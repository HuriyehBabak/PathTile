#!/usr/bin/env python3
"""Training-only development replay for selecting PathTile measurement shapes.

Copy to scripts/ in the paper repository and run on the CPU:
    python -m scripts.prototype_pathtile_shape_queries

This research prototype compares a complete sweep, random subsets, random
subsets with a cost model, and two adaptive shape-query policies. All budgeted
methods query the same number of shape/path/candidate cells. GPU measurements
are replayed from existing three-trial aggregates. The prototype performs no
new CUDA runs, reads no official holdout files, and modifies no source data.

Four-fold development validation partitions the original TRAINING shapes.
Each selection uses only its development-training partition. Configurations
are frozen before scoring the internal validation partition. Reported regret
is a development metric, not the paper's official held-out regret. Ensemble
spread is a heuristic, not a calibrated confidence interval. Compiler-resource
occupancy estimates are features, not measurements of achieved occupancy.

Any reduction in queried cells or estimated kernel-execution work is NOT a
measured reduction in tuning wall time. Compilation, validation, allocation,
zero-fill, host overhead, and model cost require a separate live experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path


PHASES = ("fwd", "bwd_input", "bwd_weight")
METHODS = ("full_sweep", "random_subset", "random_model",
           "active_generic", "active_resources")
EXPECTED = {"fwd": (4, 8, 16), "bwd_input": (4, 8, 16),
            "bwd_weight": (96, 128, 192, 224, 256, 320, 384)}


def integer(value):
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"Expected an integer, found {value!r}")
    return int(number)


def table(path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty input: {path}")
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def useful_products(B, H, L, K):
    """Mathematical non-padding products, not a hardware instruction count."""
    terms = 0
    for j in range(K):
        shift = j - K // 2
        terms += max(0, min(L, L - shift) - max(0, -shift))
    return B * H * terms


@dataclass
class PhaseData:
    candidates: tuple
    mean: object
    std: object
    resources: list


def load_training(root):
    import numpy as np

    root = root.resolve()
    if not root.name.endswith("_train") or any("holdout" in p.lower() for p in root.parts):
        raise ValueError("Use a *_train directory. Official holdout directories are excluded.")
    summary_path = root / "aggregate_candidate_summary.csv"
    rows = table(summary_path)
    sources = [summary_path]
    metas = []
    validation_B = set()
    correctness_tables = []
    for trial in (1, 2, 3):
        path = root / f"trial{trial}" / "metadata.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        if integer(meta["trial_id"]) != trial:
            raise ValueError(f"Wrong trial identity in {path}")
        metas.append(meta)
        sources.append(path)
        path = root / f"trial{trial}" / "correctness.csv"
        checks = table(path)
        correctness_tables.append((path, checks))
        sources.append(path)
        if any(row["allclose"] != "True" for row in checks):
            raise ValueError(f"Recorded candidate correctness failure in {path}")
        validation_B.update(integer(row["validation_B"]) for row in checks)
    keys = ("gpu", "sm_count", "compute_capability", "warmup", "runs", "seed",
            "warps_candidates", "bwdk_block_candidates", "bwdk_btile", "fast_math",
            "tf32_matmul", "tf32_cudnn", "measurement_contract")
    if any(any(meta[key] != metas[0][key] for key in keys) for meta in metas[1:]):
        raise ValueError("Measurement protocol differs between training trials")
    meta = metas[0]
    if meta["fast_math"] is not False or meta["tf32_matmul"] is not False or meta["tf32_cudnn"] is not False:
        raise ValueError("Expected the original strict FP32 candidate-sweep protocol")
    if integer(meta["bwdk_btile"]) != 64 or integer(meta["sm_count"]) < 1:
        raise ValueError("Expected B_TILE=64 and a positive recorded SM count")
    shapes = {}
    cells = {}
    for row in rows:
        phase = row["phase"]
        if phase not in PHASES:
            raise ValueError(f"Unknown phase {phase!r}")
        dimensions = tuple(integer(row[name]) for name in ("B", "H", "L", "K"))
        if min(dimensions) < 1 or max(dimensions[2:]) > 48:
            raise ValueError(f"Shape outside the evaluated specialization: {dimensions}")
        if integer(row["trials"]) != 3 or row["device_name"] != meta["gpu"]:
            raise ValueError("Expected three-trial means from one recorded GPU")
        kind = "bwdk_block" if phase == "bwd_weight" else "warps_per_block"
        if row["candidate_kind"] != kind or integer(row["bwdk_btile"]) != 64:
            raise ValueError("Unexpected candidate axis or batch tile")
        identifier = row["config_id"]
        if identifier in shapes and shapes[identifier] != dimensions:
            raise ValueError(f"Shape identity mismatch: {identifier}")
        shapes[identifier] = dimensions
        candidate = integer(row["candidate_value"])
        key = (identifier, phase, candidate)
        if key in cells:
            raise ValueError(f"Duplicate measurement cell: {key}")
        mean, std = float(row["mean_ms"]), float(row["between_trial_std_ms"])
        if not math.isfinite(mean) or mean <= 0 or not math.isfinite(std) or std < 0:
            raise ValueError(f"Invalid measurement: {key}")
        cells[key] = (mean, std)
    identifiers = sorted(shapes)
    dimensions = [shapes[key] for key in identifiers]
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Duplicate tensor shapes must be grouped before development validation")
    expected = {(sid, phase, c) for sid in identifiers for phase in PHASES for c in EXPECTED[phase]}
    if set(cells) != expected:
        raise ValueError("Training matrix has missing or unexpected shape/path/candidate cells")
    for path, checks in correctness_tables:
        identities = [(row["config_id"], row["phase"], integer(row["candidate_value"])) for row in checks]
        if len(identities) != len(expected) or set(identities) != expected:
            raise ValueError(f"Incomplete or duplicate recorded correctness checks in {path}")
        if any(not 1 <= integer(row["validation_B"]) <= shapes[row["config_id"]][0] for row in checks):
            raise ValueError(f"Invalid recorded validation batch in {path}")
    resources = {}
    for trial in (1, 2, 3):
        path = root / f"trial{trial}" / "kernel_resources_occupancy.csv"
        source_rows = table(path)
        sources.append(path)
        this_trial = {}
        for row in source_rows:
            phase, c = row["kernel"], integer(row["candidate_value"])
            if phase not in PHASES or c not in EXPECTED[phase]:
                raise ValueError(f"Unexpected resource record in {path}")
            key = (phase, c)
            if key in this_trial:
                raise ValueError(f"Duplicate resource record: {key}")
            attributes = {name: integer(row[name]) for name in
                          ("threads_per_block", "resident_blocks_per_sm_est",
                           "active_warps_per_sm_est", "regs_per_thread",
                           "static_smem_bytes", "local_memory_bytes")}
            if min(attributes.values()) < 0 or min(attributes[name] for name in
                    ("threads_per_block", "resident_blocks_per_sm_est", "active_warps_per_sm_est")) < 1:
                raise ValueError(f"Invalid compiler-resource estimate: {key}")
            this_trial[key] = attributes
        if set(this_trial) != {(p, c) for p in PHASES for c in EXPECTED[p]}:
            raise ValueError(f"Incomplete compiler-resource records in {path}")
        if trial == 1:
            resources = this_trial
        elif resources != this_trial:
            raise ValueError("Compiled resources differ between original trials")
    phases = {}
    for phase in PHASES:
        values = EXPECTED[phase]
        phases[phase] = PhaseData(values,
            np.asarray([[cells[(sid, phase, c)][0] for c in values] for sid in identifiers]),
            np.asarray([[cells[(sid, phase, c)][1] for c in values] for sid in identifiers]),
            [resources[(phase, c)] for c in values])
    provenance = {str(path): sha256(path) for path in sources}
    return identifiers, dimensions, phases, meta, provenance, sorted(validation_B)


def features(dimensions, phase, data, sm_count, with_resources):
    import numpy as np

    rows = []
    for B, H, L, K in dimensions:
        cases = []
        for c, r in zip(data.candidates, data.resources):
            b, h, l, k, v = (math.log2(n) for n in (B, H, L, K, c))
            values = [b, h, l, k, v, v * v, v * b, v * h, v * l, v * k]
            if with_resources:
                blocks = H * ((B + 63) // 64) if phase == "bwd_weight" else H * ((B + c - 1) // c)
                capacity = sm_count * r["resident_blocks_per_sm_est"]
                waves = blocks / capacity
                serial = (64 * useful_products(1, 1, L, K) / r["threads_per_block"]
                          if phase == "bwd_weight" else math.ceil(L / 32) * K)
                values.extend([math.log1p(waves), waves / (1 + waves),
                               math.log1p(r["active_warps_per_sm_est"]),
                               math.log1p(r["regs_per_thread"]), math.log1p(r["static_smem_bytes"]),
                               math.log1p(r["local_memory_bytes"]), math.log1p(serial)])
            cases.append(values)
        rows.append(cases)
    return np.asarray(rows, dtype=np.float64)


class QueryOracle:
    """Only eligible development-training rows may be requested by a selector."""
    def __init__(self, data, eligible):
        self._data = data
        self._eligible = list(map(int, eligible))
        self.calls = []

    def query(self, local_index):
        local_index = int(local_index)
        if not 0 <= local_index < len(self._eligible):
            raise ValueError("Selector attempted to query outside its eligible training rows")
        if local_index in self.calls:
            raise ValueError("Duplicate shape query")
        self.calls.append(local_index)
        index = self._eligible[local_index]
        return self._data.mean[index].copy(), self._data.std[index].copy()


def predictions(x, base, cached, rng, count, ridge):
    import numpy as np

    picked = sorted(cached)
    coefficients = x.shape[2] + 1
    design = np.concatenate((np.ones((*x.shape[:2], 1)), x), axis=2)
    penalty = np.eye(coefficients) * ridge
    penalty[0, 0] = 0
    result = []
    for _ in range(count):
        sampled = rng.choice(picked, size=len(picked), replace=True)
        A = design[sampled].reshape(-1, coefficients)
        means = np.stack([cached[int(i)][0] for i in sampled])
        stds = np.stack([cached[int(i)][1] for i in sampled])
        target = (np.log(means) - np.log(base[sampled, None])).ravel()
        # A fixed 1% floor prevents almost-zero empirical variances dominating.
        # These weights and bootstrap spreads are heuristics, not formal bounds.
        weights = 1 / (0.01 ** 2 + (stds / (means * math.sqrt(3))) ** 2)
        weights = weights.ravel()
        weights /= weights.mean()
        A_weighted = A * np.sqrt(weights)[:, None]
        theta = np.linalg.solve(A_weighted.T @ A_weighted + penalty,
                                A_weighted.T @ (target * np.sqrt(weights)))
        log_ms = np.einsum("sck,k->sc", design, theta) + np.log(base[:, None])
        p = np.exp(np.clip(log_ms, -30, 30))
        for i in picked:
            p[i] = cached[i][0]
        result.append(p)
    return np.stack(result)


def choose_configuration(raw_features, base, query, budget, method, seed,
                         ensemble=8, ridge=1.0):
    import numpy as np

    n = len(base)
    if not 2 <= budget <= n:
        raise ValueError("Use a shape budget between 2 and the training-pool size")
    rng = np.random.default_rng(seed)
    flat = raw_features.reshape(-1, raw_features.shape[2])
    center, scale = flat.mean(axis=0), flat.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (raw_features - center) / scale
    cached = {}

    def observe(i):
        cached[int(i)] = query(int(i))

    if method == "full_sweep":
        for i in range(n):
            observe(i)
    elif method.startswith("random"):
        for i in rng.choice(n, budget, replace=False):
            observe(i)
    else:
        # Same initial anchors in both adaptive methods. Metadata only.
        first = int(np.argmax(base))
        observe(first)
        geometry = x[:, :, :4].mean(axis=1)
        distance = ((geometry - geometry[first]) ** 2).sum(axis=1)
        distance[first] = -1
        observe(int(np.argmax(distance)))
        while len(cached) < budget:
            p = predictions(x, base, cached, rng, ensemble, ridge)
            objectives = p.sum(axis=1)
            contenders = set(map(int, objectives.argmin(axis=1)))
            contenders.update(map(int, np.argsort(objectives.mean(axis=0))[:2]))
            candidates = sorted(contenders)
            spread = np.zeros(n)
            for i, a in enumerate(candidates):
                for b in candidates[i + 1:]:
                    spread = np.maximum(spread, (p[:, :, a] - p[:, :, b]).std(axis=0))
            estimated_query_ms = np.maximum(p.mean(axis=0).sum(axis=1), 1e-12)
            score = spread ** 2 / estimated_query_ms
            for i in cached:
                score[i] = -1
            if float(score.max()) <= 1e-15:
                geometry = x.mean(axis=1)
                score = np.min([((geometry - geometry[i]) ** 2).sum(axis=1)
                                for i in cached], axis=0)
                for i in cached:
                    score[i] = -1
            observe(int(np.argmax(score)))
    if len(cached) == n or method == "random_subset":
        total = np.stack([item[0] for item in cached.values()]).sum(axis=0)
    else:
        total = predictions(x, base, cached, rng, ensemble, ridge).mean(axis=0).sum(axis=0)
    chosen = int(np.argmin(total))
    queried_ms = math.fsum(float(item[0].sum()) for item in cached.values())
    return chosen, list(cached), queried_ms


def development(ids, dimensions, phases, meta, args):
    import numpy as np

    n = len(ids)
    base = np.asarray([useful_products(*shape) for shape in dimensions], dtype=np.float64)
    all_features = {(phase, resource): features(dimensions, phase, phases[phase],
                    integer(meta["sm_count"]), resource) for phase in PHASES for resource in (False, True)}
    records = []
    work_per_cell = 3 * (integer(meta["warmup"]) + integer(meta["runs"]))
    for repeat in range(args.repeats):
        permutation = np.random.default_rng(args.seed + repeat).permutation(n)
        folds = np.array_split(permutation, args.folds)
        for fold, validation in enumerate(folds):
            pool = np.asarray(sorted(set(range(n)) - set(map(int, validation))), dtype=int)
            if args.shape_budget > len(pool):
                raise ValueError("Shape budget exceeds an internal training-pool size")
            for method in METHODS:
                policy, traces = {}, {}
                cells, queried_ms = 0, 0.0
                for pi, phase in enumerate(PHASES):
                    data = phases[phase]
                    oracle = QueryOracle(data, pool)
                    resource = method in ("active_resources", "random_model")
                    budget = len(pool) if method == "full_sweep" else args.shape_budget
                    chosen, queries, ms = choose_configuration(
                        all_features[(phase, resource)][pool], base[pool], oracle.query,
                        budget, method, args.seed + 10000 * repeat + 100 * fold + pi,
                        args.ensemble, args.ridge)
                    policy[phase] = chosen
                    traces[phase] = [ids[int(pool[i])] for i in queries]
                    cells += len(queries) * len(data.candidates)
                    queried_ms += ms
                # All three choices are frozen before any validation latency is scored.
                totals = np.zeros(len(validation))
                best = np.zeros(len(validation))
                for phase in PHASES:
                    totals += phases[phase].mean[validation, policy[phase]]
                    best += phases[phase].mean[validation].min(axis=1)
                regret = 100 * (totals / best - 1)
                records.append({"repeat": repeat, "fold": fold, "method": method,
                    "policy": {p: phases[p].candidates[policy[p]] for p in PHASES},
                    "queried_shapes": traces, "queried_cells": cells,
                    "complete_pool_cells": len(pool) * sum(len(p.candidates) for p in phases.values()),
                    "kernel_execution_ms_estimate": queried_ms * work_per_cell,
                    "validation_shapes": [ids[int(i)] for i in validation],
                    "validation_regret_pct": [float(v) for v in regret],
                    "validation_latency_ms": [float(v) for v in totals]})
        print(f"Development repeat {repeat + 1}/{args.repeats} complete", flush=True)
    summary = []
    for method in METHODS:
        rows = [row for row in records if row["method"] == method]
        regrets = [v for row in rows for v in row["validation_regret_pct"]]
        repeat_means = [float(np.mean([v for row in rows if row["repeat"] == repeat
                       for v in row["validation_regret_pct"]])) for repeat in range(args.repeats)]
        summary.append({"method": method, "unique_development_validation_shapes": n,
            "repeat_mean_regret_pct": float(np.mean(regrets)), "max_observed_regret_pct": max(regrets),
            "repeat_mean_regret_std_pct": float(np.std(repeat_means, ddof=1)) if args.repeats > 1 else None,
            "mean_queried_cells": float(np.mean([row["queried_cells"] for row in rows])),
            "mean_query_fraction": float(np.mean([row["queried_cells"] / row["complete_pool_cells"] for row in rows])),
            "mean_kernel_execution_ms_estimate": float(np.mean([row["kernel_execution_ms_estimate"] for row in rows]))})
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-root", type=Path, default=Path("results/pathtile_candidates/a6000_train"))
    parser.add_argument("--shape-budget", type=int, default=4)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ensemble", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if min(args.shape_budget, args.folds, args.ensemble) < 2 or args.repeats < 1 or args.seed < 0:
        parser.error("Use budget, folds, ensemble >=2, repeats>=1 and seed>=0")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("Ridge must be finite and positive")
    destination = args.out or Path("results/pathtile_selector/prototype_shape_queries") / args.train_root.name
    if destination.exists():
        raise FileExistsError(f"{destination} already exists. Use a new --out directory")
    import numpy as np

    start = time.perf_counter()
    ids, dimensions, phases, meta, provenance, validation_B = load_training(args.train_root)
    if not 2 <= args.folds <= len(ids) - 2:
        parser.error("Use fewer folds than the number of training shapes minus one")
    print(f"GPU data: {meta['gpu']} | training shapes: {len(ids)} | original trials: 3")
    print("Mode: training-only cached replay with internal development validation")
    print(f"Budget: {args.shape_budget} queried shapes per path, folds={args.folds}, repeats={args.repeats}")
    print(f"Recorded correctness validation_B values: {validation_B}")
    summary, records = development(ids, dimensions, phases, meta, args)
    original_policy = {p: phases[p].candidates[int(phases[p].mean.sum(axis=0).argmin())] for p in PHASES}
    report = {"development_only": True, "official_holdout_read": False,
        "new_cuda_measurements": False, "measured_tuning_wall_time": False,
        "training_root": str(args.train_root.resolve()), "original_training_policy_audit_only": original_policy,
        "source_sha256": provenance, "script_sha256": sha256(Path(__file__)),
        "numpy_version": np.__version__, "arguments": {key: str(v) if isinstance(v, Path) else v
                                                       for key, v in vars(args).items()},
        "metadata": meta, "recorded_correctness_validation_B": validation_B,
        "model_assumptions": {"target": "log latency per useful mathematical product",
            "training_objective": "sum of predicted latency over the eligible training pool",
            "query_score": "ensemble pairwise-gap variance divided by predicted query latency",
            "noise_floor_relative": 0.01, "log_latency_clip": [-30, 30],
            "ensemble_spread_is_calibrated_confidence": False,
            "occupancy_features_are_analytical_estimates": True},
        "kernel_execution_estimate": "3*(warmup+runs)*sum of queried aggregate kernel latencies",
        "validation_design": "repeated internal folds of original training shapes, not independent new workloads",
        "summary": summary, "records": records,
        "cpu_replay_wall_seconds": time.perf_counter() - start}
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "development_replay.json"
    with path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print("\nDevelopment summary:")
    for row in summary:
        print(f"  {row['method']:18s} mean_regret={row['repeat_mean_regret_pct']:.4f}% "
              f"max={row['max_observed_regret_pct']:.4f}% "
              f"query_fraction={row['mean_query_fraction']:.3f}")
    print("Full original-training policy (audit only):", original_policy)
    print(f"Saved: {path}")
    print("These are development replay results. New-workload validation and live tuning cost are not measured.")


if __name__ == "__main__":
    main()
