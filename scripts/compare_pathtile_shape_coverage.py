#!/usr/bin/env python3
"""Compare adaptive and metadata-only queries in cached development replay.

Copy beside scripts/prototype_pathtile_shape_queries.py and run:
    python -m scripts.compare_pathtile_shape_coverage

Both limited-budget policies select configurations by the sum of measured
latencies on their queried shapes, for all three paths. The only difference
is shape selection. No official holdout, CUDA execution, file writes, or
wall-time saving measurement is performed. Repeated folds reuse the original
training shapes and do not represent independent new workloads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def fold_records(records, method):
    result = {}
    for row in records:
        if row["method"] != method:
            continue
        key = (row["repeat"], row["fold"])
        if key in result:
            raise ValueError(f"Duplicate fold record for {method}: {key}")
        result[key] = row
    if not result:
        raise ValueError(f"Missing method: {method}")
    return result


def coverage_queries(pool, dimensions, budget, useful_products):
    """Largest useful work first, then farthest coverage of log2(B,H,L,K)."""
    if not 2 <= budget <= len(pool):
        raise ValueError("Invalid development query budget")
    geometry = np.log2(np.asarray([dimensions[i] for i in pool], dtype=float))
    scale = geometry.std(axis=0)
    scale[scale < 1e-12] = 1
    geometry = (geometry - geometry.mean(axis=0)) / scale
    work = [useful_products(*dimensions[i]) for i in pool]
    picked = [int(np.argmax(work))]
    while len(picked) < budget:
        distances = np.stack([
            ((geometry - geometry[i]) ** 2).sum(axis=1) for i in picked
        ]).min(axis=0)
        distances[picked] = -1
        picked.append(int(np.argmax(distances)))
    return [pool[i] for i in picked]


def measured_policy(data, phases, queries, pool, budget):
    policy = {}
    for phase in phases:
        positions = queries[phase]
        if len(positions) != budget or len(set(positions)) != budget:
            raise ValueError("Incorrect number of unique queried shapes")
        if not set(positions).issubset(pool):
            raise ValueError("A query is outside the development training pool")
        choice = int(data[phase].mean[positions].sum(axis=0).argmin())
        policy[phase] = data[phase].candidates[choice]
    return policy


def score_policy(data, phases, policy, positions):
    latency = sum(
        data[p].mean[positions, data[p].candidates.index(policy[p])]
        for p in phases
    )
    oracle = sum(data[p].mean[positions].min(axis=1) for p in phases)
    return latency, 100 * (latency / oracle - 1)


def compare_gpu(gpu, args, proto):
    root = args.train_parent / f"{gpu}_train"
    report_path = args.replay_parent / f"{gpu}_train" / "development_replay.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["development_only"] is not True or report["official_holdout_read"] is not False:
        raise ValueError("Expected training-only development replay")
    if proto.sha256(Path(proto.__file__)) != report["script_sha256"]:
        raise ValueError("Prototype source changed since the saved replay")
    ids, dimensions, data, _, provenance, _ = proto.load_training(root)
    if provenance != report["source_sha256"]:
        raise ValueError(f"Source data changed for {gpu}")
    lookup = {sid: i for i, sid in enumerate(ids)}
    budget = int(report["arguments"]["shape_budget"])
    full = fold_records(report["records"], "full_sweep")
    active = fold_records(report["records"], "active_resources")
    if set(full) != set(active):
        raise ValueError("Adaptive and full-sweep folds differ")
    methods = ("full_sweep", "active_observed", "coverage_observed")
    scores = {method: [] for method in methods}
    matches = {method: 0 for method in methods}
    cell_counts = {method: [] for method in methods}
    candidates_per_shape = sum(len(data[p].candidates) for p in proto.PHASES)

    for key, row in full.items():
        other = active[key]
        if row["validation_shapes"] != other["validation_shapes"]:
            raise ValueError("Validation partitions differ")
        validation = [lookup[sid] for sid in row["validation_shapes"]]
        pool = [i for i in range(len(ids)) if i not in set(validation)]
        if len(set(validation)) != len(validation) or not validation:
            raise ValueError("Invalid validation partition")
        coverage = coverage_queries(pool, dimensions, budget, proto.useful_products)
        queries = {
            "active_observed": {
                p: [lookup[sid] for sid in other["queried_shapes"][p]]
                for p in proto.PHASES
            },
            "coverage_observed": {p: coverage for p in proto.PHASES},
        }
        policies = {
            "full_sweep": measured_policy(
                data, proto.PHASES, {p: pool for p in proto.PHASES},
                pool, len(pool)
            )
        }
        if policies["full_sweep"] != row["policy"]:
            raise ValueError("Recomputed full-sweep policy differs from replay")
        for method, method_queries in queries.items():
            policies[method] = measured_policy(
                data, proto.PHASES, method_queries, pool, budget
            )
        for method, policy in policies.items():
            latency, regret = score_policy(data, proto.PHASES, policy, validation)
            if method == "full_sweep":
                if not np.allclose(latency, row["validation_latency_ms"], rtol=1e-12, atol=1e-12):
                    raise ValueError("Saved full-sweep latency differs from source")
                if not np.allclose(regret, row["validation_regret_pct"], rtol=1e-10, atol=1e-9):
                    raise ValueError("Saved full-sweep regret differs from source")
            scores[method].extend(regret.tolist())
            matches[method] += int(policy == policies["full_sweep"])
            count = len(pool) if method == "full_sweep" else budget
            cell_counts[method].append(count * candidates_per_shape)

    print(f"\nGPU: {gpu}")
    print("Both budgeted methods use observed sums for FWD, dX, and dK")
    for method in methods:
        print(
            f"{method:18s} mean_regret={np.mean(scores[method]):.4f}% "
            f"max={max(scores[method]):.4f}% "
            f"full_policy_matches={matches[method]}/{len(full)} "
            f"mean_cells={np.mean(cell_counts[method]):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs="+", choices=("a100", "a6000", "v100"),
                        default=("a100", "a6000", "v100"))
    parser.add_argument("--train-parent", type=Path,
                        default=Path("results/pathtile_candidates"))
    parser.add_argument("--replay-parent", type=Path,
                        default=Path("results/pathtile_selector/prototype_shape_queries"))
    args = parser.parse_args()
    from scripts import prototype_pathtile_shape_queries as proto

    print("Training-only cached development comparison. No new CUDA measurements.")
    for gpu in args.gpus:
        compare_gpu(gpu, args, proto)


if __name__ == "__main__":
    main()
