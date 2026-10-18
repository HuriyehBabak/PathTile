#!/usr/bin/env python3
"""Pilot live four-shape calibration using the existing tuning-cost timer.

Copy to benchmark/ and run on an allocated GPU:
    python -m benchmark.run_pathtile_coverage_cost --gpu a6000 --trial-id 1

The original CUDA kernels, candidates, warmups, repetitions, seed, validation,
and timing code are reused. Only the training-shape list is filtered. The
existing timer covers base.run_trial and excludes interpreter/module startup,
coverage planning, and final configuration-selection CPU time. This pilot is
not a paired full-vs-subset comparison or new-workload accuracy validation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "benchmark/benchmark_pathtile_tuning_cost.py": "0a7abcd946146526aaed402b72a697eee39cc644825cf4d58308162702a22cee",
    "benchmark/benchmark_pathtile_candidate_sweep.py": "2bf122d7bbf8f1a8fbed8760801a641699a0cfa7491e26e3098575567a442bd6",
    "benchmark/benchmark_shape_sweep_ampere.py": "332d606637c7acfa905dc718c28b5686a266c18a11cf783f6ed0e061bd10e23d",
    "kernels/warp_tiled_ampere.py": "8f3bbc6e43a590255d2c93218e61d4cb07a56b6d34d3f7a19466742f17a01144",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def absolute(path):
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def filter_cases(cases, selected_shapes):
    lookup = {case.config_id: case for case in cases}
    if len(lookup) != len(cases):
        raise ValueError("Duplicate original training shape IDs")
    selected = []
    for row in selected_shapes:
        case = lookup[row["config_id"]]
        if tuple(getattr(case, name) for name in ("B", "H", "L", "K")) != tuple(
            row[name] for name in ("B", "H", "L", "K")
        ):
            raise ValueError(f"Shape dimensions changed: {case.config_id}")
        selected.append(case)
    if len(selected) != 4 or len({case.config_id for case in selected}) != 4:
        raise ValueError("Expected four unique frozen coverage shapes")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("a100", "a6000", "v100"), default="a6000")
    parser.add_argument("--trial-id", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--policy-file", type=Path, default=Path(
        "results/pathtile_selector/coverage_frozen_v1/coverage_policy.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for relative, expected in SOURCE_SHA256.items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Source differs from the reviewed archive: {relative}")

    policy_path = absolute(args.policy_file)
    bundle = json.loads(policy_path.read_text(encoding="utf-8"))
    if bundle["schema_version"] != 1 or bundle["kernel_family"] != "Warp-tiled":
        raise ValueError("Unexpected frozen-policy format or kernel family")
    items = [item for item in bundle["gpu_policies"] if item["gpu_key"] == args.gpu]
    if len(items) != 1:
        raise ValueError("Expected one frozen entry for the requested GPU")
    policy = items[0]
    meta = policy["original_measurement_metadata"]
    if policy["training_shape_count"] != 20 or int(meta["bwdk_btile"]) != 64:
        raise ValueError("Expected the original 20-shape, B_TILE=64 protocol")
    if any(meta[name] is not False for name in ("fast_math", "tf32_matmul", "tf32_cudnn")):
        raise ValueError("Expected the original strict FP32 protocol")
    warps = meta["warps_candidates"]
    blocks = meta["bwdk_block_candidates"]
    candidates = policy["candidate_values"]
    if candidates["fwd"] != warps or candidates["bwd_input"] != warps or candidates["bwd_weight"] != blocks:
        raise ValueError("Candidate definitions differ from the frozen policy")
    cells = 4 * (2 * len(warps) + len(blocks))
    if cells != policy["queried_aggregate_cells"]:
        raise ValueError("Frozen measurement budget is inconsistent")

    output = absolute(args.output_dir) if args.output_dir else (
        ROOT / "results/pathtile_selector/coverage_live_cost_v1" / args.gpu
    )
    trial = output / f"trial{args.trial_id}"
    if trial.exists():
        raise FileExistsError(f"{trial} already exists. Use a new --output-dir")
    cache = output / "pycuda_cache"
    if args.trial_id == 1 and cache.exists() and any(cache.rglob("*")):
        raise ValueError("Trial 1 requires a new empty PyCUDA compilation cache")
    if args.trial_id > 1 and not (output / "trial1/tuning_timing.json").exists():
        raise ValueError("Run trial 1 first for this cache/output series")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["PYCUDA_CACHE_DIR"] = str(cache)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from benchmark import benchmark_pathtile_tuning_cost as cost

    canonical = cost.base.cases_for_preset("pathtile-train")
    if len(canonical) != 20:
        raise ValueError("Original training preset no longer contains 20 shapes")
    selected = filter_cases(canonical, policy["selected_shapes"])
    if int(cost.base.DEFAULT_BWDK_BTILE) != int(meta["bwdk_btile"]):
        raise ValueError("Compiled batch tile differs from the frozen protocol")

    def coverage_cases(preset):
        if preset != "pathtile-train":
            raise ValueError("Only the original training preset is permitted")
        return list(selected)

    cost.base.cases_for_preset = coverage_cases
    sys.argv = ["benchmark_pathtile_tuning_cost", "--mode", "run",
        "--preset", "pathtile-train", "--trial-id", str(args.trial_id),
        "--device", str(args.device), "--warmup", str(meta["warmup"]),
        "--runs", str(meta["runs"]), "--seed", str(meta["seed"]),
        "--warps-candidates", ",".join(map(str, warps)),
        "--bwdk-block-candidates", ",".join(map(str, blocks)),
        "--output-dir", str(output)]
    print("LIVE COVERAGE CALIBRATION PILOT")
    print("Frozen shapes:", [case.config_id for case in selected])
    print(f"Expected path/candidate evaluations: {cells}")
    cost.main()

    timing = json.loads((trial / "tuning_timing.json").read_text())
    live_meta = json.loads((trial / "metadata.json").read_text())
    if timing["status"] != "PASS" or timing["training_shapes"] != 4 or timing["path_candidate_evaluations"] != cells:
        raise ValueError("Pilot failed or measurement counts differ from the frozen budget")
    if timing["gpu"] != policy["device_name"] or live_meta["compute_capability"] != meta["compute_capability"]:
        raise ValueError("Allocated GPU differs from the frozen device")
    with (trial / "correctness.csv").open(newline="", encoding="utf-8") as f:
        checks = list(csv.DictReader(f))
    if len(checks) != cells or any(row["allclose"] != "True" for row in checks):
        raise ValueError("Incomplete or failed original correctness checks")
    audit = {
        "status": "PASS",
        "experiment": "four-shape calibration pilot",
        "paired_full_sweep_comparison": False,
        "new_workload_validation": False,
        "timer_scope": "Existing base.run_trial wall timer, excludes interpreter/module startup, coverage planning, and final configuration-selection CPU time",
        "frozen_policy_sha256": digest(policy_path),
        "wrapper_sha256": digest(Path(__file__)),
        "reviewed_source_sha256": SOURCE_SHA256,
        "selected_shape_ids": [case.config_id for case in selected],
        "recorded_correctness_validation_B": sorted({int(row["validation_B"]) for row in checks}),
        "timing": timing,
    }
    with (trial / "coverage_cost_audit.json").open("x", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"\nPILOT PASS: calibration wall time={timing['total_wall_seconds']:.3f} s")
    print("Original correctness validation_B:", audit["recorded_correctness_validation_B"])
    print("Pilot output:", trial)


if __name__ == "__main__":
    main()
