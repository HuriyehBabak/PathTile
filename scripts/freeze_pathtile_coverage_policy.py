#!/usr/bin/env python3
"""Freeze metadata-coverage PathTile policies for prospective evaluation.

Copy beside the two existing scripts and run:
    python -m scripts.freeze_pathtile_coverage_policy

Uses only original training data. Four shapes are selected from dimensions
and useful mathematical work. Each path's configuration minimizes measured
latency summed over those four shapes. No model predictions or new CUDA
measurements are used. The saved policy is a candidate for future validation,
not a new validated paper result or a measurement of tuning wall-time saving.
Existing policies and source data are never overwritten.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def build_gpu_policy(gpu, args, proto, comparison):
    root = args.train_parent / f"{gpu}_train"
    replay_path = args.replay_parent / f"{gpu}_train" / "development_replay.json"
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    if replay["development_only"] is not True or replay["official_holdout_read"] is not False:
        raise ValueError("Expected original training-only development replay")
    if proto.sha256(Path(proto.__file__)) != replay["script_sha256"]:
        raise ValueError("Prototype source changed since development replay")
    ids, dimensions, data, meta, provenance, validation_B = proto.load_training(root)
    if provenance != replay["source_sha256"]:
        raise ValueError(f"Original training files changed for {gpu}")
    if len(ids) != 20:
        raise ValueError("Expected the original 20 training shapes")
    if gpu.upper() not in meta["gpu"].upper():
        raise ValueError(f"GPU identity mismatch for {gpu}: {meta['gpu']}")
    pool = list(range(len(ids)))
    selected = comparison.coverage_queries(pool, dimensions, 4, proto.useful_products)
    policy = comparison.measured_policy(
        data, proto.PHASES, {p: selected for p in proto.PHASES}, pool, 4
    )
    shapes = [
        {"config_id": ids[i], **dict(zip(("B", "H", "L", "K"), dimensions[i]))}
        for i in selected
    ]
    sums = {
        p: float(data[p].mean[selected, data[p].candidates.index(policy[p])].sum())
        for p in proto.PHASES
    }
    count = sum(len(data[p].candidates) for p in proto.PHASES)
    return {
        "gpu_key": gpu,
        "device_name": meta["gpu"],
        "training_root": str(root.resolve()),
        "training_shape_count": len(ids),
        "selected_shapes": shapes,
        "candidate_values": {p: list(data[p].candidates) for p in proto.PHASES},
        "selected_candidate_values": policy,
        "launch_configuration": {
            "fwd": {"warps_per_block": policy["fwd"]},
            "bwd_input": {"warps_per_block": policy["bwd_input"]},
            "bwd_weight": {
                "bwdk_block": policy["bwd_weight"],
                "bwdk_btile": int(meta["bwdk_btile"]),
            },
        },
        "selected_shape_latency_sum_ms": sums,
        "queried_aggregate_cells": 4 * count,
        "complete_training_aggregate_cells": len(ids) * count,
        "recorded_correctness_validation_B": validation_B,
        "original_measurement_metadata": meta,
        "source_sha256": provenance,
        "development_replay_sha256": proto.sha256(replay_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parent", type=Path,
                        default=Path("results/pathtile_candidates"))
    parser.add_argument("--replay-parent", type=Path,
                        default=Path("results/pathtile_selector/prototype_shape_queries"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/pathtile_selector/coverage_frozen_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} already exists. Use a new --out directory")
    from scripts import prototype_pathtile_shape_queries as proto
    from scripts import compare_pathtile_shape_coverage as comparison

    policies = [build_gpu_policy(gpu, args, proto, comparison)
                for gpu in ("a100", "a6000", "v100")]
    result = {
        "schema_version": 1,
        "status": "frozen candidate for prospective evaluation",
        "kernel_family": "Warp-tiled",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_holdout_read": False,
        "new_cuda_measurements": False,
        "measured_tuning_wall_time": False,
        "accuracy_scope": "Original recorded correctness checks, not full-shape recertification",
        "selection_rule": {
            "shape_budget": 4,
            "first_shape": "Maximum useful non-padding mathematical product count",
            "remaining_shapes": "Farthest minimum squared distance in standardized log2(B,H,L,K)",
            "standardization_population": "All 20 original training shapes, dimensions only",
            "shape_tie_break": "First shape in sorted config_id order",
            "same_shapes_for_all_paths": True,
            "configuration_objective": "Sum of aggregate measured latency over the four selected shapes",
            "configuration_tie_break": "Smallest candidate value",
            "runtime_shape_adaptation": False,
        },
        "code_sha256": {
            "freeze_pathtile_coverage_policy.py": proto.sha256(Path(__file__)),
            "compare_pathtile_shape_coverage.py": proto.sha256(Path(comparison.__file__)),
            "prototype_pathtile_shape_queries.py": proto.sha256(Path(proto.__file__)),
        },
        "gpu_policies": policies,
    }
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "coverage_policy.json"
    with path.open("x", encoding="utf-8") as f:
        f.write(serialized)
    for item in policies:
        print(f"\nGPU: {item['gpu_key']} | {item['device_name']}")
        print("Selected shapes:", [s["config_id"] for s in item["selected_shapes"]])
        print("Frozen policy:", item["selected_candidate_values"])
        print(f"Aggregate cells: {item['queried_aggregate_cells']}/"
              f"{item['complete_training_aggregate_cells']}")
    print(f"\nSaved frozen candidate: {path}")
    print("No new-workload validation or live tuning-cost measurement has been performed.")


if __name__ == "__main__":
    main()
