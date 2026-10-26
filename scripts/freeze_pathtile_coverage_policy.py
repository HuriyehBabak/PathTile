#!/usr/bin/env python3
"""Freeze metadata-coverage PathTile policies for prospective evaluation.

Run:
    python -m scripts.freeze_pathtile_coverage_policy

Uses only the original training data. Six representative shapes are selected
from workload dimensions and useful mathematical work. Each path's launch
configuration minimizes measured latency summed over those six shapes.

The shape budget of six was chosen from training-only repeated four-fold
replay, without access to the official holdout set. No model predictions or
new CUDA measurements are used here.

The saved policy is frozen for prospective holdout evaluation. It is not yet
a new validated holdout result and does not establish measured tuning wall-time
savings. Existing policies and source data are never overwritten.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SHAPE_BUDGET = 6


def build_gpu_policy(gpu, args, proto, comparison):
    root = args.train_parent / f"{gpu}_train"
    replay_path = (
        args.replay_parent
        / f"{gpu}_train"
        / "development_replay.json"
    )

    replay = json.loads(
        replay_path.read_text(encoding="utf-8")
    )

    if (
        replay["development_only"] is not True
        or replay["official_holdout_read"] is not False
    ):
        raise ValueError(
            "Expected original training-only development replay"
        )

    if proto.sha256(Path(proto.__file__)) != replay["script_sha256"]:
        raise ValueError(
            "Prototype source changed since development replay"
        )

    (
        ids,
        dimensions,
        data,
        meta,
        provenance,
        validation_B,
    ) = proto.load_training(root)

    if provenance != replay["source_sha256"]:
        raise ValueError(
            f"Original training files changed for {gpu}"
        )

    if len(ids) != 20:
        raise ValueError(
            "Expected the original 20 training shapes"
        )

    if gpu.upper() not in meta["gpu"].upper():
        raise ValueError(
            f"GPU identity mismatch for {gpu}: {meta['gpu']}"
        )

    pool = list(range(len(ids)))

    selected = comparison.coverage_queries(
        pool,
        dimensions,
        SHAPE_BUDGET,
        proto.useful_products,
    )

    if len(selected) != SHAPE_BUDGET:
        raise ValueError(
            f"{gpu}: expected {SHAPE_BUDGET} selected shapes, "
            f"got {len(selected)}"
        )

    policy = comparison.measured_policy(
        data,
        proto.PHASES,
        {
            phase: selected
            for phase in proto.PHASES
        },
        pool,
        len(selected),
    )

    shapes = [
        {
            "config_id": ids[i],
            **dict(
                zip(
                    ("B", "H", "L", "K"),
                    dimensions[i],
                )
            ),
        }
        for i in selected
    ]

    sums = {
        phase: float(
            data[phase]
            .mean[
                selected,
                data[phase]
                .candidates
                .index(policy[phase]),
            ]
            .sum()
        )
        for phase in proto.PHASES
    }

    candidates_per_shape = sum(
        len(data[phase].candidates)
        for phase in proto.PHASES
    )

    queried_cells = (
        SHAPE_BUDGET * candidates_per_shape
    )

    complete_cells = (
        len(ids) * candidates_per_shape
    )

    return {
        "gpu_key": gpu,
        "device_name": meta["gpu"],
        "training_root": str(root.resolve()),
        "training_shape_count": len(ids),
        "shape_budget": SHAPE_BUDGET,
        "selected_shapes": shapes,
        "candidate_values": {
            phase: list(data[phase].candidates)
            for phase in proto.PHASES
        },
        "selected_candidate_values": policy,
        "launch_configuration": {
            "fwd": {
                "warps_per_block": policy["fwd"],
            },
            "bwd_input": {
                "warps_per_block":
                    policy["bwd_input"],
            },
            "bwd_weight": {
                "bwdk_block":
                    policy["bwd_weight"],
                "bwdk_btile":
                    int(meta["bwdk_btile"]),
            },
        },
        "selected_shape_latency_sum_ms": sums,
        "queried_aggregate_cells": queried_cells,
        "complete_training_aggregate_cells":
            complete_cells,
        "measurement_cell_reduction_pct":
            100.0
            * (1.0 - queried_cells / complete_cells),
        "recorded_correctness_validation_B":
            validation_B,
        "original_measurement_metadata": meta,
        "source_sha256": provenance,
        "development_replay_sha256":
            proto.sha256(replay_path),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--train-parent",
        type=Path,
        default=Path(
            "results/pathtile_candidates"
        ),
    )

    parser.add_argument(
        "--replay-parent",
        type=Path,
        default=Path(
            "results/pathtile_selector/"
            "prototype_shape_queries"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/pathtile_selector/"
            "coverage_frozen_budget6_v1"
        ),
    )

    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(
            f"{args.out} already exists. "
            "Use a new --out directory"
        )

    from scripts import (
        prototype_pathtile_shape_queries
        as proto
    )
    from scripts import (
        compare_pathtile_shape_coverage
        as comparison
    )

    policies = [
        build_gpu_policy(
            gpu,
            args,
            proto,
            comparison,
        )
        for gpu in ("a100", "a6000", "v100")
    ]

    result = {
        "schema_version": 2,
        "status":
            "frozen candidate for prospective evaluation",
        "kernel_family": "Warp-tiled",
        "frozen_at_utc":
            datetime.now(timezone.utc).isoformat(),
        "shape_budget": SHAPE_BUDGET,
        "budget_selection_basis":
            (
                "Training-only repeated four-fold replay; "
                "official holdout not accessed"
            ),
        "official_holdout_read": False,
        "new_cuda_measurements": False,
        "measured_tuning_wall_time": False,
        "accuracy_scope":
            (
                "Original recorded correctness checks, "
                "not full-shape recertification"
            ),
        "selection_rule": {
            "shape_budget": SHAPE_BUDGET,
            "first_shape":
                (
                    "Maximum useful non-padding "
                    "mathematical product count"
                ),
            "remaining_shapes":
                (
                    "Farthest minimum squared distance "
                    "in standardized log2(B,H,L,K)"
                ),
            "standardization_population":
                (
                    "All 20 original training shapes, "
                    "dimensions only"
                ),
            "shape_tie_break":
                (
                    "First shape in sorted config_id order"
                ),
            "same_shapes_for_all_paths": True,
            "configuration_objective":
                (
                    "Sum of aggregate measured latency "
                    f"over the {SHAPE_BUDGET} selected shapes"
                ),
            "configuration_tie_break":
                "Smallest candidate value",
            "runtime_shape_adaptation": False,
        },
        "code_sha256": {
            "freeze_pathtile_coverage_policy.py":
                proto.sha256(Path(__file__)),
            "compare_pathtile_shape_coverage.py":
                proto.sha256(
                    Path(comparison.__file__)
                ),
            "prototype_pathtile_shape_queries.py":
                proto.sha256(
                    Path(proto.__file__)
                ),
        },
        "gpu_policies": policies,
    }

    serialized = (
        json.dumps(
            result,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )

    args.out.mkdir(
        parents=True,
        exist_ok=False,
    )

    path = (
        args.out
        / "coverage_policy.json"
    )

    with path.open(
        "x",
        encoding="utf-8",
    ) as file:
        file.write(serialized)

    for item in policies:
        print(
            f"\nGPU: {item['gpu_key']} | "
            f"{item['device_name']}"
        )

        print(
            "Selected shapes:",
            [
                shape["config_id"]
                for shape
                in item["selected_shapes"]
            ],
        )

        print(
            "Frozen policy:",
            item["selected_candidate_values"],
        )

        print(
            "Aggregate cells: "
            f"{item['queried_aggregate_cells']}/"
            f"{item['complete_training_aggregate_cells']}"
        )

        print(
            "Measurement-cell reduction: "
            f"{item['measurement_cell_reduction_pct']:.1f}%"
        )

    print(
        f"\nSaved frozen candidate: {path}"
    )

    print(
        "Budget selection and policy construction "
        "used training data only."
    )

    print(
        "No official-holdout evaluation or live "
        "tuning-cost measurement has been performed."
    )


if __name__ == "__main__":
    main()