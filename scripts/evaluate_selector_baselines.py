#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate_pathtile_results import (
    GPU_CONFIGS,
    PHASES,
    DEFAULT_WARPS,
    DEFAULT_BWDK_BLOCK,
    aggregate_candidate_root,
    select_architecture_policy,
    _selected_phase_time,
)
from scripts.artifact_common import geometric_mean


ROOT = Path.cwd()

OUTPUT_DIR = (
    ROOT
    / "results/pathtile_selector/selector_baselines_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------
# Load candidate measurements
# ---------------------------------------------------------------------

training_data = {}
holdout_data = {}
architecture_policies = {}

for gpu_label, gpu_key, device_name in GPU_CONFIGS:
    _, train_candidates, _, train_oracle = aggregate_candidate_root(
        ROOT,
        Path(f"results/pathtile_candidates/{gpu_key}_train"),
    )

    _, holdout_candidates, _, holdout_oracle = aggregate_candidate_root(
        ROOT,
        Path(f"results/pathtile_candidates/{gpu_key}_holdout"),
    )

    training_data[gpu_label] = (
        train_candidates,
        train_oracle,
        device_name,
    )

    holdout_data[gpu_label] = (
        holdout_candidates,
        holdout_oracle,
        device_name,
    )

    architecture_policies[gpu_label] = (
        select_architecture_policy(train_candidates)
    )


# ---------------------------------------------------------------------
# Default policy
# ---------------------------------------------------------------------

default_policy = {
    "fwd_warps": DEFAULT_WARPS,
    "bwd_input_warps": DEFAULT_WARPS,
    "bwd_weight_block": DEFAULT_BWDK_BLOCK,
}


# ---------------------------------------------------------------------
# Select a true global-static policy from TRAINING DATA ONLY
# ---------------------------------------------------------------------
#
# This mirrors the architecture-static selection rule:
# minimize the sum of candidate latency over the development set,
# but now pool all three GPUs.
#
# A candidate must be present for every relevant GPU/configuration.
# ---------------------------------------------------------------------

phase_to_output = {
    "fwd": "fwd_warps",
    "bwd_input": "bwd_input_warps",
    "bwd_weight": "bwd_weight_block",
}

global_policy = {}

for phase in PHASES:
    phase_frames = []

    for gpu_label, (
        candidates,
        _,
        _,
    ) in training_data.items():
        subset = candidates[
            candidates["phase"] == phase
        ].copy()

        subset["gpu"] = gpu_label

        phase_frames.append(subset)

    pooled = pd.concat(
        phase_frames,
        ignore_index=True,
    )

    # Number of GPU-shape cases that every candidate must cover.
    expected_cases = (
        pooled[["gpu", "config_id"]]
        .drop_duplicates()
        .shape[0]
    )

    candidate_summary = (
        pooled
        .groupby(
            "candidate_value",
            as_index=False,
        )
        .agg(
            total_ms=("mean_ms", "sum"),
            covered_cases=(
                "config_id",
                "size",
            ),
        )
    )

    # Keep only candidates measured for every GPU-shape case.
    candidate_summary = candidate_summary[
        candidate_summary["covered_cases"]
        == expected_cases
    ].copy()

    if candidate_summary.empty:
        raise RuntimeError(
            f"No globally comparable candidate for {phase}"
        )

    candidate_summary = (
        candidate_summary
        .sort_values(
            ["total_ms", "candidate_value"],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    winner = int(
        candidate_summary.iloc[0]["candidate_value"]
    )

    global_policy[
        phase_to_output[phase]
    ] = winner


# ---------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------

def schedule_string(policy: dict[str, int]) -> str:
    return (
        f"fwd={policy['fwd_warps']},"
        f"dx={policy['bwd_input_warps']},"
        f"dk={policy['bwd_weight_block']}"
    )


def static_total(
    candidates: pd.DataFrame,
    config_id: str,
    policy: dict[str, int],
) -> float:
    fwd = _selected_phase_time(
        candidates,
        config_id,
        "fwd",
        policy["fwd_warps"],
    )

    dx = _selected_phase_time(
        candidates,
        config_id,
        "bwd_input",
        policy["bwd_input_warps"],
    )

    dk = _selected_phase_time(
        candidates,
        config_id,
        "bwd_weight",
        policy["bwd_weight_block"],
    )

    return fwd + dx + dk

def candidate_values_for_phase(
    candidates: pd.DataFrame,
    phase: str,
) -> list[int]:
    values = (
        candidates.loc[
            candidates["phase"] == phase,
            "candidate_value",
        ]
        .drop_duplicates()
        .astype(int)
        .sort_values()
        .tolist()
    )
    return values


def enumerate_static_policies(
    candidates: pd.DataFrame,
) -> list[dict[str, int]]:
    fwd_values = candidate_values_for_phase(
        candidates,
        "fwd",
    )
    dx_values = candidate_values_for_phase(
        candidates,
        "bwd_input",
    )
    dk_values = candidate_values_for_phase(
        candidates,
        "bwd_weight",
    )

    return [
        {
            "fwd_warps": fwd,
            "bwd_input_warps": dx,
            "bwd_weight_block": dk,
        }
        for fwd in fwd_values
        for dx in dx_values
        for dk in dk_values
    ]


# ---------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------

def select_majority_winner_policy(
    candidates: pd.DataFrame,
) -> dict[str, int]:
    policy = {}

    for phase in PHASES:
        subset = candidates[
            candidates["phase"] == phase
        ].copy()

        # Best candidate for each training shape.
        winners = (
            subset
            .sort_values(
                ["config_id", "mean_ms", "candidate_value"],
                kind="stable",
            )
            .groupby(
                "config_id",
                as_index=False,
            )
            .first()
        )

        # Candidate that wins on the largest number of shapes.
        counts = (
            winners["candidate_value"]
            .value_counts()
            .rename_axis("candidate_value")
            .reset_index(name="wins")
            .sort_values(
                ["wins", "candidate_value"],
                ascending=[False, True],
                kind="stable",
            )
            .reset_index(drop=True)
        )

        winner = int(
            counts.iloc[0]["candidate_value"]
        )

        policy[
            phase_to_output[phase]
        ] = winner

    return policy

majority_winner_policies = {}

for gpu_label, (
    train_candidates,
    _,
    _,
) in training_data.items():
    majority_winner_policies[gpu_label] = (
        select_majority_winner_policy(
            train_candidates
        )
    )

# ---------------------------------------------------------------------
# Evaluate all policies on HOLDOUT ONLY
# ---------------------------------------------------------------------

rows = []

for gpu_label, (
    candidates,
    oracle,
    device_name,
) in holdout_data.items():

    pathtile_policy = architecture_policies[
        gpu_label
    ]
    majority_winner_policy = majority_winner_policies[
        gpu_label
    ]
    random_fixed_policies = enumerate_static_policies(
        training_data[gpu_label][0]
    )
    
    if len(random_fixed_policies) != 63:
        raise RuntimeError(
            f"Expected 63 random-fixed policies for {gpu_label}, "
            f"got {len(random_fixed_policies)}"
        )

    for oracle_row in oracle.itertuples(
        index=False
    ):
        config_id = oracle_row.config_id

        oracle_total = float(
            oracle_row.oracle_total_ms
        )
        random_fixed_latencies = np.array(
            [
                static_total(
                    candidates,
                    config_id,
                    policy,
                )
                for policy in random_fixed_policies
            ],
            dtype=float,
        )

        random_fixed_latency = float(
            np.mean(random_fixed_latencies)
        )

        random_fixed_regret = (
            random_fixed_latency / oracle_total - 1.0
        ) * 100.0

        oracle_schedule = (
            f"fwd={int(oracle_row.fwd_winner_warps)},"
            f"dx={int(oracle_row.bwd_input_winner_warps)},"
            f"dk={int(oracle_row.bwd_weight_winner_block)}"
        )

        methods = [
            (
                "Default",
                default_policy,
                static_total(
                    candidates,
                    config_id,
                    default_policy,
                ),
            ),
            (
                "Global-static",
                global_policy,
                static_total(
                    candidates,
                    config_id,
                    global_policy,
                ),
            ),
            (
                "Majority-winner",
                majority_winner_policy,
                static_total(
                    candidates,
                    config_id,
                    majority_winner_policy,
                ),
            ),
            (
                "PathTile",
                pathtile_policy,
                static_total(
                    candidates,
                    config_id,
                    pathtile_policy,
                ),
            ),
        ]

        for (
            method,
            policy,
            latency,
        ) in methods:
            regret = (
                latency / oracle_total - 1.0
            ) * 100.0

            rows.append(
                {
                    "gpu": gpu_label,
                    "device_name": device_name,
                    "config_id": config_id,
                    "B": int(oracle_row.B),
                    "H": int(oracle_row.H),
                    "L": int(oracle_row.L),
                    "K": int(oracle_row.K),
                    "method": method,
                    "selected_schedule": (
                        schedule_string(policy)
                    ),
                    "oracle_schedule": oracle_schedule,
                    "selected_latency_ms": latency,
                    "oracle_latency_ms": oracle_total,
                    "regret_pct": regret,
                }
            )

        
        rows.append(
            {
                "gpu": gpu_label,
                "device_name": device_name,
                "config_id": config_id,
                "B": int(oracle_row.B),
                "H": int(oracle_row.H),
                "L": int(oracle_row.L),
                "K": int(oracle_row.K),
                "method": "Random-fixed (expected)",
                "selected_schedule": (
                    f"uniform over {len(random_fixed_policies)} "
                    "fixed policies"
                ),
                "oracle_schedule": oracle_schedule,
                "selected_latency_ms": random_fixed_latency,
                "oracle_latency_ms": oracle_total,
                "regret_pct": random_fixed_regret,
            }
        )

        # ORACLE BLOCK
        rows.append(
            {
                "gpu": gpu_label,
                "device_name": device_name,
                "config_id": config_id,
                "B": int(oracle_row.B),
                "H": int(oracle_row.H),
                "L": int(oracle_row.L),
                "K": int(oracle_row.K),
                "method": "Oracle",
                "selected_schedule": oracle_schedule,
                "oracle_schedule": oracle_schedule,
                "selected_latency_ms": oracle_total,
                "oracle_latency_ms": oracle_total,
                "regret_pct": 0.0,
            }
        )


per_shape = pd.DataFrame(rows)

method_order = [
    "Default",
    "Global-static",
    "Random-fixed (expected)",
    "Majority-winner",
    "PathTile",
    "Oracle",
]

per_shape["method"] = pd.Categorical(
    per_shape["method"],
    categories=method_order,
    ordered=True,
)

per_shape = (
    per_shape
    .sort_values(
        ["gpu", "config_id", "method"],
        kind="stable",
    )
    .reset_index(drop=True)
)

per_shape["method"] = (
    per_shape["method"].astype(str)
)


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

summary_rows = []

groups = []

for (
    gpu,
    method,
), group in per_shape.groupby(
    ["gpu", "method"],
    sort=False,
):
    groups.append(
        (gpu, method, group)
    )

for method, group in per_shape.groupby(
    "method",
    sort=False,
):
    groups.append(
        ("ALL", method, group)
    )


for gpu, method, group in groups:
    regret = group[
        "regret_pct"
    ].to_numpy(dtype=float)

    ratios = (
        group["selected_latency_ms"]
        .to_numpy(dtype=float)
        /
        group["oracle_latency_ms"]
        .to_numpy(dtype=float)
    )

    summary_rows.append(
        {
            "gpu": gpu,
            "method": method,
            "n_shapes": len(group),
            "mean_regret_pct": float(
                np.mean(regret)
            ),
            "median_regret_pct": float(
                np.median(regret)
            ),
            "p95_regret_pct": float(
                np.quantile(
                    regret,
                    0.95,
                )
            ),
            "max_regret_pct": float(
                np.max(regret)
            ),
            "within_1pct": int(
                np.sum(regret <= 1.0)
            ),
            "within_2pct": int(
                np.sum(regret <= 2.0)
            ),
            "oracle_geomean_speedup": (
                geometric_mean(ratios)
            ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ---------------------------------------------------------------------
# Save policy provenance
# ---------------------------------------------------------------------

policy_record = {
    "selection_split": "training only",
    "global_static_objective": (
        "minimum summed mean latency over all "
        "training GPU-shape cases, independently per path"
    ),
    "random_fixed_definition": (
        "uniform expectation over all fixed configuration "
        "triplets in the training candidate space"
    ),
    "random_fixed_evaluation": (
        "no holdout measurements used for selection; "
        "expected latency evaluated on holdout only"
    ),
    "majority_winner_definition": (
        "for each GPU and execution path, select the candidate "
        "that is fastest on the largest number of training shapes"
    ),
    "majority_winner_policies": (
        majority_winner_policies
    ),
    "default_policy": default_policy,
    "global_static_policy": global_policy,
    "architecture_static_policies": (
        architecture_policies
    ),
}

# ---------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------

per_shape_path = (
    OUTPUT_DIR
    / "selector_baselines_per_shape.csv"
)

summary_path = (
    OUTPUT_DIR
    / "selector_baselines_summary.csv"
)

policy_path = (
    OUTPUT_DIR
    / "selector_baselines_policy.json"
)

per_shape.to_csv(
    per_shape_path,
    index=False,
)

summary.to_csv(
    summary_path,
    index=False,
)

policy_path.write_text(
    json.dumps(
        policy_record,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print("=" * 80)
print("SELECTOR BASELINE EVALUATION")
print("=" * 80)

print("\nDefault:")
print(
    json.dumps(
        default_policy,
        indent=2,
    )
)

print("\nGlobal-static selected from training only:")
print(
    json.dumps(
        global_policy,
        indent=2,
    )
)

print("\nPathTile architecture-static policies:")
print(
    json.dumps(
        architecture_policies,
        indent=2,
    )
)

print("\nHoldout summary:")
print(
    summary.to_string(
        index=False
    )
)

print("\nOutputs:")
print(per_shape_path)
print(summary_path)
print(policy_path)
