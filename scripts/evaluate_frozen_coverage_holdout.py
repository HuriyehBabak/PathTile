#!/usr/bin/env python3
"""Evaluate the locked budget-6 coverage policy on existing holdout data.

This evaluation is CPU-only. It reuses the previously measured official
holdout candidate results and does not execute CUDA kernels, retune launch
configurations, or modify the frozen policy.

The coverage policy must have been constructed from training data only and
frozen before this evaluation script reads the holdout measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scripts.aggregate_pathtile_results import (
    GPU_CONFIGS,
    _selected_phase_time,
    aggregate_candidate_root,
)
from scripts.artifact_common import (
    geometric_mean,
    normalize_roundoff,
)


EXPECTED_BUDGET = 6


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def load_frozen_policy(
    policy_path: Path,
) -> tuple[dict, dict[str, dict]]:
    """Load and validate the locked training-only coverage policy."""

    policy = json.loads(
        policy_path.read_text(encoding="utf-8")
    )

    if policy.get("shape_budget") != EXPECTED_BUDGET:
        raise ValueError(
            f"Expected shape budget {EXPECTED_BUDGET}, "
            f"found {policy.get('shape_budget')}"
        )

    if policy.get("official_holdout_read") is not False:
        raise ValueError(
            "Frozen policy metadata indicates holdout access"
        )

    if policy.get("new_cuda_measurements") is not False:
        raise ValueError(
            "Unexpected CUDA-measurement marker in frozen policy"
        )

    gpu_items = policy.get("gpu_policies")
    if not isinstance(gpu_items, list):
        raise ValueError(
            "Frozen policy does not contain gpu_policies"
        )

    by_gpu: dict[str, dict] = {}

    for item in gpu_items:
        gpu_key = str(item["gpu_key"])

        if gpu_key in by_gpu:
            raise ValueError(
                f"Duplicate frozen policy for {gpu_key}"
            )

        if int(item.get("shape_budget", EXPECTED_BUDGET)) != EXPECTED_BUDGET:
            raise ValueError(
                f"{gpu_key}: unexpected shape budget"
            )

        selected_shapes = item.get("selected_shapes", [])

        if len(selected_shapes) != EXPECTED_BUDGET:
            raise ValueError(
                f"{gpu_key}: expected {EXPECTED_BUDGET} "
                f"selected shapes, found {len(selected_shapes)}"
            )

        selected = item.get(
            "selected_candidate_values"
        )

        required = {
            "fwd",
            "bwd_input",
            "bwd_weight",
        }

        if (
            not isinstance(selected, dict)
            or set(selected) != required
        ):
            raise ValueError(
                f"{gpu_key}: malformed selected policy: "
                f"{selected}"
            )

        by_gpu[gpu_key] = item

    expected_gpu_keys = {
        gpu_key
        for _, gpu_key, _ in GPU_CONFIGS
    }

    if set(by_gpu) != expected_gpu_keys:
        raise ValueError(
            "Frozen policy GPU set mismatch: "
            f"expected {sorted(expected_gpu_keys)}, "
            f"found {sorted(by_gpu)}"
        )

    return policy, by_gpu


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(
            "results/pathtile_selector/"
            "coverage_frozen_budget6_v1/"
            "coverage_policy.json"
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/pathtile_selector/"
            "coverage_budget6_holdout_eval_v1"
        ),
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    policy_path = resolve_path(
        root,
        args.policy,
    )

    out = resolve_path(
        root,
        args.out,
    )

    if not policy_path.exists():
        raise FileNotFoundError(
            f"Frozen policy not found: {policy_path}"
        )

    if out.exists():
        raise FileExistsError(
            f"{out} already exists. "
            "Use a new --out directory."
        )

    # Important:
    # Load and validate the locked policy before opening
    # any official holdout measurements.
    frozen_policy, policies_by_gpu = (
        load_frozen_policy(policy_path)
    )

    policy_hash = sha256(policy_path)

    rows: list[dict] = []

    for (
        gpu_label,
        gpu_key,
        expected_device_name,
    ) in GPU_CONFIGS:

        policy_item = policies_by_gpu[gpu_key]

        if (
            policy_item["device_name"]
            != expected_device_name
        ):
            raise ValueError(
                f"{gpu_key}: device mismatch: "
                f"{policy_item['device_name']} != "
                f"{expected_device_name}"
            )

        selected_raw = (
            policy_item[
                "selected_candidate_values"
            ]
        )

        selected = {
            "fwd": int(selected_raw["fwd"]),
            "bwd_input":
                int(selected_raw["bwd_input"]),
            "bwd_weight":
                int(selected_raw["bwd_weight"]),
        }

        # Holdout data is accessed only here,
        # after policy validation has completed.
        relative = Path(
            f"results/pathtile_candidates/"
            f"{gpu_key}_holdout"
        )

        (
            _,
            candidates,
            _,
            combined,
        ) = aggregate_candidate_root(
            root,
            relative,
        )

        for oracle_row in combined.itertuples(
            index=False
        ):
            fwd_ms = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "fwd",
                selected["fwd"],
            )

            dx_ms = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_input",
                selected["bwd_input"],
            )

            dk_ms = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_weight",
                selected["bwd_weight"],
            )

            frozen_total_ms = (
                fwd_ms
                + dx_ms
                + dk_ms
            )

            oracle_total_ms = float(
                oracle_row.oracle_total_ms
            )

            absolute_gap_us = normalize_roundoff(
                (
                    frozen_total_ms
                    - oracle_total_ms
                )
                * 1000.0
            )

            regret_pct = normalize_roundoff(
                (
                    frozen_total_ms
                    / oracle_total_ms
                    - 1.0
                )
                * 100.0
            )

            exact_oracle_schedule = bool(
                selected["fwd"]
                == int(
                    oracle_row.fwd_winner_warps
                )
                and selected["bwd_input"]
                == int(
                    oracle_row
                    .bwd_input_winner_warps
                )
                and selected["bwd_weight"]
                == int(
                    oracle_row
                    .bwd_weight_winner_block
                )
            )

            rows.append(
                {
                    "gpu": gpu_label,
                    "gpu_key": gpu_key,
                    "device_name":
                        expected_device_name,
                    "config_id":
                        oracle_row.config_id,
                    "sweep_group":
                        oracle_row.sweep_group,
                    "B": int(oracle_row.B),
                    "H": int(oracle_row.H),
                    "L": int(oracle_row.L),
                    "K": int(oracle_row.K),
                    "selected_fwd_warps":
                        selected["fwd"],
                    "selected_bwd_input_warps":
                        selected["bwd_input"],
                    "selected_bwd_weight_block":
                        selected["bwd_weight"],
                    "oracle_fwd_warps":
                        int(
                            oracle_row
                            .fwd_winner_warps
                        ),
                    "oracle_bwd_input_warps":
                        int(
                            oracle_row
                            .bwd_input_winner_warps
                        ),
                    "oracle_bwd_weight_block":
                        int(
                            oracle_row
                            .bwd_weight_winner_block
                        ),
                    "frozen_fwd_ms": fwd_ms,
                    "frozen_bwd_input_ms":
                        dx_ms,
                    "frozen_bwd_weight_ms":
                        dk_ms,
                    "frozen_total_ms":
                        frozen_total_ms,
                    "oracle_total_ms":
                        oracle_total_ms,
                    "absolute_gap_us":
                        absolute_gap_us,
                    "regret_pct":
                        regret_pct,
                    "exact_oracle_schedule":
                        exact_oracle_schedule,
                }
            )

    per_shape = pd.DataFrame(rows)

    if len(per_shape) != 30:
        raise ValueError(
            "Expected 30 official holdout "
            f"shape-GPU cases, found "
            f"{len(per_shape)}"
        )

    summary_rows: list[dict] = []

    groups = [
        (gpu, group)
        for gpu, group
        in per_shape.groupby(
            "gpu",
            sort=False,
        )
    ]

    groups.append(
        ("ALL", per_shape)
    )

    for label, group in groups:
        regret = group["regret_pct"]

        worst_index = regret.idxmax()
        worst = group.loc[worst_index]

        summary_rows.append(
            {
                "gpu": label,
                "cases": len(group),
                "exact_oracle_schedule_cases":
                    int(
                        group[
                            "exact_oracle_schedule"
                        ].sum()
                    ),
                "mean_regret_pct":
                    float(regret.mean()),
                "median_regret_pct":
                    float(regret.median()),
                "max_regret_pct":
                    float(regret.max()),
                "within_1pct_cases":
                    int(
                        (regret <= 1.0).sum()
                    ),
                "within_2pct_cases":
                    int(
                        (regret <= 2.0).sum()
                    ),
                "runtime_weighted_regret_pct":
                    (
                        float(
                            group[
                                "frozen_total_ms"
                            ].sum()
                        )
                        / float(
                            group[
                                "oracle_total_ms"
                            ].sum()
                        )
                        - 1.0
                    )
                    * 100.0,
                "oracle_geomean_speedup_vs_frozen":
                    geometric_mean(
                        group[
                            "frozen_total_ms"
                        ]
                        / group[
                            "oracle_total_ms"
                        ]
                    ),
                "worst_config_id":
                    str(
                        worst["config_id"]
                    ),
                "worst_absolute_gap_us":
                    float(
                        worst[
                            "absolute_gap_us"
                        ]
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    metadata = {
        "schema_version": 1,
        "evaluation_created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),
        "evaluation_type":
            (
                "locked-policy evaluation "
                "using existing official "
                "holdout measurements"
            ),
        "shape_budget":
            EXPECTED_BUDGET,
        "policy_path":
            str(policy_path),
        "policy_sha256":
            policy_hash,
        "policy_frozen_at_utc":
            frozen_policy.get(
                "frozen_at_utc"
            ),
        "policy_official_holdout_read":
            frozen_policy.get(
                "official_holdout_read"
            ),
        "policy_new_cuda_measurements":
            frozen_policy.get(
                "new_cuda_measurements"
            ),
        "evaluation_cuda_execution":
            False,
        "evaluation_retuning":
            False,
        "holdout_cases":
            len(per_shape),
        "selection_measurement_cells_per_gpu":
            78,
        "full_training_measurement_cells_per_gpu":
            260,
        "selection_measurement_cell_reduction_pct":
            70.0,
        "note":
            (
                "The locked budget-6 coverage "
                "policy is evaluated without "
                "modification. Holdout data are "
                "used only for post-freeze "
                "performance evaluation."
            ),
        "script_sha256":
            sha256(
                Path(__file__)
            ),
    }

    out.mkdir(
        parents=True,
        exist_ok=False,
    )

    per_shape.to_csv(
        out / "per_shape.csv",
        index=False,
    )

    summary.to_csv(
        out / "summary.csv",
        index=False,
    )

    (
        out
        / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print(
        "\nFrozen configurations:"
    )

    for (
        gpu_label,
        gpu_key,
        _,
    ) in GPU_CONFIGS:
        selected = (
            policies_by_gpu[gpu_key][
                "selected_candidate_values"
            ]
        )

        print(
            f"{gpu_label}: "
            f"FWD={selected['fwd']}, "
            f"dX={selected['bwd_input']}, "
            f"dK={selected['bwd_weight']}"
        )

    print(
        f"\nSaved evaluation to: {out}"
    )

    print(
        "No CUDA execution or retuning "
        "was performed."
    )


if __name__ == "__main__":
    main()
