#!/usr/bin/env python3
"""Rebuild PathTile candidate, frozen-policy, and holdout aggregates CPU-only."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.artifact_common import (
    geometric_mean,
    normalize_roundoff,
    read_csv,
    resolve_project_root,
    sample_std,
    write_csv,
)


PHASES = ("fwd", "bwd_input", "bwd_weight")
DEFAULT_WARPS = 16
DEFAULT_BWDK_BLOCK = 128
GPU_CONFIGS = [
    ("A100", "a100", "NVIDIA A100-PCIE-40GB"),
    ("A6000", "a6000", "NVIDIA RTX A6000"),
    ("V100", "v100", "Tesla V100-SXM3-32GB"),
]


AGGREGATE_GROUP_COLUMNS = [
    "config_id",
    "sweep_group",
    "B",
    "H",
    "L",
    "K",
    "device_name",
    "candidate_kind",
    "candidate_value",
    "schedule_id",
    "warps_per_block",
    "bwdk_btile",
    "bwdk_block",
    "phase",
]


def oracle_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "config_id",
        "sweep_group",
        "B",
        "H",
        "L",
        "K",
        "device_name",
        "phase",
        "candidate_kind",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in summary.groupby(group_columns, sort=False):
        ordered = group.sort_values("mean_ms", kind="stable").reset_index(drop=True)
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else best
        default_value = (
            DEFAULT_WARPS
            if str(best["candidate_kind"]) == "warps_per_block"
            else DEFAULT_BWDK_BLOCK
        )
        default_rows = ordered[ordered["candidate_value"] == default_value]
        default_ms = (
            float(default_rows.iloc[0]["mean_ms"])
            if not default_rows.empty
            else float("nan")
        )
        best_ms = float(best["mean_ms"])
        second_ms = float(second["mean_ms"])
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "winner_schedule_id": best["schedule_id"],
                "winner_value": int(best["candidate_value"]),
                "winner_mean_ms": best_ms,
                "winner_within_trial_std_ms": float(best.get("std_ms", float("nan"))),
                "winner_between_trial_std_ms": float(
                    best.get("between_trial_std_ms", float("nan"))
                ),
                "second_schedule_id": second["schedule_id"],
                "second_value": int(second["candidate_value"]),
                "second_mean_ms": second_ms,
                "winner_margin_vs_second_pct": (
                    (second_ms / best_ms - 1.0) * 100.0
                    if best_ms > 0.0
                    else float("nan")
                ),
                "paper_default_value": default_value,
                "paper_default_mean_ms": default_ms,
                "paper_default_regret_pct": (
                    (default_ms / best_ms - 1.0) * 100.0
                    if best_ms > 0.0 and np.isfinite(default_ms)
                    else float("nan")
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    order = {phase: index for index, phase in enumerate(PHASES)}
    result["_phase_order"] = result["phase"].map(order)
    return (
        result.sort_values(["config_id", "_phase_order"], kind="stable")
        .drop(columns="_phase_order")
        .reset_index(drop=True)
    )


def combined_oracle(oracle: pd.DataFrame) -> pd.DataFrame:
    index_columns = ["config_id", "sweep_group", "B", "H", "L", "K", "device_name"]
    rows: list[dict[str, Any]] = []
    for keys, group in oracle.groupby(index_columns, sort=False):
        if set(group["phase"]) != set(PHASES):
            raise RuntimeError(f"Missing path result for {keys}")
        phase_rows = {row.phase: row for row in group.itertuples(index=False)}
        oracle_total = float(group["winner_mean_ms"].sum())
        default_values = group["paper_default_mean_ms"].to_numpy(dtype=float)
        default_total = (
            float(default_values.sum())
            if np.isfinite(default_values).all()
            else float("nan")
        )
        row = dict(zip(index_columns, keys, strict=True))
        row.update(
            {
                "fwd_winner_warps": int(phase_rows["fwd"].winner_value),
                "bwd_input_winner_warps": int(
                    phase_rows["bwd_input"].winner_value
                ),
                "bwd_weight_winner_block": int(
                    phase_rows["bwd_weight"].winner_value
                ),
                "oracle_total_ms": oracle_total,
                "paper_default_total_ms": default_total,
                "paper_default_total_regret_pct": (
                    (default_total / oracle_total - 1.0) * 100.0
                    if oracle_total > 0.0 and np.isfinite(default_total)
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config_id", kind="stable").reset_index(
        drop=True
    )


def aggregate_candidate_root(
    root: Path, relative_root: Path
) -> tuple[dict[Path, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_root = root / relative_root
    paths = sorted(output_root.glob("trial*/trial_summary.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"{relative_root}: expected 3 trials, found {len(paths)}")
    combined_trials = pd.concat([read_csv(path) for path in paths], ignore_index=True)
    trial_ids = sorted(int(value) for value in combined_trials["trial_id"].unique())
    if trial_ids != [1, 2, 3]:
        raise RuntimeError(f"{relative_root}: invalid trial IDs {trial_ids}")

    completeness = combined_trials.groupby(
        ["config_id", "phase", "candidate_kind", "candidate_value"], sort=False
    )["trial_id"].nunique()
    if not bool((completeness == 3).all()):
        raise RuntimeError(f"{relative_root}: incomplete candidate trials")

    rows: list[dict[str, Any]] = []
    for keys, group in combined_trials.groupby(AGGREGATE_GROUP_COLUMNS, sort=False):
        trial_means = group["mean_ms"].to_numpy(dtype=float)
        row = dict(zip(AGGREGATE_GROUP_COLUMNS, keys, strict=True))
        row.update(
            {
                "trials": int(group["trial_id"].nunique()),
                "mean_ms": float(trial_means.mean()),
                "between_trial_std_ms": sample_std(trial_means),
                "min_trial_mean_ms": float(trial_means.min()),
                "max_trial_mean_ms": float(trial_means.max()),
                "mean_within_trial_std_ms": float(group["std_ms"].mean()),
            }
        )
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    oracle = oracle_from_summary(aggregate)
    combined = combined_oracle(oracle)
    winner_frequency = (
        oracle.groupby(["phase", "candidate_kind", "winner_value"], as_index=False)
        .size()
        .rename(columns={"size": "shapes_won"})
        .sort_values(["phase", "shapes_won"], ascending=[True, False])
    )

    correctness = pd.concat(
        [
            read_csv(output_root / f"trial{trial}/correctness.csv", ["allclose"])
            for trial in trial_ids
        ],
        ignore_index=True,
    )
    if not bool(correctness["allclose"].all()):
        raise AssertionError(f"{relative_root}: candidate correctness failure")

    outputs = {
        relative_root / "all_trial_summaries.csv": combined_trials,
        relative_root / "aggregate_candidate_summary.csv": aggregate,
        relative_root / "aggregate_oracle.csv": oracle,
        relative_root / "aggregate_combined_oracle.csv": combined,
        relative_root / "winner_frequency.csv": winner_frequency,
    }
    return outputs, aggregate, oracle, combined


def select_architecture_policy(training: pd.DataFrame) -> dict[str, int]:
    selected: dict[str, int] = {}
    names = {
        "fwd": "fwd_warps",
        "bwd_input": "bwd_input_warps",
        "bwd_weight": "bwd_weight_block",
    }
    for phase, output_name in names.items():
        totals = (
            training[training["phase"] == phase]
            .groupby("candidate_value", as_index=False)["mean_ms"]
            .sum()
            .sort_values(["mean_ms", "candidate_value"], kind="stable")
        )
        if totals.empty:
            raise ValueError(f"No training candidates for {phase}")
        selected[output_name] = int(totals.iloc[0]["candidate_value"])
    return selected


def _selected_phase_time(
    candidates: pd.DataFrame, config_id: str, phase: str, value: int
) -> float:
    rows = candidates[
        (candidates["config_id"] == config_id)
        & (candidates["phase"] == phase)
        & (candidates["candidate_value"] == value)
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"Expected one row for {config_id}/{phase}/{value}, found {len(rows)}"
        )
    return float(rows.iloc[0]["mean_ms"])


def verify_frozen_policy(
    root: Path, selected_by_device: dict[str, dict[str, int]]
) -> dict[str, Any]:
    path = root / "results/pathtile_selector/frozen_policy/architecture_static_policy.json"
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("frozen_before_holdout") is not True:
        raise AssertionError("Frozen-policy marker is missing")
    if policy.get("profiles") != selected_by_device:
        raise AssertionError(
            "Stored frozen policy does not match a training-only reconstruction:\n"
            f"stored={policy.get('profiles')}\nrecomputed={selected_by_device}"
        )

    frozen_at = datetime.fromisoformat(str(policy["frozen_at_utc"]))
    for _, gpu_key, _ in GPU_CONFIGS:
        for trial in (1, 2, 3):
            metadata_path = (
                root
                / f"results/pathtile_candidates/{gpu_key}_holdout/trial{trial}/metadata.json"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            holdout_at = datetime.fromisoformat(str(metadata["created_at_utc"]))
            if not frozen_at < holdout_at:
                raise AssertionError(
                    f"Policy timestamp {frozen_at} is not before {metadata_path} ({holdout_at})"
                )
    return policy


def build_training_policy_outputs(
    training_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    selected_by_device: dict[str, dict[str, int]],
) -> dict[Path, pd.DataFrame]:
    per_shape_rows: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []
    for gpu_label, _, device_name in GPU_CONFIGS:
        candidates, combined = training_data[gpu_label]
        selected = selected_by_device[device_name]
        for oracle_row in combined.itertuples(index=False):
            fwd = _selected_phase_time(
                candidates, oracle_row.config_id, "fwd", selected["fwd_warps"]
            )
            dx = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_input",
                selected["bwd_input_warps"],
            )
            dk = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_weight",
                selected["bwd_weight_block"],
            )
            arch_total = fwd + dx + dk
            oracle_total = float(oracle_row.oracle_total_ms)
            global_total = float(oracle_row.paper_default_total_ms)
            per_shape_rows.append(
                {
                    "gpu": gpu_label,
                    "config_id": oracle_row.config_id,
                    "architecture_static_total_ms": arch_total,
                    "global_static_total_ms": global_total,
                    "oracle_total_ms": oracle_total,
                    "architecture_static_regret_pct": (
                        arch_total / oracle_total - 1.0
                    )
                    * 100.0,
                    "global_static_regret_pct": (
                        global_total / oracle_total - 1.0
                    )
                    * 100.0,
                }
            )

        regrets = combined["paper_default_total_regret_pct"]
        paper_rows.append(
            {
                "gpu": gpu_label,
                "shapes": len(combined),
                "nonzero_shapes": int((regrets != 0.0).sum()),
                "mean_static_regret_pct": float(regrets.mean()),
                "median_static_regret_pct": float(regrets.median()),
                "max_static_regret_pct": float(regrets.max()),
                "oracle_geomean_speedup": geometric_mean(
                    combined["paper_default_total_ms"] / combined["oracle_total_ms"]
                ),
            }
        )

    per_shape = pd.DataFrame(per_shape_rows)
    summary_rows: list[dict[str, Any]] = []
    for gpu_label, group in per_shape.groupby("gpu", sort=False):
        regret = group["architecture_static_regret_pct"]
        summary_rows.append(
            {
                "gpu": gpu_label,
                "shapes": len(group),
                "nonzero_shapes": int((regret != 0.0).sum()),
                "mean_arch_static_regret_pct": float(regret.mean()),
                "median_arch_static_regret_pct": float(regret.median()),
                "max_arch_static_regret_pct": float(regret.max()),
                "oracle_geomean_speedup_vs_arch_static": geometric_mean(
                    group["architecture_static_total_ms"] / group["oracle_total_ms"]
                ),
            }
        )
    return {
        Path("results/pathtile_selector/three_gpu_training_oracle_summary.csv"): pd.DataFrame(
            paper_rows
        ),
        Path("results/pathtile_selector/architecture_static_training/per_shape.csv"): per_shape,
        Path("results/pathtile_selector/architecture_static_training/summary.csv"): pd.DataFrame(
            summary_rows
        ),
    }


def build_holdout_outputs(
    holdout_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    selected_by_device: dict[str, dict[str, int]],
) -> dict[Path, pd.DataFrame]:
    frozen_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    for gpu_label, _, device_name in GPU_CONFIGS:
        candidates, combined = holdout_data[gpu_label]
        selected = selected_by_device[device_name]
        for oracle_row in combined.itertuples(index=False):
            fwd = _selected_phase_time(
                candidates, oracle_row.config_id, "fwd", selected["fwd_warps"]
            )
            dx = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_input",
                selected["bwd_input_warps"],
            )
            dk = _selected_phase_time(
                candidates,
                oracle_row.config_id,
                "bwd_weight",
                selected["bwd_weight_block"],
            )
            frozen_total = fwd + dx + dk
            oracle_total = float(oracle_row.oracle_total_ms)
            gap_us = normalize_roundoff((frozen_total - oracle_total) * 1000.0)
            regret = normalize_roundoff((frozen_total / oracle_total - 1.0) * 100.0)
            common = {
                "gpu": gpu_label,
                "device_name": device_name,
                "config_id": oracle_row.config_id,
                "sweep_group": oracle_row.sweep_group,
                "B": int(oracle_row.B),
                "H": int(oracle_row.H),
                "L": int(oracle_row.L),
                "K": int(oracle_row.K),
            }
            frozen_rows.append(
                {
                    **common,
                    "selected_fwd_warps": selected["fwd_warps"],
                    "selected_bwd_input_warps": selected["bwd_input_warps"],
                    "selected_bwd_weight_block": selected["bwd_weight_block"],
                    "static_fwd_ms": fwd,
                    "static_bwd_input_ms": dx,
                    "static_bwd_weight_ms": dk,
                    "static_total_ms": frozen_total,
                    "oracle_total_ms": oracle_total,
                    "absolute_gap_us": gap_us,
                    "regret_pct": regret,
                }
            )
            evaluation_rows.append(
                {
                    "config_id": oracle_row.config_id,
                    **{key: value for key, value in common.items() if key != "config_id"},
                    "fwd_winner_warps": int(oracle_row.fwd_winner_warps),
                    "bwd_input_winner_warps": int(
                        oracle_row.bwd_input_winner_warps
                    ),
                    "bwd_weight_winner_block": int(
                        oracle_row.bwd_weight_winner_block
                    ),
                    "oracle_total_ms": oracle_total,
                    "frozen_fwd_ms": fwd,
                    "frozen_bwd_input_ms": dx,
                    "frozen_bwd_weight_ms": dk,
                    "frozen_total_ms": frozen_total,
                    "absolute_gap_us": gap_us,
                    "frozen_regret_pct": regret,
                    "exact_oracle_schedule": bool(
                        selected["fwd_warps"] == int(oracle_row.fwd_winner_warps)
                        and selected["bwd_input_warps"]
                        == int(oracle_row.bwd_input_winner_warps)
                        and selected["bwd_weight_block"]
                        == int(oracle_row.bwd_weight_winner_block)
                    ),
                    "frozen_fwd_warps": selected["fwd_warps"],
                    "frozen_bwd_input_warps": selected["bwd_input_warps"],
                    "frozen_bwd_weight_block": selected["bwd_weight_block"],
                }
            )

    frozen = pd.DataFrame(frozen_rows)
    evaluation = pd.DataFrame(evaluation_rows)

    frozen_summary_rows: list[dict[str, Any]] = []
    groups = [(label, group) for label, group in frozen.groupby("gpu", sort=False)]
    groups.append(("ALL", frozen))
    for label, group in groups:
        regret = group["regret_pct"]
        worst = group.loc[regret.idxmax()]
        frozen_summary_rows.append(
            {
                "gpu": label,
                "shapes": len(group),
                "nonzero_shapes": int((regret != 0.0).sum()),
                "mean_regret_pct": float(regret.mean()),
                "median_regret_pct": float(regret.median()),
                "max_regret_pct": float(regret.max()),
                "runtime_weighted_regret_pct": (
                    float(group["static_total_ms"].sum())
                    / float(group["oracle_total_ms"].sum())
                    - 1.0
                )
                * 100.0,
                "oracle_geomean_speedup_vs_static": geometric_mean(
                    group["static_total_ms"] / group["oracle_total_ms"]
                ),
                "shapes_within_1pct": int((regret <= 1.0).sum()),
                "worst_config_id": worst["config_id"],
                "worst_absolute_gap_us": float(worst["absolute_gap_us"]),
            }
        )

    evaluation_summary_rows: list[dict[str, Any]] = []
    for label, group in evaluation.groupby("gpu", sort=False):
        regret = group["frozen_regret_pct"]
        evaluation_summary_rows.append(
            {
                "gpu": label,
                "shapes": len(group),
                "exact_oracle_schedule_shapes": int(
                    group["exact_oracle_schedule"].sum()
                ),
                "within_1pct_shapes": int((regret <= 1.0).sum()),
                "within_2pct_shapes": int((regret <= 2.0).sum()),
                "mean_frozen_regret_pct": float(regret.mean()),
                "median_frozen_regret_pct": float(regret.median()),
                "max_frozen_regret_pct": float(regret.max()),
                "mean_absolute_gap_us": float(group["absolute_gap_us"].mean()),
                "max_absolute_gap_us": float(group["absolute_gap_us"].max()),
                "oracle_geomean_speedup_vs_frozen": geometric_mean(
                    group["frozen_total_ms"] / group["oracle_total_ms"]
                ),
            }
        )

    return {
        Path("results/pathtile_selector/frozen_policy/holdout_per_shape.csv"): frozen,
        Path("results/pathtile_selector/frozen_policy/holdout_summary.csv"): pd.DataFrame(
            frozen_summary_rows
        ),
        Path(
            "results/pathtile_selector/holdout_evaluation/three_gpu_holdout_per_shape.csv"
        ): evaluation,
        Path(
            "results/pathtile_selector/holdout_evaluation/three_gpu_holdout_summary.csv"
        ): pd.DataFrame(evaluation_summary_rows),
    }


def build_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    outputs: dict[Path, pd.DataFrame] = {}
    training_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    holdout_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    selected_by_device: dict[str, dict[str, int]] = {}

    # Policy selection is reconstructed only from training results.
    for gpu_label, gpu_key, device_name in GPU_CONFIGS:
        relative = Path(f"results/pathtile_candidates/{gpu_key}_train")
        generated, aggregate, _, combined = aggregate_candidate_root(root, relative)
        outputs.update(generated)
        training_data[gpu_label] = (aggregate, combined)
        selected_by_device[device_name] = select_architecture_policy(aggregate)

    verify_frozen_policy(root, selected_by_device)

    # Holdout data is opened only after the stored pre-holdout policy is verified.
    for gpu_label, gpu_key, _ in GPU_CONFIGS:
        relative = Path(f"results/pathtile_candidates/{gpu_key}_holdout")
        generated, aggregate, _, combined = aggregate_candidate_root(root, relative)
        outputs.update(generated)
        holdout_data[gpu_label] = (aggregate, combined)

    outputs.update(build_training_policy_outputs(training_data, selected_by_device))
    outputs.update(build_holdout_outputs(holdout_data, selected_by_device))
    return outputs


def write_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    outputs = build_outputs(root)
    for relative_path, frame in outputs.items():
        write_csv(frame, root / relative_path)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    root = resolve_project_root(parse_args().project_root)
    outputs = write_outputs(root)
    print(f"PathTile training/frozen-holdout reconstruction: PASS ({len(outputs)} files)")


if __name__ == "__main__":
    main()
