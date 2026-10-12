#!/usr/bin/env python3
"""Rebuild A6000/V100 shape-sweep aggregates from saved trial summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.artifact_common import read_csv, resolve_project_root, sample_std, write_csv


PYTORCH_METHOD = "PyTorch-cuDNN"
WARP_METHOD = "Warp"
SWEEP_ROOTS = [
    Path("results/shape_sweep_ampere/a6000_full"),
    Path("results/shape_sweep_volta/v100_full_w16_b256"),
]


def aggregate_one(root: Path, relative_root: Path) -> dict[Path, pd.DataFrame]:
    output_root = root / relative_root
    paths = sorted(output_root.glob("trial*/trial_summary.csv"))
    if len(paths) != 3:
        raise RuntimeError(f"{relative_root}: expected 3 trial summaries, found {len(paths)}")

    combined = pd.concat([read_csv(path) for path in paths], ignore_index=True)
    trial_ids = sorted(int(value) for value in combined["trial_id"].unique())
    if trial_ids != [1, 2, 3]:
        raise RuntimeError(f"{relative_root}: invalid trial IDs {trial_ids}")

    completeness = combined.groupby(
        ["config_id", "method", "phase"], sort=False
    )["trial_id"].nunique()
    if not bool((completeness == 3).all()):
        raise RuntimeError(
            f"{relative_root}: incomplete shape/method/phase trials:\n"
            f"{completeness[completeness != 3]}"
        )

    group_columns = [
        "config_id",
        "sweep_group",
        "B",
        "H",
        "L",
        "K",
        "device_name",
        "method",
        "phase",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in combined.groupby(group_columns, sort=False):
        trial_means = group["mean_ms"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys, strict=True))
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

    index_columns = ["config_id", "sweep_group", "B", "H", "L", "K", "phase"]
    comparison = (
        aggregate.pivot_table(
            index=index_columns,
            columns="method",
            values="mean_ms",
            aggfunc="first",
        )
        .reset_index()
        .rename(
            columns={
                PYTORCH_METHOD: "pytorch_mean_ms",
                WARP_METHOD: "warp_mean_ms",
            }
        )
    )
    if {"pytorch_mean_ms", "warp_mean_ms"} - set(comparison.columns):
        raise ValueError(f"{relative_root}: missing PyTorch or Warp measurements")
    comparison["warp_speedup_vs_pytorch"] = (
        comparison["pytorch_mean_ms"] / comparison["warp_mean_ms"]
    )
    comparison["warp_reduction_vs_pytorch_pct"] = (
        1.0 - comparison["warp_mean_ms"] / comparison["pytorch_mean_ms"]
    ) * 100.0
    comparison = comparison.sort_values(index_columns, kind="stable").reset_index(
        drop=True
    )

    correctness_frames = []
    for trial in trial_ids:
        path = output_root / f"trial{trial}/correctness.csv"
        correctness_frames.append(read_csv(path, ["all_correct"]))
    correctness = pd.concat(correctness_frames, ignore_index=True)
    if len(correctness) != 30 or not bool(correctness["all_correct"].all()):
        raise AssertionError(
            f"{relative_root}: expected 30/30 correctness rows, got "
            f"{int(correctness['all_correct'].sum())}/{len(correctness)}"
        )

    total = comparison[comparison["phase"] == "total"].drop(columns="phase")
    total = total.reset_index(drop=True)
    return {
        relative_root / "all_trial_summaries.csv": combined,
        relative_root / "aggregate_summary.csv": aggregate,
        relative_root / "aggregate_comparison.csv": comparison,
        relative_root / "aggregate_total_comparison.csv": total,
    }


def build_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    outputs: dict[Path, pd.DataFrame] = {}
    for relative_root in SWEEP_ROOTS:
        outputs.update(aggregate_one(root, relative_root))
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
    print(f"Shape-sweep aggregation/correctness: PASS ({len(outputs)} files)")


if __name__ == "__main__":
    main()
