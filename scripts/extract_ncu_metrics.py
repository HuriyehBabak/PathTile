#!/usr/bin/env python3
"""Extract the canonical A100 Nsight Compute metrics from exported details CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.artifact_common import resolve_project_root, write_csv


PHASE_FILES = [
    ("Forward", "results/nsight/a100_main/fwd/a100_main_fwd_details.csv"),
    (
        "Bwd-Input",
        "results/nsight/a100_main/bwd_input/a100_main_bwd_input_details.csv",
    ),
    (
        "Bwd-Weight",
        "results/nsight/a100_main/bwd_weight/a100_main_bwd_weight_details.csv",
    ),
]

METRICS = [
    ("GPU Speed Of Light Throughput", "Memory Throughput", "%"),
    ("GPU Speed Of Light Throughput", "DRAM Throughput", "%"),
    ("GPU Speed Of Light Throughput", "Duration", "ms"),
    ("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%"),
    ("Memory Workload Analysis", "Memory Throughput", "Gbyte/s"),
    ("Memory Workload Analysis", "L1/TEX Hit Rate", "%"),
    ("Memory Workload Analysis", "L2 Hit Rate", "%"),
    ("Scheduler Statistics", "Active Warps Per Scheduler", "warp"),
    ("Scheduler Statistics", "Eligible Warps Per Scheduler", "warp"),
    ("Warp State Statistics", "Warp Cycles Per Issued Instruction", "cycle"),
    ("Launch Statistics", "Registers Per Thread", "register/thread"),
    ("Launch Statistics", "Waves Per SM", ""),
    ("Occupancy", "Theoretical Occupancy", "%"),
    ("Occupancy", "Achieved Occupancy", "%"),
]


def build_output(root: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for phase, relative_path in PHASE_FILES:
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        details = pd.read_csv(path, dtype=str, keep_default_na=False)
        required = {"Section Name", "Metric Name", "Metric Unit", "Metric Value"}
        missing = sorted(required - set(details.columns))
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        if details["Kernel Name"].nunique() != 1:
            raise ValueError(f"{path} contains more than one profiled kernel")
        for section, metric, unit in METRICS:
            selected = details[
                (details["Section Name"] == section)
                & (details["Metric Name"] == metric)
                & (details["Metric Unit"] == unit)
            ]
            if len(selected) != 1:
                raise ValueError(
                    f"{path}: expected one row for {(section, metric, unit)}, "
                    f"found {len(selected)}"
                )
            rows.append(
                {
                    "phase": phase,
                    "Section Name": section,
                    "Metric Name": metric,
                    "Metric Unit": unit,
                    "Metric Value": selected.iloc[0]["Metric Value"],
                }
            )
    return pd.DataFrame(rows)


def build_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    return {
        Path("results/nsight/a100_main/a100_main_ncu_key_metrics.csv"): build_output(
            root
        )
    }


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
    print(f"A100 Nsight metric extraction: PASS ({len(outputs)} file)")


if __name__ == "__main__":
    main()
