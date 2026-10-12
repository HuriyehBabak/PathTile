#!/usr/bin/env python3
"""Rebuild and verify every paper artifact from saved measurements, CPU-only."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.aggregate_end_to_end import write_outputs as write_end_to_end
from scripts.aggregate_main_results import write_outputs as write_main
from scripts.aggregate_pathtile_results import write_outputs as write_pathtile
from scripts.aggregate_shape_results import write_outputs as write_shapes
from scripts.artifact_common import resolve_project_root
from scripts.extract_ncu_metrics import write_outputs as write_ncu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Keep existing paper figures/tables instead of regenerating them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = resolve_project_root(args.project_root)

    producers = [
        ("main/cross-GPU/torch.compile", write_main),
        ("shape generalization", write_shapes),
        ("PathTile training and holdout", write_pathtile),
        ("Nsight Compute", write_ncu),
        ("end-to-end", write_end_to_end),
    ]
    total = 0
    for label, producer in producers:
        outputs = producer(root)
        total += len(outputs)
        print(f"Rebuilt {label}: {len(outputs)} CSV files")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.aggregate_external_baselines",
            "--project-root",
            str(root),
        ],
        cwd=root,
        check=True,
    )
    external_outputs = [
        root
        / "results/external_baselines/final_summary"
        / name
        for name in (
            "external_baselines_all_metrics.csv",
            "external_baselines_paper_table.csv",
            "external_baselines_protocol_notes.csv",
            "external_baselines_paper_table.tex",
            "external_baselines_audit.json",
        )
    ]
    missing_external = [str(path) for path in external_outputs if not path.is_file()]
    if missing_external:
        raise FileNotFoundError(
            f"Missing reconstructed external-baseline outputs: {missing_external}"
        )
    total += len(external_outputs)
    print(f"Rebuilt external compiler baselines: {len(external_outputs)} outputs")

    if not args.skip_figures:
        subprocess.run(
            [
                sys.executable,
                str(root / "paper/generate_final_figures.py"),
                "--project-root",
                str(root),
            ],
            cwd=root,
            check=True,
        )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.verify_artifact",
            "--project-root",
            str(root),
        ],
        cwd=root,
        check=True,
    )
    print(f"Full CPU-only artifact rebuild: PASS ({total} canonical outputs written)")


if __name__ == "__main__":
    main()
