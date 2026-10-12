#!/usr/bin/env python3
"""Rebuild the final paired A6000 end-to-end aggregates without a GPU."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.artifact_common import (
    geometric_mean,
    read_csv,
    resolve_project_root,
    sample_std,
    write_csv,
)


BACKENDS = ("warp", "pytorch")
STEADY_EPOCHS = [2, 3, 4, 5]
SERIAL_ROOT = Path("results/end_to_end/a6000_final_serial")


def build_per_trial(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in (1, 2, 3):
        backend_losses: dict[str, pd.DataFrame] = {}
        backend_rmsle: dict[str, float] = {}
        for backend in BACKENDS:
            trial_root = root / SERIAL_ROOT / f"trial{trial}/{backend}"
            epochs = read_csv(
                trial_root / "training_epoch_system_summary.csv",
                ["epoch", "train_wall_time_s", "peak_memory_gib"],
            )
            stages = read_csv(
                trial_root / "training_batch_stage_times.csv",
                ["epoch", "batch_id", "model_fwd_gpu_ms", "backward_gpu_ms"],
            )
            losses = read_csv(
                trial_root / "training_kernel_epoch_summary.csv",
                ["epoch", "train_loss"],
            )
            test = read_csv(
                trial_root / "training_final_test_metrics.csv",
                ["test_rmsle", "batch_size", "saved_epochs", "train_backend"],
            )

            epoch_ids = sorted(int(value) for value in epochs["epoch"].tolist())
            if epoch_ids != STEADY_EPOCHS:
                raise ValueError(f"{trial_root}: expected epochs {STEADY_EPOCHS}, got {epoch_ids}")
            if len(test) != 1 or str(test.iloc[0]["train_backend"]) != backend:
                raise ValueError(f"{trial_root}: invalid test metadata")

            batch_counts = stages.groupby("epoch")["batch_id"].nunique()
            if list(batch_counts.index.astype(int)) != STEADY_EPOCHS:
                raise ValueError(f"{trial_root}: invalid stage epochs")
            if batch_counts.nunique() != 1:
                raise ValueError(f"{trial_root}: inconsistent batches per epoch")
            batches_per_epoch = int(batch_counts.iloc[0])
            if batches_per_epoch != 39:
                raise ValueError(f"{trial_root}: expected 39 batches, got {batches_per_epoch}")

            gpu_columns = [column for column in stages if column.endswith("_gpu_ms")]
            expected_gpu_columns = {
                "transfer_gpu_ms",
                "model_fwd_gpu_ms",
                "loss_gpu_ms",
                "backward_gpu_ms",
                "clip_grad_gpu_ms",
                "optimizer_step_gpu_ms",
            }
            if set(gpu_columns) != expected_gpu_columns:
                raise ValueError(f"{trial_root}: unexpected GPU-stage columns {gpu_columns}")

            epoch_wall_mean = float(epochs["train_wall_time_s"].mean())
            batch_size = int(test.iloc[0]["batch_size"])
            test_rmsle = float(test.iloc[0]["test_rmsle"])
            rows.append(
                {
                    "trial": trial,
                    "backend": backend,
                    "saved_epochs": len(epochs),
                    "batches_per_epoch": float(batches_per_epoch),
                    "epoch_wall_mean_s": epoch_wall_mean,
                    "epoch_wall_within_std_s": sample_std(
                        epochs["train_wall_time_s"]
                    ),
                    "throughput_samples_s": batch_size
                    * batches_per_epoch
                    / epoch_wall_mean,
                    "model_fwd_mean_ms": float(stages["model_fwd_gpu_ms"].mean()),
                    "backward_mean_ms": float(stages["backward_gpu_ms"].mean()),
                    "gpu_stages_mean_ms": float(
                        stages[gpu_columns].sum(axis=1).mean()
                    ),
                    "peak_memory_gib": float(epochs["peak_memory_gib"].max()),
                    "test_rmsle": test_rmsle,
                }
            )
            backend_losses[backend] = losses.sort_values("epoch").reset_index(drop=True)
            backend_rmsle[backend] = test_rmsle

        warp_losses = backend_losses["warp"]
        pytorch_losses = backend_losses["pytorch"]
        if not np.allclose(
            warp_losses["train_loss"],
            pytorch_losses["train_loss"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"Trial {trial}: train losses differ between backends")
        validation_columns = [
            column
            for column in ("val_batch_mean_rmsle", "val_loss")
            if column in warp_losses.columns and column in pytorch_losses.columns
        ]
        if len(validation_columns) != 1:
            raise ValueError(f"Trial {trial}: ambiguous validation metric columns")
        validation_column = validation_columns[0]
        if not np.allclose(
            warp_losses[validation_column],
            pytorch_losses[validation_column],
            rtol=0.0,
            atol=1e-12,
        ):
            raise AssertionError(f"Trial {trial}: validation metrics differ")
        if abs(backend_rmsle["warp"] - backend_rmsle["pytorch"]) > 1e-12:
            raise AssertionError(f"Trial {trial}: global test RMSLE differs")

    return pd.DataFrame(rows)


def build_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    per_trial = build_per_trial(root)

    paired_rows: list[dict[str, float | int]] = []
    for trial in (1, 2, 3):
        rows = per_trial[per_trial["trial"] == trial].set_index("backend")
        warp = rows.loc["warp"]
        pytorch = rows.loc["pytorch"]
        paired_rows.append(
            {
                "trial": trial,
                "epoch_speedup": float(pytorch["epoch_wall_mean_s"])
                / float(warp["epoch_wall_mean_s"]),
                "epoch_reduction_pct": (
                    1.0
                    - float(warp["epoch_wall_mean_s"])
                    / float(pytorch["epoch_wall_mean_s"])
                )
                * 100.0,
                "throughput_speedup": float(warp["throughput_samples_s"])
                / float(pytorch["throughput_samples_s"]),
                "model_fwd_speedup": float(pytorch["model_fwd_mean_ms"])
                / float(warp["model_fwd_mean_ms"]),
                "backward_speedup": float(pytorch["backward_mean_ms"])
                / float(warp["backward_mean_ms"]),
                "gpu_stages_speedup": float(pytorch["gpu_stages_mean_ms"])
                / float(warp["gpu_stages_mean_ms"]),
                "peak_memory_delta_gib": float(warp["peak_memory_gib"])
                - float(pytorch["peak_memory_gib"]),
                "rmsle_abs_difference": abs(
                    float(warp["test_rmsle"]) - float(pytorch["test_rmsle"])
                ),
            }
        )
    paired = pd.DataFrame(paired_rows)

    backend_rows: list[dict[str, Any]] = []
    metrics = [
        "epoch_wall_mean_s",
        "throughput_samples_s",
        "model_fwd_mean_ms",
        "backward_mean_ms",
        "gpu_stages_mean_ms",
        "peak_memory_gib",
        "test_rmsle",
    ]
    for backend in BACKENDS:
        group = per_trial[per_trial["backend"] == backend]
        row: dict[str, Any] = {"backend": backend, "trials": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_between_trial_std"] = sample_std(group[metric])
        backend_rows.append(row)
    backend = pd.DataFrame(backend_rows)

    paper = pd.DataFrame(
        [
            {
                "trials": len(paired),
                "epoch_speedup_geomean": geometric_mean(paired["epoch_speedup"]),
                "epoch_speedup_min": float(paired["epoch_speedup"].min()),
                "epoch_speedup_max": float(paired["epoch_speedup"].max()),
                "epoch_reduction_mean_pct": float(
                    paired["epoch_reduction_pct"].mean()
                ),
                "model_fwd_speedup_geomean": geometric_mean(
                    paired["model_fwd_speedup"]
                ),
                "backward_speedup_geomean": geometric_mean(
                    paired["backward_speedup"]
                ),
                "gpu_stages_speedup_geomean": geometric_mean(
                    paired["gpu_stages_speedup"]
                ),
                "max_rmsle_abs_difference": float(
                    paired["rmsle_abs_difference"].max()
                ),
                "max_abs_memory_delta_gib": float(
                    paired["peak_memory_delta_gib"].abs().max()
                ),
            }
        ]
    )

    return {
        SERIAL_ROOT / "final_per_trial.csv": per_trial,
        SERIAL_ROOT / "final_paired_speedups.csv": paired,
        SERIAL_ROOT / "final_backend_aggregate.csv": backend,
        SERIAL_ROOT / "final_paper_summary.csv": paper,
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
    print(f"Serial end-to-end aggregation/correctness: PASS ({len(outputs)} files)")


if __name__ == "__main__":
    main()
