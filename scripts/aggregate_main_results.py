#!/usr/bin/env python3
"""Rebuild the main three-GPU and torch.compile aggregates without a GPU."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.artifact_common import read_csv, resolve_project_root, sample_std, write_csv


METHOD_ORDER = ["PyTorch", "Naive", "GMC", "Shared", "Warp"]
PHASE_ORDER = ["fwd", "bwd_input", "bwd_weight"]
PHASE_ALIASES = {
    "forward": "fwd",
    "Forward": "fwd",
    "fwd": "fwd",
    "backward_input": "bwd_input",
    "Backward Input": "bwd_input",
    "bwd_input": "bwd_input",
    "backward_kernel": "bwd_weight",
    "Backward Kernel": "bwd_weight",
    "bwd_weight": "bwd_weight",
}


@dataclass(frozen=True)
class MethodSource:
    root: str
    filename: str


@dataclass(frozen=True)
class GPUConfig:
    key: str
    device_name: str
    architecture: str
    compute_capability: str
    warp_w: int
    warp_bwdk_block: int
    summary_path: str
    sources: dict[str, MethodSource]


GPU_CONFIGS = [
    GPUConfig(
        key="a100",
        device_name="NVIDIA A100-PCIE-40GB",
        architecture="Ampere",
        compute_capability="8.0",
        warp_w=16,
        warp_bwdk_block=128,
        summary_path="results/a100_final_summary/a100_aggregate_summary.csv",
        sources={
            "PyTorch": MethodSource("results/a100_baselines/pytorch", "pytorch_runtime_summary_a100.csv"),
            "Naive": MethodSource("results/a100_baselines/naive", "naive_runtime_summary_a100.csv"),
            "GMC": MethodSource("results/a100_baselines/coalesced", "coalesced_runtime_summary.csv"),
            "Shared": MethodSource("results/a100_baselines/shared", "shared_runtime_summary.csv"),
            "Warp": MethodSource("results/warp_tiled_ampere/a100_final_w16_b128", "benchmark_summary.csv"),
        },
    ),
    GPUConfig(
        key="a6000",
        device_name="NVIDIA RTX A6000",
        architecture="Ampere",
        compute_capability="8.6",
        warp_w=16,
        warp_bwdk_block=128,
        summary_path="results/a6000_final_summary/a6000_aggregate_summary.csv",
        sources={
            "PyTorch": MethodSource("results/a6000_final_baselines/pytorch", "pytorch_runtime_summary_a6000.csv"),
            "Naive": MethodSource("results/a6000_final_baselines/naive", "naive_runtime_summary_a6000.csv"),
            "GMC": MethodSource("results/a6000_final_baselines/coalesced", "coalesced_runtime_summary.csv"),
            "Shared": MethodSource("results/a6000_final_baselines/shared", "shared_runtime_summary.csv"),
            "Warp": MethodSource("results/warp_tiled_ampere/a6000_final_w16_b128", "benchmark_summary.csv"),
        },
    ),
    GPUConfig(
        key="v100",
        device_name="Tesla V100-SXM3-32GB",
        architecture="Volta",
        compute_capability="7.0",
        warp_w=16,
        warp_bwdk_block=256,
        summary_path="results/v100_final_summary/v100_aggregate_summary.csv",
        sources={
            "PyTorch": MethodSource("results/v100_final_baselines/pytorch", "pytorch_runtime_summary_v100.csv"),
            "Naive": MethodSource("results/v100_final_baselines/naive", "naive_runtime_summary_v100.csv"),
            "GMC": MethodSource("results/v100_final_baselines/coalesced", "coalesced_runtime_summary.csv"),
            "Shared": MethodSource("results/v100_final_baselines/shared", "shared_runtime_summary.csv"),
            "Warp": MethodSource("results/warp_tiled_volta/v100_final_w16_b256", "benchmark_summary.csv"),
        },
    ),
]


SUMMARY_COLUMNS = [
    "method",
    "trials",
    "fwd_mean_ms",
    "fwd_between_trial_std_ms",
    "bwd_input_mean_ms",
    "bwd_input_between_trial_std_ms",
    "bwd_weight_mean_ms",
    "bwd_weight_between_trial_std_ms",
    "total_mean_ms",
    "total_between_trial_std_ms",
    "fwd_speedup_vs_pytorch",
    "bwd_input_speedup_vs_pytorch",
    "bwd_weight_speedup_vs_pytorch",
    "total_speedup_vs_pytorch",
    "total_reduction_vs_pytorch_pct",
]


def _trial_number(path: Path) -> int:
    match = re.fullmatch(r"trial(\d+)", path.parent.name)
    if match is None:
        raise ValueError(f"Cannot infer trial number from {path}")
    return int(match.group(1))


def build_trial_means(root: Path, config: GPUConfig) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for method in METHOD_ORDER:
        source = config.sources[method]
        paths = sorted((root / source.root).glob(f"trial*/{source.filename}"))
        if len(paths) != 3:
            raise RuntimeError(
                f"{config.key}/{method}: expected 3 trial summaries, found {len(paths)}"
            )
        for path in paths:
            frame = read_csv(path, ["phase", "mean_ms"])
            canonical_phase = frame["phase"].map(PHASE_ALIASES)
            if canonical_phase.isna().any():
                unknown = frame.loc[canonical_phase.isna(), "phase"].tolist()
                raise ValueError(f"Unknown phases in {path}: {unknown}")
            values = frame.assign(canonical_phase=canonical_phase).set_index(
                "canonical_phase"
            )["mean_ms"]
            if set(values.index) != set(PHASE_ORDER):
                raise ValueError(f"{path} does not contain exactly {PHASE_ORDER}")
            row: dict[str, float | int | str] = {
                "method": method,
                "trial": _trial_number(path),
            }
            for phase in PHASE_ORDER:
                row[f"{phase}_ms"] = float(values.loc[phase])
            row["total_ms"] = sum(float(row[f"{phase}_ms"]) for phase in PHASE_ORDER)
            rows.append(row)

    result = pd.DataFrame(rows)
    result["method"] = pd.Categorical(result["method"], METHOD_ORDER, ordered=True)
    return result.sort_values(["method", "trial"]).reset_index(drop=True).assign(
        method=lambda frame: frame["method"].astype("object")
    )


def build_gpu_summary(trials: pd.DataFrame) -> pd.DataFrame:
    pytorch = trials[trials["method"] == "PyTorch"]
    if len(pytorch) != 3:
        raise RuntimeError("PyTorch reference must contain exactly three trials")
    references = {
        phase: float(pytorch[f"{phase}_ms"].mean())
        for phase in (*PHASE_ORDER, "total")
    }

    rows: list[dict[str, float | int | str]] = []
    for method in METHOD_ORDER:
        group = trials[trials["method"] == method]
        if len(group) != 3:
            raise RuntimeError(f"{method}: expected three trials")
        row: dict[str, float | int | str] = {
            "method": method,
            "trials": int(group["trial"].nunique()),
        }
        for phase in (*PHASE_ORDER, "total"):
            values = group[f"{phase}_ms"].to_numpy(dtype=float)
            mean = float(values.mean())
            row[f"{phase}_mean_ms"] = mean
            row[f"{phase}_between_trial_std_ms"] = sample_std(values)
            row[f"{phase}_speedup_vs_pytorch"] = references[phase] / mean
        row["total_reduction_vs_pytorch_pct"] = (
            1.0 - float(row["total_mean_ms"]) / references["total"]
        ) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def build_torch_compile(root: Path, a6000: pd.DataFrame) -> dict[Path, pd.DataFrame]:
    compile_root = root / "results/a6000_final_baselines/torch_compile"
    rows: list[dict[str, float | int]] = []
    phase_map = {
        "forward": "fwd_mean_ms",
        "backward_input": "dx_mean_ms",
        "backward_kernel": "dk_mean_ms",
    }
    pass_pattern = re.compile(
        r"^(forward|backward_input|backward_kernel): .*\bPASS\s*$", re.MULTILINE
    )
    for trial in (1, 2, 3):
        trial_root = compile_root / f"trial{trial}"
        summary = read_csv(
            trial_root / "torch_compile_runtime_summary_a6000.csv",
            ["phase", "mean_ms"],
        )
        values = dict(zip(summary["phase"], summary["mean_ms"], strict=True))
        if set(values) != set(phase_map):
            raise ValueError(f"Unexpected torch.compile phases in trial {trial}: {values}")
        log_path = trial_root / "benchmark.log"
        passes = pass_pattern.findall(log_path.read_text(encoding="utf-8"))
        if sorted(passes) != sorted(phase_map):
            raise AssertionError(f"torch.compile correctness evidence failed in {log_path}")
        row: dict[str, float | int] = {
            "trial": trial,
            "correctness_passes": len(passes),
        }
        for phase, column in phase_map.items():
            row[column] = float(values[phase])
        row["total_mean_ms"] = sum(float(row[column]) for column in phase_map.values())
        rows.append(row)

    per_trial = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [
            {
                "trials": len(per_trial),
                "correctness_passes": int(per_trial["correctness_passes"].sum()),
                "fwd_mean_ms": float(per_trial["fwd_mean_ms"].mean()),
                "fwd_between_trial_std_ms": sample_std(per_trial["fwd_mean_ms"]),
                "dx_mean_ms": float(per_trial["dx_mean_ms"].mean()),
                "dx_between_trial_std_ms": sample_std(per_trial["dx_mean_ms"]),
                "dk_mean_ms": float(per_trial["dk_mean_ms"].mean()),
                "dk_between_trial_std_ms": sample_std(per_trial["dk_mean_ms"]),
                "total_mean_ms": float(per_trial["total_mean_ms"].mean()),
                "total_between_trial_std_ms": sample_std(per_trial["total_mean_ms"]),
            }
        ]
    )

    indexed = a6000.set_index("method")
    pytorch = indexed.loc["PyTorch"]
    warp = indexed.loc["Warp"]
    compiled = aggregate.iloc[0]
    compile_total = float(compiled["total_mean_ms"])
    comparison_rows = []
    values = [
        (
            "PyTorch",
            float(pytorch["fwd_mean_ms"]),
            float(pytorch["bwd_input_mean_ms"]),
            float(pytorch["bwd_weight_mean_ms"]),
            float(pytorch["total_mean_ms"]),
            float(pytorch["total_between_trial_std_ms"]),
        ),
        (
            "torch.compile",
            float(compiled["fwd_mean_ms"]),
            float(compiled["dx_mean_ms"]),
            float(compiled["dk_mean_ms"]),
            compile_total,
            float(compiled["total_between_trial_std_ms"]),
        ),
        (
            "Warp",
            float(warp["fwd_mean_ms"]),
            float(warp["bwd_input_mean_ms"]),
            float(warp["bwd_weight_mean_ms"]),
            float(warp["total_mean_ms"]),
            float(warp["total_between_trial_std_ms"]),
        ),
    ]
    for method, fwd, dx, dk, total, total_std in values:
        comparison_rows.append(
            {
                "method": method,
                "fwd_mean_ms": fwd,
                "bwd_input_mean_ms": dx,
                "bwd_weight_mean_ms": dk,
                "total_mean_ms": total,
                "total_between_trial_std_ms": total_std,
                "fwd_speedup_vs_pytorch": float(pytorch["fwd_mean_ms"]) / fwd,
                "bwd_input_speedup_vs_pytorch": float(pytorch["bwd_input_mean_ms"]) / dx,
                "bwd_weight_speedup_vs_pytorch": float(pytorch["bwd_weight_mean_ms"]) / dk,
                "total_speedup_vs_pytorch": float(pytorch["total_mean_ms"]) / total,
                "total_reduction_vs_pytorch_pct": (
                    1.0 - total / float(pytorch["total_mean_ms"])
                )
                * 100.0,
                "total_speedup_vs_torch_compile": compile_total / total,
            }
        )

    return {
        Path("results/a6000_final_baselines/torch_compile/final_per_trial.csv"): per_trial,
        Path("results/a6000_final_baselines/torch_compile/final_aggregate.csv"): aggregate,
        Path("results/a6000_final_summary/a6000_torch_compile_comparison.csv"): pd.DataFrame(
            comparison_rows
        ),
    }


def build_outputs(root: Path) -> dict[Path, pd.DataFrame]:
    outputs: dict[Path, pd.DataFrame] = {}
    summaries: dict[str, pd.DataFrame] = {}
    for config in GPU_CONFIGS:
        trials = build_trial_means(root, config)
        summary = build_gpu_summary(trials)
        summaries[config.key] = summary
        outputs[Path(config.summary_path)] = summary
        outputs[
            Path(f"results/{config.key}_final_summary/{config.key}_trial_means.csv")
        ] = trials

    all_rows: list[dict[str, float | int | str]] = []
    paper_rows: list[dict[str, float | int | str]] = []
    for config in GPU_CONFIGS:
        summary = summaries[config.key]
        for row in summary.itertuples(index=False):
            all_rows.append(
                {
                    "gpu": config.device_name,
                    "architecture": config.architecture,
                    "compute_capability": config.compute_capability,
                    **row._asdict(),
                    "warp_w": config.warp_w,
                    "warp_bwdk_block": config.warp_bwdk_block,
                }
            )
        indexed = summary.set_index("method")
        pytorch = indexed.loc["PyTorch"]
        warp = indexed.loc["Warp"]
        paper_rows.append(
            {
                "gpu": config.device_name,
                "architecture": config.architecture,
                "compute_capability": config.compute_capability,
                "pytorch_total_ms": float(pytorch["total_mean_ms"]),
                "warp_total_ms": float(warp["total_mean_ms"]),
                "warp_total_between_trial_std_ms": float(
                    warp["total_between_trial_std_ms"]
                ),
                "warp_speedup_vs_pytorch": float(warp["total_speedup_vs_pytorch"]),
                "warp_reduction_vs_pytorch_pct": float(
                    warp["total_reduction_vs_pytorch_pct"]
                ),
                "warp_w": config.warp_w,
                "warp_bwdk_block": config.warp_bwdk_block,
            }
        )

    cross_columns = [
        "gpu",
        "architecture",
        "compute_capability",
        *SUMMARY_COLUMNS[:-1],
        "warp_w",
        "warp_bwdk_block",
        "total_reduction_vs_pytorch_pct",
    ]
    outputs[
        Path("results/cross_gpu_final_summary/cross_gpu_3gpu_all_metrics.csv")
    ] = pd.DataFrame(all_rows)[cross_columns]
    outputs[
        Path("results/cross_gpu_final_summary/cross_gpu_3gpu_paper_table.csv")
    ] = pd.DataFrame(paper_rows)
    outputs.update(build_torch_compile(root, summaries["a6000"]))
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
    print(f"Main/cross-GPU aggregation: PASS ({len(outputs)} files)")


if __name__ == "__main__":
    main()
