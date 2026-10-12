#!/usr/bin/env python3
"""Controlled PyTorch/cuDNN versus Warp CUDA shape sweep on Ampere GPUs.

This benchmark extends the paper workload without changing the measurement
contract used by the final RTX A6000 and A100 experiments:

* strict FP32 (TF32 disabled),
* explicit asymmetric SAME padding for the PyTorch/cuDNN reference,
* forward, input-gradient, and weight-gradient timed independently,
* graph construction excluded from the two backward CUDA-event intervals,
* custom weight-gradient zero-fill excluded from its kernel interval,
* 50 warm-up iterations and 100 measured iterations by default,
* one process invocation per independent trial.

Run this file from the repository root as::

    python -m benchmark.benchmark_shape_sweep_ampere --mode run ...

The final Warp implementation remains specialized for L <= 48 and K <= 48.
The presets therefore test generalization only inside that declared domain.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import socket
import zlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from kernels.warp_tiled_ampere import (
    DEFAULT_BWDK_BLOCK,
    DEFAULT_BWDK_BTILE,
    DEFAULT_WARPS_PER_BLOCK,
    KERNEL_EVENTS,
    AmpereWarpRuntime,
    KernelEventCollector,
    configure_torch,
    depthwise_conv_ampere,
    install_runtime,
    max_abs_diff,
    normalized_max_abs,
)


PYTORCH_METHOD = "PyTorch-cuDNN"
WARP_METHOD = "Warp"
PHASES = ("fwd", "bwd_input", "bwd_weight")


@dataclass(frozen=True)
class ShapeCase:
    config_id: str
    B: int
    H: int
    L: int
    K: int
    sweep_group: str

    def validate(self) -> None:
        if min(self.B, self.H, self.L, self.K) <= 0:
            raise ValueError(f"All dimensions must be positive: {self}")
        if self.L > 48 or self.K > 48:
            raise ValueError(
                f"{self.config_id}: Warp paper kernel requires L<=48 and K<=48"
            )


FULL_CASES = [
    ShapeCase("main_b16384_h128_l48_k48", 16_384, 128, 48, 48, "main"),
    ShapeCase("batch_b2048_h128_l48_k48", 2_048, 128, 48, 48, "batch"),
    ShapeCase("batch_b8192_h128_l48_k48", 8_192, 128, 48, 48, "batch"),
    ShapeCase("channel_b8192_h64_l48_k48", 8_192, 64, 48, 48, "channel"),
    ShapeCase("channel_b8192_h256_l48_k48", 8_192, 256, 48, 48, "channel"),
    ShapeCase("sequence_b8192_h128_l16_k16", 8_192, 128, 16, 16, "sequence"),
    ShapeCase("sequence_b8192_h128_l24_k24", 8_192, 128, 24, 24, "sequence"),
    ShapeCase("sequence_b8192_h128_l32_k32", 8_192, 128, 32, 32, "sequence"),
    ShapeCase("kernel_b8192_h128_l48_k8", 8_192, 128, 48, 8, "kernel"),
    ShapeCase("kernel_b8192_h128_l48_k24", 8_192, 128, 48, 24, "kernel"),
]

PATHTILE_TRAIN_CASES = [
    *FULL_CASES,
    ShapeCase(
        "mix_b4096_h64_l32_k16",
        4_096, 64, 32, 16, "mixed",
    ),
    ShapeCase(
        "mix_b4096_h256_l48_k24",
        4_096, 256, 48, 24, "mixed",
    ),
    ShapeCase(
        "mix_b2048_h64_l24_k8",
        2_048, 64, 24, 8, "mixed",
    ),
    ShapeCase(
        "mix_b2048_h256_l32_k32",
        2_048, 256, 32, 32, "mixed",
    ),
    ShapeCase(
        "mix_b8192_h64_l32_k24",
        8_192, 64, 32, 24, "mixed",
    ),
    ShapeCase(
        "mix_b8192_h256_l24_k16",
        8_192, 256, 24, 16, "mixed",
    ),
    ShapeCase(
        "mix_b16384_h64_l48_k8",
        16_384, 64, 48, 8, "mixed",
    ),
    ShapeCase(
        "mix_b16384_h256_l16_k16",
        16_384, 256, 16, 16, "mixed",
    ),
    ShapeCase(
        "mix_b16384_h128_l32_k24",
        16_384, 128, 32, 24, "mixed",
    ),
    ShapeCase(
        "mix_b4096_h128_l48_k32",
        4_096, 128, 48, 32, "mixed",
    ),
]


PATHTILE_HOLDOUT_CASES = [
    ShapeCase(
        "holdout_b4096_h96_l40_k20",
        4_096, 96, 40, 20, "holdout",
    ),
    ShapeCase(
        "holdout_b12288_h192_l40_k32",
        12_288, 192, 40, 32, "holdout",
    ),
    ShapeCase(
        "holdout_b12288_h96_l20_k12",
        12_288, 96, 20, 12, "holdout",
    ),
    ShapeCase(
        "holdout_b4096_h192_l28_k24",
        4_096, 192, 28, 24, "holdout",
    ),
    ShapeCase(
        "holdout_b12288_h64_l40_k40",
        12_288, 64, 40, 40, "holdout",
    ),
    ShapeCase(
        "holdout_b4096_h256_l20_k16",
        4_096, 256, 20, 16, "holdout",
    ),
    ShapeCase(
        "holdout_b2048_h192_l40_k32",
        2_048, 192, 40, 32, "holdout",
    ),
    ShapeCase(
        "holdout_b12288_h128_l28_k20",
        12_288, 128, 28, 20, "holdout",
    ),
    ShapeCase(
        "holdout_b4096_h64_l40_k12",
        4_096, 64, 40, 12, "holdout",
    ),
    ShapeCase(
        "holdout_b12288_h256_l32_k24",
        12_288, 256, 32, 24, "holdout",
    ),
]

CROSS_ARCH_CASE_IDS = {
    "main_b16384_h128_l48_k48",
    "batch_b8192_h128_l48_k48",
    "channel_b8192_h64_l48_k48",
    "channel_b8192_h256_l48_k48",
    "sequence_b8192_h128_l16_k16",
    "sequence_b8192_h128_l32_k32",
    "kernel_b8192_h128_l48_k8",
    "kernel_b8192_h128_l48_k24",
}


SMOKE_CASES = [
    ShapeCase("smoke_b64_h8_l16_k8", 64, 8, 16, 8, "smoke"),
    ShapeCase("smoke_b128_h16_l48_k48", 128, 16, 48, 48, "smoke"),
]


def cases_for_preset(preset: str) -> list[ShapeCase]:
    if preset == "full":
        cases = list(FULL_CASES)
    elif preset == "cross-arch":
        cases = [
            case
            for case in FULL_CASES
            if case.config_id in CROSS_ARCH_CASE_IDS
        ]
    elif preset == "pathtile-train":
        cases = list(PATHTILE_TRAIN_CASES)
    elif preset == "pathtile-holdout":
        cases = list(PATHTILE_HOLDOUT_CASES)
    elif preset == "smoke":
        cases = list(SMOKE_CASES)
    else:
        raise ValueError(f"Unknown preset: {preset}")

    for case in cases:
        case.validate()
    return cases

def pytorch_depthwise_same(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Exact PyTorch baseline used in the final paper runs.

    x has shape [B,H,L], k has shape [H,K].  For even K, the explicit
    asymmetric padding returns exactly L outputs and matches the custom CUDA
    indexing convention.
    """

    K = int(k.shape[1])
    left = K // 2
    right = K - 1 - left
    x_pad = F.pad(x, (left, right))
    return F.conv1d(
        x_pad,
        k.unsqueeze(1),
        bias=None,
        stride=1,
        padding=0,
        groups=int(x.shape[1]),
    )


def _stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty measurement group")

    mean = float(np.mean(values))
    median = float(np.median(values))
    if n > 1:
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)
        margin = 1.96 * sem
    else:
        std = 0.0
        sem = 0.0
        margin = 0.0

    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "sem_ms": sem,
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
        "min_ms": float(np.min(values)),
        "p05_ms": float(np.quantile(values, 0.05)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def _case_columns(case: ShapeCase) -> dict[str, Any]:
    return asdict(case)


def _collector_context(
    case: ShapeCase,
    method: str,
    trial_id: int,
    device_name: str,
) -> dict[str, Any]:
    return {
        **_case_columns(case),
        "method": method,
        "trial_id": trial_id,
        "device_name": device_name,
    }


def _warmup_pytorch(
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    warmup: int,
) -> None:
    x_dx = x.detach().requires_grad_(True)
    k_dx = k.detach()
    x_dw = x.detach()
    k_dw = k.detach().requires_grad_(True)

    for _ in range(warmup):
        _ = pytorch_depthwise_same(x, k)

        y_dx = pytorch_depthwise_same(x_dx, k_dx)
        _ = torch.autograd.grad(
            y_dx,
            x_dx,
            grad_outputs=gy,
            retain_graph=False,
            create_graph=False,
        )[0]

        y_dw = pytorch_depthwise_same(x_dw, k_dw)
        _ = torch.autograd.grad(
            y_dw,
            k_dw,
            grad_outputs=gy,
            retain_graph=False,
            create_graph=False,
        )[0]

    torch.cuda.synchronize()


def _warmup_warp(
    runtime: AmpereWarpRuntime,
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    warmup: int,
) -> None:
    y = torch.empty_like(x)
    gx = torch.empty_like(x)
    gk = torch.zeros_like(k)

    KERNEL_EVENTS.enabled = False
    KERNEL_EVENTS.clear()
    for _ in range(warmup):
        runtime.launch_fwd(x, k, y)
        runtime.launch_bwd_input(gy, k, gx)
        gk.zero_()
        runtime.launch_bwd_weight(gy, x, gk, int(k.shape[1]))
    torch.cuda.synchronize()
    del y, gx, gk


def _measure_pytorch(
    case: ShapeCase,
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    runs: int,
    trial_id: int,
    device_name: str,
) -> pd.DataFrame:
    collector = KernelEventCollector()
    collector.enabled = True
    common = _collector_context(case, PYTORCH_METHOD, trial_id, device_name)
    holders: dict[str, torch.Tensor] = {}

    for requested_run in range(1, runs + 1):
        collector.set_context(**common, run_id_requested=requested_run)

        def launch_forward() -> None:
            holders["fwd"] = pytorch_depthwise_same(x, k)

        collector.launch("fwd", launch_forward)

    x_dx = x.detach().requires_grad_(True)
    k_dx = k.detach()
    for requested_run in range(1, runs + 1):
        y_dx = pytorch_depthwise_same(x_dx, k_dx)
        collector.set_context(**common, run_id_requested=requested_run)

        def launch_bwd_input(y: torch.Tensor = y_dx) -> None:
            holders["bwd_input"] = torch.autograd.grad(
                y,
                x_dx,
                grad_outputs=gy,
                retain_graph=False,
                create_graph=False,
            )[0]

        collector.launch("bwd_input", launch_bwd_input)

    x_dw = x.detach()
    k_dw = k.detach().requires_grad_(True)
    for requested_run in range(1, runs + 1):
        y_dw = pytorch_depthwise_same(x_dw, k_dw)
        collector.set_context(**common, run_id_requested=requested_run)

        def launch_bwd_weight(y: torch.Tensor = y_dw) -> None:
            holders["bwd_weight"] = torch.autograd.grad(
                y,
                k_dw,
                grad_outputs=gy,
                retain_graph=False,
                create_graph=False,
            )[0]

        collector.launch("bwd_weight", launch_bwd_weight)

    raw = collector.resolve(synchronize=True)
    del holders, x_dx, k_dx, x_dw, k_dw
    return raw


def _measure_warp(
    runtime: AmpereWarpRuntime,
    case: ShapeCase,
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    runs: int,
    trial_id: int,
    device_name: str,
) -> pd.DataFrame:
    y = torch.empty_like(x)
    gx = torch.empty_like(x)
    gk = torch.zeros_like(k)

    collector = KernelEventCollector()
    collector.enabled = True
    common = _collector_context(case, WARP_METHOD, trial_id, device_name)

    for requested_run in range(1, runs + 1):
        collector.set_context(**common, run_id_requested=requested_run)
        runtime.launch_fwd(x, k, y, collector)

    for requested_run in range(1, runs + 1):
        collector.set_context(**common, run_id_requested=requested_run)
        runtime.launch_bwd_input(gy, k, gx, collector)

    for requested_run in range(1, runs + 1):
        gk.zero_()
        collector.set_context(**common, run_id_requested=requested_run)
        runtime.launch_bwd_weight(gy, x, gk, int(k.shape[1]), collector)

    raw = collector.resolve(synchronize=True)
    del y, gx, gk
    return raw


def validate_case_correctness(
    case: ShapeCase,
    seed: int,
) -> dict[str, Any]:
    """Validate the full shape in H/L/K and a capped batch of 128."""

    validation_B = min(case.B, 128)
    torch.manual_seed(seed)
    np.random.seed(seed)
    KERNEL_EVENTS.enabled = False
    KERNEL_EVENTS.clear()

    x = torch.randn(
        validation_B,
        case.H,
        case.L,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    k = torch.randn(
        case.H,
        case.K,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    gy = torch.randn_like(x)

    x_ref = x.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)

    y = depthwise_conv_ampere(x, k)
    y_ref = pytorch_depthwise_same(x_ref, k_ref)
    gx, gk = torch.autograd.grad(y, (x, k), gy)
    gx_ref, gk_ref = torch.autograd.grad(y_ref, (x_ref, k_ref), gy)
    torch.cuda.synchronize()

    row = {
        **_case_columns(case),
        "validation_B": validation_B,
        "fwd_max_abs": max_abs_diff(y, y_ref),
        "bwd_input_max_abs": max_abs_diff(gx, gx_ref),
        "bwd_weight_max_abs": max_abs_diff(gk, gk_ref),
        "fwd_normalized_max_abs": normalized_max_abs(y, y_ref),
        "bwd_input_normalized_max_abs": normalized_max_abs(gx, gx_ref),
        "bwd_weight_normalized_max_abs": normalized_max_abs(gk, gk_ref),
        "fwd_allclose": bool(torch.allclose(y, y_ref, atol=2e-3, rtol=2e-3)),
        "bwd_input_allclose": bool(
            torch.allclose(gx, gx_ref, atol=2e-3, rtol=2e-3)
        ),
        "bwd_weight_allclose": bool(
            torch.allclose(gk, gk_ref, atol=1e-2, rtol=5e-4)
        ),
    }
    row["all_correct"] = bool(
        row["fwd_allclose"]
        and row["bwd_input_allclose"]
        and row["bwd_weight_allclose"]
    )

    del x, k, gy, x_ref, k_ref, y, y_ref, gx, gk, gx_ref, gk_ref
    torch.cuda.empty_cache()
    return row


def summarize_trial(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "trial_id",
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
    for keys, group in raw.groupby(group_columns, sort=False):
        row = dict(zip(group_columns, keys, strict=True))
        row.update(_stats(group["runtime_ms"].to_numpy(dtype=float)))
        rows.append(row)

    total_group_columns = group_columns[:-1]
    total_raw = (
        raw.groupby(total_group_columns + ["run_id"], sort=False, as_index=False)[
            "runtime_ms"
        ]
        .sum()
        .rename(columns={"runtime_ms": "total_runtime_ms"})
    )
    for keys, group in total_raw.groupby(total_group_columns, sort=False):
        row = dict(zip(total_group_columns, keys, strict=True))
        row["phase"] = "total"
        row.update(_stats(group["total_runtime_ms"].to_numpy(dtype=float)))
        rows.append(row)

    result = pd.DataFrame(rows)
    phase_order = {phase: index for index, phase in enumerate((*PHASES, "total"))}
    result["_phase_order"] = result["phase"].map(phase_order)
    result = result.sort_values(
        ["config_id", "method", "_phase_order"], kind="stable"
    ).drop(columns="_phase_order")
    return result.reset_index(drop=True)


def comparison_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "config_id",
        "sweep_group",
        "B",
        "H",
        "L",
        "K",
        "phase",
    ]
    pivot = summary.pivot_table(
        index=index_columns,
        columns="method",
        values="mean_ms",
        aggfunc="first",
    ).reset_index()
    if PYTORCH_METHOD not in pivot.columns or WARP_METHOD not in pivot.columns:
        raise ValueError("Both PyTorch-cuDNN and Warp results are required")

    pivot = pivot.rename(
        columns={
            PYTORCH_METHOD: "pytorch_mean_ms",
            WARP_METHOD: "warp_mean_ms",
        }
    )
    pivot["warp_speedup_vs_pytorch"] = (
        pivot["pytorch_mean_ms"] / pivot["warp_mean_ms"]
    )
    pivot["warp_reduction_vs_pytorch_pct"] = (
        1.0 - pivot["warp_mean_ms"] / pivot["pytorch_mean_ms"]
    ) * 100.0
    return pivot.sort_values(index_columns, kind="stable").reset_index(drop=True)


def _save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _trial_output_dir(root: Path, trial_id: int) -> Path:
    return root / f"trial{trial_id}"


def run_trial(args: argparse.Namespace) -> None:
    if args.trial_id <= 0:
        raise ValueError("--trial-id must be positive")
    if args.warmup < 1 or args.runs < 2:
        raise ValueError("Use at least one warm-up and two measured runs")

    canonical_cases = cases_for_preset(args.preset)
    cases = (
        canonical_cases
        if args.trial_id % 2 == 1
        else list(reversed(canonical_cases))
    )
    output_root = Path(args.output_dir)
    trial_dir = _trial_output_dir(output_root, args.trial_id)
    known_outputs = [
        trial_dir / "raw.csv",
        trial_dir / "trial_summary.csv",
        trial_dir / "trial_comparison.csv",
    ]
    if not args.overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            f"Trial output already exists in {trial_dir}; use --overwrite intentionally"
        )

    torch.cuda.set_device(args.device)
    configure_torch()
    torch.backends.cudnn.deterministic = False

    runtime = AmpereWarpRuntime(
        warps_per_block=args.warps_per_block,
        bwdk_btile=DEFAULT_BWDK_BTILE,
        bwdk_block=args.bwdk_block,
        fast_math=False,
    )
    install_runtime(runtime)

    trial_dir.mkdir(parents=True, exist_ok=True)

    device_name = torch.cuda.get_device_name(args.device)
    props = torch.cuda.get_device_properties(args.device)
    resources = runtime.resource_report()
    resources.to_csv(trial_dir / "kernel_resources_occupancy.csv", index=False)
    pd.DataFrame([asdict(case) for case in canonical_cases]).to_csv(
        trial_dir / "config_manifest.csv", index=False
    )

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": device_name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "preset": args.preset,
        "trial_id": args.trial_id,
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "warps_per_block": runtime.warps_per_block,
        "bwdk_btile": runtime.bwdk_btile,
        "bwdk_block": runtime.bwdk_block,
        "fast_math": runtime.fast_math,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "measurement_contract": {
            "pytorch_forward_includes_explicit_pad": True,
            "backward_graph_construction_timed": False,
            "custom_bwd_weight_zero_fill_timed": False,
            "aggregate_total_definition": "sum of separately timed phase events",
            "config_order": "forward for odd trials; reverse for even trials",
        },
    }
    _save_json(trial_dir / "metadata.json", metadata)

    correctness_rows: list[dict[str, Any]] = []
    raw_frames: list[pd.DataFrame] = []

    print("=" * 96)
    print("AMPERE PYTORCH-vs-WARP SHAPE SWEEP")
    print("=" * 96)
    print(f"GPU:       {device_name}")
    print(f"Preset:    {args.preset}")
    print(f"Trial:     {args.trial_id}")
    print(f"Protocol:  warmup={args.warmup}, runs={args.runs}, strict FP32")
    print(f"Warp:      W={runtime.warps_per_block}, BWDK_BLOCK={runtime.bwdk_block}")
    print(f"Cases:     {len(cases)}")

    try:
        for case_index, case in enumerate(cases):
            case_seed = args.seed + zlib.crc32(case.config_id.encode("utf-8"))
            print("\n" + "-" * 96)
            print(
                f"[{case_index + 1}/{len(cases)}] {case.config_id}: "
                f"B={case.B}, H={case.H}, L={case.L}, K={case.K}"
            )

            if not args.skip_correctness:
                correctness = validate_case_correctness(case, case_seed)
                correctness["trial_id"] = args.trial_id
                correctness_rows.append(correctness)
                print(
                    "Correctness: "
                    f"fwd={correctness['fwd_max_abs']:.3e}, "
                    f"dx={correctness['bwd_input_max_abs']:.3e}, "
                    f"dk={correctness['bwd_weight_max_abs']:.3e}, "
                    f"ok={correctness['all_correct']}"
                )
                if not correctness["all_correct"]:
                    raise AssertionError(f"Correctness failed for {case.config_id}")

            torch.manual_seed(case_seed)
            np.random.seed(case_seed)
            x = torch.randn(
                case.B, case.H, case.L, device="cuda", dtype=torch.float32
            )
            k = torch.randn(case.H, case.K, device="cuda", dtype=torch.float32)
            gy = torch.randn_like(x)

            print("Warm-up PyTorch/cuDNN...")
            _warmup_pytorch(x, k, gy, args.warmup)
            print("Warm-up Warp...")
            _warmup_warp(runtime, x, k, gy, args.warmup)

            method_order = (
                (PYTORCH_METHOD, WARP_METHOD)
                if args.trial_id % 2 == 1
                else (WARP_METHOD, PYTORCH_METHOD)
            )
            print("Measured method order:", " -> ".join(method_order))

            for method in method_order:
                if method == PYTORCH_METHOD:
                    frame = _measure_pytorch(
                        case, x, k, gy, args.runs, args.trial_id, device_name
                    )
                else:
                    frame = _measure_warp(
                        runtime,
                        case,
                        x,
                        k,
                        gy,
                        args.runs,
                        args.trial_id,
                        device_name,
                    )
                raw_frames.append(frame)

            interim_raw = pd.concat(raw_frames, ignore_index=True)
            interim_raw.to_csv(trial_dir / "raw.csv", index=False)

            case_summary = summarize_trial(
                interim_raw[interim_raw["config_id"] == case.config_id]
            )
            case_total = comparison_from_summary(case_summary)
            case_total = case_total[case_total["phase"] == "total"]
            print("\nTotal comparison:")
            print(case_total.to_string(index=False))

            del x, k, gy, interim_raw, case_summary, case_total
            gc.collect()
            torch.cuda.empty_cache()

        raw = pd.concat(raw_frames, ignore_index=True)
        summary = summarize_trial(raw)
        comparison = comparison_from_summary(summary)

        raw.to_csv(trial_dir / "raw.csv", index=False)
        summary.to_csv(trial_dir / "trial_summary.csv", index=False)
        comparison.to_csv(trial_dir / "trial_comparison.csv", index=False)
        if correctness_rows:
            pd.DataFrame(correctness_rows).to_csv(
                trial_dir / "correctness.csv", index=False
            )

        print("\n" + "=" * 96)
        print("TRIAL TOTALS")
        print("=" * 96)
        print(
            comparison[comparison["phase"] == "total"].to_string(index=False)
        )
        print("\nSaved:")
        for name in [
            "raw.csv",
            "trial_summary.csv",
            "trial_comparison.csv",
            "correctness.csv",
            "config_manifest.csv",
            "kernel_resources_occupancy.csv",
            "metadata.json",
        ]:
            path = trial_dir / name
            if path.exists():
                print(path)
    finally:
        KERNEL_EVENTS.enabled = False
        runtime.close()


def aggregate_trials(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    paths = sorted(output_root.glob("trial*/trial_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No trial summaries found below {output_root}")

    frames = [pd.read_csv(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    trial_ids = sorted(int(value) for value in combined["trial_id"].unique())
    if len(trial_ids) != args.expected_trials:
        raise RuntimeError(
            f"Expected {args.expected_trials} trials, found {trial_ids}"
        )

    completeness_columns = ["config_id", "method", "phase"]
    completeness = combined.groupby(completeness_columns, sort=False)[
        "trial_id"
    ].nunique()
    incomplete = completeness[completeness != args.expected_trials]
    if not incomplete.empty:
        raise RuntimeError(
            "At least one configuration/method/phase is missing a trial:\n"
            f"{incomplete.to_string()}"
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
                "mean_ms": float(np.mean(trial_means)),
                "between_trial_std_ms": (
                    float(np.std(trial_means, ddof=1))
                    if trial_means.size > 1
                    else 0.0
                ),
                "min_trial_mean_ms": float(np.min(trial_means)),
                "max_trial_mean_ms": float(np.max(trial_means)),
                "mean_within_trial_std_ms": float(np.mean(group["std_ms"])),
            }
        )
        rows.append(row)

    aggregate = pd.DataFrame(rows)
    comparison = comparison_from_summary(aggregate)

    combined.to_csv(output_root / "all_trial_summaries.csv", index=False)
    aggregate.to_csv(output_root / "aggregate_summary.csv", index=False)
    comparison.to_csv(output_root / "aggregate_comparison.csv", index=False)

    print("=" * 96)
    print("AGGREGATED PYTORCH-vs-WARP SHAPE SWEEP")
    print("=" * 96)
    print(f"Trials: {trial_ids}")
    print(
        comparison[comparison["phase"] == "total"].to_string(index=False)
    )
    print("\nSaved:")
    print(output_root / "all_trial_summaries.csv")
    print(output_root / "aggregate_summary.csv")
    print(output_root / "aggregate_comparison.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument(
        "--preset",
        choices=[
            "smoke",
            "full",
            "cross-arch",
            "pathtile-train",
            "pathtile-holdout",
        ],
        default="smoke",
    )
    parser.add_argument(
        "--output-dir",
        default="results/shape_sweep_ampere",
        help="Root containing trial1, trial2, ... subdirectories.",
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--warps-per-block", type=int, default=DEFAULT_WARPS_PER_BLOCK
    )
    parser.add_argument("--bwdk-block", type=int, default=DEFAULT_BWDK_BLOCK)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_trial(args)
    else:
        aggregate_trials(args)


if __name__ == "__main__":
    main()
