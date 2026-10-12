"""Correctness, timing, and aggregation for the FP32 Triton baseline.

Compilation and Triton autotuning are completed before CUDA-event timing.
Existing PyTorch/cuDNN and PathTile measurements are not repeated by this
script.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import triton

from kernels.triton_depthwise import (
    backward_input,
    backward_weight,
    backward_weight_workspace_shape,
    forward,
    selected_configs,
)


DEFAULT_B = 16384
DEFAULT_H = 128
DEFAULT_L = 48
DEFAULT_K = 48
DEFAULT_WARMUP = 50
DEFAULT_RUNS = 100
DEFAULT_SEED = 42
PHASES = ("fwd", "bwd_input", "bwd_weight")


def configure_strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def pytorch_depthwise_same(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    H, K = map(int, weight.shape)
    left = K // 2
    right = K - 1 - left
    return F.conv1d(
        F.pad(x, (left, right)),
        weight.unsqueeze(1),
        bias=None,
        stride=1,
        padding=0,
        groups=H,
    )


def max_abs_diff(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max().item())


def normalized_max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(expected.abs().max().item()), 1e-12)
    return max_abs_diff(actual, expected) / denominator


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual - expected).float())
    denominator = torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
    return float((numerator / denominator).item())


def device_slug() -> str:
    name = torch.cuda.get_device_name().lower()
    if "a100" in name:
        return "a100"
    if "a6000" in name:
        return "a6000"
    cleaned = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    return cleaned or "unknown_gpu"


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty measurement")
    n = int(values.size)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    margin = 1.96 * sem
    return {
        "runs": n,
        "mean_ms": mean,
        "median_ms": float(np.median(values)),
        "std_ms": std,
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
        "min_ms": float(np.min(values)),
        "p05_ms": float(np.quantile(values, 0.05)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def make_inputs(
    B: int,
    H: int,
    L: int,
    K: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(B, H, L, device="cuda", dtype=torch.float32)
    weight = torch.randn(H, K, device="cuda", dtype=torch.float32)
    grad_y = torch.randn_like(x)
    return x, weight, grad_y


def validate_case(
    B: int,
    H: int,
    L: int,
    K: int,
    seed: int,
) -> pd.DataFrame:
    x, weight, grad_y = make_inputs(B, H, L, K, seed)

    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    y_ref = pytorch_depthwise_same(x_ref, weight_ref)
    gx_ref, gk_ref = torch.autograd.grad(
        y_ref,
        (x_ref, weight_ref),
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )

    y = forward(x, weight)
    gx = backward_input(grad_y, weight)
    gk, workspace = backward_weight(grad_y, x, K)
    torch.cuda.synchronize()

    comparisons = (
        ("fwd", y, y_ref, 2e-3, 2e-3),
        ("bwd_input", gx, gx_ref, 2e-3, 2e-3),
        ("bwd_weight", gk, gk_ref, 2e-2, 2e-3),
    )

    rows: list[dict[str, object]] = []
    for phase, actual, expected, atol, rtol in comparisons:
        row = {
            "B": B,
            "H": H,
            "L": L,
            "K": K,
            "phase": phase,
            "max_abs": max_abs_diff(actual, expected),
            "normalized_max_abs": normalized_max_abs(actual, expected),
            "relative_l2": relative_l2(actual, expected),
            "atol": atol,
            "rtol": rtol,
            "allclose": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        }
        rows.append(row)

    del (
        x,
        weight,
        grad_y,
        x_ref,
        weight_ref,
        y_ref,
        gx_ref,
        gk_ref,
        y,
        gx,
        gk,
        workspace,
    )
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def measure_calls(
    phase: str,
    call: Callable[[], object],
    runs: int,
) -> list[dict[str, object]]:
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    for _ in range(runs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        event_pairs.append((start, end))

    torch.cuda.synchronize()
    return [
        {
            "phase": phase,
            "run": run_id,
            "runtime_ms": float(start.elapsed_time(end)),
        }
        for run_id, (start, end) in enumerate(event_pairs, start=1)
    ]


def benchmark_main(
    B: int,
    H: int,
    L: int,
    K: int,
    warmup: int,
    runs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    x, weight, grad_y = make_inputs(B, H, L, K, seed)
    y = torch.empty_like(x)
    gx = torch.empty_like(x)
    gk = torch.empty_like(weight)
    workspace = torch.empty(
        backward_weight_workspace_shape(B, H, L, K),
        device="cuda",
        dtype=torch.float32,
    )

    print("Compiling and autotuning outside measured events...")
    forward(x, weight, y)
    backward_input(grad_y, weight, gx)
    backward_weight(grad_y, x, K, gk, workspace)
    torch.cuda.synchronize()

    print(f"Warm-up: {warmup} iterations")
    for _ in range(warmup):
        forward(x, weight, y)
        backward_input(grad_y, weight, gx)
        backward_weight(grad_y, x, K, gk, workspace)
    torch.cuda.synchronize()

    print(f"Measured runs per phase: {runs}")
    rows: list[dict[str, object]] = []
    rows.extend(measure_calls("fwd", lambda: forward(x, weight, y), runs))
    rows.extend(
        measure_calls(
            "bwd_input",
            lambda: backward_input(grad_y, weight, gx),
            runs,
        )
    )
    rows.extend(
        measure_calls(
            "bwd_weight",
            lambda: backward_weight(grad_y, x, K, gk, workspace),
            runs,
        )
    )

    raw = pd.DataFrame(rows)
    raw.insert(0, "technique", "triton_autotuned")
    raw.insert(1, "gpu", torch.cuda.get_device_name())
    raw.insert(2, "B", B)
    raw.insert(3, "H", H)
    raw.insert(4, "L", L)
    raw.insert(5, "K", K)

    summary_rows: list[dict[str, object]] = []
    for phase in PHASES:
        values = raw.loc[raw["phase"] == phase, "runtime_ms"].to_numpy(float)
        summary_rows.append(
            {
                "technique": "triton_autotuned",
                "gpu": torch.cuda.get_device_name(),
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": phase,
                **stats(values),
            }
        )
    summary = pd.DataFrame(summary_rows)

    config_metadata: dict[str, object] = {
        "selected_configs": selected_configs(),
        "workspace_shape": list(workspace.shape),
        "workspace_bytes": int(workspace.numel() * workspace.element_size()),
    }

    del x, weight, grad_y, y, gx, gk, workspace
    torch.cuda.empty_cache()
    return raw, summary, config_metadata


def environment_metadata(args: argparse.Namespace) -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "technique": "triton_autotuned",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": triton.__version__,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": f"{props.major}.{props.minor}",
        "global_memory_bytes": int(props.total_memory),
        "multiprocessor_count": int(props.multi_processor_count),
        "dtype": "torch.float32",
        "B": args.B,
        "H": args.H,
        "L": args.L,
        "K": args.K,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "seed": args.seed,
        "compile_and_autotune_timed": False,
        "timing": "CUDA events on the current PyTorch stream",
        "semantics": "explicit same padding; left=K//2; output length=L",
    }


def run_smoke(args: argparse.Namespace) -> None:
    slug = device_slug()
    output_dir = args.output_root / slug / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    correctness = validate_case(B=8, H=8, L=16, K=8, seed=args.seed)
    correctness.to_csv(output_dir / "correctness.csv", index=False)
    print(correctness.to_string(index=False))
    if not bool(correctness["allclose"].all()):
        raise AssertionError("Triton smoke correctness failed")
    print(f"Triton smoke: PASS on {torch.cuda.get_device_name()}")
    print(f"Saved: {output_dir / 'correctness.csv'}")


def run_trial(args: argparse.Namespace) -> None:
    slug = device_slug()
    output_dir = args.output_root / slug / f"trial{args.trial_id}"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty trial directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_B = min(args.B, args.validation_batch)
    print(
        "Correctness against PyTorch/cuDNN: "
        f"B={validation_B}, H={args.H}, L={args.L}, K={args.K}"
    )
    correctness = validate_case(
        validation_B,
        args.H,
        args.L,
        args.K,
        args.seed,
    )
    correctness.to_csv(output_dir / "correctness.csv", index=False)
    print(correctness.to_string(index=False))
    if not bool(correctness["allclose"].all()):
        raise AssertionError("Triton correctness failed; benchmark was not run")

    raw, summary, extra_metadata = benchmark_main(
        args.B,
        args.H,
        args.L,
        args.K,
        args.warmup,
        args.runs,
        args.seed,
    )
    raw.to_csv(output_dir / "raw.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)

    metadata = environment_metadata(args)
    metadata.update(extra_metadata)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    total_ms = float(summary["mean_ms"].sum())
    print("\nTriton FP32 main-workload summary")
    print(summary[["phase", "mean_ms", "std_ms"]].to_string(index=False))
    print(f"Sum of phase means: {total_ms:.6f} ms")
    print(f"Saved trial directory: {output_dir}")


def aggregate_trials(args: argparse.Namespace) -> None:
    if args.gpu_slug is None:
        raise ValueError("--gpu-slug is required in aggregate mode")
    root = args.output_root / args.gpu_slug
    trial_rows: list[dict[str, object]] = []

    for trial_id in range(1, args.expected_trials + 1):
        path = root / f"trial{trial_id}" / "summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        by_phase = frame.set_index("phase")["mean_ms"]
        missing = [phase for phase in PHASES if phase not in by_phase.index]
        if missing:
            raise ValueError(f"{path} is missing phases: {missing}")
        trial_rows.append(
            {
                "trial": trial_id,
                "fwd_mean_ms": float(by_phase["fwd"]),
                "bwd_input_mean_ms": float(by_phase["bwd_input"]),
                "bwd_weight_mean_ms": float(by_phase["bwd_weight"]),
                "total_mean_ms": float(sum(by_phase[phase] for phase in PHASES)),
            }
        )

    per_trial = pd.DataFrame(trial_rows)
    per_trial.to_csv(root / "final_per_trial.csv", index=False)

    aggregate: dict[str, object] = {
        "gpu_slug": args.gpu_slug,
        "method": "Triton",
        "trials": len(per_trial),
    }
    for column in (
        "fwd_mean_ms",
        "bwd_input_mean_ms",
        "bwd_weight_mean_ms",
        "total_mean_ms",
    ):
        stem = column.removesuffix("_mean_ms")
        aggregate[f"{stem}_mean_ms"] = float(per_trial[column].mean())
        aggregate[f"{stem}_between_trial_std_ms"] = float(
            per_trial[column].std(ddof=1)
        )

    aggregate_frame = pd.DataFrame([aggregate])
    aggregate_frame.to_csv(root / "final_aggregate.csv", index=False)
    print("Triton aggregation: PASS")
    print(per_trial.to_string(index=False))
    print("\n" + aggregate_frame.to_string(index=False))
    print(f"Saved: {root / 'final_per_trial.csv'}")
    print(f"Saved: {root / 'final_aggregate.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run", "aggregate"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/external_baselines/triton"),
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--gpu-slug", choices=("a100", "a6000"))
    parser.add_argument("--B", type=int, default=DEFAULT_B)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--validation-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in ("B", "H", "L", "K", "warmup", "runs", "validation_batch"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.trial_id <= 0 or args.expected_trials <= 0:
        parser.error("trial counts must be positive")
    return args


def main() -> None:
    args = parse_args()
    configure_strict_fp32()

    if args.mode == "aggregate":
        aggregate_trials(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(
            "Triton 3.6 officially requires NVIDIA compute capability 8.0+; "
            f"active GPU is sm_{major}{minor}"
        )

    print("Triton:", triton.__version__)
    print("GPU:", torch.cuda.get_device_name())
    print("Compute capability:", f"{major}.{minor}")
    if args.mode == "smoke":
        run_smoke(args)
    else:
        run_trial(args)


if __name__ == "__main__":
    main()
