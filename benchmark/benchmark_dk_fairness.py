#!/usr/bin/env python3
"""Supplementary dK timing-boundary experiment for PathTile.

Self-contained *benchmark harness*; uses the project's existing CUDA kernels,
does not modify primary results, and does not re-tune launch parameters.

Run on an allocated NVIDIA GPU, from the s4convd-paper repo root:
    python -m benchmark.benchmark_dk_fairness --trials 3 --warmup 50 --runs 100

Interpretation: the atomic-full result includes a GPU-side zero-fill followed
by one atomic dK kernel in a single CUDA-event interval. The atomic-kernel
result excludes zero-fill by design; it is only a kernel microbenchmark.
The noatomic timings include both kernels but not tensor allocation. All modes
reuse preallocated buffers. CUDA/PyCUDA launch-stream semantics are inherited
from the existing project runtimes; validate them if stream settings change.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

from kernels import warp_tiled_ampere
from kernels import warp_tiled_noatomic
from kernels import warp_tiled_noatomic_optimized


@dataclass
class Mode:
    name: str
    launch: Callable[[], None]
    prepare: Callable[[], None]
    output: torch.Tensor | None


def event_measure(mode: Mode, warmup: int, runs: int) -> list[float]:
    # The pre-event operation is deliberately excluded from kernel-only timings.
    for _ in range(warmup):
        mode.prepare()
        mode.launch()
    torch.cuda.synchronize()

    pairs = []
    for _ in range(runs):
        mode.prepare()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        mode.launch()
        end.record()
        pairs.append((start, end))
    torch.cuda.synchronize()
    return [float(start.elapsed_time(end)) for start, end in pairs]


def reference_dk(x: torch.Tensor, k: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    # Cross-correlation with the project's even-K asymmetric SAME padding.
    K = k.shape[1]
    left = K // 2
    right = K - 1 - left
    weight = k.detach().clone().requires_grad_(True)
    y = F.conv1d(F.pad(x, (left, right)), weight.unsqueeze(1), groups=k.shape[0])
    return torch.autograd.grad(y, weight, grad_outputs=gy)[0].detach()


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available; run inside an allocated GPU job")
    torch.cuda.set_device(args.device)
    gpu = torch.cuda.get_device_name(args.device)
    major, minor = torch.cuda.get_device_capability(args.device)
    if (major, minor) not in {(7, 0), (8, 0), (8, 6)}:
        raise RuntimeError(f"GPU compute capability {(major, minor)} not in primary study")
    dK_threads = 256 if (major, minor) == (7, 0) else 128
    torch.manual_seed(args.seed)
    B, H, L, K = args.B, args.H, args.L, args.K
    if L > 48 or K > 48:
        raise ValueError("Specialized kernels require L,K <= 48")
    x = torch.randn((B, H, L), device="cuda", dtype=torch.float32)
    k = torch.randn((H, K), device="cuda", dtype=torch.float32)
    gy = torch.randn_like(x)

    gpu_id = f"sm{major}{minor}"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out / f"{gpu_id}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "gpu": gpu,
        "compute_capability": f"{major}.{minor}",
        "torch_version": torch.__version__,
        "B": B, "H": H, "L": L, "K": K,
        "warps_per_block": 16,
        "dK_threads_per_block": dK_threads,
        "dK_batch_tile": 64,
        "trials": args.trials, "warmup": args.warmup, "runs": args.runs,
        "seed": args.seed,
        "timestamp_utc": stamp,
        "preallocated_outputs": True,
        "reference": "PyTorch autograd FP32 grouped Conv1D (asymmetric SAME)",
        "atol": args.atol, "rtol": args.rtol,
        "scope": {
            "atomic_kernel": "zero-fill excluded; one atomic dK kernel",
            "atomic_full": "zero-fill and one atomic dK kernel in single event interval",
            "atomic_zero_only": "zero-fill of dK output, separately",
            "noatomic": "both partial and final reduction kernels; no allocation",
            "noatomic_optimized": "both partial and final reduction kernels; no allocation",
        },
        "limitations": "GPU CUDA-event timings; includes neither allocation nor compilation. "
                       "Not a replacement for any existing primary paper aggregates.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    runtimes = []
    try:
        for label, module in (
            ("atomic", warp_tiled_ampere),
            ("noatomic", warp_tiled_noatomic),
            ("noatomic_optimized", warp_tiled_noatomic_optimized),
        ):
            rt = module.AmpereWarpRuntime(warps_per_block=16, bwdk_btile=64, bwdk_block=dK_threads)
            runtimes.append((label, rt))

        outputs = {name: torch.empty_like(k) for name, _ in runtimes}
        rt_atomic = runtimes[0][1]
        rt_na = runtimes[1][1]
        rt_opt = runtimes[2][1]
        g_atomic = outputs["atomic"]

        modes = [
            Mode("atomic_kernel", lambda: rt_atomic.launch_bwd_weight(gy, x, g_atomic, K),
                 g_atomic.zero_, g_atomic),
            Mode("atomic_full", lambda: (g_atomic.zero_(), rt_atomic.launch_bwd_weight(gy, x, g_atomic, K)),
                 lambda: None, g_atomic),
            Mode("atomic_zero_only", g_atomic.zero_, lambda: None, None),
            Mode("noatomic", lambda: rt_na.launch_bwd_weight(gy, x, outputs["noatomic"], K),
                 lambda: None, outputs["noatomic"]),
            Mode("noatomic_optimized", lambda: rt_opt.launch_bwd_weight(gy, x, outputs["noatomic_optimized"], K),
                 lambda: None, outputs["noatomic_optimized"]),
        ]

        # Validation before timing. Never report performance of an incorrect kernel.
        print(f"GPU: {gpu} | B,H,L,K={B},{H},{L},{K} | dK threads={dK_threads}", flush=True)
        reference = reference_dk(x, k, gy)
        torch.cuda.synchronize()
        correctness = []
        for name, runtime in runtimes:
            gk = outputs[name]
            if name == "atomic":
                gk.zero_()
            runtime.launch_bwd_weight(gy, x, gk, K)
            torch.cuda.synchronize()
            max_abs = float((gk - reference).abs().max().item())
            is_close = bool(torch.allclose(gk, reference, atol=args.atol, rtol=args.rtol))
            correctness.append({"design": name, "max_abs": max_abs, "allclose": is_close})
            print(f"Correctness {name:>20}: allclose={is_close}, max_abs={max_abs:.6g}", flush=True)
        import csv
        with (out_dir / "correctness.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["design", "max_abs", "allclose"])
            writer.writeheader()
            writer.writerows(correctness)
        if not all(row["allclose"] for row in correctness):
            raise RuntimeError("Correctness failed; no benchmark timings will be reported")

        raw = []
        summaries = []
        for trial_id in range(1, args.trials + 1):
            # Rotate execution order between trials to reduce systematic drift bias.
            ordered = modes[trial_id - 1:] + modes[:trial_id - 1]
            for mode in ordered:
                times = event_measure(mode, args.warmup, args.runs)
                for run_id, ms in enumerate(times, 1):
                    raw.append({"trial": trial_id, "mode": mode.name, "run": run_id, "runtime_ms": ms})
                mean_ms = statistics.mean(times)
                std_ms = statistics.stdev(times) if len(times) > 1 else 0.0
                summaries.append({"trial": trial_id, "mode": mode.name,
                                  "mean_ms": mean_ms, "std_ms": std_ms})
                print(f"trial {trial_id}: {mode.name:>20} {mean_ms:.5f} +/- {std_ms:.5f} ms", flush=True)
        for filename, rows, fieldnames in (
            ("raw.csv", raw, ["trial", "mode", "run", "runtime_ms"]),
            ("trial_summary.csv", summaries, ["trial", "mode", "mean_ms", "std_ms"]),
        ):
            with (out_dir / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        print("\nMean of trial means (not a new paper aggregate):", flush=True)
        for mode in modes:
            trial_means = [item["mean_ms"] for item in summaries if item["mode"] == mode.name]
            print(f"  {mode.name:>20}: {statistics.mean(trial_means):.5f} ms", flush=True)
        print(f"Saved: {out_dir}", flush=True)
    finally:
        for _, runtime in reversed(runtimes):
            runtime.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=Path("results/dk_fairness"))
    args = parser.parse_args()
    for key in ("B", "H", "L", "K", "trials", "warmup", "runs"):
        if getattr(args, key) < (0 if key == "warmup" else 1):
            parser.error(f"--{key} must be positive (warmup may be zero)")
    run(args)


if __name__ == "__main__":
    main()
