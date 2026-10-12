from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from kernels.shared import (
    DepthwiseConvShared,
    FWD_TIMES,
    BWD_IN_TIMES,
    BWD_K_TIMES,
    THREADS_PER_BLOCK,
    TTILE,
    HTILE,
    BWD_K_B_TILE,
    shared_memory_bytes,
    bwd_kernel_shared_memory_bytes,
    reset_timing_collectors,
)


SEED = 42

B = 16384
H = 128
L = 48
K = 48

WARMUP = 50
RUNS = 100

OUT_DIR = Path("results/v100_final_baselines/shared/current")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stats(values):
    a = np.asarray(values, dtype=np.float64)

    mean = float(a.mean())
    std = float(a.std(ddof=1))

    # Student-t critical value for 95% CI, df=29
    tcrit = 2.045229642
    half = tcrit * std / math.sqrt(len(a))

    return {
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": mean - half,
        "ci95_high_ms": mean + half,
    }


def run_once(x, k, gy):
    y = DepthwiseConvShared.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
        retain_graph=False,
        create_graph=False,
    )

    torch.cuda.synchronize()

    del y, gx, gk


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)

    gpu = torch.cuda.get_device_name(0)

    if "V100" not in gpu:
        raise RuntimeError(
            f"Benchmark must run on Tesla V100; current GPU: {gpu}"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    x = torch.randn(
        B, H, L,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    k = torch.randn(
        H, K,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    gy = torch.randn(
        B, H, L,
        device="cuda",
        dtype=torch.float32,
    )

    print("=" * 72)
    print("SHARED MEMORY BENCHMARK")
    print("=" * 72)
    print(f"GPU: {gpu}")
    print(f"PyTorch: {torch.__version__}")
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(f"FP32")
    print(f"TPB={THREADS_PER_BLOCK}")
    print(f"TTILE={TTILE}")
    print(f"HTILE={HTILE}")
    print(f"BWD_K_B_TILE={BWD_K_B_TILE}")
    print(f"FWD/BWD-IN shared={shared_memory_bytes(K)} bytes/block")
    print(f"BWD-K shared={bwd_kernel_shared_memory_bytes(L)} bytes/block")
    print(f"Warm-up={WARMUP}")
    print(f"Measured runs={RUNS}")
    print("=" * 72)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    reset_timing_collectors()

    print("\nWarm-up...")

    for i in range(WARMUP):
        run_once(x, k, gy)
        print(f"  warm-up {i + 1}/{WARMUP}")

    # Discard warm-up timings
    reset_timing_collectors()

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    print("\nMeasured runs...")

    for i in range(RUNS):
        run_once(x, k, gy)

        print(
            f"  run {i + 1:2d}/{RUNS} | "
            f"fwd={FWD_TIMES[-1]:.6f} ms | "
            f"bwd_in={BWD_IN_TIMES[-1]:.6f} ms | "
            f"bwd_k={BWD_K_TIMES[-1]:.6f} ms"
        )

    assert len(FWD_TIMES) == RUNS
    assert len(BWD_IN_TIMES) == RUNS
    assert len(BWD_K_TIMES) == RUNS

    results = {
        "forward": stats(FWD_TIMES),
        "backward_input": stats(BWD_IN_TIMES),
        "backward_kernel": stats(BWD_K_TIMES),
    }

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    labels = [
        ("Forward", "forward"),
        ("Backward Input", "backward_input"),
        ("Backward Kernel", "backward_kernel"),
    ]

    for label, key in labels:
        s = results[key]

        print(f"\n{label}:")
        print(f"Mean = {s['mean_ms']:.6f} ms")
        print(f"Std  = {s['std_ms']:.6f} ms")
        print(
            "95% CI = "
            f"[{s['ci95_low_ms']:.6f}, "
            f"{s['ci95_high_ms']:.6f}]"
        )

    # --------------------------------------------------------
    # Raw CSV
    # --------------------------------------------------------

    raw_path = OUT_DIR / "shared_runtime_raw.csv"

    with raw_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "run",
            "forward_ms",
            "backward_input_ms",
            "backward_kernel_ms",
        ])

        for i in range(RUNS):
            writer.writerow([
                i + 1,
                FWD_TIMES[i],
                BWD_IN_TIMES[i],
                BWD_K_TIMES[i],
            ])

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_path = OUT_DIR / "shared_runtime_summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "phase",
            "mean_ms",
            "std_ms",
            "ci95_low_ms",
            "ci95_high_ms",
        ])

        for label, key in labels:
            s = results[key]

            writer.writerow([
                label,
                s["mean_ms"],
                s["std_ms"],
                s["ci95_low_ms"],
                s["ci95_high_ms"],
            ])

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "gpu": gpu,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
        "B": B,
        "H": H,
        "L": L,
        "K": K,
        "dtype": "float32",
        "threads_per_block": THREADS_PER_BLOCK,
        "ttile": TTILE,
        "htile": HTILE,
        "bwd_k_b_tile": BWD_K_B_TILE,
        "fwd_bwd_input_shared_bytes":
            shared_memory_bytes(K),
        "bwd_kernel_shared_bytes":
            bwd_kernel_shared_memory_bytes(L),
        "warmup_runs": WARMUP,
        "measured_runs": RUNS,
    }

    metadata_path = (
        OUT_DIR / "shared_runtime_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nSaved:")
    print(raw_path)
    print(summary_path)
    print(metadata_path)


if __name__ == "__main__":
    main()
