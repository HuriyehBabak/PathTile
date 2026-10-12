from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from kernels.coalesced import (
    BWD_IN_TIMES,
    BWD_K_TIMES,
    FWD_TIMES,
    HTILE,
    THREADS_PER_BLOCK,
    TTILE,
    DepthwiseConvCoalesced,
    reset_timing_collectors,
)


# ============================================================
# Fixed paper workload
# ============================================================

SEED = 42

B = 16384
H = 128
L = 48
K = 48

WARMUP = 50
MEASURED_RUNS = 100

DTYPE = torch.float32

OUT_DIR = Path("results/a6000_final_baselines/coalesced/current")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Statistics
# ============================================================

def summary_stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)

    n = len(arr)

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))

    sem = std / math.sqrt(n)

    # Student-t critical value for:
    # 95% CI, df = 29 (n = 30)
    t_critical = 2.045229642

    margin = t_critical * sem

    return {
        "n": n,
        "mean_ms": mean,
        "std_ms": std,
        "sem_ms": sem,
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
    }


def print_summary(
    name: str,
    values: list[float],
) -> dict[str, float]:

    s = summary_stats(values)

    print()
    print(f"{name}:")
    print(f"Mean = {s['mean_ms']:.6f} ms")
    print(f"Std  = {s['std_ms']:.6f} ms")
    print(
        "95% CI = "
        f"[{s['ci95_low_ms']:.6f}, "
        f"{s['ci95_high_ms']:.6f}]"
    )

    return s


# ============================================================
# Main benchmark
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)

    print()
    print("Global Memory Coalescing benchmark")
    print("==================================")
    print(f"GPU: {gpu_name}")
    print(f"PyTorch: {torch.__version__}")
    print(
        "Compute Capability: "
        f"{capability[0]}.{capability[1]}"
    )

    print(
        f"B={B}, H={H}, L={L}, K={K}"
    )

    print(
        f"TPB={THREADS_PER_BLOCK}, "
        f"TTILE={TTILE}, HTILE={HTILE}"
    )

    print(
        f"FP32, seed={SEED}, "
        f"warm-up={WARMUP}, "
        f"measured runs={MEASURED_RUNS}"
    )

    # --------------------------------------------------------
    # Guard against accidentally benchmarking another GPU.
    # --------------------------------------------------------

    if "A6000" not in gpu_name:
        raise RuntimeError(
            "Final GMC runtime benchmark must run on "
            "the NVIDIA RTX A6000 to match the Naive baseline. "
            f"Current GPU: {gpu_name}"
        )

    if THREADS_PER_BLOCK != 512:
        raise RuntimeError(
            f"Expected TPB=512, got {THREADS_PER_BLOCK}"
        )

    if TTILE != 32 or HTILE != 16:
        raise RuntimeError(
            f"Unexpected mapping: TTILE={TTILE}, HTILE={HTILE}"
        )

    # --------------------------------------------------------
    # Reproducible inputs
    # --------------------------------------------------------

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    x = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    ).contiguous().requires_grad_(True)

    kernel = torch.randn(
        H,
        K,
        device="cuda",
        dtype=DTYPE,
    ).contiguous().requires_grad_(True)

    grad_out = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    ).contiguous()

    torch.cuda.synchronize()

    print()
    print("Input shapes:")
    print(f"x          = {tuple(x.shape)}")
    print(f"kernel     = {tuple(kernel.shape)}")
    print(f"grad_out   = {tuple(grad_out.shape)}")

    # ========================================================
    # Warm-up
    # ========================================================

    print()
    print("Warm-up...")

    reset_timing_collectors()

    for i in range(WARMUP):
        x.grad = None
        kernel.grad = None

        y = DepthwiseConvCoalesced.apply(
            x,
            kernel,
        )

        y.backward(
            grad_out
        )

        torch.cuda.synchronize()

        print(
            f"  warm-up {i + 1}/{WARMUP}"
        )

    # Discard every warm-up timing.
    reset_timing_collectors()

    # ========================================================
    # Measured runs
    # ========================================================

    print()
    print("Measured runs...")

    for i in range(MEASURED_RUNS):
        x.grad = None
        kernel.grad = None

        y = DepthwiseConvCoalesced.apply(
            x,
            kernel,
        )

        y.backward(
            grad_out
        )

        torch.cuda.synchronize()

        print(
            f"  run {i + 1}/{MEASURED_RUNS}"
        )

    # ========================================================
    # Sanity checks
    # ========================================================

    if len(FWD_TIMES) != MEASURED_RUNS:
        raise RuntimeError(
            "Unexpected number of Forward timings: "
            f"{len(FWD_TIMES)}"
        )

    if len(BWD_IN_TIMES) != MEASURED_RUNS:
        raise RuntimeError(
            "Unexpected number of Backward Input timings: "
            f"{len(BWD_IN_TIMES)}"
        )

    if len(BWD_K_TIMES) != MEASURED_RUNS:
        raise RuntimeError(
            "Unexpected number of Backward Kernel timings: "
            f"{len(BWD_K_TIMES)}"
        )

    # Make independent Python copies.
    fwd_times = [
        float(v)
        for v in FWD_TIMES
    ]

    bwd_in_times = [
        float(v)
        for v in BWD_IN_TIMES
    ]

    bwd_k_times = [
        float(v)
        for v in BWD_K_TIMES
    ]

    # ========================================================
    # Statistics
    # ========================================================

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)

    fwd_stats = print_summary(
        "Forward",
        fwd_times,
    )

    bwd_in_stats = print_summary(
        "Backward Input",
        bwd_in_times,
    )

    bwd_k_stats = print_summary(
        "Backward Kernel",
        bwd_k_times,
    )

    # ========================================================
    # Compare numerically to frozen Naive baseline
    # ========================================================

    naive_fwd_ms = 6.213768
    naive_bwd_in_ms = 6.077338
    naive_bwd_k_ms = 32.366277

    speedup_fwd = (
        naive_fwd_ms
        / fwd_stats["mean_ms"]
    )

    speedup_bwd_in = (
        naive_bwd_in_ms
        / bwd_in_stats["mean_ms"]
    )

    speedup_bwd_k = (
        naive_bwd_k_ms
        / bwd_k_stats["mean_ms"]
    )

    print()
    print("=" * 72)
    print("COMPARISON WITH FROZEN NAIVE BASELINE")
    print("=" * 72)

    print(
        f"Forward:        "
        f"Naive {naive_fwd_ms:.6f} ms | "
        f"GMC {fwd_stats['mean_ms']:.6f} ms | "
        f"speedup {speedup_fwd:.4f}x"
    )

    print(
        f"Backward Input: "
        f"Naive {naive_bwd_in_ms:.6f} ms | "
        f"GMC {bwd_in_stats['mean_ms']:.6f} ms | "
        f"speedup {speedup_bwd_in:.4f}x"
    )

    print(
        f"Backward Kernel:"
        f" Naive {naive_bwd_k_ms:.6f} ms | "
        f"GMC {bwd_k_stats['mean_ms']:.6f} ms | "
        f"speedup {speedup_bwd_k:.4f}x"
    )

    # ========================================================
    # Save raw CSV
    # ========================================================

    raw_csv = (
        OUT_DIR
        / "coalesced_runtime_raw.csv"
    )

    with raw_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "run_id",
                "forward_ms",
                "backward_input_ms",
                "backward_kernel_ms",
            ]
        )

        for i in range(MEASURED_RUNS):
            writer.writerow(
                [
                    i + 1,
                    fwd_times[i],
                    bwd_in_times[i],
                    bwd_k_times[i],
                ]
            )

    # ========================================================
    # Save summary CSV
    # ========================================================

    summary_csv = (
        OUT_DIR
        / "coalesced_runtime_summary.csv"
    )

    rows = [
        (
            "forward",
            fwd_stats,
            naive_fwd_ms,
            speedup_fwd,
        ),
        (
            "backward_input",
            bwd_in_stats,
            naive_bwd_in_ms,
            speedup_bwd_in,
        ),
        (
            "backward_kernel",
            bwd_k_stats,
            naive_bwd_k_ms,
            speedup_bwd_k,
        ),
    ]

    with summary_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "phase",
                "n",
                "mean_ms",
                "std_ms",
                "sem_ms",
                "ci95_low_ms",
                "ci95_high_ms",
                "naive_mean_ms",
                "speedup_vs_naive",
            ]
        )

        for (
            phase,
            stats,
            naive_ms,
            speedup,
        ) in rows:
            writer.writerow(
                [
                    phase,
                    stats["n"],
                    stats["mean_ms"],
                    stats["std_ms"],
                    stats["sem_ms"],
                    stats["ci95_low_ms"],
                    stats["ci95_high_ms"],
                    naive_ms,
                    speedup,
                ]
            )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata = {
        "technique": "Global Memory Coalescing",
        "gpu": gpu_name,
        "compute_capability": (
            f"{capability[0]}.{capability[1]}"
        ),
        "pytorch": torch.__version__,
        "dtype": "float32",
        "seed": SEED,
        "B": B,
        "H": H,
        "L": L,
        "K": K,
        "threads_per_block": THREADS_PER_BLOCK,
        "ttile": TTILE,
        "htile": HTILE,
        "warmup_runs": WARMUP,
        "measured_runs": MEASURED_RUNS,
        "notes": (
            "No shared memory, no warp shuffle, "
            "no partial reductions. "
            "Backward-kernel algorithm retained "
            "from frozen Naive baseline."
        ),
    }

    metadata_path = (
        OUT_DIR
        / "coalesced_runtime_metadata.json"
    )

    with metadata_path.open("w") as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print("=" * 72)
    print("SAVED")
    print("=" * 72)

    print(raw_csv)
    print(summary_csv)
    print(metadata_path)


if __name__ == "__main__":
    main()
