from __future__ import annotations

import csv
import math
import platform
from pathlib import Path

import numpy as np
import torch

from kernels.naive import (
    DepthwiseConvNaive,
    clear_timings,
    get_runtime_data,
)


# ============================================================
# Experiment configuration
# ============================================================

TECHNIQUE = "naive"

# Original/main paper configuration
B = 16384
H = 128
L = 48
K = 48

DTYPE = torch.float32
TPB = 512

# Initial statistically repeated benchmark.
# Later we can increase N_RUNS if needed.
N_WARMUP = 5
N_RUNS = 30

SEED = 42

RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "naive"
    / "runtime"
)


# ============================================================
# Statistics
# ============================================================

def summarize(values):
    values = np.asarray(values, dtype=np.float64)

    n = len(values)

    if n == 0:
        raise ValueError("No runtime measurements found.")

    mean = float(np.mean(values))
    median = float(np.median(values))

    if n > 1:
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)

        # 95% normal-approximation CI.
        margin = 1.96 * sem
    else:
        std = 0.0
        margin = 0.0

    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
    }


# ============================================================
# Analytical memory-traffic model
# ============================================================

FLOAT_BYTES = 4


def unique_fwd_bytes(B, H, L, K):
    """
    Minimum/unique tensor footprint for forward:

        read x      : [B,H,L]
        read kernel : [H,K]
        write y     : [B,H,L]

    This is NOT the number of logical load instructions and
    should not be called measured DRAM traffic.
    """
    input_bytes = B * H * L * FLOAT_BYTES
    kernel_bytes = H * K * FLOAT_BYTES
    output_bytes = B * H * L * FLOAT_BYTES

    return (
        input_bytes
        + kernel_bytes
        + output_bytes
    )


def unique_bwd_input_bytes(B, H, L, K):
    """
    Unique tensor footprint for grad-input:

        read grad_y : [B,H,L]
        read kernel : [H,K]
        write grad_x: [B,H,L]
    """
    grad_y_bytes = B * H * L * FLOAT_BYTES
    kernel_bytes = H * K * FLOAT_BYTES
    grad_x_bytes = B * H * L * FLOAT_BYTES

    return (
        grad_y_bytes
        + kernel_bytes
        + grad_x_bytes
    )


def unique_bwd_kernel_bytes(B, H, L, K):
    """
    Unique tensor footprint for grad-kernel:

        read grad_y : [B,H,L]
        read x      : [B,H,L]
        write grad_k: [H,K]
    """
    grad_y_bytes = B * H * L * FLOAT_BYTES
    input_bytes = B * H * L * FLOAT_BYTES
    grad_k_bytes = H * K * FLOAT_BYTES

    return (
        grad_y_bytes
        + input_bytes
        + grad_k_bytes
    )


def bytes_to_mb(value):
    return value / 1e6

# ============================================================
# One complete FWD + backward measurement
# ============================================================

def run_once(
    x_base,
    k_base,
    grad_y,
):
    x = (
        x_base
        .detach()
        .clone()
        .requires_grad_(True)
    )

    k = (
        k_base
        .detach()
        .clone()
        .requires_grad_(True)
    )

    y = DepthwiseConvNaive.apply(
        x,
        k,
    )

    y.backward(
        grad_y
    )

    torch.cuda.synchronize()


# ============================================================
# CSV writers
# ============================================================

def save_raw_csv(runtime_data):
    path = RESULT_DIR / "naive_raw_kernel_times_a6000.csv"

    rows = []

    phase_map = {
        "fwd_ms": "forward",
        "bwd_input_ms": "backward_input",
        "bwd_kernel_ms": "backward_kernel",
    }

    for key, phase in phase_map.items():
        for run_id, runtime_ms in enumerate(
            runtime_data[key],
            start=1,
        ):
            rows.append(
                {
                    "technique": TECHNIQUE,
                    "gpu": torch.cuda.get_device_name(0),
                    "B": B,
                    "H": H,
                    "L": L,
                    "K": K,
                    "threads_per_block": TPB,
                    "phase": phase,
                    "run_id": run_id,
                    "runtime_ms": runtime_ms,
                }
            )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_summary_csv(runtime_data):
    path = RESULT_DIR / "naive_runtime_summary_a6000.csv"

    phase_map = {
        "forward": runtime_data["fwd_ms"],
        "backward_input": runtime_data["bwd_input_ms"],
        "backward_kernel": runtime_data["bwd_kernel_ms"],
    }

    rows = []

    for phase, values in phase_map.items():
        stats = summarize(values)

        rows.append(
            {
                "technique": TECHNIQUE,
                "gpu": torch.cuda.get_device_name(0),
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "threads_per_block": TPB,
                "warmup_runs": N_WARMUP,
                "phase": phase,
                **stats,
            }
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    return path


def save_environment_csv():
    path = RESULT_DIR / "naive_environment_a6000.csv"

    props = torch.cuda.get_device_properties(0)

    row = {
        "technique": TECHNIQUE,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": (
            f"{props.major}.{props.minor}"
        ),
        "gpu_memory_gib": (
            props.total_memory / 1024**3
        ),
        "B": B,
        "H": H,
        "L": L,
        "K": K,
        "dtype": str(DTYPE),
        "threads_per_block": TPB,
        "warmup_runs": N_WARMUP,
        "measured_runs": N_RUNS,
        "seed": SEED,

        "fwd_unique_bytes": unique_fwd_bytes(B, H, L, K),
        "fwd_unique_mb": bytes_to_mb(
            unique_fwd_bytes(B, H, L, K)
        ),

        "bwd_input_unique_bytes": unique_bwd_input_bytes(
            B, H, L, K
        ),
        "bwd_input_unique_mb": bytes_to_mb(
            unique_bwd_input_bytes(B, H, L, K)
        ),

        "bwd_kernel_unique_bytes": unique_bwd_kernel_bytes(
            B, H, L, K
        ),
        "bwd_kernel_unique_mb": bytes_to_mb(
            unique_bwd_kernel_bytes(B, H, L, K)
        ),
    }

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys(),
        )

        writer.writeheader()
        writer.writerow(row)

    return path


# ============================================================
# Main
# ============================================================

def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. "
            "Run this benchmark inside a SLURM GPU job."
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DepthwiseConvNaive.THREADS_PER_BLOCK = TPB

    print()
    print("Naive CUDA benchmark")
    print("====================")

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Capability:",
        torch.cuda.get_device_capability(0),
    )

    print(
        f"Configuration: "
        f"B={B}, H={H}, L={L}, K={K}"
    )

    print("TPB:", TPB)
    print("Warm-up runs:", N_WARMUP)
    print("Measured runs:", N_RUNS)
    print()
    print("Analytical unique tensor footprint")
    print("----------------------------------")

    print(
        f"Forward:         "
        f"{bytes_to_mb(unique_fwd_bytes(B, H, L, K)):.3f} MB"
    )

    print(
        f"Backward input:  "
        f"{bytes_to_mb(unique_bwd_input_bytes(B, H, L, K)):.3f} MB"
    )

    print(
        f"Backward kernel: "
        f"{bytes_to_mb(unique_bwd_kernel_bytes(B, H, L, K)):.3f} MB"
    )

    # --------------------------------------------------------
    # Allocate input once
    # --------------------------------------------------------

    x_base = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    )

    k_base = torch.randn(
        H,
        K,
        device="cuda",
        dtype=DTYPE,
    )

    grad_y = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    )

    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print()
    print("Warm-up...")

    clear_timings()

    for i in range(N_WARMUP):
        run_once(
            x_base,
            k_base,
            grad_y,
        )

        print(
            f"  warm-up {i + 1}/{N_WARMUP}"
        )

    # Discard ALL warm-up measurements.
    clear_timings()

    # --------------------------------------------------------
    # Measured runs
    # --------------------------------------------------------

    print()
    print("Measured runs...")

    for i in range(N_RUNS):
        run_once(
            x_base,
            k_base,
            grad_y,
        )

        print(
            f"  run {i + 1}/{N_RUNS}"
        )

    runtime_data = get_runtime_data()

    # --------------------------------------------------------
    # Validate number of measurements
    # --------------------------------------------------------

    assert len(runtime_data["fwd_ms"]) == N_RUNS
    assert len(runtime_data["bwd_input_ms"]) == N_RUNS
    assert len(runtime_data["bwd_kernel_ms"]) == N_RUNS

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("Results")
    print("=======")

    phase_map = {
        "Forward": runtime_data["fwd_ms"],
        "Backward input": runtime_data["bwd_input_ms"],
        "Backward kernel": runtime_data["bwd_kernel_ms"],
    }

    for phase, values in phase_map.items():
        stats = summarize(values)

        print()
        print(phase)
        print("-" * len(phase))

        print(
            f"N:       {stats['n']}"
        )

        print(
            f"Mean:    {stats['mean_ms']:.6f} ms"
        )

        print(
            f"Median:  {stats['median_ms']:.6f} ms"
        )

        print(
            f"Std:     {stats['std_ms']:.6f} ms"
        )

        print(
            f"95% CI:  "
            f"[{stats['ci95_low_ms']:.6f}, "
            f"{stats['ci95_high_ms']:.6f}] ms"
        )

        print(
            f"Min/Max: "
            f"{stats['min_ms']:.6f} / "
            f"{stats['max_ms']:.6f} ms"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    raw_path = save_raw_csv(
        runtime_data
    )

    summary_path = save_summary_csv(
        runtime_data
    )

    env_path = save_environment_csv()

    print()
    print("Saved:")
    print(" ", raw_path)
    print(" ", summary_path)
    print(" ", env_path)


if __name__ == "__main__":
    main()