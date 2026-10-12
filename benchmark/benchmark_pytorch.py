from __future__ import annotations

import csv
import math
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


TECHNIQUE = "pytorch_cudnn"

B = 16384
H = 128
L = 48
K = 48

DTYPE = torch.float32

N_WARMUP = 5
N_RUNS = 30
SEED = 42

RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "pytorch"
    / "runtime"
)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)

    mean = float(np.mean(values))
    median = float(np.median(values))

    if n > 1:
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)
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


def depthwise_conv1d_same(x, weight):
    left = K // 2
    right = K - 1 - left

    x_pad = F.pad(
        x,
        (left, right),
    )

    return F.conv1d(
        x_pad,
        weight,
        bias=None,
        stride=1,
        padding=0,
        groups=H,
    )


def timed_forward(x, weight):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    y = depthwise_conv1d_same(
        x,
        weight,
    )

    end.record()
    torch.cuda.synchronize()

    return y, start.elapsed_time(end)


def timed_bwd_input(x, weight, grad_y):
    y = depthwise_conv1d_same(
        x,
        weight,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    grad_x = torch.autograd.grad(
        y,
        x,
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )[0]

    end.record()
    torch.cuda.synchronize()

    return grad_x, start.elapsed_time(end)


def timed_bwd_kernel(x, weight, grad_y):
    y = depthwise_conv1d_same(
        x,
        weight,
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    grad_weight = torch.autograd.grad(
        y,
        weight,
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )[0]

    end.record()
    torch.cuda.synchronize()

    return grad_weight, start.elapsed_time(end)


def save_summary(results):
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULT_DIR / "pytorch_runtime_summary_a6000.csv"

    rows = []

    for phase, values in results.items():
        stats = summarize(values)

        rows.append(
            {
                "technique": TECHNIQUE,
                "gpu": torch.cuda.get_device_name(0),
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": phase,
                "warmup_runs": N_WARMUP,
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


def save_environment():
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = RESULT_DIR / "pytorch_environment_a6000.csv"

    props = torch.cuda.get_device_properties(0)

    row = {
        "technique": TECHNIQUE,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": (
            f"{props.major}.{props.minor}"
        ),
        "B": B,
        "H": H,
        "L": L,
        "K": K,
        "dtype": str(DTYPE),
        "warmup_runs": N_WARMUP,
        "measured_runs": N_RUNS,
        "seed": SEED,
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


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print()
    print("PyTorch/cuDNN depthwise Conv1D benchmark")
    print("=======================================")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("cuDNN:", torch.backends.cudnn.version())
    print(
        f"B={B}, H={H}, L={L}, K={K}"
    )

    x_base = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    )

    weight_base = torch.randn(
        H,
        1,
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

    results = {
        "forward": [],
        "backward_input": [],
        "backward_kernel": [],
    }

    print()
    print("Warm-up...")

    for i in range(N_WARMUP):
        x = x_base.detach().clone().requires_grad_(True)
        w = weight_base.detach().clone().requires_grad_(True)

        y = depthwise_conv1d_same(x, w)
        y.backward(grad_y)

        torch.cuda.synchronize()

        print(f"  warm-up {i + 1}/{N_WARMUP}")

    print()
    print("Measured runs...")

    for i in range(N_RUNS):
        # Forward
        x = x_base.detach().clone().requires_grad_(True)
        w = weight_base.detach().clone().requires_grad_(True)

        _, fwd_ms = timed_forward(
            x,
            w,
        )

        results["forward"].append(fwd_ms)

        # Backward input
        x = x_base.detach().clone().requires_grad_(True)
        w = weight_base.detach().clone().requires_grad_(True)

        _, bwd_input_ms = timed_bwd_input(
            x,
            w,
            grad_y,
        )

        results["backward_input"].append(
            bwd_input_ms
        )

        # Backward kernel
        x = x_base.detach().clone().requires_grad_(True)
        w = weight_base.detach().clone().requires_grad_(True)

        _, bwd_kernel_ms = timed_bwd_kernel(
            x,
            w,
            grad_y,
        )

        results["backward_kernel"].append(
            bwd_kernel_ms
        )

        print(f"  run {i + 1}/{N_RUNS}")

    print()
    print("Results")
    print("=======")

    for phase, values in results.items():
        stats = summarize(values)

        print()
        print(phase)
        print("-" * len(phase))
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

    summary_path = save_summary(results)
    env_path = save_environment()

    print()
    print("Saved:")
    print(" ", summary_path)
    print(" ", env_path)


if __name__ == "__main__":
    main()
