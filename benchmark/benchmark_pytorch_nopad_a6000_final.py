from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

# Strict FP32 comparison
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


B = 16384
H = 128
L = 48
K = 48

DTYPE = torch.float32

N_WARMUP = 50
N_RUNS = 100
SEED = 42


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


def depthwise_conv1d_nopad(x_pad, weight):
    return F.conv1d(
        x_pad,
        weight,
        bias=None,
        stride=1,
        padding=0,
        groups=H,
    )


def timed_forward(x_pad, weight):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    y = depthwise_conv1d_nopad(
        x_pad,
        weight,
    )

    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def timed_bwd_input(x_pad, weight, grad_y):
    x_local = (
        x_pad
        .detach()
        .clone()
        .requires_grad_(True)
    )

    w_local = (
        weight
        .detach()
        .clone()
        .requires_grad_(False)
    )

    y = depthwise_conv1d_nopad(
        x_local,
        w_local,
    )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    grad_x_pad = torch.autograd.grad(
        y,
        x_local,
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )[0]

    end.record()
    torch.cuda.synchronize()

    _ = grad_x_pad

    return start.elapsed_time(end)


def timed_bwd_kernel(x_pad, weight, grad_y):
    x_local = (
        x_pad
        .detach()
        .clone()
        .requires_grad_(False)
    )

    w_local = (
        weight
        .detach()
        .clone()
        .requires_grad_(True)
    )

    y = depthwise_conv1d_nopad(
        x_local,
        w_local,
    )

    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    grad_weight = torch.autograd.grad(
        y,
        w_local,
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )[0]

    end.record()
    torch.cuda.synchronize()

    _ = grad_weight

    return start.elapsed_time(end)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    print()
    print("PyTorch/cuDNN no-pad benchmark")
    print("==============================")
    print("GPU:", torch.cuda.get_device_name(0))
    print("PyTorch:", torch.__version__)
    print("cuDNN:", torch.backends.cudnn.version())
    print(f"B={B}, H={H}, L={L}, K={K}")

    x = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    )

    weight = torch.randn(
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

    left = K // 2
    right = K - 1 - left

    x_pad = F.pad(
        x,
        (left, right),
    )

    torch.cuda.synchronize()

    print()
    print("Original input shape:", tuple(x.shape))
    print("Padded input shape:  ", tuple(x_pad.shape))

    results = {
        "forward": [],
        "backward_input": [],
        "backward_kernel": [],
    }

    print()
    print("Warm-up...")

    for i in range(N_WARMUP):
        _ = timed_forward(
            x_pad,
            weight,
        )

        _ = timed_bwd_input(
            x_pad,
            weight,
            grad_y,
        )

        _ = timed_bwd_kernel(
            x_pad,
            weight,
            grad_y,
        )

        print(f"  warm-up {i + 1}/{N_WARMUP}")

    print()
    print("Measured runs...")

    for i in range(N_RUNS):
        results["forward"].append(
            timed_forward(
                x_pad,
                weight,
            )
        )

        results["backward_input"].append(
            timed_bwd_input(
                x_pad,
                weight,
                grad_y,
            )
        )

        results["backward_kernel"].append(
            timed_bwd_kernel(
                x_pad,
                weight,
                grad_y,
            )
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
        print(f"N:       {stats['n']}")
        print(f"Mean:    {stats['mean_ms']:.6f} ms")
        print(f"Median:  {stats['median_ms']:.6f} ms")
        print(f"Std:     {stats['std_ms']:.6f} ms")
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


if __name__ == "__main__":
    main()