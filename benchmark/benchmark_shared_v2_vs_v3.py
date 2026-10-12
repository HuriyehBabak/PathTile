from __future__ import annotations

import math

import numpy as np
import torch

import kernels.shared as v2
import kernels.shared_v3 as v3


B = 16384
H = 128
L = 48
K = 48

WARMUP = 5
RUNS = 30
SEED = 42


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    tcrit = 2.045229642
    half = tcrit * std / math.sqrt(len(x))
    return mean, std, mean - half, mean + half


def run_v2(x, k, gy):
    v2.reset_timing_collectors()

    y = v2.DepthwiseConvShared.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
    )

    torch.cuda.synchronize()

    out = (
        float(v2.FWD_TIMES[-1]),
        float(v2.BWD_IN_TIMES[-1]),
        float(v2.BWD_K_TIMES[-1]),
    )

    del y, gx, gk

    return out


def run_v3(x, k, gy):
    v3.reset_timing_collectors()

    y = v3.DepthwiseConvShared.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
    )

    torch.cuda.synchronize()

    out = (
        float(v3.FWD_TIMES[-1]),
        float(v3.BWD_IN_TIMES[-1]),
        float(v3.BWD_K_TIMES[-1]),
    )

    del y, gx, gk

    return out


def main():
    gpu = torch.cuda.get_device_name(0)

    if "A6000" not in gpu:
        raise RuntimeError(f"Expected RTX A6000, got: {gpu}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

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

    print("=" * 82)
    print("SHARED-v2 vs SHARED-v3")
    print("=" * 82)
    print("GPU:", gpu)
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(f"Warm-up={WARMUP}, Runs={RUNS}")
    print()
    print("v2 FWD/BWD-IN SMEM:", v2.shared_memory_bytes(K))
    print("v3 FWD/BWD-IN SMEM:", v3.shared_memory_bytes(K))

    print("\nWarm-up...")

    for i in range(WARMUP):
        run_v2(x, k, gy)
        run_v3(x, k, gy)
        print(f"  {i+1}/{WARMUP}")

    r2 = []
    r3 = []

    print("\nMeasured runs...")

    for i in range(RUNS):

        # Alternate order to reduce thermal/clock bias.
        if i % 2 == 0:
            a = run_v2(x, k, gy)
            b = run_v3(x, k, gy)
        else:
            b = run_v3(x, k, gy)
            a = run_v2(x, k, gy)

        r2.append(a)
        r3.append(b)

        print(
            f"  run {i+1:2d}/{RUNS} | "
            f"v2 {a[0]:.3f}/{a[1]:.3f}/{a[2]:.3f} ms | "
            f"v3 {b[0]:.3f}/{b[1]:.3f}/{b[2]:.3f} ms"
        )

    r2 = np.asarray(r2)
    r3 = np.asarray(r3)

    names = [
        "Forward",
        "Backward Input",
        "Backward Kernel",
    ]

    print("\n" + "=" * 82)
    print("RESULTS")
    print("=" * 82)

    for j, name in enumerate(names):
        m2, s2, lo2, hi2 = stats(r2[:, j])
        m3, s3, lo3, hi3 = stats(r3[:, j])

        print(f"\n{name}")
        print(
            f"  v2: {m2:.6f} ms "
            f"(std {s2:.6f}, CI [{lo2:.6f}, {hi2:.6f}])"
        )
        print(
            f"  v3: {m3:.6f} ms "
            f"(std {s3:.6f}, CI [{lo3:.6f}, {hi3:.6f}])"
        )
        print(f"  v3 speedup vs v2: {m2/m3:.4f}x")

    total2 = r2.mean(axis=0).sum()
    total3 = r3.mean(axis=0).sum()

    print("\nThree-kernel total")
    print(f"  v2: {total2:.6f} ms")
    print(f"  v3: {total3:.6f} ms")
    print(f"  v3 speedup vs v2: {total2/total3:.4f}x")


if __name__ == "__main__":
    main()
