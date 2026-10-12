from __future__ import annotations

import math

import numpy as np
import torch

import kernels.shared_btile16 as s16
import kernels.shared_btile32 as s32
import kernels.shared as s64
import kernels.shared_btile128 as s128


B = 16384
H = 128
L = 48
K = 48

SEED = 42
WARMUP = 5
RUNS = 30


VARIANTS = [
    ("16", s16),
    ("32", s32),
    ("64", s64),
    ("128", s128),
]


def stats(values):
    x = np.asarray(values, dtype=np.float64)

    mean = float(x.mean())
    std = float(x.std(ddof=1))

    tcrit = 2.045229642
    half = tcrit * std / math.sqrt(len(x))

    return mean, std, mean - half, mean + half


def run(module, x, k, gy):
    module.reset_timing_collectors()

    y = module.DepthwiseConvShared.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
    )

    torch.cuda.synchronize()

    bwd_k_ms = float(module.BWD_K_TIMES[-1])

    return bwd_k_ms, gk


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    gpu = torch.cuda.get_device_name(0)

    if "A6000" not in gpu:
        raise RuntimeError(
            f"Expected NVIDIA RTX A6000, got {gpu}"
        )

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
    print("SHARED BWD-K B_TILE SWEEP")
    print("=" * 82)
    print("GPU:", gpu)
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(f"Warm-up={WARMUP}, Runs={RUNS}")

    print("\nShared memory per block:")

    for name, module in VARIANTS:
        print(
            f"  B_TILE={name:>3}: "
            f"{module.bwd_kernel_shared_memory_bytes(L)} bytes"
        )

    # ========================================================
    # Correctness
    # ========================================================

    print("\n" + "=" * 82)
    print("CORRECTNESS")
    print("=" * 82)

    _, ref = run(s64, x, k, gy)

    for name, module in VARIANTS:
        _, out = run(module, x, k, gy)

        max_abs = (out - ref).abs().max().item()
        exact = torch.equal(out, ref)

        print(
            f"B_TILE={name:>3}: "
            f"max abs={max_abs:.9e} | "
            f"exact={exact}"
        )

        if not exact:
            raise RuntimeError(
                f"B_TILE={name} differs from baseline B_TILE=64"
            )

    print("Correctness: PASS")

    # ========================================================
    # Warm-up
    # ========================================================

    print("\nWarm-up...")

    for w in range(WARMUP):
        for _, module in VARIANTS:
            run(module, x, k, gy)

        print(f"  warm-up {w+1}/{WARMUP}")

    # ========================================================
    # Timed sweep
    # ========================================================

    results = {
        name: []
        for name, _ in VARIANTS
    }

    print("\nMeasured runs...")

    for r in range(RUNS):

        # Rotate execution order each run.
        shift = r % len(VARIANTS)

        order = (
            VARIANTS[shift:]
            + VARIANTS[:shift]
        )

        current = {}

        for name, module in order:
            ms, _ = run(module, x, k, gy)

            results[name].append(ms)
            current[name] = ms

        print(
            f"  run {r+1:2d}/{RUNS} | "
            f"16={current['16']:.3f} | "
            f"32={current['32']:.3f} | "
            f"64={current['64']:.3f} | "
            f"128={current['128']:.3f} ms"
        )

    # ========================================================
    # Summary
    # ========================================================

    print("\n" + "=" * 82)
    print("RESULTS")
    print("=" * 82)

    summary = {}

    for name, _ in VARIANTS:
        mean, std, lo, hi = stats(results[name])

        summary[name] = mean

        print(
            f"B_TILE={name:>3}: "
            f"{mean:.6f} ms | "
            f"std={std:.6f} | "
            f"95% CI=[{lo:.6f}, {hi:.6f}]"
        )

    baseline = summary["64"]

    print("\nRelative to current B_TILE=64:")

    for name in ["16", "32", "64", "128"]:
        speedup = baseline / summary[name]

        print(
            f"  B_TILE={name:>3}: "
            f"{speedup:.4f}x"
        )

    best = min(
        summary,
        key=summary.get,
    )

    print()
    print(
        f"BEST: B_TILE={best} "
        f"({summary[best]:.6f} ms)"
    )


if __name__ == "__main__":
    main()
