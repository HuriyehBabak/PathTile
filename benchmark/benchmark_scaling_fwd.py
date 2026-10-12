from __future__ import annotations

import math
import numpy as np
import torch

from kernels.coalesced import (
    DepthwiseConvCoalesced,
    FWD_TIMES as GMC_FWD,
    reset_timing_collectors as reset_gmc,
)

from kernels.shared import (
    DepthwiseConvShared,
    FWD_TIMES as SHARED_FWD,
    reset_timing_collectors as reset_shared,
)


SEED = 42
H = 128
WARMUP = 5
RUNS = 30

CASES = [
    (16384, 48, 48),
    (8192, 96, 96),
    (4096, 192, 192),
    (2048, 384, 384),
]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    half = 2.045229642 * std / math.sqrt(len(x))
    return mean, std, mean-half, mean+half


def benchmark_one(fn, collector, reset_fn, x, k):
    reset_fn()

    for _ in range(WARMUP):
        _ = fn.apply(x, k)
        torch.cuda.synchronize()

    reset_fn()

    for _ in range(RUNS):
        _ = fn.apply(x, k)
        torch.cuda.synchronize()

    return stats(collector)


def main():
    torch.cuda.set_device(0)

    print("=" * 90)
    print("FORWARD SCALING STUDY: GMC vs SHARED")
    print("=" * 90)
    print("GPU:", torch.cuda.get_device_name(0))
    print("H =", H)
    print("Warm-up =", WARMUP)
    print("Runs =", RUNS)

    results = []

    for B, L, K in CASES:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        print("\n" + "-" * 90)
        print(f"B={B}, H={H}, L={L}, K={K}, B*L={B*L}")
        print("-" * 90)

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

        gmc = benchmark_one(
            DepthwiseConvCoalesced,
            GMC_FWD,
            reset_gmc,
            x,
            k,
        )

        shared = benchmark_one(
            DepthwiseConvShared,
            SHARED_FWD,
            reset_shared,
            x,
            k,
        )

        speedup = gmc[0] / shared[0]

        print(f"GMC mean:     {gmc[0]:.6f} ms")
        print(f"Shared mean:  {shared[0]:.6f} ms")
        print(f"Shared speedup vs GMC: {speedup:.4f}x")

        results.append(
            (B, L, K, gmc[0], shared[0], speedup)
        )

        del x, k
        torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"{'B':>8} {'L':>6} {'K':>6} "
        f"{'GMC ms':>12} {'Shared ms':>12} {'Speedup':>10}"
    )

    for B, L, K, gmc, shared, speedup in results:
        print(
            f"{B:8d} {L:6d} {K:6d} "
            f"{gmc:12.6f} {shared:12.6f} {speedup:10.4f}x"
        )


if __name__ == "__main__":
    main()
