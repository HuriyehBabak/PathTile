from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

import kernels.coalesced as gmc
import kernels.shared_l192 as shared


SEED = 42

B = 16384
H = 128
L = 192
K = 192

WARMUP = 5
RUNS = 30

OUT_DIR = Path("results/shared/l192")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def stats(values):
    x = np.asarray(values, dtype=np.float64)

    mean = float(x.mean())
    std = float(x.std(ddof=1))

    # 95% CI, df=29
    tcrit = 2.045229642
    half = tcrit * std / math.sqrt(len(x))

    return {
        "mean_ms": mean,
        "std_ms": std,
        "ci95_low_ms": mean - half,
        "ci95_high_ms": mean + half,
    }


def run_gmc(x, k, gy):
    gmc.reset_timing_collectors()

    y = gmc.DepthwiseConvCoalesced.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
        retain_graph=False,
        create_graph=False,
    )

    torch.cuda.synchronize()

    result = (
        float(gmc.FWD_TIMES[-1]),
        float(gmc.BWD_IN_TIMES[-1]),
        float(gmc.BWD_K_TIMES[-1]),
    )

    del y, gx, gk

    return result


def run_shared(x, k, gy):
    shared.reset_timing_collectors()

    y = shared.DepthwiseConvShared.apply(x, k)

    gx, gk = torch.autograd.grad(
        y,
        (x, k),
        grad_outputs=gy,
        retain_graph=False,
        create_graph=False,
    )

    torch.cuda.synchronize()

    result = (
        float(shared.FWD_TIMES[-1]),
        float(shared.BWD_IN_TIMES[-1]),
        float(shared.BWD_K_TIMES[-1]),
    )

    del y, gx, gk

    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)

    gpu = torch.cuda.get_device_name(0)

    if "A6000" not in gpu:
        raise RuntimeError(
            f"Must run on RTX A6000; current GPU: {gpu}"
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

    print("=" * 84)
    print("GMC vs SHARED — FULL L192 BENCHMARK")
    print("=" * 84)
    print("GPU:", gpu)
    print("PyTorch:", torch.__version__)
    print(f"B={B}, H={H}, L={L}, K={K}")
    print("dtype=FP32")
    print(f"Warm-up={WARMUP}")
    print(f"Measured runs={RUNS}")
    print()
    print("GMC:")
    print(f"  TPB={gmc.THREADS_PER_BLOCK}")
    print(f"  TTILE={gmc.TTILE}")
    print(f"  HTILE={gmc.HTILE}")
    print()
    print("Shared:")
    print(f"  TPB={shared.THREADS_PER_BLOCK}")
    print(f"  TTILE={shared.TTILE}")
    print(f"  HTILE={shared.HTILE}")
    print(f"  BWD_K_B_TILE={shared.BWD_K_B_TILE}")
    print(
        "  FWD/BWD-IN shared bytes/block="
        f"{shared.shared_memory_bytes(K)}"
    )
    print(
        "  BWD-K shared bytes/block="
        f"{shared.bwd_kernel_shared_memory_bytes(L)}"
    )
    print("=" * 84)

    # ========================================================
    # Warm-up both implementations
    # ========================================================

    print("\nWarm-up...")

    for i in range(WARMUP):
        run_gmc(x, k, gy)
        run_shared(x, k, gy)

        print(f"  warm-up {i+1}/{WARMUP}")

    # ========================================================
    # Measured paired/interleaved runs
    # ========================================================

    rows = []

    print("\nMeasured runs...")

    for i in range(RUNS):

        # Alternate order to reduce systematic clock/thermal bias.
        if i % 2 == 0:
            g = run_gmc(x, k, gy)
            s = run_shared(x, k, gy)
        else:
            s = run_shared(x, k, gy)
            g = run_gmc(x, k, gy)

        row = {
            "run": i + 1,

            "gmc_fwd_ms": g[0],
            "gmc_bwd_input_ms": g[1],
            "gmc_bwd_kernel_ms": g[2],

            "shared_fwd_ms": s[0],
            "shared_bwd_input_ms": s[1],
            "shared_bwd_kernel_ms": s[2],
        }

        rows.append(row)

        print(
            f"  run {i+1:2d}/{RUNS} | "
            f"GMC: "
            f"{g[0]:.3f} / {g[1]:.3f} / {g[2]:.3f} ms | "
            f"Shared: "
            f"{s[0]:.3f} / {s[1]:.3f} / {s[2]:.3f} ms"
        )

    # ========================================================
    # Statistics
    # ========================================================

    gmc_fwd = [r["gmc_fwd_ms"] for r in rows]
    gmc_bi = [r["gmc_bwd_input_ms"] for r in rows]
    gmc_bk = [r["gmc_bwd_kernel_ms"] for r in rows]

    sh_fwd = [r["shared_fwd_ms"] for r in rows]
    sh_bi = [r["shared_bwd_input_ms"] for r in rows]
    sh_bk = [r["shared_bwd_kernel_ms"] for r in rows]

    summary = {
        "gmc_forward": stats(gmc_fwd),
        "gmc_backward_input": stats(gmc_bi),
        "gmc_backward_kernel": stats(gmc_bk),

        "shared_forward": stats(sh_fwd),
        "shared_backward_input": stats(sh_bi),
        "shared_backward_kernel": stats(sh_bk),
    }

    gmcf = summary["gmc_forward"]["mean_ms"]
    gmci = summary["gmc_backward_input"]["mean_ms"]
    gmck = summary["gmc_backward_kernel"]["mean_ms"]

    shf = summary["shared_forward"]["mean_ms"]
    shi = summary["shared_backward_input"]["mean_ms"]
    shk = summary["shared_backward_kernel"]["mean_ms"]

    gmc_total = gmcf + gmci + gmck
    shared_total = shf + shi + shk

    print("\n" + "=" * 84)
    print("RESULTS")
    print("=" * 84)

    print("\nForward")
    print(f"  GMC:    {gmcf:.6f} ms")
    print(f"  Shared: {shf:.6f} ms")
    print(f"  Speedup Shared/GMC: {gmcf/shf:.4f}x")

    print("\nBackward Input")
    print(f"  GMC:    {gmci:.6f} ms")
    print(f"  Shared: {shi:.6f} ms")
    print(f"  Speedup Shared/GMC: {gmci/shi:.4f}x")

    print("\nBackward Kernel")
    print(f"  GMC:    {gmck:.6f} ms")
    print(f"  Shared: {shk:.6f} ms")
    print(f"  Speedup Shared/GMC: {gmck/shk:.4f}x")

    print("\nThree-kernel total")
    print(f"  GMC:    {gmc_total:.6f} ms")
    print(f"  Shared: {shared_total:.6f} ms")
    print(
        f"  Overall Shared speedup: "
        f"{gmc_total/shared_total:.4f}x"
    )

    print(
        f"  Total runtime reduction: "
        f"{100.0 * (1.0 - shared_total/gmc_total):.2f}%"
    )

    # ========================================================
    # Save raw data
    # ========================================================

    raw_path = OUT_DIR / "gmc_vs_shared_l192_raw.csv"

    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(rows)

    # ========================================================
    # Save metadata / summary
    # ========================================================

    payload = {
        "gpu": gpu,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "seed": SEED,
        "B": B,
        "H": H,
        "L": L,
        "K": K,
        "dtype": "float32",
        "warmup": WARMUP,
        "runs": RUNS,
        "shared_bwd_k_b_tile": shared.BWD_K_B_TILE,
        "shared_fwd_bwd_input_bytes":
            shared.shared_memory_bytes(K),
        "shared_bwd_kernel_bytes":
            shared.bwd_kernel_shared_memory_bytes(L),
        "summary": summary,
        "gmc_total_ms": gmc_total,
        "shared_total_ms": shared_total,
        "overall_speedup": gmc_total / shared_total,
        "runtime_reduction_pct":
            100.0 * (1.0 - shared_total / gmc_total),
    }

    json_path = OUT_DIR / "gmc_vs_shared_l192_summary.json"

    json_path.write_text(
        json.dumps(payload, indent=2)
    )

    print("\nSaved:")
    print(raw_path)
    print(json_path)


if __name__ == "__main__":
    main()
