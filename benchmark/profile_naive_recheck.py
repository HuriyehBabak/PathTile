from __future__ import annotations

import argparse

import numpy as np
import torch

import kernels.naive as n


SEED = 42

B = 16384
H = 128
L = 48
K = 48

TPB = 512
WARMUP = 5


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "forward",
            "bwd-input",
            "bwd-kernel",
        ],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)

    if "A6000" not in gpu_name:
        raise RuntimeError(
            "Naive Nsight recheck must run on RTX A6000. "
            f"Current GPU: {gpu_name}"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    n.DepthwiseConvNaive.THREADS_PER_BLOCK = TPB

    print("=" * 72)
    print("NAIVE NSIGHT RECHECK")
    print("=" * 72)
    print(f"GPU:   {gpu_name}")
    print(f"Phase: {args.phase}")
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(f"TPB={TPB}")
    print(f"Warm-up launches: {WARMUP}")
    print("Target launch:     1")
    print("=" * 72)

    # ========================================================
    # Fixed workload
    # ========================================================

    x = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    k = torch.randn(
        H,
        K,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    gy = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    y = torch.empty_like(x)
    gx = torch.empty_like(x)
    gk = torch.empty_like(k)

    torch.cuda.synchronize()

    # ========================================================
    # Get the already-defined Naive CUDA kernels
    # ========================================================

    fwd_kernel, bwd_input_kernel, bwd_kernel_kernel = (
        n.get_kernels()
    )

    # ========================================================
    # Torch -> PyCUDA zero-copy views
    #
    # kernels.naive already uses the active CUDA context
    # directly, so NO CudaContextGuard is required.
    # ========================================================

    x_g = n.TorchCudaInterop.as_gpuarray(x)
    k_g = n.TorchCudaInterop.as_gpuarray(k)
    gy_g = n.TorchCudaInterop.as_gpuarray(gy)

    y_g = n.TorchCudaInterop.as_gpuarray(y)
    gx_g = n.TorchCudaInterop.as_gpuarray(gx)
    gk_g = n.TorchCudaInterop.as_gpuarray(gk)

    # ========================================================
    # Launch configurations
    #
    # These reproduce the frozen Naive mapping.
    # ========================================================

    block = (
        TPB,
        1,
        1,
    )

    grid_fwd = (
        ceil_div(L, TPB),
        H,
        B,
    )

    grid_bwd_input = (
        ceil_div(L, TPB),
        H,
        B,
    )

    grid_bwd_kernel = (
        ceil_div(K, TPB),
        H,
        1,
    )

    # ========================================================
    # Select exactly one kernel
    # ========================================================

    if args.phase == "forward":

        kernel_fn = fwd_kernel

        kernel_args = (
            x_g.gpudata,
            k_g.gpudata,
            y_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        grid = grid_fwd

    elif args.phase == "bwd-input":

        kernel_fn = bwd_input_kernel

        kernel_args = (
            gy_g.gpudata,
            k_g.gpudata,
            gx_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        grid = grid_bwd_input

    else:

        kernel_fn = bwd_kernel_kernel

        kernel_args = (
            gy_g.gpudata,
            x_g.gpudata,
            gk_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        grid = grid_bwd_kernel

    # ========================================================
    # Warm-up
    #
    # Exactly five launches of the SAME kernel.
    # Nsight:
    #
    #   --launch-skip 5
    #   --launch-count 1
    #
    # therefore profiles only launch #6.
    # ========================================================

    print()
    print("Warm-up...")

    for i in range(WARMUP):

        kernel_fn(
            *kernel_args,
            block=block,
            grid=grid,
        )

        torch.cuda.synchronize()

        print(
            f"  warm-up {i + 1}/{WARMUP}"
        )

    # ========================================================
    # Target launch
    # ========================================================

    print()
    print("Target launch...")

    kernel_fn(
        *kernel_args,
        block=block,
        grid=grid,
    )

    torch.cuda.synchronize()

    print("Target launch complete.")


if __name__ == "__main__":
    main()
