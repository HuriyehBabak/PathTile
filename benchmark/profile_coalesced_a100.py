from __future__ import annotations

import argparse

import numpy as np
import torch

import kernels.coalesced as c


SEED = 42

B = 16384
H = 128
L = 48
K = 48

WARMUP = 5


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "fwd",
            "bwd_input",
            "bwd_kernel",
        ],
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)

    gpu_name = torch.cuda.get_device_name(0)

    if "A100" not in gpu_name:
        raise RuntimeError(
            "Nsight measurements must use NVIDIA A100. "
            f"Current GPU: {gpu_name}"
        )

    if c.THREADS_PER_BLOCK != 512:
        raise RuntimeError(
            f"Expected TPB=512, got {c.THREADS_PER_BLOCK}"
        )

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print("GMC NSIGHT PROFILE")
    print("=" * 72)
    print(f"GPU:   {gpu_name}")
    print(f"Phase: {args.phase}")
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(
        f"TPB={c.THREADS_PER_BLOCK}, "
        f"TTILE={c.TTILE}, HTILE={c.HTILE}"
    )
    print(f"Warm-up launches: {WARMUP}")
    print("Target launch:     1")
    print("=" * 72)

    # --------------------------------------------------------
    # Allocate fixed workload
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Compile/get kernels once
    # --------------------------------------------------------

    fwd_kernel, bwd_input_kernel, bwd_kernel_kernel = (
        c._get_kernels()
    )

    # --------------------------------------------------------
    # Zero-copy Torch -> PyCUDA views
    # --------------------------------------------------------

    with c.CudaContextGuard():
        x_g = c.TorchCudaInterop.as_gpuarray(x)
        k_g = c.TorchCudaInterop.as_gpuarray(k)
        gy_g = c.TorchCudaInterop.as_gpuarray(gy)

        y_g = c.TorchCudaInterop.as_gpuarray(y)
        gx_g = c.TorchCudaInterop.as_gpuarray(gx)
        gk_g = c.TorchCudaInterop.as_gpuarray(gk)

    # --------------------------------------------------------
    # Launch configuration
    # --------------------------------------------------------

    block_coalesced = (
        c.THREADS_PER_BLOCK,
        1,
        1,
    )

    grid_coalesced = (
        ceil_div(L, c.TTILE),
        ceil_div(H, c.HTILE),
        B,
    )

    block_kernel = (
        c.THREADS_PER_BLOCK,
        1,
        1,
    )

    grid_kernel = (
        ceil_div(K, c.THREADS_PER_BLOCK),
        H,
        1,
    )

    # --------------------------------------------------------
    # Select ONE kernel type
    # --------------------------------------------------------

    if args.phase == "fwd":

        kernel = fwd_kernel

        kernel_args = (
            x_g.gpudata,
            k_g.gpudata,
            y_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        block = block_coalesced
        grid = grid_coalesced

    elif args.phase == "bwd_input":

        kernel = bwd_input_kernel

        kernel_args = (
            gy_g.gpudata,
            k_g.gpudata,
            gx_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        block = block_coalesced
        grid = grid_coalesced

    else:

        kernel = bwd_kernel_kernel

        kernel_args = (
            gy_g.gpudata,
            x_g.gpudata,
            gk_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        block = block_kernel
        grid = grid_kernel

    # --------------------------------------------------------
    # Warm-up:
    #
    # Exactly five launches of the SAME target kernel.
    # Nsight will later use:
    #
    #   --launch-skip 5
    #   --launch-count 1
    #
    # to profile only the sixth launch.
    # --------------------------------------------------------

    print()
    print("Warm-up...")

    for i in range(WARMUP):
        with c.CudaContextGuard():
            kernel(
                *kernel_args,
                block=block,
                grid=grid,
            )

            c.cuda.Context.synchronize()

        print(f"  warm-up {i + 1}/{WARMUP}")

    # --------------------------------------------------------
    # Target launch
    # --------------------------------------------------------

    print()
    print("Target launch...")

    with c.CudaContextGuard():
        kernel(
            *kernel_args,
            block=block,
            grid=grid,
        )

        c.cuda.Context.synchronize()

    print("Target launch complete.")


if __name__ == "__main__":
    main()
