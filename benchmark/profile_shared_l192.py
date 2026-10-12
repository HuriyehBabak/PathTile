from __future__ import annotations

import argparse

import numpy as np
import torch

import kernels.shared_l192 as s


SEED = 42

B = 16384
H = 128
L = 192
K = 192

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

    gpu = torch.cuda.get_device_name(0)

    if "A6000" not in gpu:
        raise RuntimeError(
            f"Profiler must run on RTX A6000; current GPU: {gpu}"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    print("=" * 72)
    print("SHARED MEMORY NSIGHT PROFILER")
    print("=" * 72)
    print(f"GPU:   {gpu}")
    print(f"Phase: {args.phase}")
    print(f"B={B}, H={H}, L={L}, K={K}")
    print(f"TPB={TPB}")
    print(f"TTILE={s.TTILE}")
    print(f"HTILE={s.HTILE}")
    print(f"BWD_K_B_TILE={s.BWD_K_B_TILE}")
    print(
        "FWD/BWD-IN shared bytes/block:",
        s.shared_memory_bytes(K),
    )
    print(
        "BWD-K shared bytes/block:",
        s.bwd_kernel_shared_memory_bytes(L),
    )
    print(f"Warm-up launches: {WARMUP}")
    print("Target launch:     1")
    print("=" * 72)

    # ========================================================
    # Fixed tensors
    # ========================================================

    x = torch.randn(
        B, H, L,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    k = torch.randn(
        H, K,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    gy = torch.randn(
        B, H, L,
        device="cuda",
        dtype=torch.float32,
    ).contiguous()

    y = torch.empty_like(x)
    gx = torch.empty_like(x)
    gk = torch.empty_like(k)

    torch.cuda.synchronize()

    # ========================================================
    # Compile/get kernels
    # ========================================================

    fwd_kernel, bwd_input_kernel, bwd_kernel = s._get_kernels()

    # ========================================================
    # Torch -> PyCUDA zero-copy
    # ========================================================

    with s.CudaContextGuard():
        x_g = s.TorchCudaInterop.as_gpuarray(x)
        k_g = s.TorchCudaInterop.as_gpuarray(k)
        gy_g = s.TorchCudaInterop.as_gpuarray(gy)

        y_g = s.TorchCudaInterop.as_gpuarray(y)
        gx_g = s.TorchCudaInterop.as_gpuarray(gx)
        gk_g = s.TorchCudaInterop.as_gpuarray(gk)

    block = (
        TPB,
        1,
        1,
    )

    # ========================================================
    # Exact Shared-v2 launch configurations
    # ========================================================

    grid_temporal = (
        ceil_div(L, s.TTILE),
        ceil_div(H, s.HTILE),
        B,
    )

    grid_kernel = (
        ceil_div(K, TPB),
        H,
        1,
    )

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

        grid = grid_temporal
        shared_bytes = s.shared_memory_bytes(K)

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

        grid = grid_temporal
        shared_bytes = s.shared_memory_bytes(K)

    else:

        kernel_fn = bwd_kernel

        kernel_args = (
            gy_g.gpudata,
            x_g.gpudata,
            gk_g.gpudata,
            np.int32(B),
            np.int32(H),
            np.int32(L),
            np.int32(K),
        )

        grid = grid_kernel
        shared_bytes = s.bwd_kernel_shared_memory_bytes(L)

    # ========================================================
    # 5 identical warm-up launches
    # ========================================================

    print("\nWarm-up...")

    for i in range(WARMUP):

        with s.CudaContextGuard():
            kernel_fn(
                *kernel_args,
                block=block,
                grid=grid,
                shared=shared_bytes,
            )

        torch.cuda.synchronize()

        print(f"  warm-up {i + 1}/{WARMUP}")

    # ========================================================
    # One target launch
    # ========================================================

    print("\nTarget launch...")

    with s.CudaContextGuard():
        kernel_fn(
            *kernel_args,
            block=block,
            grid=grid,
            shared=shared_bytes,
        )

    torch.cuda.synchronize()

    print("Target launch complete.")


if __name__ == "__main__":
    main()
