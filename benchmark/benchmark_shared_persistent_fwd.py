from __future__ import annotations

import math
import numpy as np
import torch

import pycuda.driver as cuda
from pycuda.compiler import SourceModule

import kernels.shared as s


SEED = 42

B = 16384
H = 128
L = 48
K = 48

TPB = 256
GRID_Y = 1792

WARMUP = 5
RUNS = 30


CUDA_CODE = r"""
extern "C" {

__global__ void dwconv1d_fwd_shared_persistent(
    const float* __restrict__ x,
    const float* __restrict__ kHK,
    float* __restrict__ y,
    const int L,
    const int K,
    const int H,
    const int num_signals)
{
    const int t =
        blockIdx.x * blockDim.x + threadIdx.x;

    const bool valid_t =
        (t < L);

    const int pad =
        K >> 1;

    const int tile_len =
        blockDim.x + K - 1;

    const int in_start =
        blockIdx.x * blockDim.x - pad;

    extern __shared__ float smem[];

    float* tile =
        smem;

    float* k_s =
        smem + tile_len;

    /*
        Old persistent/grid-stride mapping.
    */
    for (
        int s = blockIdx.y;
        s < num_signals;
        s += gridDim.y
    )
    {
        const int h =
            s % H;

        const float* xb =
            x + (size_t)s * L;

        const float* kb =
            kHK + (size_t)h * K;

        for (
            int i = threadIdx.x;
            i < tile_len;
            i += blockDim.x
        )
        {
            const int gidx =
                in_start + i;

            tile[i] =
                (
                    gidx >= 0
                    && gidx < L
                )
                ? __ldg(&xb[gidx])
                : 0.0f;
        }

        for (
            int j = threadIdx.x;
            j < K;
            j += blockDim.x
        )
        {
            k_s[j] =
                kb[j];
        }

        __syncthreads();

        if (valid_t)
        {
            float acc =
                0.0f;

            #pragma unroll 4
            for (
                int j = 0;
                j < K;
                ++j
            )
            {
                acc = fmaf(
                    tile[threadIdx.x + j],
                    k_s[j],
                    acc
                );
            }

            y[(size_t)s * L + t] =
                acc;
        }

        __syncthreads();
    }
}

}
"""


def stats(values):
    a = np.asarray(values, dtype=np.float64)

    mean = float(a.mean())
    std = float(a.std(ddof=1))

    tcrit = 2.045229642
    half = tcrit * std / math.sqrt(len(a))

    return mean, std, mean - half, mean + half


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.cuda.set_device(0)

    gpu = torch.cuda.get_device_name(0)

    if "A6000" not in gpu:
        raise RuntimeError(
            f"Must run on RTX A6000; got {gpu}"
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    major, minor = torch.cuda.get_device_capability(0)

    with s.CudaContextGuard():
        mod = SourceModule(
            CUDA_CODE,
            options=[
                "-O3",
                f"-arch=sm_{major}{minor}",
                "-Xptxas=-v",
                "-ftz=true",
                "-prec-div=false",
                "-prec-sqrt=false",
            ],
            no_extern_c=True,
        )

        kernel = mod.get_function(
            "dwconv1d_fwd_shared_persistent"
        )

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

    y = torch.empty_like(x)

    with s.CudaContextGuard():
        x_g = s.TorchCudaInterop.as_gpuarray(x)
        k_g = s.TorchCudaInterop.as_gpuarray(k)
        y_g = s.TorchCudaInterop.as_gpuarray(y)

    num_signals = B * H

    block = (
        TPB,
        1,
        1,
    )

    grid = (
        (L + TPB - 1) // TPB,
        GRID_Y,
        1,
    )

    tile_len = TPB + K - 1

    shared_bytes = (
        tile_len + K
    ) * 4

    args = (
        x_g.gpudata,
        k_g.gpudata,
        y_g.gpudata,
        np.int32(L),
        np.int32(K),
        np.int32(H),
        np.int32(num_signals),
    )

    print("=" * 72)
    print("OLD SHARED PERSISTENT-MAPPING FORWARD")
    print("=" * 72)
    print("GPU:", gpu)
    print(f"B={B}, H={H}, L={L}, K={K}")
    print("TPB =", TPB)
    print("grid =", grid)
    print("num_signals =", num_signals)
    print("shared bytes/block =", shared_bytes)
    print("warm-up =", WARMUP)
    print("runs =", RUNS)
    print("=" * 72)

    # --------------------------------------------------------
    # Correctness against Shared-v2
    # --------------------------------------------------------

    y_ref = s.DepthwiseConvShared.apply(
        x,
        k,
    )

    with s.CudaContextGuard():
        kernel(
            *args,
            block=block,
            grid=grid,
            shared=shared_bytes,
        )

    torch.cuda.synchronize()

    max_diff = (
        y - y_ref
    ).abs().max().item()

    exact = torch.equal(
        y,
        y_ref,
    )

    print("\nCorrectness:")
    print(
        f"max abs diff = {max_diff:.9e}"
    )
    print(
        "exact =",
        exact,
    )

    if not torch.allclose(
        y,
        y_ref,
        atol=1e-3,
        rtol=1e-3,
    ):
        raise RuntimeError(
            "Persistent forward correctness FAILED"
        )

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print("\nWarm-up...")

    for i in range(WARMUP):
        s._time_kernel(
            kernel,
            args,
            block,
            grid,
            shared=shared_bytes,
        )

        print(
            f"  warm-up {i+1}/{WARMUP}"
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    values = []

    print("\nMeasured runs...")

    for i in range(RUNS):

        t = s._time_kernel(
            kernel,
            args,
            block,
            grid,
            shared=shared_bytes,
        )

        values.append(t)

        print(
            f"  run {i+1:2d}/{RUNS}: "
            f"{t:.6f} ms"
        )

    mean, std, low, high = stats(values)

    print("\n" + "=" * 72)
    print("RESULT")
    print("=" * 72)

    print(f"Mean = {mean:.6f} ms")
    print(f"Std  = {std:.6f} ms")
    print(
        f"95% CI = [{low:.6f}, {high:.6f}]"
    )

    print("\nReference:")
    print("GMC Forward       = 5.265164 ms")
    print("Shared-v2 Forward = 5.798159 ms")


if __name__ == "__main__":
    main()
