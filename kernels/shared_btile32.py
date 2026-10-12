from __future__ import annotations

import numpy as np
import torch

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule


# ============================================================
# CUDA / PyCUDA setup
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU is required. Run inside a SLURM GPU node."
    )

torch.cuda.set_device(0)
torch.cuda.init()

cuda.init()

_DEVICE_INDEX = torch.cuda.current_device()
_PYCUDA_DEVICE = cuda.Device(_DEVICE_INDEX)
_PRIMARY_CONTEXT = _PYCUDA_DEVICE.retain_primary_context()


class CudaContextGuard:
    def __enter__(self):
        _PRIMARY_CONTEXT.push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _PRIMARY_CONTEXT.pop()
        return False


# ============================================================
# Torch <-> PyCUDA zero-copy
# ============================================================

class TorchCudaInterop:
    class TorchPointer(cuda.PointerHolderBase):
        def __init__(self, ptr: int):
            super().__init__()
            self.gpudata = int(ptr)

        def __int__(self):
            return self.gpudata

    @staticmethod
    def as_gpuarray(tensor: torch.Tensor) -> gpuarray.GPUArray:
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA")

        if tensor.dtype != torch.float32:
            raise TypeError(
                f"Expected torch.float32, got {tensor.dtype}"
            )

        if not tensor.is_contiguous():
            raise ValueError(
                "Tensor must be contiguous"
            )

        holder = TorchCudaInterop.TorchPointer(
            int(tensor.data_ptr())
        )

        strides = tuple(
            stride * tensor.element_size()
            for stride in tensor.stride()
        )

        arr = gpuarray.GPUArray(
            shape=tuple(tensor.shape),
            dtype=np.float32,
            gpudata=holder,
            strides=strides,
        )

        arr._torch_tensor_ref = tensor
        arr._torch_pointer_ref = holder

        return arr


# ============================================================
# Experimental configuration
#
# EXACTLY inherited from GMC:
#
#   TTILE = 32
#   HTILE = 16
#   TPB   = 512
#
# New optimization:
#
#   Shared-memory input tile + halo
#   Shared-memory kernel coefficients
#
# NO warp tiling
# NO warp shuffle
# NO partial reduction
# ============================================================

TTILE = 32
HTILE = 16

THREADS_PER_BLOCK = TTILE * HTILE  # 512

# Backward-weight batch tile.
# x and gy for BWD_K_B_TILE samples are cooperatively cached
# in shared memory while preserving the original b -> t
# accumulation order.
BWD_K_B_TILE = 32


# ============================================================
# CUDA
# ============================================================

KERNEL_CODE = r"""
#define TTILE 32
#define HTILE 16


extern "C" {


/*
==============================================================
FORWARD -- WARP-SYNCHRONOUS SHARED MEMORY
==============================================================

GMC thread mapping is preserved:

    one warp -> one channel
    one lane -> one temporal output

Only the overlapping input tile is cached in shared memory.

Kernel coefficients remain on the normal read-only/cache path.

Each warp owns a disjoint shared-memory region, so block-wide
__syncthreads() is unnecessary. __syncwarp() is sufficient.
*/

__global__ void dwconv1d_fwd_shared(
    const float* __restrict__ x,
    const float* __restrict__ k,
    float* __restrict__ y,
    int B,
    int H,
    int L,
    int K)
{
    const int tid =
        (int)threadIdx.x;

    const int lane =
        tid % TTILE;

    const int warp_local =
        tid / TTILE;

    const int b =
        (int)blockIdx.z;

    const int h =
        (int)blockIdx.y * HTILE
        + warp_local;

    const int t0 =
        (int)blockIdx.x * TTILE;

    const int t =
        t0 + lane;

    const bool valid_h =
        (b < B && h < H);

    const bool valid_output =
        valid_h && (t < L);

    const int half =
        K / 2;

    const int tile_capacity =
        TTILE + K - 1;

    /*
        Last temporal tile may contain fewer than 32 outputs.

        For L=48:
            tile 0 -> 32 outputs
            tile 1 -> 16 outputs

        Therefore we do not unnecessarily initialize all
        79 entries for the final tile.
    */

    const int out_count =
        min(TTILE, L - t0);

    const int needed_len =
        out_count + K - 1;

    extern __shared__ float smem[];

    float* tile =
        smem
        + warp_local * tile_capacity;

    const int in_start =
        t0 - half;

    const int base =
        valid_h
        ? (b * H + h) * L
        : 0;

    /*
        Warp-cooperative input + halo load.
    */

    for (
        int i = lane;
        i < needed_len;
        i += TTILE
    )
    {
        const int gi =
            in_start + i;

        float value =
            0.0f;

        if (
            valid_h
            && (unsigned)gi < (unsigned)L
        )
        {
            value =
                x[base + gi];
        }

        tile[i] =
            value;
    }

    /*
        No cross-warp data dependency exists.
    */

    __syncwarp();

    if (valid_output)
    {
        float acc =
            0.0f;

        for (
            int j = 0;
            j < K;
            ++j
        )
        {
            acc = fmaf(
                tile[lane + j],
                k[h * K + j],
                acc
            );
        }

        y[base + t] =
            acc;
    }
}


/*
==============================================================
BACKWARD INPUT -- WARP-SYNCHRONOUS SHARED MEMORY
==============================================================

GMC mapping is preserved.

Only grad_y is cached in Shared Memory.
Kernel coefficients stay on the normal cache path.

Each warp owns its own tile, therefore __syncwarp() is enough.
*/

__global__ void dwconv1d_bwd_input_shared(
    const float* __restrict__ gy,
    const float* __restrict__ k,
    float* __restrict__ gx,
    int B,
    int H,
    int L,
    int K)
{
    const int tid =
        (int)threadIdx.x;

    const int lane =
        tid % TTILE;

    const int warp_local =
        tid / TTILE;

    const int b =
        (int)blockIdx.z;

    const int h =
        (int)blockIdx.y * HTILE
        + warp_local;

    const int t0 =
        (int)blockIdx.x * TTILE;

    const int t =
        t0 + lane;

    const bool valid_h =
        (b < B && h < H);

    const bool valid_output =
        valid_h && (t < L);

    const int half =
        K / 2;

    const int left_halo =
        K - 1 - half;

    const int tile_capacity =
        TTILE + K - 1;

    const int out_count =
        min(TTILE, L - t0);

    const int needed_len =
        out_count + K - 1;

    extern __shared__ float smem[];

    float* tile =
        smem
        + warp_local * tile_capacity;

    const int in_start =
        t0 - left_halo;

    const int base =
        valid_h
        ? (b * H + h) * L
        : 0;

    /*
        Warp-cooperative grad_y + halo load.
    */

    for (
        int i = lane;
        i < needed_len;
        i += TTILE
    )
    {
        const int gi =
            in_start + i;

        float value =
            0.0f;

        if (
            valid_h
            && (unsigned)gi < (unsigned)L
        )
        {
            value =
                gy[base + gi];
        }

        tile[i] =
            value;
    }

    __syncwarp();

    if (valid_output)
    {
        float acc =
            0.0f;

        for (
            int j = 0;
            j < K;
            ++j
        )
        {
            const int local_idx =
                lane + K - 1 - j;

            acc = fmaf(
                tile[local_idx],
                k[h * K + j],
                acc
            );
        }

        gx[base + t] =
            acc;
    }
}


/*
==============================================================
BACKWARD KERNEL -- SHARED-MEMORY BATCH TILING
==============================================================

The computational mapping remains intentionally close to GMC:

    one block  -> one channel h
    one thread -> one kernel tap j

The accumulation order for every gk[h,j] remains:

    b = 0 ... B-1
        t = 0 ... L-1

The only new optimization is cooperative caching of x and gy
for a small batch tile in shared memory.

No:
    partial buffer
    atomics
    warp shuffle
    reduction redesign
    extra output decomposition
*/

__global__ void dwconv1d_bwd_kernel_shared(
    const float* __restrict__ gy,
    const float* __restrict__ x,
    float* __restrict__ gk,
    int B,
    int H,
    int L,
    int K)
{
    const int h =
        (int)blockIdx.y;

    const int tid =
        (int)threadIdx.x;

    const int j =
        (int)blockIdx.x * (int)blockDim.x
        + tid;

    /*
        h is identical for every thread in the block,
        therefore this block-wide return is safe before
        any __syncthreads().
    */
    if (h >= H)
        return;

    const bool valid_j =
        (j < K);

    const int half =
        K / 2;

    const int B_TILE =
        32;

    /*
        Dynamic shared-memory layout:

            x_s  [B_TILE, L]
            gy_s [B_TILE, L]

        For L=48 and B_TILE=64:

            2 * 64 * 48 * 4
            = 24,576 bytes
            = 24 KiB
    */

    extern __shared__ float smem[];

    float* x_s =
        smem;

    float* gy_s =
        smem + B_TILE * L;

    float acc =
        0.0f;

    /*
        Process batches in increasing order exactly as in the
        frozen GMC implementation.
    */

    for (
        int b0 = 0;
        b0 < B;
        b0 += B_TILE
    )
    {
        const int nb =
            min(B_TILE, B - b0);

        const int tile_elems =
            nb * L;

        /*
            All 512 threads cooperate in loading the current
            x/gy batch tile.

            Even threads with j >= K participate here.
        */

        for (
            int idx = tid;
            idx < tile_elems;
            idx += (int)blockDim.x
        )
        {
            const int b_local =
                idx / L;

            const int t_local =
                idx - b_local * L;

            const int b =
                b0 + b_local;

            const int base =
                (b * H + h) * L;

            x_s[idx] =
                x[base + t_local];

            gy_s[idx] =
                gy[base + t_local];
        }

        __syncthreads();

        /*
            Preserve the original accumulation order:

                b increasing
                then t increasing

            Only the storage hierarchy has changed:
            values now come from shared memory.
        */

        if (valid_j)
        {
            for (
                int b_local = 0;
                b_local < nb;
                ++b_local
            )
            {
                const int shared_base =
                    b_local * L;

                for (
                    int t = 0;
                    t < L;
                    ++t
                )
                {
                    const int xi =
                        t + j - half;

                    if (
                        (unsigned)xi
                        < (unsigned)L
                    )
                    {
                        acc = fmaf(
                            gy_s[shared_base + t],
                            x_s[shared_base + xi],
                            acc
                        );
                    }
                }
            }
        }

        /*
            Required before the next batch tile overwrites smem.
        */

        __syncthreads();
    }

    if (valid_j)
    {
        gk[h * K + j] =
            acc;
    }
}


} // extern "C"
"""


# ============================================================
# Compile
# ============================================================

_MODULE = None
_FWD_KERNEL = None
_BWD_INPUT_KERNEL = None
_BWD_KERNEL_KERNEL = None


def _compile_module():
    major, minor = torch.cuda.get_device_capability(
        torch.cuda.current_device()
    )

    arch_flag = f"-arch=sm_{major}{minor}"

    with CudaContextGuard():
        module = SourceModule(
            KERNEL_CODE,
            options=[
                "-O3",
                arch_flag,
                "-Xptxas=-v",
                "-ftz=true",
                "-prec-div=false",
                "-prec-sqrt=false",
            ],
            keep=False,
            no_extern_c=True,
        )

    return module


def _get_kernels():
    global _MODULE
    global _FWD_KERNEL
    global _BWD_INPUT_KERNEL
    global _BWD_KERNEL_KERNEL

    if _MODULE is None:
        _MODULE = _compile_module()

        with CudaContextGuard():
            _FWD_KERNEL = _MODULE.get_function(
                "dwconv1d_fwd_shared"
            )

            _BWD_INPUT_KERNEL = _MODULE.get_function(
                "dwconv1d_bwd_input_shared"
            )

            _BWD_KERNEL_KERNEL = _MODULE.get_function(
                "dwconv1d_bwd_kernel_shared"
            )

    return (
        _FWD_KERNEL,
        _BWD_INPUT_KERNEL,
        _BWD_KERNEL_KERNEL,
    )


# ============================================================
# Shared memory size
# ============================================================

def bwd_kernel_shared_memory_bytes(L: int) -> int:
    """
    Backward-weight shared memory:

        x  : [BWD_K_B_TILE, L]
        gy : [BWD_K_B_TILE, L]

    FP32 = 4 bytes.
    """

    return (
        2
        * BWD_K_B_TILE
        * L
        * 4
    )


def shared_memory_bytes(K: int) -> int:
    """
    Forward / backward-input shared memory.

    Only the temporal input/gradient tile is cached.

        HTILE * (TTILE + K - 1)

    FP32 = 4 bytes.
    """

    tile_len = (
        TTILE
        + K
        - 1
    )

    return (
        HTILE
        * tile_len
        * 4
    )


# ============================================================
# Timing collectors
# ============================================================

FWD_TIMES: list[float] = []
BWD_IN_TIMES: list[float] = []
BWD_K_TIMES: list[float] = []


def reset_timing_collectors():
    FWD_TIMES.clear()
    BWD_IN_TIMES.clear()
    BWD_K_TIMES.clear()


def _time_kernel(
    kernel,
    args,
    block,
    grid,
    shared: int = 0,
) -> float:

    torch.cuda.synchronize()

    with CudaContextGuard():
        start = cuda.Event()
        end = cuda.Event()

        start.record()

        kernel(
            *args,
            block=block,
            grid=grid,
            shared=shared,
        )

        end.record()
        end.synchronize()

        elapsed_ms = float(
            start.time_till(end)
        )

        del start
        del end

    return elapsed_ms


# ============================================================
# Autograd
# ============================================================

class DepthwiseConvShared(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:

        if not x.is_cuda or not kernel.is_cuda:
            raise ValueError(
                "x and kernel must be CUDA tensors"
            )

        if x.dtype != torch.float32:
            raise TypeError(
                "x must be float32"
            )

        if kernel.dtype != torch.float32:
            raise TypeError(
                "kernel must be float32"
            )

        if x.ndim != 3:
            raise ValueError(
                f"x must be [B,H,L], got {tuple(x.shape)}"
            )

        if kernel.ndim != 2:
            raise ValueError(
                f"kernel must be [H,K], got {tuple(kernel.shape)}"
            )

        x = x.contiguous()
        kernel = kernel.contiguous()

        B, H, L = x.shape
        Hk, K = kernel.shape

        if H != Hk:
            raise ValueError(
                f"Channel mismatch: x H={H}, kernel H={Hk}"
            )

        ctx.save_for_backward(
            x,
            kernel,
        )

        ctx.params = (
            B,
            H,
            L,
            K,
        )

        y = torch.empty_like(x)

        with CudaContextGuard():
            x_g = TorchCudaInterop.as_gpuarray(x)
            k_g = TorchCudaInterop.as_gpuarray(kernel)
            y_g = TorchCudaInterop.as_gpuarray(y)

        fwd, _, _ = _get_kernels()

        block = (
            THREADS_PER_BLOCK,
            1,
            1,
        )

        grid = (
            (L + TTILE - 1) // TTILE,
            (H + HTILE - 1) // HTILE,
            B,
        )

        smem_bytes = shared_memory_bytes(K)

        elapsed_ms = _time_kernel(
            fwd,
            (
                x_g.gpudata,
                k_g.gpudata,
                y_g.gpudata,
                np.int32(B),
                np.int32(H),
                np.int32(L),
                np.int32(K),
            ),
            block,
            grid,
            shared=smem_bytes,
        )

        FWD_TIMES.append(
            elapsed_ms
        )

        return y

    @staticmethod
    def backward(
        ctx,
        grad_y: torch.Tensor,
    ):

        x, kernel = ctx.saved_tensors

        B, H, L, K = ctx.params

        grad_y = grad_y.contiguous()

        gx = torch.empty_like(x)
        gk = torch.empty_like(kernel)

        with CudaContextGuard():
            gy_g = TorchCudaInterop.as_gpuarray(
                grad_y
            )

            x_g = TorchCudaInterop.as_gpuarray(
                x
            )

            k_g = TorchCudaInterop.as_gpuarray(
                kernel
            )

            gx_g = TorchCudaInterop.as_gpuarray(
                gx
            )

            gk_g = TorchCudaInterop.as_gpuarray(
                gk
            )

        _, bwd_input, bwd_kernel = _get_kernels()

        # ====================================================
        # Backward Input -- Shared Memory
        # ====================================================

        block_input = (
            THREADS_PER_BLOCK,
            1,
            1,
        )

        grid_input = (
            (L + TTILE - 1) // TTILE,
            (H + HTILE - 1) // HTILE,
            B,
        )

        smem_bytes = shared_memory_bytes(K)

        bwd_input_ms = _time_kernel(
            bwd_input,
            (
                gy_g.gpudata,
                k_g.gpudata,
                gx_g.gpudata,
                np.int32(B),
                np.int32(H),
                np.int32(L),
                np.int32(K),
            ),
            block_input,
            grid_input,
            shared=smem_bytes,
        )

        BWD_IN_TIMES.append(
            bwd_input_ms
        )

        # ====================================================
        # Backward Kernel -- unchanged control path
        # ====================================================

        block_kernel = (
            THREADS_PER_BLOCK,
            1,
            1,
        )

        grid_kernel = (
            (
                K
                + THREADS_PER_BLOCK
                - 1
            )
            // THREADS_PER_BLOCK,
            H,
            1,
        )

        bwd_kernel_ms = _time_kernel(
            bwd_kernel,
            (
                gy_g.gpudata,
                x_g.gpudata,
                gk_g.gpudata,
                np.int32(B),
                np.int32(H),
                np.int32(L),
                np.int32(K),
            ),
            block_kernel,
            grid_kernel,
            shared=bwd_kernel_shared_memory_bytes(L),
        )

        BWD_K_TIMES.append(
            bwd_kernel_ms
        )

        return gx, gk


def depthwise_conv_shared(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:

    return DepthwiseConvShared.apply(
        x,
        kernel,
    )


__all__ = [
    "DepthwiseConvShared",
    "depthwise_conv_shared",
    "THREADS_PER_BLOCK",
    "TTILE",
    "HTILE",
    "shared_memory_bytes",
    "FWD_TIMES",
    "BWD_IN_TIMES",
    "BWD_K_TIMES",
    "reset_timing_collectors",
]
