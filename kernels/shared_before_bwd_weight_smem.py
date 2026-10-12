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


# ============================================================
# CUDA
# ============================================================

KERNEL_CODE = r"""
#define TTILE 32
#define HTILE 16


extern "C" {


/*
==============================================================
FORWARD -- SHARED MEMORY
==============================================================

Each warp handles:

    one channel h
    32 consecutive temporal outputs

For K taps, those 32 outputs need:

    TTILE + K - 1

input elements.

Instead of each output thread repeatedly loading the overlapping
input window from global memory, the warp cooperatively loads
that tile once into shared memory.

Shared layout:

    tiles[HTILE][TTILE + K - 1]
    kernels[HTILE][K]

No inter-warp data reuse is assumed.
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

    const int tile_len =
        TTILE + K - 1;

    extern __shared__ float smem[];

    float* tile_all =
        smem;

    float* kernel_all =
        smem + HTILE * tile_len;

    float* tile =
        tile_all + warp_local * tile_len;

    float* k_s =
        kernel_all + warp_local * K;

    /*
        Forward:

            xi = t + j - half

        For the first output in this warp tile:

            t = t0

        therefore input tile begins at:

            t0 - half
    */

    const int in_start =
        t0 - half;

    const int base =
        valid_h
        ? (b * H + h) * L
        : 0;

    /*
        Cooperative input + halo load.

        For K=48:

            tile_len = 32 + 48 - 1
                     = 79

        Each warp therefore performs only a few cooperative
        global-load rounds instead of every output thread
        independently loading K values.
    */

    for (
        int i = lane;
        i < tile_len;
        i += TTILE
    )
    {
        const int gi =
            in_start + i;

        float value = 0.0f;

        if (
            valid_h
            && gi >= 0
            && gi < L
        )
        {
            value =
                x[base + gi];
        }

        tile[i] = value;
    }

    /*
        Kernel coefficients are also shared by all lanes of
        this warp.
    */

    for (
        int j = lane;
        j < K;
        j += TTILE
    )
    {
        float value = 0.0f;

        if (valid_h)
        {
            value =
                k[h * K + j];
        }

        k_s[j] = value;
    }

    __syncthreads();

    if (valid_output)
    {
        float acc = 0.0f;

        for (int j = 0; j < K; ++j)
        {
            acc = fmaf(
                tile[lane + j],
                k_s[j],
                acc
            );
        }

        y[base + t] = acc;
    }
}


/*
==============================================================
BACKWARD INPUT -- SHARED MEMORY
==============================================================

Forward:

    y[q] += x[q + j - half] * k[j]

Therefore:

    gx[t] += gy[t - j + half] * k[j]

For a warp computing t0..t0+31, the required gy interval is:

    lowest:
        t0 - (K - 1 - half)

    highest:
        t0 + 31 + half

Total length:

    TTILE + K - 1

Again the whole interval is cooperatively cached in shared
memory.
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

    const int tile_len =
        TTILE + K - 1;

    extern __shared__ float smem[];

    float* tile_all =
        smem;

    float* kernel_all =
        smem + HTILE * tile_len;

    float* tile =
        tile_all + warp_local * tile_len;

    float* k_s =
        kernel_all + warp_local * K;

    const int in_start =
        t0 - left_halo;

    const int base =
        valid_h
        ? (b * H + h) * L
        : 0;

    /*
        Cooperative grad-output tile load.
    */

    for (
        int i = lane;
        i < tile_len;
        i += TTILE
    )
    {
        const int gi =
            in_start + i;

        float value = 0.0f;

        if (
            valid_h
            && gi >= 0
            && gi < L
        )
        {
            value =
                gy[base + gi];
        }

        tile[i] = value;
    }

    /*
        Cooperative kernel load.
    */

    for (
        int j = lane;
        j < K;
        j += TTILE
    )
    {
        float value = 0.0f;

        if (valid_h)
        {
            value =
                k[h * K + j];
        }

        k_s[j] = value;
    }

    __syncthreads();

    if (valid_output)
    {
        float acc = 0.0f;

        /*
            Local shared-memory index derivation:

                gi = t - j + half

            and:

                in_start = t0 - (K - 1 - half)

            therefore:

                gi - in_start
                    = lane + K - 1 - j
        */

        for (int j = 0; j < K; ++j)
        {
            const int local_idx =
                lane + K - 1 - j;

            acc = fmaf(
                tile[local_idx],
                k_s[j],
                acc
            );
        }

        gx[base + t] = acc;
    }
}


/*
==============================================================
BACKWARD KERNEL -- CONTROL PATH
==============================================================

INTENTIONALLY IDENTICAL IN ALGORITHM TO THE FROZEN
NAIVE/GMC WEIGHT-GRADIENT IMPLEMENTATION.

No Shared Memory optimization is introduced here.

Reason:

Changing its decomposition would mix Shared Memory with a new
parallel reduction strategy and prevent clean attribution.

One thread computes one (h,j).
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

    const int j =
        (int)blockIdx.x
        * (int)blockDim.x
        + (int)threadIdx.x;

    if (h >= H || j >= K)
        return;

    const int half =
        K / 2;

    float acc = 0.0f;

    for (int b = 0; b < B; ++b)
    {
        const int base =
            (b * H + h) * L;

        for (int t = 0; t < L; ++t)
        {
            const int xi =
                t + j - half;

            if (
                xi >= 0
                && xi < L
            )
            {
                acc = fmaf(
                    gy[base + t],
                    x[base + xi],
                    acc
                );
            }
        }
    }

    gk[h * K + j] = acc;
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

def shared_memory_bytes(K: int) -> int:
    """
    Shared memory per block:

        HTILE * (TTILE + K - 1) input/gradient tile
      + HTILE * K               kernel coefficients

    FP32 = 4 bytes.
    """

    tile_len = (
        TTILE
        + K
        - 1
    )

    floats = (
        HTILE * tile_len
        + HTILE * K
    )

    return (
        floats
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
            shared=0,
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
