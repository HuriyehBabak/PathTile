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
    """
    Push/pop the CUDA primary context used by Torch.
    """

    def __enter__(self):
        _PRIMARY_CONTEXT.push()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _PRIMARY_CONTEXT.pop()
        return False


# ============================================================
# Torch <-> PyCUDA zero-copy interoperability
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
            raise ValueError("Tensor must be contiguous")

        if tensor.device.index != torch.cuda.current_device():
            raise ValueError(
                "Tensor device must match current CUDA device"
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

        # Keep underlying Torch tensor alive.
        arr._torch_tensor_ref = tensor
        arr._torch_pointer_ref = holder

        return arr


# ============================================================
# Experimental configuration
# ============================================================
#
# Paper workload baseline:
#
#   TPB = 512
#
# Coalesced mapping:
#
#   one warp = one channel
#   lane 0..31 = consecutive temporal positions
#
# IMPORTANT:
#
#   NO shared memory
#   NO warp shuffle
#   NO warp reduction
#   NO partial buffer
#   NO CHUNK decomposition
#
# ============================================================

TTILE = 32
HTILE = 16

THREADS_PER_BLOCK = TTILE * HTILE   # 512


# ============================================================
# CUDA kernels
# ============================================================

KERNEL_CODE = r"""
#define TTILE 32
#define HTILE 16


extern "C" {


/*
==============================================================
Forward
==============================================================

Tensor layout:

    x : [B,H,L]
    k : [H,K]
    y : [B,H,L]

Warp mapping:

    lane 0  -> t
    lane 1  -> t+1
    ...
    lane 31 -> t+31

Thus for a fixed kernel tap j, adjacent lanes access
adjacent temporal global-memory locations.
*/

__global__ void dwconv1d_fwd_coalesced(
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

    const int lane_t =
        tid % TTILE;

    const int lane_h =
        tid / TTILE;

    const int b =
        (int)blockIdx.z;

    const int h =
        (int)blockIdx.y * HTILE
        + lane_h;

    const int t =
        (int)blockIdx.x * TTILE
        + lane_t;

    if (b >= B || h >= H || t >= L)
        return;

    const int half =
        K / 2;

    const int base =
        (b * H + h) * L;

    float acc = 0.0f;

    for (int j = 0; j < K; ++j)
    {
        const int xi =
            t + j - half;

        if ((unsigned)xi < (unsigned)L)
        {
            acc = fmaf(
                x[base + xi],
                k[h * K + j],
                acc
            );
        }
    }

    y[base + t] = acc;
}


/*
==============================================================
Backward w.r.t. input
==============================================================

Forward definition:

    y[q] += x[q + j - half] * k[j]

For a fixed input index t:

    t = q + j - half

therefore:

    q = t - j + half

Thus:

    gx[t] += gy[t - j + half] * k[j]

This formula works for BOTH odd and even K.
*/

__global__ void dwconv1d_bwd_input_coalesced(
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

    const int lane_t =
        tid % TTILE;

    const int lane_h =
        tid / TTILE;

    const int b =
        (int)blockIdx.z;

    const int h =
        (int)blockIdx.y * HTILE
        + lane_h;

    const int t =
        (int)blockIdx.x * TTILE
        + lane_t;

    if (b >= B || h >= H || t >= L)
        return;

    const int half =
        K / 2;

    const int base =
        (b * H + h) * L;

    float acc = 0.0f;

    for (int j = 0; j < K; ++j)
    {
        const int gi =
            t - j + half;

        if ((unsigned)gi < (unsigned)L)
        {
            acc = fmaf(
                gy[base + gi],
                k[h * K + j],
                acc
            );
        }
    }

    gx[base + t] = acc;
}


/*
==============================================================
Backward w.r.t. kernel
==============================================================

INTENTIONALLY retains the Naive algorithm.

One CUDA thread computes one (h,j) gradient coefficient.

The thread performs the complete B x L accumulation.

No:

    shared memory
    warp shuffle
    parallel reduction
    temporary partials
    chunk decomposition

For K=48 and H=128 with TPB=512:

    grid.x = 1
    grid.y = 128

=> approximately 128 blocks, matching the Naive baseline
parallel structure.
*/

__global__ void dwconv1d_bwd_kernel_coalesced(
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
        (int)blockIdx.x * (int)blockDim.x
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

            if ((unsigned)xi < (unsigned)L)
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
# Compile CUDA module exactly once
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
                "dwconv1d_fwd_coalesced"
            )

            _BWD_INPUT_KERNEL = _MODULE.get_function(
                "dwconv1d_bwd_input_coalesced"
            )

            _BWD_KERNEL_KERNEL = _MODULE.get_function(
                "dwconv1d_bwd_kernel_coalesced"
            )

    return (
        _FWD_KERNEL,
        _BWD_INPUT_KERNEL,
        _BWD_KERNEL_KERNEL,
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
) -> float:
    """
    Time one PyCUDA kernel using CUDA events.

    Events are explicitly destroyed while the CUDA primary
    context is still active. This avoids PyCUDA cleanup warnings.
    """

    torch.cuda.synchronize()

    with CudaContextGuard():
        start = cuda.Event()
        end = cuda.Event()

        start.record()

        kernel(
            *args,
            block=block,
            grid=grid,
        )

        end.record()
        end.synchronize()

        elapsed_ms = float(
            start.time_till(end)
        )

        # Make sure PyCUDA event objects are destroyed while
        # the CUDA context is still pushed.
        del start
        del end

    return elapsed_ms

# ============================================================
# PyTorch autograd interface
# ============================================================

class DepthwiseConvCoalesced(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if not x.is_cuda:
            raise ValueError(
                "x must be a CUDA tensor"
            )

        if not kernel.is_cuda:
            raise ValueError(
                "kernel must be a CUDA tensor"
            )

        if x.dtype != torch.float32:
            raise TypeError(
                f"x must be float32, got {x.dtype}"
            )

        if kernel.dtype != torch.float32:
            raise TypeError(
                f"kernel must be float32, got {kernel.dtype}"
            )

        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B,H,L], got {tuple(x.shape)}"
            )

        if kernel.ndim != 2:
            raise ValueError(
                f"kernel must have shape [H,K], got {tuple(kernel.shape)}"
            )

        # ----------------------------------------------------
        # Contiguous layout
        # ----------------------------------------------------

        x = x.contiguous()
        kernel = kernel.contiguous()

        B, H, L = x.shape
        H_kernel, K = kernel.shape

        if H_kernel != H:
            raise ValueError(
                f"Channel mismatch: x has H={H}, "
                f"kernel has H={H_kernel}"
            )

        # ----------------------------------------------------
        # Save for backward
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        y = torch.empty_like(x)

        # ----------------------------------------------------
        # PyCUDA views
        # ----------------------------------------------------

        with CudaContextGuard():
            x_g = TorchCudaInterop.as_gpuarray(x)
            k_g = TorchCudaInterop.as_gpuarray(kernel)
            y_g = TorchCudaInterop.as_gpuarray(y)

        # ----------------------------------------------------
        # Kernel
        # ----------------------------------------------------

        fwd_kernel, _, _ = _get_kernels()

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

        elapsed_ms = _time_kernel(
            fwd_kernel,
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

        # ----------------------------------------------------
        # Gradient validation
        # ----------------------------------------------------

        if not grad_y.is_cuda:
            raise ValueError(
                "grad_y must be a CUDA tensor"
            )

        if grad_y.dtype != torch.float32:
            raise TypeError(
                f"grad_y must be float32, got {grad_y.dtype}"
            )

        grad_y = grad_y.contiguous()

        # ----------------------------------------------------
        # Allocate outputs
        # ----------------------------------------------------

        gx = torch.empty_like(x)
        gk = torch.empty_like(kernel)

        # ----------------------------------------------------
        # PyCUDA views
        # ----------------------------------------------------

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

        _, bwd_input_kernel, bwd_kernel_kernel = (
            _get_kernels()
        )

        # ====================================================
        # Backward input
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

        bwd_input_ms = _time_kernel(
            bwd_input_kernel,
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
        )

        BWD_IN_TIMES.append(
            bwd_input_ms
        )

        # ====================================================
        # Backward kernel
        #
        # Keep launch configuration consistent with Naive.
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
            bwd_kernel_kernel,
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
        )

        BWD_K_TIMES.append(
            bwd_kernel_ms
        )

        return gx, gk


# ============================================================
# Public convenience function
# ============================================================

def depthwise_conv_coalesced(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    return DepthwiseConvCoalesced.apply(
        x,
        kernel,
    )


__all__ = [
    "DepthwiseConvCoalesced",
    "depthwise_conv_coalesced",
    "THREADS_PER_BLOCK",
    "TTILE",
    "HTILE",
    "FWD_TIMES",
    "BWD_IN_TIMES",
    "BWD_K_TIMES",
    "reset_timing_collectors",
]
