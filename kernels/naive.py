from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch


# ============================================================
# Lazy PyCUDA initialization
# ============================================================
#
# Important for L3S/Pascal:
# - pascal head node has no CUDA driver library available
# - GPU nodes provide libcuda.so.1
#
# Therefore PyCUDA is imported only when an actual GPU
# operation is requested.
# ============================================================

_CUDA = None
_GPUARRAY = None
_SOURCE_MODULE = None
_PRIMARY_CONTEXT = None
_TORCH_POINTER_TYPE = None


def _load_pycuda():
    global _CUDA
    global _GPUARRAY
    global _SOURCE_MODULE

    if _CUDA is not None:
        return _CUDA, _GPUARRAY, _SOURCE_MODULE

    try:
        import pycuda.driver as cuda
        import pycuda.gpuarray as gpuarray
        from pycuda.compiler import SourceModule
    except ImportError as exc:
        raise RuntimeError(
            "PyCUDA could not be loaded. "
            "On Pascal, CUDA/PyCUDA operations must be run "
            "inside a SLURM GPU job."
        ) from exc

    _CUDA = cuda
    _GPUARRAY = gpuarray
    _SOURCE_MODULE = SourceModule

    return _CUDA, _GPUARRAY, _SOURCE_MODULE


def _ensure_primary_context():
    """
    Make PyTorch and PyCUDA use the same CUDA primary context.
    """

    global _PRIMARY_CONTEXT

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. "
            "Run CUDA code inside a SLURM GPU job."
        )

    if _PRIMARY_CONTEXT is not None:
        return _PRIMARY_CONTEXT

    # Initialize PyTorch CUDA first.
    torch.cuda.init()

    cuda, _, _ = _load_pycuda()
    cuda.init()

    device_index = torch.cuda.current_device()

    device = cuda.Device(device_index)

    # PyTorch also uses the CUDA primary context.
    _PRIMARY_CONTEXT = device.retain_primary_context()

    return _PRIMARY_CONTEXT


@contextmanager
def _cuda_primary_context():
    """
    Temporarily make the shared CUDA primary context current.
    """

    context = _ensure_primary_context()

    context.push()

    try:
        yield context
    finally:
        context.pop()


# ============================================================
# Torch <-> PyCUDA zero-copy interoperability
# ============================================================

class TorchCudaInterop:
    """
    Zero-copy view of a PyTorch CUDA tensor as a PyCUDA GPUArray.

    The tensor remains owned by PyTorch.
    PyCUDA only receives a view of the same device memory.
    """

    @staticmethod
    def _get_pointer_type():
        global _TORCH_POINTER_TYPE

        if _TORCH_POINTER_TYPE is not None:
            return _TORCH_POINTER_TYPE

        cuda, _, _ = _load_pycuda()

        class TorchPointer(cuda.PointerHolderBase):
            def __init__(self, ptr: int):
                super().__init__()
                self.gpudata = int(ptr)

            def __int__(self):
                return self.gpudata

        _TORCH_POINTER_TYPE = TorchPointer

        return _TORCH_POINTER_TYPE

    @staticmethod
    def as_gpuarray(
        tensor: torch.Tensor,
        dtype=np.float32,
    ):
        if not tensor.is_cuda:
            raise ValueError(
                "Tensor must be on CUDA."
            )

        if tensor.dtype != torch.float32:
            raise TypeError(
                f"Expected torch.float32, got {tensor.dtype}"
            )

        if not tensor.is_contiguous():
            raise ValueError(
                "Tensor must be contiguous."
            )

        _, gpuarray, _ = _load_pycuda()

        pointer_type = TorchCudaInterop._get_pointer_type()

        holder = pointer_type(
            tensor.data_ptr()
        )

        strides = tuple(
            stride * tensor.element_size()
            for stride in tensor.stride()
        )

        with _cuda_primary_context():
            return gpuarray.GPUArray(
                shape=tuple(tensor.shape),
                dtype=dtype,
                gpudata=holder,
                strides=strides,
            )


# ============================================================
# CUDA kernels
# ============================================================

KERNEL_CODE = r"""
extern "C" {


/*
============================================================
Forward
============================================================

Input:
    x : [B, H, L]

Kernel:
    k : [H, K]

Output:
    y : [B, H, L]

Naive mapping:
    one CUDA thread -> one output element y[b,h,t]

No:
    - shared memory
    - tiling
    - warp cooperation
    - explicit data reuse
============================================================
*/

__global__ void dwconv1d_fwd_naive(
    const float* x,
    const float* k,
    float* y,
    int B,
    int H,
    int L,
    int K)
{
    const int b = (int)blockIdx.z;
    const int h = (int)blockIdx.y;

    const int t =
        (int)blockIdx.x * (int)blockDim.x
        + (int)threadIdx.x;

    if (b >= B || h >= H || t >= L)
        return;

    const int half = K / 2;

    const int base =
        (b * H + h) * L;

    float acc = 0.0f;

    for (int j = 0; j < K; ++j)
    {
        const int xi =
            t + j - half;

        if (xi >= 0 && xi < L)
        {
            acc +=
                x[base + xi]
                * k[h * K + j];
        }
    }

    y[base + t] = acc;
}


/*
============================================================
Backward: gradient w.r.t. input
============================================================

Forward relation:

    xi = output_t + j - half

For a fixed input position t:

    output_t = t - j + half

This formulation works correctly for both:
    - even K, e.g. K = 48
    - odd K
============================================================
*/

__global__ void dwconv1d_bwd_input_naive(
    const float* gy,
    const float* k,
    float* gx,
    int B,
    int H,
    int L,
    int K)
{
    const int b = (int)blockIdx.z;
    const int h = (int)blockIdx.y;

    const int t =
        (int)blockIdx.x * (int)blockDim.x
        + (int)threadIdx.x;

    if (b >= B || h >= H || t >= L)
        return;

    const int half = K / 2;

    const int base =
        (b * H + h) * L;

    float acc = 0.0f;

    for (int j = 0; j < K; ++j)
    {
        const int gi =
            t - j + half;

        if (gi >= 0 && gi < L)
        {
            acc +=
                gy[base + gi]
                * k[h * K + j];
        }
    }

    gx[base + t] = acc;
}


/*
============================================================
Backward: gradient w.r.t. kernel
============================================================

Naive mapping:

    one CUDA thread -> one gk[h,j]

Each thread performs a serial reduction over:

    B x L

For the main configuration:

    B = 16384
    L = 48

this means a very large reduction per thread.
============================================================
*/

__global__ void dwconv1d_bwd_kernel_naive(
    const float* gy,
    const float* x,
    float* gk,
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

    const int half = K / 2;

    float acc = 0.0f;

    for (int b = 0; b < B; ++b)
    {
        const int base =
            (b * H + h) * L;

        for (int t = 0; t < L; ++t)
        {
            const int xi =
                t + j - half;

            if (xi >= 0 && xi < L)
            {
                acc +=
                    gy[base + t]
                    * x[base + xi];
            }
        }
    }

    gk[h * K + j] = acc;
}


}  // extern "C"
"""


# ============================================================
# CUDA module compilation
# ============================================================

_MODULE = None

_FWD_KERNEL = None
_BWD_INPUT_KERNEL = None
_BWD_KERNEL_KERNEL = None

_COMPILED_ARCH = None


def compile_naive_module():
    """
    Compile the CUDA kernels for the GPU used by the current
    SLURM job.

    Examples:
        RTX A6000 -> sm_86
        A100      -> sm_80
    """

    global _COMPILED_ARCH

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required. "
            "Run this inside a SLURM GPU job."
        )

    major, minor = torch.cuda.get_device_capability()

    _COMPILED_ARCH = f"sm_{major}{minor}"

    arch_flag = f"-arch={_COMPILED_ARCH}"

    _, _, SourceModule = _load_pycuda()

    with _cuda_primary_context():
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

            # KERNEL_CODE already contains extern "C".
            no_extern_c=True,
        )

    return module


def get_kernels():
    """
    Compile once per Python process and cache all kernel handles.
    """

    global _MODULE
    global _FWD_KERNEL
    global _BWD_INPUT_KERNEL
    global _BWD_KERNEL_KERNEL

    if _MODULE is None:
        _MODULE = compile_naive_module()

        _FWD_KERNEL = _MODULE.get_function(
            "dwconv1d_fwd_naive"
        )

        _BWD_INPUT_KERNEL = _MODULE.get_function(
            "dwconv1d_bwd_input_naive"
        )

        _BWD_KERNEL_KERNEL = _MODULE.get_function(
            "dwconv1d_bwd_kernel_naive"
        )

    return (
        _FWD_KERNEL,
        _BWD_INPUT_KERNEL,
        _BWD_KERNEL_KERNEL,
    )


# ============================================================
# Runtime collectors
# ============================================================

FWD_TIMES: list[float] = []
BWD_IN_TIMES: list[float] = []
BWD_K_TIMES: list[float] = []


def clear_timings():
    """
    Clear all raw kernel runtime measurements.
    """

    FWD_TIMES.clear()
    BWD_IN_TIMES.clear()
    BWD_K_TIMES.clear()


def _time_kernel(
    kernel,
    args,
    block,
    grid,
) -> float:

    cuda, _, _ = _load_pycuda()

    torch.cuda.synchronize()

    with _cuda_primary_context():
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

        # Destroy events while CUDA context is still current.
        del start
        del end

    return elapsed_ms

# ============================================================
# PyTorch autograd Function
# ============================================================

class DepthwiseConvNaive(torch.autograd.Function):
    """
    Naive CUDA implementation of the depthwise 1-D convolution.

    Default TPB remains 512 to reproduce the original
    experimental configuration.

    Later experiments can explicitly change this value to study
    launch-configuration sensitivity.
    """

    THREADS_PER_BLOCK = 512

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        kernel: torch.Tensor,
    ):
        if not x.is_cuda:
            raise ValueError(
                "x must be a CUDA tensor."
            )

        if not kernel.is_cuda:
            raise ValueError(
                "kernel must be a CUDA tensor."
            )

        if x.dtype != torch.float32:
            raise TypeError(
                f"x must be float32, got {x.dtype}"
            )

        if kernel.dtype != torch.float32:
            raise TypeError(
                "kernel must be float32, "
                f"got {kernel.dtype}"
            )

        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [B,H,L], got {x.shape}"
            )

        if kernel.ndim != 2:
            raise ValueError(
                "kernel must have shape [H,K], "
                f"got {kernel.shape}"
            )

        x = x.contiguous()
        kernel = kernel.contiguous()

        B, H, L = x.shape
        H_kernel, K = kernel.shape

        if H_kernel != H:
            raise ValueError(
                "Kernel channel dimension does not match input: "
                f"x H={H}, kernel H={H_kernel}"
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

        x_g = TorchCudaInterop.as_gpuarray(x)
        k_g = TorchCudaInterop.as_gpuarray(kernel)
        y_g = TorchCudaInterop.as_gpuarray(y)

        fwd, _, _ = get_kernels()

        TPB = int(
            DepthwiseConvNaive.THREADS_PER_BLOCK
        )

        if TPB <= 0 or TPB > 1024:
            raise ValueError(
                "THREADS_PER_BLOCK must be in [1, 1024]."
            )

        block = (
            TPB,
            1,
            1,
        )

        grid = (
            (L + TPB - 1) // TPB,
            H,
            B,
        )

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
            block=block,
            grid=grid,
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

        if not grad_y.is_cuda:
            raise ValueError(
                "grad_y must be a CUDA tensor."
            )

        if grad_y.dtype != torch.float32:
            raise TypeError(
                "grad_y must be float32, "
                f"got {grad_y.dtype}"
            )

        if tuple(grad_y.shape) != (B, H, L):
            raise ValueError(
                "Unexpected grad_y shape: "
                f"{grad_y.shape}; expected {(B, H, L)}"
            )

        grad_y = grad_y.contiguous()

        gx = torch.empty_like(x)
        gk = torch.empty_like(kernel)

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

        _, bwd_input, bwd_kernel = get_kernels()

        TPB = int(
            DepthwiseConvNaive.THREADS_PER_BLOCK
        )

        block = (
            TPB,
            1,
            1,
        )

        # ====================================================
        # Gradient w.r.t. input
        # ====================================================

        grid_input = (
            (L + TPB - 1) // TPB,
            H,
            B,
        )

        elapsed_ms = _time_kernel(
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
            block=block,
            grid=grid_input,
        )

        BWD_IN_TIMES.append(
            elapsed_ms
        )

        # ====================================================
        # Gradient w.r.t. kernel
        # ====================================================

        grid_kernel = (
            (K + TPB - 1) // TPB,
            H,
            1,
        )

        elapsed_ms = _time_kernel(
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
            block=block,
            grid=grid_kernel,
        )

        BWD_K_TIMES.append(
            elapsed_ms
        )

        # Gradients corresponding to forward inputs:
        #     forward(ctx, x, kernel)
        return gx, gk


# ============================================================
# Configuration helpers
# ============================================================

def set_threads_per_block(value: int):
    """
    Explicitly configure TPB.

    Keep 512 for reproduction of the original experiment.
    Later use e.g. 32/64/128/256 for the launch sweep.
    """

    value = int(value)

    if value <= 0 or value > 1024:
        raise ValueError(
            "threads_per_block must be in [1, 1024]."
        )

    DepthwiseConvNaive.THREADS_PER_BLOCK = value


def get_runtime_data():
    """
    Return copies so callers cannot modify the global collectors.
    """

    return {
        "fwd_ms": list(FWD_TIMES),
        "bwd_input_ms": list(BWD_IN_TIMES),
        "bwd_kernel_ms": list(BWD_K_TIMES),
    }


# ============================================================
# Direct execution / smoke test
# ============================================================

if __name__ == "__main__":

    print("Naive CUDA kernel module")

    if not torch.cuda.is_available():
        print(
            "CUDA GPU is not available on this node."
        )
        print(
            "This is expected on the Pascal head node."
        )
        print(
            "Use validate_naive.py inside a SLURM GPU job."
        )

    else:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

        print(
            "Compute capability:",
            torch.cuda.get_device_capability(0),
        )

        print(
            "Threads per block:",
            DepthwiseConvNaive.THREADS_PER_BLOCK,
        )

        # Compile kernels, but do not run the full benchmark.
        get_kernels()

        print(
            "Compiled architecture:",
            _COMPILED_ARCH,
        )

        print(
            "Naive CUDA kernel compilation: PASS"
        )