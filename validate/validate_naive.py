import torch
import torch.nn.functional as F

from kernels.naive import (
    DepthwiseConvNaive,
    clear_timings,
    get_runtime_data,
)


def torch_reference_dwconv(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    PyTorch reference matching the CUDA convention:

        xi = t + j - K//2

    CUDA kernel:
        y[b,h,t] += x[b,h,xi] * kernel[h,j]

    For even K, padding is asymmetric.

    Example K=48:
        left  = 24
        right = 23
    """

    if x.ndim != 3:
        raise ValueError(
            f"x must be [B,H,L], got {x.shape}"
        )

    if kernel.ndim != 2:
        raise ValueError(
            f"kernel must be [H,K], got {kernel.shape}"
        )

    B, H, L = x.shape
    H_kernel, K = kernel.shape

    if H != H_kernel:
        raise ValueError(
            f"Channel mismatch: {H} vs {H_kernel}"
        )

    left = K // 2
    right = K - 1 - left

    x_padded = F.pad(
        x,
        (left, right),
    )

    # [H,K] -> [H,1,K]
    weight = kernel.unsqueeze(1)

    return F.conv1d(
        x_padded,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=H,
    )


def max_abs_error(
    actual: torch.Tensor,
    reference: torch.Tensor,
) -> float:
    return float(
        (actual - reference)
        .abs()
        .max()
        .item()
    )


def validate_case(
    B: int,
    H: int,
    L: int,
    K: int,
):
    print()
    print("=" * 70)

    print(
        f"Testing B={B}, H={H}, L={L}, K={K}"
    )

    print(
        "TPB:",
        DepthwiseConvNaive.THREADS_PER_BLOCK,
    )

    print("=" * 70)

    torch.manual_seed(42)

    clear_timings()

    # ========================================================
    # Input data
    # ========================================================

    x_base = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
    )

    k_base = torch.randn(
        H,
        K,
        device="cuda",
        dtype=torch.float32,
    )

    grad_y = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
    )

    # ========================================================
    # Custom CUDA implementation
    # ========================================================

    x_cuda = (
        x_base
        .clone()
        .detach()
        .requires_grad_(True)
    )

    k_cuda = (
        k_base
        .clone()
        .detach()
        .requires_grad_(True)
    )

    y_cuda = DepthwiseConvNaive.apply(
        x_cuda,
        k_cuda,
    )

    y_cuda.backward(
        grad_y
    )

    gx_cuda = (
        x_cuda.grad
        .detach()
        .clone()
    )

    gk_cuda = (
        k_cuda.grad
        .detach()
        .clone()
    )

    # ========================================================
    # PyTorch ground truth
    # ========================================================

    x_ref = (
        x_base
        .clone()
        .detach()
        .requires_grad_(True)
    )

    k_ref = (
        k_base
        .clone()
        .detach()
        .requires_grad_(True)
    )

    y_ref = torch_reference_dwconv(
        x_ref,
        k_ref,
    )

    y_ref.backward(
        grad_y
    )

    gx_ref = (
        x_ref.grad
        .detach()
        .clone()
    )

    gk_ref = (
        k_ref.grad
        .detach()
        .clone()
    )

    # ========================================================
    # Numerical errors
    # ========================================================

    fwd_error = max_abs_error(
        y_cuda.detach(),
        y_ref.detach(),
    )

    gx_error = max_abs_error(
        gx_cuda,
        gx_ref,
    )

    gk_error = max_abs_error(
        gk_cuda,
        gk_ref,
    )

    print()
    print("Numerical comparison")
    print("--------------------")

    print(
        f"Forward max abs error:     "
        f"{fwd_error:.8e}"
    )

    print(
        f"Grad-input max abs error:  "
        f"{gx_error:.8e}"
    )

    print(
        f"Grad-kernel max abs error: "
        f"{gk_error:.8e}"
    )

    # ========================================================
    # allclose validation
    # ========================================================

    rtol = 2e-4
    atol = 2e-4

    fwd_ok = torch.allclose(
        y_cuda.detach(),
        y_ref.detach(),
        rtol=rtol,
        atol=atol,
    )

    gx_ok = torch.allclose(
        gx_cuda,
        gx_ref,
        rtol=rtol,
        atol=atol,
    )

    gk_ok = torch.allclose(
        gk_cuda,
        gk_ref,
        rtol=rtol,
        atol=atol,
    )

    print()
    print("Correctness")
    print("-----------")

    print(
        "Forward:     ",
        "PASS" if fwd_ok else "FAIL",
    )

    print(
        "Grad input:  ",
        "PASS" if gx_ok else "FAIL",
    )

    print(
        "Grad kernel: ",
        "PASS" if gk_ok else "FAIL",
    )

    # ========================================================
    # Kernel runtime generated during correctness test
    # ========================================================

    timings = get_runtime_data()

    print()
    print("Kernel runtime")
    print("--------------")

    if timings["fwd_ms"]:
        print(
            f"Forward:     "
            f"{timings['fwd_ms'][-1]:.6f} ms"
        )

    if timings["bwd_input_ms"]:
        print(
            f"Grad input:  "
            f"{timings['bwd_input_ms'][-1]:.6f} ms"
        )

    if timings["bwd_kernel_ms"]:
        print(
            f"Grad kernel: "
            f"{timings['bwd_kernel_ms'][-1]:.6f} ms"
        )

    # ========================================================
    # Hard failure
    # ========================================================

    if not fwd_ok:
        raise AssertionError(
            "Forward correctness validation FAILED"
        )

    if not gx_ok:
        raise AssertionError(
            "Gradient-input correctness validation FAILED"
        )

    if not gk_ok:
        raise AssertionError(
            "Gradient-kernel correctness validation FAILED"
        )

    print()
    print("Case result: PASS")


def main():

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. "
            "Run validate_naive.py inside a SLURM GPU job."
        )

    print()
    print("Naive CUDA correctness validation")
    print("=================================")

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "PyTorch CUDA:",
        torch.version.cuda,
    )

    print(
        "Threads per block:",
        DepthwiseConvNaive.THREADS_PER_BLOCK,
    )

    # --------------------------------------------------------
    # Test 1
    # Even kernel length.
    #
    # Important because the original implementation had an
    # alignment issue in grad-input for even K.
    # --------------------------------------------------------

    validate_case(
        B=2,
        H=3,
        L=17,
        K=6,
    )

    # --------------------------------------------------------
    # Test 2
    # Odd K to verify both padding conventions.
    # --------------------------------------------------------

    validate_case(
        B=2,
        H=3,
        L=17,
        K=7,
    )

    # --------------------------------------------------------
    # Test 3
    # Main paper sequence/kernel dimensions.
    #
    # Keep B small here because this is a correctness test,
    # not a performance benchmark.
    # --------------------------------------------------------

    validate_case(
        B=2,
        H=8,
        L=48,
        K=48,
    )

    print()
    print("=" * 70)
    print(
        "ALL NAIVE CUDA CORRECTNESS TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()