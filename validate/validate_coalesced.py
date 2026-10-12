from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from kernels.coalesced import (
    DepthwiseConvCoalesced,
    FWD_TIMES,
    BWD_IN_TIMES,
    BWD_K_TIMES,
    reset_timing_collectors,
)


SEED = 42
ATOL = 1e-3
RTOL = 1e-3


def reference_depthwise_conv(
    x: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    """
    Reference matching the CUDA convention exactly:

        y[t] = sum_j x[t + j - floor(K/2)] * k[j]

    For even K, padding is asymmetric.

    Example K=48:
        left  = 24
        right = 23
    """

    K = kernel.shape[1]
    half = K // 2

    pad_left = half
    pad_right = K - 1 - half

    x_padded = F.pad(
        x,
        (pad_left, pad_right),
    )

    return F.conv1d(
        x_padded,
        kernel.unsqueeze(1),
        padding=0,
        groups=x.shape[1],
    )


def run_case(
    B: int,
    H: int,
    L: int,
    K: int,
) -> bool:

    print()
    print("=" * 78)
    print(f"Testing B={B}, H={H}, L={L}, K={K}")
    print("=" * 78)

    torch.manual_seed(SEED)

    # Keep values moderate to make gradient comparisons stable.
    x_base = (
        0.1
        * torch.randn(
            B,
            H,
            L,
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous()

    k_base = (
        0.1
        * torch.randn(
            H,
            K,
            device="cuda",
            dtype=torch.float32,
        )
    ).contiguous()

    grad_out = (
        0.1
        * torch.randn_like(x_base)
    ).contiguous()

    # ========================================================
    # Custom CUDA
    # ========================================================

    x_custom = (
        x_base.detach()
        .clone()
        .requires_grad_(True)
    )

    k_custom = (
        k_base.detach()
        .clone()
        .requires_grad_(True)
    )

    reset_timing_collectors()

    y_custom = DepthwiseConvCoalesced.apply(
        x_custom,
        k_custom,
    )

    y_custom.backward(
        grad_out
    )

    torch.cuda.synchronize()

    # ========================================================
    # PyTorch reference
    # ========================================================

    x_ref = (
        x_base.detach()
        .clone()
        .requires_grad_(True)
    )

    k_ref = (
        k_base.detach()
        .clone()
        .requires_grad_(True)
    )

    y_ref = reference_depthwise_conv(
        x_ref,
        k_ref,
    )

    y_ref.backward(
        grad_out
    )

    torch.cuda.synchronize()

    # ========================================================
    # Differences
    # ========================================================

    fwd_diff = (
        y_custom.detach()
        - y_ref.detach()
    ).abs()

    gx_diff = (
        x_custom.grad.detach()
        - x_ref.grad.detach()
    ).abs()

    gk_diff = (
        k_custom.grad.detach()
        - k_ref.grad.detach()
    ).abs()

    fwd_max = float(fwd_diff.max().item())
    gx_max = float(gx_diff.max().item())
    gk_max = float(gk_diff.max().item())

    # ========================================================
    # allclose checks
    # ========================================================

    fwd_ok = bool(
        torch.allclose(
            y_custom.detach(),
            y_ref.detach(),
            atol=ATOL,
            rtol=RTOL,
        )
    )

    gx_ok = bool(
        torch.allclose(
            x_custom.grad.detach(),
            x_ref.grad.detach(),
            atol=ATOL,
            rtol=RTOL,
        )
    )

    gk_ok = bool(
        torch.allclose(
            k_custom.grad.detach(),
            k_ref.grad.detach(),
            atol=ATOL,
            rtol=RTOL,
        )
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        f"Forward:         "
        f"{'PASS' if fwd_ok else 'FAIL'}"
        f" | max abs diff = {fwd_max:.6e}"
    )

    print(
        f"Backward Input:  "
        f"{'PASS' if gx_ok else 'FAIL'}"
        f" | max abs diff = {gx_max:.6e}"
    )

    print(
        f"Backward Kernel: "
        f"{'PASS' if gk_ok else 'FAIL'}"
        f" | max abs diff = {gk_max:.6e}"
    )

    print()
    print("Single-run CUDA timings (validation only):")

    if FWD_TIMES:
        print(
            f"  Forward:        "
            f"{FWD_TIMES[-1]:.6f} ms"
        )

    if BWD_IN_TIMES:
        print(
            f"  Backward Input: "
            f"{BWD_IN_TIMES[-1]:.6f} ms"
        )

    if BWD_K_TIMES:
        print(
            f"  Backward Kernel:"
            f" {BWD_K_TIMES[-1]:.6f} ms"
        )

    passed = (
        fwd_ok
        and gx_ok
        and gk_ok
    )

    # Cleanup
    del (
        x_base,
        k_base,
        grad_out,
        x_custom,
        k_custom,
        y_custom,
        x_ref,
        k_ref,
        y_ref,
    )

    torch.cuda.empty_cache()

    return passed


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help=(
            "Also validate the full paper workload: "
            "B=16384,H=128,L=48,K=48"
        ),
    )

    args = parser.parse_args()

    assert torch.cuda.is_available()

    torch.cuda.set_device(0)

    print("=" * 78)
    print("GLOBAL MEMORY COALESCING CORRECTNESS")
    print("=" * 78)

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print(
        "Compute capability:",
        torch.cuda.get_device_capability(0),
    )

    print(
        f"Tolerance: atol={ATOL}, rtol={RTOL}"
    )

    # ========================================================
    # Small correctness cases
    #
    # Explicitly test:
    #   odd K
    #   even K
    #   odd K
    # ========================================================

    cases = [
        (2, 4, 48, 47),
        (2, 4, 48, 48),
        (2, 4, 48, 49),
        (8, 16, 48, 48),
    ]

    all_ok = True

    for case in cases:
        passed = run_case(*case)
        all_ok = all_ok and passed

    # ========================================================
    # Full paper workload
    # ========================================================

    if args.full:

        print()
        print("#" * 78)
        print("FULL PAPER WORKLOAD")
        print("#" * 78)

        passed = run_case(
            B=16384,
            H=128,
            L=48,
            K=48,
        )

        all_ok = (
            all_ok
            and passed
        )

    print()
    print("=" * 78)

    if all_ok:
        print("OVERALL: PASS")
    else:
        print("OVERALL: FAIL")

    print("=" * 78)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
