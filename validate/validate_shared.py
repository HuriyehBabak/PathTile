from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from kernels.shared import (
    DepthwiseConvShared,
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
    Match the custom kernel convention exactly.

    Custom forward:
        xi = t + j - K//2

    For even K, PyTorch padding=K//2 would produce L+1 outputs.
    Therefore use asymmetric explicit padding:

        left  = K//2
        right = K - 1 - K//2
    """

    K = kernel.shape[1]

    pad_left = K // 2
    pad_right = K - 1 - pad_left

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

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    reset_timing_collectors()

    # ========================================================
    # Custom tensors
    # ========================================================

    x_custom = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    k_custom = torch.randn(
        H,
        K,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    grad_y = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=torch.float32,
    )

    # ========================================================
    # Reference tensors
    # ========================================================

    x_ref = (
        x_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    k_ref = (
        k_custom
        .detach()
        .clone()
        .requires_grad_(True)
    )

    # ========================================================
    # Forward
    # ========================================================

    y_custom = DepthwiseConvShared.apply(
        x_custom,
        k_custom,
    )

    y_ref = reference_depthwise_conv(
        x_ref,
        k_ref,
    )

    torch.cuda.synchronize()

    fwd_diff = (
        y_custom - y_ref
    ).abs().max().item()

    fwd_ok = torch.allclose(
        y_custom,
        y_ref,
        atol=ATOL,
        rtol=RTOL,
    )

    # ========================================================
    # Backward
    # ========================================================

    y_custom.backward(
        grad_y
    )

    y_ref.backward(
        grad_y
    )

    torch.cuda.synchronize()

    gx_diff = (
        x_custom.grad
        - x_ref.grad
    ).abs().max().item()

    gk_diff = (
        k_custom.grad
        - k_ref.grad
    ).abs().max().item()

    gx_ok = torch.allclose(
        x_custom.grad,
        x_ref.grad,
        atol=ATOL,
        rtol=RTOL,
    )

    gk_ok = torch.allclose(
        k_custom.grad,
        k_ref.grad,
        atol=ATOL,
        rtol=RTOL,
    )

    # Relative scale for the long FP32 weight-gradient reduction.
    gk_ref_max = (
        k_ref.grad
        .abs()
        .max()
        .item()
    )

    gk_rel = (
        gk_diff
        / (gk_ref_max + 1e-12)
    )

    overall = (
        fwd_ok
        and gx_ok
        and gk_ok
    )

    print("-" * 72)

    print(
        f"B={B}, H={H}, L={L}, K={K}"
    )

    print(
        f"Forward:         "
        f"{'PASS' if fwd_ok else 'FAIL'} "
        f"| max abs = {fwd_diff:.6e}"
    )

    print(
        f"Backward Input:  "
        f"{'PASS' if gx_ok else 'FAIL'} "
        f"| max abs = {gx_diff:.6e}"
    )

    print(
        f"Backward Kernel: "
        f"{'PASS' if gk_ok else 'FAIL'} "
        f"| max abs = {gk_diff:.6e} "
        f"| rel-scale = {gk_rel:.6e}"
    )

    print(
        f"OVERALL:         "
        f"{'PASS' if overall else 'FAIL'}"
    )

    return bool(overall)


def quick_validation() -> bool:

    print("=" * 72)
    print("SHARED MEMORY — QUICK CORRECTNESS")
    print("=" * 72)

    cases = [
        # odd K
        (2, 4, 48, 47),

        # exact paper kernel size
        (2, 4, 48, 48),

        # odd K on other side
        (2, 4, 48, 49),

        # somewhat larger channel count
        (8, 16, 48, 48),
    ]

    results = []

    for case in cases:
        results.append(
            run_case(*case)
        )

    print("=" * 72)

    overall = all(results)

    print(
        "QUICK VALIDATION:",
        "PASS" if overall else "FAIL",
    )

    print("=" * 72)

    return overall


def full_validation() -> bool:

    print("=" * 72)
    print("SHARED MEMORY — FULL PAPER WORKLOAD")
    print("=" * 72)

    result = run_case(
        B=16384,
        H=128,
        L=48,
        K=48,
    )

    print("=" * 72)

    print(
        "FULL VALIDATION:",
        "PASS" if result else "FAIL",
    )

    print("=" * 72)

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full B=16384,H=128,L=48,K=48 validation",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    gpu = torch.cuda.get_device_name(0)

    print(
        f"GPU: {gpu}"
    )

    if args.full:
        ok = full_validation()
    else:
        ok = quick_validation()

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
