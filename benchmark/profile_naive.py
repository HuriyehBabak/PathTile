from __future__ import annotations

import argparse

import torch

from kernels.naive import (
    DepthwiseConvNaive,
    clear_timings,
)


B = 16384
H = 128
L = 48
K = 48

DTYPE = torch.float32
TPB = 512
SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=[
            "forward",
            "bwd-input",
            "bwd-kernel",
        ],
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. "
            "Run this inside a SLURM GPU job."
        )

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    DepthwiseConvNaive.THREADS_PER_BLOCK = TPB

    print("GPU:", torch.cuda.get_device_name(0))
    print(f"B={B}, H={H}, L={L}, K={K}, TPB={TPB}")
    print("Phase:", args.phase)

    x = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )

    kernel = torch.randn(
        H,
        K,
        device="cuda",
        dtype=DTYPE,
        requires_grad=True,
    )

    grad_y = torch.randn(
        B,
        H,
        L,
        device="cuda",
        dtype=DTYPE,
    )

    # --------------------------------------------------------
    # Warm-up / compile
    # --------------------------------------------------------

    y = DepthwiseConvNaive.apply(x, kernel)
    y.backward(grad_y)

    torch.cuda.synchronize()

    x.grad = None
    kernel.grad = None

    clear_timings()

    # --------------------------------------------------------
    # Target launch
    # --------------------------------------------------------

    if args.phase == "forward":
        y = DepthwiseConvNaive.apply(x, kernel)
        torch.cuda.synchronize()

    elif args.phase == "bwd-input":
        y = DepthwiseConvNaive.apply(x, kernel)

        grad_x = torch.autograd.grad(
            y,
            x,
            grad_outputs=grad_y,
            retain_graph=False,
            create_graph=False,
        )[0]

        torch.cuda.synchronize()

        del grad_x

    elif args.phase == "bwd-kernel":
        y = DepthwiseConvNaive.apply(x, kernel)

        grad_kernel = torch.autograd.grad(
            y,
            kernel,
            grad_outputs=grad_y,
            retain_graph=False,
            create_graph=False,
        )[0]

        torch.cuda.synchronize()

        del grad_kernel

    print("Profile workload complete.")


if __name__ == "__main__":
    main()