"""Create an independent PyTorch/cuDNN reference for the TVM baseline.

This helper is intentionally executed with the project's existing PyTorch
environment, while ``benchmark_tvm_main.py`` runs in the isolated TVM Python
3.11 environment.  Inputs and results cross the process boundary as NPZ files;
the helper itself is never part of a timed region.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def depthwise_same(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    channels, kernel_size = map(int, weight.shape)
    left = kernel_size // 2
    right = kernel_size - 1 - left
    return F.conv1d(
        F.pad(x, (left, right)),
        weight.unsqueeze(1),
        bias=None,
        stride=1,
        padding=0,
        groups=channels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for the reference but is unavailable")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    with np.load(args.input) as arrays:
        x_numpy = np.ascontiguousarray(arrays["x"], dtype=np.float32)
        weight_numpy = np.ascontiguousarray(arrays["weight"], dtype=np.float32)
        grad_y_numpy = np.ascontiguousarray(arrays["grad_y"], dtype=np.float32)

    x = torch.from_numpy(x_numpy).to(args.device).requires_grad_(True)
    weight = torch.from_numpy(weight_numpy).to(args.device).requires_grad_(True)
    grad_y = torch.from_numpy(grad_y_numpy).to(args.device)

    y = depthwise_same(x, weight)
    grad_x, grad_weight = torch.autograd.grad(
        y,
        (x, weight),
        grad_outputs=grad_y,
        retain_graph=False,
        create_graph=False,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        y=y.detach().cpu().numpy(),
        grad_x=grad_x.detach().cpu().numpy(),
        grad_weight=grad_weight.detach().cpu().numpy(),
    )


if __name__ == "__main__":
    main()
