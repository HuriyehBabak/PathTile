"""Triton baseline for the short depthwise Conv1D operator used in PathTile.

The operator follows the exact semantics used by the CUDA and PyTorch
implementations in this artifact:

    y[b, h, t] = sum_k x[b, h, t + k - floor(K / 2)] * w[h, k]

Out-of-range input positions are zero.  This is cross-correlation with explicit
``same`` padding (left = K // 2) and an output length equal to L.

This module intentionally starts with a strict FP32 implementation.  Mixed
precision is evaluated separately so the external-baseline comparison does not
silently change numerical semantics.
"""

import torch
import triton
import triton.language as tl


MAX_L = 64
MAX_K = 64
DK_BLOCK_K = 4
DK_BLOCK_N_PER_PROGRAM = 4096


_FWD_CONFIGS = [
    triton.Config({"BLOCK_BH": 1}, num_warps=1, num_stages=1),
    triton.Config({"BLOCK_BH": 2}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_BH": 4}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_BH": 8}, num_warps=8, num_stages=1),
]


_DK_PARTIAL_CONFIGS = [
    triton.Config({"BLOCK_N": 64}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_N": 256}, num_warps=8, num_stages=1),
]


@triton.autotune(configs=_FWD_CONFIGS, key=["N_BH", "H", "L", "K"])
@triton.jit
def _depthwise_fwd_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    N_BH: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_BH: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_bh = pid * BLOCK_BH + tl.arange(0, BLOCK_BH)
    offs_t = tl.arange(0, BLOCK_T)

    valid_bh = offs_bh < N_BH
    valid_t = offs_t < L
    h = offs_bh % H
    base = offs_bh[:, None] * L
    pad = K // 2

    acc = tl.zeros((BLOCK_BH, BLOCK_T), dtype=tl.float32)

    for kk in range(0, K):
        xi = offs_t + kk - pad
        valid_x = (xi >= 0) & (xi < L)
        xv = tl.load(
            x_ptr + base + xi[None, :],
            mask=valid_bh[:, None] & valid_t[None, :] & valid_x[None, :],
            other=0.0,
        )
        wv = tl.load(
            w_ptr + h * K + kk,
            mask=valid_bh,
            other=0.0,
        )
        acc += xv.to(tl.float32) * wv[:, None].to(tl.float32)

    tl.store(
        y_ptr + base + offs_t[None, :],
        acc,
        mask=valid_bh[:, None] & valid_t[None, :],
    )


@triton.autotune(configs=_FWD_CONFIGS, key=["N_BH", "H", "L", "K"])
@triton.jit
def _depthwise_bwd_input_kernel(
    gy_ptr,
    w_ptr,
    gx_ptr,
    N_BH: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_BH: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_bh = pid * BLOCK_BH + tl.arange(0, BLOCK_BH)
    offs_t = tl.arange(0, BLOCK_T)

    valid_bh = offs_bh < N_BH
    valid_t = offs_t < L
    h = offs_bh % H
    base = offs_bh[:, None] * L
    pad = K // 2

    acc = tl.zeros((BLOCK_BH, BLOCK_T), dtype=tl.float32)

    for kk in range(0, K):
        tout = offs_t - kk + pad
        valid_gy = (tout >= 0) & (tout < L)
        gyv = tl.load(
            gy_ptr + base + tout[None, :],
            mask=valid_bh[:, None] & valid_t[None, :] & valid_gy[None, :],
            other=0.0,
        )
        wv = tl.load(
            w_ptr + h * K + kk,
            mask=valid_bh,
            other=0.0,
        )
        acc += gyv.to(tl.float32) * wv[:, None].to(tl.float32)

    tl.store(
        gx_ptr + base + offs_t[None, :],
        acc,
        mask=valid_bh[:, None] & valid_t[None, :],
    )


@triton.autotune(configs=_DK_PARTIAL_CONFIGS, key=["B", "H", "L", "K"])
@triton.jit
def _depthwise_bwd_weight_partial_kernel(
    gy_ptr,
    x_ptr,
    partial_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    L: tl.constexpr,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_PER_PROGRAM: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_hk = tl.program_id(0)
    pid_part = tl.program_id(1)

    k_blocks = tl.cdiv(K, BLOCK_K)
    h = pid_hk // k_blocks
    k_block = pid_hk - h * k_blocks
    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    valid_k = offs_k < K
    pad = K // 2

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    lane_n = tl.arange(0, BLOCK_N)
    part_start = pid_part * N_PER_PROGRAM

    for block_offset in range(0, N_PER_PROGRAM, BLOCK_N):
        offs_n = part_start + block_offset + lane_n
        valid_n = offs_n < B * L
        b = offs_n // L
        t = offs_n - b * L

        gyv = tl.load(
            gy_ptr + (b * H + h) * L + t,
            mask=valid_n,
            other=0.0,
        ).to(tl.float32)

        xi = t[None, :] + offs_k[:, None] - pad
        valid_x = (
            valid_k[:, None]
            & valid_n[None, :]
            & (xi >= 0)
            & (xi < L)
        )
        xv = tl.load(
            x_ptr + (b[None, :] * H + h) * L + xi,
            mask=valid_x,
            other=0.0,
        ).to(tl.float32)

        acc += tl.sum(xv * gyv[None, :], axis=1)

    tl.store(
        partial_ptr + (h * K + offs_k) * NUM_PARTIALS + pid_part,
        acc,
        mask=valid_k,
    )


@triton.jit
def _depthwise_bwd_weight_reduce_kernel(
    partial_ptr,
    gk_ptr,
    H: tl.constexpr,
    K: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_hk = tl.program_id(0)
    k_blocks = tl.cdiv(K, BLOCK_K)
    h = pid_hk // k_blocks
    k_block = pid_hk - h * k_blocks

    offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_p = tl.arange(0, BLOCK_P)
    valid_k = offs_k < K
    valid_p = offs_p < NUM_PARTIALS

    values = tl.load(
        partial_ptr
        + (h * K + offs_k[:, None]) * NUM_PARTIALS
        + offs_p[None, :],
        mask=valid_k[:, None] & valid_p[None, :],
        other=0.0,
    )
    result = tl.sum(values.to(tl.float32), axis=1)
    tl.store(gk_ptr + h * K + offs_k, result, mask=valid_k)


def _check_supported_device() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError(
            "Triton 3.6 supports NVIDIA compute capability 8.0 or newer; "
            f"the active device is sm_{major}{minor}"
        )


def _check_x_weight(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int, int, int]:
    _check_supported_device()
    if not x.is_cuda or not weight.is_cuda:
        raise ValueError("x and weight must be CUDA tensors")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("The strict baseline supports FP32 only")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("x and weight must be contiguous")
    if x.ndim != 3 or weight.ndim != 2:
        raise ValueError("Expected x[B,H,L] and weight[H,K]")

    B, H, L = map(int, x.shape)
    H_weight, K = map(int, weight.shape)
    if H != H_weight:
        raise ValueError("Input and weight channel dimensions do not match")
    if B <= 0 or H <= 0 or L <= 0 or K <= 0:
        raise ValueError("All dimensions must be positive")
    if L > MAX_L or K > MAX_K:
        raise ValueError(f"This baseline requires L<={MAX_L} and K<={MAX_K}")
    return B, H, L, K


def _check_same_shape(a: torch.Tensor, b: torch.Tensor, names: str) -> None:
    if a.shape != b.shape:
        raise ValueError(f"{names} must have identical shapes")
    if a.dtype != b.dtype or a.device != b.device:
        raise ValueError(f"{names} must have identical dtype and device")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError(f"{names} must be contiguous")


def backward_weight_workspace_shape(B: int, H: int, L: int, K: int) -> tuple[int, int, int]:
    num_partials = triton.cdiv(B * L, DK_BLOCK_N_PER_PROGRAM)
    return H, K, num_partials


def forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, L, K = _check_x_weight(x, weight)
    if out is None:
        out = torch.empty_like(x)
    _check_same_shape(x, out, "x and out")

    n_bh = B * H
    block_t = triton.next_power_of_2(L)
    grid = lambda meta: (triton.cdiv(n_bh, meta["BLOCK_BH"]),)
    _depthwise_fwd_kernel[grid](
        x,
        weight,
        out,
        N_BH=n_bh,
        H=H,
        L=L,
        K=K,
        BLOCK_T=block_t,
    )
    return out


def backward_input(
    grad_y: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    B, H, L, K = _check_x_weight(grad_y, weight)
    if out is None:
        out = torch.empty_like(grad_y)
    _check_same_shape(grad_y, out, "grad_y and out")

    n_bh = B * H
    block_t = triton.next_power_of_2(L)
    grid = lambda meta: (triton.cdiv(n_bh, meta["BLOCK_BH"]),)
    _depthwise_bwd_input_kernel[grid](
        grad_y,
        weight,
        out,
        N_BH=n_bh,
        H=H,
        L=L,
        K=K,
        BLOCK_T=block_t,
    )
    return out


def backward_weight(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    K: int,
    out: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if grad_y.ndim != 3 or x.ndim != 3:
        raise ValueError("Expected grad_y[B,H,L] and x[B,H,L]")
    _check_same_shape(grad_y, x, "grad_y and x")
    B, H, L = map(int, x.shape)
    _check_supported_device()
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError("grad_y and x must be CUDA FP32 tensors")
    if B <= 0 or H <= 0 or L <= 0 or K <= 0:
        raise ValueError("All dimensions must be positive")
    if L > MAX_L or K > MAX_K:
        raise ValueError(f"This baseline requires L<={MAX_L} and K<={MAX_K}")

    if out is None:
        out = torch.empty((H, K), device=x.device, dtype=torch.float32)
    if (
        out.shape != (H, K)
        or out.dtype != torch.float32
        or out.device != x.device
        or not out.is_contiguous()
    ):
        raise ValueError("out must be a contiguous FP32 tensor with shape [H,K]")

    expected_workspace_shape = backward_weight_workspace_shape(B, H, L, K)
    if workspace is None:
        workspace = torch.empty(
            expected_workspace_shape,
            device=x.device,
            dtype=torch.float32,
        )
    if (
        tuple(workspace.shape) != expected_workspace_shape
        or workspace.dtype != torch.float32
        or workspace.device != x.device
        or not workspace.is_contiguous()
    ):
        raise ValueError(
            "workspace must be a contiguous FP32 tensor with shape "
            f"{expected_workspace_shape}"
        )

    num_partials = expected_workspace_shape[2]
    k_blocks = triton.cdiv(K, DK_BLOCK_K)
    partial_grid = (H * k_blocks, num_partials)
    _depthwise_bwd_weight_partial_kernel[partial_grid](
        grad_y,
        x,
        workspace,
        B=B,
        H=H,
        L=L,
        K=K,
        NUM_PARTIALS=num_partials,
        BLOCK_K=DK_BLOCK_K,
        N_PER_PROGRAM=DK_BLOCK_N_PER_PROGRAM,
    )

    reduce_grid = (H * k_blocks,)
    _depthwise_bwd_weight_reduce_kernel[reduce_grid](
        workspace,
        out,
        H=H,
        K=K,
        NUM_PARTIALS=num_partials,
        BLOCK_K=DK_BLOCK_K,
        BLOCK_P=triton.next_power_of_2(num_partials),
        num_warps=4,
        num_stages=1,
    )
    return out, workspace


class _DepthwiseConv1dTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_c = x.contiguous()
        weight_c = weight.contiguous()
        y = forward(x_c, weight_c)
        ctx.save_for_backward(x_c, weight_c)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x, weight = ctx.saved_tensors
        grad_y_c = grad_y.contiguous()
        grad_x = backward_input(grad_y_c, weight)
        grad_weight, _ = backward_weight(grad_y_c, x, int(weight.shape[1]))
        return grad_x, grad_weight


def depthwise_conv1d_same(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Autograd-enabled strict-FP32 Triton baseline."""

    return _DepthwiseConv1dTriton.apply(x, weight)


def _config_to_dict(kernel) -> dict[str, object] | None:
    config = getattr(kernel, "best_config", None)
    if config is None:
        return None
    return {
        **dict(getattr(config, "kwargs", {})),
        "num_warps": getattr(config, "num_warps", None),
        "num_stages": getattr(config, "num_stages", None),
    }


def selected_configs() -> dict[str, dict[str, object] | None]:
    """Return the most recently selected autotune configuration per path."""

    return {
        "forward": _config_to_dict(_depthwise_fwd_kernel),
        "backward_input": _config_to_dict(_depthwise_bwd_input_kernel),
        "backward_weight_partial": _config_to_dict(
            _depthwise_bwd_weight_partial_kernel
        ),
        "backward_weight_reduce": {
            "BLOCK_K": DK_BLOCK_K,
            "N_PER_PROGRAM": DK_BLOCK_N_PER_PROGRAM,
            "num_warps": 4,
            "num_stages": 1,
        },
    }
