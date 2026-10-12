"""TVM-TIRx kernels for PathTile's strict-FP32 depthwise Conv1D workload.

The operator is cross-correlation with explicit same padding::

    y[b, h, t] = sum_k x[b, h, t + k - floor(K / 2)] * weight[h, k]

Out-of-range input positions are zero and the output length is exactly ``L``.
The forward and input-gradient paths use one CUDA thread per padded time slot.
The weight-gradient path mirrors the Triton comparison: deterministic partial
reductions followed by a second reduction kernel.  Candidate compilation and
selection are intentionally exposed so the benchmark can keep them outside the
measured CUDA events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tvm

try:
    # TVM v0.25's new device-kernel dialect is exported as ``tirx``.
    from tvm.script import tirx as T
except ImportError:  # Compatibility with older TVM releases.
    from tvm.script import tir as T


MAX_L = 64
MAX_K = 64
DK_BLOCK_K = 4
DK_N_PER_CTA = 4096
FWD_BLOCK_BH_CANDIDATES = (1, 2, 4, 8)
DK_THREADS_CANDIDATES = (64, 128, 256)
DK_REDUCE_THREADS = 256


@dataclass(frozen=True)
class CompiledKernel:
    """One compiled CUDA candidate and the configuration that produced it."""

    phase: str
    name: str
    config: dict[str, int]
    executable: Any
    cuda_source_chars: int

    def __call__(self, *args: Any) -> Any:
        return self.executable(*args)


@dataclass(frozen=True)
class CompiledWeightKernel:
    """A two-launch dK candidate (partial accumulation plus final reduction)."""

    phase: str
    name: str
    config: dict[str, int]
    partial: CompiledKernel
    reduce: CompiledKernel

    def __call__(self, grad_y: Any, x: Any, workspace: Any, grad_weight: Any) -> None:
        self.partial(grad_y, x, workspace)
        self.reduce(workspace, grad_weight)


@dataclass(frozen=True)
class CompiledCandidateSet:
    """All candidates for one static ``(B,H,L,K)`` specialization."""

    B: int
    H: int
    L: int
    K: int
    target: str
    forward: tuple[CompiledKernel, ...]
    backward_input: tuple[CompiledKernel, ...]
    backward_weight: tuple[CompiledWeightKernel, ...]

    @property
    def workspace_shape(self) -> tuple[int, int, int]:
        return backward_weight_workspace_shape(self.B, self.H, self.L, self.K)


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _check_shape(B: int, H: int, L: int, K: int) -> None:
    if min(B, H, L, K) <= 0:
        raise ValueError("B, H, L, and K must be positive")
    if L > MAX_L or K > MAX_K:
        raise ValueError(f"This baseline requires L<={MAX_L} and K<={MAX_K}")


def backward_weight_workspace_shape(
    B: int,
    H: int,
    L: int,
    K: int,
) -> tuple[int, int, int]:
    """Return ``[H,K,num_partials]`` for the deterministic dK workspace."""

    _check_shape(B, H, L, K)
    return H, K, _ceil_div(B * L, DK_N_PER_CTA)


def _compile_source(source: str, phase: str, name: str, config: dict[str, int], target: tvm.target.Target) -> CompiledKernel:
    parsed = tvm.script.from_source(source, extra_vars={"T": T})
    ir_module = parsed if isinstance(parsed, tvm.IRModule) else tvm.IRModule({"main": parsed})
    compiled_artifact = tvm.compile(
        ir_module,
        target=target,
        tir_pipeline="tirx",
    )
    jit = getattr(compiled_artifact, "jit", None)
    runtime_module = jit() if callable(jit) else compiled_artifact

    entry: Any = runtime_module
    if not callable(entry):
        try:
            entry = runtime_module["main"]
        except (KeyError, TypeError, AttributeError) as error:
            raise RuntimeError(
                f"Could not resolve callable entry point for {name}"
            ) from error

    cuda_source = ""
    source_candidates = [runtime_module]
    source_candidates.extend(list(getattr(runtime_module, "imported_modules", ())))
    for source_module in source_candidates:
        inspect_source = getattr(source_module, "inspect_source", None)
        get_source = getattr(source_module, "get_source", None)
        candidate_source = ""
        try:
            if callable(inspect_source):
                candidate_source = str(inspect_source())
            elif callable(get_source):
                candidate_source = str(get_source())
        except (AttributeError, RuntimeError, TypeError):
            continue
        if "__global__" in candidate_source:
            cuda_source = candidate_source
            break
    return CompiledKernel(
        phase=phase,
        name=name,
        config=dict(config),
        executable=entry,
        cuda_source_chars=len(cuda_source),
    )


def _pointwise_source(
    *,
    B: int,
    H: int,
    L: int,
    K: int,
    block_bh: int,
    backward_input: bool,
) -> str:
    block_t = _next_power_of_two(L)
    # Match Triton's candidate launch widths: BLOCK_BH={1,2,4,8} maps to
    # {1,2,4,8} warps.  For the main L=48 specialization, every thread owns
    # two positions from the BLOCK_BH x 64 logical tile.
    threads = block_bh * 32
    tile_elements = block_bh * block_t
    items_per_thread = _ceil_div(tile_elements, threads)
    n_bh = B * H
    grid = _ceil_div(n_bh, block_bh)
    if threads > 1024:
        raise ValueError(f"Invalid CUDA block size: {threads}")

    phase_name = "bwd_input" if backward_input else "fwd"
    input_name = "grad_y_ptr" if backward_input else "x_ptr"
    input_buffer = "GradY" if backward_input else "X"
    output_name = "grad_x_ptr" if backward_input else "y_ptr"
    output_buffer = "GradX" if backward_input else "Y"
    input_position = (
        f"t - kk + {K // 2}"
        if backward_input
        else f"t + kk - {K // 2}"
    )

    return f'''\
@T.prim_func
def tvm_{phase_name}_bh{block_bh}({input_name}: T.handle, weight_ptr: T.handle, {output_name}: T.handle):
    T.func_attr({{"global_symbol": "main"}})
    {input_buffer} = T.match_buffer({input_name}, ({B}, {H}, {L}), "float32")
    Weight = T.match_buffer(weight_ptr, ({H}, {K}), "float32")
    {output_buffer} = T.match_buffer({output_name}, ({B}, {H}, {L}), "float32")
    T.device_entry()
    acc = T.alloc_local((1,), "float32")
    block = T.cta_id([{grid}])
    tx = T.thread_id([{threads}])
    for item in range({items_per_thread}):
        local = item * {threads} + tx
        local_bh = local // {block_t}
        t = local - local_bh * {block_t}
        bh = block * {block_bh} + local_bh
        b = bh // {H}
        h = bh - b * {H}
        acc[0] = T.float32(0)
        if local < {tile_elements}:
            if bh < {n_bh}:
                if t < {L}:
                    for kk in range({K}):
                        input_t = {input_position}
                        if input_t >= 0:
                            if input_t < {L}:
                                acc[0] = acc[0] + {input_buffer}[b, h, input_t] * Weight[h, kk]
                    {output_buffer}[b, h, t] = acc[0]
'''


def _dk_partial_source(
    *,
    B: int,
    H: int,
    L: int,
    K: int,
    threads: int,
) -> str:
    num_partials = _ceil_div(B * L, DK_N_PER_CTA)
    k_blocks = _ceil_div(K, DK_BLOCK_K)
    grid = H * k_blocks * num_partials
    iterations = DK_N_PER_CTA // threads
    if DK_N_PER_CTA % threads != 0:
        raise ValueError("DK_N_PER_CTA must be divisible by candidate threads")

    reduction_lines: list[str] = []
    stride = threads // 2
    while stride > 0:
        reduction_lines.extend(
            (
                f"    if tx < {stride}:",
                f"        shared0[tx] = shared0[tx] + shared0[tx + {stride}]",
                f"        shared1[tx] = shared1[tx] + shared1[tx + {stride}]",
                f"        shared2[tx] = shared2[tx] + shared2[tx + {stride}]",
                f"        shared3[tx] = shared3[tx] + shared3[tx + {stride}]",
                "    T.cuda.cta_sync()",
            )
        )
        stride //= 2
    reduction = "\n".join(reduction_lines)

    return f'''\
@T.prim_func
def tvm_bwd_weight_partial_t{threads}(grad_y_ptr: T.handle, x_ptr: T.handle, partial_ptr: T.handle):
    T.func_attr({{"global_symbol": "main"}})
    GradY = T.match_buffer(grad_y_ptr, ({B}, {H}, {L}), "float32")
    X = T.match_buffer(x_ptr, ({B}, {H}, {L}), "float32")
    Partial = T.match_buffer(partial_ptr, ({H}, {K}, {num_partials}), "float32")
    T.device_entry()
    shared0 = T.alloc_shared(({threads},), "float32")
    shared1 = T.alloc_shared(({threads},), "float32")
    shared2 = T.alloc_shared(({threads},), "float32")
    shared3 = T.alloc_shared(({threads},), "float32")
    acc0 = T.alloc_local((1,), "float32")
    acc1 = T.alloc_local((1,), "float32")
    acc2 = T.alloc_local((1,), "float32")
    acc3 = T.alloc_local((1,), "float32")
    block = T.cta_id([{grid}])
    tx = T.thread_id([{threads}])
    pid_hk = block // {num_partials}
    pid_part = block - pid_hk * {num_partials}
    h = pid_hk // {k_blocks}
    k_block = pid_hk - h * {k_blocks}
    k0 = k_block * {DK_BLOCK_K}
    acc0[0] = T.float32(0)
    acc1[0] = T.float32(0)
    acc2[0] = T.float32(0)
    acc3[0] = T.float32(0)
    for iteration in range({iterations}):
        n = pid_part * {DK_N_PER_CTA} + iteration * {threads} + tx
        if n < {B * L}:
            b = n // {L}
            t = n - b * {L}
            gy_value = GradY[b, h, t]
            input_t0 = t + k0 - {K // 2}
            if k0 < {K}:
                if input_t0 >= 0:
                    if input_t0 < {L}:
                        acc0[0] = acc0[0] + gy_value * X[b, h, input_t0]
            input_t1 = t + k0 + 1 - {K // 2}
            if k0 + 1 < {K}:
                if input_t1 >= 0:
                    if input_t1 < {L}:
                        acc1[0] = acc1[0] + gy_value * X[b, h, input_t1]
            input_t2 = t + k0 + 2 - {K // 2}
            if k0 + 2 < {K}:
                if input_t2 >= 0:
                    if input_t2 < {L}:
                        acc2[0] = acc2[0] + gy_value * X[b, h, input_t2]
            input_t3 = t + k0 + 3 - {K // 2}
            if k0 + 3 < {K}:
                if input_t3 >= 0:
                    if input_t3 < {L}:
                        acc3[0] = acc3[0] + gy_value * X[b, h, input_t3]
    shared0[tx] = acc0[0]
    shared1[tx] = acc1[0]
    shared2[tx] = acc2[0]
    shared3[tx] = acc3[0]
    T.cuda.cta_sync()
{reduction}
    if tx == 0:
        if k0 < {K}:
            Partial[h, k0, pid_part] = shared0[0]
        if k0 + 1 < {K}:
            Partial[h, k0 + 1, pid_part] = shared1[0]
        if k0 + 2 < {K}:
            Partial[h, k0 + 2, pid_part] = shared2[0]
        if k0 + 3 < {K}:
            Partial[h, k0 + 3, pid_part] = shared3[0]
'''


def _dk_reduce_source(*, H: int, K: int, num_partials: int) -> str:
    rounds = _ceil_div(num_partials, DK_REDUCE_THREADS)
    grid = H * K
    reduction_lines: list[str] = []
    stride = DK_REDUCE_THREADS // 2
    while stride > 0:
        reduction_lines.extend(
            (
                f"    if tx < {stride}:",
                f"        shared[tx] = shared[tx] + shared[tx + {stride}]",
                "    T.cuda.cta_sync()",
            )
        )
        stride //= 2
    reduction = "\n".join(reduction_lines)
    return f'''\
@T.prim_func
def tvm_bwd_weight_reduce(partial_ptr: T.handle, grad_weight_ptr: T.handle):
    T.func_attr({{"global_symbol": "main"}})
    Partial = T.match_buffer(partial_ptr, ({H}, {K}, {num_partials}), "float32")
    GradWeight = T.match_buffer(grad_weight_ptr, ({H}, {K}), "float32")
    T.device_entry()
    shared = T.alloc_shared(({DK_REDUCE_THREADS},), "float32")
    acc = T.alloc_local((1,), "float32")
    block = T.cta_id([{grid}])
    tx = T.thread_id([{DK_REDUCE_THREADS}])
    h = block // {K}
    k = block - h * {K}
    acc[0] = T.float32(0)
    for iteration in range({rounds}):
        partial_index = iteration * {DK_REDUCE_THREADS} + tx
        if partial_index < {num_partials}:
            acc[0] = acc[0] + Partial[h, k, partial_index]
    shared[tx] = acc[0]
    T.cuda.cta_sync()
{reduction}
    if tx == 0:
        GradWeight[h, k] = shared[0]
'''


def compile_candidates(
    B: int,
    H: int,
    L: int,
    K: int,
    target: Any = None,
) -> CompiledCandidateSet:
    """Compile the full static candidate family for one shape.

    This function does no timing.  The caller must compile all candidates and
    select among them before the final warm-up and measured CUDA events.
    """

    _check_shape(B, H, L, K)
    if target is None:
        target = {"kind": "cuda"}
    target_object = (
        target
        if isinstance(target, tvm.target.Target)
        else tvm.target.Target(target)
    )

    forward: list[CompiledKernel] = []
    backward_input: list[CompiledKernel] = []
    for block_bh in FWD_BLOCK_BH_CANDIDATES:
        threads = block_bh * 32
        config = {
            "BLOCK_BH": block_bh,
            "num_warps": block_bh,
            "threads": threads,
        }
        forward.append(
            _compile_source(
                _pointwise_source(
                    B=B,
                    H=H,
                    L=L,
                    K=K,
                    block_bh=block_bh,
                    backward_input=False,
                ),
                "fwd",
                f"fwd_bh{block_bh}",
                config,
                target_object,
            )
        )
        backward_input.append(
            _compile_source(
                _pointwise_source(
                    B=B,
                    H=H,
                    L=L,
                    K=K,
                    block_bh=block_bh,
                    backward_input=True,
                ),
                "bwd_input",
                f"bwd_input_bh{block_bh}",
                config,
                target_object,
            )
        )

    num_partials = _ceil_div(B * L, DK_N_PER_CTA)
    reduce_kernel = _compile_source(
        _dk_reduce_source(H=H, K=K, num_partials=num_partials),
        "bwd_weight_reduce",
        "bwd_weight_reduce",
        {
            "BLOCK_K": DK_BLOCK_K,
            "NUM_PARTIALS": num_partials,
            "threads": DK_REDUCE_THREADS,
        },
        target_object,
    )
    backward_weight: list[CompiledWeightKernel] = []
    for threads in DK_THREADS_CANDIDATES:
        config = {
            "BLOCK_K": DK_BLOCK_K,
            "N_PER_CTA": DK_N_PER_CTA,
            "BLOCK_N": threads,
            "num_warps": threads // 32,
            "partial_threads": threads,
            "reduce_threads": DK_REDUCE_THREADS,
        }
        partial_kernel = _compile_source(
            _dk_partial_source(B=B, H=H, L=L, K=K, threads=threads),
            "bwd_weight_partial",
            f"bwd_weight_partial_t{threads}",
            config,
            target_object,
        )
        backward_weight.append(
            CompiledWeightKernel(
                phase="bwd_weight",
                name=f"bwd_weight_t{threads}",
                config=config,
                partial=partial_kernel,
                reduce=reduce_kernel,
            )
        )

    return CompiledCandidateSet(
        B=B,
        H=H,
        L=L,
        K=K,
        target=str(target_object),
        forward=tuple(forward),
        backward_input=tuple(backward_input),
        backward_weight=tuple(backward_weight),
    )
