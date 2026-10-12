#!/usr/bin/env python3

"""Volta/Ampere implementation of the S4 depthwise-convolution experiment.

The script has three independent modes:

* ``validate``: report kernel resources/estimated occupancy and check numerical

  correctness against PyTorch.

* ``benchmark``: validate, then measure the three custom CUDA kernels with

  warm-up and repeated CUDA-event timing.

* ``train``: validate, then train the four-block S4 model and export steady-

  state measurements for epochs 2..N.

The CUDA mapping is specialized for the paper workload L <= 48 and K <= 48:

* Forward and input-gradient: one block owns several batches of one channel;

  one warp owns one (batch, channel) sequence; one lane accumulates output t

  and, when present, t+32 in registers in the same K loop.

* Weight-gradient: one block owns one channel and one batch tile; x and gy are

  staged in shared memory; each thread accumulates four neighboring taps;

  warp shuffles and one final warp reduce produce one block partial per tap.

  Block partials are written to a global workspace and reduced by a second

  kernel. The second stage assigns one warp to one (channel, tap) coefficient,

  eliminating cross-warp synchronization in the final reduction. No atomic

  operations are used.

Hardware limits are selected automatically for NVIDIA Tesla V100-SXM3-32GB

(compute capability 7.0), A100 (compute capability 8.0), and RTX A6000

(compute capability 8.6). Controlled sweeps on the two Ampere paper GPUs

selected the common final launch defaults WARPS_PER_BLOCK=16 and

BWDK_BLOCK=128. Those defaults are retained on Volta as an explicit starting

point, while the corresponding CLI options remain available for architecture-

specific sensitivity studies and reproducibility.

"""

from __future__ import annotations

import argparse

import gc

import math

import threading

import time

import warnings

from contextlib import contextmanager

from dataclasses import dataclass

from pathlib import Path

from typing import Any, Callable, Iterator

import numpy as np

import pandas as pd

import pycuda.driver as cuda

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from pycuda.compiler import SourceModule

from torch.utils.data import DataLoader

try:

    import pynvml

except ImportError:  # NVML is optional for validate/benchmark.

    pynvml = None



# =============================================================================

# Paper workload and supported GPU hardware profiles

# =============================================================================

WARP_SIZE = 32

MAX_L = 48

MAX_K = 48

DEFAULT_B = 16_384

DEFAULT_H = 128

DEFAULT_L = 48

DEFAULT_K = 48

DEFAULT_WARPS_PER_BLOCK = 16

DEFAULT_BWDK_BTILE = 64

DEFAULT_BWDK_KPACK = 4

DEFAULT_BWDK_BLOCK = 128

@dataclass(frozen=True)

class GPUProfile:

    key: str

    display_name: str

    compute_capability: tuple[int, int]

    peak_bandwidth_gbps: float

    max_threads_per_sm: int

    max_blocks_per_sm: int

    max_regs_per_sm: int

    max_smem_per_sm: int

    max_warps_per_sm: int

    @property

    def limits(self) -> dict[str, int]:

        return {

            "max_threads_per_sm": self.max_threads_per_sm,

            "max_blocks_per_sm": self.max_blocks_per_sm,

            "max_regs_per_sm": self.max_regs_per_sm,

            "max_smem_per_sm": self.max_smem_per_sm,

            "max_warps_per_sm": self.max_warps_per_sm,

        }



GPU_PROFILES = {

    (7, 0): GPUProfile(

        key="v100_sxm3_32gb",

        display_name="NVIDIA Tesla V100-SXM3-32GB",

        compute_capability=(7, 0),

        peak_bandwidth_gbps=900.0,

        max_threads_per_sm=2048,

        max_blocks_per_sm=32,

        max_regs_per_sm=65_536,

        max_smem_per_sm=96 * 1024,

        max_warps_per_sm=64,

    ),

    (8, 0): GPUProfile(

        key="a100",

        display_name="NVIDIA A100 40GB",

        compute_capability=(8, 0),

        peak_bandwidth_gbps=1555.0,

        max_threads_per_sm=2048,

        max_blocks_per_sm=32,

        max_regs_per_sm=65_536,

        max_smem_per_sm=164 * 1024,

        max_warps_per_sm=64,

    ),

    (8, 6): GPUProfile(

        key="rtx_a6000",

        display_name="NVIDIA RTX A6000",

        compute_capability=(8, 6),

        peak_bandwidth_gbps=768.0,

        max_threads_per_sm=1536,

        max_blocks_per_sm=16,

        max_regs_per_sm=65_536,

        max_smem_per_sm=100 * 1024,

        max_warps_per_sm=48,

    ),

}

# Historical public name retained for scripts that imported it directly.

AMPERE_PROFILES = GPU_PROFILES



def detect_gpu_profile(device_id: int) -> GPUProfile:

    capability = tuple(torch.cuda.get_device_capability(device_id))

    if capability not in GPU_PROFILES:

        raise RuntimeError(

            "This experiment currently supports Tesla V100/CC 7.0, "

            "A100/CC 8.0, and RTX A6000/CC 8.6; detected CC "

            f"{capability[0]}.{capability[1]}."

        )

    profile = GPU_PROFILES[capability]

    name = torch.cuda.get_device_name(device_id)

    total_gib = torch.cuda.get_device_properties(device_id).total_memory / 1024**3

    if capability == (7, 0) and not ("V100" in name and 28.0 <= total_gib <= 36.0):

        warnings.warn(

            "CC 7.0 occupancy limits are valid, but the built-in 900 GB/s "

            "bandwidth reference specifically targets the detected paper "

            "node's Tesla V100-SXM3-32GB configuration.",

            stacklevel=2,

        )

    elif capability == (8, 0) and not (

        "A100" in name and 35.0 <= total_gib <= 45.0

    ):

        warnings.warn(

            "CC 8.0 occupancy limits are valid, but the built-in 1555 GB/s "

            "bandwidth reference specifically targets the detected paper node's "

            "A100 40GB configuration.",

            stacklevel=2,

        )

    return profile



def ceil_div(a: int, b: int) -> int:

    return (a + b - 1) // b



def round_up(value: int, quantum: int) -> int:

    return ceil_div(value, quantum) * quantum



def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:

    return float((a - b).abs().max().item())



def normalized_max_abs(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:

    scale = max(float(b.abs().max().item()), eps)

    return max_abs_diff(a, b) / scale



def estimate_gpu_occupancy(

    threads_per_block: int,

    regs_per_thread: int,

    smem_per_block: int,

    profile: GPUProfile,

) -> dict[str, Any]:

    """Estimate occupancy using compiler-reported resources and GPU limits.

    Register allocation is rounded per warp to 256 registers. Shared memory is

    rounded to 256 bytes. This is an analytical estimate; Nsight Compute remains

    authoritative for achieved occupancy.

    """

    if threads_per_block <= 0 or threads_per_block > 1024:

        raise ValueError("threads_per_block must be in [1, 1024]")

    limits = profile.limits

    warps_per_block = ceil_div(threads_per_block, WARP_SIZE)

    regs_per_warp = round_up(regs_per_thread * WARP_SIZE, 256)

    regs_per_block = regs_per_warp * warps_per_block

    allocated_smem = round_up(smem_per_block, 256) if smem_per_block else 0

    by_threads = limits["max_threads_per_sm"] // threads_per_block

    by_warps = limits["max_warps_per_sm"] // warps_per_block

    by_regs = (

        limits["max_regs_per_sm"] // regs_per_block

        if regs_per_block

        else limits["max_blocks_per_sm"]

    )

    by_smem = (

        limits["max_smem_per_sm"] // allocated_smem

        if allocated_smem

        else limits["max_blocks_per_sm"]

    )

    constraints = {

        "threads": by_threads,

        "warps": by_warps,

        "registers": by_regs,

        "shared_memory": by_smem,

        "architectural_block_limit": limits["max_blocks_per_sm"],

    }

    blocks_per_sm = max(0, min(constraints.values()))

    active_warps = min(

        blocks_per_sm * warps_per_block,

        limits["max_warps_per_sm"],

    )

    occupancy = active_warps / limits["max_warps_per_sm"]

    limiting_value = min(constraints.values())

    limiting_resources = [

        key for key, value in constraints.items() if value == limiting_value

    ]

    return {

        "threads_per_block": threads_per_block,

        "warps_per_block": warps_per_block,

        "regs_per_thread": regs_per_thread,

        "allocated_regs_per_block": regs_per_block,

        "static_smem_bytes": smem_per_block,

        "allocated_smem_bytes_est": allocated_smem,

        "resident_blocks_per_sm_est": blocks_per_sm,

        "active_warps_per_sm_est": active_warps,

        "occupancy_est": occupancy,

        "limiting_resources_est": ",".join(limiting_resources),

        "gpu_profile": profile.key,

        "compute_capability": (

            f"{profile.compute_capability[0]}.{profile.compute_capability[1]}"

        ),

    }



# Backward-compatible alias for existing analysis scripts.

estimate_ampere_occupancy = estimate_gpu_occupancy



def minimum_algorithmic_bytes(B: int, H: int, L: int, K: int) -> int:

    """Unique input/output bytes, not profiler-measured DRAM traffic."""

    return (2 * B * H * L + H * K) * 4



def algorithmic_bandwidth_gbps(num_bytes: int, time_ms: float) -> float:

    if time_ms <= 0:

        return float("nan")

    return num_bytes / (time_ms / 1000.0) / 1e9



# =============================================================================

# CUDA-event collection without per-kernel synchronization

# =============================================================================

class KernelEventCollector:

    def __init__(self) -> None:

        self.enabled = False

        self._context: dict[str, Any] = {}

        self._events: list[

            tuple[str, torch.cuda.Event, torch.cuda.Event, dict[str, Any]]

        ] = []

    def set_context(self, **context: Any) -> None:

        self._context = dict(context)

    def clear(self) -> None:

        self._events.clear()

        self._context.clear()

    def launch(

        self,

        phase: str,

        launch_fn: Callable[[], None],

        metadata: dict[str, Any] | None = None,

    ) -> None:

        if not self.enabled:

            launch_fn()

            return

        start = torch.cuda.Event(enable_timing=True)

        end = torch.cuda.Event(enable_timing=True)

        start.record()

        launch_fn()

        end.record()

        row = dict(self._context)

        if metadata:

            row.update(metadata)

        self._events.append((phase, start, end, row))

    def resolve(self, synchronize: bool = True) -> pd.DataFrame:

        if synchronize:

            torch.cuda.synchronize()

        rows: list[dict[str, Any]] = []

        phase_counts: dict[str, int] = {}

        for phase, start, end, metadata in self._events:

            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            rows.append(

                {

                    **metadata,

                    "phase": phase,

                    "run_id": phase_counts[phase],

                    "runtime_ms": float(start.elapsed_time(end)),

                }

            )

        return pd.DataFrame(rows)



KERNEL_EVENTS = KernelEventCollector()



# =============================================================================

# CUDA kernels

# =============================================================================

CUDA_TEMPLATE = r"""

#include <stdint.h>

extern "C" {

#define WARP 32

#define MAX_L 48

#define MAX_K 48

#define WARPS_PER_BLOCK __WARPS_PER_BLOCK__

#define FWD_BLOCK (WARPS_PER_BLOCK * WARP)

#define BWDK_BTILE __BWDK_BTILE__

#define BWDK_KPACK 4

#define BWDK_BLOCK __BWDK_BLOCK__
#define REDUCE_BLOCK 256



__device__ __forceinline__ float warp_reduce_sum(float value) {

    #pragma unroll

    for (int offset = 16; offset > 0; offset >>= 1) {

        value += __shfl_down_sync(0xffffffff, value, offset);

    }

    return value;

}



// -----------------------------------------------------------------------------

// Forward: block tile = WARPS_PER_BLOCK batches x one channel x L positions.

// Warp tile = one batch x one channel x up to 48 positions.

// Thread tile = t0=lane and t1=lane+32, accumulated in the same K loop.

// k remains in global/read-only cache: prior paper measurements showed that

// staging k per block did not pay for its synchronization overhead.

// -----------------------------------------------------------------------------

__global__ void dwconv1d_warptile_fwd(

    const float* __restrict__ x,

    const float* __restrict__ kHK,

    float* __restrict__ y,

    const int B,

    const int H,

    const int L,

    const int K)

{

    const int tid = threadIdx.x;

    const int lane = tid & 31;

    const int warp_id = tid >> 5;

    const int b = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const int h = blockIdx.y;

    const int pad = K >> 1;

    if (warp_id >= WARPS_PER_BLOCK || b >= B || h >= H) return;

    if (L > MAX_L || K > MAX_K) return;

    __shared__ float sX[WARPS_PER_BLOCK][MAX_L];

    const float* xb = x + ((size_t)b * H + h) * L;

    const float* kb = kHK + (size_t)h * K;

    float* yb = y + ((size_t)b * H + h) * L;

    for (int i = lane; i < L; i += WARP) {

        sX[warp_id][i] = xb[i];

    }

    __syncwarp();

    const int t0 = lane;

    const int t1 = lane + WARP;

    const bool valid0 = t0 < L;

    const bool valid1 = t1 < L;

    float acc0 = 0.0f;

    float acc1 = 0.0f;

    #pragma unroll

    for (int kk = 0; kk < MAX_K; ++kk) {

        if (kk < K) {

            const float kval = __ldg(kb + kk);

            if (valid0) {

                const int xi0 = t0 + kk - pad;

                if ((unsigned)xi0 < (unsigned)L) {

                    acc0 = fmaf(sX[warp_id][xi0], kval, acc0);

                }

            }

            if (valid1) {

                const int xi1 = t1 + kk - pad;

                if ((unsigned)xi1 < (unsigned)L) {

                    acc1 = fmaf(sX[warp_id][xi1], kval, acc1);

                }

            }

        }

    }

    if (valid0) yb[t0] = acc0;

    if (valid1) yb[t1] = acc1;

}



// -----------------------------------------------------------------------------

// Input-gradient: same block/warp/thread hierarchy as Forward.

// -----------------------------------------------------------------------------

__global__ void dwconv1d_warptile_bwd_input(

    const float* __restrict__ gy,

    const float* __restrict__ kHK,

    float* __restrict__ gx,

    const int B,

    const int H,

    const int L,

    const int K)

{

    const int tid = threadIdx.x;

    const int lane = tid & 31;

    const int warp_id = tid >> 5;

    const int b = blockIdx.x * WARPS_PER_BLOCK + warp_id;

    const int h = blockIdx.y;

    const int pad = K >> 1;

    if (warp_id >= WARPS_PER_BLOCK || b >= B || h >= H) return;

    if (L > MAX_L || K > MAX_K) return;

    __shared__ float sGY[WARPS_PER_BLOCK][MAX_L];

    const float* gy_bh = gy + ((size_t)b * H + h) * L;

    const float* kb = kHK + (size_t)h * K;

    float* gx_bh = gx + ((size_t)b * H + h) * L;

    for (int i = lane; i < L; i += WARP) {

        sGY[warp_id][i] = gy_bh[i];

    }

    __syncwarp();

    const int t0 = lane;

    const int t1 = lane + WARP;

    const bool valid0 = t0 < L;

    const bool valid1 = t1 < L;

    float acc0 = 0.0f;

    float acc1 = 0.0f;

    #pragma unroll

    for (int kk = 0; kk < MAX_K; ++kk) {

        if (kk < K) {

            const float kval = __ldg(kb + kk);

            if (valid0) {

                const int tout0 = t0 - kk + pad;

                if ((unsigned)tout0 < (unsigned)L) {

                    acc0 = fmaf(sGY[warp_id][tout0], kval, acc0);

                }

            }

            if (valid1) {

                const int tout1 = t1 - kk + pad;

                if ((unsigned)tout1 < (unsigned)L) {

                    acc1 = fmaf(sGY[warp_id][tout1], kval, acc1);

                }

            }

        }

    }

    if (valid0) gx_bh[t0] = acc0;

    if (valid1) gx_bh[t1] = acc1;

}



// -----------------------------------------------------------------------------

// Weight-gradient stage 1 (no atomics): one channel and one batch tile per block.
// Four adjacent kernel taps are accumulated per thread to reuse gy and expose ILP.
// Each block writes one reduced partial per valid tap to global memory.
//
// Partial layout: [H, K, num_btiles].  The batch-tile index is contiguous so the
// second reduction kernel reads block partials coalescently across q.
// -----------------------------------------------------------------------------

__global__ void dwconv1d_warptile_bwd_weight_partials(

    const float* __restrict__ gy,

    const float* __restrict__ x,

    float* __restrict__ partials,

    const int B,

    const int H,

    const int L,

    const int K,

    const int num_btiles)

{

    const int tid = threadIdx.x;

    const int lane = tid & 31;

    const int warp_id = tid >> 5;

    const int num_warps = BWDK_BLOCK / WARP;

    const int h = blockIdx.x;

    const int btile = blockIdx.y;

    const int pad = K >> 1;

    if (h >= H || btile >= num_btiles) return;

    if (L > MAX_L || K > MAX_K) return;

    const int b_start = btile * BWDK_BTILE;

    const int b_count = min(BWDK_BTILE, B - b_start);

    if (b_count <= 0) return;

    __shared__ float sGY[BWDK_BTILE][MAX_L];

    __shared__ float sX[BWDK_BTILE][MAX_L];

    __shared__ float sWarpSums[BWDK_KPACK][BWDK_BLOCK / WARP];

    for (int linear = tid; linear < b_count * L; linear += BWDK_BLOCK) {

        const int bb = linear / L;

        const int t = linear - bb * L;

        const size_t base = ((size_t)(b_start + bb) * H + h) * L;

        sGY[bb][t] = gy[base + t];

        sX[bb][t] = x[base + t];

    }

    __syncthreads();

    for (int k0 = 0; k0 < K; k0 += BWDK_KPACK) {

        float acc0 = 0.0f;

        float acc1 = 0.0f;

        float acc2 = 0.0f;

        float acc3 = 0.0f;

        for (int linear = tid; linear < b_count * L; linear += BWDK_BLOCK) {

            const int bb = linear / L;

            const int t = linear - bb * L;

            const float gyv = sGY[bb][t];

            const int j0 = k0 + 0;

            const int j1 = k0 + 1;

            const int j2 = k0 + 2;

            const int j3 = k0 + 3;

            if (j0 < K) {

                const int xi = t + j0 - pad;

                if ((unsigned)xi < (unsigned)L)

                    acc0 = fmaf(gyv, sX[bb][xi], acc0);

            }

            if (j1 < K) {

                const int xi = t + j1 - pad;

                if ((unsigned)xi < (unsigned)L)

                    acc1 = fmaf(gyv, sX[bb][xi], acc1);

            }

            if (j2 < K) {

                const int xi = t + j2 - pad;

                if ((unsigned)xi < (unsigned)L)

                    acc2 = fmaf(gyv, sX[bb][xi], acc2);

            }

            if (j3 < K) {

                const int xi = t + j3 - pad;

                if ((unsigned)xi < (unsigned)L)

                    acc3 = fmaf(gyv, sX[bb][xi], acc3);

            }

        }

        acc0 = warp_reduce_sum(acc0);

        acc1 = warp_reduce_sum(acc1);

        acc2 = warp_reduce_sum(acc2);

        acc3 = warp_reduce_sum(acc3);

        if (lane == 0) {

            sWarpSums[0][warp_id] = acc0;

            sWarpSums[1][warp_id] = acc1;

            sWarpSums[2][warp_id] = acc2;

            sWarpSums[3][warp_id] = acc3;

        }

        __syncthreads();

        if (warp_id == 0) {

            float value0 = lane < num_warps ? sWarpSums[0][lane] : 0.0f;

            float value1 = lane < num_warps ? sWarpSums[1][lane] : 0.0f;

            float value2 = lane < num_warps ? sWarpSums[2][lane] : 0.0f;

            float value3 = lane < num_warps ? sWarpSums[3][lane] : 0.0f;

            value0 = warp_reduce_sum(value0);

            value1 = warp_reduce_sum(value1);

            value2 = warp_reduce_sum(value2);

            value3 = warp_reduce_sum(value3);

            if (lane == 0) {

                if (k0 + 0 < K) {

                    const size_t index =
                        ((size_t)h * K + (k0 + 0)) * num_btiles + btile;

                    partials[index] = value0;

                }

                if (k0 + 1 < K) {

                    const size_t index =
                        ((size_t)h * K + (k0 + 1)) * num_btiles + btile;

                    partials[index] = value1;

                }

                if (k0 + 2 < K) {

                    const size_t index =
                        ((size_t)h * K + (k0 + 2)) * num_btiles + btile;

                    partials[index] = value2;

                }

                if (k0 + 3 < K) {

                    const size_t index =
                        ((size_t)h * K + (k0 + 3)) * num_btiles + btile;

                    partials[index] = value3;

                }

            }

        }

        __syncthreads();

    }

}



// -----------------------------------------------------------------------------
// Weight-gradient stage 2 (optimized, no atomics): one warp reduces all
// batch-tile partials for one (h, k) coefficient.  A 256-thread block contains
// eight independent warps and therefore reduces up to eight coefficients.
//
// For num_btiles=256, each lane reads eight values.  The q dimension is
// contiguous in the [H,K,num_btiles] workspace, so each warp performs
// coalesced 32-float loads at every loop iteration.  No shared memory,
// __syncthreads(), or cross-warp reduction is required in this stage.
// -----------------------------------------------------------------------------

__global__ void dwconv1d_warptile_reduce_partials(

    const float* __restrict__ partials,

    float* __restrict__ gk,

    const int H,

    const int K,

    const int num_btiles)

{

    const int h = blockIdx.x;

    const int tid = threadIdx.x;

    const int lane = tid & 31;

    const int warp_id = tid >> 5;

    const int warps_per_block = blockDim.x / WARP;

    const int kk = blockIdx.y * warps_per_block + warp_id;

    if (h >= H || kk >= K) return;

    const size_t base = ((size_t)h * K + kk) * num_btiles;

    float acc = 0.0f;

    for (int q = lane; q < num_btiles; q += WARP) {

        acc += partials[base + q];

    }

    acc = warp_reduce_sum(acc);

    if (lane == 0) {

        gk[(size_t)h * K + kk] = acc;

    }

}

} // extern "C"

"""



class AmpereWarpRuntime:

    """Own the PyCUDA primary-context reference and compiled kernel functions.

    CUDA's current-context stack is local to each host thread.  PyTorch's

    autograd engine may invoke ``backward`` on a different host thread from

    ``forward``, so retaining and pushing the primary context once in

    ``__init__`` is insufficient.  Compilation, resource queries, and every

    PyCUDA launch therefore activate the shared primary context on the calling

    thread for only the duration of the driver operation.

    """

    def __init__(

        self,

        warps_per_block: int = DEFAULT_WARPS_PER_BLOCK,

        bwdk_btile: int = DEFAULT_BWDK_BTILE,

        bwdk_block: int = DEFAULT_BWDK_BLOCK,

        fast_math: bool = False,

    ) -> None:

        if warps_per_block not in {1, 2, 4, 8, 16, 32}:

            raise ValueError("warps_per_block must be one of 1,2,4,8,16,32")

        if bwdk_btile != 64:

            raise ValueError("This paper-confirmed version fixes BWDK_BTILE=64")

        if not (WARP_SIZE <= bwdk_block <= 1024) or bwdk_block % WARP_SIZE != 0:

            raise ValueError(

                "bwdk_block must be a multiple of 32 in the inclusive range "

                "[32, 1024]"

            )

        self.warps_per_block = warps_per_block

        self.fwd_block = warps_per_block * WARP_SIZE

        self.bwdk_btile = bwdk_btile

        self.bwdk_block = bwdk_block

        self.fast_math = fast_math

        self._closed = False

        cuda.init()

        torch.cuda.init()

        self.device_id = torch.cuda.current_device()

        self.device = cuda.Device(self.device_id)

        self.profile = detect_gpu_profile(self.device_id)

        self.technique = f"WarpNoAtomicOpt-{self.profile.key}"

        self.context = self.device.retain_primary_context()

        # Reused global workspace for the two-stage no-atomic dK path.
        # Layout: [H, K, num_btiles]. Keeping it on the runtime also keeps
        # the storage alive across asynchronous PyCUDA launches.
        self._partials_workspace: torch.Tensor | None = None
        self._partials_shape: tuple[int, int, int] | None = None
        self.reduce_block = 256
        self.reduce_warps = self.reduce_block // WARP_SIZE

        source = (

CUDA_TEMPLATE.replace("__WARPS_PER_BLOCK__", str(warps_per_block))

            .replace("__BWDK_BTILE__", str(bwdk_btile))

            .replace("__BWDK_BLOCK__", str(bwdk_block))

        )

        major, minor = self.profile.compute_capability

        options = [

            "-O3",

            f"-arch=sm_{major}{minor}",

            "-lineinfo",

            "-Xptxas=-v",

            "-ftz=true",

            "-prec-div=false",

            "-prec-sqrt=false",

        ]

        if fast_math:

            options.append("-use_fast_math")

        with self._active_primary_context():

            self.module = SourceModule(source, options=options, no_extern_c=True)

            self.fwd = self.module.get_function("dwconv1d_warptile_fwd")

            self.bwd_input = self.module.get_function("dwconv1d_warptile_bwd_input")

            self.bwd_weight_partials = self.module.get_function(
                "dwconv1d_warptile_bwd_weight_partials"
            )

            self.reduce_partials = self.module.get_function(
                "dwconv1d_warptile_reduce_partials"
            )

    @contextmanager

    def _active_primary_context(self) -> Iterator[None]:

        """Make the PyTorch/PyCUDA primary context current on this thread."""

        self.context.push()

        try:

            yield

        finally:

            cuda.Context.pop()

    @staticmethod

    def _check_shapes(x: torch.Tensor, k: torch.Tensor) -> tuple[int, int, int, int]:

        if not x.is_cuda or not k.is_cuda:

            raise ValueError("x and k must be CUDA tensors")

        if x.dtype != torch.float32 or k.dtype != torch.float32:

            raise ValueError("Only FP32 is supported")

        if x.ndim != 3 or k.ndim != 2:

            raise ValueError("Expected x[B,H,L] and k[H,K]")

        B, H, L = map(int, x.shape)

        Hk, K = map(int, k.shape)

        if H != Hk:

            raise ValueError("x and k channel dimensions do not match")

        if L > MAX_L or K > MAX_K:

            raise ValueError(f"Specialized kernels require L<={MAX_L}, K<={MAX_K}")

        return B, H, L, K

    @staticmethod

    def _ptr(tensor: torch.Tensor) -> np.uintp:

        return np.uintp(tensor.data_ptr())

    def launch_fwd(

        self,

        x: torch.Tensor,

        k: torch.Tensor,

        y: torch.Tensor,

        collector: KernelEventCollector | None = None,

    ) -> None:

        B, H, L, K = self._check_shapes(x, k)

        def launch() -> None:

            with self._active_primary_context():

                self.fwd(

                    self._ptr(x),

                    self._ptr(k),

                    self._ptr(y),

                    np.int32(B),

                    np.int32(H),

                    np.int32(L),

                    np.int32(K),

                    block=(self.fwd_block, 1, 1),

                    grid=(ceil_div(B, self.warps_per_block), H, 1),

                )

        metadata = {"B": B, "H": H, "L": L, "K": K}

        (collector or KERNEL_EVENTS).launch("fwd", launch, metadata)

    def launch_bwd_input(

        self,

        gy: torch.Tensor,

        k: torch.Tensor,

        gx: torch.Tensor,

        collector: KernelEventCollector | None = None,

    ) -> None:

        B, H, L, K = self._check_shapes(gy, k)

        def launch() -> None:

            with self._active_primary_context():

                self.bwd_input(

                    self._ptr(gy),

                    self._ptr(k),

                    self._ptr(gx),

                    np.int32(B),

                    np.int32(H),

                    np.int32(L),

                    np.int32(K),

                    block=(self.fwd_block, 1, 1),

                    grid=(ceil_div(B, self.warps_per_block), H, 1),

                )

        metadata = {"B": B, "H": H, "L": L, "K": K}

        (collector or KERNEL_EVENTS).launch("bwd_input", launch, metadata)

    def _get_partials_workspace(
        self,
        H: int,
        K: int,
        num_btiles: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return a reusable [H,K,num_btiles] FP32 workspace."""

        shape = (H, K, num_btiles)

        if (
            self._partials_workspace is None
            or self._partials_shape != shape
            or self._partials_workspace.device != device
        ):
            self._partials_workspace = torch.empty(
                shape,
                device=device,
                dtype=torch.float32,
            )
            self._partials_shape = shape

        return self._partials_workspace

    def launch_bwd_weight(
        self,
        gy: torch.Tensor,
        x: torch.Tensor,
        gk: torch.Tensor,
        K: int,
        collector: KernelEventCollector | None = None,
    ) -> None:
        """Launch the complete two-stage no-atomic dK path.

        The single event named ``bwd_weight`` brackets both CUDA kernels, so
        its elapsed time is directly comparable with the atomic single-kernel
        implementation.
        """

        if not gy.is_cuda or not x.is_cuda or not gk.is_cuda:
            raise ValueError("gy, x, and gk must be CUDA tensors")

        if (
            gy.dtype != torch.float32
            or x.dtype != torch.float32
            or gk.dtype != torch.float32
        ):
            raise ValueError("Only FP32 is supported")

        if gy.shape != x.shape:
            raise ValueError("gy and x must have the same [B,H,L] shape")

        B, H, L = map(int, x.shape)

        if gk.shape != (H, K):
            raise ValueError(f"gk must have shape {(H, K)}")

        if L > MAX_L or K > MAX_K:
            raise ValueError(
                f"Specialized kernels require L<={MAX_L}, K<={MAX_K}"
            )

        num_btiles = ceil_div(B, self.bwdk_btile)
        partials = self._get_partials_workspace(
            H=H,
            K=K,
            num_btiles=num_btiles,
            device=x.device,
        )

        def launch() -> None:
            with self._active_primary_context():
                self.bwd_weight_partials(
                    self._ptr(gy),
                    self._ptr(x),
                    self._ptr(partials),
                    np.int32(B),
                    np.int32(H),
                    np.int32(L),
                    np.int32(K),
                    np.int32(num_btiles),
                    block=(self.bwdk_block, 1, 1),
                    grid=(H, num_btiles, 1),
                )

                self.reduce_partials(
                    self._ptr(partials),
                    self._ptr(gk),
                    np.int32(H),
                    np.int32(K),
                    np.int32(num_btiles),
                    block=(self.reduce_block, 1, 1),
                    grid=(H, ceil_div(K, self.reduce_warps), 1),
                )

        metadata = {
            "B": B,
            "H": H,
            "L": L,
            "K": K,
            "num_btiles": num_btiles,
            "partial_workspace_bytes": int(partials.numel() * partials.element_size()),
            "reduction": "two_stage_no_atomic_warp_per_coefficient",
        }

        (collector or KERNEL_EVENTS).launch("bwd_weight", launch, metadata)

    def launch_bwd_weight_stages(
        self,
        gy: torch.Tensor,
        x: torch.Tensor,
        gk: torch.Tensor,
        K: int,
        collector: KernelEventCollector,
    ) -> None:
        """Launch and time the two no-atomic dK stages separately.

        This method is used only for microbenchmark breakdowns.  The regular
        ``launch_bwd_weight`` method keeps a single event around both stages so
        the reported dK time remains directly comparable with the atomic
        single-kernel implementation.
        """

        if not gy.is_cuda or not x.is_cuda or not gk.is_cuda:
            raise ValueError("gy, x, and gk must be CUDA tensors")
        if (
            gy.dtype != torch.float32
            or x.dtype != torch.float32
            or gk.dtype != torch.float32
        ):
            raise ValueError("Only FP32 is supported")
        if gy.shape != x.shape:
            raise ValueError("gy and x must have the same [B,H,L] shape")

        B, H, L = map(int, x.shape)
        if gk.shape != (H, K):
            raise ValueError(f"gk must have shape {(H, K)}")
        if L > MAX_L or K > MAX_K:
            raise ValueError(
                f"Specialized kernels require L<={MAX_L}, K<={MAX_K}"
            )

        num_btiles = ceil_div(B, self.bwdk_btile)
        partials = self._get_partials_workspace(
            H=H,
            K=K,
            num_btiles=num_btiles,
            device=x.device,
        )

        metadata = {
            "B": B,
            "H": H,
            "L": L,
            "K": K,
            "num_btiles": num_btiles,
            "partial_workspace_bytes": int(
                partials.numel() * partials.element_size()
            ),
            "reduction": "two_stage_no_atomic_warp_per_coefficient",
        }

        def launch_partials() -> None:
            with self._active_primary_context():
                self.bwd_weight_partials(
                    self._ptr(gy),
                    self._ptr(x),
                    self._ptr(partials),
                    np.int32(B),
                    np.int32(H),
                    np.int32(L),
                    np.int32(K),
                    np.int32(num_btiles),
                    block=(self.bwdk_block, 1, 1),
                    grid=(H, num_btiles, 1),
                )

        def launch_reduce() -> None:
            with self._active_primary_context():
                self.reduce_partials(
                    self._ptr(partials),
                    self._ptr(gk),
                    np.int32(H),
                    np.int32(K),
                    np.int32(num_btiles),
                    block=(self.reduce_block, 1, 1),
                    grid=(H, ceil_div(K, self.reduce_warps), 1),
                )

        collector.launch(
            "bwd_weight_partials",
            launch_partials,
            metadata,
        )
        collector.launch(
            "bwd_weight_reduce",
            launch_reduce,
            metadata,
        )

    def resource_report(self) -> pd.DataFrame:

        rows = []

        with self._active_primary_context():

            for name, fn, block in [

                ("fwd", self.fwd, self.fwd_block),

                ("bwd_input", self.bwd_input, self.fwd_block),

                ("bwd_weight_partials", self.bwd_weight_partials, self.bwdk_block),

                ("bwd_weight_reduce", self.reduce_partials, self.reduce_block),

            ]:

                row = {"kernel": name}

                row.update(

                    estimate_gpu_occupancy(

                        threads_per_block=block,

                        regs_per_thread=int(fn.num_regs),

                        smem_per_block=int(fn.shared_size_bytes),

                        profile=self.profile,

                    )

                )

                row["local_memory_bytes"] = int(fn.local_size_bytes)

                rows.append(row)

        return pd.DataFrame(rows)

    def close(self) -> None:

        if self._closed:

            return

        torch.cuda.synchronize()

        with self._active_primary_context():

            self.fwd = None

            self.bwd_input = None

            self.bwd_weight_partials = None

            self.reduce_partials = None

            self._partials_workspace = None

            self._partials_shape = None

            self.module = None

            gc.collect()

        self.context.detach()

        self._closed = True



_RUNTIME: AmpereWarpRuntime | None = None



def install_runtime(runtime: AmpereWarpRuntime) -> None:

    global _RUNTIME

    _RUNTIME = runtime



def get_runtime() -> AmpereWarpRuntime:

    if _RUNTIME is None:

        raise RuntimeError("CUDA runtime has not been installed")

    return _RUNTIME



class DepthConvAmpere(torch.autograd.Function):

    @staticmethod

    def forward(ctx: Any, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

        runtime = get_runtime()

        x_c = x.contiguous()

        k_c = k.contiguous()

        B, H, L, K = runtime._check_shapes(x_c, k_c)

        y = torch.empty_like(x_c)

        runtime.launch_fwd(x_c, k_c, y)

        ctx.save_for_backward(x_c, k_c)

        ctx.params = (B, H, L, K)

        return y

    @staticmethod

    def backward(ctx: Any, gy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        runtime = get_runtime()

        x_c, k_c = ctx.saved_tensors

        B, H, L, K = ctx.params

        gy_c = gy.contiguous()

        if tuple(gy_c.shape) != (B, H, L):

            raise ValueError("Unexpected gradient shape")

        gx = torch.empty_like(x_c)

        runtime.launch_bwd_input(gy_c, k_c, gx)

        # The second no-atomic reduction kernel writes every gk[h,k]

        # coefficient exactly once, so no zero-fill is required.

        gk = torch.empty_like(k_c)

        runtime.launch_bwd_weight(gy_c, x_c, gk, K)

        return gx, gk



def depthwise_conv_ampere(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

    return DepthConvAmpere.apply(x, k)



def reference_depthwise_conv(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:

    y = F.conv1d(x, k.unsqueeze(1), padding=k.shape[1] // 2, groups=x.shape[1])

    return y[..., : x.shape[-1]].contiguous()



# =============================================================================

# Correctness and microbenchmark

# =============================================================================

def validate_correctness(

    output_dir: Path,

    cases: list[tuple[int, int, int, int]] | None = None,

    seed: int = 0,

) -> pd.DataFrame:

    torch.manual_seed(seed)

    np.random.seed(seed)

    KERNEL_EVENTS.enabled = False

    KERNEL_EVENTS.clear()

    if cases is None:

        cases = [

            (2, 4, 16, 8),

            (4, 8, 31, 16),

            (8, 16, 48, 24),

            (32, 64, 48, 48),

            (128, 128, 48, 48),

        ]

    rows: list[dict[str, Any]] = []

    print("\n" + "=" * 88)

    print("CORRECTNESS AGAINST PYTORCH FP32")

    print("=" * 88)

    for B, H, L, K in cases:

        x = torch.randn(B, H, L, device="cuda", dtype=torch.float32, requires_grad=True)

        k = torch.randn(H, K, device="cuda", dtype=torch.float32, requires_grad=True)

        gy = torch.randn(B, H, L, device="cuda", dtype=torch.float32)

        x_ref = x.detach().clone().requires_grad_(True)

        k_ref = k.detach().clone().requires_grad_(True)

        y = depthwise_conv_ampere(x, k)

        y_ref = reference_depthwise_conv(x_ref, k_ref)

        gx, gk = torch.autograd.grad(y, (x, k), gy)

        gx_ref, gk_ref = torch.autograd.grad(y_ref, (x_ref, k_ref), gy)

        torch.cuda.synchronize()

        row = {

            "B": B,

            "H": H,

            "L": L,

            "K": K,

            "fwd_max_abs": max_abs_diff(y, y_ref),

            "bwd_input_max_abs": max_abs_diff(gx, gx_ref),

            "bwd_weight_max_abs": max_abs_diff(gk, gk_ref),

            "fwd_normalized_max_abs": normalized_max_abs(y, y_ref),

            "bwd_input_normalized_max_abs": normalized_max_abs(gx, gx_ref),

            "bwd_weight_normalized_max_abs": normalized_max_abs(gk, gk_ref),

            "fwd_allclose": bool(torch.allclose(y, y_ref, atol=2e-3, rtol=2e-3)),

            "bwd_input_allclose": bool(

                torch.allclose(gx, gx_ref, atol=2e-3, rtol=2e-3)

            ),

            "bwd_weight_allclose": bool(

                torch.allclose(gk, gk_ref, atol=1e-2, rtol=5e-4)

            ),

        }

        rows.append(row)

        print(

            f"B={B:4d} H={H:3d} L={L:2d} K={K:2d} | "

            f"fwd={row['fwd_max_abs']:.3e} "

            f"dx={row['bwd_input_max_abs']:.3e} "

            f"dk={row['bwd_weight_max_abs']:.3e} | "

            f"ok={row['fwd_allclose'] and row['bwd_input_allclose'] and row['bwd_weight_allclose']}"

        )

        del x, k, gy, x_ref, k_ref, y, y_ref, gx, gk, gx_ref, gk_ref

    result = pd.DataFrame(rows)

    output_dir.mkdir(parents=True, exist_ok=True)

    result.to_csv(output_dir / "correctness.csv", index=False)

    if not bool(

        result[["fwd_allclose", "bwd_input_allclose", "bwd_weight_allclose"]]

        .to_numpy()

        .all()

    ):

        raise AssertionError("At least one correctness case failed")

    return result



def summarize_kernel_times(raw: pd.DataFrame) -> pd.DataFrame:

    columns = [

        "phase",

        "runs",

        "mean_ms",

        "median_ms",

        "std_ms",

        "p05_ms",

        "p95_ms",

    ]

    if raw.empty:

        return pd.DataFrame(columns=columns)

    rows = []

    for phase, group in raw.groupby("phase", sort=False):

        values = group["runtime_ms"].to_numpy(dtype=float)

        rows.append(

            {

                "phase": phase,

                "runs": len(values),

                "mean_ms": float(np.mean(values)),

                "median_ms": float(np.median(values)),

                "std_ms": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,

                "p05_ms": float(np.quantile(values, 0.05)),

                "p95_ms": float(np.quantile(values, 0.95)),

            }

        )

    return pd.DataFrame(rows, columns=columns)



def benchmark_kernels(

    runtime: AmpereWarpRuntime,

    output_dir: Path,

    B: int = DEFAULT_B,

    H: int = DEFAULT_H,

    L: int = DEFAULT_L,

    K: int = DEFAULT_K,

    warmup: int = 5,

    runs: int = 100,

    seed: int = 0,

) -> tuple[pd.DataFrame, pd.DataFrame]:

    torch.manual_seed(seed)

    x = torch.randn(B, H, L, device="cuda", dtype=torch.float32)

    k = torch.randn(H, K, device="cuda", dtype=torch.float32)

    gy = torch.randn_like(x)

    y = torch.empty_like(x)

    gx = torch.empty_like(x)

    gk = torch.empty_like(k)

    KERNEL_EVENTS.enabled = False

    for _ in range(warmup):

        runtime.launch_fwd(x, k, y)

        runtime.launch_bwd_input(gy, k, gx)

        runtime.launch_bwd_weight(gy, x, gk, K)

    torch.cuda.synchronize()

    collector = KernelEventCollector()

    collector.enabled = True

    common_context = {

        "technique": runtime.technique,

        "gpu_profile": runtime.profile.key,

        "device_name": torch.cuda.get_device_name(runtime.device_id),

        "warps_per_block": runtime.warps_per_block,

        "bwdk_btile": runtime.bwdk_btile,

        "bwdk_block": runtime.bwdk_block,

        "bwdk_reduction": "two_stage_no_atomic_warp_per_coefficient",

    }

    for run_id in range(1, runs + 1):

        collector.set_context(**common_context, run_id_requested=run_id)

        runtime.launch_fwd(x, k, y, collector)

    for run_id in range(1, runs + 1):

        collector.set_context(**common_context, run_id_requested=run_id)

        runtime.launch_bwd_input(gy, k, gx, collector)

    for run_id in range(1, runs + 1):

        collector.set_context(**common_context, run_id_requested=run_id)

        runtime.launch_bwd_weight(gy, x, gk, K, collector)

    raw = collector.resolve(synchronize=True)

    summary = summarize_kernel_times(raw)

    algorithmic_bytes = minimum_algorithmic_bytes(B, H, L, K)

    summary["algorithmic_bytes"] = algorithmic_bytes

    summary["algorithmic_bandwidth_gbps"] = summary["mean_ms"].map(

        lambda value: algorithmic_bandwidth_gbps(algorithmic_bytes, value)

    )

    summary["pct_of_peak_algorithmic"] = (

        100.0

        * summary["algorithmic_bandwidth_gbps"]

        / runtime.profile.peak_bandwidth_gbps

    )

    summary["technique"] = runtime.technique

    summary["gpu_profile"] = runtime.profile.key

    summary["device_name"] = torch.cuda.get_device_name(runtime.device_id)

    summary["warps_per_block"] = runtime.warps_per_block

    summary["bwdk_btile"] = runtime.bwdk_btile

    summary["bwdk_block"] = runtime.bwdk_block

    total_mean = float(summary["mean_ms"].sum())

    total_median_sum = float(summary["median_ms"].sum())

    print("\n" + "=" * 88)

    print(f"{runtime.profile.display_name} KERNEL MICROBENCHMARK")

    print("=" * 88)

    print(f"B={B}, H={H}, L={L}, K={K}, warmup={warmup}, runs={runs}")

    print(summary.to_string(index=False))

    print(f"\nSum of phase means:   {total_mean:.4f} ms")

    print(f"Sum of phase medians: {total_median_sum:.4f} ms")

    print(

        "Note: algorithmic bandwidth uses unique minimum bytes; "

        "Nsight Compute is required for actual DRAM traffic."

    )

    # Measure the two no-atomic dK stages separately as a diagnostic.
    # This is a second measurement pass and therefore does not affect the
    # headline ``bwd_weight`` timing above.
    breakdown = KernelEventCollector()
    breakdown.enabled = True

    # Warm the exact stage-specific launch path without collecting events.
    for _ in range(warmup):
        num_btiles = ceil_div(B, runtime.bwdk_btile)
        partials = runtime._get_partials_workspace(H, K, num_btiles, x.device)
        with runtime._active_primary_context():
            runtime.bwd_weight_partials(
                runtime._ptr(gy),
                runtime._ptr(x),
                runtime._ptr(partials),
                np.int32(B),
                np.int32(H),
                np.int32(L),
                np.int32(K),
                np.int32(num_btiles),
                block=(runtime.bwdk_block, 1, 1),
                grid=(H, num_btiles, 1),
            )
            runtime.reduce_partials(
                runtime._ptr(partials),
                runtime._ptr(gk),
                np.int32(H),
                np.int32(K),
                np.int32(num_btiles),
                block=(runtime.reduce_block, 1, 1),
                grid=(H, ceil_div(K, runtime.reduce_warps), 1),
            )
    torch.cuda.synchronize()

    for run_id in range(1, runs + 1):
        breakdown.set_context(
            **common_context,
            run_id_requested=run_id,
        )
        runtime.launch_bwd_weight_stages(
            gy,
            x,
            gk,
            K,
            breakdown,
        )

    breakdown_raw = breakdown.resolve(synchronize=True)
    breakdown_summary = summarize_kernel_times(breakdown_raw)

    print("\n" + "=" * 88)
    print("NO-ATOMIC dK BREAKDOWN")
    print("=" * 88)
    print(breakdown_summary.to_string(index=False))
    if not breakdown_summary.empty:
        breakdown_total = float(breakdown_summary["mean_ms"].sum())
        print(f"\nSum of dK stage means: {breakdown_total:.4f} ms")

    output_dir.mkdir(parents=True, exist_ok=True)

    raw.to_csv(output_dir / "benchmark_raw.csv", index=False)

    summary.to_csv(output_dir / "benchmark_summary.csv", index=False)

    breakdown_raw.to_csv(
        output_dir / "bwd_weight_breakdown_raw.csv",
        index=False,
    )
    breakdown_summary.to_csv(
        output_dir / "bwd_weight_breakdown_summary.csv",
        index=False,
    )

    del x, k, gy, y, gx, gk

    torch.cuda.empty_cache()

    return raw, summary



# =============================================================================

# S4 model

# =============================================================================

TRAIN_CONV_BACKEND = "warp"



class S4ConvKernel(nn.Module):

    def __init__(self, hidden_dim: int, state_dim: int) -> None:

        super().__init__()

        self.H = hidden_dim

        self.N = state_dim

        self.log_dt = nn.Parameter(

            torch.rand(hidden_dim) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)

        )

        self.C_re = nn.Parameter(torch.randn(hidden_dim, state_dim // 2))

        self.C_im = nn.Parameter(torch.randn(hidden_dim, state_dim // 2))

        self.log_A_re = nn.Parameter(torch.randn(hidden_dim, state_dim // 2))

        self.A_im = nn.Parameter(torch.ones(hidden_dim, state_dim // 2))

    def build_kernel(self, taps: int, device: torch.device) -> torch.Tensor:

        dt = torch.exp(self.log_dt)

        C = self.C_re + 1j * self.C_im

        A = -torch.exp(self.log_A_re) + 1j * self.A_im

        dtA = A * dt.unsqueeze(-1)

        times = torch.arange(taps, device=device, dtype=torch.float32)

        K_exp = torch.exp(dtA.unsqueeze(-1) * times)

        C_eff = C * (torch.exp(dtA) - 1.0) / A

        return (2 * torch.einsum("hn,hnk->hk", C_eff, K_exp).real).contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        _, H, L = x.shape

        if H != self.H:

            raise ValueError(f"Expected H={self.H}, got {H}")

        if L > MAX_L:

            raise ValueError(f"Warp-tiled paper kernel supports L<={MAX_L}")

        k = self.build_kernel(L, x.device)

        if TRAIN_CONV_BACKEND == "warp":

            return depthwise_conv_ampere(

                x.contiguous(),

                k,

            )

        if TRAIN_CONV_BACKEND == "pytorch":

            return reference_depthwise_conv(

                x.contiguous(),

                k,

            )

        raise RuntimeError(

            f"Unknown training backend: "

            f"{TRAIN_CONV_BACKEND}"

        )



class S4ConvBlock(nn.Module):

    def __init__(self, hidden_dim: int, state_dim: int, dropout: float) -> None:

        super().__init__()

        self.skip = nn.Parameter(torch.randn(hidden_dim))

        self.kernel_layer = S4ConvKernel(hidden_dim, state_dim)

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(dropout)

        self.output_linear = nn.Sequential(

            nn.Conv1d(hidden_dim, 2 * hidden_dim, kernel_size=1),

            nn.GLU(dim=1),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        residual = x

        x_hl = x.transpose(1, 2).contiguous()

        y = self.kernel_layer(x_hl)

        y = y + x_hl * self.skip.unsqueeze(-1)

        y = self.output_linear(self.activation(self.dropout(y)))

        return y.transpose(1, 2) + residual



class Model(nn.Module):

    def __init__(

        self,

        state_dim: int = 64,

        hidden_dim: int = 128,

        input_dim: int = 4,

        output_dim: int = 1,

        num_blocks: int = 4,

        dropout: float = 0.01,

    ) -> None:

        super().__init__()

        self.encoder = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(

            [S4ConvBlock(hidden_dim, state_dim, dropout) for _ in range(num_blocks)]

        )

        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Dataset contract: x is [B, L, input_dim].

        x = self.encoder(x)

        for block in self.blocks:

            x = block(x)

        return self.decoder(x)[:, -1, :]



def rmsle(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    prediction = prediction.reshape(-1)

    target = target.reshape(-1)

    prediction = torch.clamp(prediction, min=0.0)

    target = torch.clamp(target, min=0.0)

    squared = (torch.log1p(prediction) - torch.log1p(target)).square()

    return torch.sqrt(squared.mean() + 1e-12)



# =============================================================================

# Non-serializing training instrumentation

# =============================================================================

class StageEventCollector:

    def __init__(self) -> None:

        self.events: list[

            tuple[int, int, str, torch.cuda.Event, torch.cuda.Event]

        ] = []

        self.zero_grad_cpu_rows: list[dict[str, Any]] = []

    def measure(

        self,

        epoch: int,

        batch_id: int,

        stage: str,

        fn: Callable[[], Any],

    ) -> Any:

        start = torch.cuda.Event(enable_timing=True)

        end = torch.cuda.Event(enable_timing=True)

        start.record()

        result = fn()

        end.record()

        self.events.append((epoch, batch_id, stage, start, end))

        return result

    def add_zero_grad_cpu(self, epoch: int, batch_id: int, runtime_ms: float) -> None:

        self.zero_grad_cpu_rows.append(

            {

                "epoch": epoch,

                "batch_id": batch_id,

                "zero_grad_cpu_ms": runtime_ms,

            }

        )

    def resolve(self, synchronize: bool = True) -> pd.DataFrame:

        if synchronize:

            torch.cuda.synchronize()

        rows: dict[tuple[int, int], dict[str, Any]] = {}

        for item in self.zero_grad_cpu_rows:

            key = (int(item["epoch"]), int(item["batch_id"]))

            rows[key] = dict(item)

        for epoch, batch_id, stage, start, end in self.events:

            key = (epoch, batch_id)

            rows.setdefault(key, {"epoch": epoch, "batch_id": batch_id})

            rows[key][f"{stage}_gpu_ms"] = float(start.elapsed_time(end))

        return pd.DataFrame(rows.values()).sort_values(["epoch", "batch_id"])



class NVMLSampler:

    def __init__(self, device_id: int, interval_s: float = 0.1) -> None:

        self.interval_s = interval_s

        self.samples: list[dict[str, float]] = []

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._handle = None

        if pynvml is not None:

            try:

                pynvml.nvmlInit()

                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)

            except Exception:

                self._handle = None

    def start(self) -> None:

        self.samples.clear()

        self._stop.clear()

        if self._handle is None:

            return

        self._thread = threading.Thread(target=self._run, daemon=True)

        self._thread.start()

    def _run(self) -> None:

        while not self._stop.is_set():

            try:

                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)

                memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)

                self.samples.append(

                    {

                        "util_gpu_pct": float(util.gpu),

                        "util_mem_pct": float(util.memory),

                        "nvml_used_gib": float(memory.used / 1024**3),

                        "nvml_total_gib": float(memory.total / 1024**3),

                    }

                )

            except Exception:

                pass

            self._stop.wait(self.interval_s)

    def stop(self) -> dict[str, float]:

        self._stop.set()

        if self._thread is not None:

            self._thread.join(timeout=max(1.0, 2 * self.interval_s))

            self._thread = None

        if not self.samples:

            return {

                "util_gpu_avg_pct": float("nan"),

                "util_gpu_min_pct": float("nan"),

                "util_gpu_max_pct": float("nan"),

                "util_mem_avg_pct": float("nan"),

                "nvml_used_peak_gib": float("nan"),

                "nvml_total_gib": float("nan"),

            }

        frame = pd.DataFrame(self.samples)

        return {

            "util_gpu_avg_pct": float(frame["util_gpu_pct"].mean()),

            "util_gpu_min_pct": float(frame["util_gpu_pct"].min()),

            "util_gpu_max_pct": float(frame["util_gpu_pct"].max()),

            "util_mem_avg_pct": float(frame["util_mem_pct"].mean()),

            "nvml_used_peak_gib": float(frame["nvml_used_gib"].max()),

            "nvml_total_gib": float(frame["nvml_total_gib"].max()),

        }



@dataclass

class TrainConfig:

    epochs: int = 5

    batch_size: int = DEFAULT_B

    data_fraction: float = 0.1

    num_workers: int = 4

    prefetch_factor: int = 18

    learning_rate: float = 1e-3

    momentum: float = 0.9

    seed: int = 0



def build_dataloaders(config: TrainConfig) -> tuple[Any, Any, Any, pd.DataFrame]:

    try:

        from dataset import ASHRAEDataset, load_local_data

    except ImportError as exc:

        raise RuntimeError(

            "Training mode requires the project's dataset.py with "

            "ASHRAEDataset and load_local_data."

        ) from exc

    train_df = load_local_data("train", fraction=config.data_fraction)

    val_df = load_local_data("val", fraction=config.data_fraction)

    test_df = load_local_data("test", fraction=config.data_fraction)

    train_dataset = ASHRAEDataset(train_df)

    val_dataset = ASHRAEDataset(val_df)

    test_dataset = ASHRAEDataset(test_df)

    loader_kwargs: dict[str, Any] = {

        "num_workers": config.num_workers,

        "pin_memory": True,

    }

    if config.num_workers > 0:

        loader_kwargs.update(

            {

                "prefetch_factor": config.prefetch_factor,

                "persistent_workers": True,

            }

        )

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.batch_size,

        shuffle=True,

        drop_last=True,

        **loader_kwargs,

    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=config.batch_size,

        shuffle=False,

        drop_last=False,

        **loader_kwargs,

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=config.batch_size,

        shuffle=False,

        drop_last=False,

        **loader_kwargs,

    )

    if len(train_loader) == 0:

        raise RuntimeError("Training loader is empty; lower --batch-size or increase data")

    return train_loader, val_loader, test_loader, test_df



def evaluate_loader(model: nn.Module, loader: DataLoader) -> float:

    """Return the unweighted mean of per-batch RMSLE values.

    This is intentionally named and recorded as a batch-mean metric.  It is not

    the same statistic as the global test RMSLE, which concatenates every test

    prediction and target before taking the final square root.

    """

    model.eval()

    losses: list[torch.Tensor] = []

    with torch.no_grad():

        for batch in loader:

            x = batch["input_sequence"].to("cuda", non_blocking=True)

            target = batch["label_sequence"].to("cuda", non_blocking=True)

            losses.append(rmsle(model(x), target).detach())

    if not losses:

        return float("nan")

    return float(torch.stack(losses).mean().item())



def run_training(

    runtime: AmpereWarpRuntime,

    output_dir: Path,

    config: TrainConfig,

) -> dict[str, pd.DataFrame]:

    torch.manual_seed(config.seed)

    np.random.seed(config.seed)

    train_loader, val_loader, test_loader, _ = build_dataloaders(config)

    model = Model().to(device="cuda", dtype=torch.float32)

    optimizer = optim.SGD(

        model.parameters(),

        lr=config.learning_rate,

        momentum=config.momentum,

    )

    raw_kernel_frames: list[pd.DataFrame] = []

    batch_stage_frames: list[pd.DataFrame] = []

    kernel_epoch_rows: list[dict[str, Any]] = []

    epoch_system_rows: list[dict[str, Any]] = []

    sampler = NVMLSampler(runtime.device_id)

    for epoch_idx in range(config.epochs):

        epoch = epoch_idx + 1

        save_epoch = epoch >= 2

        model.train()

        torch.cuda.reset_peak_memory_stats()

        stage_events = StageEventCollector()

        KERNEL_EVENTS.clear()

        KERNEL_EVENTS.enabled = True

        KERNEL_EVENTS.set_context(technique=runtime.technique, epoch=epoch)

        losses: list[torch.Tensor] = []

        torch.cuda.synchronize()

        sampler.start()

        wall_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader, start=1):

            cpu_start = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)

            stage_events.add_zero_grad_cpu(

                epoch,

                batch_idx,

                (time.perf_counter() - cpu_start) * 1000.0,

            )

            transferred = stage_events.measure(

                epoch,

                batch_idx,

                "transfer",

                lambda: (

                    batch["input_sequence"].to("cuda", non_blocking=True),

                    batch["label_sequence"].to("cuda", non_blocking=True),

                ),

            )

            x, target = transferred

            prediction = stage_events.measure(

                epoch,

                batch_idx,

                "model_fwd",

                lambda: model(x),

            )

            loss = stage_events.measure(

                epoch,

                batch_idx,

                "loss",

                lambda: rmsle(prediction, target),

            )

            stage_events.measure(epoch, batch_idx, "backward", loss.backward)

            stage_events.measure(

                epoch,

                batch_idx,

                "clip_grad",

                lambda: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),

            )

            stage_events.measure(epoch, batch_idx, "optimizer_step", optimizer.step)

            losses.append(loss.detach())

        torch.cuda.synchronize()

        train_wall_s = time.perf_counter() - wall_start

        nvml_metrics = sampler.stop()

        train_loss = float(torch.stack(losses).mean().item())

        peak_memory_gib = float(torch.cuda.max_memory_allocated() / 1024**3)

        raw_kernel = KERNEL_EVENTS.resolve(synchronize=False)

        stage_frame = stage_events.resolve(synchronize=False)

        KERNEL_EVENTS.enabled = False

        # Validation is deliberately excluded from the training kernel collector.

        val_batch_mean_rmsle = evaluate_loader(model, val_loader)

        print(

            f"Epoch {epoch}/{config.epochs}: train={train_loss:.5f}, "

            f"val_batch_mean_rmsle={val_batch_mean_rmsle:.5f}, "

            f"train_wall={train_wall_s:.2f}s, "

            f"peak={peak_memory_gib:.2f}GiB"

        )

        if save_epoch:

            raw_kernel_frames.append(raw_kernel)

            batch_stage_frames.append(stage_frame.assign(technique=runtime.technique))

            summary = summarize_kernel_times(raw_kernel)

            summary_map = summary.set_index("phase")["mean_ms"].to_dict()

            kernel_epoch_rows.append(

                {

                    "technique": runtime.technique,

                    "epoch": epoch,

                    "fwd_mean_ms": summary_map.get("fwd", np.nan),

                    "bwd_input_mean_ms": summary_map.get("bwd_input", np.nan),

                    "bwd_weight_mean_ms": summary_map.get("bwd_weight", np.nan),

                    "train_loss": train_loss,

                    "val_batch_mean_rmsle": val_batch_mean_rmsle,

                    "peak_memory_gib": peak_memory_gib,

                }

            )

            epoch_system_rows.append(

                {

                    "technique": runtime.technique,

                    "epoch": epoch,

                    "train_wall_time_s": train_wall_s,

                    "peak_memory_gib": peak_memory_gib,

                    **nvml_metrics,

                }

            )

    KERNEL_EVENTS.enabled = False

    # Test RMSLE uses labels from the loader directly; no index realignment.

    model.eval()

    predictions: list[np.ndarray] = []

    targets: list[np.ndarray] = []

    with torch.no_grad():

        for batch in test_loader:

            x = batch["input_sequence"].to("cuda", non_blocking=True)

            target = batch["label_sequence"].to("cuda", non_blocking=True)

            prediction = model(x)

            predictions.append(prediction.detach().cpu().reshape(-1).numpy())

            targets.append(target.detach().cpu().reshape(-1).numpy())

    prediction_np = np.clip(np.concatenate(predictions), 0.0, None)

    target_np = np.clip(np.concatenate(targets), 0.0, None)

    test_rmsle = float(

        np.sqrt(np.mean((np.log1p(prediction_np) - np.log1p(target_np)) ** 2))

    )

    raw_kernel_columns = [

        "technique",

        "epoch",

        "B",

        "H",

        "L",

        "K",

        "phase",

        "run_id",

        "runtime_ms",

    ]

    raw_kernel_all = (

        pd.concat(raw_kernel_frames, ignore_index=True)

        if raw_kernel_frames

        else pd.DataFrame(columns=raw_kernel_columns)

    )

    # PyTorch/cuDNN produces no custom-kernel events. Preserve an explicit

    # empty schema instead of writing a zero-byte/headerless CSV.

    raw_kernel_all = raw_kernel_all.reindex(columns=raw_kernel_columns)

    batch_stage_all = (

        pd.concat(batch_stage_frames, ignore_index=True)

        if batch_stage_frames

        else pd.DataFrame()

    )

    kernel_epoch = pd.DataFrame(kernel_epoch_rows)

    epoch_system = pd.DataFrame(epoch_system_rows)

    final_test = pd.DataFrame(

        [

            {

                "technique": runtime.technique,

                "test_rmsle": test_rmsle,

                "epochs_total": config.epochs,

                "saved_epochs": f"2-{config.epochs}",

                "batch_size": config.batch_size,

                "data_fraction": config.data_fraction,

                "train_backend": TRAIN_CONV_BACKEND,

                "warps_per_block": (

                    runtime.warps_per_block

                    if TRAIN_CONV_BACKEND == "warp"

                    else np.nan

                ),

                "bwdk_btile": (

                    runtime.bwdk_btile

                    if TRAIN_CONV_BACKEND == "warp"

                    else np.nan

                ),

                "bwdk_block": (

                    runtime.bwdk_block

                    if TRAIN_CONV_BACKEND == "warp"

                    else np.nan

                ),

                "bwdk_reduction": (
                    "two_stage_no_atomic"
                    if TRAIN_CONV_BACKEND == "warp"
                    else "pytorch"
                ),

                "gpu_profile": runtime.profile.key,

                "device_name": torch.cuda.get_device_name(runtime.device_id),

                "compute_capability": (

                    f"{runtime.profile.compute_capability[0]}."

                    f"{runtime.profile.compute_capability[1]}"

                ),

            }

        ]

    )

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {

        "training_raw_kernel_times": raw_kernel_all,

        "training_batch_stage_times": batch_stage_all,

        "training_kernel_epoch_summary": kernel_epoch,

        "training_epoch_system_summary": epoch_system,

        "training_final_test_metrics": final_test,

    }

    for name, frame in outputs.items():

        frame.to_csv(output_dir / f"{name}.csv", index=False)

    print(f"Test RMSLE: {test_rmsle:.6f}")

    return outputs



# =============================================================================

# CLI

# =============================================================================

def configure_torch() -> None:

    if not torch.cuda.is_available():

        raise RuntimeError("CUDA is not available")

    torch.backends.cudnn.allow_tf32 = False

    torch.backends.cuda.matmul.allow_tf32 = False

    torch.backends.cudnn.benchmark = True



def print_device_report(runtime: AmpereWarpRuntime) -> None:

    device_id = torch.cuda.current_device()

    name = torch.cuda.get_device_name(device_id)

    capability = torch.cuda.get_device_capability(device_id)

    props = torch.cuda.get_device_properties(device_id)

    print("\n" + "=" * 88)

    print("GPU")

    print("=" * 88)

    print(f"Name:               {name}")

    print(f"Compute capability: {capability[0]}.{capability[1]}")

    print(f"SM count:           {props.multi_processor_count}")

    print(f"Total memory:       {props.total_memory / 1024**3:.2f} GiB")

    print(f"Selected profile:   {runtime.profile.key}")

    print(f"Peak BW reference:  {runtime.profile.peak_bandwidth_gbps:.1f} GB/s")



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(

        "--mode",

        choices=["validate", "benchmark", "train", "all"],

        default="benchmark",

    )

    parser.add_argument("--output-dir", default="results/warp_tiled_noatomic")

    parser.add_argument("--warps-per-block", type=int, default=DEFAULT_WARPS_PER_BLOCK)

    parser.add_argument("--bwdk-block", type=int, default=DEFAULT_BWDK_BLOCK)

    parser.add_argument(

        "--fast-math",

        action="store_true",

        help="Opt in to -use_fast_math; disabled by default to match GMC/Shared.",

    )

    parser.add_argument("--skip-correctness", action="store_true")

    parser.add_argument("--B", type=int, default=DEFAULT_B)

    parser.add_argument("--H", type=int, default=DEFAULT_H)

    parser.add_argument("--L", type=int, default=DEFAULT_L)

    parser.add_argument("--K", type=int, default=DEFAULT_K)

    parser.add_argument("--warmup", type=int, default=5)

    parser.add_argument("--runs", type=int, default=100)

    parser.add_argument(

        "--train-backend",

        choices=["warp", "pytorch"],

        default="warp",

        help=(

            "Depthwise-convolution backend used "

            "by S4 training."

        ),

    )

    parser.add_argument("--epochs", type=int, default=5)

    parser.add_argument("--data-fraction", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=DEFAULT_B)

    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--prefetch-factor", type=int, default=18)

    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()



def main() -> None:

    global TRAIN_CONV_BACKEND

    args = parse_args()

    TRAIN_CONV_BACKEND = args.train_backend

    configure_torch()

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = AmpereWarpRuntime(

        warps_per_block=args.warps_per_block,

        bwdk_btile=DEFAULT_BWDK_BTILE,

        bwdk_block=args.bwdk_block,

        fast_math=args.fast_math,

    )

    install_runtime(runtime)

    try:

        print_device_report(runtime)

        resources = runtime.resource_report()

        print("\n" + "=" * 88)

        print("COMPILER RESOURCES AND ESTIMATED GPU OCCUPANCY")

        print("=" * 88)

        print(resources.to_string(index=False))

        resources.to_csv(output_dir / "kernel_resources_occupancy.csv", index=False)

        if not args.skip_correctness:

            validate_correctness(output_dir, seed=args.seed)

        if args.mode in {"benchmark", "all"}:

            benchmark_kernels(

                runtime,

                output_dir,

                B=args.B,

                H=args.H,

                L=args.L,

                K=args.K,

                warmup=args.warmup,

                runs=args.runs,

                seed=args.seed,

            )

        if args.mode in {"train", "all"}:

            if args.train_backend == "pytorch":

                runtime.technique = (

                    f"PyTorch-cuDNN-"

                    f"{runtime.profile.key}"

                )

            run_training(

                runtime,

                output_dir,

                TrainConfig(

                    epochs=args.epochs,

                    batch_size=args.batch_size,

                    data_fraction=args.data_fraction,

                    num_workers=args.num_workers,

                    prefetch_factor=args.prefetch_factor,

                    seed=args.seed,

                ),

            )

    finally:

        KERNEL_EVENTS.enabled = False

        runtime.close()



if __name__ == "__main__":

    main()
