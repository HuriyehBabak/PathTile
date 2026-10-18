#!/usr/bin/env python3
"""Paired dK batch sweep using the existing Shared and Warp CUDA kernels.

Place this file in benchmark/ in the s4convd-paper repository.
Run each trial in a separate process on an allocated GPU:
    python -m benchmark.benchmark_dk_batch_compare --trial-id 1 --expected-gpu A6000
    python -m benchmark.benchmark_dk_batch_compare --trial-id 2 --expected-gpu A6000
    python -m benchmark.benchmark_dk_batch_compare --trial-id 3 --expected-gpu A6000
Then aggregate:
    python -m benchmark.benchmark_dk_batch_compare --mode aggregate

Defaults: B={2048,8192,16384}, H=128, L=48, K=48, three trials,
50 warm-ups and 100 measured launches per method and shape.
Both methods use identical inputs within each shape and trial.
CUDA-event intervals exclude allocation, compilation, and validation.
Warp full dK includes output zero-initialization. Shared overwrites its
output and needs no zero-initialization. These supplementary timings
must not be pooled with the paper's kernel-only primary measurements.

This is a comparison of two existing designs, not a single-factor
ablation of every difference between them. Compiled resource attributes
are recorded, but achieved occupancy is not measured by this script.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty measurement table")
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_trial(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import pycuda.driver as cuda

    if not torch.cuda.is_available():
        raise RuntimeError("Run in an allocated CUDA GPU job, not a CPU-only session")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in {(8, 0), (8, 6), (7, 0)}:
        raise RuntimeError(f"GPU {gpu}, capability {capability}, is outside the primary study")
    if args.expected_gpu and args.expected_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"Expected GPU containing {args.expected_gpu!r}, found {gpu!r}")

    trial_dir = args.out / f"trial{args.trial_id}"
    if trial_dir.exists():
        raise FileExistsError(f"{trial_dir} already exists. Choose a new --out directory")

    from kernels import shared
    from kernels import warp_tiled_ampere as warp

    warp.configure_torch()
    stream = torch.cuda.default_stream(0)
    if int(stream.cuda_stream) != 0:
        raise RuntimeError("This harness requires the CUDA default stream shared by both runtimes")

    threads = 256 if capability == (7, 0) else 128
    runtime = warp.AmpereWarpRuntime(
        warps_per_block=16, bwdk_btile=64, bwdk_block=threads, fast_math=False
    )
    raw_rows, summary_rows, correctness_rows = [], [], []
    try:
        shared_fn = shared._get_kernels()[2]
        warp_fn = runtime.bwd_weight
        shared_threads = int(shared.THREADS_PER_BLOCK)
        shared_tile = int(shared.BWD_K_B_TILE)
        if shared_tile != 64:
            raise RuntimeError(f"Expected Shared B_TILE=64, found {shared_tile}")
        with shared.CudaContextGuard():
            resources = {
                "Shared": {
                    "registers_per_thread": int(shared_fn.num_regs),
                    "static_shared_bytes": int(shared_fn.shared_size_bytes),
                    "dynamic_shared_bytes": int(shared.bwd_kernel_shared_memory_bytes(args.L)),
                    "threads_per_block": shared_threads,
                },
                "Warp": {
                    "registers_per_thread": int(warp_fn.num_regs),
                    "static_shared_bytes": int(warp_fn.shared_size_bytes),
                    "dynamic_shared_bytes": 0,
                    "threads_per_block": threads,
                },
            }

        metadata = {
            "trial_id": args.trial_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "device_name": gpu,
            "compute_capability": list(capability),
            "multiprocessors": int(torch.cuda.get_device_properties(0).multi_processor_count),
            "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "batches": args.batches,
            "H": args.H, "L": args.L, "K": args.K,
            "warmup": args.warmup, "runs": args.runs, "seed": args.seed,
            "atol": args.atol, "rtol": args.rtol,
            "batch_tile": 64, "warp_k_pack": 4,
            "timing_scope": "full_dk",
            "warp_zero_initialization_included": True,
            "cuda_default_stream_handle": int(stream.cuda_stream),
            "kernel_sha256": {"Shared": digest(shared.__file__), "Warp": digest(warp.__file__)},
            "resources": resources,
        }
        trial_dir.mkdir(parents=True, exist_ok=False)
        (trial_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        batches = list(args.batches)
        if args.trial_id % 2 == 0:
            batches.reverse()

        with torch.cuda.stream(stream), shared.CudaContextGuard():
            for index, B in enumerate(batches):
                seed = args.seed + B
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                x = torch.randn(B, args.H, args.L, device="cuda", dtype=torch.float32)
                gy = torch.randn_like(x)
                weight = torch.randn(args.H, args.K, device="cuda", dtype=torch.float32,
                                     requires_grad=True)
                y = F.conv1d(
                    F.pad(x, (args.K // 2, args.K - 1 - args.K // 2)),
                    weight.unsqueeze(1), groups=args.H,
                )
                reference = torch.autograd.grad(y, weight, grad_outputs=gy)[0].detach()
                del y, weight
                out_shared = torch.empty_like(reference)
                out_warp = torch.empty_like(reference)
                ptr = lambda tensor: np.uintp(tensor.data_ptr())
                dimensions = tuple(np.int32(v) for v in (B, args.H, args.L, args.K))
                ntiles = (B + 63) // 64
                shared_grid = ((args.K + shared_threads - 1) // shared_threads, args.H, 1)

                def launch_shared():
                    shared_fn(
                        ptr(gy), ptr(x), ptr(out_shared), *dimensions,
                        block=(shared_threads, 1, 1), grid=shared_grid,
                        shared=resources["Shared"]["dynamic_shared_bytes"],
                    )

                def launch_warp():
                    out_warp.zero_()
                    warp_fn(
                        ptr(gy), ptr(x), ptr(out_warp), *dimensions, np.int32(ntiles),
                        block=(threads, 1, 1), grid=(args.H, ntiles, 1),
                    )

                launches = {"Shared": launch_shared, "Warp": launch_warp}
                outputs = {"Shared": out_shared, "Warp": out_warp}
                blocks = {"Shared": math.prod(shared_grid), "Warp": args.H * ntiles}
                for method in ("Shared", "Warp"):
                    outputs[method].fill_(float("nan"))
                    launches[method]()
                    torch.cuda.synchronize()
                    result = outputs[method]
                    difference = float((result - reference).abs().max().item())
                    scale = max(float(reference.abs().max().item()), 1.0)
                    passed = bool(torch.allclose(result, reference, atol=args.atol, rtol=args.rtol))
                    correctness_rows.append({
                        "trial_id": args.trial_id, "B": B, "H": args.H, "L": args.L,
                        "K": args.K, "method": method, "validation_B": B,
                        "max_abs_error": difference, "normalized_max_abs": difference / scale,
                        "atol": args.atol, "rtol": args.rtol, "passed": passed,
                    })
                    if not passed:
                        write_csv(trial_dir / "correctness.csv", correctness_rows)
                        raise RuntimeError(
                            f"{method} correctness FAILED at B={B}, max_abs={difference:.6g}"
                        )

                del reference
                order = ["Shared", "Warp"]
                if (args.trial_id + index) % 2 == 0:
                    order.reverse()
                for method in order:
                    launch = launches[method]
                    for _ in range(args.warmup):
                        launch()
                    torch.cuda.synchronize()
                    pairs = [(cuda.Event(), cuda.Event()) for _ in range(args.runs)]
                    for start, end in pairs:
                        start.record()
                        launch()
                        end.record()
                    pairs[-1][1].synchronize()
                    values = [float(start.time_till(end)) for start, end in pairs]
                    if any(not math.isfinite(v) or v <= 0 for v in values):
                        raise RuntimeError(f"Invalid CUDA-event timings for {method}, B={B}")
                    common = {
                        "trial_id": args.trial_id,
                        "config_id": f"batch_b{B}_h{args.H}_l{args.L}_k{args.K}",
                        "B": B, "H": args.H, "L": args.L, "K": args.K,
                        "device_name": gpu, "method": method, "phase": "bwd_weight",
                        "timing_scope": "full_dk", "seed": seed,
                        "grid_blocks": blocks[method], **resources[method],
                    }
                    for run_id, value in enumerate(values, 1):
                        raw_rows.append({**common, "run_id": run_id, "runtime_ms": value})
                    mean = statistics.mean(values)
                    summary_rows.append({
                        **common, "runs": args.runs, "warmup": args.warmup,
                        "mean_ms": mean, "within_trial_std_ms": statistics.stdev(values),
                    })
                    print(f"trial={args.trial_id} B={B:5d} {method:6s} "
                          f"full_dK={mean:.6f} ms blocks={blocks[method]}", flush=True)
                    del pairs
                del launches, outputs, out_shared, out_warp, x, gy
                gc.collect()
                torch.cuda.empty_cache()

        write_csv(trial_dir / "raw.csv", raw_rows)
        write_csv(trial_dir / "trial_summary.csv", summary_rows)
        write_csv(trial_dir / "correctness.csv", correctness_rows)
        print(f"Saved: {trial_dir}", flush=True)
    finally:
        runtime.close()


def aggregate(args):
    frames, signatures = [], []
    for trial_id in range(1, args.expected_trials + 1):
        directory = args.out / f"trial{trial_id}"
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata["trial_id"] != trial_id:
            raise ValueError(f"Trial identity mismatch in {directory}")
        signatures.append({
            key: metadata[key] for key in (
                "device_name", "compute_capability", "torch_version", "cuda_version",
                "cudnn_version", "batches", "H", "L", "K", "warmup", "runs", "seed",
                "atol", "rtol", "batch_tile", "warp_k_pack", "timing_scope",
                "warp_zero_initialization_included", "kernel_sha256", "resources",
            )
        })
        with (directory / "correctness.csv").open(newline="") as f:
            checks = list(csv.DictReader(f))
        if len(checks) != 2 * len(metadata["batches"]) or any(
                row["passed"] != "True" for row in checks):
            raise ValueError(f"Missing or failed correctness checks in {directory}")
        with (directory / "trial_summary.csv").open(newline="") as f:
            rows = list(csv.DictReader(f))
        expected = {(B, method) for B in metadata["batches"] for method in ("Shared", "Warp")}
        observed = [(int(row["B"]), row["method"]) for row in rows]
        if set(observed) != expected or len(observed) != len(expected):
            raise ValueError(f"Incomplete or duplicate measurements in {directory}")
        if any(int(row["trial_id"]) != trial_id or row["timing_scope"] != "full_dk"
               for row in rows):
            raise ValueError(f"Inconsistent timing scope or trial identity in {directory}")
        frames.extend(rows)
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError("GPU, configuration, source, or measurement protocol differs across trials")
    output = []
    for B in sorted(signatures[0]["batches"]):
        paired = {}
        for trial_id in range(1, args.expected_trials + 1):
            paired[trial_id] = {
                row["method"]: float(row["mean_ms"]) for row in frames
                if int(row["B"]) == B and int(row["trial_id"]) == trial_id
            }
        s = [paired[i]["Shared"] for i in paired]
        w = [paired[i]["Warp"] for i in paired]
        if any(not math.isfinite(v) or v <= 0 for v in s + w):
            raise ValueError("Invalid trial means")
        output.append({
            "B": B, "H": signatures[0]["H"], "L": signatures[0]["L"],
            "K": signatures[0]["K"], "device_name": signatures[0]["device_name"],
            "phase": "bwd_weight", "timing_scope": "full_dk",
            "trials": args.expected_trials,
            "shared_mean_ms": statistics.mean(s),
            "shared_between_trial_std_ms": statistics.stdev(s),
            "warp_full_mean_ms": statistics.mean(w),
            "warp_between_trial_std_ms": statistics.stdev(w),
            "warp_speedup_geomean": math.exp(statistics.mean(
                math.log(a / b) for a, b in zip(s, w))),
        })
    target = args.out / "aggregate_comparison.csv"
    write_csv(target, output)
    for row in output:
        print(f"B={row['B']:5d} Shared={row['shared_mean_ms']:.6f} ms "
              f"Warp(full)={row['warp_full_mean_ms']:.6f} ms "
              f"paired speedup={row['warp_speedup_geomean']:.4f}x")
    print(f"Saved: {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("run", "aggregate"), default="run")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--batches", type=int, nargs="+", default=[2048, 8192, 16384])
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--expected-gpu", default="")
    parser.add_argument("--out", type=Path, default=Path("results/dk_batch_compare/a6000"))
    args = parser.parse_args()
    if min(args.H, args.L, args.K, *args.batches) <= 0 or max(args.L, args.K) > 48:
        parser.error("Positive dimensions and L,K<=48 are required")
    if args.warmup < 0 or args.runs < 2 or args.trial_id < 1 or args.expected_trials < 2:
        parser.error("Use warmup>=0, runs>=2, trial-id>=1, and expected-trials>=2")
    if len(args.batches) != len(set(args.batches)):
        parser.error("Batch sizes must be distinct")
    if not math.isfinite(args.atol) or not math.isfinite(args.rtol) or min(args.atol, args.rtol) < 0:
        parser.error("Finite, nonnegative correctness tolerances are required")
    if args.mode == "run":
        run_trial(args)
    else:
        aggregate(args)


if __name__ == "__main__":
    main()
