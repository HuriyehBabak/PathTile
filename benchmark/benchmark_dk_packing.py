#!/usr/bin/env python3
"""Controlled dK packing ablation for the existing PathTile CUDA kernel.

Place in benchmark/ and run from the s4convd-paper repository root:
    python -m benchmark.benchmark_dk_packing --trials 3 --warmup 50 --runs 100

The original kernel file is only read. Pack=4 preserves its CUDA template;
pack=1/2 remove unused tap lanes from that same explicitly unrolled kernel.
All variants compute the full dK tensor with the same grid, block, inputs,
batch tile, compiler flags, and reduction/atomic strategy. Packing necessarily
changes the number of tap-group iterations, barriers, and live accumulators.

Results are supplementary within-harness comparisons, not replacements for
the primary paper measurements. Both kernel-only and zero-fill-plus-kernel
times are recorded on one explicit PyCUDA stream. Allocations, compilation,
and correctness checks are outside timing. No training dataset is needed.

CPU-only source verification (does not compile or run CUDA):
    python -m benchmark.benchmark_dk_packing --check-source
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import hashlib
import json
import math
import platform
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


PACKS = (1, 2, 4)
BTILE = 64
FUNCTION_NAME = "dwconv1d_warptile_bwd_weight"


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_template(path: Path) -> tuple[str, str]:
    text = path.read_text(encoding="utf-8")
    values = [
        node.value
        for node in ast.parse(text).body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "CUDA_TEMPLATE" for t in node.targets)
    ]
    if len(values) != 1:
        raise ValueError("Expected one literal CUDA_TEMPLATE in the kernel file")
    template = ast.literal_eval(values[0])
    if not isinstance(template, str):
        raise ValueError("CUDA_TEMPLATE must be a literal string")
    return template, sha256(text)


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Unsupported kernel layout: expected exactly one {old!r}")
    return text.replace(old, new, 1)


def remove_tap_branch(body: str, tap: int) -> str:
    marker = f"            if (j{tap} < K) {{"
    if body.count(marker) != 1:
        raise ValueError(f"Unsupported tap-{tap} branch in CUDA kernel")
    start = body.index(marker)
    brace = body.index("{", start)
    depth = 1
    end = brace + 1
    while end < len(body) and depth:
        depth += (body[end] == "{") - (body[end] == "}")
        end += 1
    if depth:
        raise ValueError("Unbalanced braces in CUDA tap branch")
    if body[end:end + 1] == "\n":
        end += 1
    return body[:start] + body[end:]


def make_source(template: str, pack: int, block: int) -> str:
    if pack not in PACKS or block < 32 or block > 1024 or block % 32:
        raise ValueError("Pack must be 1/2/4; block must be a multiple of 32 in [32,1024]")
    source = replace_once(template, "#define BWDK_KPACK 4", f"#define BWDK_KPACK {pack}")
    marker = f"__global__ void {FUNCTION_NAME}("
    if source.count(marker) != 1:
        raise ValueError("Could not uniquely locate the existing dK kernel")
    offset = source.index(marker)
    prefix, body = source[:offset], source[offset:]
    for tap in range(pack, 4):
        body = remove_tap_branch(body, tap)
        for fragment in (
            f"        float acc{tap} = 0.0f;\n",
            f"            const int j{tap} = k0 + {tap};\n",
            f"        acc{tap} = warp_reduce_sum(acc{tap});\n",
            f"            sWarpSums[{tap}][warp_id] = acc{tap};\n",
            f"            float value{tap} = lane < num_warps ? sWarpSums[{tap}][lane] : 0.0f;\n",
            f"            value{tap} = warp_reduce_sum(value{tap});\n",
            f"                if (k0 + {tap} < K) atomicAdd(&gk[h * K + k0 + {tap}], value{tap});\n",
        ):
            body = replace_once(body, fragment, "")
        if re.search(rf"\b(?:acc{tap}|j{tap}|value{tap})\b", body):
            raise ValueError(f"An unused tap-{tap} reference remains")
    source = prefix + body
    for token, value in (
        ("__WARPS_PER_BLOCK__", 16),
        ("__BWDK_BTILE__", BTILE),
        ("__BWDK_BLOCK__", block),
    ):
        source = replace_once(source, token, str(value))
    if pack == 4:
        baseline = template.replace("__WARPS_PER_BLOCK__", "16").replace(
            "__BWDK_BTILE__", str(BTILE)
        ).replace("__BWDK_BLOCK__", str(block))
        if source != baseline:
            raise AssertionError("Pack=4 must preserve the original generated CUDA source")
    return source


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def summarize(trial_rows: list[dict], resources: list[dict]) -> list[dict]:
    result = []
    for scope in ("kernel", "full"):
        baseline = {r["trial"]: r["mean_ms"] for r in trial_rows if r["pack"] == 1 and r["scope"] == scope}
        for pack in PACKS:
            selected = [r for r in trial_rows if r["pack"] == pack and r["scope"] == scope]
            means = [r["mean_ms"] for r in selected]
            ratios = [baseline[r["trial"]] / r["mean_ms"] for r in selected]
            resource = next(r for r in resources if r["pack"] == pack)
            result.append({
                "scope": scope, "pack": pack, "trials": len(means),
                "mean_ms": statistics.mean(means),
                "between_trial_std_ms": statistics.stdev(means) if len(means) > 1 else 0.0,
                "paired_geomean_speedup_vs_pack1": math.exp(statistics.mean(map(math.log, ratios))),
                "paired_speedup_min": min(ratios), "paired_speedup_max": max(ratios),
                "registers_per_thread": resource["registers_per_thread"],
                "shared_memory_bytes": resource["shared_memory_bytes"],
                "local_memory_bytes": resource["local_memory_bytes"],
            })
    return result


def run_gpu(args: argparse.Namespace, template: str, kernel_file_hash: str) -> None:
    # Keep GPU dependencies optional for --help and --check-source.
    import torch
    import torch.nn.functional as F
    import pycuda
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    if not torch.cuda.is_available():
        raise RuntimeError("Run this benchmark inside an allocated CUDA GPU job")
    torch.cuda.set_device(args.device)
    torch.cuda.init()
    cc = torch.cuda.get_device_capability(args.device)
    if cc not in {(7, 0), (8, 0), (8, 6)}:
        raise ValueError(f"This initial experiment targets V100/A100/RTX A6000, got sm_{cc[0]}{cc[1]}")
    block = args.block_size or (256 if cc == (7, 0) else 128)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out_dir = args.out / f"sm{cc[0]}{cc[1]}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    options = ["-O3", f"-arch=sm_{cc[0]}{cc[1]}", "-lineinfo", "-Xptxas=-v",
               "-ftz=true", "-prec-div=false", "-prec-sqrt=false"]
    sources = {pack: make_source(template, pack, block) for pack in PACKS}
    metadata = {
        "status": "started", "gpu": torch.cuda.get_device_name(args.device),
        "compute_capability": list(cc), "python": platform.python_version(),
        "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(), "pycuda_version": pycuda.VERSION_TEXT,
        "shape": [args.B, args.H, args.L, args.K], "packs": list(PACKS),
        "block_size": block, "batch_tile": BTILE, "seed": args.seed,
        "trials": args.trials, "warmup": args.warmup, "runs": args.runs,
        "compiler_options": options, "kernel_file": str(args.kernel_file.resolve()),
        "kernel_file_sha256": kernel_file_hash, "template_sha256": sha256(template),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cuda_source_sha256": {str(pack): sha256(sources[pack]) for pack in PACKS},
        "timing_stream": "one explicit PyCUDA stream for memset, kernels, and events",
        "scope_kernel": "dK kernel only; output zeroed before start event on same stream",
        "scope_full": "output zero-fill plus dK kernel in one event interval",
        "reference": "PyTorch FP32 conv1d_weight, TF32 disabled, asymmetric zero padding",
        "atol": args.atol, "rtol": args.rtol,
        "occupancy": "not measured; resource counts are not achieved occupancy",
        "source_level_counts": {
            str(pack): {"tap_groups": math.ceil(args.K / pack),
                        "block_barriers_per_block": 1 + 2 * math.ceil(args.K / pack),
                        "logical_gy_reads_total": args.B * args.H * args.L * math.ceil(args.K / pack),
                        "atomic_updates_total": args.H * args.K * math.ceil(args.B / BTILE)}
            for pack in PACKS
        },
    }
    for pack, source in sources.items():
        (out_dir / f"dk_pack{pack}.cu").write_text(source, encoding="utf-8")
    write_json(out_dir / "metadata.json", metadata)
    cuda.init()
    context = cuda.Device(args.device).retain_primary_context()
    context.push()
    modules, functions, resources = {}, {}, []
    raw, trial_rows, correctness = [], [], []
    stream = None
    try:
        metadata["cuda_driver_version"] = cuda.get_driver_version()
        stream = cuda.Stream()
        for pack in PACKS:
            module = SourceModule(sources[pack], options=options, no_extern_c=True)
            fn = module.get_function(FUNCTION_NAME)
            fn.prepare("PPPiiiii")
            modules[pack], functions[pack] = module, fn
            resources.append({"pack": pack, "registers_per_thread": int(fn.num_regs),
                              "shared_memory_bytes": int(fn.shared_size_bytes),
                              "local_memory_bytes": int(fn.local_size_bytes)})
        write_csv(out_dir / "resources.csv", resources)
        print(f"GPU: {metadata['gpu']} | block={block}, B_TILE={BTILE}", flush=True)
        print(f"Output: {out_dir}", flush=True)
        for row in resources:
            print(f"pack={row['pack']}: regs={row['registers_per_thread']}, "
                  f"smem={row['shared_memory_bytes']} B, local={row['local_memory_bytes']} B", flush=True)

        def inputs(shape: tuple[int, int, int, int], seed: int):
            B, H, L, K = shape
            torch.manual_seed(seed)
            x = torch.randn((B, H, L), device="cuda", dtype=torch.float32)
            gy = torch.randn_like(x)
            gk = torch.empty((H, K), device="cuda", dtype=torch.float32)
            padded = F.pad(x, (K // 2, K - 1 - K // 2))
            ref = torch.nn.grad.conv1d_weight(padded, (H, 1, K), gy, groups=H).squeeze(1).detach()
            torch.cuda.synchronize()
            return x, gy, gk, ref

        def launch(pack, tensors, shape):
            x, gy, gk, _ = tensors
            B, H, L, K = shape
            tiles = math.ceil(B / BTILE)
            functions[pack].prepared_async_call(
                (H, tiles, 1), (block, 1, 1), stream,
                gy.data_ptr(), x.data_ptr(), gk.data_ptr(), B, H, L, K, tiles)

        def zero(tensors):
            cuda.memset_d32_async(tensors[2].data_ptr(), 0, tensors[2].numel(), stream)

        def check(pack, tensors, shape, phase):
            stream.synchronize()
            actual, ref = tensors[2], tensors[3]
            error = (actual - ref).abs()
            finite = bool(torch.isfinite(actual).all().item())
            max_abs = float(error.max().item()) if finite else None
            norm = float(ref.abs().max().item())
            ok = finite and bool(torch.allclose(actual, ref, atol=args.atol, rtol=args.rtol))
            correctness.append({"phase": phase, "pack": pack, "B": shape[0], "H": shape[1],
                                "L": shape[2], "K": shape[3], "allclose": ok,
                                "max_abs": max_abs,
                                "normalized_max_abs": max_abs / max(norm, 1e-30) if finite else None})
            write_csv(out_dir / "correctness.csv", correctness)
            if not ok:
                raise RuntimeError(f"Correctness failed: {phase}, pack={pack}, shape={shape}, max_abs={max_abs}")

        # Nondivisible B/K and a one-tap edge case exercise the removed lanes.
        for case_id, shape in enumerate(((1, 1, 1, 1), (65, 3, 17, 13))):
            tensors = inputs(shape, args.seed + case_id + 1)
            for pack in PACKS:
                zero(tensors)
                launch(pack, tensors, shape)
                check(pack, tensors, shape, "edge_validation")
            del tensors
        shape = (args.B, args.H, args.L, args.K)
        tensors = inputs(shape, args.seed)
        for pack in PACKS:
            zero(tensors)
            launch(pack, tensors, shape)
            check(pack, tensors, shape, "main_validation")
        print("Correctness: PASS for all packs on main and edge cases", flush=True)

        def measure(pack: int, scope: str) -> list[float]:
            for _ in range(args.warmup):
                zero(tensors)
                launch(pack, tensors, shape)
            stream.synchronize()
            pairs = [(cuda.Event(), cuda.Event()) for _ in range(args.runs)]
            for start, end in pairs:
                if scope == "kernel":
                    zero(tensors)
                start.record(stream)
                if scope == "full":
                    zero(tensors)
                launch(pack, tensors, shape)
                end.record(stream)
            stream.synchronize()
            return [float(start.time_till(end)) for start, end in pairs]

        for trial in range(1, args.trials + 1):
            offset = (trial - 1) % len(PACKS)
            order = PACKS[offset:] + PACKS[:offset]
            scopes = ("kernel", "full") if trial % 2 else ("full", "kernel")
            for pack in order:
                for scope in scopes:
                    times = measure(pack, scope)
                    check(pack, tensors, shape, f"after_trial_{trial}_{scope}")
                    raw.extend({"trial": trial, "pack": pack, "scope": scope,
                                "run": index, "runtime_ms": ms} for index, ms in enumerate(times, 1))
                    row = {"trial": trial, "pack": pack, "scope": scope,
                           "mean_ms": statistics.mean(times),
                           "within_trial_std_ms": statistics.stdev(times) if len(times) > 1 else 0.0}
                    trial_rows.append(row)
                    print(f"trial={trial} pack={pack} {scope:6s}: {row['mean_ms']:.6f} ms", flush=True)
            write_csv(out_dir / "raw.csv", raw)
            write_csv(out_dir / "trial_summary.csv", trial_rows)
        summary = summarize(trial_rows, resources)
        write_csv(out_dir / "summary.csv", summary)
        print("\nMean of trial means; SD is between trials; speedup baseline is pack=1:", flush=True)
        for row in summary:
            print(f"{row['scope']:6s} pack={row['pack']}: {row['mean_ms']:.6f} "
                  f"+/- {row['between_trial_std_ms']:.6f} ms | "
                  f"speedup={row['paired_geomean_speedup_vs_pack1']:.4f}x", flush=True)
        metadata["status"] = "complete"
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = str(error)
        raise
    finally:
        write_json(out_dir / "metadata.json", metadata)
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            functions.clear()
            modules.clear()
            gc.collect()
            cuda.Context.pop()
            context.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel-file", type=Path, default=Path("kernels/warp_tiled_ampere.py"))
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument("--block-size", type=int, help="Fixed across packs; default 128 on Ampere, 256 on V100")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--out", type=Path, default=Path("results/dk_packing"))
    parser.add_argument("--check-source", action="store_true", help="Verify source transformations without GPU dependencies")
    parser.add_argument("--emit-dir", type=Path, help="With --check-source, save generated sources for inspection")
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in ("B", "H", "L", "K", "trials", "runs")) or args.warmup < 0:
        parser.error("Dimensions/trials/runs must be positive; warmup must be nonnegative")
    if args.L > 48 or args.K > 48:
        parser.error("The existing CUDA kernel supports only L,K <= 48")
    if args.block_size is not None and (args.block_size < 32 or args.block_size > 1024 or args.block_size % 32):
        parser.error("--block-size must be a multiple of 32 in [32,1024]")
    if args.atol < 0 or args.rtol < 0 or not math.isfinite(args.atol + args.rtol):
        parser.error("Tolerances must be finite and nonnegative")
    template, kernel_file_hash = load_template(args.kernel_file)
    # Validate the exact source layout before importing or allocating GPU resources.
    sources = {pack: make_source(template, pack, args.block_size or 128) for pack in PACKS}
    if args.check_source:
        if args.emit_dir is not None:
            args.emit_dir.mkdir(parents=True, exist_ok=True)
            for pack, source in sources.items():
                (args.emit_dir / f"dk_pack{pack}.cu").write_text(source, encoding="utf-8")
        print("Source checks PASS: pack=4 is unchanged; unused tap lanes removed for pack=1/2.")
        print("CUDA compilation and numerical validation still require the GPU run.")
        return
    if args.emit_dir is not None:
        parser.error("--emit-dir is only used with --check-source")
    run_gpu(args, template, kernel_file_hash)


if __name__ == "__main__":
    main()
