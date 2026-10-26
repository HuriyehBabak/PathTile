"""Batch-tile/block-size experiment; tap-pack=4."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from benchmark.benchmark_dk_packing import (
    load_template, make_source, write_csv,
)

TILES = (32, 64)
FUNCTION = "dwconv1d_warptile_bwd_weight"


def build_sources(path, block):
    template, kernel_hash = load_template(path)
    baseline = make_source(template, pack=4, block=block)
    marker = "#define BWDK_BTILE 64"
    if baseline.count(marker) != 1:
        raise ValueError("Unexpected batch-tile definition")
    sources = {}
    for tile in TILES:
        generated = baseline.replace(
            marker, f"#define BWDK_BTILE {tile}", 1
        )
        if generated.replace(
            f"#define BWDK_BTILE {tile}", marker, 1
        ) != baseline:
            raise AssertionError("Unexpected source changes")
        sources[tile] = generated
    return sources, kernel_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--L", type=int, default=48)
    parser.add_argument("--K", type=int, default=48)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--block-size", type=int, choices=(64, 128, 256), default=128)
    parser.add_argument("--check-source", action="store_true")
    parser.add_argument("--out", type=Path,
                        default=Path("results/dk_btile_block"))
    args = parser.parse_args()
    if min(args.B, args.H, args.L, args.K, args.runs) < 1:
        parser.error("Dimensions and runs must be positive")
    if args.trials < 2 or args.warmup < 0:
        parser.error("Use at least two trials and nonnegative warmup")
    if args.L > 48 or args.K > 48:
        parser.error("L and K must be at most 48")

    kernel_path = Path("kernels/warp_tiled_ampere.py")
    sources, kernel_hash = build_sources(kernel_path, args.block_size)
    if args.check_source:
        for tile in TILES:
            print(f"Source PASS: tile={tile}, block={args.block_size}, pack=4")
        print("No CUDA compilation or GPU execution performed.")
        return

    import torch
    import torch.nn.functional as F
    import pycuda
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    torch.cuda.set_device(args.device)
    gpu = torch.cuda.get_device_name(args.device)
    if "RTX A6000" not in gpu:
        raise RuntimeError(f"Initial experiment requires RTX A6000; got {gpu}")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=False)
    options = ["-O3", "-arch=sm_86", "-lineinfo", "-Xptxas=-v",
               "-ftz=true", "-prec-div=false", "-prec-sqrt=false"]
    meta = {
        "status": "started", "gpu": gpu,
        "shape": [args.B, args.H, args.L, args.K],
        "tiles": list(TILES), "block": args.block_size, "pack": 4,
        "trials": args.trials, "warmup": args.warmup,
        "runs": args.runs, "seed": args.seed,
        "atol": 1e-2, "rtol": 5e-4,
        "kernel_sha256": kernel_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "pycuda": pycuda.VERSION_TEXT, "compiler_options": options,
        "kernel_scope": "zero before start event on same stream",
        "full_scope": "zero and kernel inside event interval",
        "resource_note": "resource counts are not achieved occupancy",
    }
    for tile, source in sources.items():
        (out / f"dk_btile{tile}.cu").write_text(source)
    raw, trials, checks, resources = [], [], [], []
    modules, functions = {}, {}
    cuda.init()
    context = cuda.Device(args.device).retain_primary_context()
    context.push()
    stream = None
    try:
        stream = cuda.Stream()
        for tile in TILES:
            module = SourceModule(sources[tile], options=options,
                                  no_extern_c=True)
            fn = module.get_function(FUNCTION)
            fn.prepare("PPPiiiii")
            modules[tile], functions[tile] = module, fn
            resources.append({
                "tile": tile, "registers_per_thread": int(fn.num_regs),
                "shared_memory_bytes": int(fn.shared_size_bytes),
                "local_memory_bytes": int(fn.local_size_bytes),
                "grid_blocks": args.H * math.ceil(args.B / tile),
                "atomic_updates": args.H * args.K * math.ceil(args.B / tile),
            })
        write_csv(out / "resources.csv", resources)
        print(f"GPU: {gpu}\nOutput: {out}", flush=True)

        def inputs(shape, seed):
            B, H, L, K = shape
            torch.manual_seed(seed)
            x = torch.randn(B, H, L, device=f"cuda:{args.device}")
            gy = torch.randn_like(x)
            gk = torch.empty(H, K, device=x.device)
            padded = F.pad(x, (K // 2, K - 1 - K // 2))
            ref = torch.nn.grad.conv1d_weight(
                padded, (H, 1, K), gy, groups=H
            ).squeeze(1).detach()
            torch.cuda.synchronize(args.device)
            return x, gy, gk, ref

        def zero(tensors):
            cuda.memset_d32_async(
                tensors[2].data_ptr(), 0, tensors[2].numel(), stream
            )

        def launch(tile, tensors, shape):
            B, H, L, K = shape
            count = math.ceil(B / tile)
            functions[tile].prepared_async_call(
                (H, count, 1), (args.block_size, 1, 1), stream,
                tensors[1].data_ptr(), tensors[0].data_ptr(),
                tensors[2].data_ptr(), B, H, L, K, count,
            )

        def check(tile, tensors, shape, phase):
            stream.synchronize()
            actual, ref = tensors[2], tensors[3]
            finite = bool(torch.isfinite(actual).all().item())
            ok = finite and torch.allclose(
                actual, ref, atol=1e-2, rtol=5e-4
            )
            checks.append({
                "tile": tile, "shape": str(shape), "phase": phase,
                "allclose": bool(ok),
                "max_abs": float((actual-ref).abs().max().item()),
            })
            write_csv(out / "correctness.csv", checks)
            if not ok:
                raise RuntimeError(f"Correctness failed: tile={tile}, {phase}")

        primary = (args.B, args.H, args.L, args.K)
        for index, shape in enumerate(((1, 1, 1, 1), (65, 3, 17, 13), primary)):
            tensors = inputs(shape, args.seed + index)
            for tile in TILES:
                zero(tensors)
                launch(tile, tensors, shape)
                check(tile, tensors, shape, "pre_timing")
            if shape != primary:
                del tensors
        print("Correctness PASS: all tiles, primary and edge cases", flush=True)

        for trial in range(1, args.trials + 1):
            offset = (trial - 1) % len(TILES)
            order = TILES[offset:] + TILES[:offset]
            scopes = ("kernel", "full") if trial % 2 else ("full", "kernel")
            for tile in order:
                for scope in scopes:
                    for _ in range(args.warmup):
                        zero(tensors)
                        launch(tile, tensors, primary)
                    stream.synchronize()
                    pairs = [(cuda.Event(), cuda.Event()) for _ in range(args.runs)]
                    for start, end in pairs:
                        if scope == "kernel":
                            zero(tensors)
                        start.record(stream)
                        if scope == "full":
                            zero(tensors)
                        launch(tile, tensors, primary)
                        end.record(stream)
                    stream.synchronize()
                    times = [float(a.time_till(b)) for a, b in pairs]
                    check(tile, tensors, primary, f"trial{trial}_{scope}")
                    raw.extend({
                        "trial": trial, "tile": tile, "scope": scope,
                        "run": i, "runtime_ms": value
                    } for i, value in enumerate(times, 1))
                    row = {
                        "trial": trial, "tile": tile, "scope": scope,
                        "mean_ms": statistics.mean(times),
                        "within_trial_std_ms": statistics.stdev(times)
                            if len(times) > 1 else 0.0,
                    }
                    trials.append(row)
                    write_csv(out / "raw.csv", raw)
                    write_csv(out / "trial_summary.csv", trials)
                    print(f"trial={trial} tile={tile} {scope}: "
                          f"{row['mean_ms']:.6f} ms", flush=True)

        summary = []
        for scope in ("kernel", "full"):
            base = {r["trial"]: r["mean_ms"] for r in trials
                    if r["tile"] == 64 and r["scope"] == scope}
            for tile in TILES:
                selected = [r for r in trials
                            if r["tile"] == tile and r["scope"] == scope]
                means = [r["mean_ms"] for r in selected]
                ratios = [base[r["trial"]] / r["mean_ms"] for r in selected]
                row = {
                    "tile": tile, "scope": scope, "trials": len(means),
                    "mean_ms": statistics.mean(means),
                    "between_trial_std_ms": statistics.stdev(means),
                    "paired_geomean_speedup_vs_tile64":
                        math.exp(statistics.mean(map(math.log, ratios))),
                }
                summary.append(row)
                print(row, flush=True)
        write_csv(out / "summary.csv", summary)
        meta["status"] = "complete"
    except BaseException as error:
        meta["status"] = "failed"
        meta["error"] = str(error)
        raise
    finally:
        try:
            if stream is not None:
                stream.synchronize()
        finally:
            functions.clear()
            modules.clear()
            cuda.Context.pop()
            context.detach()
            (out / "metadata.json").write_text(
                json.dumps(meta, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
