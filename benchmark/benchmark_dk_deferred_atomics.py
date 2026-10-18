#!/usr/bin/env python3
"""Exploratory comparison of the existing Warp dK and a deferred atomic epilogue.

Install this separate file in benchmark/ in the original s4convd-paper repo.
Run on an allocated GPU with the original s4convd Python environment:
    python -m benchmark.benchmark_dk_deferred_atomics --expected-gpu A6000

The original kernel file is read and imported, never rewritten. The variant
keeps B_TILE=64, K_PACK=4, arithmetic, reductions, and launch dimensions.
It buffers block results in 48 shared floats and lets tid<K issue one atomic
update per coefficient after the packing loop. Scalar atomic count is unchanged.

Before timing, both variants must pass the existing Warp dK tolerances against
PyTorch FP32 and an independent direct FP64 reference over the FULL batch.
Two full-batch seeds, three executions each, and two small boundary cases are
checked. A few direct FP64 coefficients are audited with CPU math.fsum.
CUDA-event timing includes zero-initialization plus dK for BOTH variants.
This is one exploratory trial, not a demonstrated new algorithm or paper result.
"""
from __future__ import annotations

import argparse
import ast
import csv
import difflib
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


REVIEWED_SHA256 = "8f3bbc6e43a590255d2c93218e61d4cb07a56b6d34d3f7a19466742f17a01144"
ATOL, RTOL = 1e-2, 5e-4
ORIGINAL_NAME = "dwconv1d_warptile_bwd_weight"
VARIANT_NAME = "dwconv1d_warptile_bwd_weight_deferred"
METHODS = ("Baseline", "Deferred")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_template(path):
    data = path.read_bytes()
    if sha256(data) != REVIEWED_SHA256:
        raise RuntimeError("The Warp kernel source differs from the reviewed source. "
                           "Do not bypass this guard. Provide the current file for review.")
    tree = ast.parse(data.decode("utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CUDA_TEMPLATE" for t in node.targets
        ):
            template = ast.literal_eval(node.value)
            if not isinstance(template, str):
                break
            return template
    raise RuntimeError("CUDA_TEMPLATE was not found as a literal string")


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("A reviewed CUDA source anchor is missing or ambiguous")
    return text.replace(old, new, 1)


def deferred_template(template):
    start = template.index("__global__ void " + ORIGINAL_NAME + "(")
    # dK is the final function in the reviewed CUDA_TEMPLATE.
    end = template.index('\n} // extern "C"', start)
    original = template[start:end]
    modified = replace_once(original, ORIGINAL_NAME, VARIANT_NAME)
    anchor = "    __shared__ float sWarpSums[BWDK_KPACK][BWDK_BLOCK / WARP];"
    modified = replace_once(modified, anchor,
                            anchor + "\n    __shared__ float sBlockSums[MAX_K];")
    for tap in range(4):
        old = (f"if (k0 + {tap} < K) "
               f"atomicAdd(&gk[h * K + k0 + {tap}], value{tap});")
        new = f"if (k0 + {tap} < K) sBlockSums[k0 + {tap}] = value{tap};"
        modified = replace_once(modified, old, new)
    ending = "        __syncthreads();\n    }\n}\n"
    if not modified.endswith(ending):
        raise RuntimeError("Unexpected dK function ending")
    epilogue = ("        __syncthreads();\n    }\n\n"
                "    // The final packing-loop barrier makes all stored sums visible.\n"
                "    if (tid < K) atomicAdd(&gk[h * K + tid], sBlockSums[tid]);\n"
                "}\n")
    modified = modified[:-len(ending)] + epilogue
    variant = template[:start] + modified + template[end:]
    if variant[:start] != template[:start] or variant.endswith(template[end:]) is False:
        raise RuntimeError("Transformation affected CUDA outside dK")
    return variant


def render(template, threads):
    source = template.replace("__WARPS_PER_BLOCK__", "16")
    source = source.replace("__BWDK_BTILE__", "64")
    return source.replace("__BWDK_BLOCK__", str(threads))


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def write_csv(path, rows):
    if not rows:
        raise RuntimeError("No rows available for " + str(path))
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(actual, reference):
    import numpy as np

    a, r = np.asarray(actual, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(r).all():
        raise RuntimeError("Nonfinite result or reference")
    error = np.abs(a - r)
    allowed = ATOL + RTOL * np.abs(r)
    failed = int(np.count_nonzero(error > allowed))
    return {"max_abs_error": float(error.max()),
            "normalized_max_abs": float(error.max() / max(float(np.abs(r).max()), 1.0)),
            "max_error_to_allowance": float((error / allowed).max()),
            "failed_coefficients": failed, "passed": failed == 0}


def reference_fp64(torch, x, gy, K):
    B, H, L = map(int, x.shape)
    result = torch.zeros((H, K), device=x.device, dtype=torch.float64)
    for j in range(K):
        shift = j - K // 2
        lo, hi = max(0, -shift), min(L, L - shift)
        if hi <= lo:
            continue
        for b0 in range(0, B, 512):
            xp = x[b0:b0 + 512, :, lo + shift:hi + shift].to(torch.float64)
            gp = gy[b0:b0 + 512, :, lo:hi].to(torch.float64)
            result[:, j].add_((xp * gp).sum(dim=(0, 2), dtype=torch.float64))
    return result


def audit_reference(x, gy, reference):
    import numpy as np

    H, K = reference.shape
    L = int(x.shape[2])
    selected = {(0, 0), (H - 1, K // 2),
                tuple(map(int, np.unravel_index(np.argmin(np.abs(reference)), reference.shape)))}
    rows = []
    for h, j in sorted(selected):
        shift = j - K // 2
        lo, hi = max(0, -shift), min(L, L - shift)
        xp = x[:, h, lo + shift:hi + shift].cpu().numpy().astype(np.float64)
        gp = gy[:, h, lo:hi].cpu().numpy().astype(np.float64)
        cpu_sum = math.fsum((xp * gp).ravel())
        error = abs(float(reference[h, j]) - cpu_sum)
        passed = error <= 1e-8 + 1e-12 * abs(cpu_sum)
        rows.append({"h": h, "j": j, "direct_fp64": float(reference[h, j]),
                     "cpu_fsum": cpu_sum, "max_abs_error": error, "passed": passed})
    return rows


def run_trial(args, template, variant):
    import numpy as np
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("Use the s4convd Python environment in an allocated CUDA GPU job")
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from kernels import warp_tiled_ampere as warp

    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    if args.expected_gpu.lower() not in gpu.lower():
        raise RuntimeError(f"Expected {args.expected_gpu}, found {gpu}")
    if Path(warp.__file__).resolve() != args.kernel_source.resolve():
        raise RuntimeError("The imported Warp module is not the reviewed source path")
    if read_template(Path(warp.__file__)) != template or warp.CUDA_TEMPLATE != template:
        raise RuntimeError("The imported CUDA template differs from the reviewed template")
    capability = tuple(torch.cuda.get_device_capability(0))
    if capability not in {(7, 0), (8, 0), (8, 6)}:
        raise RuntimeError("This pilot supports only the three paper GPU architectures")
    threads = 256 if capability == (7, 0) else 128
    warp.configure_torch()
    stream = torch.cuda.default_stream(0)
    if int(stream.cuda_stream) != 0:
        raise RuntimeError("This harness requires the shared CUDA default stream")
    runtime = warp.AmpereWarpRuntime(warps_per_block=16, bwdk_btile=64,
                                     bwdk_block=threads, fast_math=False)
    folder = (args.out or Path("results/dk_deferred_atomics") / runtime.profile.key)
    folder = folder / f"trial{args.trial_id}"
    try:
        folder.mkdir(parents=True, exist_ok=False)
    except Exception:
        runtime.close()
        raise
    module = functions = None
    checks, audits, raw, summaries, blocks = [], [], [], [], []
    try:
        baseline_source, deferred_source = render(template, threads), render(variant, threads)
        (folder / "baseline.cu").write_text(baseline_source, encoding="utf-8")
        (folder / "deferred.cu").write_text(deferred_source, encoding="utf-8")
        (folder / "source.diff").write_text("".join(difflib.unified_diff(
            baseline_source.splitlines(True), deferred_source.splitlines(True),
            fromfile="baseline.cu", tofile="deferred.cu")), encoding="utf-8")
        major, minor = capability
        options = ["-O3", f"-arch=sm_{major}{minor}", "-lineinfo", "-Xptxas=-v",
                   "-ftz=true", "-prec-div=false", "-prec-sqrt=false"]
        with runtime._active_primary_context():
            module = SourceModule(deferred_source, options=options, no_extern_c=True)
            functions = {"Baseline": runtime.bwd_weight,
                         "Deferred": module.get_function(VARIANT_NAME)}
            resources = {name: {"threads_per_block": threads,
                                "registers_per_thread": int(fn.num_regs),
                                "static_shared_bytes": int(fn.shared_size_bytes),
                                "local_bytes_per_thread": int(fn.local_size_bytes)}
                         for name, fn in functions.items()}
        metadata = {
            "experiment": "deferred_atomic_epilogue_v1", "exploratory": True,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "trial_id": args.trial_id, "device_name": gpu,
            "compute_capability": list(capability), "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "kernel_file_sha256": sha256(args.kernel_source.read_bytes()),
            "harness_sha256": sha256(Path(__file__).read_bytes()),
            "cuda_source_sha256": {"Baseline": sha256(baseline_source.encode()),
                                   "Deferred": sha256(deferred_source.encode())},
            "compile_options": options, "resources": resources,
            "B": args.B, "H": 128, "L": 48, "K": 48,
            "B_TILE": 64, "K_PACK": 4, "atol": ATOL, "rtol": RTOL,
            "full_batch_validation_seeds": [42, 16426], "validation_repetitions": 3,
            "warmup": 50, "runs_per_method": 100, "paired_blocks": 10,
            "timing_scope": "full_dk_zero_fill_plus_kernel",
            "compilation_and_validation_in_timing": False,
            "atomic_updates_per_block": 48, "atomic_count_reduced": False,
            "achieved_occupancy_measured": False,
        }
        write_json(folder / "metadata.json", metadata)
        print(f"GPU: {gpu} | dK threads/block={threads} | B_TILE=64 K_PACK=4", flush=True)
        print(f"Full-batch dK tolerance: atol={ATOL} rtol={RTOL}", flush=True)
        for name, values in resources.items():
            print(f"{name}: {values}", flush=True)

        def inputs(shape, seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            x = torch.randn(*shape, device="cuda", dtype=torch.float32)
            return x, torch.randn_like(x)

        def launch(name, x, gy, out, K):
            B, H, L = map(int, x.shape)
            out.zero_()
            functions[name](np.uintp(gy.data_ptr()), np.uintp(x.data_ptr()),
                            np.uintp(out.data_ptr()), *map(np.int32, (B, H, L, K)),
                            np.int32((B + 63) // 64), block=(threads, 1, 1),
                            grid=(H, (B + 63) // 64, 1))

        def validate(B, H, L, K, seed, label):
            x, gy = inputs((B, H, L), seed)
            print(f"Validation {label}: B={B} H={H} L={L} K={K} seed={seed}", flush=True)
            padded = F.pad(x, (K // 2, K - 1 - K // 2))
            ref32 = torch.nn.grad.conv1d_weight(padded, (H, 1, K), gy, groups=H).squeeze(1)
            del padded
            ref64 = reference_fp64(torch, x, gy, K)
            torch.cuda.synchronize()
            refs = {"pytorch_fp32": ref32.cpu().numpy(), "direct_fp64": ref64.cpu().numpy()}
            coefficient_audits = audit_reference(x, gy, refs["direct_fp64"])
            audits.extend({"case": label, "seed": seed, **row} for row in coefficient_audits)
            common = {"case": label, "B": B, "H": H, "L": L, "K": K,
                      "validation_B": B, "seed": seed}
            checks.append({**common, "method": "PyTorch_FP32", "repeat": 1,
                           "reference": "direct_fp64", **metrics(refs["pytorch_fp32"], refs["direct_fp64"])})
            out = torch.empty((H, K), device="cuda", dtype=torch.float32)
            for name in METHODS:
                rows = []
                for repeat in range(1, 4):
                    launch(name, x, gy, out, K)
                    torch.cuda.synchronize()
                    actual = out.cpu().numpy()
                    for ref_name, ref in refs.items():
                        row = {**common, "method": name, "repeat": repeat,
                               "reference": ref_name, **metrics(actual, ref)}
                        checks.append(row)
                        rows.append(row)
                print(f"  {name}: failed_coefficients(max)="
                      f"{max(row['failed_coefficients'] for row in rows)} "
                      f"max_abs(max)={max(row['max_abs_error'] for row in rows):.9g}", flush=True)
            if not all(row["passed"] for row in checks) or not all(row["passed"] for row in audits):
                raise RuntimeError("Correctness gate failed. No benchmark timings were collected")
            return refs

        with torch.no_grad(), torch.cuda.stream(stream), runtime._active_primary_context():
            validate(1, 2, 1, 1, 42, "minimal")
            validate(65, 3, 17, 7, 42, "partial_batch_tile_and_coefficient_pack")
            refs = validate(args.B, 128, 48, 48, 42, "full_batch")
            validate(args.B, 128, 48, 48, 16426, "full_batch")
            print("Full-batch correctness PASS. Starting paired timing...", flush=True)
            x, gy = inputs((args.B, 128, 48), 42)
            outputs = {name: torch.empty((128, 48), device="cuda", dtype=torch.float32)
                       for name in METHODS}
            for _ in range(50):
                for name in METHODS:
                    launch(name, x, gy, outputs[name], 48)
            torch.cuda.synchronize()
            events = {name: [(cuda.Event(), cuda.Event()) for _ in range(100)] for name in METHODS}
            for block_id in range(10):
                order = list(METHODS)
                if (block_id + args.trial_id) % 2 == 0:
                    order.reverse()
                for name in order:
                    for run_id in range(block_id * 10, (block_id + 1) * 10):
                        start, end = events[name][run_id]
                        start.record()
                        launch(name, x, gy, outputs[name], 48)
                        end.record()
                torch.cuda.synchronize()
            for name in METHODS:
                values = [float(start.time_till(end)) for start, end in events[name]]
                if any(not math.isfinite(v) or v <= 0 for v in values):
                    raise RuntimeError("Invalid CUDA-event measurements")
                raw.extend({"method": name, "run_id": i + 1, "paired_block": i // 10 + 1,
                            "runtime_ms": value} for i, value in enumerate(values))
                summaries.append({"method": name, "mean_ms": statistics.mean(values),
                                  "within_trial_std_ms": statistics.stdev(values), "runs": 100})
                for ref_name, ref in refs.items():
                    checks.append({"case": "after_timing", "B": args.B, "H": 128, "L": 48,
                                   "K": 48, "validation_B": args.B, "seed": 42,
                                   "method": name, "repeat": 1, "reference": ref_name,
                                   **metrics(outputs[name].cpu().numpy(), ref)})
            if not all(row["passed"] for row in checks):
                raise RuntimeError("Post-timing validation failed. Measurements are not valid results")
            for block_id in range(1, 11):
                means = {name: statistics.mean(row["runtime_ms"] for row in raw
                                               if row["method"] == name and row["paired_block"] == block_id)
                         for name in METHODS}
                blocks.append({"paired_block": block_id, "baseline_ms": means["Baseline"],
                               "deferred_ms": means["Deferred"],
                               "baseline_over_deferred": means["Baseline"] / means["Deferred"]})
            events = None
        means = {row["method"]: row["mean_ms"] for row in summaries}
        write_csv(folder / "raw.csv", raw)
        write_csv(folder / "trial_summary.csv", summaries)
        write_csv(folder / "paired_blocks.csv", blocks)
        write_json(folder / "status.json", {"status": "PILOT_PASS", "exploratory": True,
                                            "baseline_over_deferred": means["Baseline"] / means["Deferred"]})
        for row in summaries:
            print(f"{row['method']:8s} full_dK={row['mean_ms']:.6f} ms", flush=True)
        print(f"Baseline/Deferred={means['Baseline'] / means['Deferred']:.6f}x", flush=True)
        print(f"Saved: {folder}\nPILOT PASS. This is one trial, not a publication claim.", flush=True)
    except Exception as exc:
        write_json(folder / "status.json", {"status": "FAILED", "error": str(exc), "exploratory": True})
        raise
    finally:
        if checks:
            write_csv(folder / "correctness.csv", checks)
        if audits:
            write_json(folder / "fp64_fsum_audit.json", audits)
        functions = None
        with runtime._active_primary_context():
            module = None
        runtime.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpu", default="A6000")
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--kernel-source", type=Path, default=Path("kernels/warp_tiled_ampere.py"))
    parser.add_argument("--check-source", action="store_true", help="CPU-only source checks, no CUDA imports")
    args = parser.parse_args()
    if args.B < 1 or args.trial_id < 1:
        parser.error("B and trial-id must be positive")
    template = read_template(args.kernel_source)
    variant = deferred_template(template)
    if args.check_source:
        print("Source guard PASS. Transformation is confined to dK.")
        print("Compute packing and reductions are unchanged. Shared buffer: 48 FP32 values.")
        print("CUDA compilation, full-batch validation, and performance require the allocated GPU.")
        return
    run_trial(args, template, variant)


if __name__ == "__main__":
    main()