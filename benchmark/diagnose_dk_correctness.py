#!/usr/bin/env python3
"""Diagnose dK numerical differences without changing kernels or tolerances.

Place in benchmark/ beside benchmark_dk_batch_compare.py. Run on the same
allocated GPU with --trial-dir pointing to the incomplete benchmark trial.
This script measures correctness only and does not generate timing results.
It reproduces the trial inputs, compares Shared with the existing GMC and
Warp kernels, and computes an independent direct FP64 dK reference. Selected
coefficients are also checked on the CPU with math.fsum of FP64 products.
The original incomplete trial and its files are left intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def time_range(L, K, j):
    """Valid t for x[b,h,t+j-K//2] in a length-L sequence."""
    shift = j - K // 2
    return max(0, -shift), min(L, L - shift), shift


def source_hash(module):
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def metrics(actual, reference, atol, rtol):
    import numpy as np

    a = np.asarray(actual, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if not (np.isfinite(a).all() and np.isfinite(r).all()):
        raise ValueError("Nonfinite values found in a diagnostic output or reference")
    error = np.abs(a - r)
    allowance = atol + rtol * np.abs(r)
    ratio = np.divide(error, allowance, out=np.full_like(error, np.inf),
                      where=allowance > 0)
    ratio[(allowance == 0) & (error == 0)] = 0
    worst = tuple(int(v) for v in np.unravel_index(np.argmax(error), error.shape))
    return {
        "coefficients": int(error.size),
        "failed_coefficients": int(np.count_nonzero(error > allowance)),
        "max_abs_error": float(error.max()),
        "normalized_max_abs": float(error.max() / max(float(np.abs(r).max()), 1.0)),
        "max_error_to_allowance": float(ratio.max()),
        "relative_l2_error": float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-300)),
        "max_abs_error_at_h_j": list(worst),
    }


def reference_fp64(torch, x, gy, K, chunk):
    """Direct dK formula with FP64 products and sums, independent of cuDNN."""
    B, H, L = x.shape
    answer = torch.zeros((H, K), device=x.device, dtype=torch.float64)
    for j in range(K):
        lo, hi, shift = time_range(L, K, j)
        if hi <= lo:
            continue
        for b0 in range(0, B, chunk):
            x_part = x[b0:b0 + chunk, :, lo + shift:hi + shift].to(torch.float64)
            gy_part = gy[b0:b0 + chunk, :, lo:hi].to(torch.float64)
            answer[:, j].add_((x_part * gy_part).sum(dim=(0, 2), dtype=torch.float64))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path,
                        default=Path("results/dk_batch_compare/a6000/trial1"))
    parser.add_argument("--B", type=int, default=16384)
    parser.add_argument("--chunk", type=int, default=512)
    args = parser.parse_args()
    if args.B < 1 or args.chunk < 1:
        parser.error("B and chunk must be positive")
    original = json.loads((args.trial_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.B not in original["batches"]:
        parser.error("B must be present in the original trial metadata")

    import numpy as np
    import torch
    import torch.nn.functional as F
    from kernels import shared, coalesced
    from kernels import warp_tiled_ampere as warp

    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    if gpu != original["device_name"]:
        raise RuntimeError(f"Expected original GPU {original['device_name']!r}, found {gpu!r}")
    for name, module in (("Shared", shared), ("Warp", warp)):
        if source_hash(module) != original["kernel_sha256"][name]:
            raise RuntimeError(f"{name} source changed since the incomplete trial")
    versions = {"torch_version": str(torch.__version__), "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version()}
    if any(versions[key] != original[key] for key in versions):
        raise RuntimeError("Use the same Python/CUDA/cuDNN environment as the incomplete trial")
    warp.configure_torch()
    stream = torch.cuda.default_stream(0)
    if int(stream.cuda_stream) != 0:
        raise RuntimeError("This diagnostic requires the CUDA default stream")
    B, H, L, K = args.B, original["H"], original["L"], original["K"]
    atol, rtol = float(original["atol"]), float(original["rtol"])
    seed = int(original["seed"]) + B
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(B, H, L, device="cuda", dtype=torch.float32)
    gy = torch.randn_like(x)
    weight = torch.randn(H, K, device="cuda", dtype=torch.float32, requires_grad=True)
    y = F.conv1d(F.pad(x, (K // 2, K - 1 - K // 2)), weight.unsqueeze(1), groups=H)
    reference32 = torch.autograd.grad(y, weight, grad_outputs=gy)[0].detach()
    del y, weight

    threads = int(original["resources"]["Warp"]["threads_per_block"])
    runtime = warp.AmpereWarpRuntime(warps_per_block=16, bwdk_btile=64,
                                     bwdk_block=threads, fast_math=False)
    try:
        outputs = {name: torch.full_like(reference32, float("nan"))
                   for name in ("Shared", "Shared_repeat", "GMC", "Warp")}
        dimensions = tuple(np.int32(v) for v in (B, H, L, K))
        ptr = lambda tensor: np.uintp(tensor.data_ptr())
        with torch.cuda.stream(stream), shared.CudaContextGuard(), torch.no_grad():
            shared_fn = shared._get_kernels()[2]
            gmc_fn = coalesced._get_kernels()[2]
            for name in ("Shared", "Shared_repeat"):
                block = int(shared.THREADS_PER_BLOCK)
                shared_fn(ptr(gy), ptr(x), ptr(outputs[name]), *dimensions,
                          block=(block, 1, 1), grid=((K + block - 1) // block, H, 1),
                          shared=int(shared.bwd_kernel_shared_memory_bytes(L)))
            block = int(coalesced.THREADS_PER_BLOCK)
            gmc_fn(ptr(gy), ptr(x), ptr(outputs["GMC"]), *dimensions,
                   block=(block, 1, 1), grid=((K + block - 1) // block, H, 1))
            ntiles = (B + 63) // 64
            outputs["Warp"].zero_()
            runtime.bwd_weight(ptr(gy), ptr(x), ptr(outputs["Warp"]), *dimensions,
                              np.int32(ntiles), block=(threads, 1, 1), grid=(H, ntiles, 1))
            torch.cuda.synchronize()
            print(f"GPU: {gpu}\nB={B} H={H} L={L} K={K} seed={seed}", flush=True)
            print(f"Original tolerance: atol={atol} rtol={rtol}", flush=True)
            print("Computing direct FP64 reference, without timing...", flush=True)
            reference64 = reference_fp64(torch, x, gy, K, args.chunk)
            torch.cuda.synchronize()

        ref32 = reference32.cpu().numpy().astype(np.float64)
        ref64 = reference64.cpu().numpy()
        arrays = {name: tensor.cpu().numpy().astype(np.float64)
                  for name, tensor in outputs.items()}
        report = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "original_trial_dir": str(args.trial_dir), "B": B, "H": H, "L": L, "K": K,
            "device_name": gpu, "seed": seed, "atol": atol, "rtol": rtol, **versions,
            "source_sha256": {name: source_hash(module) for name, module in
                              (("Shared", shared), ("GMC", coalesced), ("Warp", warp))},
            "fp64_reference": "direct formula, FP64 products and reductions over full batch",
            "fp64_chunk": args.chunk,
            "shared_repeat_exact": bool(torch.equal(outputs["Shared"], outputs["Shared_repeat"])),
            "shared_gmc_exact": bool(torch.equal(outputs["Shared"], outputs["GMC"])),
            "shared_gmc_max_abs": float(np.abs(arrays["Shared"] - arrays["GMC"]).max()),
            "vs_pytorch_fp32": {}, "vs_direct_fp64": {}, "selected_coefficients": [],
        }
        print(f"Shared repeat exact: {report['shared_repeat_exact']}")
        print(f"Shared/GMC exact: {report['shared_gmc_exact']} | "
              f"max abs={report['shared_gmc_max_abs']:.9g}")
        for name in ("Shared", "GMC", "Warp"):
            original_close = bool(torch.allclose(outputs[name], reference32, atol=atol, rtol=rtol))
            m32 = metrics(arrays[name], ref32, atol, rtol)
            m32["torch_allclose_original"] = original_close
            report["vs_pytorch_fp32"][name] = m32
            report["vs_direct_fp64"][name] = metrics(arrays[name], ref64, atol, rtol)
        report["vs_direct_fp64"]["PyTorch_FP32"] = metrics(ref32, ref64, atol, rtol)
        for reference_name in ("vs_pytorch_fp32", "vs_direct_fp64"):
            print(reference_name + ":")
            for name, row in report[reference_name].items():
                print(f"  {name:12s} failed={row['failed_coefficients']}/{row['coefficients']} "
                      f"max_abs={row['max_abs_error']:.9g} "
                      f"normalized={row['normalized_max_abs']:.9g}")

        error = np.abs(arrays["Shared"] - ref32)
        allowance = atol + rtol * np.abs(ref32)
        ranking = error / allowance
        selected = list(np.argsort(ranking.ravel())[::-1][:4])
        selected.extend([int(np.argmax(error)), K // 2, 0])
        selected = list(dict.fromkeys(int(v) for v in selected))
        print("Selected coefficients (same original tolerance):", flush=True)
        for flat in selected:
            h, j = np.unravel_index(flat, (H, K))
            lo, hi, shift = time_range(L, K, int(j))
            if hi > lo:
                x_cpu = x[:, h, lo + shift:hi + shift].cpu().numpy().astype(np.float64)
                gy_cpu = gy[:, h, lo:hi].cpu().numpy().astype(np.float64)
                cpu_sum = math.fsum((x_cpu * gy_cpu).ravel())
            else:
                cpu_sum = 0.0
            row = {"h": int(h), "j": int(j), "terms": B * max(hi - lo, 0),
                   "Shared": float(arrays["Shared"][h, j]), "GMC": float(arrays["GMC"][h, j]),
                   "Warp": float(arrays["Warp"][h, j]), "PyTorch_FP32": float(ref32[h, j]),
                   "direct_FP64": float(ref64[h, j]), "CPU_fsum_FP64": cpu_sum,
                   "original_abs_error": float(error[h, j]),
                   "original_allowance": float(allowance[h, j]),
                   "fp64_fsum_agree": bool(abs(float(ref64[h, j]) - cpu_sum)
                                             <= 1e-8 + 1e-12 * abs(cpu_sum))}
            report["selected_coefficients"].append(row)
            print(f"  h={h} j={j} Shared={row['Shared']:.9g} "
                  f"Warp={row['Warp']:.9g} Torch={row['PyTorch_FP32']:.9g} "
                  f"FP64={row['direct_FP64']:.12g} fsum={cpu_sum:.12g} "
                  f"err={row['original_abs_error']:.9g} "
                  f"allow={row['original_allowance']:.9g} fsum_agree={row['fp64_fsum_agree']}")
        directory = args.trial_dir.parent / "diagnostics"
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / ("dk_b" + str(B) + "_" +
                                 datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        with filename.open("x", encoding="utf-8") as f:
            json.dump(report, f, indent=2, allow_nan=False)
        print(f"Saved diagnostic: {filename}")
        print("Diagnostic complete. No benchmark timing was recorded and no tolerance was changed.")
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
