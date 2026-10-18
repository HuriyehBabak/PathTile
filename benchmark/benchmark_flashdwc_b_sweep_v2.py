import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
torch.cuda.init()
print("GPU:", torch.cuda.get_device_name(0), flush=True)

lib_path = Path(
    "results/flashdwc/build_v2_121790/libflashdwc_fp32.so"
).resolve()
lib = ctypes.CDLL(str(lib_path))

fn = lib.flash_dk_fp32
fn.argtypes = (
    [ctypes.c_void_p] * 3
    + [ctypes.c_int] * 5
    + [ctypes.c_void_p]
)
fn.restype = ctypes.c_int

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--B", type=int, choices=(64, 1024, 4096), required=True)
args = parser.parse_args()

B, H, L, K = args.B, 128, 48, 48
gz_values = [
    2**i for i in range(B.bit_length())
    if B % (2**i) == 0
]
print(f"SHAPE={(B, H, L, K)}, GZ_CANDIDATES={gz_values}", flush=True)
generator = torch.Generator().manual_seed(42)
x_cpu = torch.randn(B, H, L, generator=generator)
dy_cpu = torch.randn(B, H, L, generator=generator)

# Direct FP64 reference:
# dK[h,k] = sum_{b,t} dy[b,h,t] * x[b,h,t+k-K//2]
left = K // 2
right = K - 1 - left
x_pad = F.pad(x_cpu.double(), (left, right))
dy_ref = dy_cpu.double()

reference = torch.stack([
    (x_pad[:, :, k:k + L] * dy_ref).sum(dim=(0, 2))
    for k in range(K)
], dim=1)

# Flash-DWC layout: [B, 1, L, H].
x = x_cpu.permute(0, 2, 1).unsqueeze(1).contiguous().cuda()
dy = dy_cpu.permute(0, 2, 1).unsqueeze(1).contiguous().cuda()
dw = torch.empty((1, K, H), device="cuda", dtype=torch.float32)

atol, rtol = 1e-3, 5e-4
all_passed = True

for gz in gz_values:
    dw.zero_()
    stream = torch.cuda.current_stream()

    status = fn(
        x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
        B, H, L, K, gz, stream.cuda_stream,
    )
    if status != 0:
        raise RuntimeError(f"Flash-DWC returned {status}, GZ={gz}")

    torch.cuda.synchronize()
    actual = dw[0].T.contiguous().cpu().double()

    finite = bool(torch.isfinite(actual).all())
    close = torch.isclose(actual, reference, atol=atol, rtol=rtol)
    passed = finite and bool(close.all())
    max_abs = (actual - reference).abs().max().item()
    normalized = max_abs / max(reference.abs().max().item(), 1e-30)

    print(
        f"GZ={gz}: PASS={passed}, "
        f"max_abs={max_abs:.9e}, "
        f"normalized_max_abs={normalized:.9e}, "
        f"failed_elements={(~close).sum().item()}, "
        f"atol={atol}, rtol={rtol}",
        flush=True,
    )
    all_passed &= passed

print("Flash-DWC checks finished; testing PathTile pack=4", flush=True)

import gc
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from benchmark.benchmark_dk_packing import (
    load_template, make_source, FUNCTION_NAME, BTILE,
)

# Reuse exactly the same CPU inputs and FP64 reference.
x_ours = x_cpu.contiguous().cuda()
dy_ours = dy_cpu.contiguous().cuda()
dk_ours = torch.empty((H, K), device="cuda", dtype=torch.float32)
torch.cuda.synchronize()

cuda.init()
context = cuda.Device(0).retain_primary_context()
context.push()
module = kernel = cu_stream = None

try:
    template, source_hash = load_template(
        Path("kernels/warp_tiled_ampere.py")
    )
    block = 128
    cc = torch.cuda.get_device_capability(0)
    options = [
        "-O3", f"-arch=sm_{cc[0]}{cc[1]}",
        "-lineinfo", "-Xptxas=-v",
        "-ftz=true", "-prec-div=false", "-prec-sqrt=false",
    ]
    module = SourceModule(
        make_source(template, 4, block),
        options=options,
        no_extern_c=True,
    )
    kernel = module.get_function(FUNCTION_NAME)
    kernel.prepare("PPPiiiii")
    cu_stream = cuda.Stream()
    tiles = (B + BTILE - 1) // BTILE

    print("PathTile source SHA256:", source_hash, flush=True)

    for repeat in range(1, 4):
        cuda.memset_d32_async(
            dk_ours.data_ptr(), 0, dk_ours.numel(), cu_stream
        )
        kernel.prepared_async_call(
            (H, tiles, 1), (block, 1, 1), cu_stream,
            dy_ours.data_ptr(), x_ours.data_ptr(),
            dk_ours.data_ptr(), B, H, L, K, tiles,
        )
        cu_stream.synchronize()

        actual = dk_ours.cpu().double()
        error = (actual - reference).abs()
        allowed = atol + rtol * reference.abs()
        close = torch.isclose(
            actual, reference, atol=atol, rtol=rtol
        )
        passed = bool(torch.isfinite(actual).all() and close.all())

        print(
            f"PathTile pack=4 repeat={repeat}: PASS={passed}, "
            f"max_abs={error.max().item():.9e}, "
            f"normalized_max_abs="
            f"{error.max().item()/reference.abs().max().item():.9e}, "
            f"failed_elements={(~close).sum().item()}, "
            f"max_error_to_allowed={(error/allowed).max().item():.3f}, "
            f"atol={atol}, rtol={rtol}",
            flush=True,
        )


    import csv
    import json
    import os
    import random
    import statistics

    out_dir = Path("results/flashdwc") / (
        f"b_sweep_v2_B{B}_" + os.environ["SLURM_JOB_ID"]
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    TRIALS, WARMUP, RUNS = 3, 10, 20
    methods = [("PathTile", 0)] + [
        (f"Flash_GZ{g}", g) for g in gz_values
    ]
    scopes = ("native_full", "reuse_x_full")

    metadata = {
        "shape": [B, H, L, K],
        "seed": 42, "gz_candidates": gz_values,
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "trials": TRIALS,
        "warmup_per_measurement": WARMUP,
        "runs_per_measurement": RUNS,
        "atol": atol,
        "rtol": rtol,
        "reference": "direct CPU FP64",
        "pathtile_source_sha256": source_hash,
        "pathtile_compiler_options": options,
        "flash_library": str(lib_path),
        "flash_commit": "74d31d8fae0fb189385c0b8bc39e11d7e81ba408",
        "flash_compiler_options": "-O3 --use_fast_math sm_86",
        "timing": "CUDA events; one sample per launch",
        "allocation_included": False,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    # Preallocated layout-conversion buffers.
    # x and dy already have Flash-DWC's contiguous NHWC layout.
    x_bhl_view = x_ours.permute(0, 2, 1).unsqueeze(1)
    dy_bhl_view = dy_ours.permute(0, 2, 1).unsqueeze(1)
    dw_hk_view = dw[0].T
    layout_output = torch.empty_like(dk_ours)
    torch.cuda.synchronize()

    shared_stream = torch.cuda.ExternalStream(cu_stream.handle, device=0)
    start_event = cuda.Event()
    end_event = cuda.Event()

    def zero_output(gz):
        output = dk_ours if gz == 0 else dw
        cuda.memset_d32_async(
            output.data_ptr(), 0, output.numel(), cu_stream
        )

    def launch_method(gz):
        if gz == 0:
            kernel.prepared_async_call(
                (H, tiles, 1), (block, 1, 1), cu_stream,
                dy_ours.data_ptr(), x_ours.data_ptr(),
                dk_ours.data_ptr(), B, H, L, K, tiles,
            )
        else:
            status = fn(
                x.data_ptr(), dy.data_ptr(), dw.data_ptr(),
                B, H, L, K, gz, cu_stream.handle,
            )
            if status != 0:
                raise RuntimeError(f"Flash-DWC status={status}, GZ={gz}")

    def operation(gz, scope):
        if scope == "reuse_x_full" and gz != 0:
            dy.copy_(dy_bhl_view)
        if scope != "kernel":
            zero_output(gz)
        launch_method(gz)
        if scope == "reuse_x_full" and gz != 0:
            layout_output.copy_(dw_hk_view)

    rows = []
    print(f"TIMING_STARTED: {out_dir}", flush=True)

    with (
        (out_dir / "samples.csv").open("w", newline="") as raw_file,
        (out_dir / "trials.csv").open("w", newline="") as trial_file,
    ):
        raw_writer = csv.writer(raw_file)
        raw_writer.writerow(["trial", "method", "scope", "run", "ms"])
        trial_writer = csv.writer(trial_file)
        trial_writer.writerow([
            "trial", "method", "scope", "mean_ms",
            "last_output_pass", "failed_elements", "max_abs",
        ])

        for trial in range(1, TRIALS + 1):
            tasks = [
                (name, gz, scope)
                for name, gz in methods for scope in scopes
            ]
            random.Random(1000 + trial).shuffle(tasks)

            for name, gz, scope in tasks:
                samples = []

                with torch.cuda.stream(shared_stream):
                    for _ in range(WARMUP):
                        if scope == "kernel":
                            zero_output(gz)
                        operation(gz, scope)
                    cu_stream.synchronize()

                    for run in range(RUNS):
                        if scope == "kernel":
                            zero_output(gz)

                        start_event.record(cu_stream)
                        operation(gz, scope)
                        end_event.record(cu_stream)
                        end_event.synchronize()

                        ms = start_event.time_till(end_event)
                        samples.append(ms)
                        raw_writer.writerow([trial, name, scope, run, ms])

                # Outside timing: check the final measured output.
                if gz == 0:
                    result = dk_ours
                elif scope == "reuse_x_full":
                    result = layout_output
                else:
                    result = dw_hk_view

                measured = result.cpu().double()
                close = torch.isclose(
                    measured, reference, atol=atol, rtol=rtol
                )
                ok = bool(torch.isfinite(measured).all() and close.all())
                failed = int((~close).sum())
                max_abs = (measured - reference).abs().max().item()
                mean_ms = statistics.mean(samples)

                rows.append((name, scope, mean_ms, ok))
                trial_writer.writerow([
                    trial, name, scope, mean_ms, ok, failed, max_abs
                ])
                raw_file.flush()
                trial_file.flush()

                print(
                    f"trial={trial} {name} {scope}: "
                    f"{mean_ms:.6f} ms | "
                    f"PASS_last={ok} failed={failed}",
                    flush=True,
                )

    print("\nSUMMARY: mean of trial means +/- between-trial SD", flush=True)
    for name, _ in methods:
        for scope in scopes:
            selected = [r for r in rows if r[0] == name and r[1] == scope]
            values = [r[2] for r in selected]
            passed = sum(r[3] for r in selected)
            print(
                f"{name:14s} {scope:12s}: "
                f"{statistics.mean(values):.6f} +/- "
                f"{statistics.stdev(values):.6f} ms | "
                f"last-output checks={passed}/{TRIALS}",
                flush=True,
            )

    shared_stream = None
    print(f"TIMING_COMPLETE: {out_dir}", flush=True)

finally:
    kernel = module = cu_stream = None
    gc.collect()
    cuda.Context.pop()
    context.detach()

print("COMPARISON_COMPLETE: inspect PASS for each configuration", flush=True)
