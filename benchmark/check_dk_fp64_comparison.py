import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
torch.cuda.init()
print("GPU:", torch.cuda.get_device_name(0), flush=True)

lib_path = Path(
    "results/flashdwc/build_121405/libflashdwc_fp32.so"
).resolve()
lib = ctypes.CDLL(str(lib_path))

fn = lib.flash_dk_fp32
fn.argtypes = (
    [ctypes.c_void_p] * 3
    + [ctypes.c_int] * 5
    + [ctypes.c_void_p]
)
fn.restype = ctypes.c_int

B, H, L, K = 16384, 128, 48, 48
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

for gz in (1, 4, 16, 64, 256, 1024):
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

finally:
    kernel = module = cu_stream = None
    gc.collect()
    cuda.Context.pop()
    context.detach()

print("COMPARISON_COMPLETE: inspect PASS for each configuration", flush=True)
