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
    error = (actual - reference).abs()
    budget = atol + rtol * reference.abs()
    ratio = error / budget

    for index in ratio.flatten().topk(3).indices.tolist():
        h, k = divmod(index, K)
        print(
            f"  h={h} k={k}: "
            f"ref={reference[h,k].item():.9e}, "
            f"actual={actual[h,k].item():.9e}, "
            f"abs_error={error[h,k].item():.9e}, "
            f"allowed={budget[h,k].item():.9e}, "
            f"ratio={ratio[h,k].item():.3f}",
            flush=True,
        )

    all_passed &= passed

if not all_passed:
    raise SystemExit("CORRECTNESS_FAILED")

print("CORRECTNESS_PASS: all tested GZ values", flush=True)
