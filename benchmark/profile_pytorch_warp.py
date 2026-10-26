"""Profile one warmed-up execution path, excluding setup."""
import argparse

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("pytorch", "warp"), required=True)
    parser.add_argument("--phase", choices=("fwd", "dx", "dk"), required=True)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(42)
    B, H, L, K = 16384, 128, 48, 48

    runtime = None
    if args.backend == "warp":
        from kernels.warp_tiled_ampere import AmpereWarpRuntime
        runtime = AmpereWarpRuntime(
            warps_per_block=16,
            bwdk_btile=64,
            bwdk_block=128,
        )

    # Match the primary RTX A6000 PyTorch baseline.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    x = torch.randn(B, H, L, device="cuda", requires_grad=True)
    w = torch.randn(H, 1, K, device="cuda", requires_grad=True)
    gy = torch.randn_like(x)
    output = torch.empty_like(x)
    gk = torch.empty(H, K, device="cuda")
    k = w.detach().view(H, K)

    def forward():
        padded = F.pad(x, (K // 2, K - 1 - K // 2))
        return F.conv1d(padded, w, groups=H)

    def prepare_call():
        if args.backend == "pytorch":
            if args.phase == "fwd":
                return forward
            # Graph construction remains outside the profiling interval.
            y = forward()
            target = x if args.phase == "dx" else w
            return lambda: torch.autograd.grad(
                y, target, grad_outputs=gy,
                retain_graph=False, create_graph=False,
            )[0]

        if args.phase == "fwd":
            return lambda: runtime.launch_fwd(x.detach(), k, output)
        if args.phase == "dx":
            return lambda: runtime.launch_bwd_input(gy, k, output)

        # Match the primary kernel-only dK scope.
        gk.zero_()
        return lambda: runtime.launch_bwd_weight(
            gy, x.detach(), gk, K
        )

    for _ in range(args.warmup):
        call = prepare_call()
        result = call()
        torch.cuda.synchronize()
        del call, result

    call = prepare_call()
    torch.cuda.synchronize()

    print(
        f"Profiling: {args.backend}/{args.phase}, "
        f"GPU={torch.cuda.get_device_name(0)}",
        flush=True,
    )
    torch.cuda.profiler.start()
    try:
        result = call()
        torch.cuda.synchronize()
    finally:
        torch.cuda.profiler.stop()

    print("Profiled invocation completed.", flush=True)


if __name__ == "__main__":
    main()
