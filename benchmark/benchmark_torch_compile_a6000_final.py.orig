from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmark import benchmark_pytorch_a6000_final as eager


TECHNIQUE = "torch_compile_inductor"
COMPILE_BACKEND = "inductor"
COMPILE_MODE = "max-autotune-no-cudagraphs"

RESULT_DIR = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "a6000_final_baselines"
    / "torch_compile"
    / "current"
)

eager_conv = eager.depthwise_conv1d_same

compiled_conv = torch.compile(
    eager_conv,
    backend=COMPILE_BACKEND,
    mode=COMPILE_MODE,
    fullgraph=True,
    dynamic=False,
)


def validate_correctness() -> None:
    torch.manual_seed(eager.SEED)
    torch.cuda.manual_seed_all(eager.SEED)

    batch = 256

    x = torch.randn(
        batch,
        eager.H,
        eager.L,
        device="cuda",
        dtype=eager.DTYPE,
    )
    weight = torch.randn(
        eager.H,
        1,
        eager.K,
        device="cuda",
        dtype=eager.DTYPE,
    )
    grad_y = torch.randn_like(x)

    x_ref = x.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)

    y_ref = eager_conv(x_ref, w_ref)
    gx_ref, gw_ref = torch.autograd.grad(
        y_ref,
        (x_ref, w_ref),
        grad_outputs=grad_y,
    )

    x_cmp = x.detach().clone().requires_grad_(True)
    w_cmp = weight.detach().clone().requires_grad_(True)

    y_cmp = compiled_conv(x_cmp, w_cmp)
    gx_cmp, gw_cmp = torch.autograd.grad(
        y_cmp,
        (x_cmp, w_cmp),
        grad_outputs=grad_y,
    )

    torch.cuda.synchronize()

    tensors = {
        "forward": (y_cmp, y_ref),
        "backward_input": (gx_cmp, gx_ref),
        "backward_kernel": (gw_cmp, gw_ref),
    }

    print("\nCorrectness against PyTorch eager")
    print("=================================")

    for phase, (actual, reference) in tensors.items():
        max_abs = float(
            (actual - reference).abs().max().item()
        )
        torch.testing.assert_close(
            actual,
            reference,
            rtol=1e-4,
            atol=1e-4,
        )
        print(
            f"{phase}: max_abs={max_abs:.9e}, PASS"
        )

    del (
        x,
        weight,
        grad_y,
        x_ref,
        w_ref,
        y_ref,
        gx_ref,
        gw_ref,
        x_cmp,
        w_cmp,
        y_cmp,
        gx_cmp,
        gw_cmp,
    )
    torch.cuda.empty_cache()


def rename_outputs() -> None:
    replacements = {
        "pytorch_runtime_summary_a6000.csv":
            "torch_compile_runtime_summary_a6000.csv",
        "pytorch_environment_a6000.csv":
            "torch_compile_environment_a6000.csv",
    }

    for old_name, new_name in replacements.items():
        old_path = RESULT_DIR / old_name
        new_path = RESULT_DIR / new_name

        if not old_path.exists():
            raise FileNotFoundError(old_path)

        old_path.replace(new_path)


def save_compile_metadata() -> Path:
    path = RESULT_DIR / "torch_compile_metadata.json"

    metadata = {
        "technique": TECHNIQUE,
        "backend": COMPILE_BACKEND,
        "mode": COMPILE_MODE,
        "fullgraph": True,
        "dynamic": False,
        "compile_overhead_timed": False,
        "correctness_validation_batch": 256,
        "warmup_runs": eager.N_WARMUP,
        "measured_runs": eager.N_RUNS,
        "B": eager.B,
        "H": eager.H,
        "L": eager.L,
        "K": eager.K,
        "dtype": str(eager.DTYPE),
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
    }

    path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    gpu = torch.cuda.get_device_name(0)
    if "RTX A6000" not in gpu:
        raise RuntimeError(
            f"Expected RTX A6000, found {gpu}"
        )

    validate_correctness()

    eager.TECHNIQUE = TECHNIQUE
    eager.RESULT_DIR = RESULT_DIR
    eager.depthwise_conv1d_same = compiled_conv

    eager.main()

    rename_outputs()
    metadata_path = save_compile_metadata()

    print("\ntorch.compile metadata:")
    print(" ", metadata_path)


if __name__ == "__main__":
    main()