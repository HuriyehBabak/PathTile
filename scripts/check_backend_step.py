import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from kernels import warp_tiled_ampere_imputed as m

def compare(label, a, b):
    a, b = a.double(), b.double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError(f"Nonfinite values: {label}")
    d = a - b
    relative = d.norm() / b.norm().clamp_min(1e-12)
    print(
        f"{label}: max_abs={d.abs().max().item():.9e} "
        f"relative_L2={relative.item():.9e}",
        flush=True,
    )

m.configure_torch()
runtime = m.AmpereWarpRuntime(
    warps_per_block=16,
    bwdk_btile=m.DEFAULT_BWDK_BTILE,
    bwdk_block=128,
    fast_math=False,
)
m.install_runtime(runtime)
m.KERNEL_EVENTS.enabled = False

try:
    torch.manual_seed(0)
    config = m.TrainConfig(
        seed=0, batch_size=16384,
        data_fraction=0.1, num_workers=0,
    )
    train_loader, _, _, _ = m.build_dataloaders(config)
    model = m.Model().cuda().float()
    initial = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
    }

    batch = next(iter(train_loader))
    x = batch["input_sequence"].cuda()
    target = batch["label_sequence"].cuda()
    print("Batch:", tuple(x.shape), flush=True)

    # Both backends begin each comparison with identical RNG states.
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()

    for training in (True, False):
        print(
            "\nMODE: " + ("train, dropout enabled" if training
                         else "eval, dropout disabled"),
            flush=True,
        )
        results = {}

        for backend in ("warp", "pytorch"):
            m.TRAIN_CONV_BACKEND = backend
            model.load_state_dict(initial)
            model.train(training)
            model.zero_grad(set_to_none=True)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=1e-3, momentum=0.9
            )
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state_all(cuda_rng)

            prediction = model(x)
            loss = m.log_mse(prediction, target)
            loss.backward()
            torch.cuda.synchronize()

            gradients = {
                name: p.grad.detach().cpu().clone()
                for name, p in model.named_parameters()
                if p.grad is not None
            }
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            torch.cuda.synchronize()

            results[backend] = {
                "prediction": prediction.detach().cpu().clone(),
                "loss": loss.detach().cpu().clone(),
                "gradients": gradients,
                "grad_norm": grad_norm.detach().cpu().clone(),
                "parameters": {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                },
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
            }
            print(
                backend, "loss =", loss.item(),
                "gradient_norm_before_clipping =", grad_norm.item(),
                flush=True,
            )
            del prediction, loss, optimizer

        a, b = results["warp"], results["pytorch"]
        compare("prediction", a["prediction"], b["prediction"])
        compare("loss", a["loss"], b["loss"])
        compare("gradient_norm", a["grad_norm"], b["grad_norm"])

        if a["gradients"].keys() != b["gradients"].keys():
            raise RuntimeError("Different sets of parameters received gradients")

        for name in a["gradients"]:
            compare(
                "gradient/" + name,
                a["gradients"][name], b["gradients"][name],
            )
        for name in a["parameters"]:
            compare(
                "parameter_after_step/" + name,
                a["parameters"][name], b["parameters"][name],
            )

        print(
            "RNG states after step equal:",
            torch.equal(a["cpu_rng"], b["cpu_rng"])
            and all(torch.equal(u, v) for u, v in
                    zip(a["cuda_rng"], b["cuda_rng"])),
            flush=True,
        )
finally:
    m.KERNEL_EVENTS.enabled = False
    runtime.close()
