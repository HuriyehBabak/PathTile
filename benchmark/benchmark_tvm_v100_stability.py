"""Batched stability remeasurement for the TVM-TIRx V100 baseline.

This protocol is intentionally separate from the initial V100 run and both
single-call stability remeasurements. The initial allocation is used only to
freeze the majority/consistent schedule:

* forward: ``fwd_bh4``;
* backward input: ``bwd_input_bh1``;
* backward weight: ``bwd_weight_t64``.

All candidates are correctness-checked once, the main-workload candidates are
compiled once, and then three timing trials use the same compiled kernels.
Each timing sample brackets a fixed batch of calls with one CUDA-event pair and
reports elapsed time divided by calls per sample. An excluded same-stream
lead-in keeps work queued while Python submits the measured sequence. This
standard amortized device-time protocol prevents a brief system interruption
from being mistaken for a single slow kernel invocation. Compilation,
validation, allocation, burn-in, warm-up, lead-in calls, file I/O, and
GPU-state queries are outside the measured CUDA-event intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tvm
from cuda.bindings import runtime as cudart

from benchmark.benchmark_tvm_main import (
    DEFAULT_B,
    DEFAULT_H,
    DEFAULT_K,
    DEFAULT_L,
    DEFAULT_SEED,
    DEFAULT_WARMUP,
    PHASES,
    _create_event,
    _cuda_result,
    _destroy_event,
    _device_sync,
    _empty_device,
    _record_event,
    allocate_main_inputs,
    cuda_target,
    device_metadata,
    print_correctness,
    stats,
    validate_candidates,
    validate_gpu_slug,
    write_csv,
)
from kernels.tvm_depthwise import compile_candidates


TECHNIQUE = "tvm_tirx_frozen_v100_batched"
FIXED_NAMES = {
    "fwd": "fwd_bh4",
    "bwd_input": "bwd_input_bh1",
    "bwd_weight": "bwd_weight_t64",
}


def measure_batched_cuda_calls(
    call: Callable[[], object],
    samples: int,
    calls_per_sample: int,
    queue_lead_in: int,
) -> list[float]:
    """Return amortized per-call device times for fixed-size call batches.

    Event objects are allocated before GPU work is enqueued. The lead-in calls
    precede the first start event on the same default stream and therefore
    cannot contribute to any result. Each returned value is the elapsed time
    for one event-bracketed batch divided by ``calls_per_sample``.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if calls_per_sample <= 0:
        raise ValueError("calls_per_sample must be positive")
    if queue_lead_in <= 0:
        raise ValueError("queue_lead_in must be positive")

    event_pairs: list[tuple[Any, Any]] = []
    try:
        # Event creation can involve host-side runtime work, so complete it
        # before constructing the queue whose timing we want to inspect.
        for _ in range(samples):
            event_pairs.append((_create_event(), _create_event()))

        # These calls are before every start event in stream order and are
        # therefore rigorously excluded from all elapsed-time measurements.
        for _ in range(queue_lead_in):
            call()

        for start, end in event_pairs:
            _record_event(start)
            for _ in range(calls_per_sample):
                call()
            _record_event(end)

        _cuda_result(
            cudart.cudaEventSynchronize(event_pairs[-1][1]),
            "cudaEventSynchronize",
        )
        milliseconds: list[float] = []
        for start, end in event_pairs:
            values = _cuda_result(
                cudart.cudaEventElapsedTime(start, end),
                "cudaEventElapsedTime",
            )
            if len(values) != 1:
                raise RuntimeError(
                    "cudaEventElapsedTime returned an unexpected result"
                )
            milliseconds.append(float(values[0]) / calls_per_sample)
        return milliseconds
    finally:
        for start, end in event_pairs:
            _destroy_event(start)
            _destroy_event(end)


def candidate_named(candidates: Sequence[Any], name: str) -> Any:
    matches = [candidate for candidate in candidates if candidate.name == name]
    if len(matches) != 1:
        available = [candidate.name for candidate in candidates]
        raise RuntimeError(
            f"Expected exactly one candidate {name!r}; available={available}"
        )
    return matches[0]


def gpu_snapshot(label: str) -> dict[str, object]:
    """Capture read-only NVIDIA state outside measured regions."""

    fields = (
        "timestamp,index,uuid,name,pstate,temperature.gpu,utilization.gpu,"
        "power.draw,power.limit,clocks.current.graphics,clocks.current.sm,"
        "clocks.current.memory"
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={fields}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "label": label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def prepare_output_root(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output root: {path}")
    path.mkdir(parents=True, exist_ok=True)


def selected_config(candidate: Any) -> dict[str, object]:
    return {"candidate": candidate.name, **dict(candidate.config)}


def aggregate(
    output_root: Path,
    trial_rows: list[dict[str, object]],
    audit_rows: list[dict[str, object]],
    between_trial_cv_limit_pct: float,
    within_trial_phase_cv_limit_pct: float,
) -> bool:
    write_csv(output_root / "final_per_trial.csv", trial_rows)

    result: dict[str, object] = {
        "gpu_slug": "v100",
        "method": "TVM-TIRx (frozen schedule, batched timing)",
        "trials": len(trial_rows),
    }
    for column in (
        "fwd_mean_ms",
        "bwd_input_mean_ms",
        "bwd_weight_mean_ms",
        "total_mean_ms",
    ):
        stem = column.removesuffix("_mean_ms")
        values = np.asarray(
            [float(row[column]) for row in trial_rows], dtype=np.float64
        )
        mean = float(np.mean(values))
        standard_deviation = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
        result[f"{stem}_mean_ms"] = mean
        result[f"{stem}_between_trial_std_ms"] = standard_deviation

    write_csv(output_root / "final_aggregate.csv", [result])
    write_csv(output_root / "final_audit.csv", audit_rows)

    total_mean = float(result["total_mean_ms"])
    total_std = float(result["total_between_trial_std_ms"])
    total_cv_pct = 100.0 * total_std / total_mean
    maximum_within_cv_pct = max(
        float(row["max_phase_within_trial_cv_pct"]) for row in audit_rows
    )
    stable = (
        total_cv_pct <= between_trial_cv_limit_pct
        and maximum_within_cv_pct <= within_trial_phase_cv_limit_pct
    )
    stability = {
        "between_trial_total_cv_pct": total_cv_pct,
        "between_trial_total_cv_limit_pct": between_trial_cv_limit_pct,
        "maximum_phase_within_trial_cv_pct": maximum_within_cv_pct,
        "phase_within_trial_cv_limit_pct": within_trial_phase_cv_limit_pct,
        "stability_pass": stable,
    }
    write_csv(output_root / "final_stability.csv", [stability])

    print("\nBatched frozen-schedule V100 aggregate")
    print("trial  fwd_ms      dX_ms       dK_ms       total_ms")
    for row in trial_rows:
        print(
            f"{int(row['trial']):5d}  "
            f"{float(row['fwd_mean_ms']):10.6f}  "
            f"{float(row['bwd_input_mean_ms']):10.6f}  "
            f"{float(row['bwd_weight_mean_ms']):10.6f}  "
            f"{float(row['total_mean_ms']):10.6f}"
        )
    print("\nAggregate:")
    print(json.dumps(result, indent=2))
    print("\nStability:")
    print(json.dumps(stability, indent=2))
    print(f"V100 stability remeasurement: {'PASS' if stable else 'FAIL'}")
    return stable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/external_baselines/tvm/v100_stability_v3"),
    )
    parser.add_argument("--B", type=int, default=DEFAULT_B)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--burnin", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--calls-per-sample", type=int, default=32)
    parser.add_argument("--queue-lead-in", type=int, default=64)
    parser.add_argument("--validation-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path("/home/huriyeh.babak/micromamba/envs/s4convd/bin/python"),
    )
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--between-trial-cv-limit-pct", type=float, default=2.0)
    parser.add_argument("--within-trial-phase-cv-limit-pct", type=float, default=5.0)
    args = parser.parse_args()
    for name in (
        "B",
        "H",
        "L",
        "K",
        "trials",
        "burnin",
        "warmup",
        "samples",
        "calls_per_sample",
        "queue_lead_in",
        "validation_batch",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "between_trial_cv_limit_pct",
        "within_trial_phase_cv_limit_pct",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    prepare_output_root(args.output_root)

    device = tvm.cuda(0)
    if not bool(device.exist):
        raise RuntimeError("TVM CUDA device 0 is unavailable")
    gpu = device_metadata(device)
    validate_gpu_slug("v100", str(gpu["gpu"]))
    target = cuda_target(gpu["compute_capability"])

    print("TVM:", tvm.__version__)
    print("GPU:", gpu["gpu"])
    print("Compute capability:", gpu["compute_capability"])
    print("Target:", json.dumps(target, sort_keys=True))
    print(
        "Protocol: compile once; fixed schedule; queue-backed batched "
        "CUDA-event timing; three continuous trials"
    )
    print(
        f"Timing samples per phase: {args.samples}; "
        f"calls per sample: {args.calls_per_sample}"
    )

    validation_batch = min(args.B, args.validation_batch)
    correctness, _ = validate_candidates(
        B=validation_batch,
        H=args.H,
        L=args.L,
        K=args.K,
        seed=args.seed,
        device=device,
        target=target,
        reference_python=args.reference_python,
        reference_device=args.reference_device,
    )
    write_csv(args.output_root / "correctness.csv", correctness)
    print_correctness(correctness)
    if not all(bool(row["allclose"]) for row in correctness):
        raise AssertionError("TVM fixed-schedule correctness failed")

    print("Compiling main-workload candidates once outside measured events...")
    compiled = compile_candidates(args.B, args.H, args.L, args.K, target)
    selected_fwd = candidate_named(compiled.forward, FIXED_NAMES["fwd"])
    selected_dx = candidate_named(compiled.backward_input, FIXED_NAMES["bwd_input"])
    selected_dk = candidate_named(compiled.backward_weight, FIXED_NAMES["bwd_weight"])

    x, weight, grad_y = allocate_main_inputs(
        args.B, args.H, args.L, args.K, args.seed, device
    )
    output = _empty_device((args.B, args.H, args.L), device)
    grad_x = _empty_device((args.B, args.H, args.L), device)
    grad_weight = _empty_device((args.H, args.K), device)
    workspace = _empty_device(compiled.workspace_shape, device)

    calls = {
        "fwd": lambda: selected_fwd(x, weight, output),
        "bwd_input": lambda: selected_dx(grad_y, weight, grad_x),
        "bwd_weight": lambda: selected_dk(grad_y, x, workspace, grad_weight),
    }
    configs = {
        "forward": selected_config(selected_fwd),
        "backward_input": selected_config(selected_dx),
        "backward_weight": selected_config(selected_dk),
    }
    print("Frozen configurations:")
    print(json.dumps(configs, indent=2))

    snapshots = [gpu_snapshot("before_burnin")]
    print(f"Continuous burn-in outside timing: {args.burnin} cycles")
    for _ in range(args.burnin):
        for phase in PHASES:
            calls[phase]()
    _device_sync(device)
    snapshots.append(gpu_snapshot("after_burnin"))

    trial_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for trial in range(1, args.trials + 1):
        print(f"\nTrial {trial}/{args.trials}: {args.warmup} warm-up cycles")
        for _ in range(args.warmup):
            for phase in PHASES:
                calls[phase]()
        _device_sync(device)
        snapshots.append(gpu_snapshot(f"trial{trial}_before"))

        raw_rows: list[dict[str, object]] = []
        summary_rows: list[dict[str, object]] = []
        phase_means: dict[str, float] = {}
        phase_cvs: dict[str, float] = {}
        for phase in PHASES:
            measurements = measure_batched_cuda_calls(
                calls[phase],
                args.samples,
                args.calls_per_sample,
                args.queue_lead_in,
            )
            phase_stats = stats(measurements)
            phase_stats["samples"] = phase_stats.pop("runs")
            phase_means[phase] = float(phase_stats["mean_ms"])
            phase_cvs[phase] = (
                100.0
                * float(phase_stats["std_ms"])
                / float(phase_stats["mean_ms"])
            )
            for sample, milliseconds in enumerate(measurements, start=1):
                raw_rows.append(
                    {
                        "technique": TECHNIQUE,
                        "gpu": gpu["gpu"],
                        "B": args.B,
                        "H": args.H,
                        "L": args.L,
                        "K": args.K,
                        "phase": phase,
                        "sample": sample,
                        "calls_per_sample": args.calls_per_sample,
                        "runtime_ms_per_call": milliseconds,
                    }
                )
            summary_rows.append(
                {
                    "technique": TECHNIQUE,
                    "gpu": gpu["gpu"],
                    "B": args.B,
                    "H": args.H,
                    "L": args.L,
                    "K": args.K,
                    "phase": phase,
                    "calls_per_sample": args.calls_per_sample,
                    "runtime_unit": "amortized_ms_per_call",
                    **phase_stats,
                }
            )

        snapshots.append(gpu_snapshot(f"trial{trial}_after"))
        trial_dir = args.output_root / f"trial{trial}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        write_csv(trial_dir / "raw.csv", raw_rows)
        write_csv(trial_dir / "summary.csv", summary_rows)
        write_csv(trial_dir / "correctness.csv", correctness)

        metadata = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tvm": tvm.__version__,
            "gpu": gpu,
            "target": target,
            "dtype": "float32",
            "B": args.B,
            "H": args.H,
            "L": args.L,
            "K": args.K,
            "trial": trial,
            "timing_samples_per_phase": args.samples,
            "calls_per_timing_sample": args.calls_per_sample,
            "measured_calls_per_phase": args.samples * args.calls_per_sample,
            "burnin_cycles": args.burnin,
            "warmup_cycles": args.warmup,
            "unmeasured_queue_lead_in_calls_per_phase": args.queue_lead_in,
            "compile_timed": False,
            "validation_timed": False,
            "allocation_timed": False,
            "autotuning_in_remeasurement": False,
            "schedule_selection_source": "independent initial run job 113124",
            "selected_configs": configs,
            "backward_weight_launches_per_call": 2,
            "timing": (
                "amortized per-call CUDA-event time for fixed-size batches "
                "on the default stream, with an excluded same-stream "
                "queue lead-in"
            ),
            "semantics": "explicit same padding; left=K//2; output length=L",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        (trial_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )

        total = sum(phase_means.values())
        trial_rows.append(
            {
                "trial": trial,
                "fwd_mean_ms": phase_means["fwd"],
                "bwd_input_mean_ms": phase_means["bwd_input"],
                "bwd_weight_mean_ms": phase_means["bwd_weight"],
                "total_mean_ms": total,
            }
        )
        audit_rows.append(
            {
                "trial": trial,
                "correctness_rows": len(correctness),
                "correctness_pass": True,
                "fwd_samples": args.samples,
                "bwd_input_samples": args.samples,
                "bwd_weight_samples": args.samples,
                "calls_per_sample": args.calls_per_sample,
                "measured_calls_per_phase": args.samples * args.calls_per_sample,
                "fixed_configs_match": True,
                "fwd_within_trial_cv_pct": phase_cvs["fwd"],
                "bwd_input_within_trial_cv_pct": phase_cvs["bwd_input"],
                "bwd_weight_within_trial_cv_pct": phase_cvs["bwd_weight"],
                "max_phase_within_trial_cv_pct": max(phase_cvs.values()),
            }
        )
        print(
            f"Trial {trial}: total={total:.6f} ms; "
            f"max phase CV={max(phase_cvs.values()):.3f}%"
        )

    snapshots.append(gpu_snapshot("finished"))
    (args.output_root / "gpu_snapshots.json").write_text(
        json.dumps(snapshots, indent=2) + "\n", encoding="utf-8"
    )
    protocol = {
        "purpose": "independent V100 batched stability remeasurement",
        "initial_results_preserved_at": (
            "results/external_baselines/tvm/v100_unstable_job113124"
        ),
        "earlier_stability_results_preserved_at": [
            "results/external_baselines/tvm/v100_stability",
            "results/external_baselines/tvm/v100_stability_v2",
        ],
        "fixed_candidates": FIXED_NAMES,
        "compile_count_main_shape": 1,
        "trials": args.trials,
        "burnin_cycles": args.burnin,
        "warmup_cycles_per_trial": args.warmup,
        "timing_samples_per_phase_per_trial": args.samples,
        "calls_per_timing_sample": args.calls_per_sample,
        "measured_calls_per_phase_per_trial": args.samples * args.calls_per_sample,
        "unmeasured_queue_lead_in_calls_per_phase": args.queue_lead_in,
        "queue_lead_in_included_in_event_time": False,
        "between_trial_total_cv_limit_pct": args.between_trial_cv_limit_pct,
        "within_trial_phase_cv_limit_pct": args.within_trial_phase_cv_limit_pct,
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n", encoding="utf-8"
    )

    stable = aggregate(
        args.output_root,
        trial_rows,
        audit_rows,
        args.between_trial_cv_limit_pct,
        args.within_trial_phase_cv_limit_pct,
    )
    if not stable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
