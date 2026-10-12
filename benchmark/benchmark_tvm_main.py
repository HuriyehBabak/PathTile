"""Correctness, autotuning, timing, and aggregation for the TVM-TIRx baseline.

The protocol intentionally matches ``benchmark_triton_main.py``:

* strict FP32 and explicit same-padding semantics;
* independent forward, backward-input, and backward-weight paths;
* candidate compilation/autotuning outside final timing;
* 50 warm-up iterations and 100 CUDA-event measurements by default;
* two-stage dK timing includes both launches and excludes allocation;
* correctness is checked against PyTorch/cuDNN in the existing project env.

The script runs under the isolated TVM Python environment and does not repeat
any existing PyTorch, PathTile, or Triton performance measurements.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import re
import socket
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import tvm
from cuda.bindings import runtime as cudart

from kernels.tvm_depthwise import (
    CompiledCandidateSet,
    compile_candidates,
)


DEFAULT_B = 16384
DEFAULT_H = 128
DEFAULT_L = 48
DEFAULT_K = 48
DEFAULT_WARMUP = 50
DEFAULT_RUNS = 100
DEFAULT_TUNE_WARMUP = 5
DEFAULT_TUNE_RUNS = 20
DEFAULT_SEED = 42
PHASES = ("fwd", "bwd_input", "bwd_weight")
TECHNIQUE = "tvm_tirx_autotuned"


def _status_code(status: Any) -> int:
    value = getattr(status, "value", status)
    return int(value)


def _cuda_result(result: Any, operation: str) -> tuple[Any, ...]:
    items = result if isinstance(result, tuple) else (result,)
    if not items:
        raise RuntimeError(f"{operation} returned no CUDA status")
    status = items[0]
    if _status_code(status) != 0:
        label = getattr(status, "name", str(status))
        raise RuntimeError(f"{operation} failed with {label}")
    return tuple(items[1:])


def _default_stream() -> Any:
    stream_type = getattr(cudart, "cudaStream_t", None)
    return stream_type(0) if stream_type is not None else 0


def _create_event() -> Any:
    values = _cuda_result(cudart.cudaEventCreate(), "cudaEventCreate")
    if len(values) != 1:
        raise RuntimeError("cudaEventCreate returned an unexpected result")
    return values[0]


def _record_event(event: Any) -> None:
    _cuda_result(
        cudart.cudaEventRecord(event, _default_stream()),
        "cudaEventRecord",
    )


def _destroy_event(event: Any) -> None:
    _cuda_result(cudart.cudaEventDestroy(event), "cudaEventDestroy")


def measure_cuda_calls(call: Callable[[], object], runs: int) -> list[float]:
    """Measure calls with independent event pairs on CUDA's default stream."""

    if runs <= 0:
        raise ValueError("runs must be positive")
    event_pairs: list[tuple[Any, Any]] = []
    try:
        for _ in range(runs):
            start = _create_event()
            end = _create_event()
            event_pairs.append((start, end))
            _record_event(start)
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
                raise RuntimeError("cudaEventElapsedTime returned an unexpected result")
            milliseconds.append(float(values[0]))
        return milliseconds
    finally:
        for start, end in event_pairs:
            _destroy_event(start)
            _destroy_event(end)


def _to_device(array: np.ndarray, device: tvm.runtime.Device) -> Any:
    contiguous = np.ascontiguousarray(array)
    tensor_constructor = getattr(tvm.runtime, "tensor", None)
    if tensor_constructor is not None:
        try:
            return tensor_constructor(contiguous, device=device)
        except TypeError:
            return tensor_constructor(contiguous, device)
    return tvm.nd.array(contiguous, device=device)


def _empty_device(shape: Sequence[int], device: tvm.runtime.Device) -> Any:
    # This compatibility path works with both the new TVM Tensor API and the
    # legacy NDArray API.  The temporary host array is never in a timed region.
    return _to_device(np.empty(tuple(shape), dtype=np.float32), device)


def _tensor_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "numpy"):
        return np.asarray(tensor.numpy())
    return np.asarray(tensor.asnumpy())


def _device_sync(device: tvm.runtime.Device) -> None:
    sync = getattr(device, "sync", None)
    if callable(sync):
        sync()
    else:
        _cuda_result(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def _device_property(device: tvm.runtime.Device, name: str) -> Any | None:
    try:
        value = getattr(device, name)
        value = value() if callable(value) else value
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        if isinstance(value, np.generic):
            return value.item()
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)
    except (AttributeError, RuntimeError, TypeError):
        return None


def device_metadata(device: tvm.runtime.Device) -> dict[str, object]:
    result: dict[str, object] = {
        "gpu": _device_property(device, "device_name") or "unknown CUDA GPU",
        "compute_capability": _device_property(device, "compute_version") or "unknown",
        "global_memory_bytes": _device_property(device, "total_global_memory"),
        "multiprocessor_count": _device_property(device, "multi_processor_count"),
        "max_threads_per_block": _device_property(device, "max_threads_per_block"),
    }

    if result["gpu"] == "unknown CUDA GPU" or result["compute_capability"] == "unknown":
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,compute_cap,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if query.returncode == 0 and query.stdout.strip():
            first = [field.strip() for field in query.stdout.splitlines()[0].split(",")]
            if len(first) >= 4:
                result["gpu"] = first[0]
                result["compute_capability"] = first[1]
                result["nvidia_smi_memory_mib"] = first[2]
                result["nvidia_driver"] = first[3]
    return result


def validate_gpu_slug(slug: str, gpu_name: str) -> None:
    normalized = gpu_name.lower()
    expected = {
        "a100": "a100",
        "a6000": "a6000",
        "v100": "v100",
    }[slug]
    if expected not in normalized:
        raise RuntimeError(
            f"--gpu-slug={slug} does not match active device {gpu_name!r}"
        )


def cuda_target(compute_capability: object) -> dict[str, str]:
    match = re.search(r"(\d+)\.(\d+)", str(compute_capability))
    if match is None:
        raise RuntimeError(
            f"Could not derive a CUDA architecture from {compute_capability!r}"
        )
    return {
        "kind": "cuda",
        "arch": f"sm_{match.group(1)}{match.group(2)}",
    }


def make_numpy_inputs(
    B: int,
    H: int,
    L: int,
    K: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, H, L), dtype=np.float32)
    weight = rng.standard_normal((H, K), dtype=np.float32)
    grad_y = rng.standard_normal((B, H, L), dtype=np.float32)
    return x, weight, grad_y


def pytorch_reference(
    x: np.ndarray,
    weight: np.ndarray,
    grad_y: np.ndarray,
    reference_python: Path,
    reference_device: str,
) -> dict[str, np.ndarray]:
    helper = Path(__file__).with_name("tvm_pytorch_reference.py").resolve()
    if not helper.exists():
        raise FileNotFoundError(helper)
    if not reference_python.exists():
        raise FileNotFoundError(reference_python)

    with tempfile.TemporaryDirectory(prefix="tvm-reference-") as temporary:
        input_path = Path(temporary) / "input.npz"
        output_path = Path(temporary) / "output.npz"
        np.savez(input_path, x=x, weight=weight, grad_y=grad_y)
        subprocess.run(
            [
                str(reference_python),
                str(helper),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--device",
                reference_device,
            ],
            check=True,
        )
        with np.load(output_path) as arrays:
            return {
                "fwd": np.asarray(arrays["y"], dtype=np.float32).copy(),
                "bwd_input": np.asarray(arrays["grad_x"], dtype=np.float32).copy(),
                "bwd_weight": np.asarray(arrays["grad_weight"], dtype=np.float32).copy(),
            }


def comparison_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    atol: float,
    rtol: float,
) -> dict[str, object]:
    actual64 = np.asarray(actual, dtype=np.float64)
    expected64 = np.asarray(expected, dtype=np.float64)
    difference = actual64 - expected64
    max_abs = float(np.max(np.abs(difference)))
    denominator = max(float(np.max(np.abs(expected64))), 1e-12)
    l2_denominator = max(float(np.linalg.norm(expected64.ravel())), 1e-12)
    return {
        "max_abs": max_abs,
        "normalized_max_abs": max_abs / denominator,
        "relative_l2": float(np.linalg.norm(difference.ravel())) / l2_denominator,
        "atol": atol,
        "rtol": rtol,
        "allclose": bool(np.allclose(actual64, expected64, atol=atol, rtol=rtol)),
    }


def validate_candidates(
    B: int,
    H: int,
    L: int,
    K: int,
    seed: int,
    device: tvm.runtime.Device,
    target: object,
    reference_python: Path,
    reference_device: str,
) -> tuple[list[dict[str, object]], CompiledCandidateSet]:
    x_numpy, weight_numpy, grad_y_numpy = make_numpy_inputs(B, H, L, K, seed)
    reference = pytorch_reference(
        x_numpy,
        weight_numpy,
        grad_y_numpy,
        reference_python,
        reference_device,
    )
    compiled = compile_candidates(B, H, L, K, target)

    x = _to_device(x_numpy, device)
    weight = _to_device(weight_numpy, device)
    grad_y = _to_device(grad_y_numpy, device)
    output = _empty_device((B, H, L), device)
    grad_x = _empty_device((B, H, L), device)
    grad_weight = _empty_device((H, K), device)
    workspace = _empty_device(compiled.workspace_shape, device)

    rows: list[dict[str, object]] = []
    tolerances = {
        "fwd": (2e-3, 2e-3),
        "bwd_input": (2e-3, 2e-3),
        "bwd_weight": (2e-2, 2e-3),
    }
    for candidate in compiled.forward:
        candidate(x, weight, output)
        _device_sync(device)
        atol, rtol = tolerances["fwd"]
        rows.append(
            {
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": "fwd",
                "candidate": candidate.name,
                "candidate_config": json.dumps(candidate.config, sort_keys=True),
                **comparison_metrics(_tensor_numpy(output), reference["fwd"], atol, rtol),
            }
        )
    for candidate in compiled.backward_input:
        candidate(grad_y, weight, grad_x)
        _device_sync(device)
        atol, rtol = tolerances["bwd_input"]
        rows.append(
            {
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": "bwd_input",
                "candidate": candidate.name,
                "candidate_config": json.dumps(candidate.config, sort_keys=True),
                **comparison_metrics(
                    _tensor_numpy(grad_x),
                    reference["bwd_input"],
                    atol,
                    rtol,
                ),
            }
        )
    for candidate in compiled.backward_weight:
        candidate(grad_y, x, workspace, grad_weight)
        _device_sync(device)
        atol, rtol = tolerances["bwd_weight"]
        rows.append(
            {
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": "bwd_weight",
                "candidate": candidate.name,
                "candidate_config": json.dumps(candidate.config, sort_keys=True),
                **comparison_metrics(
                    _tensor_numpy(grad_weight),
                    reference["bwd_weight"],
                    atol,
                    rtol,
                ),
            }
        )
    return rows, compiled


def stats(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot summarize an empty measurement")
    count = int(array.size)
    mean = float(np.mean(array))
    standard_deviation = float(np.std(array, ddof=1)) if count > 1 else 0.0
    sem = standard_deviation / math.sqrt(count) if count > 1 else 0.0
    margin = 1.96 * sem
    return {
        "runs": count,
        "mean_ms": mean,
        "median_ms": float(np.median(array)),
        "std_ms": standard_deviation,
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
        "min_ms": float(np.min(array)),
        "p05_ms": float(np.quantile(array, 0.05)),
        "p95_ms": float(np.quantile(array, 0.95)),
        "max_ms": float(np.max(array)),
    }


def tune_candidates(
    candidates: Sequence[Any],
    call_for: Callable[[Any], Callable[[], object]],
    device: tvm.runtime.Device,
    warmup: int,
    runs: int,
) -> tuple[Any, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    selected: Any | None = None
    selected_median = math.inf
    for candidate in candidates:
        call = call_for(candidate)
        for _ in range(warmup):
            call()
        _device_sync(device)
        measurements = measure_cuda_calls(call, runs)
        candidate_stats = stats(measurements)
        rows.append(
            {
                "candidate": candidate.name,
                "config": dict(candidate.config),
                **candidate_stats,
            }
        )
        median = float(candidate_stats["median_ms"])
        if median < selected_median:
            selected = candidate
            selected_median = median
    if selected is None:
        raise RuntimeError("No candidate was selected")
    return selected, rows


def allocate_main_inputs(
    B: int,
    H: int,
    L: int,
    K: int,
    seed: int,
    device: tvm.runtime.Device,
) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(seed)
    x_numpy = rng.standard_normal((B, H, L), dtype=np.float32)
    x = _to_device(x_numpy, device)
    del x_numpy
    weight_numpy = rng.standard_normal((H, K), dtype=np.float32)
    weight = _to_device(weight_numpy, device)
    del weight_numpy
    grad_y_numpy = rng.standard_normal((B, H, L), dtype=np.float32)
    grad_y = _to_device(grad_y_numpy, device)
    del grad_y_numpy
    return x, weight, grad_y


def benchmark_main(
    B: int,
    H: int,
    L: int,
    K: int,
    warmup: int,
    runs: int,
    tune_warmup: int,
    tune_runs: int,
    seed: int,
    device: tvm.runtime.Device,
    target: object,
    gpu_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    x, weight, grad_y = allocate_main_inputs(B, H, L, K, seed, device)
    output = _empty_device((B, H, L), device)
    grad_x = _empty_device((B, H, L), device)
    grad_weight = _empty_device((H, K), device)

    print("Compiling TVM-TIRx candidates outside measured events...")
    compiled = compile_candidates(B, H, L, K, target)
    workspace = _empty_device(compiled.workspace_shape, device)

    print(
        "Autotuning outside final timing: "
        f"{tune_warmup} warm-up + {tune_runs} measured calls per candidate"
    )
    selected_fwd, fwd_tuning = tune_candidates(
        compiled.forward,
        lambda candidate: lambda: candidate(x, weight, output),
        device,
        tune_warmup,
        tune_runs,
    )
    selected_dx, dx_tuning = tune_candidates(
        compiled.backward_input,
        lambda candidate: lambda: candidate(grad_y, weight, grad_x),
        device,
        tune_warmup,
        tune_runs,
    )
    selected_dk, dk_tuning = tune_candidates(
        compiled.backward_weight,
        lambda candidate: lambda: candidate(grad_y, x, workspace, grad_weight),
        device,
        tune_warmup,
        tune_runs,
    )

    calls: dict[str, Callable[[], object]] = {
        "fwd": lambda: selected_fwd(x, weight, output),
        "bwd_input": lambda: selected_dx(grad_y, weight, grad_x),
        "bwd_weight": lambda: selected_dk(grad_y, x, workspace, grad_weight),
    }
    print(f"Final warm-up: {warmup} iterations")
    for _ in range(warmup):
        for phase in PHASES:
            calls[phase]()
    _device_sync(device)

    print(f"Final measured runs per phase: {runs}")
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for phase in PHASES:
        measurements = measure_cuda_calls(calls[phase], runs)
        for run_id, milliseconds in enumerate(measurements, start=1):
            raw_rows.append(
                {
                    "technique": TECHNIQUE,
                    "gpu": gpu_name,
                    "B": B,
                    "H": H,
                    "L": L,
                    "K": K,
                    "phase": phase,
                    "run": run_id,
                    "runtime_ms": milliseconds,
                }
            )
        summary_rows.append(
            {
                "technique": TECHNIQUE,
                "gpu": gpu_name,
                "B": B,
                "H": H,
                "L": L,
                "K": K,
                "phase": phase,
                **stats(measurements),
            }
        )

    extra_metadata: dict[str, object] = {
        "target": compiled.target,
        "selected_configs": {
            "forward": {
                "candidate": selected_fwd.name,
                **selected_fwd.config,
            },
            "backward_input": {
                "candidate": selected_dx.name,
                **selected_dx.config,
            },
            "backward_weight": {
                "candidate": selected_dk.name,
                **selected_dk.config,
            },
        },
        "autotune_measurements": {
            "forward": fwd_tuning,
            "backward_input": dx_tuning,
            "backward_weight": dk_tuning,
        },
        "workspace_shape": list(compiled.workspace_shape),
        "workspace_bytes": int(np.prod(compiled.workspace_shape) * 4),
        "backward_weight_launches_per_call": 2,
        "generated_cuda_source_chars": {
            "forward": selected_fwd.cuda_source_chars,
            "backward_input": selected_dx.cuda_source_chars,
            "backward_weight_partial": selected_dk.partial.cuda_source_chars,
            "backward_weight_reduce": selected_dk.reduce.cuda_source_chars,
        },
    }
    return raw_rows, summary_rows, extra_metadata


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError(f"Rows for {path} do not have a consistent schema")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_correctness(rows: Sequence[dict[str, object]]) -> None:
    print("phase       candidate                 max_abs      relative_l2  allclose")
    for row in rows:
        print(
            f"{str(row['phase']):12s} "
            f"{str(row['candidate']):24s} "
            f"{float(row['max_abs']):.6e}  "
            f"{float(row['relative_l2']):.6e}  "
            f"{str(row['allclose']):>8s}"
        )


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata(
    args: argparse.Namespace,
    gpu: dict[str, object],
    target: object,
) -> dict[str, object]:
    libinfo: dict[str, object] = {}
    try:
        libinfo = dict(tvm.support.libinfo())
    except (AttributeError, RuntimeError, TypeError):
        pass
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "technique": TECHNIQUE,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tvm": tvm.__version__,
        "tvm_git_commit": libinfo.get("GIT_COMMIT_HASH"),
        "tvm_use_cuda": libinfo.get("USE_CUDA"),
        "tvm_use_llvm": libinfo.get("USE_LLVM"),
        "apache_tvm_ffi": package_version("apache-tvm-ffi"),
        "cuda_bindings": package_version("cuda-bindings"),
        **gpu,
        "target": target,
        "dtype": "float32",
        "B": args.B,
        "H": args.H,
        "L": args.L,
        "K": args.K,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "autotune_warmup_runs": args.tune_warmup,
        "autotune_measured_runs": args.tune_runs,
        "seed": args.seed,
        "compile_and_autotune_timed": False,
        "allocation_timed": False,
        "timing": "CUDA events on the default stream",
        "semantics": "explicit same padding; left=K//2; output length=L",
        "pytorch_reference_python": str(args.reference_python),
        "pytorch_reference_device": args.reference_device,
    }


def prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite non-empty output directory: {path}"
        )
    path.mkdir(parents=True, exist_ok=True)


def run_smoke(
    args: argparse.Namespace,
    device: tvm.runtime.Device,
    gpu: dict[str, object],
    target: object,
) -> None:
    output_dir = args.output_root / args.gpu_slug / "smoke"
    prepare_output_directory(output_dir, args.overwrite)
    rows, compiled = validate_candidates(
        B=8,
        H=8,
        L=16,
        K=8,
        seed=args.seed,
        device=device,
        target=target,
        reference_python=args.reference_python,
        reference_device=args.reference_device,
    )
    write_csv(output_dir / "correctness.csv", rows)
    print_correctness(rows)
    if not all(bool(row["allclose"]) for row in rows):
        raise AssertionError("TVM smoke correctness failed")
    metadata = environment_metadata(args, gpu, target)
    metadata["validated_candidates"] = {
        "forward": len(compiled.forward),
        "backward_input": len(compiled.backward_input),
        "backward_weight": len(compiled.backward_weight),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TVM-TIRx smoke: PASS on {gpu['gpu']}")
    print(f"Saved: {output_dir / 'correctness.csv'}")


def run_trial(
    args: argparse.Namespace,
    device: tvm.runtime.Device,
    gpu: dict[str, object],
    target: object,
) -> None:
    output_dir = args.output_root / args.gpu_slug / f"trial{args.trial_id}"
    prepare_output_directory(output_dir, args.overwrite)

    validation_batch = min(args.B, args.validation_batch)
    print(
        "Correctness against PyTorch/cuDNN: "
        f"B={validation_batch}, H={args.H}, L={args.L}, K={args.K}"
    )
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
    write_csv(output_dir / "correctness.csv", correctness)
    print_correctness(correctness)
    if not all(bool(row["allclose"]) for row in correctness):
        raise AssertionError("TVM correctness failed; performance benchmark was not run")

    raw, summary, extra_metadata = benchmark_main(
        B=args.B,
        H=args.H,
        L=args.L,
        K=args.K,
        warmup=args.warmup,
        runs=args.runs,
        tune_warmup=args.tune_warmup,
        tune_runs=args.tune_runs,
        seed=args.seed,
        device=device,
        target=target,
        gpu_name=str(gpu["gpu"]),
    )
    write_csv(output_dir / "raw.csv", raw)
    write_csv(output_dir / "summary.csv", summary)
    metadata = environment_metadata(args, gpu, target)
    metadata.update(extra_metadata)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    total = sum(float(row["mean_ms"]) for row in summary)
    print("\nTVM-TIRx FP32 main-workload summary")
    print("phase          mean_ms      std_ms")
    for row in summary:
        print(
            f"{str(row['phase']):12s} "
            f"{float(row['mean_ms']):11.6f} "
            f"{float(row['std_ms']):11.6f}"
        )
    print(f"Sum of phase means: {total:.6f} ms")
    print(f"Saved trial directory: {output_dir}")


def aggregate_trials(args: argparse.Namespace) -> None:
    root = args.output_root / args.gpu_slug
    trial_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for trial_id in range(1, args.expected_trials + 1):
        trial_dir = root / f"trial{trial_id}"
        summary_path = trial_dir / "summary.csv"
        raw_path = trial_dir / "raw.csv"
        correctness_path = trial_dir / "correctness.csv"
        metadata_path = trial_dir / "metadata.json"
        for path in (summary_path, raw_path, correctness_path, metadata_path):
            if not path.exists():
                raise FileNotFoundError(path)

        correctness = read_csv(correctness_path)
        if not correctness or any(row["allclose"].lower() != "true" for row in correctness):
            raise AssertionError(f"Correctness audit failed: {correctness_path}")
        summary = read_csv(summary_path)
        means = {row["phase"]: float(row["mean_ms"]) for row in summary}
        missing = [phase for phase in PHASES if phase not in means]
        if missing:
            raise ValueError(f"{summary_path} is missing phases: {missing}")
        raw = read_csv(raw_path)
        measured_counts = {
            phase: sum(row["phase"] == phase for row in raw) for phase in PHASES
        }
        if any(count != args.runs for count in measured_counts.values()):
            raise ValueError(
                f"{raw_path} has unexpected per-phase counts: {measured_counts}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("compile_and_autotune_timed") is not False:
            raise ValueError(f"Unexpected timing contract in {metadata_path}")
        if metadata.get("backward_weight_launches_per_call") != 2:
            raise ValueError(f"Unexpected dK launch count in {metadata_path}")

        trial_rows.append(
            {
                "trial": trial_id,
                "fwd_mean_ms": means["fwd"],
                "bwd_input_mean_ms": means["bwd_input"],
                "bwd_weight_mean_ms": means["bwd_weight"],
                "total_mean_ms": sum(means[phase] for phase in PHASES),
            }
        )
        audit_rows.append(
            {
                "trial": trial_id,
                "correctness_rows": len(correctness),
                "correctness_pass": True,
                "fwd_runs": measured_counts["fwd"],
                "bwd_input_runs": measured_counts["bwd_input"],
                "bwd_weight_runs": measured_counts["bwd_weight"],
                "selected_configs_present": bool(metadata.get("selected_configs")),
            }
        )

    write_csv(root / "final_per_trial.csv", trial_rows)
    aggregate: dict[str, object] = {
        "gpu_slug": args.gpu_slug,
        "method": "TVM-TIRx",
        "trials": len(trial_rows),
    }
    for column in (
        "fwd_mean_ms",
        "bwd_input_mean_ms",
        "bwd_weight_mean_ms",
        "total_mean_ms",
    ):
        stem = column.removesuffix("_mean_ms")
        values = np.asarray([float(row[column]) for row in trial_rows], dtype=np.float64)
        aggregate[f"{stem}_mean_ms"] = float(np.mean(values))
        aggregate[f"{stem}_between_trial_std_ms"] = (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        )
    write_csv(root / "final_aggregate.csv", [aggregate])
    write_csv(root / "final_audit.csv", audit_rows)

    print("TVM-TIRx aggregation and audit: PASS")
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
    print(json.dumps(aggregate, indent=2))
    print(f"Saved: {root / 'final_per_trial.csv'}")
    print(f"Saved: {root / 'final_aggregate.csv'}")
    print(f"Saved: {root / 'final_audit.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "run", "aggregate"), required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/external_baselines/tvm"),
    )
    parser.add_argument("--gpu-slug", choices=("a100", "a6000", "v100"), required=True)
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--B", type=int, default=DEFAULT_B)
    parser.add_argument("--H", type=int, default=DEFAULT_H)
    parser.add_argument("--L", type=int, default=DEFAULT_L)
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--tune-warmup", type=int, default=DEFAULT_TUNE_WARMUP)
    parser.add_argument("--tune-runs", type=int, default=DEFAULT_TUNE_RUNS)
    parser.add_argument("--validation-batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--reference-python",
        type=Path,
        default=Path("/home/huriyeh.babak/micromamba/envs/s4convd/bin/python"),
    )
    parser.add_argument("--reference-device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for name in (
        "B",
        "H",
        "L",
        "K",
        "warmup",
        "runs",
        "tune_warmup",
        "tune_runs",
        "validation_batch",
        "trial_id",
        "expected_trials",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "aggregate":
        aggregate_trials(args)
        return

    device = tvm.cuda(0)
    if not bool(device.exist):
        raise RuntimeError("TVM CUDA device 0 is unavailable")
    gpu = device_metadata(device)
    validate_gpu_slug(args.gpu_slug, str(gpu["gpu"]))
    target = cuda_target(gpu["compute_capability"])
    print("TVM:", tvm.__version__)
    print("GPU:", gpu["gpu"])
    print("Compute capability:", gpu["compute_capability"])
    print("Target:", json.dumps(target, sort_keys=True))

    if args.mode == "smoke":
        run_smoke(args, device, gpu, target)
    else:
        run_trial(args, device, gpu, target)


if __name__ == "__main__":
    main()
