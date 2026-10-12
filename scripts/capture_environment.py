#!/usr/bin/env python3
"""Capture software/hardware metadata; performs no benchmark measurements."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def package_version(import_name: str, distribution_name: str | None = None) -> str | None:
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    if value is not None:
        return str(value)
    try:
        return importlib.metadata.version(distribution_name or import_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def command_output(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect() -> dict[str, Any]:
    packages = {
        name: package_version(import_name, distribution)
        for name, import_name, distribution in [
            ("numpy", "numpy", None),
            ("pandas", "pandas", None),
            ("matplotlib", "matplotlib", None),
            ("torch", "torch", None),
            ("pycuda", "pycuda", None),
            ("nvidia_ml_py", "pynvml", "nvidia-ml-py"),
        ]
    }
    payload: dict[str, Any] = {
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "scheduler": {
            key: os.environ.get(key)
            for key in [
                "SLURM_JOB_ID",
                "SLURM_JOB_NODELIST",
                "SLURM_JOB_PARTITION",
                "CUDA_VISIBLE_DEVICES",
                "CONDA_PREFIX",
            ]
        },
        "commands": {
            "nvcc": command_output(["nvcc", "--version"]),
            "ncu": command_output(["ncu", "--version"]),
            "nvidia_smi": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,compute_cap,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
    }

    try:
        import torch

        torch_data: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            torch_data.update(
                {
                    "device_index": device,
                    "device_name": torch.cuda.get_device_name(device),
                    "compute_capability": ".".join(
                        str(value) for value in torch.cuda.get_device_capability(device)
                    ),
                    "sm_count": properties.multi_processor_count,
                    "total_memory_bytes": properties.total_memory,
                }
            )
        payload["torch_runtime"] = torch_data
    except Exception as exc:
        payload["torch_runtime"] = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifact/environment_capture.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2) + "\n", encoding="utf-8")
    print(f"Environment capture: PASS ({args.output})")


if __name__ == "__main__":
    main()
