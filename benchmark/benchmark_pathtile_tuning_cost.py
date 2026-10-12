#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import benchmark.benchmark_pathtile_candidate_sweep as base
from kernels.warp_tiled_ampere import (
    AmpereWarpRuntime as OriginalAmpereWarpRuntime,
)


TIMING: dict[str, float | int] = {
    "module_build_seconds": 0.0,
    "module_build_calls": 0,
    "validation_seconds": 0.0,
    "validation_calls": 0,
    "warmup_seconds": 0.0,
    "warmup_calls": 0,
    "benchmark_seconds": 0.0,
    "benchmark_calls": 0,
}

MODULE_BUILDS: list[dict[str, Any]] = []


def cache_file_count(path: str | None) -> int | None:
    if not path:
        return None

    root = Path(path)

    if not root.exists():
        return 0

    return sum(
        1
        for p in root.rglob("*")
        if p.is_file()
    )


class TimedAmpereWarpRuntime(OriginalAmpereWarpRuntime):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        start = time.perf_counter()

        try:
            super().__init__(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start

            TIMING["module_build_seconds"] = (
                float(TIMING["module_build_seconds"])
                + elapsed
            )
            TIMING["module_build_calls"] = (
                int(TIMING["module_build_calls"])
                + 1
            )

        MODULE_BUILDS.append(
            {
                "warps_per_block": int(self.warps_per_block),
                "bwdk_btile": int(self.bwdk_btile),
                "bwdk_block": int(self.bwdk_block),
                "seconds": elapsed,
            }
        )


def timed_wrapper(
    fn,
    seconds_key: str,
    calls_key: str,
):
    def wrapped(*args: Any, **kwargs: Any):
        start = time.perf_counter()

        try:
            return fn(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start

            TIMING[seconds_key] = (
                float(TIMING[seconds_key])
                + elapsed
            )
            TIMING[calls_key] = (
                int(TIMING[calls_key])
                + 1
            )

    return wrapped


# Patch only the symbols used by the original run_trial().
base.AmpereWarpRuntime = TimedAmpereWarpRuntime

base.validate_schedule = timed_wrapper(
    base.validate_schedule,
    "validation_seconds",
    "validation_calls",
)

base.warmup_schedule = timed_wrapper(
    base.warmup_schedule,
    "warmup_seconds",
    "warmup_calls",
)

base.measure_schedule = timed_wrapper(
    base.measure_schedule,
    "benchmark_seconds",
    "benchmark_calls",
)


def main() -> None:
    args = base.parse_args()

    if args.mode != "run":
        raise ValueError(
            "benchmark_pathtile_tuning_cost.py supports "
            "--mode run only. Use the original benchmark "
            "script for aggregation."
        )

    cases = base.cases_for_preset(args.preset)
    specs = base.candidate_specs(args)

    source_configs = set()

    for kind, value in specs:
        warps = (
            value
            if kind == base.FWD_DX_KIND
            else base.DEFAULT_WARPS_PER_BLOCK
        )

        block = (
            value
            if kind == base.BWDK_KIND
            else base.DEFAULT_BWDK_BLOCK
        )

        source_configs.add(
            (
                int(warps),
                int(base.DEFAULT_BWDK_BTILE),
                int(block),
            )
        )

    n_shapes = len(cases)

    path_candidate_evaluations = (
        len(args.warps_candidates) * n_shapes * 2
        + len(args.bwdk_block_candidates) * n_shapes
    )

    schedule_shape_evaluations = (
        len(specs) * n_shapes
    )

    measured_kernel_launches = (
        path_candidate_evaluations
        * args.runs
    )

    warmup_kernel_launches = (
        path_candidate_evaluations
        * args.warmup
    )

    cache_dir = os.environ.get(
        "PYCUDA_CACHE_DIR"
    )

    cache_before = cache_file_count(
        cache_dir
    )

    total_start = time.perf_counter()
    status = "PASS"
    error = None

    try:
        base.run_trial(args)

    except Exception as exc:
        status = "FAIL"
        error = repr(exc)
        raise

    finally:
        total_wall_seconds = (
            time.perf_counter()
            - total_start
        )

        cache_after = cache_file_count(
            cache_dir
        )

        accounted = (
            float(TIMING["module_build_seconds"])
            + float(TIMING["validation_seconds"])
            + float(TIMING["warmup_seconds"])
            + float(TIMING["benchmark_seconds"])
        )

        other_wall_seconds = max(
            0.0,
            total_wall_seconds - accounted,
        )

        record = {
            "status": status,
            "error": error,
            "gpu": (
                base.torch.cuda.get_device_name(
                    args.device
                )
                if base.torch.cuda.is_available()
                else None
            ),
            "trial_id": int(args.trial_id),
            "preset": args.preset,
            "training_shapes": n_shapes,
            "candidate_specs": len(specs),
            "unique_source_configs": len(
                source_configs
            ),
            "schedule_shape_evaluations": (
                schedule_shape_evaluations
            ),
            "path_candidate_evaluations": (
                path_candidate_evaluations
            ),
            "warmup_repetitions": int(
                args.warmup
            ),
            "measurement_repetitions": int(
                args.runs
            ),
            "warmup_kernel_launches": (
                warmup_kernel_launches
            ),
            "measured_kernel_launches": (
                measured_kernel_launches
            ),
            "pycuda_cache_dir": cache_dir,
            "cache_files_before": (
                cache_before
            ),
            "cache_files_after": (
                cache_after
            ),
            **TIMING,
            "other_wall_seconds": (
                other_wall_seconds
            ),
            "total_wall_seconds": (
                total_wall_seconds
            ),
        }

        trial_output = (
            Path(args.output_dir)
            / f"trial{args.trial_id}"
        )
        trial_output.mkdir(
            parents=True,
            exist_ok=True,
        )

        json_path = (
            trial_output
            / "tuning_timing.json"
        )

        csv_path = (
            trial_output
            / "tuning_timing.csv"
        )

        build_path = (
            trial_output
            / "module_build_times.csv"
        )

        json_path.write_text(
            json.dumps(
                record,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        pd.DataFrame(
            [record]
        ).to_csv(
            csv_path,
            index=False,
        )

        pd.DataFrame(
            MODULE_BUILDS
        ).to_csv(
            build_path,
            index=False,
        )

        print("\n" + "=" * 80)
        print("PATHTILE TUNING-COST TIMING")
        print("=" * 80)

        keys = [
            "training_shapes",
            "candidate_specs",
            "unique_source_configs",
            "path_candidate_evaluations",
            "warmup_kernel_launches",
            "measured_kernel_launches",
            "cache_files_before",
            "cache_files_after",
            "module_build_seconds",
            "validation_seconds",
            "warmup_seconds",
            "benchmark_seconds",
            "other_wall_seconds",
            "total_wall_seconds",
        ]

        for key in keys:
            print(
                f"{key}: {record[key]}"
            )

        print("\nTiming file:")
        print(json_path)


if __name__ == "__main__":
    main()
