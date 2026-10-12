from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import socket
import zlib
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from benchmark.benchmark_shape_sweep_ampere import (
    ShapeCase,
    cases_for_preset,
    pytorch_depthwise_same,
)
from kernels.warp_tiled_ampere import (
    DEFAULT_BWDK_BLOCK,
    DEFAULT_BWDK_BTILE,
    DEFAULT_WARPS_PER_BLOCK,
    KERNEL_EVENTS,
    AmpereWarpRuntime,
    KernelEventCollector,
    configure_torch,
    max_abs_diff,
    normalized_max_abs,
)


FWD_DX_KIND = "warps_per_block"
BWDK_KIND = "bwdk_block"
PHASES = ("fwd", "bwd_input", "bwd_weight")
ALLOWED_WARPS = {1, 2, 4, 8, 16, 32}


def parse_int_csv(text: str, option_name: str) -> list[int]:
    """Parse a comma-separated integer list while preserving user order."""

    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option_name} must be a comma-separated integer list"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{option_name} cannot be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError(f"{option_name} contains duplicates")
    return values


def validate_candidates(warps: Iterable[int], blocks: Iterable[int]) -> None:
    invalid_warps = [value for value in warps if value not in ALLOWED_WARPS]
    if invalid_warps:
        raise ValueError(
            f"Unsupported WARPS_PER_BLOCK values {invalid_warps}; "
            f"allowed={sorted(ALLOWED_WARPS)}"
        )

    invalid_blocks = [
        value for value in blocks if value < 32 or value > 1024 or value % 32
    ]
    if invalid_blocks:
        raise ValueError(
            "BWDK_BLOCK values must be multiples of 32 in [32, 1024]; "
            f"invalid={invalid_blocks}"
        )


def stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        raise ValueError("Cannot summarize an empty measurement group")

    mean = float(np.mean(values))
    median = float(np.median(values))
    if n > 1:
        std = float(np.std(values, ddof=1))
        sem = std / math.sqrt(n)
        margin = 1.96 * sem
    else:
        std = 0.0
        sem = 0.0
        margin = 0.0

    return {
        "n": n,
        "mean_ms": mean,
        "median_ms": median,
        "std_ms": std,
        "sem_ms": sem,
        "ci95_low_ms": mean - margin,
        "ci95_high_ms": mean + margin,
        "min_ms": float(np.min(values)),
        "p05_ms": float(np.quantile(values, 0.05)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(np.max(values)),
    }


def case_seed(base_seed: int, case: ShapeCase) -> int:
    return int(base_seed + zlib.crc32(case.config_id.encode("utf-8")))


def make_inputs(
    case: ShapeCase,
    seed: int,
    batch_override: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    B = int(case.B if batch_override is None else batch_override)
    x = torch.randn(B, case.H, case.L, device="cuda", dtype=torch.float32)
    k = torch.randn(case.H, case.K, device="cuda", dtype=torch.float32)
    gy = torch.randn_like(x)
    return x, k, gy


def schedule_id(candidate_kind: str, candidate_value: int) -> str:
    prefix = "w" if candidate_kind == FWD_DX_KIND else "b"
    return f"{prefix}{candidate_value}"


def schedule_context(
    case: ShapeCase,
    trial_id: int,
    device_name: str,
    candidate_kind: str,
    candidate_value: int,
    runtime: AmpereWarpRuntime,
) -> dict[str, Any]:
    return {
        **asdict(case),
        "trial_id": trial_id,
        "device_name": device_name,
        "candidate_kind": candidate_kind,
        "candidate_value": candidate_value,
        "schedule_id": schedule_id(candidate_kind, candidate_value),
        "warps_per_block": runtime.warps_per_block,
        "bwdk_btile": runtime.bwdk_btile,
        "bwdk_block": runtime.bwdk_block,
    }


def validate_schedule(
    runtime: AmpereWarpRuntime,
    case: ShapeCase,
    seed: int,
    candidate_kind: str,
    candidate_value: int,
    trial_id: int,
    device_name: str,
) -> list[dict[str, Any]]:
    """Compare the relevant candidate paths with the exact PyTorch reference."""

    validation_B = min(case.B, 128)
    x, k, gy = make_inputs(case, seed, batch_override=validation_B)
    rows: list[dict[str, Any]] = []
    context = schedule_context(
        case,
        trial_id,
        device_name,
        candidate_kind,
        candidate_value,
        runtime,
    )

    KERNEL_EVENTS.enabled = False
    KERNEL_EVENTS.clear()

    if candidate_kind == FWD_DX_KIND:
        y = torch.empty_like(x)
        gx = torch.empty_like(x)
        runtime.launch_fwd(x, k, y)
        runtime.launch_bwd_input(gy, k, gx)

        x_ref = x.detach().clone().requires_grad_(True)
        y_ref = pytorch_depthwise_same(x_ref, k)
        gx_ref = torch.autograd.grad(
            y_ref,
            x_ref,
            grad_outputs=gy,
            retain_graph=False,
            create_graph=False,
        )[0]
        torch.cuda.synchronize()

        for phase, actual, expected in (
            ("fwd", y, y_ref),
            ("bwd_input", gx, gx_ref),
        ):
            rows.append(
                {
                    **context,
                    "validation_B": validation_B,
                    "phase": phase,
                    "max_abs": max_abs_diff(actual, expected),
                    "normalized_max_abs": normalized_max_abs(actual, expected),
                    "allclose": bool(
                        torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)
                    ),
                }
            )
        del y, gx, x_ref, y_ref, gx_ref
    else:
        gk = torch.zeros_like(k)
        runtime.launch_bwd_weight(gy, x, gk, case.K)

        k_ref = k.detach().clone().requires_grad_(True)
        y_ref = pytorch_depthwise_same(x, k_ref)
        gk_ref = torch.autograd.grad(
            y_ref,
            k_ref,
            grad_outputs=gy,
            retain_graph=False,
            create_graph=False,
        )[0]
        torch.cuda.synchronize()
        rows.append(
            {
                **context,
                "validation_B": validation_B,
                "phase": "bwd_weight",
                "max_abs": max_abs_diff(gk, gk_ref),
                "normalized_max_abs": normalized_max_abs(gk, gk_ref),
                "allclose": bool(
                    torch.allclose(gk, gk_ref, atol=1e-2, rtol=5e-4)
                ),
            }
        )
        del gk, k_ref, y_ref, gk_ref

    del x, k, gy
    torch.cuda.empty_cache()
    return rows


def warmup_schedule(
    runtime: AmpereWarpRuntime,
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    candidate_kind: str,
    warmup: int,
) -> None:
    KERNEL_EVENTS.enabled = False
    KERNEL_EVENTS.clear()

    if candidate_kind == FWD_DX_KIND:
        y = torch.empty_like(x)
        gx = torch.empty_like(x)
        for _ in range(warmup):
            runtime.launch_fwd(x, k, y)
            runtime.launch_bwd_input(gy, k, gx)
        torch.cuda.synchronize()
        del y, gx
    else:
        gk = torch.zeros_like(k)
        for _ in range(warmup):
            gk.zero_()
            runtime.launch_bwd_weight(gy, x, gk, int(k.shape[1]))
        torch.cuda.synchronize()
        del gk


def measure_schedule(
    runtime: AmpereWarpRuntime,
    case: ShapeCase,
    x: torch.Tensor,
    k: torch.Tensor,
    gy: torch.Tensor,
    candidate_kind: str,
    candidate_value: int,
    runs: int,
    trial_id: int,
    device_name: str,
) -> pd.DataFrame:
    collector = KernelEventCollector()
    collector.enabled = True
    common = schedule_context(
        case,
        trial_id,
        device_name,
        candidate_kind,
        candidate_value,
        runtime,
    )

    if candidate_kind == FWD_DX_KIND:
        y = torch.empty_like(x)
        gx = torch.empty_like(x)
        for run_id in range(1, runs + 1):
            collector.set_context(**common, run_id_requested=run_id)
            runtime.launch_fwd(x, k, y, collector)
        for run_id in range(1, runs + 1):
            collector.set_context(**common, run_id_requested=run_id)
            runtime.launch_bwd_input(gy, k, gx, collector)
        raw = collector.resolve(synchronize=True)
        del y, gx
    else:
        gk = torch.zeros_like(k)
        for run_id in range(1, runs + 1):
            gk.zero_()
            collector.set_context(**common, run_id_requested=run_id)
            runtime.launch_bwd_weight(gy, x, gk, int(k.shape[1]), collector)
        raw = collector.resolve(synchronize=True)
        del gk

    return raw


SUMMARY_GROUP_COLUMNS = [
    "trial_id",
    "config_id",
    "sweep_group",
    "B",
    "H",
    "L",
    "K",
    "device_name",
    "candidate_kind",
    "candidate_value",
    "schedule_id",
    "warps_per_block",
    "bwdk_btile",
    "bwdk_block",
    "phase",
]


def summarize_raw(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in raw.groupby(SUMMARY_GROUP_COLUMNS, sort=False):
        row = dict(zip(SUMMARY_GROUP_COLUMNS, keys, strict=True))
        row.update(stats(group["runtime_ms"].to_numpy(dtype=float)))
        rows.append(row)
    result = pd.DataFrame(rows)
    order = {phase: index for index, phase in enumerate(PHASES)}
    result["_phase_order"] = result["phase"].map(order)
    result = result.sort_values(
        ["config_id", "_phase_order", "candidate_value"], kind="stable"
    ).drop(columns="_phase_order")
    return result.reset_index(drop=True)


def oracle_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Select the fastest candidate independently for every shape and path."""

    group_columns = [
        "config_id",
        "sweep_group",
        "B",
        "H",
        "L",
        "K",
        "device_name",
        "phase",
        "candidate_kind",
    ]
    rows: list[dict[str, Any]] = []

    for keys, group in summary.groupby(group_columns, sort=False):
        ordered = group.sort_values("mean_ms", kind="stable").reset_index(drop=True)
        best = ordered.iloc[0]
        second = ordered.iloc[1] if len(ordered) > 1 else best
        default_value = (
            DEFAULT_WARPS_PER_BLOCK
            if best["candidate_kind"] == FWD_DX_KIND
            else DEFAULT_BWDK_BLOCK
        )
        default_rows = ordered[ordered["candidate_value"] == default_value]
        default_ms = (
            float(default_rows.iloc[0]["mean_ms"])
            if not default_rows.empty
            else float("nan")
        )
        best_ms = float(best["mean_ms"])
        second_ms = float(second["mean_ms"])
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "winner_schedule_id": best["schedule_id"],
                "winner_value": int(best["candidate_value"]),
                "winner_mean_ms": best_ms,
                "winner_within_trial_std_ms": float(
                    best.get("std_ms", float("nan"))
                ),
                "winner_between_trial_std_ms": float(
                    best.get("between_trial_std_ms", float("nan"))
                ),
                "second_schedule_id": second["schedule_id"],
                "second_value": int(second["candidate_value"]),
                "second_mean_ms": second_ms,
                "winner_margin_vs_second_pct": (
                    (second_ms / best_ms - 1.0) * 100.0
                    if best_ms > 0.0
                    else float("nan")
                ),
                "paper_default_value": default_value,
                "paper_default_mean_ms": default_ms,
                "paper_default_regret_pct": (
                    (default_ms / best_ms - 1.0) * 100.0
                    if best_ms > 0.0 and np.isfinite(default_ms)
                    else float("nan")
                ),
            }
        )
        rows.append(row)

    result = pd.DataFrame(rows)
    order = {phase: index for index, phase in enumerate(PHASES)}
    result["_phase_order"] = result["phase"].map(order)
    result = result.sort_values(
        ["config_id", "_phase_order"], kind="stable"
    ).drop(columns="_phase_order")
    return result.reset_index(drop=True)


def combined_oracle(oracle: pd.DataFrame) -> pd.DataFrame:
    index_columns = [
        "config_id",
        "sweep_group",
        "B",
        "H",
        "L",
        "K",
        "device_name",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in oracle.groupby(index_columns, sort=False):
        phases = set(group["phase"])
        if phases != set(PHASES):
            raise RuntimeError(
                f"Missing path result for {keys}: expected={PHASES}, found={phases}"
            )
        phase_rows = {row.phase: row for row in group.itertuples(index=False)}
        oracle_total = float(group["winner_mean_ms"].sum())
        default_values = group["paper_default_mean_ms"].to_numpy(dtype=float)
        default_total = (
            float(np.sum(default_values))
            if np.isfinite(default_values).all()
            else float("nan")
        )
        row = dict(zip(index_columns, keys, strict=True))
        row.update(
            {
                "fwd_winner_warps": int(phase_rows["fwd"].winner_value),
                "bwd_input_winner_warps": int(
                    phase_rows["bwd_input"].winner_value
                ),
                "bwd_weight_winner_block": int(
                    phase_rows["bwd_weight"].winner_value
                ),
                "oracle_total_ms": oracle_total,
                "paper_default_total_ms": default_total,
                "paper_default_total_regret_pct": (
                    (default_total / oracle_total - 1.0) * 100.0
                    if oracle_total > 0.0 and np.isfinite(default_total)
                    else float("nan")
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("config_id", kind="stable").reset_index(
        drop=True
    )


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def trial_dir(output_root: Path, trial_id: int) -> Path:
    return output_root / f"trial{trial_id}"


def candidate_specs(args: argparse.Namespace) -> list[tuple[str, int]]:
    specs = [(FWD_DX_KIND, value) for value in args.warps_candidates]
    specs.extend((BWDK_KIND, value) for value in args.bwdk_block_candidates)
    if args.trial_id % 2 == 0:
        specs.reverse()
    return specs


def run_trial(args: argparse.Namespace) -> None:
    if args.trial_id <= 0:
        raise ValueError("--trial-id must be positive")
    if args.warmup < 1 or args.runs < 2:
        raise ValueError("Use at least one warm-up and two measured runs")
    validate_candidates(args.warps_candidates, args.bwdk_block_candidates)

    canonical_cases = cases_for_preset(args.preset)
    cases = canonical_cases if args.trial_id % 2 else list(reversed(canonical_cases))
    output_root = Path(args.output_dir)
    output_dir = trial_dir(output_root, args.trial_id)
    known_outputs = [
        output_dir / "raw.csv",
        output_dir / "trial_summary.csv",
        output_dir / "trial_oracle.csv",
    ]
    if not args.overwrite and any(path.exists() for path in known_outputs):
        raise FileExistsError(
            f"Trial output already exists in {output_dir}; "
            "use --overwrite intentionally"
        )

    torch.cuda.set_device(args.device)
    configure_torch()
    torch.backends.cudnn.deterministic = False
    KERNEL_EVENTS.enabled = False
    KERNEL_EVENTS.clear()

    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = torch.cuda.get_device_name(args.device)
    props = torch.cuda.get_device_properties(args.device)

    pd.DataFrame([asdict(case) for case in canonical_cases]).to_csv(
        output_dir / "shape_manifest.csv", index=False
    )
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": device_name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "total_memory_bytes": props.total_memory,
        "preset": args.preset,
        "trial_id": args.trial_id,
        "warmup": args.warmup,
        "runs": args.runs,
        "seed": args.seed,
        "warps_candidates": args.warps_candidates,
        "bwdk_block_candidates": args.bwdk_block_candidates,
        "bwdk_btile": DEFAULT_BWDK_BTILE,
        "fast_math": False,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "factorization": {
            "fwd_bwd_input_axis": "WARPS_PER_BLOCK",
            "bwd_weight_axis": "BWDK_BLOCK",
            "reason": "The compile-time axes affect disjoint kernel paths.",
        },
        "measurement_contract": {
            "strict_fp32": True,
            "cuda_event_kernel_timing": True,
            "allocation_timed": False,
            "compilation_timed": False,
            "validation_timed": False,
            "bwd_weight_zero_fill_timed": False,
            "candidate_order": "listed on odd trials; fully reversed on even trials",
            "shape_order": "listed on odd trials; fully reversed on even trials",
        },
    }
    save_json(output_dir / "metadata.json", metadata)

    raw_frames: list[pd.DataFrame] = []
    correctness_rows: list[dict[str, Any]] = []
    resource_frames: list[pd.DataFrame] = []

    print("=" * 100)
    print("PATHTILE FACTORIZED CANDIDATE SWEEP")
    print("=" * 100)
    print(f"GPU:        {device_name}")
    print(f"Preset:     {args.preset}")
    print(f"Trial:      {args.trial_id}")
    print(f"Protocol:   warmup={args.warmup}, runs={args.runs}, strict FP32")
    print(f"W candidates:     {args.warps_candidates}")
    print(f"BWDK candidates:  {args.bwdk_block_candidates}")
    print(f"Shapes:            {len(cases)}")

    try:
        specs = candidate_specs(args)
        for spec_index, (kind, value) in enumerate(specs, start=1):
            runtime = AmpereWarpRuntime(
                warps_per_block=(
                    value if kind == FWD_DX_KIND else DEFAULT_WARPS_PER_BLOCK
                ),
                bwdk_btile=DEFAULT_BWDK_BTILE,
                bwdk_block=(value if kind == BWDK_KIND else DEFAULT_BWDK_BLOCK),
                fast_math=False,
            )
            sid = schedule_id(kind, value)
            relevant_phases = (
                ("fwd", "bwd_input") if kind == FWD_DX_KIND else ("bwd_weight",)
            )
            print("\n" + "-" * 100)
            print(
                f"Candidate {spec_index}/{len(specs)}: {sid} "
                f"({kind}={value}), phases={relevant_phases}"
            )

            resources = runtime.resource_report()
            resources = resources[resources["kernel"].isin(relevant_phases)].copy()
            resources["trial_id"] = args.trial_id
            resources["candidate_kind"] = kind
            resources["candidate_value"] = value
            resources["schedule_id"] = sid
            resources["warps_per_block"] = runtime.warps_per_block
            resources["bwdk_btile"] = runtime.bwdk_btile
            resources["bwdk_block"] = runtime.bwdk_block
            resource_frames.append(resources)

            try:
                for case_index, case in enumerate(cases, start=1):
                    seed = case_seed(args.seed, case)
                    print(
                        f"[{case_index}/{len(cases)}] {case.config_id}: "
                        f"B={case.B}, H={case.H}, L={case.L}, K={case.K}"
                    )

                    if not args.skip_correctness:
                        candidate_correctness = validate_schedule(
                            runtime,
                            case,
                            seed,
                            kind,
                            value,
                            args.trial_id,
                            device_name,
                        )
                        correctness_rows.extend(candidate_correctness)
                        for row in candidate_correctness:
                            print(
                                f"  correctness {row['phase']}: "
                                f"max_abs={row['max_abs']:.3e}, "
                                f"ok={row['allclose']}"
                            )
                        if not all(row["allclose"] for row in candidate_correctness):
                            raise AssertionError(
                                f"Correctness failed for {case.config_id}/{sid}"
                            )

                    x, k, gy = make_inputs(case, seed)
                    warmup_schedule(runtime, x, k, gy, kind, args.warmup)
                    frame = measure_schedule(
                        runtime,
                        case,
                        x,
                        k,
                        gy,
                        kind,
                        value,
                        args.runs,
                        args.trial_id,
                        device_name,
                    )
                    raw_frames.append(frame)

                    interim_raw = pd.concat(raw_frames, ignore_index=True)
                    interim_raw.to_csv(output_dir / "raw.csv", index=False)
                    measured = summarize_raw(frame)
                    print(
                        measured[
                            ["phase", "mean_ms", "median_ms", "std_ms"]
                        ].to_string(index=False)
                    )

                    del x, k, gy, frame, interim_raw, measured
                    gc.collect()
                    torch.cuda.empty_cache()
            finally:
                runtime.close()

        raw = pd.concat(raw_frames, ignore_index=True)
        summary = summarize_raw(raw)
        oracle = oracle_from_summary(summary)
        combined = combined_oracle(oracle)

        raw.to_csv(output_dir / "raw.csv", index=False)
        summary.to_csv(output_dir / "trial_summary.csv", index=False)
        oracle.to_csv(output_dir / "trial_oracle.csv", index=False)
        combined.to_csv(output_dir / "trial_combined_oracle.csv", index=False)
        pd.concat(resource_frames, ignore_index=True).to_csv(
            output_dir / "kernel_resources_occupancy.csv", index=False
        )
        if correctness_rows:
            pd.DataFrame(correctness_rows).to_csv(
                output_dir / "correctness.csv", index=False
            )

        print("\n" + "=" * 100)
        print("EMPIRICAL PER-PATH ORACLE")
        print("=" * 100)
        print(
            oracle[
                [
                    "config_id",
                    "phase",
                    "winner_schedule_id",
                    "winner_mean_ms",
                    "winner_margin_vs_second_pct",
                    "paper_default_regret_pct",
                ]
            ].to_string(index=False)
        )
        print("\nCOMBINED ORACLE")
        print(combined.to_string(index=False))
        print("\nSaved below:", output_dir)
    finally:
        KERNEL_EVENTS.enabled = False


def aggregate_trials(args: argparse.Namespace) -> None:
    output_root = Path(args.output_dir)
    paths = sorted(output_root.glob("trial*/trial_summary.csv"))
    if not paths:
        raise FileNotFoundError(f"No trial summaries found below {output_root}")

    combined_trials = pd.concat(
        [pd.read_csv(path) for path in paths], ignore_index=True
    )
    trial_ids = sorted(int(value) for value in combined_trials["trial_id"].unique())
    if len(trial_ids) != args.expected_trials:
        raise RuntimeError(
            f"Expected {args.expected_trials} trials, found trial IDs {trial_ids}"
        )

    completeness_columns = [
        "config_id",
        "phase",
        "candidate_kind",
        "candidate_value",
    ]
    completeness = combined_trials.groupby(completeness_columns, sort=False)[
        "trial_id"
    ].nunique()
    incomplete = completeness[completeness != args.expected_trials]
    if not incomplete.empty:
        raise RuntimeError(
            "At least one shape/path/candidate is missing a trial:\n"
            f"{incomplete.to_string()}"
        )

    group_columns = [
        column for column in SUMMARY_GROUP_COLUMNS if column != "trial_id"
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in combined_trials.groupby(group_columns, sort=False):
        trial_means = group["mean_ms"].to_numpy(dtype=float)
        row = dict(zip(group_columns, keys, strict=True))
        row.update(
            {
                "trials": int(group["trial_id"].nunique()),
                "mean_ms": float(np.mean(trial_means)),
                "between_trial_std_ms": (
                    float(np.std(trial_means, ddof=1))
                    if trial_means.size > 1
                    else 0.0
                ),
                "min_trial_mean_ms": float(np.min(trial_means)),
                "max_trial_mean_ms": float(np.max(trial_means)),
                "mean_within_trial_std_ms": float(np.mean(group["std_ms"])),
            }
        )
        rows.append(row)

    aggregate = pd.DataFrame(rows)
    oracle = oracle_from_summary(aggregate)
    combined = combined_oracle(oracle)
    winner_frequency = (
        oracle.groupby(
            ["phase", "candidate_kind", "winner_value"], as_index=False
        )
        .size()
        .rename(columns={"size": "shapes_won"})
        .sort_values(["phase", "shapes_won"], ascending=[True, False])
    )

    combined_trials.to_csv(output_root / "all_trial_summaries.csv", index=False)
    aggregate.to_csv(output_root / "aggregate_candidate_summary.csv", index=False)
    oracle.to_csv(output_root / "aggregate_oracle.csv", index=False)
    combined.to_csv(output_root / "aggregate_combined_oracle.csv", index=False)
    winner_frequency.to_csv(output_root / "winner_frequency.csv", index=False)

    print("=" * 100)
    print("AGGREGATED PATHTILE CANDIDATE SWEEP")
    print("=" * 100)
    print(f"Trials: {trial_ids}")
    print("\nPER-PATH ORACLE")
    print(
        oracle[
            [
                "config_id",
                "phase",
                "winner_schedule_id",
                "winner_mean_ms",
                "winner_margin_vs_second_pct",
                "paper_default_regret_pct",
            ]
        ].to_string(index=False)
    )
    print("\nWINNER FREQUENCY")
    print(winner_frequency.to_string(index=False))
    print("\nSaved below:", output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["run", "aggregate"], default="run")
    parser.add_argument(
        "--preset",
        choices=[
            "smoke",
            "full",
            "cross-arch",
            "pathtile-train",
            "pathtile-holdout",
        ],
        default="smoke",
    )
    parser.add_argument(
        "--output-dir",
        default="results/pathtile_candidates",
        help="Root containing trial1, trial2, ... subdirectories.",
    )
    parser.add_argument("--trial-id", type=int, default=1)
    parser.add_argument("--expected-trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--warps-candidates",
        default="4,8,16",
        help="Comma-separated WARPS_PER_BLOCK candidates.",
    )
    parser.add_argument(
        "--bwdk-block-candidates",
        default="96,128,192",
        help="Comma-separated BWDK_BLOCK candidates.",
    )
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.warps_candidates = parse_int_csv(
        args.warps_candidates, "--warps-candidates"
    )
    args.bwdk_block_candidates = parse_int_csv(
        args.bwdk_block_candidates, "--bwdk-block-candidates"
    )
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "run":
        run_trial(args)
    else:
        aggregate_trials(args)


if __name__ == "__main__":
    main()
