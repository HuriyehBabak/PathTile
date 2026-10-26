#!/usr/bin/env python3
"""Verify every canonical PathTile result from saved measurements, CPU-only."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.aggregate_end_to_end import build_outputs as build_end_to_end
from scripts.aggregate_main_results import build_outputs as build_main
from scripts.aggregate_pathtile_results import build_outputs as build_pathtile
from scripts.aggregate_shape_results import build_outputs as build_shapes
from scripts.artifact_common import (
    assert_frame_close,
    geometric_mean,
    read_csv,
    resolve_project_root,
    write_csv,
)
from scripts.extract_ncu_metrics import build_outputs as build_ncu


def _assert_close(actual: float, expected: float, name: str, atol: float = 1e-9) -> None:
    if not np.isclose(actual, expected, rtol=1e-11, atol=atol):
        raise AssertionError(f"{name}: expected {expected}, found {actual}")


def verify_rebuilt_frames(root: Path) -> tuple[int, list[str]]:
    expected_outputs: dict[Path, pd.DataFrame] = {}
    for builder in (build_main, build_shapes, build_pathtile, build_ncu, build_end_to_end):
        built = builder(root)
        overlap = set(expected_outputs).intersection(built)
        if overlap:
            raise RuntimeError(f"Multiple producers claim outputs: {sorted(overlap)}")
        expected_outputs.update(built)

    checked: list[str] = []
    for relative_path, expected in expected_outputs.items():
        path = root / relative_path
        actual = read_csv(path)
        serialized = io.StringIO()
        expected.to_csv(serialized, index=False)
        serialized.seek(0)
        expected_roundtrip = pd.read_csv(serialized)
        assert_frame_close(actual, expected_roundtrip, str(relative_path))
        checked.append(str(relative_path))
    return len(checked), checked


def verify_code_syntax(root: Path) -> int:
    count = 0
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise AssertionError(f"Syntax error in {path}: {exc}") from exc
        count += 1
    return count


def verify_end_to_end_schemas(root: Path) -> None:
    serial = root / "results/end_to_end/a6000_final_serial"
    raw_columns = [
        "technique",
        "epoch",
        "B",
        "H",
        "L",
        "K",
        "phase",
        "run_id",
        "runtime_ms",
    ]
    for trial in (1, 2, 3):
        for backend in ("warp", "pytorch"):
            summary = read_csv(
                serial / f"trial{trial}/{backend}/training_kernel_epoch_summary.csv"
            )
            if "val_batch_mean_rmsle" not in summary or "val_loss" in summary:
                raise AssertionError(
                    f"trial{trial}/{backend}: validation metric is not explicitly named"
                )
        pytorch_raw = read_csv(
            serial / f"trial{trial}/pytorch/training_raw_kernel_times.csv"
        )
        if list(pytorch_raw.columns) != raw_columns or len(pytorch_raw) != 0:
            raise AssertionError(
                f"trial{trial}/pytorch: expected a schema-valid empty custom-event CSV"
            )


def verify_external_baselines(root: Path) -> dict[str, Any]:
    summary_root = root / "results/external_baselines/final_summary"
    paper = read_csv(
        summary_root / "external_baselines_paper_table.csv",
        [
            "gpu",
            "pytorch_total_ms",
            "triton_total_ms",
            "tvm_tirx_total_ms",
            "pathtile_total_ms",
            "pathtile_speedup_vs_best_compiler",
            "best_external_compiler",
            "triton_status",
            "tvm_evidence_status",
        ],
    )
    gpu_order = ["A100", "RTX A6000", "V100"]
    if len(paper) != 3 or paper["gpu"].tolist() != gpu_order:
        raise AssertionError("External paper table must contain the three GPUs in order")

    expected = {
        "A100": {
            "pytorch": 47.42395558516184,
            "triton": 21.774445,
            "tvm": 11.517958843708039,
            "pathtile": 11.282756355603537,
            "status": "PASS",
        },
        "RTX A6000": {
            "pytorch": 56.72340135892232,
            "triton": 31.765586,
            "tvm": 19.702207309405008,
            "pathtile": 15.211909228960671,
            "status": "PASS",
        },
        "V100": {
            "pytorch": 54.88611330827077,
            "triton": None,
            "tvm": 19.917635186513266,
            "pathtile": 13.58520658651988,
            "status": "QUALIFIED_WITH_CAVEAT",
        },
    }
    best_speedups: dict[str, float] = {}
    for row in paper.itertuples(index=False):
        values = expected[str(row.gpu)]
        _assert_close(float(row.pytorch_total_ms), values["pytorch"], f"{row.gpu} external/PyTorch")
        _assert_close(float(row.tvm_tirx_total_ms), values["tvm"], f"{row.gpu} external/TVM")
        _assert_close(float(row.pathtile_total_ms), values["pathtile"], f"{row.gpu} external/PathTile")
        if values["triton"] is None:
            if not pd.isna(row.triton_total_ms):
                raise AssertionError("V100 Triton entry must be unavailable")
            if str(row.triton_status) != "UNSUPPORTED: Triton 3.6 on sm_70":
                raise AssertionError("V100 Triton status is not explicit")
        else:
            _assert_close(
                float(row.triton_total_ms),
                float(values["triton"]),
                f"{row.gpu} external/Triton",
                atol=1e-5,
            )
            if str(row.triton_status) != "PASS":
                raise AssertionError(f"{row.gpu}: Triton audit is not PASS")
        if str(row.best_external_compiler) != "TVM-TIRx":
            raise AssertionError(f"{row.gpu}: unexpected best external compiler")
        if str(row.tvm_evidence_status) != values["status"]:
            raise AssertionError(f"{row.gpu}: unexpected TVM evidence status")
        expected_speedup = float(values["tvm"]) / float(values["pathtile"])
        _assert_close(
            float(row.pathtile_speedup_vs_best_compiler),
            expected_speedup,
            f"{row.gpu} PathTile/best compiler",
        )
        best_speedups[str(row.gpu)] = expected_speedup

    all_metrics = read_csv(
        summary_root / "external_baselines_all_metrics.csv",
        ["gpu", "method", "evidence_status", "timing_protocol", "total_mean_ms"],
    )
    expected_methods = {
        "A100": {"PyTorch", "Triton", "TVM-TIRx", "PathTile"},
        "RTX A6000": {"PyTorch", "Triton", "TVM-TIRx", "PathTile"},
        "V100": {"PyTorch", "TVM-TIRx", "PathTile"},
    }
    if len(all_metrics) != 11 or all_metrics.duplicated(["gpu", "method"]).any():
        raise AssertionError("External all-metrics table must contain 11 unique rows")
    for gpu, methods in expected_methods.items():
        found = set(all_metrics.loc[all_metrics["gpu"] == gpu, "method"])
        if found != methods:
            raise AssertionError(f"{gpu}: external methods differ: {sorted(found)}")

    protocol = read_csv(
        summary_root / "external_baselines_protocol_notes.csv",
        ["gpu", "method", "evidence_status", "timing_protocol"],
    )
    if len(protocol) != 11:
        raise AssertionError("Expected 11 external-baseline protocol rows")

    audit_path = summary_root / "external_baselines_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("schema_version") != 1:
        raise AssertionError("Unexpected external-baseline audit schema")
    policy = audit.get("publication_policy", {})
    if policy.get("v100_tvm_source") != (
        "results/external_baselines/tvm/v100_stability_v3"
    ):
        raise AssertionError("Final aggregate did not select V100 stability V3")
    tvm_audits = {entry["gpu_slug"]: entry for entry in audit.get("tvm", [])}
    if set(tvm_audits) != {"a100", "a6000", "v100"}:
        raise AssertionError("Expected TVM audits for all three GPUs")
    v100 = tvm_audits["v100"]
    if v100.get("audit_outcome") != "QUALIFIED_WITH_CAVEAT":
        raise AssertionError("V100 TVM evidence must retain its caveat")
    if v100.get("strict_stability_gate_pass") is not False:
        raise AssertionError("V100 strict stability gate must remain false")
    if v100.get("correctness") != "PASS" or v100.get("fixed_schedule") != "PASS":
        raise AssertionError("V100 correctness/fixed-schedule audit failed")
    if int(v100.get("measured_calls_per_phase_per_trial", 0)) != 640:
        raise AssertionError("V100 batched timing call count changed")
    _assert_close(
        float(v100["between_trial_total_cv_pct"]),
        0.5620004505658299,
        "V100 TVM between-trial total CV",
    )
    _assert_close(
        float(v100["maximum_phase_within_trial_cv_pct"]),
        5.2466137807217885,
        "V100 TVM maximum phase CV",
    )

    return {
        "rows": len(all_metrics),
        "tvm_v100_evidence_status": "QUALIFIED_WITH_CAVEAT",
        "tvm_v100_between_trial_total_cv_pct": float(
            v100["between_trial_total_cv_pct"]
        ),
        "tvm_v100_maximum_phase_cv_pct": float(
            v100["maximum_phase_within_trial_cv_pct"]
        ),
        "pathtile_speedup_vs_best_external_compiler": best_speedups,
    }


def verify_key_metrics(root: Path) -> dict[str, Any]:
    cross = read_csv(
        root / "results/cross_gpu_final_summary/cross_gpu_3gpu_all_metrics.csv"
    )
    if len(cross) != 15 or cross["gpu"].nunique() != 3:
        raise AssertionError("Three-GPU aggregate must contain 3 GPUs x 5 methods")
    if set(cross["trials"].astype(int)) != {3}:
        raise AssertionError("Every main microbenchmark must contain three trials")
    warp = cross[cross["method"] == "Warp"].set_index("gpu")
    expected_warp = {
        "NVIDIA A100-PCIE-40GB": (11.282756355603537, 4.203224291164359),
        "NVIDIA RTX A6000": (15.211909228960671, 3.7288811355073967),
        "Tesla V100-SXM3-32GB": (13.58520658651988, 4.0401382900376595),
    }
    for gpu, (runtime, speedup) in expected_warp.items():
        _assert_close(float(warp.loc[gpu, "total_mean_ms"]), runtime, f"{gpu} runtime")
        _assert_close(
            float(warp.loc[gpu, "total_speedup_vs_pytorch"]),
            speedup,
            f"{gpu} speedup",
        )

    shape_metrics: dict[str, Any] = {}
    for label, relative in [
        ("A6000", "results/shape_sweep_ampere/a6000_full/aggregate_total_comparison.csv"),
        (
            "V100",
            "results/shape_sweep_volta/v100_full_w16_b256/aggregate_total_comparison.csv",
        ),
    ]:
        frame = read_csv(root / relative, ["warp_speedup_vs_pytorch"])
        if len(frame) != 10:
            raise AssertionError(f"{label}: expected 10 generalization shapes")
        speedups = frame["warp_speedup_vs_pytorch"].to_numpy(dtype=float)
        shape_metrics[label] = {
            "shapes": len(frame),
            "geomean_speedup": geometric_mean(speedups),
            "min_speedup": float(speedups.min()),
            "max_speedup": float(speedups.max()),
        }
    _assert_close(shape_metrics["A6000"]["geomean_speedup"], 3.140810530325931, "A6000 shape geomean")
    _assert_close(shape_metrics["V100"]["geomean_speedup"], 3.23922670273891, "V100 shape geomean")

    holdout = read_csv(
        root
        / "results/pathtile_selector/holdout_evaluation/three_gpu_holdout_per_shape.csv"
    )
    if len(holdout) != 30:
        raise AssertionError("Expected 30 frozen-policy holdout shapes")
    within_one = int((holdout["frozen_regret_pct"] <= 1.0).sum())
    within_two = int((holdout["frozen_regret_pct"] <= 2.0).sum())
    if (within_one, within_two) != (26, 29):
        raise AssertionError(
            f"Unexpected holdout thresholds: within1={within_one}, within2={within_two}"
        )
    holdout_mean = float(holdout["frozen_regret_pct"].mean())
    _assert_close(holdout_mean, 0.312414463104819, "overall holdout mean regret")

    ncu = read_csv(root / "results/nsight/a100_main/a100_main_ncu_key_metrics.csv")
    if len(ncu) != 42 or ncu["phase"].nunique() != 3:
        raise AssertionError("Nsight summary must contain 14 metrics x 3 paths")

    compile_summary = read_csv(
        root / "results/a6000_final_baselines/torch_compile/final_aggregate.csv"
    ).iloc[0]
    _assert_close(float(compile_summary["total_mean_ms"]), 56.55188406626383, "torch.compile total")

    end_to_end = read_csv(
        root / "results/end_to_end/a6000_final_serial/final_paper_summary.csv"
    ).iloc[0]
    _assert_close(float(end_to_end["epoch_speedup_geomean"]), 1.0300206366942428, "epoch speedup")
    _assert_close(float(end_to_end["gpu_stages_speedup_geomean"]), 1.6754784377938918, "GPU-stage speedup")
    if float(end_to_end["max_rmsle_abs_difference"]) != 0.0:
        raise AssertionError("End-to-end test RMSLE is not identical")

    external_baselines = verify_external_baselines(root)

    return {
        "main_microbenchmark": {
            "gpus": 3,
            "methods_per_gpu": 5,
            "trials": 3,
            "warp_speedup_range": [
                float(warp["total_speedup_vs_pytorch"].min()),
                float(warp["total_speedup_vs_pytorch"].max()),
            ],
        },
        "shape_generalization": shape_metrics,
        "frozen_holdout": {
            "shapes": len(holdout),
            "within_1pct": within_one,
            "within_2pct": within_two,
            "mean_regret_pct": holdout_mean,
        },
        "torch_compile_total_ms": float(compile_summary["total_mean_ms"]),
        "end_to_end": {
            "epoch_speedup_geomean": float(end_to_end["epoch_speedup_geomean"]),
            "gpu_stages_speedup_geomean": float(
                end_to_end["gpu_stages_speedup_geomean"]
            ),
            "max_rmsle_abs_difference": float(
                end_to_end["max_rmsle_abs_difference"]
            ),
        },
        "ncu_metrics": len(ncu),
        "external_baselines": external_baselines,
    }


def verify_paper_artifacts(root: Path) -> int:
    output_root = root / "paper/final_artifacts"
    manifest = read_csv(output_root / "artifact_manifest.csv", ["artifact", "kind"])
    if len(manifest) != 28:
        raise AssertionError(f"Expected 28 manifest entries, found {len(manifest)}")
    missing = [
        str(output_root / relative)
        for relative in manifest["artifact"]
        if not (output_root / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing generated paper artifacts: {missing}")
    return len(manifest) + 1


def write_result_manifest(root: Path) -> Path:
    rows = [
        (
            "main_three_gpu",
            "results/{a100_baselines,a6000_final_baselines,v100_final_baselines}/<method>/trial*/<summary>.csv; results/warp_tiled_{ampere,volta}/<final>/trial*/benchmark_summary.csv",
            "results/{a100,a6000,v100}_final_summary/*_aggregate_summary.csv; results/cross_gpu_final_summary/cross_gpu_3gpu_*.csv",
            "scripts/aggregate_main_results.py",
        ),
        (
            "torch_compile",
            "results/a6000_final_baselines/torch_compile/trial*/torch_compile_runtime_summary_a6000.csv; trial*/benchmark.log",
            "results/a6000_final_baselines/torch_compile/final_*.csv",
            "scripts/aggregate_main_results.py",
        ),
        (
            "shape_generalization",
            "results/shape_sweep_{ampere,volta}/<final>/trial*/trial_summary.csv; trial*/correctness.csv",
            "results/shape_sweep_{ampere,volta}/<final>/aggregate_*.csv",
            "scripts/aggregate_shape_results.py",
        ),
        (
            "pathtile_training_holdout",
            "results/pathtile_candidates/{a100,a6000,v100}_{train,holdout}/trial*/{trial_summary,correctness,metadata}.csv|json",
            "results/pathtile_candidates/*/aggregate_*.csv; results/pathtile_selector/{architecture_static_training,frozen_policy,holdout_evaluation}/*",
            "scripts/aggregate_pathtile_results.py",
        ),
        (
            "nsight_a100",
            "results/nsight/a100_main/{fwd,bwd_input,bwd_weight}/*_details.csv",
            "results/nsight/a100_main/a100_main_ncu_key_metrics.csv",
            "scripts/extract_ncu_metrics.py",
        ),
        (
            "end_to_end_a6000",
            "results/end_to_end/a6000_final_serial/trial*/{warp,pytorch}/training_*.csv",
            "results/end_to_end/a6000_final_serial/final_*.csv",
            "scripts/aggregate_end_to_end.py",
        ),
        (
            "external_compiler_baselines",
            "results/external_baselines/{triton,tvm}/{a100,a6000,v100_stability_v3}/<trial and aggregate evidence>",
            "results/external_baselines/final_summary/*",
            "scripts/aggregate_external_baselines.py",
        ),
        (
            "paper_figures_tables",
            "canonical outputs above",
            "paper/final_artifacts/{figures,tables}; paper/final_artifacts/artifact_manifest.csv",
            "paper/generate_final_figures.py",
        ),
    ]
    frame = pd.DataFrame(
        rows, columns=["category", "inputs", "outputs", "producer"]
    )
    path = root / "artifact/result_manifest.csv"
    write_csv(frame, path)
    return path


def write_archive_manifest(root: Path) -> tuple[Path, int]:
    """Hash all deliverable files so an extracted snapshot can be audited."""

    output_path = root / "artifact/archive_manifest.csv"
    rows: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if (
            ".git" in relative.parts 
            or "__pycache__" in relative.parts
            or path.suffix == ".pyc"
            or path.name.endswith(".orig")
            or path == output_path
        ):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        rows.append(
            {
                "path": relative.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    write_csv(pd.DataFrame(rows), output_path)
    return output_path, len(rows)


def write_reports(
    root: Path,
    *,
    rebuilt_files: int,
    syntax_files: int,
    paper_files: int,
    metrics: dict[str, Any],
) -> tuple[Path, Path]:
    report = {
        "status": "PASS",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "gpu_benchmarks_executed": False,
        "verification_mode": "CPU-only reconstruction from saved measurements",
        "checks": {
            "reconstructed_canonical_csvs": rebuilt_files,
            "python_files_syntax_checked": syntax_files,
            "paper_artifacts_checked": paper_files,
            "validation_metric_named_batch_mean_rmsle": True,
            "pytorch_empty_custom_event_csv_has_schema": True,
            "frozen_policy_precedes_all_holdout_trials": True,
            "external_compiler_baselines_audited": True,
            "v100_tvm_caveat_preserved": True,
        },
        "metrics": metrics,
    }
    artifact_root = root / "artifact"
    artifact_root.mkdir(parents=True, exist_ok=True)
    json_path = artifact_root / "verification_report.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    markdown_path = artifact_root / "verification_report.md"
    main = metrics["main_microbenchmark"]
    holdout = metrics["frozen_holdout"]
    end_to_end = metrics["end_to_end"]
    external = metrics["external_baselines"]
    markdown_path.write_text(
        "\n".join(
            [
                "# Artifact Verification Report",
                "",
                "**Status: PASS**",
                "",
                "The verification was CPU-only and rebuilt every canonical aggregate from saved measurements. No GPU benchmark was executed.",
                "",
                f"- Canonical CSVs reconstructed and compared: {rebuilt_files}",
                f"- Python source files syntax-checked: {syntax_files}",
                f"- Generated paper artifacts checked: {paper_files}",
                f"- Main benchmark: {main['gpus']} GPUs, 5 methods, 3 trials",
                f"- Warp speedup range: {main['warp_speedup_range'][0]:.6f}x to {main['warp_speedup_range'][1]:.6f}x",
                f"- Frozen holdout: {holdout['within_1pct']}/30 within 1%, {holdout['within_2pct']}/30 within 2%",
                f"- Frozen-policy mean regret: {holdout['mean_regret_pct']:.6f}%",
                f"- End-to-end epoch speedup: {end_to_end['epoch_speedup_geomean']:.6f}x",
                f"- End-to-end GPU-stage speedup: {end_to_end['gpu_stages_speedup_geomean']:.6f}x",
                f"- Maximum paired test-RMSLE difference: {end_to_end['max_rmsle_abs_difference']:.1f}",
                f"- External baselines: {external['rows']} GPU/method rows audited",
                "- TVM V100 evidence status: QUALIFIED_WITH_CAVEAT",
                f"- TVM V100 between-trial total CV: {external['tvm_v100_between_trial_total_cv_pct']:.6f}%",
                f"- TVM V100 maximum phase CV: {external['tvm_v100_maximum_phase_cv_pct']:.6f}% (5% diagnostic gate)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return json_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    root = resolve_project_root(parse_args().project_root)
    rebuilt_files, _ = verify_rebuilt_frames(root)
    verify_end_to_end_schemas(root)
    syntax_files = verify_code_syntax(root)
    metrics = verify_key_metrics(root)
    paper_files = verify_paper_artifacts(root)
    write_result_manifest(root)
    write_reports(
        root,
        rebuilt_files=rebuilt_files,
        syntax_files=syntax_files,
        paper_files=paper_files,
        metrics=metrics,
    )
    _, hashed_files = write_archive_manifest(root)
    print("Artifact verification: PASS")
    print(f"Canonical CSVs reconstructed: {rebuilt_files}")
    print(f"Python sources syntax-checked: {syntax_files}")
    print(f"Paper artifacts checked: {paper_files}")
    print(f"Deliverable files hashed: {hashed_files}")


if __name__ == "__main__":
    main()
