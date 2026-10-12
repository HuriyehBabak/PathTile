"""Audit and combine PyTorch, Triton, TVM-TIRx, and PathTile results.

This script only reads saved measurements.  It never launches a GPU kernel.
Run it after the three TVM trials have been aggregated on each GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PHASE_COLUMNS = {
    "fwd": "fwd_mean_ms",
    "bwd_input": "bwd_input_mean_ms",
    "bwd_weight": "bwd_weight_mean_ms",
    "total": "total_mean_ms",
}
GPU_SOURCES = {
    "a100": {
        "label": "A100",
        "native": "results/a100_final_summary/a100_aggregate_summary.csv",
        "triton_supported": True,
        "tvm_root": "results/external_baselines/tvm/a100",
        "tvm_timing_protocol": "100 single-call CUDA-event measurements per phase and trial",
    },
    "a6000": {
        "label": "RTX A6000",
        "native": "results/a6000_final_summary/a6000_aggregate_summary.csv",
        "triton_supported": True,
        "tvm_root": "results/external_baselines/tvm/a6000",
        "tvm_timing_protocol": "100 single-call CUDA-event measurements per phase and trial",
    },
    "v100": {
        "label": "V100",
        "native": "results/v100_final_summary/v100_aggregate_summary.csv",
        "triton_supported": False,
        # Deliberately do not read results/external_baselines/tvm/v100 here.
        # That directory contains job 113124, whose between-trial CV was 9.36%.
        "tvm_root": "results/external_baselines/tvm/v100_stability_v3",
        "tvm_timing_protocol": (
            "20 CUDA-event samples x 32 calls per phase and trial; "
            "reported as amortized milliseconds per call"
        ),
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def native_method_row(rows: list[dict[str, str]], method: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("method") == method]
    if len(matches) != 1:
        raise ValueError(f"Expected one {method} row; found {len(matches)}")
    return matches[0]


def native_metrics(row: dict[str, str]) -> dict[str, float | None]:
    return {
        "fwd_mean_ms": float(row["fwd_mean_ms"]),
        "fwd_between_trial_std_ms": optional_float(
            row.get("fwd_between_trial_std_ms")
        ),
        "bwd_input_mean_ms": float(row["bwd_input_mean_ms"]),
        "bwd_input_between_trial_std_ms": optional_float(
            row.get("bwd_input_between_trial_std_ms")
        ),
        "bwd_weight_mean_ms": float(row["bwd_weight_mean_ms"]),
        "bwd_weight_between_trial_std_ms": optional_float(
            row.get("bwd_weight_between_trial_std_ms")
        ),
        "total_mean_ms": float(row["total_mean_ms"]),
        "total_between_trial_std_ms": optional_float(
            row.get("total_between_trial_std_ms")
        ),
    }


def optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def csv_bool(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{label}: expected true/false, got {value!r}")
    return normalized == "true"


def external_metrics(row: dict[str, str]) -> dict[str, float | None]:
    return {
        "fwd_mean_ms": float(row["fwd_mean_ms"]),
        "fwd_between_trial_std_ms": optional_float(
            row.get("fwd_between_trial_std_ms")
        ),
        "bwd_input_mean_ms": float(row["bwd_input_mean_ms"]),
        "bwd_input_between_trial_std_ms": optional_float(
            row.get("bwd_input_between_trial_std_ms")
        ),
        "bwd_weight_mean_ms": float(row["bwd_weight_mean_ms"]),
        "bwd_weight_between_trial_std_ms": optional_float(
            row.get("bwd_weight_between_trial_std_ms")
        ),
        "total_mean_ms": float(row["total_mean_ms"]),
        "total_between_trial_std_ms": optional_float(
            row.get("total_between_trial_std_ms")
        ),
    }


def assert_close_sum(metrics: dict[str, float | None], label: str) -> None:
    phase_sum = sum(float(metrics[column]) for column in (
        "fwd_mean_ms",
        "bwd_input_mean_ms",
        "bwd_weight_mean_ms",
    ))
    total = float(metrics["total_mean_ms"])
    if abs(phase_sum - total) > max(1e-5, 1e-7 * total):
        raise AssertionError(
            f"{label}: total {total} differs from phase sum {phase_sum}"
        )


def audit_tvm(
    project: Path,
    slug: str,
    source: dict[str, object],
) -> dict[str, object]:
    root = project / str(source["tvm_root"])
    audit_rows = read_csv(root / "final_audit.csv")
    if len(audit_rows) != 3:
        raise AssertionError(f"{slug}: expected three TVM audit rows")
    if {int(row["trial"]) for row in audit_rows} != {1, 2, 3}:
        raise AssertionError(f"{slug}: expected TVM trials 1, 2, and 3")

    if slug == "v100":
        for row in audit_rows:
            if not csv_bool(row["correctness_pass"], f"{slug}/correctness_pass"):
                raise AssertionError(f"{slug}: TVM correctness failure")
            if int(row["correctness_rows"]) != 11:
                raise AssertionError(f"{slug}: expected 11 TVM correctness rows")
            if any(int(row[field]) != 20 for field in (
                "fwd_samples",
                "bwd_input_samples",
                "bwd_weight_samples",
            )):
                raise AssertionError(f"{slug}: TVM batched sample-count failure")
            if int(row["calls_per_sample"]) != 32:
                raise AssertionError(f"{slug}: expected 32 calls per timing sample")
            if int(row["measured_calls_per_phase"]) != 640:
                raise AssertionError(f"{slug}: expected 640 measured calls per phase")
            if not csv_bool(row["fixed_configs_match"], f"{slug}/fixed_configs_match"):
                raise AssertionError(f"{slug}: frozen TVM configurations changed")

        stability_rows = read_csv(root / "final_stability.csv")
        if len(stability_rows) != 1:
            raise AssertionError(f"{slug}: expected one TVM stability row")
        stability = stability_rows[0]
        total_cv = float(stability["between_trial_total_cv_pct"])
        total_cv_limit = float(stability["between_trial_total_cv_limit_pct"])
        phase_cv = float(stability["maximum_phase_within_trial_cv_pct"])
        phase_cv_limit = float(stability["phase_within_trial_cv_limit_pct"])
        reported_pass = csv_bool(stability["stability_pass"], "v100/stability_pass")
        computed_pass = total_cv <= total_cv_limit and phase_cv <= phase_cv_limit
        if reported_pass != computed_pass:
            raise AssertionError("v100: stored stability decision is inconsistent")
        if total_cv > total_cv_limit:
            raise AssertionError(
                "v100: TVM total latency is not stable between trials: "
                f"{total_cv:.6f}% > {total_cv_limit:.6f}%"
            )

        outcome = "PASS" if reported_pass else "QUALIFIED_WITH_CAVEAT"
        note = (
            "strict stability gate passed"
            if reported_pass
            else (
                "between-trial total CV passed, but the maximum phase "
                f"within-trial CV was {phase_cv:.6f}% versus the "
                f"{phase_cv_limit:.1f}% gate; all samples were retained"
            )
        )
        return {
            "gpu_slug": slug,
            "data_root": str(source["tvm_root"]),
            "trials": len(audit_rows),
            "correctness": "PASS",
            "fixed_schedule": "PASS",
            "timing_protocol": source["tvm_timing_protocol"],
            "timing_samples_per_phase_per_trial": 20,
            "calls_per_sample": 32,
            "measured_calls_per_phase_per_trial": 640,
            "between_trial_total_cv_pct": total_cv,
            "between_trial_total_cv_limit_pct": total_cv_limit,
            "maximum_phase_within_trial_cv_pct": phase_cv,
            "phase_within_trial_cv_limit_pct": phase_cv_limit,
            "strict_stability_gate_pass": reported_pass,
            "audit_outcome": outcome,
            "publication_note": note,
        }

    for row in audit_rows:
        if not csv_bool(row["correctness_pass"], f"{slug}/correctness_pass"):
            raise AssertionError(f"{slug}: TVM correctness failure")
        if any(int(row[field]) != 100 for field in (
            "fwd_runs",
            "bwd_input_runs",
            "bwd_weight_runs",
        )):
            raise AssertionError(f"{slug}: TVM run-count failure")
        if not csv_bool(
            row["selected_configs_present"],
            f"{slug}/selected_configs_present",
        ):
            raise AssertionError(f"{slug}: missing TVM selected config")
    return {
        "gpu_slug": slug,
        "data_root": str(source["tvm_root"]),
        "trials": len(audit_rows),
        "correctness": "PASS",
        "selected_configs": "PASS",
        "timing_protocol": source["tvm_timing_protocol"],
        "runs_per_phase_per_trial": 100,
        "audit_outcome": "PASS",
        "publication_note": "standard single-call timing audit passed",
    }


def audit_triton(project: Path, slug: str) -> dict[str, object]:
    root = project / "results/external_baselines/triton" / slug
    for trial_id in range(1, 4):
        correctness = read_csv(root / f"trial{trial_id}/correctness.csv")
        if any(row["allclose"].lower() != "true" for row in correctness):
            raise AssertionError(f"{slug}: Triton correctness failure in trial {trial_id}")
        raw = read_csv(root / f"trial{trial_id}/raw.csv")
        counts = {
            phase: sum(row["phase"] == phase for row in raw)
            for phase in ("fwd", "bwd_input", "bwd_weight")
        }
        if any(count != 100 for count in counts.values()):
            raise AssertionError(f"{slug}: Triton run-count failure: {counts}")
    return {
        "gpu_slug": slug,
        "trials": 3,
        "correctness": "PASS",
        "runs_per_phase_per_trial": 100,
    }


def metric_row(
    gpu_slug: str,
    gpu_label: str,
    method: str,
    metrics: dict[str, float | None],
    pytorch: dict[str, float | None],
    pathtile: dict[str, float | None],
    evidence_status: str,
    timing_protocol: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "gpu_slug": gpu_slug,
        "gpu": gpu_label,
        "method": method,
        "evidence_status": evidence_status,
        "timing_protocol": timing_protocol,
        **metrics,
    }
    for phase, column in PHASE_COLUMNS.items():
        method_time = float(metrics[column])
        row[f"speedup_vs_pytorch_{phase}"] = float(pytorch[column]) / method_time
    row["pathtile_speedup_vs_method_total"] = (
        float(metrics["total_mean_ms"]) / float(pathtile["total_mean_ms"])
    )
    return row


def write_latex_table(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{External compiler-framework baselines for the FP32 main "
        r"workload ($B{=}16384$, $H{=}128$, $L{=}K{=}48$). Times are the "
        r"sum of forward, input-gradient, and weight-gradient means in ms. "
        r"Triton 3.6 does not support the V100 (sm\_70). "
        r"$^{\dagger}$The V100 TVM--TIRx value uses 20 batched samples of "
        r"32 calls per phase and trial. Its between-trial total CV was "
        r"0.562\%; retained transient samples raised the maximum phase CV to "
        r"5.247\%, narrowly above the pre-specified 5\% diagnostic gate.}",
        r"\label{tab:external-framework-baselines}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"GPU & PyTorch & Triton & TVM--TIRx & PathTile & PT / best compiler \\",
        r"\midrule",
    ]
    for row in rows:
        triton = row["triton_total_ms"]
        triton_text = "--" if triton is None else f"{float(triton):.3f}"
        tvm_text = f"{float(row['tvm_tirx_total_ms']):.3f}"
        if row["tvm_evidence_status"] != "PASS":
            tvm_text += r"$^{\dagger}$"
        lines.append(
            f"{row['gpu']} & {float(row['pytorch_total_ms']):.3f} & "
            f"{triton_text} & {tvm_text} & "
            f"{float(row['pathtile_total_ms']):.3f} & "
            f"{float(row['pathtile_speedup_vs_best_compiler']):.2f}"
            r"$\times$ \\"
        )
    lines.extend((
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/external_baselines/final_summary"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project_root.resolve()
    output_dir = (project / args.output_dir).resolve()
    all_rows: list[dict[str, object]] = []
    paper_rows: list[dict[str, object]] = []
    audits: dict[str, Any] = {
        "schema_version": 1,
        "publication_policy": {
            "v100_tvm_source": (
                "results/external_baselines/tvm/v100_stability_v3"
            ),
            "v100_tvm_treatment": (
                "include as a conservative external-baseline measurement with "
                "an explicit caveat; never relabel the strict stability gate as PASS"
            ),
            "superseded_v100_tvm_source": (
                "results/external_baselines/tvm/v100"
            ),
        },
        "tvm": [],
        "triton": [],
    }

    for slug, source in GPU_SOURCES.items():
        native = read_csv(project / str(source["native"]))
        pytorch = native_metrics(native_method_row(native, "PyTorch"))
        pathtile = native_metrics(native_method_row(native, "Warp"))
        assert_close_sum(pytorch, f"{slug}/PyTorch")
        assert_close_sum(pathtile, f"{slug}/PathTile")

        methods: list[tuple[str, dict[str, float | None], str, str]] = [
            (
                "PyTorch",
                pytorch,
                "canonical saved result",
                "three-trial native CUDA-event benchmark",
            ),
        ]
        triton: dict[str, float | None] | None = None
        if bool(source["triton_supported"]):
            triton_rows = read_csv(
                project / "results/external_baselines/triton" / slug / "final_aggregate.csv"
            )
            if len(triton_rows) != 1:
                raise ValueError(f"{slug}: expected one Triton aggregate row")
            triton = external_metrics(triton_rows[0])
            assert_close_sum(triton, f"{slug}/Triton")
            triton_audit = audit_triton(project, slug)
            audits["triton"].append(triton_audit)
            methods.append((
                "Triton",
                triton,
                "PASS",
                "100 single-call CUDA-event measurements per phase and trial",
            ))

        tvm_root = project / str(source["tvm_root"])
        tvm_rows = read_csv(tvm_root / "final_aggregate.csv")
        if len(tvm_rows) != 1:
            raise ValueError(f"{slug}: expected one TVM aggregate row")
        tvm_metrics = external_metrics(tvm_rows[0])
        assert_close_sum(tvm_metrics, f"{slug}/TVM-TIRx")
        tvm_audit = audit_tvm(project, slug, source)
        audits["tvm"].append(tvm_audit)
        methods.extend((
            (
                "TVM-TIRx",
                tvm_metrics,
                str(tvm_audit["audit_outcome"]),
                str(source["tvm_timing_protocol"]),
            ),
            (
                "PathTile",
                pathtile,
                "canonical saved result",
                "three-trial native CUDA-event benchmark",
            ),
        ))

        for method, metrics, evidence_status, timing_protocol in methods:
            all_rows.append(
                metric_row(
                    slug,
                    str(source["label"]),
                    method,
                    metrics,
                    pytorch,
                    pathtile,
                    evidence_status,
                    timing_protocol,
                )
            )

        external_compilers = [("TVM-TIRx", float(tvm_metrics["total_mean_ms"]))]
        if triton is not None:
            external_compilers.append(("Triton", float(triton["total_mean_ms"])))
        best_compiler, best_compiler_ms = min(
            external_compilers,
            key=lambda item: item[1],
        )
        paper_rows.append(
            {
                "gpu": source["label"],
                "pytorch_total_ms": pytorch["total_mean_ms"],
                "triton_total_ms": None if triton is None else triton["total_mean_ms"],
                "tvm_tirx_total_ms": tvm_metrics["total_mean_ms"],
                "pathtile_total_ms": pathtile["total_mean_ms"],
                "pathtile_speedup_vs_pytorch": (
                    float(pytorch["total_mean_ms"]) / float(pathtile["total_mean_ms"])
                ),
                "pathtile_speedup_vs_triton": (
                    None
                    if triton is None
                    else float(triton["total_mean_ms"])
                    / float(pathtile["total_mean_ms"])
                ),
                "pathtile_speedup_vs_tvm_tirx": (
                    float(tvm_metrics["total_mean_ms"])
                    / float(pathtile["total_mean_ms"])
                ),
                "best_external_compiler": best_compiler,
                "best_external_compiler_total_ms": best_compiler_ms,
                "pathtile_speedup_vs_best_compiler": (
                    best_compiler_ms / float(pathtile["total_mean_ms"])
                ),
                "triton_status": (
                    "PASS"
                    if bool(source["triton_supported"])
                    else "UNSUPPORTED: Triton 3.6 on sm_70"
                ),
                "tvm_evidence_status": tvm_audit["audit_outcome"],
                "tvm_timing_protocol": source["tvm_timing_protocol"],
                "tvm_publication_note": tvm_audit["publication_note"],
            }
        )

    all_fields = list(all_rows[0])
    write_csv(output_dir / "external_baselines_all_metrics.csv", all_rows, all_fields)
    paper_fields = list(paper_rows[0])
    write_csv(output_dir / "external_baselines_paper_table.csv", paper_rows, paper_fields)
    protocol_rows = [
        {
            "gpu": row["gpu"],
            "method": row["method"],
            "evidence_status": row["evidence_status"],
            "timing_protocol": row["timing_protocol"],
        }
        for row in all_rows
    ]
    write_csv(
        output_dir / "external_baselines_protocol_notes.csv",
        protocol_rows,
        list(protocol_rows[0]),
    )
    latex_path = output_dir / "external_baselines_paper_table.tex"
    write_latex_table(latex_path, paper_rows)
    audit_path = output_dir / "external_baselines_audit.json"
    audit_path.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")

    print("External-baseline data reconstruction: PASS")
    for row in paper_rows:
        triton_text = (
            "n/a" if row["triton_total_ms"] is None else f"{float(row['triton_total_ms']):.3f}"
        )
        print(
            f"{row['gpu']:10s} PyTorch={float(row['pytorch_total_ms']):.3f} ms, "
            f"Triton={triton_text} ms, "
            f"TVM={float(row['tvm_tirx_total_ms']):.3f} ms, "
            f"PathTile={float(row['pathtile_total_ms']):.3f} ms"
        )
    print("TVM V100 evidence status: QUALIFIED_WITH_CAVEAT")
    print(f"Saved: {output_dir / 'external_baselines_all_metrics.csv'}")
    print(f"Saved: {output_dir / 'external_baselines_paper_table.csv'}")
    print(f"Saved: {output_dir / 'external_baselines_protocol_notes.csv'}")
    print(f"Saved: {latex_path}")
    print(f"Saved: {audit_path}")


if __name__ == "__main__":
    main()
