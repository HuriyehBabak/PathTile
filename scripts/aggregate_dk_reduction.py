"""Reconstruct dK reduction statistics from saved trial means."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

SOURCES = {
    "A100": "sm80_20261013T072826Z",
    "RTX A6000": "sm86_20261013T072356Z",
    "V100 SXM3": "sm70_20261013T074302Z",
}
MODES = (
    "atomic_kernel", "atomic_full", "atomic_zero_only",
    "noatomic", "noatomic_optimized",
)


def write_outputs(root: Path) -> list[Path]:
    rows = []
    hashes = {}

    for gpu, directory in SOURCES.items():
        path = root / "results/dk_fairness" / directory / "trial_summary.csv"
        hashes[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()

        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))

        if len(records) != 15:
            raise ValueError(f"Expected 15 trial/mode rows: {path}")

        for mode in MODES:
            selected = [r for r in records if r["mode"] == mode]
            if sorted(int(r["trial"]) for r in selected) != [1, 2, 3]:
                raise ValueError(f"Expected exactly trials 1,2,3: {gpu}/{mode}")

            values = [float(r["mean_ms"]) for r in selected]
            rows.append({
                "gpu": gpu,
                "mode": mode,
                "trials": len(values),
                "mean_ms": statistics.mean(values),
                "between_trial_std_ms": statistics.stdev(values),
            })

    out = root / "results/dk_fairness/paper_summary"
    out.mkdir(parents=True, exist_ok=True)
    summary = out / "reduction_summary.csv"
    audit = out / "reduction_provenance.json"

    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    audit.write_text(json.dumps({
        "source_sha256": hashes,
        "statistic": "Mean and sample SD of three trial-level means",
        "within_trial_std_ms_used": False,
        "new_gpu_measurements": False,
    }, indent=2) + "\n")

    for row in rows:
        print(
            f"{row['gpu']:10s} {row['mode']:20s} "
            f"{row['mean_ms']:.3f} +/- "
            f"{row['between_trial_std_ms']:.3f} ms"
        )
    return [summary, audit]


if __name__ == "__main__":
    for output in write_outputs(Path.cwd()):
        print(f"Saved: {output}")
