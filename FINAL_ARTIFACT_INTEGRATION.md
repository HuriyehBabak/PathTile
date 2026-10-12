# Final artifact integration

This patch integrates the completed Triton and TVM-TIRx measurements into the
paper figures, tables, documentation, and CPU-only artifact verification path.
It launches no GPU work and does not alter any raw measurement.

## What changes

- `scripts.aggregate_external_baselines` becomes a required reconstruction
  stage.
- `paper/generate_final_figures.py` adds
  `fig_external_baselines.{pdf,png}` and
  `table_external_baselines.{csv,tex}`.
- `scripts.verify_artifact` checks all 11 supported GPU/framework rows, exact
  headline values, the selected V100 V3 data root, and preservation of
  `strict_stability_gate_pass=false`.
- The verifier expects 28 generated manifest entries, or 29 files including
  `paper/final_artifacts/artifact_manifest.csv`.
- The root and artifact documentation records the external baseline protocols
  and the V100 TVM qualification.

The V100 TVM-TIRx value remains `QUALIFIED_WITH_CAVEAT`: correctness and the
fixed schedule pass, total between-trial CV is 0.562%, and the maximum phase CV
is 5.247% against the pre-specified 5% diagnostic gate. No sample is removed.

## Install

Run on the Pascal head node:

```bash
cd "$HOME/s4convd-paper"

tar --no-same-owner -xzf \
  "$HOME/s4convd_final_artifact_integration_patch.tar.gz"

sha256sum -c \
  FINAL_ARTIFACT_INTEGRATION_MANIFEST.sha256
```

The old `artifact/archive_manifest.csv` is expected to become stale immediately
after extracting a code/documentation patch. Do not run the archive-integrity
check until the rebuild below has regenerated that manifest.

## Complete CPU-only rebuild and verification

```bash
PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python

"$PY" -m py_compile \
  paper/generate_final_figures.py \
  scripts/aggregate_external_baselines.py \
  scripts/rebuild_artifact.py \
  scripts/verify_artifact.py

"$PY" -m scripts.rebuild_artifact \
  --project-root .

"$PY" -m scripts.check_archive_manifest \
  --project-root .
```

The important terminal lines are:

```text
External-baseline data reconstruction: PASS
TVM V100 evidence status: QUALIFIED_WITH_CAVEAT
Final paper artifacts: PASS
Generated files: 29
Artifact verification: PASS
Paper artifacts checked: 29
Full CPU-only artifact rebuild: PASS (66 canonical outputs written)
Archive integrity: PASS (... files)
```

The final hashed-file count depends on the complete result evidence present in
the project, so the number shown by the last command is not hard-coded.

## Inspect the new outputs

```bash
sed -n '1,8p' \
  paper/final_artifacts/tables/table_external_baselines.csv

grep -n -E \
  'external_baselines|QUALIFIED_WITH_CAVEAT|0.562|5.246' \
  artifact/verification_report.md \
  results/external_baselines/final_summary/external_baselines_audit.json

find paper/final_artifacts \
  -type f \
  | sort
```

No additional GPU job is required after this integration succeeds.
