# Artifact scope and audit status

## What is complete

The repository documents saved code and evidence supporting the paper-facing results below. This Markdown-only documentation review has not independently executed or validated those sources:

1. Three-trial main-workload comparisons across A100, RTX A6000, and V100.
2. Five-stage ablation: PyTorch, naive, GMC, shared-memory, and warp-tiled.
3. Independent correctness evidence for forward, backward-input, and
   backward-weight.
4. A6000 and V100 ten-shape generalization sweeps.
5. Twenty training shapes plus ten frozen-policy holdout shapes on each GPU.
6. Three-trial A6000 `torch.compile` baseline.
7. A100 Nsight Compute metrics for all three computational paths.
8. Three paired, serial end-to-end S4ConvD/ASHRAE trials on A6000.
9. Three-trial Triton baselines on A100 and A6000, plus three-trial TVM-TIRx
   baselines on A100, A6000, and V100.
10. Paper-ready figures and tables generated only from canonical aggregates.

Each canonical aggregate now has a persistent CPU-only producer.  The mapping
from raw inputs through canonical outputs to producer programs is stored in
`result_manifest.csv` and is recreated by the verifier.
`archive_manifest.csv` records the byte size and SHA-256 digest of every
deliverable file (excluding caches and legacy `.orig` backups).

## Audit corrections

Two representation issues were corrected without changing any measured value:

- End-to-end validation is an unweighted mean of per-batch RMSLE values, so the
  field is explicitly named `val_batch_mean_rmsle`.  Test RMSLE remains a
  global RMSLE and must not be compared directly to that validation series.
- A PyTorch backend records no custom CUDA-kernel events.  Its raw custom-event
  CSVs are therefore schema-valid, header-only files rather than zero-byte
  files.

The V100 aggregate was also normalized to the same complete schema as A100 and
A6000, including the trial count and per-path between-trial standard deviations.
These are derived from the existing three saved trials.

## External-baseline qualification

`scripts/aggregate_external_baselines.py` explicitly selects
`results/external_baselines/tvm/v100_stability_v3` and cannot silently consume
the superseded unstable V100 TVM directory. The selected V100 measurement uses
20 CUDA-event samples of 32 calls per phase and trial, or 640 measured calls
per phase in each of three trials. It passed correctness and fixed-schedule
checks, and its total between-trial CV was 0.562%.

The maximum within-trial phase CV was 5.247%, narrowly above the unchanged 5%
diagnostic gate. No sample is removed and the stored gate decision remains
false. The final CSV, audit JSON, generated table, and documentation therefore
mark this value `QUALIFIED_WITH_CAVEAT`. Triton 3.6 is explicitly recorded as
unsupported on V100 (sm_70), rather than represented as a failed experiment.

## Frozen-policy data separation

`scripts/aggregate_pathtile_results.py` derives the architecture policy from
training shapes only and verifies that the saved policy timestamp precedes all
holdout trial timestamps.  Holdout measurements are read only after that check.
This guards the intended train/holdout separation during reconstruction.

## Exclusions

- No new GPU benchmark is run by any program in `scripts/`.
- Exploratory and backup programs are not canonical result producers.
- The preprocessed ASHRAE CSV dataset is not duplicated in the audit bundle;
  end-to-end remeasurement expects the existing project files under
  `pre_preprocessed_data/`.
- Nsight `.ncu-rep` files are retained as profiler evidence; the compact
  paper-facing table is reconstructed from exported details CSVs.

Run `python -m scripts.verify_artifact --project-root .` after any result or
code change.  The verifier is designed to fail on schema drift, numeric drift, missing paper
artifacts, an invalid frozen-policy split, or Python syntax errors.

## Interpretation and timing boundaries

- PathTile is a **frozen, offline path- and device-specific launch-configuration
  policy for Warp-tiled kernels**, not a runtime per-shape autotuner or a
  selector between the four kernel families.
- Aggregate microbenchmark latency is the sum of independently timed FWD,
  dX, and dK paths, not a measured end-to-end training step.
- PathTile candidate timing excludes compilation, allocation, validation,
  and dK output-buffer zeroing; the warp-tiled dK atomic accumulation requires
  the output to be zero-initialized. Cross-implementation timing boundaries
  should therefore be checked in the acquisition code when interpreting
  performance comparisons.
- The `best-tested` held-out reference minimizes latency only over the
  candidate configurations evaluated for each path and GPU; it is not a
  global performance optimum.
- A100 Nsight profiling and A6000 end-to-end training cover distinct
  evaluation scopes. The complementary P100 study uses an earlier
  implementation and is reported separately.
- The `V100_STABILITY_REMEASUREMENT*.md` and patch `README` files are
  historical acquisition/installation records, not instructions to rerun or
  overwrite the qualified final V100 TVM evidence.

`artifact/verification_report.md` is generated by the repository verifier.
Its `PASS` reflects the saved-artifact reconstruction at the time of its
last run, not independent scientific verification of GPU benchmark fairness.
