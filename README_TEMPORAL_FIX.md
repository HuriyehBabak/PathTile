# Corrected temporal training protocol — pilot version

This package adds separate files to the existing `s4convd-paper` checkout.
It does not replace `dataset.py`, `kernels/warp_tiled_ampere.py`, or any old results.
Extract the ZIP in the project root; inspect the included patch if desired.

## First step: audit data and test the loader

From the project root on Pascal:

```bash
sbatch scripts/audit_temporal.sbatch
```

Read `temporal-audit-JOBID.log`. The CPU audit reads all three CSVs, constructs
valid windows, hashes each source file, prints split date ranges, invalid-row
counts, valid/selected windows, scalers, baseline scores and training batch count.
It stops on duplicates or overlapping/out-of-order split date ranges. Inspect
such errors before changing policy; do not silently drop duplicates or permit
overlap. This audit uses compute-node memory rather than the login node.

The supplied Slurm partition and Python path follow your previous environment.
If this partition requires a GPU even for CPU jobs, submit the audit with the
cluster's CPU partition or add `--gres=gpu:nvidia_rtx_a6000:1` to `sbatch`.

## Second step: GPU integration pilot, after the audit passes

```bash
sbatch scripts/pilot_temporal.sbatch
```

This runs Warp then PyTorch serially, seed 0, two epochs, batch 16384, the original
launch configuration, SGD 0.001/momentum 0.9, and the original model architecture.
Kernel correctness checks remain enabled. It uses a new output directory named
with the Slurm job ID. This is an integration pilot, NOT final paper evidence.
No CUDA/PyTorch execution was possible in the preparation environment.

Inspect `training_history.csv` for finite, changing loss and validation RMSLE;
inspect `parameter_change_l2`, predictions, baseline comparisons, paired differences
and `correctness.csv`. Improving over a zero baseline alone is insufficient to
claim competitive forecasting; compare persistence too. Two epochs do not promise
convergence. Choose training duration/hyperparameters using validation only,
freeze the protocol, then run the three final paired seeds. Do not tune against
test scores or reuse pilot timings as final measurements.

## Defined changes

- Group by `(building_id, meter)` and sort by timestamp. Inputs are 48 consecutive
  hourly observations; the target is the immediately following hour. All 49 rows
  must have finite features and nonnegative energy. Missing/invalid rows create
  gaps that windows cannot cross. Each split builds its own windows: no history
  is carried across split boundaries. Source row numbers identify targets.
- Sample 10% of complete valid windows without replacement, using numpy
  `default_rng(42)`; retain sorted selected-window order. The former pandas row
  sample is deliberately removed. Neighboring sampled windows can overlap within
  a split; train/validation/test time ranges must be disjoint.
- Transform the energy input using `log1p`. Standardize the four input features
  using all valid training rows only (population standard deviation; constant
  features use scale 1). Reuse exactly those statistics for validation and test.
- Targets are `log1p` of raw nonnegative energy, without standardization. The
  model predicts that log quantity. Train with **unclipped log-space MSE** so
  negative model outputs receive gradients. This changes the objective from the
  old clamped standardized-target RMSLE. Architecture and CUDA kernels are unchanged.
- Evaluate global RMSLE on raw energy, stably as
  `sqrt(mean((max(predicted_log, 0) - log1p(raw_target))**2))`.
  This equals evaluating predictions `expm1(max(predicted_log, 0))` in energy
  units, without risking exponential overflow. Float64 target logs come directly
  from the CSV energy values. Validation is now global, not a batch-mean score.
- Use a separate seeded DataLoader generator to keep shuffled order reproducible
  independently of the model's dropout RNG. Both backends use identical data,
  normalization, initial seed and loader configuration. GPU floating-point
  reductions can still differ; equal seeds do not promise identical training.
- Preserve epoch-1 timing warm-up and stage instrumentation. The actual number
  of batches is measured; it is **not assumed to remain 39**.
- Save source snapshots/hashes, data hashes, exact configuration, initial model
  hash, full epoch history, test predictions and a final checkpoint. Checkpoint
  and file-writing work happens outside timed training loops.

## Outputs and compatibility

New output CSVs label the loss `train_log_mse` and validation `val_global_rmsle`.
`training_final_test_metrics.csv` contains model, zero and persistence RMSLE,
nonpositive-prediction fraction, parameter change and actual batch count.
`test_predictions.npz` contains predicted log values, target log values,
persistence log values and source CSV target row numbers.

**Do not run the old `aggregate_end_to_end.py` on these outputs.** It assumes
39 batches, old column names, and equality within 1e-12. The included
`scripts/review_temporal_pair.py PAIR_DIRECTORY` checks pairing metadata and
reports metric and prediction differences without inventing an equivalence
tolerance. A final three-trial aggregation and manuscript update should use the
new completed runs after the pilot has been reviewed.

Old end-to-end RMSLE scores are not raw-energy accuracy evidence and should not
be mixed with corrected scores. Corrected end-to-end timings must be rerun for
both backends. Standalone kernel benchmarks are unchanged by this data fix.

## Validation performed here

11 CPU tests passed: grouping/order, target alignment, gaps, duplicate rejection,
invalid readings, minimum length, sampling reproducibility, training-only scaler,
split chronology and mathematically correct RMSLE/baselines. One optional test
was skipped because PyTorch is absent; it checks the actual loss gradient for a
negative prediction and DataLoader collation, and runs on your cluster. All Python
files compile. CUDA kernels and model code were checked byte-for-byte against
the supplied source and are unchanged. Real data and GPU training remain to be
validated on Pascal; training quality is not established by these CPU tests.
