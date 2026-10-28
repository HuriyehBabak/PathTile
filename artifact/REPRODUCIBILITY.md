# Reproducibility commands

## Scope and prerequisites

This document separates CPU-only reconstruction from fresh GPU measurements.

- CPU-only reconstruction requires the saved measurement archive in addition
  to the source code.
- The current Git checkout does not include that measurement archive.
  A download location has not yet been provided.
- GPU remeasurement requires a compatible GPU environment and the dependencies
  needed by the selected implementation.
- End-to-end training additionally requires the preprocessed ASHRAE CSV files,
  which are not included in this repository.

Run commands from the repository root. The examples below use the Python
interpreter selected by your active environment:

```bash
PY=python
```

The recorded acquisition environment is described in
[ENVIRONMENT.md](ENVIRONMENT.md). Its machine-specific paths are not required
on other systems.

## CPU-only reconstruction

After obtaining the saved measurements and restoring their original paths,
install the analysis dependencies and run:

```bash
"$PY" -m pip install -r artifact/analysis-requirements.txt

"$PY" -m scripts.rebuild_artifact --project-root .
```

This command reconstructs the canonical outputs included in the rebuild
pipeline, regenerates figures and tables, and runs artifact consistency
checks. It does not repeat GPU benchmarks or independently validate the
measurement procedure.

Individual stages can also be run independently:

```bash
"$PY" -m scripts.aggregate_main_results --project-root .
"$PY" -m scripts.aggregate_dk_reduction
"$PY" -m scripts.aggregate_shape_results --project-root .
"$PY" -m scripts.aggregate_pathtile_results --project-root .
"$PY" -m scripts.extract_ncu_metrics --project-root .
"$PY" -m scripts.aggregate_end_to_end --project-root .
"$PY" -m scripts.aggregate_external_baselines --project-root .
"$PY" paper/generate_final_figures.py --project-root .
"$PY" -m scripts.verify_artifact --project-root .
```

These commands do not run GPU benchmarks or modify raw trial measurements.
They may rewrite derived aggregates, figures, manifests, and verification
reports.

A successful rebuild verifies the outputs covered by its producers and
checks. It should not be interpreted as verification of every supplementary
experiment in the repository.

For a complete archived snapshot with its matching manifest, file integrity
can be checked using:

```bash
"$PY" -m scripts.check_archive_manifest --project-root .
```

The archive-integrity check requires the files listed in the manifest.
It is not expected to pass on a source-only checkout or after changes to
manifested files.

## Measurement contract

- Main workload: FP32 `B=16384, H=128, L=48, K=48`.
- Execution paths: forward (`FWD`), input gradient (`dX`), and weight
  gradient (`dK`).
- Unless an experiment specifies otherwise, measurements use three trials,
  50 warm-up executions, and 100 timed executions per path and trial.
- GPU timings use CUDA events. Event results are consumed after the
  required synchronization.
- Aggregate microbenchmark latency is the sum of independently measured
  FWD, dX, and dK latencies. It is not a directly timed training iteration.
- Reported means average trial means. Reported standard deviations describe
  between-trial variability unless explicitly stated otherwise.
- Speedup aggregation follows the definition given for each experiment.
  Paired training speedups are geometric means of paired trial ratios.

### Timing boundaries

For the primary PyTorch baselines, FWD timing includes explicit asymmetric
padding followed by grouped convolution. Backward timings bracket the
respective `torch.autograd.grad` calls. The preceding forward computations
used to construct the backward graphs are excluded.

Primary Warp and PathTile candidate timings use preallocated output buffers
and exclude compilation, allocation, numerical validation, and dK output
zero-initialization. Atomic dK accumulation requires the output buffer to be
zeroed before each invocation.

Separate reduction, packing, and tile/block experiments report both
kernel-only and initialization-inclusive scopes where specified.

Triton and TVM-TIRx dK timings include both partial- and final-reduction
kernels. Their output and workspace buffers are allocated before final
timing.

These comparisons characterize the evaluated implementation paths under
their documented timing boundaries.

### Precision and numerical validation

Primary PyTorch comparisons use FP32 with TF32 disabled. The custom CUDA
compilation flags and experiment-specific exceptions are documented in the
paper and acquisition metadata. In particular, the Flash-DWC comparison uses
different compilation settings.

Primary numerical validation compares FWD outputs, input gradients, and
weight gradients against PyTorch reference computations:

- FWD and dX: `atol=2e-3`, `rtol=2e-3`.
- dK: `atol=1e-2`, `rtol=5e-4`.

Shape-sweep validation retains the evaluated `H`, `L`, and `K`, with batch
size capped at `min(B, 128)`. Timing uses the full workload batch size.

These tolerances are experimental acceptance criteria, not analytical error
bounds. Flash-DWC uses a separate CPU FP64 reference and validation protocol.

## GPU acquisition entry points

The following commands identify acquisition entry points and illustrate
remeasurement. They do not automatically reconstruct every original job,
trial ordering, or environment.

Schedule one measurement process per GPU allocation. Use fresh output
directories and check each script's output settings before execution.
Some architecture-specific baseline scripts use predefined output paths.

### Main baselines

The architecture-specific scripts contain workload, warm-up, run, and seed
constants:

| GPU | PyTorch | Naive | GMC | Shared |
|---|---|---|---|---|
| A100 | `benchmark.benchmark_pytorch_a100` | `benchmark.benchmark_naive_a100` | `benchmark.benchmark_coalesced_a100` | `benchmark.benchmark_shared_a100` |
| RTX A6000 | `benchmark.benchmark_pytorch_a6000_final` | `benchmark.benchmark_naive_a6000_final` | `benchmark.benchmark_coalesced_a6000_final` | `benchmark.benchmark_shared_a6000_final` |
| V100 | `benchmark.benchmark_pytorch_v100_final` | `benchmark.benchmark_naive_v100_final` | `benchmark.benchmark_coalesced_v100_final` | `benchmark.benchmark_shared_v100_final` |

Each is invoked as a module, for example:

```bash
"$PY" -m benchmark.benchmark_pytorch_a6000_final
```

The primary V100 evaluation uses a V100 SXM3 device. The separate V100 PCIe
device is used for the supplementary tap-packing experiments.

### Warp-tiled main kernels

For A100 or RTX A6000:

```bash
"$PY" kernels/warp_tiled_ampere.py \
  --mode benchmark --B 16384 --H 128 --L 48 --K 48 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/output/directory
```

For V100, the runtime detects Volta and the selected dK block size is 256:

```bash
"$PY" kernels/warp_tiled_ampere.py \
  --mode benchmark --B 16384 --H 128 --L 48 --K 48 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 256 \
  --output-dir /new/output/directory
```

Repeat measurements according to the three-trial protocol, preserving each
trial in its own output directory.

### External compiler-framework baselines

#### Triton

The evaluated Triton 3.6 environment supports measurements on A100 and
RTX A6000. Check the script's output settings and use fresh destinations
before running:

```bash
"$PY" -m benchmark.benchmark_triton_main --mode smoke

for TRIAL in 1 2 3; do
  "$PY" -m benchmark.benchmark_triton_main \
    --mode run \
    --trial-id "$TRIAL"
done

"$PY" -m benchmark.benchmark_triton_main \
  --mode aggregate \
  --gpu-slug a6000 \
  --expected-trials 3
```

Use `--gpu-slug a100` for A100 aggregation.

The evaluated measurements use three trials, with 50 warm-up executions
followed by 100 single-call CUDA-event measurements per path and trial.
Compilation and autotuning occur outside final timing.

A usable Triton 3.6 execution path was unavailable for V100 in the evaluated
environment. This is not a universal claim about all Triton versions or
builds.

#### TVM-TIRx

The evaluated TVM build uses tag `v0.25.0.post1`. The original cluster
environment is loaded with:

```bash
source scripts/tvm_env.sh
```

This environment script contains acquisition-specific settings and must be
adapted for another machine.

For A100 or RTX A6000, the standard smoke-test and trial sequence is:

```bash
"$TVM_ENV/bin/python" -m benchmark.benchmark_tvm_main \
  --mode smoke \
  --gpu-slug a6000

for TRIAL in 1 2 3; do
  "$TVM_ENV/bin/python" -m benchmark.benchmark_tvm_main \
    --mode run \
    --gpu-slug a6000 \
    --trial-id "$TRIAL"
done

"$TVM_ENV/bin/python" -m benchmark.benchmark_tvm_main \
  --mode aggregate \
  --gpu-slug a6000 \
  --expected-trials 3
```

Use `a100` for the corresponding A100 measurements.

The final V100 result uses the separate V3 batched-timing protocol under
`results/external_baselines/tvm/v100_stability_v3`. Its acquisition entry
point is:

```text
scripts/run_tvm_v100_stability.sbatch
```

The fixed schedules are:

- FWD: `fwd_bh4`.
- dX: `bwd_input_bh1`.
- dK: `bwd_weight_t64`.

Each of three trials records 20 CUDA-event samples containing 32 calls per
path. Elapsed time is divided by 32 to obtain per-call latency.

The recorded V100 run passed correctness and fixed-schedule checks.
Its aggregate between-trial coefficient of variation was 0.562%.
The maximum within-trial phase coefficient of variation was 5.247%,
exceeding the predefined 5% diagnostic threshold.

The result is retained as `QUALIFIED_WITH_CAVEAT`. The acquisition script
intentionally exits with status 2 when the diagnostic gate is missed,
after saving the results. A failed diagnostic gate must not be relabeled
as a pass.

### Shape and configuration sweeps

Example fixed-configuration shape sweep on RTX A6000:

```bash
"$PY" -m benchmark.benchmark_shape_sweep_ampere \
  --mode run --preset full --trial-id 1 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/shape/root
```

Repeat with trial IDs 2 and 3. Use the device-specific frozen configuration
for other GPUs.

Example PathTile tuning-candidate sweep:

```bash
"$PY" -m benchmark.benchmark_pathtile_candidate_sweep \
  --mode run --preset pathtile-train --trial-id 1 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-candidates 4,8,16 \
  --bwdk-block-candidates 96,128,192,224,256,320,384 \
  --output-dir /new/candidate/root
```

Repeat the tuning sweep for three trials. Held-out measurements use the
corresponding holdout preset and separate output directories.

FWD and dX each evaluate three candidates, and dK evaluates seven.
Independent path selection therefore requires 13 path-candidate evaluations
per shape, or 260 per sweep over 20 tuning shapes.

Original PathTile selection fixes `B_tile=64` and `K_pack=4`.
The supplementary tile/block experiments do not change the frozen policies
used for the primary results.

For runs with acquisition metadata, consult the saved `metadata.json`
for the exact candidate lists, device identity, timestamps, and settings.

### Supplementary dK and profiling sources

The repository includes the following supplementary entry points:

- `benchmark/benchmark_dk_btile.py`: batch-tile experiment.
- `benchmark/benchmark_dk_btile_block.py`: joint batch-tile/block-size
  experiment.
- `benchmark/benchmark_dk_btile_confirm.py`: comparison of the selected
  tile/block candidate with the original configuration.
- `benchmark/profile_pytorch_warp.py`: PyTorch–Warp profiling harness.
- `scripts/aggregate_dk_reduction.py`: CPU-only reduction-statistics
  reconstruction.
- `scripts/evaluate_training_shape_budget.py`: cached tuning-budget replay.
- `scripts/freeze_pathtile_coverage_policy.py`: coverage-policy freezing.
- `scripts/evaluate_frozen_coverage_holdout.py`: evaluation of a frozen
  coverage policy using existing holdout measurements.

Their saved measurement outputs are maintained separately from the source
checkout. Consult the acquisition settings for each experiment rather than
assuming the primary timing defaults apply to every supplementary run.

The tile/block confirmation uses six trials, 50 warm-up calls, and
100 timed calls per scope and trial. Specify `--trials 6` when reproducing
that protocol; the script's default trial count is three.

The PyTorch–Warp profiling harness is intended for use with Nsight Compute.
It does not replace uninstrumented latency measurement. Resource and memory
profiling use separate runs, capturing one post-warm-up invocation per path
and implementation in each run.

### End-to-end S4ConvD/ASHRAE

The preprocessed input CSVs must exist in `pre_preprocessed_data/`.
The reported protocol uses FP32, batch size 16,384, five epochs, and
10% of each preprocessed training, validation, and test split.

Rows are sampled without replacement using `random_state=42`.
This dataset-sampling setting is distinct from the training seed used for
model initialization and training randomness.

The three paired trials run serially in one RTX A6000 allocation:

```text
trial1: warp -> pytorch
trial2: pytorch -> warp
trial3: warp -> pytorch
```

Example command for a new training run:

```bash
"$PY" kernels/warp_tiled_ampere.py \
  --mode train --train-backend warp \
  --epochs 5 --data-fraction 0.1 --batch-size 16384 \
  --num-workers 4 --prefetch-factor 18 --seed 0 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/end_to_end/trial1/warp
```

The example above uses seed 0, matching trial 1.
The original paired trials used training seeds 0, 1, and 2
for trials 1, 2, and 3, respectively. Both backends received
the same seed within each pair, as recorded in the Slurm
submission command for job 112060.
For trials 2 and 3, use --seed 1 and --seed 2, respectively,
with separate output directories for each trial and backend.

Replace `warp` with `pytorch` and use a separate output directory for the
paired backend. Use separate directories for subsequent trials and record
the actual seed and configuration for every new run.

Epoch 1 serves as warm-up. Epochs 2–5 provide the reported steady-state
measurements. Epoch wall time and recorded GPU-stage time are distinct
measurement scopes.

Test RMSLE is computed jointly over all test predictions and targets.
The validation field `val_batch_mean_rmsle` is an unweighted mean of
per-batch RMSLE values and is not directly comparable to global test RMSLE.

Run paired measurements serially. Concurrent runs can introduce resource
contention and change both wall-time and GPU measurements.

## Interpretation constraints

- PathTile is an offline, path- and device-specific launch-configuration
  selection procedure within the Warp-tiled family. It performs no runtime
  retuning or kernel-family selection.
- Held-out regret is relative to the best-tested candidates within the same
  finite configuration space. It does not establish proximity to a global
  performance optimum or transfer to an unevaluated GPU.
- Majority-winner selects the same configurations as PathTile on the full
  tuning sets. The reported results do not establish an advantage of
  PathTile's summed-latency objective over that heuristic.
- Reduced-budget replay quantifies candidate-measurement requirements using
  cached measurements. It does not measure actual reduced-budget tuning
  wall time.
- Occupancy, cache hit rates, instruction counts, and memory traffic must
  be interpreted together with the execution mapping. Individual counters
  do not isolate the causes of runtime differences.
- Profiler durations are excluded from primary speedup calculations.
  Profiling results depend on their replay and cache-control conditions.
- Numerical checks establish agreement for the tested inputs and tolerances,
  not universal correctness or bitwise determinism.
- Kernel-level speedup, recorded GPU-stage speedup, and training-epoch
  speedup describe different scopes and must not be treated as interchangeable.