# Reproducibility commands

## CPU-only reconstruction (recommended)

This is the authoritative paper-artifact command.  It reads saved measurements,
rewrites all canonical CSV aggregates, regenerates figures/tables, and verifies
the complete dependency chain.

```bash
cd ~/s4convd-paper

/home/huriyeh.babak/micromamba/envs/s4convd/bin/python \
  -m scripts.rebuild_artifact \
  --project-root .
```

Individual stages can also be run independently:

```bash
PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python

$PY -m scripts.aggregate_main_results --project-root .
$PY -m scripts.aggregate_shape_results --project-root .
$PY -m scripts.aggregate_pathtile_results --project-root .
$PY -m scripts.extract_ncu_metrics --project-root .
$PY -m scripts.aggregate_end_to_end --project-root .
$PY -m scripts.aggregate_external_baselines --project-root .
$PY paper/generate_final_figures.py --project-root .
$PY -m scripts.verify_artifact --project-root .
```

These commands perform no CUDA benchmark and do not modify raw trial files.

## Measurement contract

- Main tensor: FP32 `B=16384, H=128, L=48, K=48`.
- Paths: forward, backward-input (`dX`), and backward-weight (`dK`).
- Main comparison: three trials, 50 warm-ups, 100 timed iterations per path.
- GPU timing: CUDA events with synchronization before the timed measurement is
  consumed; CPU launch time is not substituted for GPU runtime.
- Total microbenchmark latency: sum of the three independently measured path
  means; this is not a directly timed training iteration.
- PathTile candidate timing excludes compilation, allocation, correctness
  validation, and zero-initialization of the dK output buffer. The warp-tiled
  dK kernel uses atomic accumulation and requires zeroed output storage.
  Treat this as a documented timing boundary; verify comparable boundaries
  in each baseline's acquisition code before inferring full-call speedup.
- Strict comparison: TF32 and custom fast math are disabled.
- Shape and PathTile candidate sweeps: seed 42 and independent correctness
  checks for all three paths.
- End-to-end: three paired trials, six runs in one serial A6000 allocation;
  steady state is epochs 2--5.  Epoch wall time is bracketed by CUDA
  synchronization, while model-forward/backward GPU stages use CUDA events.
- Reported uncertainty: sample standard deviation across trial means.
- Reported aggregate speedups: geometric mean where stated.
- Triton: version 3.6.0, three trials on A100 and A6000, with 50 warm-ups
  followed by 100 single-call CUDA-event measurements per path and trial;
  compilation and autotuning are outside final timing.
- TVM-TIRx: version 0.25.0.post1, three trials per GPU. A100 and A6000 use
  100 single-call CUDA-event measurements per path and trial. The qualified
  V100 V3 result uses 20 timing samples of 32 calls (640 measured calls) per
  path and trial, with the fixed `fwd_bh4`, `bwd_input_bh1`, and
  `bwd_weight_t64` schedule.

## Original GPU acquisition entry points

The saved results are already complete; these commands are included for artifact
review and should be scheduled one process per GPU allocation.  Output
directories should be new or explicitly backed up before remeasurement.

### Main baselines

The architecture-specific scripts contain the final workload, warm-up, run, and
seed constants:

| GPU | PyTorch | Naive | GMC | Shared |
|---|---|---|---|---|
| A100 | `benchmark.benchmark_pytorch_a100` | `benchmark.benchmark_naive_a100` | `benchmark.benchmark_coalesced_a100` | `benchmark.benchmark_shared_a100` |
| A6000 | `benchmark.benchmark_pytorch_a6000_final` | `benchmark.benchmark_naive_a6000_final` | `benchmark.benchmark_coalesced_a6000_final` | `benchmark.benchmark_shared_a6000_final` |
| V100 | `benchmark.benchmark_pytorch_v100_final` | `benchmark.benchmark_naive_v100_final` | `benchmark.benchmark_coalesced_v100_final` | `benchmark.benchmark_shared_v100_final` |

Each is invoked as a module, for example:

```bash
$PY -m benchmark.benchmark_pytorch_a6000_final
```

The final warp-tiled main kernel is acquired with:

```bash
# A100 or A6000
$PY kernels/warp_tiled_ampere.py \
  --mode benchmark --B 16384 --H 128 --L 48 --K 48 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/output/directory

# V100: the same runtime auto-detects Volta; dK uses block 256
$PY kernels/warp_tiled_ampere.py \
  --mode benchmark --B 16384 --H 128 --L 48 --K 48 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 256 \
  --output-dir /new/output/directory
```

### External compiler-framework baselines

Triton 3.6 supports the two Ampere GPUs in this artifact. From a fresh output
directory, run the smoke test and three trials on the selected A100 or A6000
node, then aggregate on the head node:

```bash
$PY -m benchmark.benchmark_triton_main --mode smoke

for TRIAL in 1 2 3; do
  $PY -m benchmark.benchmark_triton_main \
    --mode run \
    --trial-id "$TRIAL"
done

$PY -m benchmark.benchmark_triton_main \
  --mode aggregate \
  --gpu-slug a6000 \
  --expected-trials 3
```

Use `--gpu-slug a100` for A100 aggregation. A usable Triton 3.6 execution
path was not available for V100 (sm_70) in the **evaluated environment**;
therefore, no V100 Triton latency is reported. This is not a universal claim
about every Triton build or version.

TVM was built from tag `v0.25.0.post1`. Load the recorded build environment,
then run the standard smoke/trial sequence on A100 or A6000:

```bash
source scripts/tvm_env.sh

"$TVM_ENV/bin/python" -m benchmark.benchmark_tvm_main \
  --mode smoke --gpu-slug a6000

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

The V100 paper-facing TVM measurement is the preserved V3 batched-timing run
under `results/external_baselines/tvm/v100_stability_v3`. Its acquisition entry
point is `scripts/run_tvm_v100_stability.sbatch`. It intentionally exits with
status 2 when the unchanged stability gate is missed, after saving every
result. Job 113329 passed correctness and fixed-schedule checks and achieved a
0.562% total between-trial CV, but one phase reached 5.247% within-trial CV
against the 5% diagnostic gate. Do not rerun it merely to seek a passing sample
and do not relabel the stored decision as a pass.

### Shape and schedule candidates

```bash
# Fixed-schedule shape sweep; repeat trial-id 1, 2, and 3.
$PY -m benchmark.benchmark_shape_sweep_ampere \
  --mode run --preset full --trial-id 1 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/shape/root

# Candidate sweep; use the architecture's saved candidate list and repeat
# trial-id 1, 2, and 3 for pathtile-train, then independently for holdout.
$PY -m benchmark.benchmark_pathtile_candidate_sweep \
  --mode run --preset pathtile-train --trial-id 1 \
  --warmup 50 --runs 100 --seed 42 \
  --warps-candidates 4,8,16 \
  --bwdk-block-candidates 96,128,192,224,256,320,384 \
  --output-dir /new/candidate/root
```

The exact candidate lists, device identity, timestamps, and protocol for every
completed run are preserved in each trial's `metadata.json`; those files take
precedence over the illustrative command above.

### End-to-end S4ConvD/ASHRAE

The input CSVs must exist in `pre_preprocessed_data/`.  Run all six processes
serially inside one A6000 allocation using paired order:

```text
trial1: warp -> pytorch
trial2: pytorch -> warp
trial3: warp -> pytorch
```

For each run:

```bash
$PY kernels/warp_tiled_ampere.py \
  --mode train --train-backend warp \
  --epochs 5 --data-fraction 1.0 --batch-size 16384 \
  --num-workers 4 --prefetch-factor 18 --seed 0 \
  --warps-per-block 16 --bwdk-block 128 \
  --output-dir /new/end_to_end/trial1/warp
```

Replace `warp` with `pytorch` and the output directory for its paired run.  Do
not launch the paired trials concurrently: shared CPU/I/O contention changes
epoch wall time even though CUDA-stage timing remains valid.

## Interpretation constraints

- Describe PathTile as an **offline path- and device-specific
  launch-configuration policy for Warp-tiled kernels**. The holdout evidence
  supports one frozen configuration per GPU and path, not per-shape runtime
  adaptation or selection among the four kernel families.
- `val_batch_mean_rmsle` is the mean of batch RMSLE values.
  `test_rmsle` is a global RMSLE and is not directly comparable to it.
- Occupancy is not a standalone optimization objective and cannot by itself
  establish which hardware resource limits performance. The selected dK
  block size on A100 is 128 threads within the tested candidate set; resource
  explanations require supporting launch metadata and profiling evidence.
