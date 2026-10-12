# TVM-TIRx baseline handoff

This patch adds a strict-FP32 TVM baseline for the already measured workload
`B=16384, H=128, L=48, K=48`.  It does **not** rerun PyTorch, PathTile, or
Triton.  The baseline has independent forward, backward-input, and
backward-weight paths.  dK is a deterministic two-kernel partial/reduce path;
both launches are included in its CUDA-event time and workspace allocation is
excluded.

## Install into the project

From the directory containing `s4convd_tvm_baseline_patch.tar.gz`:

```bash
cd "$HOME/s4convd-paper"
tar --no-same-owner -xzf /path/to/s4convd_tvm_baseline_patch.tar.gz

PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python
"$PY" -m py_compile \
  kernels/tvm_depthwise.py \
  benchmark/tvm_pytorch_reference.py \
  benchmark/benchmark_tvm_main.py \
  scripts/aggregate_external_baselines.py
```

Do not rebuild TVM.  The existing build and CUDA smoke are already complete.

## Environment in every allocation

```bash
cd "$HOME/s4convd-paper"
source scripts/tvm_env.sh
TVM_PY="$TVM_ENV/bin/python"
```

## A6000 (node14)

Allocate a fresh, exclusive-enough GPU step; do not run trials concurrently:

```bash
srun \
  --partition=ampere \
  --nodelist=pascal-node14.l3s.intra \
  --gres=gpu:nvidia_rtx_a6000:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --pty bash
```

Inside it:

```bash
cd "$HOME/s4convd-paper"
source scripts/tvm_env.sh
TVM_PY="$TVM_ENV/bin/python"

"$TVM_PY" -m benchmark.benchmark_tvm_main \
  --mode smoke --gpu-slug a6000

for TRIAL in 1 2 3; do
  "$TVM_PY" -m benchmark.benchmark_tvm_main \
    --mode run --gpu-slug a6000 --trial-id "$TRIAL" || break
done

"$TVM_PY" -m benchmark.benchmark_tvm_main \
  --mode aggregate --gpu-slug a6000 --expected-trials 3
```

The smoke must say `TVM-TIRx smoke: PASS` before trials are run.  Send the full
error if it does not; do not add `--overwrite` blindly.

## A100 (node12)

```bash
srun \
  --partition=ampere \
  --nodelist=pascal-node12.l3s.intra \
  --gres=gpu:a100:1 \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --pty bash
```

Then use the A6000 command block with every `a6000` replaced by `a100`.

## V100 (node10)

First confirm the Slurm GRES string:

```bash
sinfo -N -h -o "%N|%P|%G|%t" | grep 'pascal-node10'
```

Allocate node10 with the exact reported V100 GRES.  Then use the same command
block with `--gpu-slug v100`.  TVM supports sm_70; the Triton 3.6 baseline was
correctly omitted there because that Triton release requires sm_80 or newer.

## Final saved-results aggregation

After all three TVM GPU aggregates exist:

```bash
cd "$HOME/s4convd-paper"
PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python

"$PY" -m scripts.aggregate_external_baselines --project-root .
```

Expected TVM files per GPU are:

```text
results/external_baselines/tvm/<gpu>/smoke/{correctness.csv,metadata.json}
results/external_baselines/tvm/<gpu>/trial{1,2,3}/{correctness.csv,raw.csv,summary.csv,metadata.json}
results/external_baselines/tvm/<gpu>/final_per_trial.csv
results/external_baselines/tvm/<gpu>/final_aggregate.csv
results/external_baselines/tvm/<gpu>/final_audit.csv
```

The cross-baseline aggregation produces:

```text
results/external_baselines/final_summary/external_baselines_all_metrics.csv
results/external_baselines/final_summary/external_baselines_paper_table.csv
results/external_baselines/final_summary/external_baselines_audit.json
```

## Interpretation rule

Report this as a framework baseline specialized and autotuned for the same
static shape.  Do not call it stock TVM, and do not imply that compilation or
autotuning is included in steady-state latency.  The exact label should be
`TVM-TIRx (autotuned)`.
