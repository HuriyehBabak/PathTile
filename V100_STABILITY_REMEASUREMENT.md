# TVM-TIRx V100 stability remeasurement

This patch adds an independent, quality-gated remeasurement for the unstable
TVM-TIRx V100 run from Slurm job 113124. It does not change the original TVM
benchmark or any existing result.

The initial run is used only to freeze the schedule selected consistently or
by majority across its three trials: `fwd_bh4`, `bwd_input_bh1`, and
`bwd_weight_t64`. The remeasurement compiles the main shape once, performs a
continuous excluded burn-in, and measures three back-to-back trials with the
same compiled kernels. Correctness, allocations, compilation, warm-up, GPU
snapshots, and file I/O remain outside CUDA-event timing.

The run passes only when total between-trial CV is at most 2% and every
within-trial phase CV is at most 5%. All raw measurements and the original
unstable result are retained.

Install from the project root and syntax-check:

```bash
tar --no-same-owner -xzf "$HOME/s4convd_tvm_v100_stability_patch.tar.gz"

/home/huriyeh.babak/micromamba/envs/s4convd/bin/python -m py_compile \
  benchmark/benchmark_tvm_v100_stability.py
```

Submit:

```bash
sbatch scripts/run_tvm_v100_stability.sbatch
```

The Slurm log is written below `results/external_baselines/tvm/`, and the new
results are written to `results/external_baselines/tvm/v100_stability/`.
