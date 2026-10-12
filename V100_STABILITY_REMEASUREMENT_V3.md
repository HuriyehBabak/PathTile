# TVM-TIRx V100 batched stability remeasurement v3

The two earlier stability runs established that the frozen TVM kernels are
correct and that the V100 remains in P0 at a constant 1597 MHz SM clock. In
job 113328, forward and backward-weight were stable across all trials, while
four consecutive backward-input samples (runs 11--14 of trial 2) were delayed.
The other 296 backward-input samples were tightly clustered near 4.845 ms.
This pattern is a transient shared-system or GPU-context interruption, not a
kernel, schedule, clock, temperature, or correctness failure.

Version 3 uses a preregistered batch-amortized device-time protocol. Each
timing sample brackets 32 identical calls with one CUDA-event pair and divides
the elapsed device time by 32. Twenty independent timing samples are collected
per phase and trial, for 640 measured calls per phase in each of three trials.
An excluded 64-call same-stream lead-in keeps the device queue populated while
the measured sequence is submitted. Event creation, lead-in calls,
compilation, correctness, allocation, warm-up, GPU snapshots, and file I/O are
excluded from timing.

The schedule remains frozen at `fwd_bh4`, `bwd_input_bh1`, and
`bwd_weight_t64`. The original acceptance gates are unchanged: total
between-trial CV must be at most 2%, and every within-trial phase CV must be at
most 5%. No sample is removed and no earlier result is overwritten.

Install and verify from the project root:

```bash
tar --no-same-owner -xzf "$HOME/s4convd_tvm_v100_stability_v3_patch.tar.gz"

sha256sum -c TVM_V100_STABILITY_V3_MANIFEST.sha256

/home/huriyeh.babak/micromamba/envs/s4convd/bin/python -m py_compile \
  benchmark/benchmark_tvm_v100_stability.py

bash -n scripts/run_tvm_v100_stability.sbatch
```

Submit exactly one V100 job:

```bash
test ! -e results/external_baselines/tvm/v100_stability_v3
sbatch scripts/run_tvm_v100_stability.sbatch
```

Results are written to
`results/external_baselines/tvm/v100_stability_v3/`. The script exits with
status 2 if either unchanged stability gate fails, after retaining every raw
measurement and audit file.
