# TVM-TIRx V100 stability remeasurement v2

Job 113296 established that the GPU itself was stable: it remained in P0 at a
constant 1597 MHz SM clock, with normal temperature and power headroom. The
per-phase medians were also nearly identical across trials. A few isolated
samples were approximately one or more scheduler quanta slower, however,
which inflated the arithmetic means and produced 35--39% within-phase CV.

The timing loop previously issued `start event -> Python TVM call -> end
event`. If the host task was descheduled after the start event had reached an
otherwise empty CUDA stream, that idle interval became part of the CUDA-event
measurement even though the kernel itself had not slowed down.

This patch retains the frozen V100 schedule (`fwd_bh4`, `bwd_input_bh1`, and
`bwd_weight_t64`) and the preregistered stability thresholds. Before each
measured phase, it enqueues 32 unmeasured calls on the same stream and creates
all event objects in advance. The lead-in calls are strictly before the first
start event in stream order, so their runtime is excluded. They keep the GPU
queue non-empty while Python submits the measured event/call sequence. The
Slurm step is additionally bound to allocated CPU cores.

No earlier result is overwritten. Version 2 writes to
`results/external_baselines/tvm/v100_stability_v2/`. The run passes only if
total between-trial CV is at most 2% and every within-trial phase CV is at most
5%.

Install from the project root and verify:

```bash
tar --no-same-owner -xzf "$HOME/s4convd_tvm_v100_stability_v2_patch.tar.gz"

sha256sum -c TVM_V100_STABILITY_V2_MANIFEST.sha256

/home/huriyeh.babak/micromamba/envs/s4convd/bin/python -m py_compile \
  benchmark/benchmark_tvm_v100_stability.py

bash -n scripts/run_tvm_v100_stability.sbatch
```

Submit exactly one V100 job:

```bash
test ! -e results/external_baselines/tvm/v100_stability_v2
sbatch scripts/run_tvm_v100_stability.sbatch
```

The script intentionally exits with status 2 if either stability gate fails,
after retaining all raw measurements, summaries, metadata, and GPU snapshots.
