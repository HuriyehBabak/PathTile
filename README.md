# PathTile: Analysis and Offline Autotuning of CUDA Kernels for Short Depthwise Conv1D

This repository contains CUDA/PyTorch implementations and experimental
measurements for short depthwise one-dimensional convolution (Conv1D)
in S4ConvD.

The study evaluates three training paths separately:

- Forward propagation (FWD)
- Input-gradient computation (dX)
- Weight-gradient computation (dK)

Repository: https://github.com/HuriyehBabak/PathTile

## Kernel designs and PathTile

The evaluated kernel families are **Naive, global-memory-coalesced (GMC),
Shared-memory-tiled, and Warp-tiled**. Their comparison examines changes
in execution mapping, operand staging, and reduction organization.

Warp-tiled FWD/dX maps complete `(batch, channel)` sequences to warps.
Warp-tiled dK uses batch-tile parallelism, tap packing, hierarchical
warp/block reduction, and atomic accumulation of block contributions.
The specialization targets `L <= 48` and `K <= 48` in FP32.

**PathTile selects launch configurations within the Warp-tiled family.**
It does not choose among the four kernel families.

For each GPU, PathTile uses 20 tuning shapes to select one configuration
independently for FWD, dX, and dK. These configurations are frozen before
evaluation on ten held-out shapes per GPU. Deployment performs no
shape-dependent configuration search or selection.

## Reconstruct the primary results (CPU only)

The following commands reconstruct aggregates from saved measurements,
regenerate figures and tables, and run the artifact consistency checks.
They do not repeat GPU benchmarks or independently validate the GPU
measurement procedure.

From the repository root:

```bash
python -m pip install -r artifact/analysis-requirements.txt
python -m scripts.rebuild_artifact --project-root .
```

The documented successful completion messages are:

```text
Artifact verification: PASS
Full CPU-only artifact rebuild: PASS
```

The rebuild writes a verification report to:

```text
artifact/verification_report.md
```

For the complete project snapshot, also run:

```bash
python -m scripts.check_archive_manifest --project-root .
```

See `artifact/REPRODUCIBILITY.md` for measurement commands and scope,
and `artifact/ENVIRONMENT.md` for recorded software versions.

The supplementary packing and Flash-DWC experiments described below
have separate measurement outputs. A successful primary artifact rebuild
should not be interpreted as verification of those outputs unless they
are explicitly included in its manifest and checks.

## Primary performance results

The primary workload is:

```text
B=16384, H=128, L=48, K=48, FP32
```

Aggregate latency is the **sum of independently measured FWD, dX, and dK
latencies**, not a directly timed training step.

| GPU | PyTorch aggregate (ms) | PathTile aggregate (ms) | Speedup | Frozen configuration |
|---|---:|---:|---:|---|
| A100 PCIe 40 GB | 47.424 | 11.283 | 4.20x | 16 / 16 / 128 |
| RTX A6000 | 56.723 | 15.212 | 3.73x | 16 / 16 / 128 |
| V100 32 GB | 54.886 | 13.585 | 4.04x | 16 / 16 / 256 |

Configuration entries specify FWD warps per block, dX warps per block,
and dK threads per block, respectively.

Across 30 held-out shape–GPU cases, frozen PathTile configurations achieve:

- **0.312% mean configuration regret**
- 26/30 cases within 1% of the best-tested reference
- 29/30 cases within 2% of the best-tested reference

The reference selects the best measured per-shape configurations within
the same finite candidate space. It is not a global optimum over arbitrary
kernels or launch configurations.

The pooled global-static policy achieves 1.231% mean regret.
The selected GPU-dependent configuration difference is concentrated
in the dK thread-block size.

## Framework and compiler comparisons

Comparisons include PyTorch and, where supported, `torch.compile`,
Triton 3.6, and the evaluated TVM-TIRx 0.25.0.post1 implementation.

| GPU | Triton aggregate (ms) | TVM-TIRx aggregate (ms) | PathTile aggregate (ms) |
|---|---:|---:|---:|
| A100 PCIe 40 GB | 21.774 | 11.518 | 11.283 |
| RTX A6000 | 31.766 | 19.702 | 15.212 |
| V100 32 GB | Unavailable in evaluated environment | 19.918† | 13.585 |

These comparisons characterize the tested implementations and
configurations, not the general performance limits of the frameworks.

† The V100 TVM-TIRx result passed correctness and fixed-schedule checks,
but its maximum within-trial phase coefficient of variation (5.247%)
exceeded the predefined 5% diagnostic threshold. The evidence is retained
with the status `QUALIFIED_WITH_CAVEAT`.

**Timing boundary:** Primary PathTile candidate measurements exclude
compilation, allocation, validation, and dK output initialization.
The atomic dK kernel requires a zero-initialized output buffer.
Separate reduction and packing experiments report measurements that
include initialization.

## Controlled dK tap-packing ablation

A controlled experiment compares jointly processing one, two, or four
filter coefficients. Within each GPU, the variants retain the same
launch grid, thread-block size, batch-tile size, and global accumulation
strategy.

At the primary workload:

| GPU | Pack 1 kernel (ms) | Pack 2 kernel (ms) | Pack 4 kernel (ms) | Pack 4 speedup over Pack 1 |
|---|---:|---:|---:|---:|
| A100 | 9.894 | 6.185 | 4.177 | 2.37x |
| RTX A6000 | 12.687 | 7.582 | 5.478 | 2.32x |
| V100 | 14.743 | 9.059 | 6.022 | 2.45x |

Measurements use three trials, 50 warm-up executions, and 100 measured
executions. All variants pass the tested main and edge-case checks.

On RTX A6000, four-tap packing reduces:

- Executed warp instructions by approximately **58.4%**
- Shared-memory load instructions by approximately **37.3%**

Achieved occupancy remains nearly unchanged: 24.92% versus 24.87%.
These instruction counts are not measurements of physical memory traffic.

Additional checks include:

- A100 warm-up sensitivity: the speedup remains approximately 2.35x
  with 200 warm-up executions.
- RTX A6000 filter-length sweep: 1.80–2.32x kernel speedups across
  `K = 8, 13, 16, 32, 48`.
- RTX A6000 batch-size sweep: 2.32–2.49x kernel speedups across
  `B = 64, 1024, 4096, 16384`.

Four-tap packing is already part of the primary Warp-tiled implementation.
These ablation speedups are not additional gains over the headline results.

## Flash-DWC comparison

A separate experiment compares FP32 dK on RTX A6000 with Flash-DWC,
pinned to commit:

```text
74d31d8fae0fb189385c0b8bc39e11d7e81ba408
```

External repository: https://github.com/GilgameshXYZ123/Flash-DWC

The experiment evaluates `B = 64, 1024, 4096, 16384` with
`H=128` and `L=K=48`, sweeping the tested partition parameter `GZ`.

At the primary workload, the best-tested Flash-DWC configuration
with `GZ=4096` gives:

| Scope | PathTile (ms) | Flash-DWC (ms) | Flash-DWC speedup |
|---|---:|---:|---:|
| Native layout | 5.450 | 2.844 | 1.92x |
| Input reuse | 5.452 | 4.141 | 1.32x |

Both scopes include output initialization.

Native-layout timing assumes inputs are already in each implementation's
required layout. Input-reuse timing assumes a pre-existing converted input
activation and includes the required upstream-gradient and output
conversions. It excludes the cost of creating and retaining the reused
input representation.

Flash-DWC achieves lower best-tested mean latency in both scopes at all
four tested batch sizes. These results establish a performance limitation
of the evaluated PathTile dK implementation.

Numerical checks use a direct CPU FP64 reference computed from the same
FP32 inputs, with `atol=1e-3` and `rtol=5e-4`. Agreement depends on the
Flash-DWC configuration and input. The primary-workload configuration
reported above passes all three final-output checks in both scopes;
these checks do not cover every timed invocation.

The comparison covers dK on RTX A6000. It does not establish relative
FWD, dX, or complete training performance.

## End-to-end training

Three paired, serial S4ConvD/ASHRAE training trials on RTX A6000 report
geometric-mean speedups of:

- **1.03x** for training-epoch wall time
- **1.68x** for recorded GPU-stage time

Paired test RMSLE is equal at the reported precision.
Recorded GPU stages and complete epochs represent different measurement
scopes. The instrumentation does not identify a unique cause for the
difference between their speedups.

## Complementary P100 study

The Tesla P100 experiment uses an earlier implementation, a separate
Naive CUDA baseline, and a restricted profiling protocol.

Its 3.26x kernel and 1.29x epoch speedups are reported separately and
are not pooled with the primary results. Arithmetic-intensity and
effective-bandwidth estimates in that study are model-derived.

## Repository layout

- `kernels/`: CUDA/PyCUDA kernels and training integration.
- `benchmark/`: GPU measurement entry points.
- `results/`: measurements, correctness checks, and profiler outputs.
- `scripts/`: reconstruction, auditing, and verification tools.
- `paper/generate_final_figures.py`: figure and table generator.
- `paper/final_artifacts/`: generated figures and tables.
- `artifact/`: environment, provenance, and reproducibility documentation.
- `TACO_FIGURE_GUIDE.md`: manuscript figure guidance under its historical
  filename.

## Historical documentation

`EXTERNAL_BASELINES_FINALIZATION.md`, `FINAL_ARTIFACT_INTEGRATION.md`,
`PUBLICATION_FIGURE_UPGRADE.md`, `TVM_BASELINE_README.md`, and
`V100_STABILITY_REMEASUREMENT*.md` describe historical acquisition
and integration steps.

Use `artifact/REPRODUCIBILITY.md` and the corresponding measurement
metadata for the current reconstruction workflow. Historical patch
commands should not be treated as the primary reproduction procedure.