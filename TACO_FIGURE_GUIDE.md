# TACO figure integration guide — synchronized with the current manuscript

This guide documents the **eight figures referenced by the current paper**:

- Two explanatory diagrams authored separately from the benchmark generator:
  `pathtile_study_overview.pdf` (Introduction) and `cuda_kernel_designs.pdf`
  (CUDA Kernel Designs).
- Six performance figures produced from saved canonical measurements by
  `paper/generate_final_figures.py`.

The current LaTeX manuscript references files under `images/`. The generator's
output is under `paper/final_artifacts/`; copy or stage the correct PDF
artifacts into the manuscript's `images/` directory before compiling. Do not
change LaTeX paths blindly: use the paths that actually exist in the paper
checkout. PNG renderings are convenient for inspection, while the vector PDFs
are intended for publication.

**Important:** Figure colors, hatching, font size, and clipping are properties
of the generated PDFs and cannot be verified from the Markdown bundle alone.
Always visually inspect the *current* PDFs at the final journal page width.
Older notes in `PUBLICATION_FIGURE_UPGRADE.md` describe a historical design
revision and may not match the latest generated figures.

For ACM accessibility, preserve descriptive `\Description{...}` alternatives.
The captions below are consistent with the current manuscript. All numeric
speedups and aggregate latencies describe the **tested implementations and
workloads**, not performance bounds for general-purpose frameworks.

## Manuscript figures (eight total)

### 1. Study overview — `images/pathtile_study_overview.pdf`

**Placement:** Introduction; single-column figure (`figure`,
`width=\linewidth`).

**Caption:** Study overview. Four CUDA kernel families are evaluated across
FWD, dX, and dK. PathTile selects one fixed configuration per execution path
and target GPU using offline training shapes; the configurations are frozen
before held-out evaluation. End-to-end training is examined separately using
the S4ConvD model and the Great Energy Predictor III dataset.

**Description:** The diagram connects the four CUDA kernel families and three
training paths to offline per-device PathTile configuration, held-out workload
evaluation, and separate application-level training measurements.

### 2. Kernel designs — `images/cuda_kernel_designs.pdf`

**Placement:** CUDA Kernel Designs; full-width figure (`figure*`,
`width=\textwidth`).

**Caption:** Execution strategies for short depthwise Conv1D. Naive and GMC
use independent output or coefficient ownership. Shared-memory tiling adds
cooperative operand staging without cross-thread dK reduction; Warp-tiled
uses sequence-level warp mapping for FWD/dX and hierarchical reduction with
block-level atomic accumulation for dK.

**Description:** The diagram contrasts Naive, GMC, Shared, and Warp-tiled CUDA
kernels across FWD, dX, and dK, distinguishing thread mapping, on-chip staging,
and the weight-gradient reduction structure. Shared dK retains sequential
per-coefficient reduction; Warp-tiled dK uses cross-thread hierarchical
reduction and atomic updates.

### 3. Main cross-GPU comparison — `images/fig_cross_gpu_latency.pdf`

**Placement:** Cross-GPU Performance; full-width.

**Caption:** Cross-GPU performance for the main workload. (a) FWD, dX, and dK
latency for PyTorch and PathTile-configured warp-tiled kernels. Error bars
show one between-trial standard deviation. (b) Path-specific and aggregate
speedups over PyTorch.

**Description:** Aggregate independently measured FWD+dX+dK latency decreases
from 47.424 to 11.283 ms on A100, from 56.723 to 15.212 ms on RTX A6000, and
from 54.886 to 13.585 ms on V100; dK has the largest path-specific gains.
The sums are microbenchmark aggregates, not directly timed training steps.

### 4. Kernel progression — `images/fig_v100_ablation.pdf`

**Placement:** Kernel-Design Progression; full-width.

**Caption:** Cumulative kernel-design progression on V100. (a) FWD, dX, and dK
latency for PyTorch, Naive, GMC, Shared, and Warp-tiled. (b) Aggregate latency
removed between adjacent implementations. The progression is not a
leave-one-out ablation.

**Description:** Aggregate latency decreases from approximately 54.9 ms for
PyTorch to 13.6 ms for Warp-tiled. The largest adjacent reduction is
GMC-to-Shared, driven principally by the dK path. Because changes are
cumulative, the graph does not attribute individual gains to isolated
primitive operations.

### 5. Framework/compiler comparison — `images/fig_external_baselines.pdf`

**Placement:** Framework and Compiler Baselines; full-width.

**Caption:** Framework and compiler baselines on (a) A100, (b) RTX A6000,
and (c) V100. A usable Triton path is unavailable on V100 in the evaluated
environment; the V100 TVM-TIRx result marked with $\dagger$ uses the qualified
timing protocol documented in the methodology.

**Description:** Tested PyTorch, Triton, TVM-TIRx, and PathTile configurations
are compared by the aggregate of independently timed FWD, dX, and dK. The
A100 PathTile-to-TVM difference is small; A6000 and V100 show larger measured
differences. V100 TVM evidence is qualified rather than a strict stability
pass. No universal framework performance limit is implied.

### 6. Workload-shape portability — `images/fig_shape_generalization.pdf`

**Placement:** Cross-Shape Performance; full-width.

**Caption:** Speedup over PyTorch across ten workload shapes on (a) RTX A6000
and (b) V100. Circles denote individual shapes, diamonds group geometric
means, and vertical spans show minimum-to-maximum ranges.

**Description:** The ten A6000 cases range from 2.24 to 3.73 times, and the
ten V100 cases from 2.31 to 4.01 times. Grouped shape variations and geometric
means describe performance within the tested workloads only.

### 7. Held-out configuration regret — `images/fig_holdout_regret.pdf`

**Placement:** Frozen-Configuration Generalization; full-width.

**Caption:** Held-out regret of frozen PathTile configurations. (a) Individual
cases and per-GPU means. (b) Empirical cumulative distributions. Overall,
26 of 30 cases are within 1% and 29 of 30 within 2% of the best-tested
reference.

**Description:** Per-GPU mean regrets are approximately 0.007% on A100,
0.486% on RTX A6000, and 0.444% on V100. The reference is the lowest measured
candidate within the finite tested configuration space, not a global optimum.

### 8. End-to-end training — `images/fig_end_to_end_a6000.pdf`

**Placement:** End-to-End Training; full-width.

**Caption:** Paired end-to-end S4ConvD training on RTX A6000. (a) Mean epoch
wall time, (b) recorded GPU-stage time per batch decomposed into model-forward,
aggregate backward, and other recorded GPU work, and (c) speedup relative to
PyTorch across measurement scopes. Backward includes dX, dK, and other training
operations rather than representing an isolated kernel measurement.

**Description:** Mean epoch wall time decreases from approximately 17.23 to
16.73 seconds, while recorded GPU-stage time decreases from 354.2 to 211.4 ms
per batch. The geometric-mean reported speedups are 1.03 times for complete
epochs and 1.68 times for recorded GPU stages. The instrumentation does **not**
identify which remaining components dominate epoch time. Paired test RMSLE is
unchanged at the reported precision.

## Publication and interpretation checks

1. Rebuild the six performance PDFs from saved canonical CSVs using the
   documented CPU-only procedure in `artifact/REPRODUCIBILITY.md`.
2. Ensure the two conceptual diagrams come from their authoritative sources;
   they are not substitutes for the six generated performance figures.
3. Compile the paper with the actual eight `images/*.pdf` paths and check
   float order, readability at page width, font embedding, labels, and legends.
4. Keep profiler-instrumented durations separate from uninstrumented CUDA-event
   timing; do not sum them or treat them as directly interchangeable.
5. Preserve the V100 TVM-TIRx qualified-timing dagger and the precise scope
   of the recorded backward interval in the end-to-end panel.
6. The historical V100 timing and patch-installation Markdown files are
   provenance records; use the current artifact documentation for reproduction.
