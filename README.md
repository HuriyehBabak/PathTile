# PathTile: Hardware-Aware CUDA Execution for Short S4ConvD Depthwise Conv1D

This repository contains CUDA/PyTorch implementations and saved experimental
measurements for short depthwise one-dimensional convolution (Conv1D) in the
evaluated S4ConvD model. The work studies three training paths separately:
forward propagation (FWD), input-gradient computation (dX), and weight-gradient
computation (dK).

The kernel-design progression is **Naive -> global-memory-coalesced (GMC) ->
Shared-memory-tiled -> Warp-tiled**. These are execution designs, not the
configuration choices made by PathTile. The warp-tiled FWD/dX kernels map
complete `(batch, channel)` sequences to warps; warp-tiled dK uses batch-tile
parallelism, packed register accumulation, hierarchical warp/block reduction,
and block-level atomic updates. The warp-tiled specialization targets
`L <= 48` and `K <= 48` in FP32.

**PathTile is an offline path- and device-specific launch-configuration policy
for the warp-tiled kernels only.** It uses 20 training workload shapes per GPU
to choose one configuration independently for FWD, dX, and dK. Configurations
are frozen before testing ten held-out shapes per GPU; no shape-dependent
configuration search or selection runs during deployment. PathTile does not
choose between Naive, GMC, Shared, and Warp-tiled implementations.

The main FP32 workload is `B=16384, H=128, L=48, K=48`. The primary GPUs are
NVIDIA A100, RTX A6000, and V100. Comparisons use PyTorch and, where supported,
`torch.compile`, Triton 3.6, and the evaluated TVM-TIRx 0.25.0.post1
implementation. These are comparisons of the **tested implementations and
configurations**, not the performance limits of their frameworks.

## Reconstruct the reported results (CPU only)

The following commands reconstruct canonical aggregates from the **saved**
measurements, regenerate the paper figures/tables, and run the repository's
consistency checks. They do **not** repeat GPU benchmarks or independently
validate the acquisition code against its underlying CUDA execution.

```bash
cd /path/to/s4convd-paper
python -m pip install -r artifact/analysis-requirements.txt
python -m scripts.rebuild_artifact --project-root .
```

The documented successful completion is:

```text
Artifact verification: PASS
Full CPU-only artifact rebuild: PASS
```

A new verification report is written to `artifact/verification_report.md`.
On the complete project snapshot, check the archive manifest after rebuilding:

```bash
python -m scripts.check_archive_manifest --project-root .
```

For original measurement commands and their scope, see
`artifact/REPRODUCIBILITY.md`; for recorded software versions, see
`artifact/ENVIRONMENT.md`.

## Canonical headline results

The aggregate latency in this table is the **sum of independently timed** FWD,
dX, and dK latencies. It is not the latency of a directly timed training step.

| GPU | PyTorch aggregate (ms) | PathTile-configured Warp aggregate (ms) | Speedup | Frozen configuration (FWD/dX warps, dK threads) |
|---|---:|---:|---:|---|
| A100 PCIe 40 GB | 47.423956 | 11.282756 | 4.203224x | 16 / 16 / 128 |
| RTX A6000 | 56.723401 | 15.211909 | 3.728881x | 16 / 16 / 128 |
| V100 SXM3 32 GB | 54.886113 | 13.585207 | 4.040138x | 16 / 16 / 256 |

Across 30 held-out shape--GPU cases, frozen PathTile has a mean latency regret
of 0.312414% relative to the **best-tested per-shape candidate within the
finite tested search space** (26/30 cases within 1%, 29/30 within 2%). The
pooled global-static configuration has 1.231% mean regret. The measured
architecture-dependent choice is concentrated in the dK block size.

In three paired, serial S4ConvD/ASHRAE training trials on RTX A6000, the
geometric-mean speedups are 1.030021x for epoch wall time and 1.675478x for
the recorded GPU stages; paired test RMSLE is equal at the reported precision.
The recorded GPU stages and complete epoch measure different scopes.

| GPU | Triton aggregate (ms) | TVM-TIRx aggregate (ms) | PathTile aggregate (ms) | PathTile vs. evaluated best compiler |
|---|---:|---:|---:|---:|
| A100 PCIe 40 GB | 21.774445 | 11.517959 | 11.282756 | 1.020846x |
| RTX A6000 | 31.765586 | 19.702207 | 15.211909 | 1.295183x |
| V100 SXM3 32 GB | unavailable in evaluated Triton 3.6 environment | 19.917635† | 13.585207 | 1.466127x |

† The V100 TVM-TIRx result passed correctness and fixed-schedule checks, but
its maximum within-trial phase coefficient of variation (5.247%) exceeded the
pre-specified 5% diagnostic threshold. The full evidence was retained and is
marked `QUALIFIED_WITH_CAVEAT`; this is not a strict stability pass.

**Timing boundary:** PathTile candidate timing excludes compilation,
allocation, correctness validation, and zero-initialization of the dK output
buffer. The warp-tiled dK kernel requires a zeroed output for its atomic
updates. This measurement boundary must be considered when interpreting or
revalidating cross-implementation comparisons; the saved-documentation bundle
alone is not proof that all compared implementations include identical setup
costs.

A complementary Tesla P100 experiment uses an earlier implementation, a
separate naive CUDA baseline, and a restricted profiling protocol. Its 3.26x
kernel and 1.29x epoch speedups are **not pooled** with the primary results.

## Repository layout and documentation

- `kernels/`: CUDA/PyCUDA kernels and training integration.
- `benchmark/`: GPU measurement entry points.
- `results/`: raw/per-trial, aggregate, correctness, and profiler evidence.
- `scripts/`: reconstruction, auditing, and verification tools.
- `paper/generate_final_figures.py`: figure and table generator.
- `paper/final_artifacts/`: generated PDF/PNG figures and CSV/LaTeX tables.
- `artifact/`: scope, environment, provenance, and reproduction guidance.
- `TACO_FIGURE_GUIDE.md`: manuscript-oriented figure captions, descriptions,
  and placement notes.

`EXTERNAL_BASELINES_FINALIZATION.md`, `FINAL_ARTIFACT_INTEGRATION.md`,
`PUBLICATION_FIGURE_UPGRADE.md`, `TVM_BASELINE_README.md`, and the
`V100_STABILITY_REMEASUREMENT*.md` files document **historical patch and
acquisition steps**. Do not use their old patch-installation or rerun commands
as the primary reproduction workflow for the completed artifact. For current
reconstruction use `artifact/REPRODUCIBILITY.md` and saved acquisition metadata.
