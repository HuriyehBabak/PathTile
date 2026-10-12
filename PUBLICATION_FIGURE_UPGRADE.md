# Publication Figure Upgrade

This patch upgrades all six final figures to a consistent ACM TACO-oriented
visual system without running or changing any GPU experiment. All plotted
values continue to come from the canonical saved CSV files.

## What changed

- `fig_cross_gpu_latency`: stacked FWD/dX/dK latency and a compact, annotated
  cross-GPU path-speedup matrix.
- `fig_v100_ablation`: stacked path decomposition plus the incremental latency
  removed by every optimization transition.
- `fig_external_baselines`: one panel per GPU, stacked path decomposition for
  PyTorch, Triton, TVM-TIRx, and PathTile; the unsupported V100 Triton case and
  qualified V100 TVM evidence are marked explicitly.
- `fig_shape_generalization`: per-shape distributions, group ranges, and group
  geometric means for both evaluated GPUs.
- `fig_holdout_regret`: per-GPU regret distributions and a complementary
  empirical CDF showing the 1% and 2% coverage claims directly.
- `fig_end_to_end_a6000`: epoch time, GPU-stage decomposition, and a
  scope-to-system speedup panel that makes dilution by non-kernel work visible.

`TACO_FIGURE_GUIDE.md` supplies ready-to-use captions, text alternatives, and
the recommended evaluation narrative.

## Install and rebuild on Pascal

From the project root:

```bash
cd "$HOME/s4convd-paper"

cp paper/generate_final_figures.py \
  "$HOME/generate_final_figures.before_publication_upgrade.py"

tar --no-same-owner -xzf \
  "$HOME/s4convd_publication_figure_upgrade.tar.gz"

sha256sum -c PUBLICATION_FIGURE_UPGRADE_MANIFEST.sha256

PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python

"$PY" -m py_compile paper/generate_final_figures.py
"$PY" -m scripts.rebuild_artifact --project-root .
"$PY" -m scripts.verify_artifact --project-root .
"$PY" -m scripts.check_archive_manifest --project-root .
```

These commands are CPU-only reconstruction and validation steps.  They do not
submit SLURM jobs or repeat benchmarks.

## Suggested LaTeX placement

Use the six PDF files as double-column `figure*` elements at
`width=\textwidth`. See `TACO_FIGURE_GUIDE.md` for captions and ACM
`\Description{}` text alternatives. PNG files are only for quick preview.

## Validation performed before packaging

- Figure generator syntax check: PASS
- Final paper artifact generation: PASS (29 files)
- Full CPU-only artifact rebuild: PASS (66 canonical outputs)
- Artifact verification: PASS
- Six vector PDFs rendered independently with Poppler: PASS
- Embedded Unicode fonts and one-page layout: PASS
- Visual overlap and clipping inspection at full resolution: PASS
