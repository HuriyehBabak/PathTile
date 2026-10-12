# External-baseline finalization

This patch finalizes the CPU-only aggregation of the saved PyTorch, Triton,
TVM-TIRx, and PathTile measurements. It does not launch a GPU kernel and no
additional benchmark should be submitted.

## Evidence policy

- A100 and RTX A6000 TVM-TIRx use their audited three-trial, single-call
  CUDA-event results.
- V100 TVM-TIRx is read explicitly from
  `results/external_baselines/tvm/v100_stability_v3`. The superseded unstable
  result in `results/external_baselines/tvm/v100` is never selected.
- The V100 V3 result passed correctness and fixed-schedule checks. It contains
  20 timing samples x 32 calls x 3 trials for each path, with no discarded
  samples. Total between-trial CV was 0.562%. The maximum within-trial phase CV
  was 5.247%, narrowly above the unchanged 5% diagnostic gate. Consequently,
  the aggregate labels this evidence `QUALIFIED_WITH_CAVEAT`; it does not
  rewrite the stored `stability_pass=False` decision.
- Triton 3.6 is reported as unsupported on the V100 (sm_70), not as a missing
  or failed measurement.

The V100 TVM-TIRx value is conservative for the PathTile comparison because
all transiently delayed timing samples remain in the arithmetic mean.

## Install and verify

Run from the Pascal head node:

```bash
cd "$HOME/s4convd-paper"

tar --no-same-owner -xzf \
  "$HOME/s4convd_external_baselines_final_patch.tar.gz"

sha256sum -c \
  EXTERNAL_BASELINES_FINALIZATION_MANIFEST.sha256

PY=/home/huriyeh.babak/micromamba/envs/s4convd/bin/python

"$PY" -m py_compile \
  scripts/aggregate_external_baselines.py
```

## Reconstruct the final tables

This command is CPU-only:

```bash
"$PY" -m scripts.aggregate_external_baselines \
  --project-root .
```

It writes:

```text
results/external_baselines/final_summary/external_baselines_all_metrics.csv
results/external_baselines/final_summary/external_baselines_paper_table.csv
results/external_baselines/final_summary/external_baselines_protocol_notes.csv
results/external_baselines/final_summary/external_baselines_paper_table.tex
results/external_baselines/final_summary/external_baselines_audit.json
```

Inspect the compact table and the audit decision:

```bash
sed -n '1,10p' \
  results/external_baselines/final_summary/external_baselines_paper_table.csv

grep -n -E \
  'v100_stability_v3|QUALIFIED_WITH_CAVEAT|strict_stability_gate_pass' \
  results/external_baselines/final_summary/external_baselines_audit.json
```

## Expected headline values

All times are the sum of forward, backward-input, and backward-weight means for
`B=16384, H=128, L=48, K=48`, FP32.

| GPU | PyTorch (ms) | Triton (ms) | TVM-TIRx (ms) | PathTile (ms) | PathTile vs best compiler |
|---|---:|---:|---:|---:|---:|
| A100 | 47.424 | 21.774 | 11.518 | 11.283 | 1.021x |
| RTX A6000 | 56.723 | 31.766 | 19.702 | 15.212 | 1.295x |
| V100 | 54.886 | unsupported | 19.918 (qualified) | 13.585 | 1.466x |

For the manuscript, place a dagger on the V100 TVM-TIRx entry and retain the
generated table caption. Do not describe the V100 strict diagnostic gate as a
pass. The PathTile-vs-PyTorch headline results remain unchanged.
