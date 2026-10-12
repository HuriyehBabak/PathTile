# Artifact Verification Report

**Status: PASS**

The verification was CPU-only and rebuilt every canonical aggregate from saved measurements. No GPU benchmark was executed.

- Canonical CSVs reconstructed and compared: 61
- Python source files syntax-checked: 93
- Generated paper artifacts checked: 29
- Main benchmark: 3 GPUs, 5 methods, 3 trials
- Warp speedup range: 3.728881x to 4.203224x
- Frozen holdout: 26/30 within 1%, 29/30 within 2%
- Frozen-policy mean regret: 0.312414%
- End-to-end epoch speedup: 1.030021x
- End-to-end GPU-stage speedup: 1.675478x
- Maximum paired test-RMSLE difference: 0.0
- External baselines: 11 GPU/method rows audited
- TVM V100 evidence status: QUALIFIED_WITH_CAVEAT
- TVM V100 between-trial total CV: 0.562000%
- TVM V100 maximum phase CV: 5.246614% (5% diagnostic gate)
