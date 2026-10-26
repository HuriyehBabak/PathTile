# Recorded environment

The final measurements were collected on the Pascal cluster with the project at
`/home/huriyeh.babak/s4convd-paper` and the interpreter at
`/home/huriyeh.babak/micromamba/envs/s4convd/bin/python`.

## GPU nodes

| Role | Node | Device | Architecture | Compute capability | Frozen schedule |
|---|---|---|---|---:|---|
| A100 | `pascal-node12.l3s.intra` | NVIDIA A100 PCIe 40 GB | Ampere | 8.0 | W16 / dK B128 |
| A6000 | `pascal-node14.l3s.intra` | NVIDIA RTX A6000 | Ampere | 8.6 | W16 / dK B128 |
| V100 | `pascal-node10.l3s.intra` | Tesla V100 SXM3 32 GB | Volta | 7.0 | W16 / dK B256 |

## Software recorded for the final artifact

| Component | Version |
|---|---|
| Python | 3.14.6 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN reported by PyTorch | 9.10.2 (raw version code `91002`) |
| NumPy | 2.5.2 |
| pandas | 3.0.5 |
| Matplotlib | 3.11.1 |
| Triton | 3.6.0 |
| TVM-TIRx | 0.25.0.post1 |
| TVM build Python | 3.11.16 |
| LLVM used for TVM build | 18.1.8 |
| CUDA toolkit used for TVM build | 12.9 |
| System CUDA toolkit used by cluster tools | 12.4 |

PyTorch and custom-kernel comparisons use FP32 with TF32 disabled.
Main, shape, and scheduling measurements use seed 42.
The actual training seeds of the original paired end-to-end trials
could not be recovered from the available run records.
Seed 0 in the remeasurement example is illustrative.
Main and sweep measurements use 50 warm-ups and 100 measured repetitions
per trial unless a saved metadata file states otherwise.

These versions and GPU nodes describe the recorded evaluation environment,
not requirements for other machines or evidence of portability to untested
architectures. The P100 complementary experiment is documented separately.

The Python environment does not need to be activated through Conda.  On Pascal,
invoke its interpreter directly:

```bash
/home/huriyeh.babak/micromamba/envs/s4convd/bin/python -m scripts.rebuild_artifact --project-root .
```

The source-built TVM environment is loaded with `source scripts/tvm_env.sh`.
Its source tree is recorded at
`/home/huriyeh.babak/third_party/apache-tvm-v0.25.0` and its isolated build
environment at `/home/huriyeh.babak/tvm-build-env-v025`. CPU-only artifact
reconstruction does not import TVM and therefore does not require those paths;
they are needed only for a fresh TVM GPU measurement.

To capture current node-specific versions without running a benchmark:

```bash
/home/huriyeh.babak/micromamba/envs/s4convd/bin/python \
  -m scripts.capture_environment \
  --output artifact/environment_capture.json
```
