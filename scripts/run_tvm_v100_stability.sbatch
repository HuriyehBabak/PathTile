#!/bin/bash
#SBATCH --job-name=tvm-v100-stability-v3
#SBATCH --partition=volta
#SBATCH --nodelist=pascal-node10.l3s.intra
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/huriyeh.babak/s4convd-paper/results/external_baselines/tvm/slurm-v100-stability-v3-%j.log

set -euo pipefail

cd /home/huriyeh.babak/s4convd-paper
source scripts/tvm_env.sh

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun \
  --ntasks=1 \
  --cpus-per-task=4 \
  --cpu-bind=cores \
  "$TVM_ENV/bin/python" \
  -m benchmark.benchmark_tvm_v100_stability \
  --output-root results/external_baselines/tvm/v100_stability_v3 \
  --samples 20 \
  --calls-per-sample 32 \
  --queue-lead-in 64
