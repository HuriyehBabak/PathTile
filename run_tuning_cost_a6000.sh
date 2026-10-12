#!/bin/bash
#SBATCH --job-name=pathtile-cost-a6000
#SBATCH --partition=ampere
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/pathtile_selector/tuning_cost_v1/a6000_slurm_%j.log

set -euo pipefail

cd ~/s4convd-paper

export PATH="/home/huriyeh.babak/micromamba/envs/s4convd/bin:$PATH"
export PYTHONPATH=.

export PYCUDA_CACHE_DIR="$PWD/results/pathtile_selector/tuning_cost_v1/cache/a6000"

rm -rf "$PYCUDA_CACHE_DIR"
mkdir -p "$PYCUDA_CACHE_DIR"

echo "HOST:"
hostname

echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader

python benchmark/benchmark_pathtile_tuning_cost.py \
  --mode run \
  --preset pathtile-train \
  --trial-id 1 \
  --warmup 50 \
  --runs 100 \
  --warps-candidates 4,8,16 \
  --bwdk-block-candidates 96,128,192,224,256,320,384 \
  --seed 42 \
  --device 0 \
  --output-dir results/pathtile_selector/tuning_cost_v1/a6000
