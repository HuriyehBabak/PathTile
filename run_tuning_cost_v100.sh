#!/bin/bash
#SBATCH --job-name=pathtile-cost-v100
#SBATCH --partition=volta
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/pathtile_selector/tuning_cost_v1/v100_slurm_%j.log

set -euo pipefail

cd ~/s4convd-paper

export PATH="/home/huriyeh.babak/micromamba/envs/s4convd/bin:$PATH"
export PYTHONPATH=.

export PYCUDA_CACHE_DIR="$PWD/results/pathtile_selector/tuning_cost_v1/cache/v100"

# Cold cache only before Trial 1.
rm -rf "$PYCUDA_CACHE_DIR"
mkdir -p "$PYCUDA_CACHE_DIR"

echo "HOST:"
hostname

echo "GPU:"
nvidia-smi --query-gpu=name --format=csv,noheader

for TRIAL in 1 2 3; do
    echo
    echo "================================================================"
    echo "TRIAL ${TRIAL}"
    echo "================================================================"

    python benchmark/benchmark_pathtile_tuning_cost.py \
      --mode run \
      --preset pathtile-train \
      --trial-id "$TRIAL" \
      --warmup 50 \
      --runs 100 \
      --warps-candidates 4,8,16 \
      --bwdk-block-candidates 96,128,192,224,256,320,384 \
      --seed 42 \
      --device 0 \
      --output-dir results/pathtile_selector/tuning_cost_v1/v100
done

echo
echo "CACHE FILES AFTER:"
find "$PYCUDA_CACHE_DIR" -type f | wc -l
