#!/bin/bash
#SBATCH --job-name=imputed-pilot
#SBATCH --partition=ampere
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=imputed-pilot-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the s4convd-paper project root}"
PYTHON_BIN="/home/huriyeh.babak/micromamba/envs/s4convd/bin/python"
RUN_ROOT="results/end_to_end/imputed_pilot_${SLURM_JOB_ID}"
# Two epochs validate integration; these are NOT final paper measurements.
for backend in warp pytorch; do
    "$PYTHON_BIN" -u -m kernels.warp_tiled_ampere_imputed \
        --mode train --train-backend "$backend" \
        --output-dir "$RUN_ROOT/$backend" \
        --warps-per-block 16 --bwdk-block 128 \
        --epochs 2 --data-fraction 0.1 --batch-size 16384 \
        --num-workers 4 --prefetch-factor 18 --seed 0
done
"$PYTHON_BIN" -u scripts/review_imputed_pair.py "$RUN_ROOT"
