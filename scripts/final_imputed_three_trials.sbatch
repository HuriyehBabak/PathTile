#!/bin/bash
#SBATCH --job-name=imputed-final
#SBATCH --partition=ampere
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=imputed-final-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="/home/huriyeh.babak/micromamba/envs/s4convd/bin/python"
RUN_ROOT="results/end_to_end/imputed_final_${SLURM_JOB_ID}"

mkdir -p "$RUN_ROOT"
cp "$0" "$RUN_ROOT/submitted_script.sbatch"

for trial in 1 2 3; do
    seed=$((trial - 1))

    if [ "$trial" -eq 2 ]; then
        backends=(pytorch warp)
    else
        backends=(warp pytorch)
    fi

    for backend in "${backends[@]}"; do
        echo "START trial=$trial seed=$seed backend=$backend"

        "$PYTHON_BIN" -u -m kernels.warp_tiled_ampere_imputed \
            --mode train \
            --train-backend "$backend" \
            --output-dir "$RUN_ROOT/trial${trial}/$backend" \
            --warps-per-block 16 \
            --bwdk-block 128 \
            --epochs 5 \
            --data-fraction 0.1 \
            --batch-size 16384 \
            --num-workers 4 \
            --prefetch-factor 18 \
            --seed "$seed"

        echo "FINISHED trial=$trial backend=$backend"
    done

    "$PYTHON_BIN" -u scripts/review_imputed_pair.py \
        "$RUN_ROOT/trial${trial}"
done

echo "ALL THREE PAIRED TRIALS COMPLETED: $RUN_ROOT"
