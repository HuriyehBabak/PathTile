#!/bin/bash
#SBATCH --job-name=temporal-audit
#SBATCH --partition=ampere
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=temporal-audit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the s4convd-paper project root}"
PYTHON_BIN="/home/huriyeh.babak/micromamba/envs/s4convd/bin/python"
"$PYTHON_BIN" -m unittest discover -s tests -p test_temporal.py -v
"$PYTHON_BIN" -u dataset_temporal.py --fraction 0.1 --batch-size 16384
