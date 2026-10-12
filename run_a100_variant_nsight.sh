#!/bin/bash
#SBATCH --job-name=a100-ncu-variants
#SBATCH --partition=ampere
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/nsight/a100_variant_comparison/slurm-%j.log

set -euo pipefail

cd "$HOME/s4convd-paper"

PYTHON="$HOME/micromamba/envs/s4convd/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found: $PYTHON"
    exit 1
fi

if ! command -v ncu >/dev/null 2>&1; then
    echo "ERROR: ncu not found in PATH"
    exit 1
fi

echo "============================================================"
echo "ENVIRONMENT"
echo "============================================================"
hostname
"$PYTHON" --version
ncu --version
nvidia-smi --query-gpu=name --format=csv,noheader

run_profile () {
    variant="$1"
    phase_dir="$2"
    kernel="$3"
    module="$4"
    phase_arg="$5"

    outdir="results/nsight/a100_variant_comparison/${variant}/${phase_dir}"
    mkdir -p "$outdir"

    base="${outdir}/a100_${variant}_${phase_dir}"

    echo
    echo "============================================================"
    echo "VARIANT: ${variant}"
    echo "PHASE:   ${phase_dir}"
    echo "KERNEL:  ${kernel}"
    echo "============================================================"

    ncu \
        --set full \
        --kernel-name-base function \
        --kernel-name "${kernel}" \
        --launch-skip 5 \
        --launch-count 1 \
        --force-overwrite \
        --export "${base}" \
        "$PYTHON" -m "${module}" --phase "${phase_arg}"

    ncu \
        --import "${base}.ncu-rep" \
        --page details \
        --csv \
        > "${base}_details.csv"
}

# GMC
run_profile \
    gmc \
    fwd \
    dwconv1d_fwd_coalesced \
    benchmark.profile_coalesced_a100 \
    fwd

run_profile \
    gmc \
    bwd_input \
    dwconv1d_bwd_input_coalesced \
    benchmark.profile_coalesced_a100 \
    bwd_input

run_profile \
    gmc \
    bwd_weight \
    dwconv1d_bwd_kernel_coalesced \
    benchmark.profile_coalesced_a100 \
    bwd_kernel

# Shared
run_profile \
    shared \
    fwd \
    dwconv1d_fwd_shared \
    benchmark.profile_shared_a100 \
    forward

run_profile \
    shared \
    bwd_input \
    dwconv1d_bwd_input_shared \
    benchmark.profile_shared_a100 \
    bwd-input

run_profile \
    shared \
    bwd_weight \
    dwconv1d_bwd_kernel_shared \
    benchmark.profile_shared_a100 \
    bwd-kernel

echo
echo "============================================================"
echo "ALL SIX A100 COMPARATIVE PROFILES COMPLETE"
echo "============================================================"
