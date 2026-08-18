#!/usr/bin/env bash
# Portable Slurm template for Qwen3-VL teacher response generation.
#
# Required environment variable:
#   QWEN3_VL_DATA_ROOT=/path/containing/models,datasets,data
# Optional:
#   CONTAINER_IMAGE=/path/to/pytorch-cuda.sif
#   SITE_PACKAGES=/path/to/site-packages

#SBATCH --job-name=qwen3-vl-teacher
#SBATCH --output=logs/gen_teacher_%j.out
#SBATCH --error=logs/gen_teacher_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100-80:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

set -euo pipefail

: "${QWEN3_VL_DATA_ROOT:?Set QWEN3_VL_DATA_ROOT before submitting this job}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GENERATOR=(
    python -u "${PROJECT_ROOT}/gen_teacher_data.py"
    --total 50000
    --tp 4
    --batch_size 32
    --resume
)

mkdir -p "${PROJECT_ROOT}/logs"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-${QWEN3_VL_DATA_ROOT}/.cache/huggingface}"

if [[ -n "${CONTAINER_IMAGE:-}" ]]; then
    singularity exec --nv "${CONTAINER_IMAGE}" bash -lc \
        "export PYTHONPATH='${SITE_PACKAGES:-}':\"\${PYTHONPATH:-}\"; ${GENERATOR[*]}"
else
    "${GENERATOR[@]}"
fi
