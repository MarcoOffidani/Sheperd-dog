#!/bin/bash

#SBATCH --job-name=evac80
#SBATCH --array=0-79
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

cd "$(dirname "$0")"
source .venv/bin/activate
export PYTHONPATH=.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
SEED=$((TASK_ID % 20))

if (( TASK_ID < 20 )); then
    VARIANT="walls_speed"
    WALL_FLAG="--walls"
    SPEED_FLAG="--speed"
elif (( TASK_ID < 40 )); then
    VARIANT="walls_nospeed"
    WALL_FLAG="--walls"
    SPEED_FLAG="--no-speed"
elif (( TASK_ID < 60 )); then
    VARIANT="nowalls_speed"
    WALL_FLAG="--no-walls"
    SPEED_FLAG="--speed"
else
    VARIANT="nowalls_nospeed"
    WALL_FLAG="--no-walls"
    SPEED_FLAG="--no-speed"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-slurm_runs}"
RUN_DIR="${OUTPUT_ROOT}/${VARIANT}/seed-${SEED}"
OUTPUT_DIR="${RUN_DIR}/output"
CHECKPOINT_DIR="${RUN_DIR}/checkpoints"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "${OUTPUT_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/stdout.log") 2> >(tee -a "${LOG_DIR}/stderr.log" >&2)

echo "task_id=${TASK_ID}"
echo "variant=${VARIANT}"
echo "seed=${SEED}"
echo "output_dir=${OUTPUT_DIR}"
echo "checkpoint_dir=${CHECKPOINT_DIR}"
echo "log_dir=${LOG_DIR}"

python main.py \
    --exp-name evac80 \
    --variant-name "${VARIANT}" \
    --seed "${SEED}" \
    -n 60 \
    --learn-timesteps 6000000 \
    --device cpu \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --log-dir "${LOG_DIR}" \
    ${WALL_FLAG} \
    ${SPEED_FLAG}
