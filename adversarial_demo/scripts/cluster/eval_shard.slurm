#!/usr/bin/env bash
#
# Array job: each task runs ONE shard = one attack on one image window.
# Submitted by submit.sh with --array=0-(NUM_ATTACKS*NUM_WINDOWS - 1).
# Task t -> attack = t / NUM_WINDOWS, window = t % NUM_WINDOWS.
#
#SBATCH --job-name=advshard
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
# ============================================================================== #
#  JEAN ZAY: FILL IN THESE PLACEHOLDERS, then uncomment (remove the leading "##").
#  The account suffix selects the GPU partition (e.g. @v100 / @a100 / @h100) and
#  must match --constraint and your allocation.
# ============================================================================== #
##SBATCH --account=CHANGE_ME@CHANGE_GPU      # e.g. abc@a100
##SBATCH --constraint=CHANGE_GPU             # e.g. a100  (omit on default V100)
##SBATCH --qos=CHANGE_QOS                    # e.g. qos_gpu-t3 (<=20h) or qos_gpu-t4 (<=100h)
# ============================================================================== #

set -euo pipefail

# CLUSTER_DIR is exported by submit.sh; default to this script's directory otherwise.
: "${CLUSTER_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${CLUSTER_DIR}/config.sh"

TASK="${SLURM_ARRAY_TASK_ID:?Run via submit.sh, or pass sbatch --array=...}"
ATTACK_IDX=$(( TASK / NUM_WINDOWS ))
WINDOW_IDX=$(( TASK % NUM_WINDOWS ))
if (( ATTACK_IDX >= NUM_ATTACKS )); then
  echo "Task ${TASK} is out of range (only ${ARRAY_SIZE} shards); nothing to do."
  exit 0
fi
ATTACK="${ATTACK_TYPES[$ATTACK_IDX]}"
WINDOW_START=$(( WINDOW_IDX * IMAGES_PER_SHARD ))

build_ablation_flags   # -> ABLATION_FLAGS (from the RUN_* toggles in config.sh)

echo "=============================================================="
echo "Shard task ${TASK}: attack='${ATTACK}' window ${WINDOW_IDX}/${NUM_WINDOWS}"
echo "  images [${WINDOW_START}:$(( WINDOW_START + IMAGES_PER_SHARD ))) of ${TOTAL_IMAGES}"
echo "  budgets=${BUDGETS_OVERRIDE}  parallel_workers=${PARALLEL_WORKERS}"
echo "  ablations: ${ABLATION_FLAGS[*]}"
echo "  overrides: ${OVERRIDE_FLAGS[*]}"
echo "=============================================================="

module purge 2>/dev/null || true
# module load <cuda / compiler modules your build needs>   # often unnecessary with conda
activate_env
export MPLBACKEND=Agg

cd "${PROJECT_DIR}"
srun python main.py evaluate-dataset-shard \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${ATTACK}" \
  --total-images "${TOTAL_IMAGES}" \
  --images-per-shard "${IMAGES_PER_SHARD}" \
  --image-shard-index "${WINDOW_IDX}" \
  --results-dir "${RESULTS_BASE}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  "${ABLATION_FLAGS[@]}" \
  "${OVERRIDE_FLAGS[@]}"

echo "Shard task ${TASK} done."
