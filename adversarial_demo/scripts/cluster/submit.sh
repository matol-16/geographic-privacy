#!/usr/bin/env bash
#
# Submit the full multi-node evaluation:
#   1. a job ARRAY of shards (one attack x one image window per task), then
#   2. a single MERGE job that runs only after the whole array succeeds.
#
# Usage:
#   ./submit.sh                 # submit array + dependent merge
#   MAX_CONCURRENT=32 ./submit.sh   # cap how many array tasks run at once (--array=...%32)
#   ./submit.sh --dry-run       # print the plan and array mapping, submit nothing
#
# Edit scripts/cluster/config.sh first (attacks, budgets, TOTAL_IMAGES, IMAGES_PER_SHARD,
# paths) and fill the #SBATCH placeholders in eval_shard.slurm / merge.slurm.

set -euo pipefail

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CLUSTER_DIR}/config.sh"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "${RESULTS_BASE}" "${PLOTS_DIR}" "${CLUSTER_DIR}/slurm_logs"

echo "=============================================================="
echo "Dataset:       ${DATASET}"
echo "Shard attacks (${NUM_ATTACKS}):  ${ATTACK_TYPES[*]}"
echo "Merge attacks (${#MERGE_ATTACK_TYPES[@]}):  ${MERGE_ATTACK_TYPES[*]}"
echo "Total images:  ${TOTAL_IMAGES}  (images/shard=${IMAGES_PER_SHARD} -> ${NUM_WINDOWS} windows)"
echo "Array size:    ${ARRAY_SIZE} jobs  (${NUM_ATTACKS} attacks x ${NUM_WINDOWS} windows)"
echo "Budgets:       ${BUDGETS_OVERRIDE}"
echo "Results base:  ${RESULTS_BASE}"
echo "=============================================================="

ARRAY_SPEC="0-$(( ARRAY_SIZE - 1 ))"
[[ -n "${MAX_CONCURRENT:-}" ]] && ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"

if (( DRY_RUN )); then
  echo "[dry-run] would submit: sbatch --array=${ARRAY_SPEC} eval_shard.slurm"
  echo "[dry-run] task -> (attack, window) mapping:"
  for (( t = 0; t < ARRAY_SIZE; t++ )); do
    ai=$(( t / NUM_WINDOWS )); wi=$(( t % NUM_WINDOWS ))
    printf '  task %3d -> attack=%-16s window=%d  images[%d:%d)\n' \
      "$t" "${ATTACK_TYPES[$ai]}" "$wi" "$(( wi * IMAGES_PER_SHARD ))" "$(( wi * IMAGES_PER_SHARD + IMAGES_PER_SHARD ))"
  done
  echo "[dry-run] would then submit merge.slurm with --dependency=afterok:<array_jobid>"
  exit 0
fi

cd "${CLUSTER_DIR}"   # so relative #SBATCH --output=slurm_logs/... resolves here

shard_jid=$(sbatch --parsable \
  --export=ALL,CLUSTER_DIR="${CLUSTER_DIR}" \
  --array="${ARRAY_SPEC}" \
  "${CLUSTER_DIR}/eval_shard.slurm")
echo "Submitted shard array job: ${shard_jid}  (array ${ARRAY_SPEC})"

merge_jid=$(sbatch --parsable \
  --export=ALL,CLUSTER_DIR="${CLUSTER_DIR}" \
  --dependency="afterok:${shard_jid}" \
  "${CLUSTER_DIR}/merge.slurm")
echo "Submitted merge job:       ${merge_jid}  (runs after ${shard_jid} succeeds)"
echo
echo "Monitor:  squeue -u \$USER"
echo "If some shards fail, re-run them (they resume from saved state), then submit"
echo "the merge alone:  sbatch --export=ALL,CLUSTER_DIR=${CLUSTER_DIR} ${CLUSTER_DIR}/merge.slurm"
