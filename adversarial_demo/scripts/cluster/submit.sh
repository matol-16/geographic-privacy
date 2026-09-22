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
#   SKIP_MERGE=1 ./submit.sh    # submit the shard array ONLY (prints the merge command)
#   AFTER_JOBID=123 ./submit.sh # queue the array behind job 123 (a run already in flight)
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

# Optional extra dependency for the shard array: queue it behind a run already in flight
# (AFTER_JOBID=<jobid of the running array>) instead of competing for the same GPUs.
SHARD_DEP=()
[[ -n "${AFTER_JOBID:-}" ]] && SHARD_DEP=(--dependency="afterany:${AFTER_JOBID}")

if (( DRY_RUN )); then
  echo "[dry-run] would submit: sbatch --array=${ARRAY_SPEC} eval_shard.slurm"
  echo "[dry-run] task -> (attack, window) mapping:"
  for (( t = 0; t < ARRAY_SIZE; t++ )); do
    ai=$(( t / NUM_WINDOWS )); wi=$(( t % NUM_WINDOWS ))
    printf '  task %3d -> attack=%-16s window=%d  images[%d:%d)\n' \
      "$t" "${ATTACK_TYPES[$ai]}" "$wi" "$(( wi * IMAGES_PER_SHARD ))" "$(( wi * IMAGES_PER_SHARD + IMAGES_PER_SHARD ))"
  done
  [[ -n "${AFTER_JOBID:-}" ]] && echo "[dry-run] array would wait for job ${AFTER_JOBID} (afterany)"
  if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
    echo "[dry-run] SKIP_MERGE=1 -> no merge job would be submitted"
  else
    echo "[dry-run] would then submit merge.slurm with --dependency=afterok:<array_jobid>"
  fi
  exit 0
fi

cd "${CLUSTER_DIR}"   # so relative #SBATCH --output=slurm_logs/... resolves here

shard_jid=$(sbatch --parsable \
  --export=ALL,CLUSTER_DIR="${CLUSTER_DIR}" \
  --array="${ARRAY_SPEC}" \
  ${SHARD_DEP[@]+"${SHARD_DEP[@]}"} \
  "${CLUSTER_DIR}/eval_shard.slurm")
echo "Submitted shard array job: ${shard_jid}  (array ${ARRAY_SPEC})"
[[ -n "${AFTER_JOBID:-}" ]] && echo "  (queued behind job ${AFTER_JOBID})"

# SKIP_MERGE: when folding an attack into a run whose OTHER shards are still being computed,
# a merge chained to this array alone would stitch an incomplete run over the existing
# combined results. Submit the array now and merge once, by hand, after everything landed.
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  cat <<EOF
No merge job submitted (SKIP_MERGE=1). Once THIS attack and the rest of the run are on
disk, stitch the full set with:

  MERGE_ATTACK_TYPES_STR="${MERGE_ATTACK_TYPES[*]}" \\
  ROBUSTNESS_ATTACK_TYPES_STR="${ROBUSTNESS_ATTACK_TYPES[*]}" \\
  RESULTS_BASE="${RESULTS_BASE}" PLOTS_DIR="${PLOTS_DIR}" \\
  sbatch --export=ALL,CLUSTER_DIR=${CLUSTER_DIR} ${CLUSTER_DIR}/merge.slurm
EOF
else
  merge_jid=$(sbatch --parsable \
    --export=ALL,CLUSTER_DIR="${CLUSTER_DIR}" \
    --dependency="afterok:${shard_jid}" \
    "${CLUSTER_DIR}/merge.slurm")
  echo "Submitted merge job:       ${merge_jid}  (runs after ${shard_jid} succeeds)"
fi
echo
echo "Monitor:  squeue -u \$USER"
echo "If some shards fail, re-run them (they resume from saved state), then submit the"
if [[ "${SKIP_MERGE:-0}" == "1" ]]; then
  # A BARE merge would use config.sh's defaults, i.e. MERGE_ATTACK_TYPES = the attacks this
  # array computed -- for a fold-in that is the new attack ALONE, which would overwrite the
  # combined results with a single-attack merge. Always re-use the command printed above.
  echo "merge with the MERGE_ATTACK_TYPES_STR / ROBUSTNESS_ATTACK_TYPES_STR / RESULTS_BASE"
  echo "/ PLOTS_DIR shown above -- a bare merge.slurm would stitch only [${ATTACK_TYPES[*]}]."
else
  echo "merge alone:  sbatch --export=ALL,CLUSTER_DIR=${CLUSTER_DIR} ${CLUSTER_DIR}/merge.slurm"
fi
