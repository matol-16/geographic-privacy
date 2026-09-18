#!/usr/bin/env bash
#
# Run ONE shard of the configured experiment on the local GPU (no SLURM) -- the exact
# command eval_shard.slurm runs, built from config.sh (so the defaults are the full
# experiment; override with the same env knobs: DATASET, ATTACK_BUDGETS_STR, DATA_ROOT,
# RESULTS_BASE, NUM_RESTARTS, IMAGES_PER_SHARD, ...).
#
#   run_shard_local.sh <attack> [window_index=0]
#   run_shard_local.sh merge            # merge every shard under RESULTS_BASE (+ plots)
#
# e.g. the 100-image YFCC pilot of one attack at 1/255:
#   DATASET=yfcc ATTACK_BUDGETS_STR=0.0039 RESULTS_BASE=/Data/$USER/results/pilot ./run_shard_local.sh dtd

set -euo pipefail
WHAT="${1:?usage: run_shard_local.sh <attack|merge> [window_index]}"; WINDOW_IDX="${2:-0}"
export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${CLUSTER_DIR}/config.sh"
export MPLBACKEND=Agg
mkdir -p "${RESULTS_BASE}"
cd "${PROJECT_DIR}"

if [[ "${WHAT}" == "merge" ]]; then
  build_ablation_flags
  echo "merge: attacks=[${MERGE_ATTACK_TYPES[*]}] -> ${RESULTS_BASE}"
  exec python main.py merge-shards \
    --config "${CONFIG_PATH}" --dataset "${DATASET}" --attack-types "${MERGE_ATTACK_TYPES[@]}" \
    --total-images "${TOTAL_IMAGES}" --results-dir "${RESULTS_BASE}" --plots-dir "${PLOTS_DIR}" \
    "${ABLATION_FLAGS[@]}" "${OVERRIDE_FLAGS[@]}"
fi

build_ablation_flags "${WINDOW_IDX}"
echo "shard: attack=${WHAT} window=${WINDOW_IDX} images=[$(( WINDOW_IDX * IMAGES_PER_SHARD )):$(( (WINDOW_IDX + 1) * IMAGES_PER_SHARD ))) budgets=${BUDGETS_OVERRIDE} -> ${RESULTS_BASE}"
echo "  ${ABLATION_FLAGS[*]}"
echo "  ${OVERRIDE_FLAGS[*]}"
exec python main.py evaluate-dataset-shard \
  --config "${CONFIG_PATH}" --dataset "${DATASET}" --attack-types "${WHAT}" \
  --total-images "${TOTAL_IMAGES}" --images-per-shard "${IMAGES_PER_SHARD}" --image-shard-index "${WINDOW_IDX}" \
  --results-dir "${RESULTS_BASE}" --parallel-workers "${PARALLEL_WORKERS}" \
  "${ABLATION_FLAGS[@]}" "${OVERRIDE_FLAGS[@]}"
