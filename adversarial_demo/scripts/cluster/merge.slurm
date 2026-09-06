#!/usr/bin/env bash
#
# Merge job: stitches every per-shard output back into the full-dataset results,
# ablation JSONs and plots. No GPU required (it only loads tensors and plots), so
# the Jean Zay "prepost" CPU partition is ideal. Submitted by submit.sh with a
# --dependency=afterok on the shard array, or run by hand once all shards finished.
#
#SBATCH --job-name=advmerge
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
# ============================================================================== #
#  JEAN ZAY: FILL IN, then uncomment. Merge needs no GPU -- the 'prepost' partition
#  (CPU, with internet) is the right place for it.
# ============================================================================== #
##SBATCH --account=CHANGE_ME@cpu
##SBATCH --partition=prepost
# ============================================================================== #

set -euo pipefail

: "${CLUSTER_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${CLUSTER_DIR}/config.sh"

module purge 2>/dev/null || true
activate_env
export MPLBACKEND=Agg

build_ablation_flags   # -> ABLATION_FLAGS (must match the shards being stitched)

echo "Merging attacks: ${MERGE_ATTACK_TYPES[*]}"
echo "  ablations: ${ABLATION_FLAGS[*]}"
echo "  overrides: ${OVERRIDE_FLAGS[*]}"

cd "${PROJECT_DIR}"
python main.py merge-shards \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${MERGE_ATTACK_TYPES[@]}" \
  --total-images "${TOTAL_IMAGES}" \
  --results-dir "${RESULTS_BASE}" \
  --plots-dir "${PLOTS_DIR}" \
  "${ABLATION_FLAGS[@]}" \
  "${OVERRIDE_FLAGS[@]}"

echo "Merge done. Results in ${RESULTS_BASE}, plots in ${PLOTS_DIR}."
