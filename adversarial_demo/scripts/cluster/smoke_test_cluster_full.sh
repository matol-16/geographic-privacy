#!/usr/bin/env bash
#
# Local (no SLURM) end-to-end smoke test of the cluster_full run: runs the EXACT shard
# command of eval_shard.slurm for every attack on a tiny window, then merge-shards, then
# the paper post-processing -- with the attack / ablation grids shrunk so it finishes in
# minutes. Catches wiring, config and pre/post-processing errors before burning cluster
# hours. Everything (results, GeoShield I/O) goes under SMOKE_BASE.
#
#   ./smoke_test_cluster_full.sh yfcc [attacks...]     # default: all 11 attacks
#   SMOKE_IMAGES=3 ./smoke_test_cluster_full.sh osv dtd geoshield
#
# Env: PYTHON (python), DATA_ROOT, SMOKE_BASE, SMOKE_IMAGES (2), NUM_RESTARTS (2),
#      SKIP_POSTPROCESS (0).

set -euo pipefail

DS="${1:?usage: smoke_test_cluster_full.sh <yfcc|osv> [attacks...]}"; shift || true
export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python}"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/cluster_full/config.yaml}"
export DATASET="${DS}"
export TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"          # same seeded pool as the real run
export IMAGES_PER_SHARD="${SMOKE_IMAGES:-2}"          # ... but a 2-image window
export ATTACK_TYPES_STR="${*:-dtd diffusion_l2 targeted_l2 sampling sampling_cfg10 encoder training_loss unidef ace dtd_blur geoshield}"
export MERGE_ATTACK_TYPES_STR="${ATTACK_TYPES_STR}"
export ATTACK_BUDGETS_STR="${BUDGETS:-0.0039 0.0157 0.0314}"
export RUN_SAMPLING_STEPS_ABLATION=1 RUN_ROBUSTNESS_ABLATION=1 RUN_MODEL_TRANSFER_ABLATION=1 RUN_CFG_ABLATION=1
export EVAL_NUM_STEPS_STR="${EVAL_NUM_STEPS:-8 250}"
export EVAL_CFGS_STR="${EVAL_CFGS:-1 10}"
export ROBUSTNESS_ATTACK_TYPES_STR="${ATTACK_TYPES_STR}"
export NUM_RESTARTS="${NUM_RESTARTS:-2}"
export SAVE_DELTAS="${SAVE_DELTAS:-1}"
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DATA}"
export SMOKE_BASE="${SMOKE_BASE:-${PROJECT_DIR}/results/cluster_full_smoke}"
export RESULTS_BASE="${SMOKE_BASE}/${DS}_full"
export PLOTS_DIR="${RESULTS_BASE}/plots"
export GEOSHIELD_IO_DIR="${RESULTS_BASE}/geoshield_io"
# Shrink the attack training + GeoShield + robustness grids (everything else is the real config).
export EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES_STR:-attack_train_args.${DS}.n_steps=2 attack_train_args.${DS}.train_batch_size=8 attack_train_args.${DS}.anchor_samples=8 attack_train_args.${DS}.clean_num_steps=8 attack_train_args.${DS}.restart_eval_batch_size=16 geoshield.steps=2 robustness.jpeg_quality_factors=[80] robustness.gaussian_blur_sigmas=[0,4]}"

source "${CLUSTER_DIR}/config.sh"
export MPLBACKEND=Agg
mkdir -p "${RESULTS_BASE}"
cd "${PROJECT_DIR}"

echo "== smoke test: ${DS} attacks=[${ATTACK_TYPES[*]}] window=[0:${IMAGES_PER_SHARD}) results=${RESULTS_BASE}"
for ATTACK in "${ATTACK_TYPES[@]}"; do
  build_ablation_flags 0
  echo; echo "---- shard: ${ATTACK}"; echo "     ${ABLATION_FLAGS[*]}"; echo "     ${OVERRIDE_FLAGS[*]}"
  "${PYTHON}" main.py evaluate-dataset-shard \
    --config "${CONFIG_PATH}" --dataset "${DATASET}" --attack-types "${ATTACK}" \
    --total-images "${TOTAL_IMAGES}" --images-per-shard "${IMAGES_PER_SHARD}" --image-shard-index 0 \
    --results-dir "${RESULTS_BASE}" --parallel-workers 1 \
    "${ABLATION_FLAGS[@]}" "${OVERRIDE_FLAGS[@]}"
done

echo; echo "---- merge"
build_ablation_flags
"${PYTHON}" main.py merge-shards \
  --config "${CONFIG_PATH}" --dataset "${DATASET}" --attack-types "${MERGE_ATTACK_TYPES[@]}" \
  --total-images "${TOTAL_IMAGES}" --results-dir "${RESULTS_BASE}" --plots-dir "${PLOTS_DIR}" \
  "${ABLATION_FLAGS[@]}" "${OVERRIDE_FLAGS[@]}"

if [[ "${SKIP_POSTPROCESS:-0}" != 1 ]]; then
  echo; echo "---- post-processing (matplotlib only)"
  YFCC_BASE="${SMOKE_BASE}/yfcc_full" OSV_BASE="${SMOKE_BASE}/osv_full" TIKZ="${TIKZ:-0}" PYTHON="${PYTHON}" \
    "${CLUSTER_DIR}/postprocess_cluster_full.sh" "${DS}"
fi
echo; echo "SMOKE_TEST_OK ${DS}"
