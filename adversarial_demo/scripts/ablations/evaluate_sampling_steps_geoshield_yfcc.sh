#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
PROJECT_DIR=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_NAME="geoshield"
ATTACK_BUDGETS=( 0.0157 0.0314 0.0628 )
EVAL_NUM_STEPS=(8 16 32 64 128 250 512)
# One clean dir per budget (can repeat the same dir if images are shared).
CLEAN_IMAGE_DIRS=(
  "/Data/mathias.ollu/hf_cache/clean_yfcc_images"
  "/Data/mathias.ollu/hf_cache/clean_yfcc_images"
  "/Data/mathias.ollu/hf_cache/clean_yfcc_images"
)
ATTACKED_IMAGE_DIRS=(
  "/Data/mathias.ollu/hf_cache/attacked_yfcc_images_geoshield_e_4/img/a4878f6e76f7a7c026a766609df999a1+geoshield/clean_yfcc_images"
  "/Data/mathias.ollu/hf_cache/attacked_yfcc_images_geoshield/img/9a2838fa9e42a4879397d1c90ae0ab69+geoshield/clean_yfcc_images"
  "/Data/mathias.ollu/hf_cache/attacked_yfcc_images_geoshield_e_16/img/058ab7f16c8d76800fbf2da75d2b2e14+geoshield/clean_yfcc_images"
)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/ablations/results_sampling_steps_geoshield"
PLOTS_DIR="${RESULTS_DIR}/plots"
OVERRIDES=(
  "plot.attack_success_rate_thresholds=[200, 750, 2500]"
)


cd "${PROJECT_DIR}"

cmd=(
  python main.py evaluate-sampling-steps-precomputed
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --attack-name "${ATTACK_NAME}"
  --attack-budgets "${ATTACK_BUDGETS[@]}"
  --clean-image-dirs "${CLEAN_IMAGE_DIRS[@]}"
  --attacked-image-dirs "${ATTACKED_IMAGE_DIRS[@]}"
  --eval-num-steps "${EVAL_NUM_STEPS[@]}"
  --n-images "${N_IMAGES}"
  --results-dir "${RESULTS_DIR}"
  --plots-dir "${PLOTS_DIR}"
)

for override in "${OVERRIDES[@]}"; do
  [[ -n "${override}" ]] && cmd+=(--override "${override}")
done

exec "${cmd[@]}"
