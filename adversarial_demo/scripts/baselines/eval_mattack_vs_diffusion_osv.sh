#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo

# Edit these values to match the mattack evaluation you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="osv"
ATTACK_NAME="mattack"
ATTACK_BUDGETS=( 0.0314 ) #8, 16/255, can be written ( 0.0314 0.0628 )
#image dirs can be set one after the other (under the other)
CLEAN_IMAGE_DIRS=(
  "/Data/mathias.ollu/hf_cache/clean_osv_images"
)
ATTACKED_IMAGE_DIRS=(
  "/Data/mathias.ollu/hf_cache/attacked_osv_images_mattack_e_8/img/692e93e6f753f2a265c37b8c41447630target/clean_osv_images"
)
RESULTS_DIR="${PROJECT_DIR}/results/results_0406_mattack"
PLOTS_DIR="${PROJECT_DIR}/results/results_0406_mattack/plots"
N_IMAGES=100
OVERRIDES=(
  # "plot.plot_success_rate=true"
  "plot.attack_success_rate_thresholds=[200, 750, 2500]"
)

cd "${PROJECT_DIR}"

cmd=(
  python main.py evaluate-geoshield-vs-diffusion
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --attack-name "${ATTACK_NAME}"
  --results-dir "${RESULTS_DIR}"
  --plots-dir "${PLOTS_DIR}"
  --n-images "${N_IMAGES}"
  --attack-budgets "${ATTACK_BUDGETS[@]}"
  --clean-image-dirs "${CLEAN_IMAGE_DIRS[@]}"
  --attacked-image-dirs "${ATTACKED_IMAGE_DIRS[@]}"
)

for override in "${OVERRIDES[@]}"; do
  [[ -n "${override}" ]] && cmd+=(--override "${override}")
done

exec "${cmd[@]}"
