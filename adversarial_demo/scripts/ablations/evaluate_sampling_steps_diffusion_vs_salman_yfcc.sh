#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(diffusion_l2 dtd sampling encoder)
N_IMAGES=50
EVAL_NUM_STEPS=(8 16 32 64 128 250 512)
RESULTS_DIR="${PROJECT_DIR}/results/ablations/results_sampling_steps_dtd_l2_cosneg_salman_encoder"
PLOTS_DIR="${RESULTS_DIR}/plots"
OVERRIDES=(
  "attack_budgets.yfcc=[0.03137]"
)


cd "${PROJECT_DIR}"

cmd=(
  python main.py evaluate-sampling-steps
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --n-images "${N_IMAGES}"
  --results-dir "${RESULTS_DIR}"
  --plots-dir "${PLOTS_DIR}"
  --eval-num-steps "${EVAL_NUM_STEPS[@]}"
)

if ((${#ATTACK_TYPES[@]})); then
  cmd+=(--attack-types "${ATTACK_TYPES[@]}")
fi

for override in "${OVERRIDES[@]}"; do
  [[ -n "${override}" ]] && cmd+=(--override "${override}")
done

exec "${cmd[@]}"
