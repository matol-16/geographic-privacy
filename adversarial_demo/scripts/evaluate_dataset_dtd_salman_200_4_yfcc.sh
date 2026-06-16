#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(diffusion_salman diffusion)
N_IMAGES=200
RESULTS_DIR="${PROJECT_DIR}/results/results_dtd_salman_200_4"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=4
USE_REAL_GPS=false
#model type: "" for RFM, "diffusion" for diffusion, "flow" for flow
MODEL_TYPE="" 
OVERRIDES=(
#   "attack_budgets.yfcc=[0.0157, 0.0314]"
  "attack_budgets.yfcc=[0.0157]"
  "model_type=${MODEL_TYPE}"
)


cd "${PROJECT_DIR}"

cmd=(
  python main.py evaluate-dataset
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --n-images "${N_IMAGES}"
  --results-dir "${RESULTS_DIR}"
  --plots-dir "${PLOTS_DIR}"
  --parallel-workers "${PARALLEL_WORKERS}"
)

if ((${#ATTACK_TYPES[@]})); then
  cmd+=(--attack-types "${ATTACK_TYPES[@]}")
fi

if [[ "${USE_REAL_GPS}" == true ]]; then
  cmd+=(--use-real-gps)
fi

for override in "${OVERRIDES[@]}"; do
  [[ -n "${override}" ]] && cmd+=(--override "${override}")
done

exec "${cmd[@]}"