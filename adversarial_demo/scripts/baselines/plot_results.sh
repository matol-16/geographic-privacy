#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the results you want to visualize.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(geoshield diffusion)
RESULTS_DIR="${PROJECT_DIR}/results/results_2405_cosine_sim"
PLOTS_DIR="${RESULTS_DIR}/plots_0306"
PLOT_TYPE="results"
OVERRIDES=(
  # "plot.gps_true=true"
)

cd "${PROJECT_DIR}"

cmd=(
  python main.py plot "${PLOT_TYPE}"
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --results-dir "${RESULTS_DIR}"
  --plots-dir "${PLOTS_DIR}"
)

if ((${#ATTACK_TYPES[@]})); then
  cmd+=(--attack-types "${ATTACK_TYPES[@]}")
fi

for override in "${OVERRIDES[@]}"; do
  [[ -n "${override}" ]] && cmd+=(--override "${override}")
done

exec "${cmd[@]}"
