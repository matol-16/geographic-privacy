#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(encoder diffusion)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results"
PLOTS_DIR="${PROJECT_DIR}/plots"
OVERRIDES=(
  # "device=cpu"
  # "n_images_to_eval=50"
)

cd "${PROJECT_DIR}"

cmd=(
  python main.py evaluate-localizability
  --config "${CONFIG_PATH}"
  --dataset "${DATASET}"
  --n-images "${N_IMAGES}"
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
