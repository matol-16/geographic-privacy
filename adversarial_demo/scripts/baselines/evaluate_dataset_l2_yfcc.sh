#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(diffusion)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/results_0706_l2_lr=0.005"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=4
USE_REAL_GPS=false
OVERRIDES=(
  "attack_train_args.yfcc.dot_product_loss=l2"
  "attack_train_args.yfcc.lr=0.005"
  "attack_budgets.yfcc=[0.0314]"
  # "device=cpu"
  "n_images_to_eval=50"
  # "use_cuda_streams=false"
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
