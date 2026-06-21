#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(ace)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/baselines/results_ace_alpha0_50_8"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=4
USE_REAL_GPS=false
#model type: "" for RFM, "diffusion" for diffusion, "flow" for flow
MODEL_TYPE=""
# ACE target image (the image whose score/encoding we pull towards) and encoder-term weight.
# alpha=0 disables the encoder l2 term (ablation: target-score alignment only).
TARGET_IMAGE="/users/eleves-b/2023/mathias.ollu/repos/plonk/.media/MIST.png"
ALPHA=0
OVERRIDES=(
  "attack_budgets.yfcc=[0.0314]"  # 8/255
  "model_type=${MODEL_TYPE}"
  "attack_train_args.yfcc.dot_product_loss=l2_target"
  "attack_train_args.yfcc.target_image=${TARGET_IMAGE}"
  "attack_train_args.yfcc.alpha=${ALPHA}"
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
