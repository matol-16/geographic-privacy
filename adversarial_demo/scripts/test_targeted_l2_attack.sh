#!/usr/bin/env bash
#
# Test the TargetedL2 attack on 100 YFCC images.
#
# TargetedL2 is the targeted counterpart of the training_loss attack. Instead of
# *maximising* the model's training loss against its own clean prediction (untargeted),
# it *minimises* the L2 (MSE) training loss towards the velocity of a single fixed
# geographic point: the point on Earth farthest from the image's real coordinates --
# i.e. their antipode. Steering the whole diffusion/flow trajectory at that antipode
# drives the model to predict the farthest possible location. Each image's real GPS is
# threaded in automatically from the dataset labels (utils/datasets); without it the
# attack falls back to the antipode of the clean model's mean prediction.
#
# training_loss is included as the untargeted sibling so the two overlay in one set of
# plots (drop it from ATTACK_TYPES below to test TargetedL2 alone).
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_targeted_l2_attack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(targeted_l2)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_targeted_l2"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314 )
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314,0.0628]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# The robustness ablation (JPEG compression + Gaussian blur, GeoShield Fig. 6 levels) is
# scoped per attack type via robustness.attack_types, which by default does NOT include
# targeted_l2. So we must both enable it (--run-robustness-ablation) AND list targeted_l2
# in --robustness-attack-types, otherwise no blur/compression is computed for it. Each
# degraded image is re-evaluated at the baseline sampling-step count; transform levels come
# from the robustness: block of config.yaml. Sampling-steps + model-transfer ablations are
# already on by config and apply to every attack, so they run for targeted_l2 regardless.
python main.py evaluate-dataset \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${ATTACK_TYPES[@]}" \
  --n-images "${N_IMAGES}" \
  --results-dir "${RESULTS_DIR}" \
  --plots-dir "${PLOTS_DIR}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --run-robustness-ablation \
  --robustness-attack-types "${ATTACK_TYPES[@]}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
