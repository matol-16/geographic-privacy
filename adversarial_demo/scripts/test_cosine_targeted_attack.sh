#!/usr/bin/env bash
#
# Test the CosineTargeted attack on 100 YFCC images.
#
# CosineTargeted is a direction-only variant of TargetedL2 (which replicates DP-Attacker's
# targeted noise-prediction loss, Eq. 6). It keeps the SAME target -- the point on Earth
# farthest from the image's real coordinates (its antipode) -- and the same forward process,
# but instead of matching the network's prediction to the target velocity in L2, it minimizes
# the NEGATIVE cosine similarity between them:
#
#   min_delta  E[ -cos( network(x_t, emb(I + delta), gamma), v_target ) ]
#
# i.e. it steers the sampling flow to POINT towards the antipode without constraining its
# magnitude (scale-invariant). Each image's real GPS is threaded in automatically from the
# dataset labels; without it the target falls back to the antipode of the clean model's mean
# prediction.
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_cosine_targeted_attack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(cosine_targeted)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_cosine_targeted"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314 0.0628)
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314,0.0628]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# The robustness ablation (JPEG compression + Gaussian blur) is scoped per attack type via
# robustness.attack_types, which by default does NOT include cosine_targeted. So we enable it
# (--run-robustness-ablation) AND list cosine_targeted in --robustness-attack-types; otherwise
# no blur/compression is computed for it. Sampling-steps + model-transfer ablations are already
# on by config and apply to every attack, so they run for cosine_targeted regardless.
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
