#!/usr/bin/env bash
#
# Test the TargetedL2 attack (v2) on 100 YFCC images.
#
# v2 makes TargetedL2 an explicit replication of the TARGETED attack from
#   "Diffusion Policy Attacker: Crafting Adversarial Attacks for Diffusion-based Policies"
#   (Chen, Xue & Chen, NeurIPS 2024), Eq. 6 / Algorithm 1 (targeted branch).
#
# The paper's targeted objective (minimized) is
#   L_tar(I) = E_{k, eps_k} || eps_theta(tau_target + eps_k, k, I + delta) - eps_k ||^2
# i.e. forward-diffuse a chosen target action, then minimize the denoising error of the
# network conditioned on the perturbed image I + delta, so the sampler reconstructs the
# target. Here the "action" is the predicted GPS and the target is the point on Earth
# farthest from the image's real coordinates (its antipode). Each image's real GPS is
# threaded in automatically from the dataset labels; without it the attack falls back to
# the antipode of the clean model's mean prediction. The diffusion (DDPM) branch is a
# line-for-line match of Eq. 6; flow / Riemannian flow matching (the PLONK_YFCC default)
# use the corresponding true velocity as eps_k.
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_targeted_l2_attack_v2.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(targeted_l2)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_targeted_l2_v2"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314 0.0628)
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314,0.0628]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# The robustness ablation (JPEG compression + Gaussian blur) is scoped per attack type via
# robustness.attack_types, which by default does NOT include targeted_l2. So we enable it
# (--run-robustness-ablation) AND list targeted_l2 in --robustness-attack-types; otherwise
# no blur/compression is computed for it. Sampling-steps + model-transfer ablations are
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
