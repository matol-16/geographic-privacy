#!/usr/bin/env bash
#
# Test the DTDBlur attack on 100 YFCC images, WITH the blur robustness ablation.
#
# DTDBlur is a blur-robust variant of DTD (Diffusion Trajectory Deviation). Plain DTD
# trains the perturbation only on the clean protected image, so blurring the protected
# image can wash the perturbation out. DTDBlur adds an expectation-over-transformation
# objective: each step the DTD loss is evaluated on both the clean protected image and a
# Gaussian-blurred copy at a random sigma in [0, 4], combined half/half. The blur is
# differentiable, so the gradient reaches the perturbation through it.
#
# We run plain `dtd` alongside as a baseline and scope the robustness ablation to BOTH,
# so the resulting robustness plots overlay dtd vs dtd_blur under JPEG/blur degradation --
# i.e. the blur robustness is tested as part of this launch.
#
# The robustness ablation is scoped per attack type (robustness.attack_types), which by
# default excludes both of these, so we pass --run-robustness-ablation together with
# --robustness-attack-types; otherwise no blur/compression would be computed for them.
# Blur sigmas / JPEG qualities come from the robustness: block of config.yaml.
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_dtd_blur_attack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
# dtd_blur is the attack under test; dtd is the (non-blur-trained) baseline to compare
# blur robustness against. Drop dtd to test dtd_blur alone.
ATTACK_TYPES=(dtd_blur)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_dtd_blur"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314 )
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314,0.0628]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

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
echo "Blur robustness: see ${PLOTS_DIR}/yfcc_robustness_results.png and ${RESULTS_DIR}/yfcc_robustness_results.json"
