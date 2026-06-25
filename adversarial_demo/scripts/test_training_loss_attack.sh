#!/usr/bin/env bash
#
# Test the training_loss attack on 200 YFCC images with restart and
# sampling-steps ablations.
#
# The training_loss attack maximises the model's own training objective
# (||network(x_t, emb(I+delta), gamma) - v_true||^2), adapting the forward
# process and true target to the model parameterisation (diffusion / flow /
# Riemannian flow matching).
#
# Activate the env first:  conda activate plonk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(training_loss)
N_IMAGES=200
RESULTS_DIR="${PROJECT_DIR}/results/test_training_loss"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1
EVAL_NUM_STEPS=(8 16 32 64 250)

ATTACK_BUDGETS=(0.0314 0.0628)
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"

export MPLBACKEND=Agg

cd "${PROJECT_DIR}"

python main.py evaluate-dataset \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${ATTACK_TYPES[@]}" \
  --n-images "${N_IMAGES}" \
  --results-dir "${RESULTS_DIR}" \
  --plots-dir "${PLOTS_DIR}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --run-sampling-steps-ablation \
  --eval-num-steps "${EVAL_NUM_STEPS[@]}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
