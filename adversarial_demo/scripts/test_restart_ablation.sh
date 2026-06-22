#!/usr/bin/env bash
#
# Small end-to-end test of the number-of-restarts ablation: a SINGLE attack (dtd) on
# YFCC, 5 images, one budget.
#
# --max-restarts sets the ablation depth: the attack trains MAX_RESTARTS restarts so the
# restart ablation (always produced) reaches that depth, while the reported/plotted MAIN
# results are selected over only the configured num_restarts
# (attack_train_args.<dataset>.num_restarts in config.yaml). Pick MAX_RESTARTS > that
# num_restarts so the split is visible: the main displacement/success-rate plots reflect
# num_restarts, and the restart-ablation curve (best-after-k) extends out to MAX_RESTARTS.
#
# Activate the env first:  conda activate plonk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(dtd encoder)
N_IMAGES=10
RESULTS_DIR="${PROJECT_DIR}/results/ablations/test_restart_ablation"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1
MAX_RESTARTS=8                             # ablation depth (> config num_restarts to see the split)

# Single budget; it is the single source of truth for the run.
ATTACK_BUDGETS=( 0.0314 0.0628 )                   # 8/255
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314]" override string

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
  --max-restarts "${MAX_RESTARTS}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
echo "Check ${DATASET}_restarts_results.json / the restart-ablation plot: best-after-k should"
echo "extend to ${MAX_RESTARTS}, while the main displacement plots reflect config num_restarts."
