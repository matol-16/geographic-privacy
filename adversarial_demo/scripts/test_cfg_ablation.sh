#!/usr/bin/env bash
#
# Evaluate the DTD and Salman sampling attacks on 100 YFCC images and record the
# guidance-scale (cfg) ablation for both.
#
# The cfg ablation re-evaluates each image's best perturbation at several
# classifier-free guidance scales (--eval-cfgs, default 1 2 5 10) while holding the
# number of sampling steps at the baseline. It measures how the attack's displacement
# holds up as the deployed model's guidance strength changes. Like the sampling-steps
# ablation it is NOT scoped per attack type -- it runs for every evaluated attack --
# so both dtd and sampling get a cfg sweep in this run.
#
# The ablation is enabled by default via config.yaml (plot.run_cfg_ablation: true and
# eval_cfgs: [1, 2, 5, 10]); we still pass --run-cfg-ablation and --eval-cfgs explicitly
# so this run always sweeps guidance regardless of the config default.
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_cfg_ablation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(dtd sampling)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_cfg_ablation"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314)
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314]" override string

# Guidance scales swept by the cfg ablation.
EVAL_CFGS=(1 2 5 10)

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
  --run-cfg-ablation \
  --eval-cfgs "${EVAL_CFGS[@]}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
echo "cfg-ablation JSON: ${RESULTS_DIR}/${DATASET}_cfg_results.json"
echo "cfg-ablation plot: ${PLOTS_DIR}/${DATASET}_cfg_success_rate.{png,pdf}"
