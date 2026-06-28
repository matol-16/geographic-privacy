#!/usr/bin/env bash
#
# End-to-end test of every attack type on osv in a SINGLE evaluate-dataset run, with
# the restart ablation (always) and the sampling-steps ablation enabled.
#
# All attacks run together and overlay in one set of plots. `ace` is targeted; its
# target image / l2_target loss / alpha come from config.yaml
# (attack_train_args.<dataset>.per_attack.ace) and are applied to `ace` only.
#
# GeoShield is now folded into evaluate-dataset as just another attack: listing
# `geoshield` in --attack-types makes the run generate the GeoShield images
# out-of-process (Geoshield/geoshield.py) on the SAME seeded image selection and
# budgets, evaluate them with the shared backbone, and overlay them in the combined
# plots -- no separate generate/eval/re-plot steps and no geoshield_common.sh glue.
# GeoShield generation settings (clean_dir, output_base, steps, epsilons) live in the
# `geoshield:` block of config.yaml; epsilons default to round(budget * 255).
#
# Activate the env first:  conda activate plonk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="osv"
# ATTACK_TYPES=(encoder sampling diffusion_l2 dtd unidef ace geoshield)
ATTACK_TYPES=(geoshield encoder dtd)
N_IMAGES=5
RESULTS_DIR="${PROJECT_DIR}/results/cluster_eval/test_2206_v3"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1
EVAL_NUM_STEPS=(8 16 32 64 250)            # sampling-steps ablation grid

# Attack budgets are the single source of truth; GeoShield's epsilon is derived as
# round(budget * 255) (0.0314 -> 8/255) unless overridden in config.geoshield.epsilons.
ATTACK_BUDGETS=( 0.0314 )
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# Single pass: trainable attacks + GeoShield, all overlaid in the combined plots.
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
