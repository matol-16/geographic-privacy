#!/usr/bin/env bash
#
# End-to-end test of every attack type on OSV (10 images x 2 budgets) in a SINGLE
# run, with the restart ablation (always) and the sampling-steps ablation enabled.
#
# All attacks run together and overlay in one set of plots. `ace` is targeted; its
# target image / l2_target loss / alpha come from config.yaml
# (attack_train_args.<dataset>.per_attack.ace) and are applied to `ace` only, so it
# runs alongside the untargeted attacks without affecting them.
#
# GeoShield is generated out-of-process (it is not a trainable attack type), so it
# runs after the main pass on the SAME seeded image selection and budgets, writes
# its results into the same RESULTS_DIR, and is then overlaid with the others in a
# final combined plot.
#
# Activate the env first:  conda activate plonk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/geoshield_common.sh"
PROJECT_DIR="$(repo_root)"
REPOS_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"   # holds both plonk/ and Geoshield/

# Edit these values to match the experiment you want to run.
CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="osv"
ATTACK_TYPES=(encoder sampling diffusion_l2 dtd unidef unidef_nofdje ace)
N_IMAGES=10
RESULTS_DIR="${PROJECT_DIR}/results/cluster_eval/test_2006"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1
EVAL_NUM_STEPS=(8 16 32 64 250)            # sampling-steps ablation grid

# Budgets are the single source of truth for both the backbone attacks and GeoShield.
# Each L-inf budget (0-1 scale) is paired with the matching GeoShield epsilon (0-255).
ATTACK_BUDGETS=( 0.0157 0.0314 )           # 4/255, 8/255
GEOSHIELD_EPSILONS=( 4 8 )                 # matching GeoShield epsilon (0-255 scale)
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0157,0.0314]" override string

# GeoShield generation settings (OSV).
GEOSHIELD_ATTACK_NAME="geoshield"
GEOSHIELD_STEPS=100
DATASET_ROOT="/Data/mathias.ollu/hf_cache/datasets/osv5m/images/test/00"
GEOSHIELD_CLEAN_DIR="/Data/mathias.ollu/hf_cache/clean_osv_images"
GEOSHIELD_OUTPUT_BASE="/Data/mathias.ollu/hf_cache/attacked_osv_images_geoshield"


export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# --- 1. Main pass: every trainable attack type, overlaid. ------------------- #
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

# --- 2. GeoShield: same seeded images & budgets, results into the same dir. -- #
geoshield_generate_and_eval \
  "$PROJECT_DIR" "$REPOS_ROOT" "$CONFIG_PATH" "$DATASET" "$DATASET_ROOT" \
  "$GEOSHIELD_CLEAN_DIR" "$GEOSHIELD_OUTPUT_BASE" "$N_IMAGES" "$GEOSHIELD_STEPS" \
  "$GEOSHIELD_ATTACK_NAME" "$RESULTS_DIR" "$PLOTS_DIR" ATTACK_BUDGETS GEOSHIELD_EPSILONS

# --- 3. Regenerate the combined plots so GeoShield overlays the other attacks. #
python main.py plot results \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${ATTACK_TYPES[@]}" "${GEOSHIELD_ATTACK_NAME}" \
  --results-dir "${RESULTS_DIR}" \
  --plots-dir "${PLOTS_DIR}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
