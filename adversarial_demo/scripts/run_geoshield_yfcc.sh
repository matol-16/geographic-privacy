#!/usr/bin/env bash
#
# Generate GeoShield adversarial images on YFCC and evaluate them with the SAME
# backbone as every other attack (seeded image selection + evaluate-geoshield-vs-
# diffusion). The heavy lifting lives in geoshield_common.sh so this script and
# test_all_attacks.sh stay in sync.
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
DATASET="yfcc"
ATTACK_NAME="geoshield"

# YFCC4k dataset root (must match config.data_dirs.yfcc); images live in <root>/images.
DATASET_ROOT="/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k"
CLEAN_IMAGES_PATH="/Data/mathias.ollu/hf_cache/clean_yfcc_images"
OUTPUT_BASE="/Data/mathias.ollu/hf_cache/attacked_yfcc_images_geoshield"

N_IMAGES_TO_EVAL=100
STEPS=100

# Parallel arrays: each L-inf budget (0-1 scale, for the evaluator) is paired with
# the matching GeoShield epsilon (integer, 0-255 scale). 0.0157~=4/255, 0.0314~=8/255.
ATTACK_BUDGETS=( 0.0157 0.0314 )
EPSILONS=( 4 8 )

RESULTS_DIR="${PROJECT_DIR}/results/cluster_eval/test_2006/geoshield"
PLOTS_DIR="${RESULTS_DIR}/plots"

export MPLBACKEND=Agg   # headless plotting

geoshield_generate_and_eval \
  "$PROJECT_DIR" "$REPOS_ROOT" "$CONFIG_PATH" "$DATASET" "$DATASET_ROOT" \
  "$CLEAN_IMAGES_PATH" "$OUTPUT_BASE" "$N_IMAGES_TO_EVAL" "$STEPS" "$ATTACK_NAME" \
  "$RESULTS_DIR" "$PLOTS_DIR" ATTACK_BUDGETS EPSILONS

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
