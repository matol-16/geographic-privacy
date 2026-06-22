#!/usr/bin/env bash
# Plot a joint sampling-steps sensitivity comparison from pre-existing JSON result files.
# Run evaluate_sampling_steps_diffusion_vs_salman_yfcc.sh and
# evaluate_sampling_steps_geoshield_yfcc.sh first to produce the result files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
PROJECT_DIR=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
PLOTS_DIR="${PROJECT_DIR}/results/ablations/plots_sampling_steps_comparison"

# Edit these paths to point to the JSON files produced by the evaluation scripts.
RESULTS_FILES=(
  "${PROJECT_DIR}/results/ablations/results_sampling_steps_diffusion_vs_salman/yfcc_sampling_steps_results.json"
  "${PROJECT_DIR}/results/ablations/results_sampling_steps_geoshield/yfcc_geoshield_sampling_steps_results.json"
)


cd "${PROJECT_DIR}"

exec python main.py plot sampling-steps \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --plots-dir "${PLOTS_DIR}" \
  --results-files "${RESULTS_FILES[@]}"
