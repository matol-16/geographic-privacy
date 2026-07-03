#!/usr/bin/env bash
#
# Regenerate every plot for an already-computed full-dataset run (default: the
# yfcc4k_full cluster run), purely from the saved .pt/.json results — no
# attacks are re-run and no GPU is needed.
#
# Produces, per attack in ATTACK_TYPES:
#   - main displacement + success-rate curves vs. budget       (plot results)
#   - success rate vs. budget                                  (plot success-rate)
#   - per-attack displacement variance box plot ("DTD" spread) (plot dtd-variance)
#   - localizability vs. attack strength (box plots)           (evaluate-localizability --stage plot)
#   - clean-vs-attacked true-GPS displacement scatter          (plot clean-vs-attacked-displacement)
#   - robustness to JPEG / Gaussian blur                       (plot robustness)
#   - cross-model transferability                              (plot model-transfer)
#   - success rate vs. sampling steps                           (plot sampling-steps)
#
# Distances/success rates are computed against the true image GPS wherever it's
# available (that's the default everywhere now); the saved run config for this
# particular run predates that default, so it's forced on here via --override.
# Robustness/model-transfer already plot both metrics side by side regardless.
#
# All ablation/variance plots are saved as both a high-dpi PNG and a vector PDF,
# plus a JSON sidecar with the underlying numbers.
#
# Usage:
#   ./plot_yfcc4k_full.sh                 # plot the default yfcc4k_full run
#   RESULTS_BASE=/path/to/other/run ./plot_yfcc4k_full.sh
#   ATTACK_TYPES="encoder dtd ace" ./plot_yfcc4k_full.sh

set -euo pipefail

DATASET="${DATASET:-yfcc}"
ATTACK_TYPES="${ATTACK_TYPES:-encoder sampling diffusion_l2 dtd ace geoshield training_loss unidef}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
# Use the plonk conda env's python (the base env lacks plonk); override with PYTHON=...
PYTHON="${PYTHON:-/Data/mathias.ollu/conda/plonk/bin/python}"
RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/${DATASET}4k_full}"
PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"

echo "=============================================================="
echo "Plotting saved results (no recomputation)"
echo "Dataset:      ${DATASET}"
echo "Attacks:      ${ATTACK_TYPES}"
echo "Results dir:  ${RESULTS_BASE}"
echo "Plots dir:    ${PLOTS_DIR}"
echo "=============================================================="

cd "${PROJECT_DIR}"

CONFIG="${RESULTS_BASE}/${DATASET}_run_config_merged.yaml"

# --- main displacement + success-rate curves vs. budget (true-GPS metric) ---
# "${PYTHON}" main.py plot results \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types ${ATTACK_TYPES} \
#   --override plot.gps_true=true

# # --- success rate vs. budget (true-GPS metric) ---
# "${PYTHON}" main.py plot success-rate \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types ${ATTACK_TYPES} \
#   --override plot.gps_true=true

# # --- per-attack displacement variance ("DTD" spread), true-GPS metric ---
# "${PYTHON}" main.py plot dtd-variance \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types ${ATTACK_TYPES} \
#   --override plot.gps_true=true

# --- training loss vs. achieved FSD (dtd/training_loss/sampling only; requires
#     results saved with the 'final_loss' field -- older runs may not have it, so
#     this step is allowed to fail without aborting the rest of the script) ---
# "${PYTHON}" main.py plot loss-vs-fsd \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types dtd training_loss sampling \
#   --override plot.gps_true=true \
#   || echo "Skipping loss-vs-fsd plot (results predate the 'final_loss' field)"

# --- clean-prediction-vs-truth displacement against perturbed-vs-truth displacement
#     (true-GPS metric); shared log-log axes with a y=x "no attack effect" reference ---
# "${PYTHON}" main.py plot clean-vs-attacked-displacement \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types ${ATTACK_TYPES} \
#   --override plot.gps_true=true

# --- localizability vs. attack strength (box plots per attack, low/med/high
#     localizability tertiles). Attack types are listed explicitly here rather
#     than left to default to the run config's `attack_types` (which is missing
#     `dtd` for this run and would silently drop its panel) ---
# "${PYTHON}" main.py evaluate-localizability --stage plot \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types encoder sampling diffusion_l2 dtd ace training_loss

# --- robustness to JPEG / Gaussian blur (reads <dataset>_robustness_results.json;
#     always plots both predicted and true metrics, solid vs. dashed) ---
"${PYTHON}" main.py plot robustness \
  --dataset "${DATASET}" \
  --results-dir "${RESULTS_BASE}" \
  --plots-dir "${PLOTS_DIR}"

# # --- cross-model transferability (reads <dataset>_model_transfer_results.json) ---
# "${PYTHON}" main.py plot model-transfer \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}"

# # --- success rate vs. sampling steps (reads <dataset>_sampling_steps_results.json) ---
# "${PYTHON}" main.py plot sampling-steps \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}"

echo
echo "Done. Plots in ${PLOTS_DIR}."
