#!/usr/bin/env bash
#
# TikZ/pgfplots twin of plot_yfcc4k_full.sh: regenerates every plot for an
# already-computed full-dataset run (default: the yfcc4k_full cluster run)
# as standalone pgfplots .tex sources instead of matplotlib figures --
# purely from the saved .pt/.json results, no attacks re-run, no GPU needed.
#
# Every "${PYTHON} main.py plot ..." call below is the exact same command as
# in plot_yfcc4k_full.sh with --tikz appended; see that script for what each
# plot shows. Outputs land in "${PLOTS_DIR}/tikz/" (kept apart from the
# matplotlib PNG/PDF of the same name):
#   <stem>.tex           -- standalone pgfplots source (paper-ready, no titles;
#                            captions are written in LaTeX)
#   <stem>.pdf/.png       -- best-effort local compile + rasterized preview,
#                            produced automatically if pdflatex/pdftoppm and
#                            the pgfplots package are available; if not, the
#                            .tex is still written and will compile wherever
#                            pgfplots is available (e.g. Overleaf)
#   <stem>_<attack>.dat   -- external data tables for the two scatter plots
#                            (loss-vs-fsd, clean-vs-attacked-displacement),
#                            which can have thousands of points per attack
#
# World-map plots (plot_gps_samples_on_map / plot_gps_trajectories_on_map) are
# not reproduced here: they draw real coastlines/borders via cartopy, which
# has no straightforward TikZ equivalent. The overall-FSD ("plot results") and
# overall-success-rate ("plot success-rate") curves are also intentionally
# left out of this script (unlike plot_yfcc4k_full.sh) -- still available via
# "main.py plot results --tikz" / "plot success-rate --tikz" if needed.
#
# Usage:
#   ./plot_yfcc4k_full_tikz.sh                 # plot the default yfcc4k_full run
#   RESULTS_BASE=/path/to/other/run ./plot_yfcc4k_full_tikz.sh
#   ATTACK_TYPES="encoder dtd ace" ./plot_yfcc4k_full_tikz.sh

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
echo "Plotting saved results as TikZ/pgfplots (no recomputation)"
echo "Dataset:      ${DATASET}"
echo "Attacks:      ${ATTACK_TYPES}"
echo "Results dir:  ${RESULTS_BASE}"
echo "Plots dir:    ${PLOTS_DIR}/tikz"
echo "=============================================================="

cd "${PROJECT_DIR}"

CONFIG="${RESULTS_BASE}/${DATASET}_run_config_merged.yaml"

# # --- per-attack displacement variance ("DTD" spread), true-GPS metric ---
# "${PYTHON}" main.py plot dtd-variance --tikz \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types ${ATTACK_TYPES} \
#   --override plot.gps_true=true

# # --- training loss vs. achieved FSD (dtd/training_loss/sampling only; requires
# #     results saved with the 'final_loss' field -- older runs may not have it, so
# #     this step is allowed to fail without aborting the rest of the script) ---
# "${PYTHON}" main.py plot loss-vs-fsd --tikz \
#   --config "${CONFIG}" \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}" \
#   --attack-types dtd training_loss sampling \
#   --override plot.gps_true=true \
#   || echo "Skipping loss-vs-fsd plot (results predate the 'final_loss' field)"

# # --- clean-prediction-vs-truth displacement against perturbed-vs-truth displacement
# #     (true-GPS metric); shared log-log axes with a y=x "no attack effect" reference ---
# "${PYTHON}" main.py plot clean-vs-attacked-displacement --tikz \
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
"${PYTHON}" main.py evaluate-localizability --stage plot --tikz \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --results-dir "${RESULTS_BASE}" \
  --plots-dir "${PLOTS_DIR}" \
  --attack-types geoshield training_loss sampling diffusion_l2 dtd

# # --- robustness to JPEG / Gaussian blur (reads <dataset>_robustness_results.json;
# #     currently restricted to the 2500km success-rate threshold) ---
# "${PYTHON}" main.py plot robustness --tikz \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}"

# # --- cross-model transferability (reads <dataset>_model_transfer_results.json) ---
# "${PYTHON}" main.py plot model-transfer --tikz \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}"

# # --- success rate vs. sampling steps (reads <dataset>_sampling_steps_results.json) ---
# "${PYTHON}" main.py plot sampling-steps --tikz \
#   --dataset "${DATASET}" \
#   --results-dir "${RESULTS_BASE}" \
#   --plots-dir "${PLOTS_DIR}"

echo
echo "Done. TikZ sources (+ best-effort PDF/PNG previews) in ${PLOTS_DIR}/tikz."
