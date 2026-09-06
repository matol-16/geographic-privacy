#!/usr/bin/env bash
#
# Rebuild the COMBINED multi-attack figures for a dataset purely from the saved
# <dataset>_<attack>_results.pt files -- no attacks re-run, no GPU. Use it to fold a
# newly computed attack (e.g. targeted_l2) into the combined figures when the other
# attacks' shards are no longer on disk (the OSV case), so a full re-merge is not an
# option. Every attack passed here just needs its <dataset>_<attack>_results.pt present
# in RESULTS_BASE.
#
# Regenerates: main displacement + success-rate curves (plot results), success rate vs
# budget (plot success-rate), per-attack displacement spread (plot dtd-variance), and
# the clean-vs-attacked displacement scatter (plot clean-vs-attacked-displacement). All
# against the true image GPS (that's what the existing combined OSV plots use).
#
# Usage:
#   ./replot_combined.sh <yfcc|osv> "<space separated attack list>" [RESULTS_BASE] [PLOTS_DIR]
#
# Example (OSV: existing six attacks + targeted_l2, integrated in place):
#   ./replot_combined.sh osv "encoder sampling dtd ace training_loss unidef targeted_l2" \
#       results/cluster_eval/osv_5k_full
#
# Env: PYTHON=/path/to/plonk/python to use a specific interpreter (skips conda
#      activation); otherwise the plonk conda env is activated via config.sh.

set -euo pipefail

DATASET="${1:?usage: replot_combined.sh <yfcc|osv> \"<attack list>\" [RESULTS_BASE] [PLOTS_DIR]}"
ATTACKS="${2:?quoted, space-separated attack list, e.g. \"encoder dtd targeted_l2\"}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config.yaml}"
RESULTS_BASE="${3:-${PROJECT_DIR}/results/cluster_eval/${DATASET}_5k_full}"
PLOTS_DIR="${4:-${RESULTS_BASE}/plots}"
GPS_TRUE="${GPS_TRUE:-true}"

# Interpreter: explicit PYTHON wins; otherwise activate the plonk conda env.
if [[ -z "${PYTHON:-}" ]]; then
  # shellcheck source=/dev/null
  source "${HERE}/config.sh"      # provides activate_env + CONDA_ENV
  activate_env
  PYTHON="python"
fi

export MPLBACKEND=Agg
cd "${PROJECT_DIR}"

echo "=============================================================="
echo "Rebuilding combined plots (no recomputation, no GPU)"
echo "Dataset:      ${DATASET}"
echo "Attacks:      ${ATTACKS}"
echo "Results dir:  ${RESULTS_BASE}"
echo "Plots dir:    ${PLOTS_DIR}"
echo "=============================================================="

# shellcheck disable=SC2086  # word-splitting ${ATTACKS} into --attack-types is intended
for PLOT in results success-rate dtd-variance clean-vs-attacked-displacement; do
  echo ">>> plot ${PLOT}"
  "${PYTHON}" main.py plot "${PLOT}" \
    --config "${CONFIG_PATH}" \
    --dataset "${DATASET}" \
    --results-dir "${RESULTS_BASE}" \
    --plots-dir "${PLOTS_DIR}" \
    --attack-types ${ATTACKS} \
    --override "plot.gps_true=${GPS_TRUE}"
done

echo
echo "Done. Combined plots in ${PLOTS_DIR}."
