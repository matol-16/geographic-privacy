#!/usr/bin/env bash
#
# Paper figures + tables for a finished cluster_full run (CPU only; reads the merged .pt /
# .json files written by merge.slurm). Run after the merge job(s) succeeded:
#
#   ./postprocess_cluster_full.sh yfcc          # results/cluster_full/yfcc_full
#   ./postprocess_cluster_full.sh osv
#   ./postprocess_cluster_full.sh all
#
# Produces, per dataset, under <RESULTS_BASE>/plots (main) and plots_appendix:
#   * FSD / success-rate vs budget for every attack, per-attack spread,
#   * clean-vs-attacked FSD heatmaps (headline attacks; all attacks in the appendix),
#   * sampling-steps, cfg (guidance scale), robustness (JPEG / blur), model-transfer and
#     restart ablation figures (+ TikZ twins where implemented),
#   * results/tables/: the success-rate / FSD table (LaTeX + Markdown) over both datasets
#     when both runs exist.
# Env: TIKZ (1), CLEAN (0), PYTHON, YFCC_BASE / OSV_BASE (override result dirs), TABLES_DIR.

set -euo pipefail

WHICH="${1:-all}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/../.." && pwd)"
cd "${PROJECT_DIR}"

export YFCC_BASE="${YFCC_BASE:-${PROJECT_DIR}/results/cluster_full/yfcc_full}"
export OSV_BASE="${OSV_BASE:-${PROJECT_DIR}/results/cluster_full/osv_full}"
export PYTHON="${PYTHON:-python}"
export TIKZ="${TIKZ:-1}"
export CLEAN="${CLEAN:-0}"
export MPLBACKEND=Agg

# Attack keys of this run (canonical; the config pinned each to its latest implementation).
export ALL_ATTACKS="${ALL_ATTACKS:-geoshield unidef training_loss encoder sampling diffusion_l2 targeted_l2 dtd_blur dtd}"
export MAIN_HEATMAP_ATTACKS="${MAIN_HEATMAP_ATTACKS:-geoshield encoder training_loss targeted_l2 dtd}"
export LOCALIZABILITY_ATTACKS="${LOCALIZABILITY_ATTACKS:-geoshield training_loss sampling diffusion_l2 dtd}"
export DPA_PILOT_DIR=""   # targeted_l2 is part of this run; never fall back to the pilot

scripts/make_paper_figures.sh "${WHICH}"

# Main results table (needs the plot sidecars written above).
runs=()
[[ "${WHICH}" != "osv"  && -d "${YFCC_BASE}/plots" ]] && runs+=(--run "yfcc=${YFCC_BASE}/plots")
[[ "${WHICH}" != "yfcc" && -d "${OSV_BASE}/plots"  ]] && runs+=(--run "osv=${OSV_BASE}/plots")
if (( ${#runs[@]} )); then
  "${PYTHON}" scripts/make_success_rate_table.py --out-dir "${TABLES_DIR:-$(dirname "${YFCC_BASE}")/tables}" "${runs[@]}"
fi
echo "Post-processing done."
