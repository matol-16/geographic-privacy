#!/usr/bin/env bash
#
# Regenerate every paper figure (main + appendix) from the saved full-run results --
# no GPU, no attacks re-run, just `main.py plot` over the merged .pt / .json files.
#
# Layout it produces, per dataset:
#
#   <BASE>/plots/                    main-paper figures (+ tikz/ twins)
#   <BASE>/plots_appendix/           appendix figures, every stem ends in "_appendix"
#
# with
#   YFCC4k  BASE = results/cluster_eval/yfcc4k_full     (main results + all ablations)
#   OSV     BASE = results/cluster_eval_2/osv_full      (main results; ablations -> appendix)
#
# What goes where:
#   * main results (displacement / success rate vs budget, per-attack spread, and the
#     clean-vs-attacked heatmap restricted to the headline attacks on one line) -> plots/
#     for BOTH datasets;
#   * ablations (sampling steps, guidance scale, model transfer, robustness, restarts,
#     localizability) -> plots/ on YFCC4k, plots_appendix/ on OSV;
#   * appendix-only variants on BOTH datasets: the robustness figures with all three
#     success-rate thresholds (main paper keeps 2500 km only) and the heatmap over every
#     attack (main paper: GeoShield, GeoFA-enc., GeoFA-AdvDM, GeoFA-DPA, GeoFA-TD).
#
# Attacks whose <ds>_<attack>_results.pt is not on disk yet (targeted_l2 until its
# cluster run lands) are skipped with a notice, so the attack lists below can already
# name them.
#
# Usage:
#   scripts/make_paper_figures.sh            # both datasets
#   scripts/make_paper_figures.sh yfcc       # one dataset
#   TIKZ=0 scripts/make_paper_figures.sh     # matplotlib only (fast preview, no pdflatex)
#   CLEAN=1 scripts/make_paper_figures.sh    # wipe plots/ + plots_appendix/ first
#
# Env knobs: PYTHON, TIKZ (default 1), CLEAN (default 0), YFCC_BASE, OSV_BASE,
#            ALL_ATTACKS, MAIN_HEATMAP_ATTACKS, LOCALIZABILITY_ATTACKS, EPS.
#
# EPS (e.g. 0.0039 for 1/255): draw the single-budget figures (ablations, spread,
# heatmaps) at that budget instead of the default 8/255, into plots_eps<N>/ and
# plots_appendix_eps<N>/ (N = EPS*255). The figures that already span every budget
# (displacement / success rate vs budget, restarts) are skipped in that mode.
#   EPS=0.0039 scripts/make_paper_figures.sh yfcc
#
# Each .pt-backed plot re-loads every attack's results file (~30 s per attack), so a
# full run is ~45 min per dataset; run the two datasets as two processes to halve it:
#   scripts/make_paper_figures.sh yfcc & scripts/make_paper_figures.sh osv & wait

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
cd "${PROJECT_DIR}"

WHICH="${1:-all}"
PYTHON="${PYTHON:-/Data/mathias.ollu/conda/plonk/bin/python}"
TIKZ="${TIKZ:-1}"
CLEAN="${CLEAN:-0}"
EPS="${EPS:-}"
YFCC_BASE="${YFCC_BASE:-${PROJECT_DIR}/results/cluster_eval/yfcc4k_full}"
OSV_BASE="${OSV_BASE:-${PROJECT_DIR}/results/cluster_eval_2/osv_full}"

# Every attack in the combined curves/tables. targeted_l2 (GeoFA-DPA) is listed so it
# folds in automatically once its results exist.
ALL_ATTACKS="${ALL_ATTACKS:-encoder sampling diffusion_l2 dtd ace geoshield training_loss unidef targeted_l2}"
# Main-paper attack subset for the figures that do not show every attack (heatmap,
# localizability, model transfer): GeoShield, GeoFA-sample, GeoFA-AdvDM, GeoFA-DPA, GeoFA-TD. Each of
# those figures also gets an all-attacks "_appendix" twin.
MAIN_ATTACKS="${MAIN_ATTACKS:-geoshield sampling training_loss targeted_l2 dtd}"
# Main-paper heatmap: the main attacks, single row (dtd is forced last by the plot code).
MAIN_HEATMAP_ATTACKS="${MAIN_HEATMAP_ATTACKS:-${MAIN_ATTACKS}}"
# Until the full targeted_l2 (GeoFA-DPA) run lands, its heatmap panel comes from the
# 100-image YFCC pilot (the panel title states n = 100). Used only when the dataset has
# no <ds>_targeted_l2_results.pt of its own; set DPA_PILOT_DIR="" to disable.
DPA_PILOT_DIR="${DPA_PILOT_DIR-${PROJECT_DIR}/results/test/test_targeted_l2}"
# Localizability panel columns.
LOCALIZABILITY_ATTACKS="${LOCALIZABILITY_ATTACKS:-${MAIN_ATTACKS}}"

export MPLBACKEND=Agg

EPS_ARGS=() PLOTS_TAG=""
if [[ -n "${EPS}" ]]; then
  EPS_ARGS=(--eps "${EPS}")
  PLOTS_TAG="_eps$("${PYTHON}" -c "print(round(${EPS} * 255))")"
fi

# Backends to render: matplotlib always, plus the pgfplots twin when TIKZ=1.
FLAG_SETS=("")
[[ "${TIKZ}" == "1" ]] && FLAG_SETS+=("--tikz")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# plot <type> <dataset> <results_dir> <plots_dir> <config> [extra main.py args...]
plot() {
  local type="$1" ds="$2" rdir="$3" pdir="$4" cfg="$5"; shift 5
  echo; echo ">>> [${ds}] plot ${type} -> ${pdir}  ($*)"
  "${PYTHON}" main.py plot "${type}" --config "${cfg}" --dataset "${ds}" \
    --results-dir "${rdir}" --plots-dir "${pdir}" ${EPS_ARGS[@]+"${EPS_ARGS[@]}"} "$@"
}

# Same, once per backend in FLAG_SETS.
plot_both() {
  local flags
  for flags in "${FLAG_SETS[@]}"; do
    # shellcheck disable=SC2086
    plot "$@" ${flags}
  done
}

# present_attacks <dataset> <results_dir> <attacks...>: those with a results file on disk.
present_attacks() {
  local ds="$1" rdir="$2" a out=(); shift 2
  for a in "$@"; do [[ -f "${rdir}/${ds}_${a}_results.pt" ]] && out+=("${a}"); done
  echo "${out[*]}"
}

# localizability <dataset> <results_dir> <plots_dir> <config> <attacks> [extra args...]
localizability() {
  local ds="$1" rdir="$2" pdir="$3" cfg="$4" attacks="$5"; shift 5
  if ! ls "${rdir}/localizability_shards/${ds}_loc__"*.pt >/dev/null 2>&1 \
     && [[ ! -f "${rdir}/${ds}_localizability.pt" ]]; then
    echo; echo ">>> [${ds}] localizability: no scores under ${rdir} -- skipping"
    return 0
  fi
  local flags
  for flags in "${FLAG_SETS[@]}"; do
    echo; echo ">>> [${ds}] localizability plot -> ${pdir}  ($* ${flags})"
    # shellcheck disable=SC2086
    "${PYTHON}" main.py evaluate-localizability --stage plot --config "${cfg}" --dataset "${ds}" \
      --attack-types ${attacks} --results-dir "${rdir}" --plots-dir "${pdir}" "$@" ${flags}
  done
}

# run_dataset <dataset> <base> <ablations: main|appendix>
run_dataset() {
  local ds="$1" base="$2" ablation_home="$3"
  local main="${base}/plots${PLOTS_TAG}" app="${base}/plots_appendix${PLOTS_TAG}"
  local cfg="${base}/${ds}_run_config_merged.yaml"
  [[ -f "${cfg}" ]] || cfg="${PROJECT_DIR}/config.yaml"
  # Where the ablation figures of this dataset live, and the stem suffix they carry.
  local abl_dir abl_suffix=()
  # Stem suffix of the all-attacks twin of a main-attacks figure. On a dataset whose
  # ablations already live in the appendix (OSV) the main-attacks version is itself
  # "_appendix", so the all-attacks one needs a distinct stem.
  local all_suffix=_appendix
  [[ "${ablation_home}" == "appendix" ]] && all_suffix=_all_appendix
  if [[ "${ablation_home}" == "appendix" ]]; then
    abl_dir="${app}"; abl_suffix=(--filename-suffix _appendix)
  else
    abl_dir="${main}"
  fi

  echo "=============================================================================="
  echo "${ds}: ${base}"
  echo "  config: ${cfg}"
  echo "  main figures      -> ${main}"
  echo "  appendix figures  -> ${app}"
  echo "  ablations         -> ${abl_dir}"
  echo "=============================================================================="
  if [[ "${CLEAN}" == "1" ]]; then
    rm -rf "${main}" "${app}"
  fi
  mkdir -p "${main}" "${app}"

  # -- main results: every attack, against the true GPS ------------------------------
  if [[ -z "${EPS}" ]]; then
    # shellcheck disable=SC2086
    plot_both results       "${ds}" "${base}" "${main}" "${cfg}" --attack-types ${ALL_ATTACKS} --override plot.gps_true=true
    # shellcheck disable=SC2086
    plot_both success-rate  "${ds}" "${base}" "${main}" "${cfg}" --attack-types ${ALL_ATTACKS} --override plot.gps_true=true
  fi
  # shellcheck disable=SC2086
  plot_both dtd-variance  "${ds}" "${base}" "${main}" "${cfg}" --attack-types ${ALL_ATTACKS} --override plot.gps_true=true

  # -- heatmap: headline attacks on one line (main), every attack (appendix) ----------
  local dpa_extra=()
  if [[ ! -f "${base}/${ds}_targeted_l2_results.pt" && -n "${DPA_PILOT_DIR}" \
        && -f "${DPA_PILOT_DIR}/${ds}_targeted_l2_results.pt" ]]; then
    dpa_extra=(--extra-results "targeted_l2=${DPA_PILOT_DIR}")
  fi
  # shellcheck disable=SC2086
  plot_both clean-vs-attacked-displacement "${ds}" "${base}" "${main}" "${cfg}" \
    --attack-types ${MAIN_HEATMAP_ATTACKS} --n-cols 5 ${dpa_extra[@]+"${dpa_extra[@]}"}
  # shellcheck disable=SC2086
  plot_both clean-vs-attacked-displacement "${ds}" "${base}" "${app}" "${cfg}" \
    --attack-types ${ALL_ATTACKS} --filename-suffix _appendix

  # -- JSON-backed ablations ----------------------------------------------------------
  if [[ -f "${base}/${ds}_sampling_steps_results.json" ]]; then
    plot_both sampling-steps "${ds}" "${base}" "${abl_dir}" "${cfg}" ${abl_suffix[@]+"${abl_suffix[@]}"}
  else
    echo ">>> [${ds}] no sampling-steps JSON -- skipping"
  fi
  if [[ -f "${base}/${ds}_model_transfer_results.json" ]]; then
    # Main attacks where the ablations live; every attack in the appendix twin.
    # shellcheck disable=SC2086
    plot_both model-transfer "${ds}" "${base}" "${abl_dir}" "${cfg}" --attack-types ${MAIN_ATTACKS} \
      ${abl_suffix[@]+"${abl_suffix[@]}"}
    plot_both model-transfer "${ds}" "${base}" "${app}" "${cfg}" --filename-suffix "${all_suffix}"
    # One panel per budget at 2500 km (TikZ only); spans every budget, so not in EPS mode.
    if [[ "${TIKZ}" == "1" && -z "${EPS}" ]]; then
      # shellcheck disable=SC2086
      plot model-transfer-by-budget "${ds}" "${base}" "${abl_dir}" "${cfg}" --tikz --attack-types ${MAIN_ATTACKS} \
        ${abl_suffix[@]+"${abl_suffix[@]}"}
      plot model-transfer-by-budget "${ds}" "${base}" "${app}" "${cfg}" --tikz --filename-suffix "${all_suffix}"
    fi
  else
    echo ">>> [${ds}] no model-transfer JSON -- skipping"
  fi
  if [[ -n "${EPS}" ]]; then
    echo ">>> [${ds}] restarts figure spans every budget -- skipped in EPS mode"
  elif [[ -f "${base}/${ds}_restarts_results.json" ]]; then
    # matplotlib only (no TikZ twin).
    plot restarts "${ds}" "${base}" "${abl_dir}" "${cfg}" ${abl_suffix[@]+"${abl_suffix[@]}"}
  else
    echo ">>> [${ds}] no restarts JSON -- skipping"
  fi
  if [[ -f "${base}/${ds}_cfg_results.json" ]]; then
    # guidance-scale sweep; matplotlib only (no TikZ twin).
    plot cfg "${ds}" "${base}" "${abl_dir}" "${cfg}" ${abl_suffix[@]+"${abl_suffix[@]}"}
  else
    echo ">>> [${ds}] no cfg JSON -- skipping"
  fi
  if [[ -f "${base}/${ds}_robustness_results.json" ]]; then
    # Main-paper version: 2500 km only. On OSV the ablations are appendix material, so
    # the 2500-km-only strip is skipped there and only the all-threshold version is made.
    if [[ "${ablation_home}" == "main" ]]; then
      plot_both robustness "${ds}" "${base}" "${main}" "${cfg}"
    fi
    # Appendix version: all three thresholds.
    plot_both robustness "${ds}" "${base}" "${app}" "${cfg}" --thresholds all --filename-suffix _appendix
    if [[ "${TIKZ}" == "1" && -f "${base}/${ds}_sampling_steps_results.json" ]]; then
      # Combined blur | JPEG | sampling-steps figure (TikZ only).
      if [[ "${ablation_home}" == "main" ]]; then
        plot robustness-and-sampling-steps "${ds}" "${base}" "${main}" "${cfg}" --tikz
      fi
      plot robustness-and-sampling-steps "${ds}" "${base}" "${app}" "${cfg}" --tikz \
        --thresholds all --filename-suffix _appendix
    fi
  else
    echo ">>> [${ds}] no robustness JSON -- skipping"
  fi

  # -- localizability (needs precomputed scores; see scripts/run_localizability.sh) ---
  # One figure per budget already, so not repeated in EPS mode. Main attacks where
  # the ablations live, every attack in the appendix.
  if [[ -z "${EPS}" ]]; then
    localizability "${ds}" "${base}" "${abl_dir}" "${cfg}" "${LOCALIZABILITY_ATTACKS}" ${abl_suffix[@]+"${abl_suffix[@]}"}
    # shellcheck disable=SC2086
    # (OSV keeps its main-attack version in the appendix too, hence the distinct stem.)
    # shellcheck disable=SC2086
    localizability "${ds}" "${base}" "${app}" "${cfg}" "$(present_attacks "${ds}" "${base}" ${ALL_ATTACKS})" \
      --filename-suffix "${all_suffix}"
  fi

  echo; echo "[${ds}] done."
}

case "${WHICH}" in
  yfcc) run_dataset yfcc "${YFCC_BASE}" main ;;
  osv)  run_dataset osv  "${OSV_BASE}"  appendix ;;
  all)  run_dataset yfcc "${YFCC_BASE}" main; run_dataset osv "${OSV_BASE}" appendix ;;
  *) echo "usage: $0 [yfcc|osv|all]" >&2; exit 1 ;;
esac

echo; echo "All figures done."
