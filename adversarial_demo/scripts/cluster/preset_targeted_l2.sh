#!/usr/bin/env bash
#
# Fully evaluate the TargetedL2 attack on the full dataset and fold it into the
# EXISTING full-run results, reusing the shard/merge infrastructure (submit.sh).
#
#   ./preset_targeted_l2.sh yfcc [--dry-run]   # integrate into results/cluster_eval/yfcc4k_full
#   ./preset_targeted_l2.sh osv  [--dry-run]   # integrate into results/cluster_eval/osv_5k_full
#
# What "fully evaluated" means here == exactly what the existing full runs computed:
#   * main displacement / success-rate results (2 budgets: 4/255 and 8/255)
#   * sampling-steps ablation on the grid [8 16 32 64 250]
#   * robustness ablation (JPEG [20 40 60 80] + Gaussian blur [0 2 4 6 8 10])
#   * cross-model transfer ('' / diffusion / flow)
# (The cfg guidance-scale sweep is NOT part of these runs -- use preset_cfg_ablation.sh
#  for that.)
#
# HOW INTEGRATION WORKS (differs per dataset because of what is still on disk):
#
#   YFCC  yfcc4k_full still holds every attack's shards, so we compute ONLY the
#         targeted_l2 shards (40 windows x 4000/100) and let the merge stitch the FULL
#         attack set -- the 8 existing attacks (read from their on-disk shards) plus
#         targeted_l2 -- back into the combined results, ablation JSONs and plots.
#         Result: yfcc_targeted_l2_results.pt appears and every combined figure in
#         yfcc4k_full/plots is regenerated to include targeted_l2. One command, done.
#
#   OSV   osv_5k_full keeps the merged per-attack .pt files but NOT the shards, so the
#         merge here can only stitch targeted_l2 (writing osv_targeted_l2_results.pt
#         next to the existing six). Its own combined figures go to a side dir
#         (plots_targeted_l2/) so nothing existing is clobbered. To rebuild the COMBINED
#         OSV figures including targeted_l2, run the companion replot_combined.sh
#         afterwards -- it reads the saved .pt files only (no GPU, seconds). The command
#         is printed at the end of this run.
#
# Override any knob from the environment, e.g.:
#   BUDGETS="0.0314" IMAGES_PER_SHARD=200 ./preset_targeted_l2.sh yfcc --dry-run
#
# Prereqs: fill the #SBATCH placeholders in eval_shard.slurm / merge.slurm first, and
# make sure config.yaml points at this cluster's data + the plonk conda env exists.

set -euo pipefail

DS="${1:?usage: preset_targeted_l2.sh <yfcc|osv> [--dry-run]}"; shift || true

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${CLUSTER_DIR}/integrate_lib.sh"

# --- common to both datasets -------------------------------------------------- #
export DATASET="${DS}"
export ATTACK_TYPES_STR="targeted_l2"                       # compute only targeted_l2 shards
export ATTACK_BUDGETS_STR="${BUDGETS:-0.0157 0.0314}"      # match the existing full runs
export IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"
export EVAL_NUM_STEPS_STR="${EVAL_NUM_STEPS:-8 16 32 64 250}"

# Full evaluation: sampling-steps + robustness (scoped to include targeted_l2) +
# model-transfer, mirroring the existing full runs. cfg stays off.
export RUN_SAMPLING_STEPS_ABLATION=1
export RUN_ROBUSTNESS_ABLATION=1
export RUN_MODEL_TRANSFER_ABLATION=1
export RUN_CFG_ABLATION=0

# Pin the robustness transform grids to the ones the original runs used, so
# targeted_l2 lines up point-for-point with the other attacks (the repo config.yaml
# may list an extra JPEG=100 level that the original run did not compute).
export EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-robustness.jpeg_quality_factors=[20,40,60,80] robustness.gaussian_blur_sigmas=[0,2,4,6,8,10]}"

case "${DS}" in
  yfcc)
    export TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"
    export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/yfcc4k_full}"
    export PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"
    # Merge every attack already sharded into RESULTS_BASE, plus targeted_l2, so it lands
    # in the combined figures. Auto-detected from the shard dirs (see integrate_lib.sh) so
    # this composes with the other integration presets in any order. Robustness scope must
    # be the union: merge_shards takes it explicitly and would otherwise drop the existing
    # curves from the re-merged robustness JSON.
    resolve_integration_lists "${RESULTS_BASE}" targeted_l2
    export MERGE_ATTACK_TYPES_STR="${MERGE_ATTACK_TYPES:-${INTEGRATION_MERGE_ATTACKS}}"
    export ROBUSTNESS_ATTACK_TYPES_STR="${ROBUSTNESS_ATTACK_TYPES:-${INTEGRATION_ROBUSTNESS_ATTACKS}}"
    ;;
  osv)
    export TOTAL_IMAGES="${TOTAL_IMAGES:-5000}"            # osv_5k_full = 5000 images x 2 budgets
    export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/osv_5k_full}"
    # Old OSV shards are gone -> stitch only targeted_l2, and send its own combined
    # figures to a side dir so the existing osv_5k_full/plots are untouched.
    export PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots_targeted_l2}"
    export MERGE_ATTACK_TYPES_STR="targeted_l2"
    export ROBUSTNESS_ATTACK_TYPES_STR="${ROBUSTNESS_ATTACK_TYPES:-targeted_l2}"
    ;;
  *)
    echo "unknown dataset '${DS}' (expected yfcc or osv)" >&2; exit 1;;
esac

"${CLUSTER_DIR}/submit.sh" "$@"

# OSV needs a second, GPU-free step to rebuild the combined figures across all attacks.
if [[ "${DS}" == "osv" && " $* " != *" --dry-run "* ]]; then
  cat <<EOF

--------------------------------------------------------------------------------
OSV: once the merge above has finished, rebuild the COMBINED plots (existing six
attacks + targeted_l2) from the saved .pt files -- no GPU, runs in seconds:

  ${CLUSTER_DIR}/replot_combined.sh osv \\
    "encoder sampling dtd ace training_loss unidef targeted_l2" \\
    "${RESULTS_BASE}"
--------------------------------------------------------------------------------
EOF
fi
