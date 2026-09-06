#!/usr/bin/env bash
#
# Fully evaluate the DTDBlur attack on the full YFCC4k pool and fold it into the existing
# yfcc4k_full run, as a blur-robustness ABLATION against plain DTD.
#
#   ./preset_dtd_blur.sh [--dry-run]
#
# THE QUESTION: does training the perturbation with additional noise (expectation over a
# Gaussian blur of random sigma) make the attack more robust to Gaussian blurring at
# evaluation time? DTDBlur evaluates the DTD loss on both the clean protected image and a
# blurred copy (sigma ~ U[blur_sigma_min, blur_sigma_max]), combined blur_loss_weight /
# (1 - blur_loss_weight); plain DTD trains on the clean image only.
#
# HOW THE COMPARISON IS PRODUCED: `dtd` already has robustness (JPEG + Gaussian blur)
# samples in yfcc4k_full from the original run. We compute only the `dtd_blur` shards, and
# the merge rebuilds the combined robustness JSON over BOTH attacks, so
# yfcc_robustness_results.{json,png,pdf} overlays dtd vs dtd_blur across the blur sweep.
# The blur grid is sigma [0 2 4 6 8 10], so it covers both in-distribution blur (<= the
# training sigma_max of 4) and out-of-distribution blur (6-10) -- the interesting part.
#
# Merge + robustness attack lists are auto-detected from the shards already on disk (see
# integrate_lib.sh), so this composes with preset_targeted_l2.sh in either order.
#
# Everything else mirrors the existing full run: 2 budgets, sampling-steps grid
# [8 16 32 64 250], model transfer, and the original JPEG/blur grids (pinned via
# EXTRA_OVERRIDES so dtd_blur lines up point-for-point with dtd).
#
# Tune the blur training params via config.yaml attack_train_args.yfcc.per_attack.dtd_blur
# (blur_sigma_min / blur_sigma_max / blur_loss_weight).
#
# Prereqs: fill the #SBATCH placeholders in eval_shard.slurm / merge.slurm first.

set -euo pipefail

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${CLUSTER_DIR}/integrate_lib.sh"

NEW_ATTACK="dtd_blur"

export DATASET="yfcc"
export TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"
export IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"
export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/yfcc4k_full}"
export PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"

export ATTACK_TYPES_STR="${NEW_ATTACK}"                    # compute only dtd_blur shards
export ATTACK_BUDGETS_STR="${BUDGETS:-0.0157 0.0314}"      # match the existing full run
export EVAL_NUM_STEPS_STR="${EVAL_NUM_STEPS:-8 16 32 64 250}"

# Merge every attack already sharded into RESULTS_BASE, plus dtd_blur; robustness scope
# must be the union (merge_shards takes it explicitly and would otherwise drop the
# existing curves -- including the `dtd` baseline this ablation compares against).
resolve_integration_lists "${RESULTS_BASE}" "${NEW_ATTACK}"
export MERGE_ATTACK_TYPES_STR="${MERGE_ATTACK_TYPES:-${INTEGRATION_MERGE_ATTACKS}}"
export ROBUSTNESS_ATTACK_TYPES_STR="${ROBUSTNESS_ATTACK_TYPES:-${INTEGRATION_ROBUSTNESS_ATTACKS}}"

export RUN_SAMPLING_STEPS_ABLATION=1
export RUN_ROBUSTNESS_ABLATION=1          # the point of this run
export RUN_MODEL_TRANSFER_ABLATION=1
export RUN_CFG_ABLATION=0                 # yfcc4k_full has no cfg data; see preset_cfg_ablation.sh

# Pin the robustness grids to the ones the original run used, so dtd_blur is directly
# comparable to dtd (config.yaml may list an extra JPEG=100 level the original lacked).
export EXTRA_OVERRIDES_STR="${EXTRA_OVERRIDES:-robustness.jpeg_quality_factors=[20,40,60,80] robustness.gaussian_blur_sigmas=[0,2,4,6,8,10]}"

echo "Folding '${NEW_ATTACK}' into ${RESULTS_BASE}"
echo "  merge attacks:      ${MERGE_ATTACK_TYPES_STR}"
echo "  robustness scope:   ${ROBUSTNESS_ATTACK_TYPES_STR}"
echo

"${CLUSTER_DIR}/submit.sh" "$@"

if [[ " $* " != *" --dry-run "* ]]; then
  cat <<MSG

--------------------------------------------------------------------------------
When the merge finishes, the blur comparison is here:
  ${RESULTS_BASE}/yfcc_robustness_results.json      (numbers: dtd vs dtd_blur)
  ${PLOTS_DIR}/yfcc_robustness_results.{png,pdf}    (overlay across sigma 0..10)
--------------------------------------------------------------------------------
MSG
fi
