#!/usr/bin/env bash
#
# THE full run: every attack of the paper, every ablation, 3 budgets, 4000 images, on one
# dataset per invocation (submit both):
#
#   ./preset_cluster_full.sh yfcc --dry-run     # print the job plan, submit nothing
#   ./preset_cluster_full.sh yfcc               # 9 attacks x 40 windows = 360 GPU jobs + merge
#   ./preset_cluster_full.sh osv                # same on OSV-5M
#
# What one shard job computes (100 images x 3 budgets of ONE attack):
#   * the attack itself: configs/cluster_full/config.yaml pins every attack to its latest
#     implementation under its canonical key (GeoFA-TD = dtd_v3_old_forward, see the config
#     header); NUM_RESTARTS (default 5, the paper protocol) restarts per image, FSD-best kept;
#   * eval-time ablations on the kept perturbation: sampling steps [8 16 32 64 250],
#     guidance scale [1 2 5 10], JPEG q [20 40 60 80 100] + Gaussian blur sigma [0 2 4 6 8 10],
#     transfer to the diffusion / Euclidean-flow variants (same encoder);
#   * clean AND attacked predictions vs the true GPS are always stored (location_results),
#     which is what the clean-vs-attacked FSD heatmaps read.
# GeoShield is one more sharded attack (needs the Geoshield repo next to this one); it gets
# the same eval-time ablations, incl. transfer, and lossless model-exact clean copies.
#
# Budgets: 1/255, 4/255, 8/255 (BUDGETS="0.0039 0.0157 0.0314").
#
# Outputs: RESULTS_BASE (default results/cluster_full/<ds>_full) with shards/, the merged
# <ds>_<attack>_results.pt, the ablation JSONs (<ds>_{sampling_steps,cfg,robustness,
# model_transfer,restarts}_results.json) and merge plots in plots/. Paper figures + tables:
# ./postprocess_cluster_full.sh <ds>  (CPU only).
#
# Knobs (env): NUM_RESTARTS (5), ABLATION_WINDOWS (all; e.g. 10 -> ablations on a random
# 1000-image subset), IMAGES_PER_SHARD (100), TOTAL_IMAGES (4000), BUDGETS, ATTACKS,
# EVAL_NUM_STEPS, EVAL_CFGS, SAVE_DELTAS (0), DATA_ROOT (<adversarial_demo>/DATA),
# RESULTS_BASE, MAX_CONCURRENT.
#
# Prereqs (see README): #SBATCH placeholders in eval_shard.slurm / merge.slurm; DATA_ROOT
# holding YFCC100M/yfcc4k and osv5m; the plonk conda env; the Geoshield repo beside this
# repo's root (drop geoshield from ATTACKS otherwise).

set -euo pipefail

DS="${1:?usage: preset_cluster_full.sh <yfcc|osv> [--dry-run]}"; shift || true
case "${DS}" in yfcc|osv) ;; *) echo "dataset must be yfcc or osv, got '${DS}'" >&2; exit 1 ;; esac

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/cluster_full/config.yaml}"
export DATASET="${DS}"
export TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"
export IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"
export PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"

# The 9 attacks (canonical keys; the config pins the implementation of each).
export ATTACK_TYPES_STR="${ATTACKS:-dtd diffusion_l2 targeted_l2 sampling encoder training_loss unidef dtd_blur geoshield}"
export MERGE_ATTACK_TYPES_STR="${ATTACK_TYPES_STR}"          # fresh run: merge exactly what we compute
export ATTACK_BUDGETS_STR="${BUDGETS:-0.0039 0.0157 0.0314}"  # 1/255, 4/255, 8/255

# Every ablation on, for every attack.
export RUN_SAMPLING_STEPS_ABLATION=1
export RUN_ROBUSTNESS_ABLATION=1
export RUN_MODEL_TRANSFER_ABLATION=1
export RUN_CFG_ABLATION=1
export EVAL_NUM_STEPS_STR="${EVAL_NUM_STEPS:-8 16 32 64 250}"
export EVAL_CFGS_STR="${EVAL_CFGS:-1 2 5 10}"
export ROBUSTNESS_ATTACK_TYPES_STR="${ATTACK_TYPES_STR}"      # robustness scoped to ALL attacks of the run
export ABLATION_WINDOWS="${ABLATION_WINDOWS:-}"

# Restarts: paper protocol (5, FSD-best kept, no extra ablation-only restarts).
export NUM_RESTARTS="${NUM_RESTARTS:-5}"
export SAVE_DELTAS="${SAVE_DELTAS:-0}"

# Machine paths.
export DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DATA}"
export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_full/${DS}_full}"
export PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"
export GEOSHIELD_IO_DIR="${GEOSHIELD_IO_DIR:-${RESULTS_BASE}/geoshield_io}"

echo "cluster_full preset: dataset=${DS} restarts=${NUM_RESTARTS} ablation_windows=${ABLATION_WINDOWS:-all}"
echo "  config=${CONFIG_PATH}"
echo "  data_root=${DATA_ROOT}"
exec "${CLUSTER_DIR}/submit.sh" "$@"
