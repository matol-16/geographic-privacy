#!/usr/bin/env bash
#
# THE full run: every attack of the paper, every ablation, 3 budgets, 4000 images, on one
# dataset per invocation (submit both). Equivalent to `DATASET=<ds> ./submit.sh` -- the full
# experiment is config.sh's default configuration.
#
#   ./preset_cluster_full.sh yfcc --dry-run     # print the job plan, submit nothing
#   ./preset_cluster_full.sh yfcc               # 11 attacks x 40 windows = 440 GPU jobs + merge
#   ./preset_cluster_full.sh osv                # same on OSV-5M
#
# What one shard job computes (100 images x 3 budgets of ONE attack):
#   * the attack itself (standard specification of each key, see the config header and
#     configs/cluster_full/ATTACK_SPECS.md); NUM_RESTARTS (default 5, the paper protocol)
#     restarts per image, the one with the highest FSD to the true location kept;
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

# The full experiment IS config.sh's default; this preset only fixes the dataset (and lets
# the usual env knobs through: NUM_RESTARTS, ABLATION_WINDOWS, DATA_ROOT, RESULTS_BASE, ...).
export DATASET="${DS}"
echo "cluster_full: dataset=${DS} restarts=${NUM_RESTARTS:-5} ablation_windows=${ABLATION_WINDOWS:-all}"
exec "${CLUSTER_DIR}/submit.sh" "$@"
