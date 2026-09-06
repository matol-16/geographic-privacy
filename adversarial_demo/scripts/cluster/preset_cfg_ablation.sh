#!/usr/bin/env bash
#
# Run the guidance-scale (cfg) ablation as a FULL sharded run on YFCC4k.
#
# The cfg ablation is computed at SHARD time: each image's best perturbation is
# re-evaluated at every scale in EVAL_CFGS. It therefore cannot be bolted onto an
# existing run whose shards were computed without it (yfcc4k_full has no cfg data) --
# the attacks must be trained and evaluated afresh with --run-cfg-ablation on. That is
# what this preset does, over the full 4000-image YFCC4k pool.
#
# It writes to its OWN results base (yfcc4k_cfg) so it never touches the established
# yfcc4k_full run. Only the cfg ablation is enabled; the heavier robustness /
# model-transfer / sampling-steps ablations are off (flip the RUN_* env vars to add
# them).
#
#   >>> WHICH ATTACKS TO SWEEP is a single knob. Set ATTACKS to the attacks you want
#   >>> to evaluate with cfg. It defaults to the pair from scripts/test_cfg_ablation.sh.
#
# Usage:
#   ./preset_cfg_ablation.sh [--dry-run]
#   ATTACKS="dtd sampling encoder" ./preset_cfg_ablation.sh
#   ATTACKS="dtd" EVAL_CFGS="1 2 5 10 15" RESULTS_BASE=/path ./preset_cfg_ablation.sh
#
# Prereqs: fill the #SBATCH placeholders in eval_shard.slurm / merge.slurm first, and
# make sure config.yaml points at this cluster's data + the plonk conda env exists.

set -euo pipefail

export CLUSTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${CLUSTER_DIR}/../.." && pwd)"

export DATASET="yfcc"
export TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"
export IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"

# Attacks to evaluate with the cfg sweep (the "specify later" knob).
export ATTACK_TYPES_STR="${ATTACKS:-dtd sampling}"
export MERGE_ATTACK_TYPES_STR="${ATTACK_TYPES_STR}"        # fresh run: merge exactly what we compute
export ATTACK_BUDGETS_STR="${BUDGETS:-0.0157 0.0314}"      # match the existing full run

# cfg ablation ON; other ablations OFF (override with RUN_*=1 if you want them too).
export RUN_CFG_ABLATION=1
export EVAL_CFGS_STR="${EVAL_CFGS:-1 2 5 10}"
export RUN_SAMPLING_STEPS_ABLATION="${RUN_SAMPLING_STEPS_ABLATION:-0}"
export RUN_ROBUSTNESS_ABLATION="${RUN_ROBUSTNESS_ABLATION:-0}"
export RUN_MODEL_TRANSFER_ABLATION="${RUN_MODEL_TRANSFER_ABLATION:-0}"

# Separate base so the established yfcc4k_full run is left intact.
export RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/yfcc4k_cfg}"
export PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"

exec "${CLUSTER_DIR}/submit.sh" "$@"
