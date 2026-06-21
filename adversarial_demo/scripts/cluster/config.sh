#!/usr/bin/env bash
#
# Shared experiment configuration for the multi-node YFCC4k / OSV-5M evaluation.
# Sourced by submit.sh, eval_shard.slurm and merge.slurm so every piece agrees on
# the attacks, budgets, image count and output paths. Edit this file to define a run.
#
# The cluster scheduling is a 2D grid: one job per (attack x image window). Window k
# of an attack covers images [k*IMAGES_PER_SHARD, (k+1)*IMAGES_PER_SHARD) of the seeded
# selection of the first TOTAL_IMAGES images. merge.slurm stitches them back together.

# --- conda environment -------------------------------------------------------- #
CONDA_ENV="plonk"

# --- project paths (auto-detected) -------------------------------------------- #
# CLUSTER_DIR is exported by submit.sh; fall back to this file's directory so the
# slurm scripts can source us even if it was not set.
_THIS_DIR="${CLUSTER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${_THIS_DIR}/../.." && pwd)"
CONFIG_PATH="${PROJECT_DIR}/config.yaml"

# --- experiment definition ---------------------------------------------------- #
DATASET="yfcc"                       # "yfcc" (4000 imgs) or "osv"
TOTAL_IMAGES=4000                    # full seeded pool to evaluate (use a subset for OSV)
IMAGES_PER_SHARD=100                 # images per cluster job (window size); tune to walltime
# Attacks from scripts/test_all_attacks.sh. GeoShield is out-of-process: each geoshield
# task generates + evaluates its window via Geoshield/geoshield.py into shard-isolated
# dirs, and merge stitches it in like any other attack. It requires the Geoshield repo
# alongside plonk/ and the geoshield.{clean_dir,output_base} settings in config.yaml.
# Drop "geoshield" from this list if you do not want it (or it isn't installed).
ATTACK_TYPES=(encoder sampling diffusion_l2 dtd unidef ace geoshield)
ATTACK_BUDGETS=(0.0157 0.0314)       # 4/255 and 8/255
EVAL_NUM_STEPS=(8 16 32 64 250)      # sampling-steps ablation grid
PARALLEL_WORKERS=1                   # per-GPU concurrency (1 = strict reproducibility)

# Where shard outputs (RESULTS_BASE/shards/...) and merged results/plots go.
RESULTS_BASE="${PROJECT_DIR}/results/cluster_eval/${DATASET}4k_full"
PLOTS_DIR="${RESULTS_BASE}/plots"

# --- derived (do not edit) ---------------------------------------------------- #
BUDGETS_OVERRIDE="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # e.g. "[0.0157,0.0314]"
NUM_WINDOWS=$(( (TOTAL_IMAGES + IMAGES_PER_SHARD - 1) / IMAGES_PER_SHARD ))
NUM_ATTACKS=${#ATTACK_TYPES[@]}
ARRAY_SIZE=$(( NUM_ATTACKS * NUM_WINDOWS ))

# Activate conda in a way that works on a bare compute node (PATH may lack conda).
activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    # Adjust if your conda lives elsewhere.
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV}"
}
