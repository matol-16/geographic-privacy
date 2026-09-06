#!/usr/bin/env bash
#
# Shared experiment configuration for the multi-node YFCC4k / OSV-5M evaluation.
# Sourced by submit.sh, eval_shard.slurm and merge.slurm so every piece agrees on
# the attacks, budgets, image count and output paths. Edit this file to define a run,
# OR export any of the variables below before calling submit.sh (every value honours
# an environment override, which is how the preset_*.sh launchers drive it).
#
# The cluster scheduling is a 2D grid: one job per (attack x image window). Window k
# of an attack covers images [k*IMAGES_PER_SHARD, (k+1)*IMAGES_PER_SHARD) of the seeded
# selection of the first TOTAL_IMAGES images. merge.slurm stitches them back together.
#
# String-list overrides use a *_STR env var (space separated), e.g.
#   ATTACK_TYPES_STR="dtd sampling" ATTACK_BUDGETS_STR="0.0157 0.0314" ./submit.sh

# --- conda environment -------------------------------------------------------- #
CONDA_ENV="${CONDA_ENV:-plonk}"

# --- project paths (auto-detected) -------------------------------------------- #
# CLUSTER_DIR is exported by submit.sh; fall back to this file's directory so the
# slurm scripts can source us even if it was not set.
_THIS_DIR="${CLUSTER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "${_THIS_DIR}/../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config.yaml}"

# --- experiment definition (all overridable from the environment) ------------- #
DATASET="${DATASET:-yfcc}"                       # "yfcc" (4000 imgs) or "osv"
TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"             # full seeded pool to evaluate
IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"      # images per cluster job (window size)
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"        # per-GPU concurrency (1 = strict reproducibility)

# Attacks COMPUTED by the shard array (one job per attack x window). GeoShield is
# out-of-process but sharded like the rest; drop it if the Geoshield repo isn't set up.
read -r -a ATTACK_TYPES <<< "${ATTACK_TYPES_STR:-encoder sampling diffusion_l2 dtd unidef ace geoshield}"

# Attacks STITCHED by the merge job. Defaults to the computed set. Set it to a SUPERSET
# (the attacks already sharded into RESULTS_BASE plus a newly computed one) to fold a
# fresh attack into an existing full run without recomputing the others -- the merge
# reads every listed attack's shards from RESULTS_BASE/shards and rebuilds the combined
# results, ablation JSONs and plots.
read -r -a MERGE_ATTACK_TYPES <<< "${MERGE_ATTACK_TYPES_STR:-${ATTACK_TYPES[*]}}"

read -r -a ATTACK_BUDGETS <<< "${ATTACK_BUDGETS_STR:-0.0157 0.0314}"   # 4/255 and 8/255
read -r -a EVAL_NUM_STEPS <<< "${EVAL_NUM_STEPS_STR:-8 16 32 64 250}"  # sampling-steps ablation grid

# --- ablations (1 = on, 0 = off) ---------------------------------------------- #
RUN_SAMPLING_STEPS_ABLATION="${RUN_SAMPLING_STEPS_ABLATION:-1}"
RUN_ROBUSTNESS_ABLATION="${RUN_ROBUSTNESS_ABLATION:-1}"
RUN_MODEL_TRANSFER_ABLATION="${RUN_MODEL_TRANSFER_ABLATION:-1}"
RUN_CFG_ABLATION="${RUN_CFG_ABLATION:-0}"                              # guidance-scale sweep
read -r -a EVAL_CFGS <<< "${EVAL_CFGS_STR:-1 2 5 10}"

# Attacks the robustness ablation is scoped to. Empty -> fall back to config.yaml's
# robustness.attack_types (the historical behaviour). Set it to a superset when folding
# a new attack into an existing run so the merged robustness JSON keeps the old curves
# and adds the new one.
read -r -a ROBUSTNESS_ATTACK_TYPES <<< "${ROBUSTNESS_ATTACK_TYPES_STR:-}"

# Extra `--override key=value` pairs forwarded verbatim to both the shard and merge
# commands (space separated; each token must be spaceless). Use to pin robustness
# transform grids etc. so a folded-in attack matches the original run exactly.
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_OVERRIDES_STR:-}"

# --- output paths ------------------------------------------------------------- #
# Where shard outputs (RESULTS_BASE/shards/...) and merged results/plots go. Point this
# at an existing full-run base to integrate a new attack into it.
RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/${DATASET}4k_full}"
PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"

# --- derived (do not edit) ---------------------------------------------------- #
BUDGETS_OVERRIDE="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # e.g. "[0.0157,0.0314]"
NUM_WINDOWS=$(( (TOTAL_IMAGES + IMAGES_PER_SHARD - 1) / IMAGES_PER_SHARD ))
NUM_ATTACKS=${#ATTACK_TYPES[@]}
ARRAY_SIZE=$(( NUM_ATTACKS * NUM_WINDOWS ))

# --override flags shared by the shard and merge commands: the budgets plus any extras.
OVERRIDE_FLAGS=(--override "attack_budgets.${DATASET}=${BUDGETS_OVERRIDE}")
for _o in ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}; do
  OVERRIDE_FLAGS+=(--override "${_o}")
done

# Assemble the ablation flags shared by the shard and merge commands from the toggles
# above, so both stay in lock-step and new ablations are enabled in one place.
# NB: uses if-blocks and a trailing `return 0` on purpose -- a bare `[[..]] && ..` as the
# function's last statement returns non-zero when false and would abort callers under
# `set -e`.
build_ablation_flags() {
  ABLATION_FLAGS=()
  if [[ "${RUN_SAMPLING_STEPS_ABLATION}" == 1 ]]; then ABLATION_FLAGS+=(--run-sampling-steps-ablation); fi
  if [[ "${RUN_ROBUSTNESS_ABLATION}" == 1 ]];     then ABLATION_FLAGS+=(--run-robustness-ablation); fi
  if [[ "${RUN_MODEL_TRANSFER_ABLATION}" == 1 ]]; then ABLATION_FLAGS+=(--run-model-transfer-ablation); fi
  if [[ "${RUN_CFG_ABLATION}" == 1 ]];            then ABLATION_FLAGS+=(--run-cfg-ablation --eval-cfgs "${EVAL_CFGS[@]}"); fi
  ABLATION_FLAGS+=(--eval-num-steps "${EVAL_NUM_STEPS[@]}")
  if (( ${#ROBUSTNESS_ATTACK_TYPES[@]} > 0 )); then ABLATION_FLAGS+=(--robustness-attack-types "${ROBUSTNESS_ATTACK_TYPES[@]}"); fi
  return 0
}

# Activate conda in a way that works on a bare compute node (PATH may lack conda).
activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    # Adjust if your conda lives elsewhere.
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  fi
  conda activate "${CONDA_ENV}"
}
