#!/usr/bin/env bash
#
# Shared experiment configuration for the multi-node YFCC4k / OSV-5M evaluation.
# Sourced by submit.sh, eval_shard.slurm and merge.slurm so every piece agrees on
# the attacks, budgets, image count and output paths. Edit this file to define a run,
# OR export any of the variables below before calling submit.sh (every value honours
# an environment override, which is how the preset_*.sh launchers drive it).
#
# THE DEFAULTS BELOW ARE THE FULL EXPERIMENT (configs/cluster_full/config.yaml): the 9
# attacks of the paper + the guided-sampler variant sampling_cfg10, budgets 1/255 4/255 8/255, 4000 images, 5 restarts selected by the
# FSD to the true location, and every ablation (sampling steps, guidance scale, JPEG / blur
# robustness, cross-model transfer) for every attack. `DATASET=yfcc ./submit.sh` and
# `DATASET=osv ./submit.sh` (or preset_cluster_full.sh <ds>) launch it; nothing to edit
# besides the machine paths (DATA_ROOT) and the #SBATCH placeholders in the .slurm files.
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
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/configs/cluster_full/config.yaml}"

# --- experiment definition (all overridable from the environment) ------------- #
DATASET="${DATASET:-yfcc}"                       # "yfcc" (4000 imgs) or "osv"
TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"             # full seeded pool to evaluate
IMAGES_PER_SHARD="${IMAGES_PER_SHARD:-100}"      # images per cluster job (window size)
PARALLEL_WORKERS="${PARALLEL_WORKERS:-1}"        # per-GPU concurrency (1 = strict reproducibility)

# Attacks COMPUTED by the shard array (one job per attack x window): the attacks of the
# paper (+ sampling_cfg10, the sampling attack trained through the guided sampler) (see configs/cluster_full/ATTACK_SPECS.md). GeoShield is out-of-process but sharded
# like the rest; drop it if the Geoshield repo isn't set up.
# NOTE `ace` (GeoFA-ACE) is part of the run. To fold it into a run that is ALREADY going
# (rather than recomputing everything), use preset_ace.sh instead of re-submitting here.
read -r -a ATTACK_TYPES <<< "${ATTACK_TYPES_STR:-dtd diffusion_l2 targeted_l2 sampling sampling_cfg10 encoder training_loss unidef ace dtd_blur geoshield}"

# Attacks STITCHED by the merge job. Defaults to the computed set. Set it to a SUPERSET
# (the attacks already sharded into RESULTS_BASE plus a newly computed one) to fold a
# fresh attack into an existing full run without recomputing the others -- the merge
# reads every listed attack's shards from RESULTS_BASE/shards and rebuilds the combined
# results, ablation JSONs and plots.
read -r -a MERGE_ATTACK_TYPES <<< "${MERGE_ATTACK_TYPES_STR:-${ATTACK_TYPES[*]}}"

read -r -a ATTACK_BUDGETS <<< "${ATTACK_BUDGETS_STR:-0.0039 0.0157 0.0314}"   # 1/255, 4/255, 8/255
read -r -a EVAL_NUM_STEPS <<< "${EVAL_NUM_STEPS_STR:-8 16 32 64 250}"  # sampling-steps ablation grid

# --- ablations (1 = on, 0 = off) ---------------------------------------------- #
RUN_SAMPLING_STEPS_ABLATION="${RUN_SAMPLING_STEPS_ABLATION:-1}"
RUN_ROBUSTNESS_ABLATION="${RUN_ROBUSTNESS_ABLATION:-1}"
RUN_MODEL_TRANSFER_ABLATION="${RUN_MODEL_TRANSFER_ABLATION:-1}"
RUN_CFG_ABLATION="${RUN_CFG_ABLATION:-1}"                              # guidance-scale sweep
read -r -a EVAL_CFGS <<< "${EVAL_CFGS_STR:-1 2 5 10}"

# Attacks the robustness ablation is scoped to: every computed attack by default. Set it
# to a superset when folding a new attack into an existing run so the merged robustness
# JSON keeps the old curves and adds the new one.
read -r -a ROBUSTNESS_ATTACK_TYPES <<< "${ROBUSTNESS_ATTACK_TYPES_STR:-${ATTACK_TYPES[*]}}"

# Extra `--override key=value` pairs forwarded verbatim to both the shard and merge
# commands (space separated; each token must be spaceless). Use to pin robustness
# transform grids etc. so a folded-in attack matches the original run exactly.
read -r -a EXTRA_OVERRIDES <<< "${EXTRA_OVERRIDES_STR:-}"

# Machine / run knobs, each turned into --override pairs below:
#   DATA_ROOT        dir holding YFCC100M/yfcc4k and osv5m  -> data_dirs.{yfcc,osv}
#                    (default: <adversarial_demo>/DATA, the layout of the previous runs)
#   NUM_RESTARTS     restarts per image (default 5, the paper protocol); sets BOTH
#                    attack_train_args.<ds>.num_restarts and max_restarts
#   SAVE_DELTAS      1 -> persist every best perturbation (save_deltas)
#   GEOSHIELD_IO_DIR base dir for GeoShield's clean copies / attacked outputs
#                    (default: RESULTS_BASE/geoshield_io)
#                    -> geoshield.clean_dir=<dir>/clean, geoshield.output_base=<dir>/attacked
#   ABLATION_WINDOWS run the (expensive) eval-time ablations only on the first N image
#                    windows of every attack (a seeded random subset of N*IMAGES_PER_SHARD
#                    images); the remaining windows compute the main result only. Empty
#                    (default) = every window. The merge builds the ablation JSONs from the
#                    windows that carry samples.
DATA_ROOT="${DATA_ROOT:-${PROJECT_DIR}/DATA}"
NUM_RESTARTS="${NUM_RESTARTS-5}"          # export NUM_RESTARTS="" to keep the config file's values
SAVE_DELTAS="${SAVE_DELTAS:-0}"
ABLATION_WINDOWS="${ABLATION_WINDOWS:-}"

# --- output paths ------------------------------------------------------------- #
# Where shard outputs (RESULTS_BASE/shards/...) and merged results/plots go. Point this
# at an existing full-run base to integrate a new attack into it.
RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_full/${DATASET}_full}"
PLOTS_DIR="${PLOTS_DIR:-${RESULTS_BASE}/plots}"
GEOSHIELD_IO_DIR="${GEOSHIELD_IO_DIR-${RESULTS_BASE}/geoshield_io}"   # export "" to keep the config file's dirs

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
if [[ -n "${DATA_ROOT}" ]]; then
  OVERRIDE_FLAGS+=(--override "data_root=${DATA_ROOT}"
                   --override "data_dirs.yfcc=${DATA_ROOT}/YFCC100M/yfcc4k"
                   --override "data_dirs.osv=${DATA_ROOT}/osv5m")
fi
if [[ -n "${NUM_RESTARTS}" ]]; then
  OVERRIDE_FLAGS+=(--override "attack_train_args.${DATASET}.num_restarts=${NUM_RESTARTS}"
                   --override "max_restarts=${NUM_RESTARTS}")
fi
if [[ "${SAVE_DELTAS}" == 1 ]]; then
  OVERRIDE_FLAGS+=(--override "save_deltas=true")
fi
if [[ -n "${GEOSHIELD_IO_DIR}" ]]; then
  OVERRIDE_FLAGS+=(--override "geoshield.clean_dir=${GEOSHIELD_IO_DIR}/clean"
                   --override "geoshield.output_base=${GEOSHIELD_IO_DIR}/attacked")
fi

# Assemble the ablation flags shared by the shard and merge commands from the toggles
# above, so both stay in lock-step and new ablations are enabled in one place.
# NB: uses if-blocks and a trailing `return 0` on purpose -- a bare `[[..]] && ..` as the
# function's last statement returns non-zero when false and would abort callers under
# `set -e`.
# build_ablation_flags [window_idx]: with a window index and ABLATION_WINDOWS set, windows
# >= ABLATION_WINDOWS get no eval-time ablation flags (main result only). The merge (no
# window) always gets the full set, so it stitches the ablation samples that exist.
build_ablation_flags() {
  local window_idx="${1:-}"
  ABLATION_FLAGS=()
  if [[ -n "${window_idx}" && -n "${ABLATION_WINDOWS}" ]] && (( window_idx >= ABLATION_WINDOWS )); then
    ABLATION_FLAGS+=(--eval-num-steps "${EVAL_NUM_STEPS[@]}")   # harmless without the run flags
    return 0
  fi
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
