#!/usr/bin/env bash
#
# Test the TruncatedSamplingAttack on 100 YFCC images.
#
# This is a memory-frugal variant of the Salman sampling attack (`sampling`). Both
# optimize the haversine distance between the clean and perturbed model predictions by
# differentiating THROUGH the sampler. The difference is how much of the trajectory is
# kept in the autograd graph:
#
#   - `sampling`            : runs a short 16-step trajectory and backprops through ALL
#                             16 steps (the forward path is a coarse 16-step model).
#   - `truncated_sampling`  : runs the full-fidelity 250-step trajectory in the forward
#                             pass but keeps the autograd graph only for the LAST 16
#                             steps (truncated backprop through time). The other 234
#                             steps run under torch.no_grad(), so activation memory is
#                             bounded by 16 steps while the sampled trajectory stays at
#                             250-step fidelity.
#
# 16-of-250 is the baseline defined by TruncatedSamplingAttack's defaults
# (backprop_steps=16, total_sampling_steps=250), so no override is needed below. To
# sweep the window, add e.g.
#   --override "attack_train_args.${DATASET}.per_attack.truncated_sampling.backprop_steps=32"
#
# Run it in the terminal:
#   conda activate plonk
#   bash scripts/test_truncated_sampling_attack.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROJECT_DIR="$(repo_root)"

CONFIG_PATH="${PROJECT_DIR}/config.yaml"
DATASET="yfcc"
ATTACK_TYPES=(truncated_sampling)
N_IMAGES=100
RESULTS_DIR="${PROJECT_DIR}/results/test/test_truncated_sampling"
PLOTS_DIR="${RESULTS_DIR}/plots"
PARALLEL_WORKERS=1

# Attack budgets (l_inf epsilon as a fraction of 255) are the single source of truth.
ATTACK_BUDGETS=(0.0314 )
BUDGETS="[$(IFS=,; echo "${ATTACK_BUDGETS[*]}")]"   # "[0.0314,0.0628]" override string

export MPLBACKEND=Agg                  # headless plotting

cd "${PROJECT_DIR}"

# Number-of-sampling-steps ablation: re-evaluate each best perturbation at every step
# count in EVAL_NUM_STEPS (no retraining). Unlike robustness it is NOT scoped per attack
# type -- it runs for every evaluated attack -- but we pass --run-sampling-steps-ablation
# and --eval-num-steps explicitly so this attack's evaluation always sweeps sampling steps
# regardless of the plot.run_sampling_steps_ablation config default.
EVAL_NUM_STEPS=(8 16 32 64 128 250)

# The robustness ablation (JPEG compression + Gaussian blur) is scoped per attack type via
# robustness.attack_types, which by default does NOT include truncated_sampling. So we
# enable it (--run-robustness-ablation) AND list truncated_sampling in
# --robustness-attack-types; otherwise no blur/compression is computed for it. The
# model-transfer ablation is on by config and applies to every attack regardless.
python main.py evaluate-dataset \
  --config "${CONFIG_PATH}" \
  --dataset "${DATASET}" \
  --attack-types "${ATTACK_TYPES[@]}" \
  --n-images "${N_IMAGES}" \
  --results-dir "${RESULTS_DIR}" \
  --plots-dir "${PLOTS_DIR}" \
  --parallel-workers "${PARALLEL_WORKERS}" \
  --run-sampling-steps-ablation \
  --eval-num-steps "${EVAL_NUM_STEPS[@]}" \
  --run-robustness-ablation \
  --robustness-attack-types "${ATTACK_TYPES[@]}" \
  --override "attack_budgets.${DATASET}=${BUDGETS}"

echo "Done. Results in ${RESULTS_DIR}, plots in ${PLOTS_DIR}."
