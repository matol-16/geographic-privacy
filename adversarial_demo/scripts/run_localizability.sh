#!/usr/bin/env bash
#
# Precompute image localizability on a single local GPU (no SLURM).
#
# Localizability is a property of the clean image alone, so it is computed + stored once
# here (keyed by image id) and later joined to the separately-produced attack results by
# `main.py evaluate-localizability --stage plot`. The whole pool is scored in one process;
# it is resumable (re-run to skip ids already done), so an interrupted run just continues.
#
# Usage:
#   ./run_localizability.sh                # YFCC4k, all 4000 images, seed 42
#   ./run_localizability.sh yfcc           # same
#   ./run_localizability.sh osv            # OSV-5M, seeded 4000-image subset, seed 42
#   SEED=7 TOTAL_IMAGES=500 ./run_localizability.sh yfcc      # override anything via env
#
# For the join at plot time to work, RESULTS_BASE / SEED must match your attack run.

set -euo pipefail

DATASET="${1:-yfcc}"                      # "yfcc" (all 4000) or "osv" (4000-image subset)
TOTAL_IMAGES="${TOTAL_IMAGES:-4000}"      # size of the seeded image pool to score
SEED="${SEED:-42}"                        # MUST equal the attack run's seed (config.yaml: 42)
NUM_MC_SAMPLES="${NUM_MC_SAMPLES:-256}"   # Monte-Carlo samples for the likelihood estimate

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${HERE}/.." && pwd)"
# Use the plonk conda env's python (the base env lacks plonk); override with PYTHON=...
PYTHON="${PYTHON:-/Data/mathias.ollu/conda/plonk/bin/python}"
# Write next to the attack results so --stage plot can find both. Override with RESULTS_BASE=...
RESULTS_BASE="${RESULTS_BASE:-${PROJECT_DIR}/results/cluster_eval/${DATASET}4k_full}"

echo "=============================================================="
echo "Localizability precompute (local, single GPU)"
echo "Dataset:      ${DATASET}"
echo "Images:       ${TOTAL_IMAGES}   seed=${SEED}   mc_samples=${NUM_MC_SAMPLES}"
echo "Results dir:  ${RESULTS_BASE}"
echo "=============================================================="

cd "${PROJECT_DIR}"
"${PYTHON}" main.py evaluate-localizability \
  --stage compute \
  --dataset "${DATASET}" \
  --n-images "${TOTAL_IMAGES}" \
  --num-mc-samples "${NUM_MC_SAMPLES}" \
  --results-dir "${RESULTS_BASE}" \
  --override "seed=${SEED}"

echo
echo "Done. Scores in ${RESULTS_BASE}/localizability_shards/"
echo "Once the attack results are on disk, plot with:"
echo "  ${PYTHON} main.py evaluate-localizability --stage plot --dataset ${DATASET} \\"
echo "    --attack-types dtd encoder unidef ace sampling diffusion_l2 geoshield \\"
echo "    --results-dir ${RESULTS_BASE} --plots-dir ${RESULTS_BASE}/plots"
