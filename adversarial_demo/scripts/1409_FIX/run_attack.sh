#!/bin/bash
# 1409_FIX: evaluate ONE attack (pixel-space fix, single run / no restarts) on the
# first 100 YFCC4k images (shard 0 of 4000), budgets 4/255 and 8/255.
# Usage: scripts/1409_FIX/run_attack.sh <attack_key>
#   attack keys (post-renaming): dtd = GeoFA-TD, encoder = GeoFA-enc., training_loss = GeoFA-AdvDM,
#   dtd_v2 = GeoFA-TD v2 (geodesic RFM forward process, tangent cosine), geoshield = GeoShield (PNG, 336 px),
#   dtd_v3 / training_loss_v3 = v2 / AdvDM with x_t noised from the TRUE location (paper Eq. 1/9/15),
#   dtd_v3_old_forward = v1 (Gaussian x_t) + TRUE x0 (isolates the x0 effect),
#   encoder_v3 = GeoFA-enc. with the shared sign-SGD PGD loop, sampling_v2 = GeoFA-Sampling vs the TRUE location,
#   l2_v2 / l2_v3 = GeoFA-L2 with geodesic x_t (clean-model / TRUE x0)
set -u
ATTACK="${1:?attack key required (dtd | encoder | training_loss | dtd_v2 | dtd_v3 | dtd_v3_old_forward | training_loss_v3 | encoder_v3 | sampling_v2 | l2_v2 | l2_v3 | geoshield)}"
ROOT=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
DATA=/Data/mathias.ollu
source ~/.bashrc
cd "$ROOT"
export MPLBACKEND=Agg TORCH_HOME=$DATA/torch_cache HF_HOME=$DATA/hf_cache
mkdir -p "$DATA/results/1409_FIX" "$DATA/logs/1409_FIX"
echo "1409_FIX start $(date -Is) host=$(hostname) attack=$ATTACK"
$DATA/conda/plonk/bin/python main.py evaluate-dataset-shard \
  --config configs/1409_FIX/config.yaml --dataset yfcc --attack-types "$ATTACK" \
  --total-images 4000 --images-per-shard 100 --image-shard-index 0 \
  --results-dir "$DATA/results/1409_FIX" --parallel-workers 1 --max-restarts 1
echo "RUN_EXIT_CODE=$? attack=$ATTACK end=$(date -Is)"
