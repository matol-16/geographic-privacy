#!/bin/bash
# 1409_FIX ablations: retrain ONE attack on the same 100 YFCC4k images / 4/255 (seed 42) with the
# eval-time ablations enabled: sampling steps (eval_num_steps), cfg (eval_cfgs) and cross-model
# transfer (model_transfer.model_types: rfm / diffusion / flow). Separate results dir so the main
# 1409_FIX shard states are untouched. Usage: scripts/1409_FIX/run_ablation.sh <attack_key>
set -u
ATTACK="${1:?attack key required}"
ROOT=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
DATA=/Data/mathias.ollu
source ~/.bashrc
cd "$ROOT"
export MPLBACKEND=Agg TORCH_HOME=$DATA/torch_cache HF_HOME=$DATA/hf_cache
mkdir -p "$DATA/results/1409_FIX_ablations" "$DATA/logs/1409_FIX"
echo "1409_FIX ablation start $(date -Is) host=$(hostname) attack=$ATTACK"
$DATA/conda/plonk/bin/python main.py evaluate-dataset-shard \
  --config configs/1409_FIX/config.yaml --dataset yfcc --attack-types "$ATTACK" \
  --total-images 4000 --images-per-shard 100 --image-shard-index 0 \
  --results-dir "$DATA/results/1409_FIX_ablations" --parallel-workers 1 --max-restarts 1 \
  --run-sampling-steps-ablation --run-cfg-ablation --run-model-transfer-ablation
echo "RUN_EXIT_CODE=$? attack=$ATTACK end=$(date -Is)"
