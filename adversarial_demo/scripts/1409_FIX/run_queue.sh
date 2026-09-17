#!/bin/bash
# 1409_FIX follow-up queue (one GPU, strictly sequential):
#   1. schedulers : offline re-evaluation of the saved 4/255 deltas under other inference
#                   schedules / integrators / step counts (no retraining)
#   2. eps 1/255  : retrain both attacks at the smaller budget (deltas saved)
#   3. transforms : offline JPEG / blur robustness of the saved 4/255 deltas
set -u
ROOT=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
DATA=/Data/mathias.ollu
LOGS=$DATA/logs/1409_FIX
source ~/.bashrc
cd "$ROOT"
export MPLBACKEND=Agg TORCH_HOME=$DATA/torch_cache HF_HOME=$DATA/hf_cache
PY=$DATA/conda/plonk/bin/python

echo "QUEUE start $(date -Is)"
$PY scripts/1409_FIX/offline_eval.py --mode schedulers \
    --attacks sampling_v2 dtd_v3_old_forward > $LOGS/offline_schedulers.log 2>&1
echo "STEP_DONE schedulers exit=$? $(date -Is)"

for A in sampling_v2 dtd_v3_old_forward; do
  echo "1409_FIX eps1 start $(date -Is) attack=$A" > $LOGS/${A}_eps1.log
  $PY main.py evaluate-dataset-shard \
    --config configs/1409_FIX/config_eps1.yaml --dataset yfcc --attack-types "$A" \
    --total-images 4000 --images-per-shard 100 --image-shard-index 0 \
    --results-dir "$DATA/results/1409_FIX_eps1" --parallel-workers 1 --max-restarts 1 \
    >> $LOGS/${A}_eps1.log 2>&1
  echo "RUN_EXIT_CODE=$? attack=$A end=$(date -Is)" >> $LOGS/${A}_eps1.log
  echo "STEP_DONE eps1_$A $(date -Is)"
done

$PY scripts/1409_FIX/offline_eval.py --mode transforms \
    --attacks sampling_v2 dtd_v3_old_forward > $LOGS/offline_transforms.log 2>&1
echo "STEP_DONE transforms exit=$? $(date -Is)"
echo "QUEUE_DONE $(date -Is)"
