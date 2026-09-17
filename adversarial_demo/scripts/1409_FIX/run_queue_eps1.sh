#!/bin/bash
# 1409_FIX: comprehensive offline evaluation of the saved 1/255 perturbations (no retraining):
# cross-model transfer, sampling steps 2..250, JPEG / blur. Clean-transformed control is reused
# from the 4/255 run (attack- and budget-independent).
set -u
cd /users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
source ~/.bashrc
export MPLBACKEND=Agg TORCH_HOME=/Data/mathias.ollu/torch_cache HF_HOME=/Data/mathias.ollu/hf_cache
PY=/Data/mathias.ollu/conda/plonk/bin/python
DELTAS=/Data/mathias.ollu/results/1409_FIX_eps1/shards
OUT=/Data/mathias.ollu/results/1409_FIX_offline_eps1
LOGS=/Data/mathias.ollu/logs/1409_FIX
Q=$LOGS/queue.log
echo "QUEUE_EPS1 start $(date -Is)" >> $Q
$PY scripts/1409_FIX/offline_eval.py --mode transfer --attacks sampling_v2 dtd_v3_old_forward --deltas-root $DELTAS --out-dir $OUT > $LOGS/offline_eps1_transfer.log 2>&1
echo "STEP_DONE eps1_transfer exit=$? $(date -Is)" >> $Q
$PY scripts/1409_FIX/offline_eval.py --mode steps --steps 2 3 4 5 6 8 10 12 16 32 64 128 250 --attacks sampling_v2 dtd_v3_old_forward --deltas-root $DELTAS --out-dir $OUT > $LOGS/offline_eps1_steps.log 2>&1
echo "STEP_DONE eps1_steps exit=$? $(date -Is)" >> $Q
$PY scripts/1409_FIX/offline_eval.py --mode transforms --attacks sampling_v2 dtd_v3_old_forward --deltas-root $DELTAS --out-dir $OUT > $LOGS/offline_eps1_transforms.log 2>&1
echo "STEP_DONE eps1_transforms exit=$? $(date -Is)" >> $Q
echo "QUEUE_EPS1_DONE $(date -Is)" >> $Q
