#!/bin/bash
# 1409_FIX: remaining attacks on the 100 images -- GeoFA-L2 v2/v3, GeoFA-L2 targeted (antipode) at 4/255
# (encoder / AdvDM v1+v3 already exist there) and encoder v3, AdvDM v3, L2 v3, L2 targeted at 1/255.
# Sequential on one GPU; launch with `at now` (survives session restarts); every run is resumable.
set -u
cd /users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
source ~/.bashrc
export MPLBACKEND=Agg TORCH_HOME=/Data/mathias.ollu/torch_cache HF_HOME=/Data/mathias.ollu/hf_cache
PY=/Data/mathias.ollu/conda/plonk/bin/python
LOGS=/Data/mathias.ollu/logs/1409_FIX
Q=$LOGS/queue.log
run() {  # run <config> <results_dir> <attack> <log>
  echo "1409_FIX start $(date -Is) attack=$3 config=$1" > "$4"
  $PY main.py evaluate-dataset-shard --config "$1" --dataset yfcc --attack-types "$3" \
    --total-images 4000 --images-per-shard 100 --image-shard-index 0 \
    --results-dir "$2" --parallel-workers 1 --max-restarts 1 >> "$4" 2>&1
  echo "RUN_EXIT_CODE=$? attack=$3 end=$(date -Is)" >> "$4"
  echo "STEP_DONE rest_$(basename "$2")_$3 $(date -Is)" >> $Q
}
echo "QUEUE_REST start $(date -Is)" >> $Q
for A in diffusion_l2_v2 diffusion_l2_v3 l2_target; do
  run configs/1409_FIX/config.yaml /Data/mathias.ollu/results/1409_FIX $A $LOGS/$A.log
done
for A in encoder_v3 training_loss_v3 diffusion_l2_v3 l2_target; do
  run configs/1409_FIX/config_eps1.yaml /Data/mathias.ollu/results/1409_FIX_eps1 $A $LOGS/${A}_eps1.log
done
echo "QUEUE_REST_DONE $(date -Is)" >> $Q
