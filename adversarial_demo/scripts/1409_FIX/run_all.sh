#!/bin/bash
# 1409_FIX: run GeoFA-TD (dtd), GeoFA-enc. (encoder), GeoFA-AdvDM (training_loss)
# sequentially on the local GPU, then build the report. Logs: /Data/mathias.ollu/logs/1409_FIX/.
ROOT=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
LOGS=/Data/mathias.ollu/logs/1409_FIX
mkdir -p "$LOGS"
for a in dtd encoder training_loss dtd_v2 geoshield; do
  bash "$ROOT/scripts/1409_FIX/run_attack.sh" "$a" > "$LOGS/$a.log" 2>&1
done
bash "$ROOT/scripts/1409_FIX/make_report.sh" > "$LOGS/report.log" 2>&1
echo "ALL_DONE $(date -Is)"
