#!/bin/bash
# 1409_FIX: copy the (small) shard state files into results/1409_FIX/data and write results/1409_FIX/report.md
ROOT=/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo
DATA=/Data/mathias.ollu/results/1409_FIX
source ~/.bashrc
cd "$ROOT"
mkdir -p results/1409_FIX/data
rsync -a --exclude='*.png' "$DATA/shards/" results/1409_FIX/data/
/Data/mathias.ollu/conda/plonk/bin/python scripts/1409_FIX/write_ablation_report.py > /dev/null
/Data/mathias.ollu/conda/plonk/bin/python scripts/1409_FIX/write_offline_reports.py > /dev/null
/Data/mathias.ollu/conda/plonk/bin/python scripts/1409_FIX/write_eps1_report.py > /dev/null
/Data/mathias.ollu/conda/plonk/bin/python scripts/1409_FIX/write_report.py
