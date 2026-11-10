#!/usr/bin/env python
"""Attack success vs number of optimisation steps (OSV-5M or YFCC4k, 1/255, single restart).

Reads the shard states of the n_steps sweep (``<sweep>/steps_<N>/shards/<attack>__w*/``,
produced by run_shard_local.sh with ``attack_train_args.osv.n_steps=N``, NUM_RESTARTS=1,
``per_task_seed=true``) and adds the 60-step point from the full cluster run: restart 0 of
``<ref>/<dataset>_<attack>_results.pt`` on the same image ids (skipped with ``--ref-dir ""``) (the single-restart protocol of
scripts/make_single_restart_tables.py). NB the 60-step reference was trained without
per_task_seed (one seed per shard), so its restart 0 is a different random draw than the
sweep points -- same protocol, different init noise.

Writes ``steps_sweep_<dataset>_eps1.{md,csv}`` into ``--out-dir`` (default: the sweep dir).
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys
import warnings

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from utils.plots.common import _display_attack_name  # noqa: E402

THRESHOLDS = [200.0, 750.0, 2500.0]
ATTACKS = ["dtd", "targeted_l2", "sampling", "training_loss"]
DEFAULT_REF = os.path.join(
    os.path.dirname(HERE), "results", "cluster_eval_24092026_final", "osv_full_results_2609"
)


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return torch.load(path, map_location="cpu", weights_only=False)


def sweep_fsd(sweep_dir: str, n_steps: int, attack: str) -> dict:
    """{image_id: FSD-to-truth (km)} for the finished images of one sweep shard."""
    out = {}
    pattern = os.path.join(sweep_dir, f"steps_{n_steps}", "shards", f"{attack}__w*", "*_eval_state.pt")
    for path in glob.glob(pattern):
        state = _load(path)
        ids = state["source_image_ids"]
        locs = (state.get("location_results") or {}).get(attack) or [[]]
        fsd = state["results"][attack]["final_step_displacement_true"][0]
        for i, img_id in enumerate(ids):
            if i < len(locs[0]) and locs[0][i] is None:
                continue  # not computed yet
            v = float(fsd[i])
            if not math.isnan(v):
                out[str(img_id)] = v
    return out


def reference_fsd(ref_dir: str, dataset: str, attack: str, ids: list, budget_idx: int) -> dict:
    """{image_id: restart-0 FSD-to-truth} of the 60-step full run for ``ids``."""
    res = _load(os.path.join(ref_dir, f"{dataset}_{attack}_results.pt"))
    index = {str(i): k for k, i in enumerate(res["image_ids"])}
    out = {}
    for img_id in ids:
        k = index.get(img_id)
        restarts = res["restart_results"][budget_idx][k] if k is not None else None
        if not restarts or restarts[0] is None:
            continue
        v = restarts[0].get("final_step_displacement_true")
        if v is not None and not math.isnan(float(v)):
            out[img_id] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="osv", choices=["osv", "yfcc"])
    ap.add_argument("--sweep-dir", default=None, help="default: /Data/mathias.ollu/results/steps_sweep_<dataset>_eps1")
    ap.add_argument("--ref-dir", default=DEFAULT_REF)
    ap.add_argument("--ref-budget-idx", type=int, default=0, help="index of 1/255 in the reference run")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    args.sweep_dir = args.sweep_dir or f"/Data/mathias.ollu/results/steps_sweep_{args.dataset}_eps1"
    out_dir = args.out_dir or args.sweep_dir
    name = f"steps_sweep_{args.dataset}_eps1"

    steps = sorted(
        int(m.group(1))
        for d in glob.glob(os.path.join(args.sweep_dir, "steps_*"))
        if (m := re.fullmatch(r"steps_(\d+)", os.path.basename(d)))
    )
    rows = []
    for attack in ATTACKS:
        ids = None
        for n in steps:
            fsd = sweep_fsd(args.sweep_dir, n, attack)
            if fsd:
                rows.append((attack, n, "sweep", fsd))
                ids = sorted(set(ids or fsd) | set(fsd))
        if ids and args.ref_dir:
            rows.append((attack, 60, "ref (full run, restart 0)",
                         reference_fsd(args.ref_dir, args.dataset, attack, ids, args.ref_budget_idx)))

    header = ["attack", "n_steps", "source", "n_images"] + [f"SR@{int(t)}km" for t in THRESHOLDS] + ["mean_FSD_km"]
    table = []
    for attack, n, source, fsd in rows:
        vals = list(fsd.values())
        sr = [100.0 * sum(v > t for v in vals) / len(vals) if vals else float("nan") for t in THRESHOLDS]
        mean = sum(vals) / len(vals) if vals else float("nan")
        table.append([_display_attack_name(attack), n, source, len(vals)] + [round(x, 1) for x in sr] + [round(mean)])

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + table)
    md = [f"_{'OSV-5M' if args.dataset == 'osv' else 'YFCC4k'}, 1/255, 1 restart, FSD to the true location; success = FSD > threshold._", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in table]
    with open(os.path.join(out_dir, f"{name}.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
