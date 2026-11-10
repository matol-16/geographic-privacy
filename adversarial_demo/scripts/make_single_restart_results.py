#!/usr/bin/env python
"""Write single-restart copies of a run's ``<ds>_<attack>_results.pt`` files.

The merged results keep the best of k restarts in their top-level tensors
(``final_step_displacement_{true,predicted}``, ``location_results``). This script
rebuilds every file from ONE restart (``--restart``, default 0) of ``restart_results``:

  final_step_displacement_{true,predicted}  that restart's value (NaN if missing)
  restart_results[b][i]                      [that restart] only
  location_results[b][i]                     best_restart = that restart, its GPS samples
  final_loss                                 NaN (only stored for the selected restart)

Pointing ``main.py plot`` / ``scripts/make_paper_figures.sh`` at ``--out-dir`` then draws
every .pt-backed figure (displacement / success rate vs budget, spread, heatmaps,
localizability) for the single-restart case. The run config and localizability scores
are symlinked next to them; the JSON ablations are NOT (they only hold best-restart
aggregates, so they cannot be redone per restart).

Usage:
  python scripts/make_single_restart_results.py --results-dir <run dir> --dataset yfcc \
      --out-dir /Data/.../single_restart_yfcc
"""
from __future__ import annotations

import argparse
import glob
import os
import warnings


def _value(r, key):
    v = None if r is None else r.get(key)
    return float("nan") if v is None else float(v)


def single_restart(res: dict, restart: int) -> dict:
    import torch

    out = {k: v for k, v in res.items()
           if k not in ("restart_results", "location_results", "final_loss")}
    pick = [[(rs[restart] if rs and restart < len(rs) else None) for rs in per_image]
            for per_image in res["restart_results"]]
    for key in ("final_step_displacement_true", "final_step_displacement_predicted"):
        if key in res:
            out[key] = torch.tensor([[_value(r, key) for r in pb] for pb in pick],
                                    dtype=res[key].dtype)
    if "final_loss" in res:
        out["final_loss"] = torch.full_like(res["final_loss"], float("nan"))
    out["restart_results"] = [[[r] if r is not None else [] for r in pb] for pb in pick]
    if "location_results" in res:
        locs = []
        for pb, loc_b in zip(pick, res["location_results"]):
            row = []
            for r, loc in zip(pb, loc_b):
                loc = dict(loc) if loc else {}
                if r is not None:
                    loc.update(best_restart=restart, true_gps=r.get("true_gps"),
                               predicted_gps_source=r.get("gps_source"),
                               predicted_gps_perturbed=r.get("gps_perturbed"))
                row.append(loc)
            locs.append(row)
        out["location_results"] = locs
    return out


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", default="yfcc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--restart", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for name in (f"{args.dataset}_run_config_merged.yaml", f"{args.dataset}_localizability.pt"):
        src, dst = os.path.join(os.path.abspath(args.results_dir), name), os.path.join(args.out_dir, name)
        if os.path.exists(src) and not os.path.lexists(dst):
            os.symlink(src, dst)

    for path in sorted(glob.glob(os.path.join(args.results_dir, f"{args.dataset}_*_results.pt"))):
        if " (" in os.path.basename(path):  # browser-download duplicates, e.g. "x (1).pt"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = torch.load(path, map_location="cpu", weights_only=False)
        if "restart_results" not in res:
            print(f"skip {os.path.basename(path)}: no restart_results")
            continue
        out = single_restart(res, args.restart)
        t = out["final_step_displacement_true"]
        means = [float(row[~torch.isnan(row)].mean()) for row in t]
        n = [int((~torch.isnan(row)).sum()) for row in t]
        torch.save(out, os.path.join(args.out_dir, os.path.basename(path)))
        print(f"{os.path.basename(path)}: n={n} mean FSD true={[round(m) for m in means]}", flush=True)


if __name__ == "__main__":
    main()
