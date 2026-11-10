#!/usr/bin/env python
"""Restart ablation against the TRUE location: success rate of best-of-k vs. k, one panel per budget.

``main.py plot restarts`` draws the ``<ds>_restarts_results.json`` curves, which hold the
displacement to the clean prediction. This figure instead uses the per-restart
``final_step_displacement_true`` stored in ``<ds>_<attack>_results.pt`` -- the metric the
restarts are selected on -- so its k = max point equals the headline success rate.

``--drop-restarts 1`` leaves restart indices out (e.g. one that duplicates restart 0), so
the x axis counts distinct restarts.

Usage:
  python scripts/plot_restart_ablation_true.py --results-dir <run> --plots-dir <run>/plots \
      [--dataset yfcc] [--threshold 2500] [--drop-restarts 1] [--filename-suffix _appendix]
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from make_ablation_tables import _best_so_far, _restart_curves_from_pt  # noqa: E402
from make_success_rate_table import budget_label, order_attacks  # noqa: E402
from utils.plots.common import _display_attack_name  # noqa: E402
from utils.plots.style import attack_color  # noqa: E402

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--plots-dir", required=True)
    ap.add_argument("--dataset", default="yfcc")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[2500.0])
    ap.add_argument("--drop-restarts", type=int, nargs="*", default=[])
    ap.add_argument("--filename-suffix", default="")
    args = ap.parse_args()

    curves, budgets, _ = _restart_curves_from_pt(args.results_dir, args.dataset, drop=tuple(args.drop_restarts))
    attacks = order_attacks({a for a, _ in curves})
    kmax = min(len(r) for rows in curves.values() for r in rows)
    ks = list(range(1, kmax + 1))

    rows_t = args.thresholds
    fig, axes = plt.subplots(len(rows_t), len(budgets), figsize=(4.2 * len(budgets), 3.4 * len(rows_t)),
                             sharex=True, sharey="row", squeeze=False)
    handles = {}
    for ti, thr in enumerate(rows_t):
        for bi, b in enumerate(budgets):
            ax = axes[ti][bi]
            for ai, a in enumerate(attacks):
                rows = [_best_so_far(r[:kmax]) for r in curves.get((a, b), []) if len(r) >= kmax]
                if not rows:
                    continue
                ys = [100 * sum(c[k - 1] > thr for c in rows) / len(rows) for k in ks]
                line, = ax.plot(ks, ys, color=attack_color(a), marker=MARKERS[ai % len(MARKERS)],
                                markersize=5, linewidth=2,
                                linestyle="--" if a == "sampling_cfg10" else "-")
                handles[a] = line
            ax.set_xticks(ks)
            ax.grid(True, alpha=0.3)
            if ti == 0:
                ax.set_title(f"$b = {budget_label(b)}$")
            if ti == len(rows_t) - 1:
                ax.set_xlabel("distinct restarts $k$" if args.drop_restarts else "restarts $k$")
            if bi == 0:
                ax.set_ylabel(f"ASR@{thr:.0f} km (%)")
    fig.legend([handles[a] for a in attacks if a in handles],
               [_display_attack_name(a) for a in attacks if a in handles],
               loc="upper center", ncol=min(5, len(handles)), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout()
    os.makedirs(args.plots_dir, exist_ok=True)
    stem = f"{args.dataset}_restarts_success_rate_true{args.filename_suffix}"
    for ext in ("png", "pdf"):
        path = os.path.join(args.plots_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
