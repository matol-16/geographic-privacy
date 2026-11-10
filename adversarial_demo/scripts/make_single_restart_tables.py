#!/usr/bin/env python
"""Rebuild the main results tables for the SINGLE-restart case (restart 0 only).

The main tables report best-of-k restarts: ``final_step_displacement_true`` in
``<ds>_<attack>_results.pt`` is the max over ``restart_results[b][i][*]``. This script
reads the first restart's true-GPS FSD instead and writes, into ``--out-dir``:

  success_rate_<ds>.{md,tex}   same format as scripts/make_success_rate_table.py
  main_results_3budgets.tex    the paper table (``--paper-template``) with the <ds>
                               block refilled (best bold, second best underlined)

``--check`` also recomputes the best-of-k numbers the same way and prints them, to
compare against the existing tables. The cfg / sampling-steps / model-transfer /
JPEG / blur ablations only store aggregates for the selected (best) restart and
cannot be rebuilt for one restart.

Usage:
  python scripts/make_single_restart_tables.py --results-dir <run dir> --dataset yfcc \
      --out-dir <run dir>/tables_single_restart \
      --paper-template <run dir>/tables/main_results_3budgets.tex
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from make_success_rate_table import order_attacks, render_latex, render_markdown  # noqa: E402

THRESHOLDS = [200.0, 750.0, 2500.0]
# Paper-table row label (as in main_results_3budgets.tex) -> attack key.
PAPER_ROWS = {
    "GeoShield": "geoshield",
    "\\fname-$\\mathcal{E}$": "encoder",
    "\\fname-Sample": "sampling",
    "\\fname-AdvDM": "training_loss",
    "\\fname-ACE": "ace",
    "\\fname-DPA": "targeted_l2",
    "\\fname-UniDef": "unidef",
    "\\fname-$L^2$": "diffusion_l2",
    "\\fname-\\mname": "dtd",
}


def load_per_image(results_dir: str, dataset: str, restart: int | None) -> tuple[list, dict]:
    """{attack: [per-budget list of FSDs]}; restart=None -> best over all restarts."""
    import torch
    import yaml

    budgets = [float(b) for b in yaml.safe_load(open(os.path.join(
        results_dir, f"{dataset}_run_config_merged.yaml")))["attack_budgets"][dataset]]
    out = {}
    for path in sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*_results.pt"))):
        attack = os.path.basename(path)[len(dataset) + 1:-len("_results.pt")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = torch.load(path, map_location="cpu", weights_only=False)
        if "restart_results" not in res:
            continue
        per_budget = []
        for per_image in res["restart_results"]:
            vals = []
            for restarts in per_image:
                d = [float(r["final_step_displacement_true"]) for r in (restarts or [])
                     if r is not None and r.get("final_step_displacement_true") is not None]
                d = [x for x in d if not math.isnan(x)]
                if not d:
                    continue
                vals.append(max(d) if restart is None else d[min(restart, len(d) - 1)])
            per_budget.append(vals)
        out[attack] = per_budget
        print(f"  {attack}: n={[len(v) for v in per_budget]}", flush=True)
    return budgets, out


def make_run(dataset: str, budgets: list, per_image: dict) -> dict:
    return {
        "dataset": dataset, "budgets": budgets, "thresholds": THRESHOLDS,
        "success": {a: {t: [sum(x > t for x in v) / len(v) for v in pb] for t in THRESHOLDS}
                    for a, pb in per_image.items()},
        "fsd_mean": {a: [sum(v) / len(v) for v in pb] for a, pb in per_image.items()},
        "n": {a: [len(v) for v in pb] for a, pb in per_image.items()},
    }


def _paper_pct(x: float) -> str:
    s = f"{100 * x:.1f}"
    return f"{100 * x:.2f}" if s == "100.0" and x < 1 else s


def refill_paper_table(template: str, run: dict, dataset_label: str) -> str:
    """Replace the numbers of the ``dataset_label`` block, recomputing bold / underline."""
    rows = [(lab, a) for lab, a in PAPER_ROWS.items() if a in run["success"]]
    # columns: per budget FSD, SR@750, SR@2500
    cols = []
    for bi in range(len(run["budgets"])):
        cols.append(lambda a, bi=bi: run["fsd_mean"][a][bi])
        cols.append(lambda a, bi=bi: run["success"][a][750.0][bi])
        cols.append(lambda a, bi=bi: run["success"][a][2500.0][bi])
    fmts = [lambda v: f"{v:.0f}", _paper_pct, _paper_pct] * len(run["budgets"])
    ranks = []
    for g in cols:
        vals = sorted({round(g(a), 12) for _, a in rows}, reverse=True)
        ranks.append(vals[:2])

    def cells(a):
        out = []
        for g, f, (best, *second) in zip(cols, fmts, ranks):
            v = round(g(a), 12)
            s = f(g(a))
            out.append(f"\\bfseries {s}" if v == best else f"\\underline{{{s}}}" if second and v == second[0] else s)
        return " & ".join(out)

    lines = template.split("\n")
    start = next(i for i, l in enumerate(lines) if f"\\textbf{{{dataset_label}}}" in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("& Clean"))
    for i in range(start, end):
        label = lines[i].lstrip("& ").split(" \\citeyearpar")[0].strip()
        a = PAPER_ROWS.get(label)
        if a is None or a not in run["success"] or not lines[i + 1].lstrip().startswith("&"):
            continue
        lines[i + 1] = "  & " + cells(a) + " \\\\"
    text = "\n".join(lines)
    text = text.replace("\\label{tab:main_results}", "\\label{tab:main_results_1restart}")
    text = re.sub(r"(Best per column in\s*\n?\\textbf\{bold\})",
                  r"Single restart (no best-of-$k$ selection). \1", text, count=1)
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", default="yfcc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--restart", type=int, default=0, help="restart index to report (0 = first)")
    ap.add_argument("--paper-template", help="main_results_3budgets.tex to refill")
    ap.add_argument("--check", action="store_true", help="also print the best-of-k table")
    args = ap.parse_args()

    label = {"yfcc": "YFCC4k", "osv": "OSV-5M"}.get(args.dataset, args.dataset)
    budgets, per_image = load_per_image(args.results_dir, args.dataset, args.restart)
    run = make_run(args.dataset, budgets, per_image)
    attacks = order_attacks(set(run["success"]))
    n_img = max(v for vs in run["n"].values() for v in vs)

    os.makedirs(args.out_dir, exist_ok=True)
    stem = f"success_rate_{args.dataset}"
    md = render_markdown([run], attacks)
    tex = render_latex([run], attacks, f"tab:{stem}_1restart",
                       f"Attack success rate (\\%, higher is better) at 200/750/2500\\,km and mean "
                       f"final-step displacement (km) on {label} ({n_img} images), per budget "
                       f"$b$, single restart (no best-of-$k$ selection).")
    written = []
    for ext, body in (("md", "_Single restart (restart index "
                       f"{args.restart}); displacement to the true location._\n\n" + md), ("tex", tex)):
        path = os.path.join(args.out_dir, f"{stem}.{ext}")
        open(path, "w").write(body + "\n")
        written.append(path)
    print(f"\n### {stem} (single restart)\n{md}")

    if args.paper_template:
        path = os.path.join(args.out_dir, os.path.basename(args.paper_template))
        open(path, "w").write(refill_paper_table(open(args.paper_template).read(), run, label))
        written.append(path)

    if args.check:
        _, best = load_per_image(args.results_dir, args.dataset, None)
        brun = make_run(args.dataset, budgets, best)
        print(f"\n### {stem} (best of all restarts, check)\n{render_markdown([brun], attacks)}")
    print("\nWritten:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
