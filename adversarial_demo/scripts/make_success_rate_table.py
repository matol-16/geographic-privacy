#!/usr/bin/env python
"""Build the paper's main results ("accuracy") table from saved plot sidecars.

Reads the ``<dataset>_<attacks>_attack_success_rate.json`` and
``<dataset>_<attacks>_final_step_displacement_true.json`` sidecars that
``main.py plot success-rate`` / ``plot results`` write next to the figures, and
emits one table per dataset plus a combined YFCC4k + OSV-5M table:

  * success rate (%) at every stored threshold (200 / 750 / 2500 km) for every budget,
  * mean final-step displacement (km) per budget,

as LaTeX (booktabs, paper-ready) and Markdown (quick reading). Attack labels go
through ``utils.plots.common._display_attack_name`` so they follow the GeoFA naming:
GeoShield (the one external baseline) is listed first, then the GeoFA instances with
the headline GeoFA-TD last and in bold, and the best value of each column in bold.

No GPU / no .pt loading: the sidecars already hold the aggregated numbers.

Usage (defaults = the current full runs):
  python scripts/make_success_rate_table.py
  python scripts/make_success_rate_table.py --out-dir results/tables \
      --run yfcc=results/cluster_eval/yfcc4k_full/plots \
      --run osv=results/cluster_eval_2/osv_full/plots
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from utils.plots.common import _display_attack_name  # noqa: E402

DATASET_LABELS = {"yfcc": "YFCC4k", "osv": "OSV-5M"}
# GeoShield (external pipeline) first, then the GeoFA instances -- prior losses
# re-implemented in the framework (ACE, UniDef, AdvDM), then the new ones, with the
# headline GeoFA-TD last. Unknown keys go in between.
BASELINES = ["geoshield"]
GEOFA_FAMILY = ["ace", "unidef", "training_loss", "encoder", "sampling", "diffusion_l2",
                "targeted_l2", "dtd_blur", "dtd"]


def _find_sidecar(plots_dir: str, dataset: str, suffix: str) -> str:
    matches = sorted(glob.glob(os.path.join(plots_dir, f"{dataset}_*_{suffix}.json")), key=len)
    if not matches:
        raise FileNotFoundError(f"no {dataset}_*_{suffix}.json under {plots_dir} "
                                f"(run `main.py plot success-rate` / `plot results` first)")
    # Prefer the sidecar covering the most attacks (longest attack list in the stem).
    return matches[-1]


def load_run(plots_dir: str, dataset: str) -> dict:
    sr = json.load(open(_find_sidecar(plots_dir, dataset, "attack_success_rate")))
    fsd_path = os.path.join(plots_dir, os.path.basename(_find_sidecar(plots_dir, dataset, "attack_success_rate"))
                            .replace("attack_success_rate", "final_step_displacement_true"))
    fsd = json.load(open(fsd_path)) if os.path.exists(fsd_path) else {"attacks": {}}
    if sr.get("metric") != "final_step_displacement_true":
        print(f"WARNING: {dataset} success-rate sidecar uses metric {sr.get('metric')!r}, "
              "not the true-GPS metric", file=sys.stderr)
    return {
        "dataset": dataset,
        "budgets": [float(b) for b in sr["attack_budgets"]],
        "thresholds": [float(t) for t in sr["thresholds_km"]],
        "success": {a: {float(t): v for t, v in d["success_rate"].items()} for a, d in sr["attacks"].items()},
        "fsd_mean": {a: d["mean"] for a, d in fsd.get("attacks", {}).items()},
        "n": {a: d["n_samples_per_budget"] for a, d in sr["attacks"].items()},
    }


def order_attacks(attacks) -> list[str]:
    known = BASELINES + GEOFA_FAMILY
    rest = sorted(a for a in attacks if a not in known)
    return [a for a in BASELINES if a in attacks] + rest + [a for a in GEOFA_FAMILY if a in attacks]


def budget_label(b: float) -> str:
    n = round(b * 255)
    return f"{n}/255" if abs(n / 255 - b) < 2e-3 else f"{b:g}"


def _fmt_pct(x) -> str:
    return "--" if x is None else f"{100 * x:.1f}"


def _fmt_km(x) -> str:
    return "--" if x is None else f"{x:,.0f}"


def build_columns(runs: list[dict]):
    """Column spec: (dataset, budget, kind, threshold|None) -> getter(attack) -> float|None."""
    cols = []
    for run in runs:
        for bi, b in enumerate(run["budgets"]):
            for t in run["thresholds"]:
                cols.append((run["dataset"], b, "sr", t,
                             (lambda r, bi, t: lambda a: (r["success"].get(a, {}).get(t) or [None] * 9)[bi])(run, bi, t)))
            if run["fsd_mean"]:
                cols.append((run["dataset"], b, "fsd", None,
                             (lambda r, bi: lambda a: (r["fsd_mean"].get(a) or [None] * 9)[bi])(run, bi)))
    return cols


def render_markdown(runs: list[dict], attacks: list[str]) -> str:
    cols = build_columns(runs)
    head = ["Attack"] + [f"{DATASET_LABELS.get(d, d)} {budget_label(b)} " +
                         (f"SR@{t:.0f}km (%)" if k == "sr" else "mean FSD (km)") for d, b, k, t, _ in cols]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    best = [max((g(a) for a in attacks if g(a) is not None), default=None) for *_, g in cols]
    for a in attacks:
        cells = []
        for (d, b, k, t, g), bmax in zip(cols, best):
            v = g(a)
            s = _fmt_pct(v) if k == "sr" else _fmt_km(v)
            cells.append(f"**{s}**" if v is not None and bmax is not None and v >= bmax else s)
        name = _display_attack_name(a)
        lines.append("| " + " | ".join([f"**{name}**" if a == "dtd" else name] + cells) + " |")
    return "\n".join(lines)


def render_latex(runs: list[dict], attacks: list[str], label: str, caption: str) -> str:
    cols = build_columns(runs)
    per_budget = [len(r["thresholds"]) + (1 if r["fsd_mean"] else 0) for r in runs]
    per_run = [len(r["budgets"]) * n for r, n in zip(runs, per_budget)]
    best = [max((g(a) for a in attacks if g(a) is not None), default=None) for *_, g in cols]

    out = []
    out.append("% Auto-generated by scripts/make_success_rate_table.py -- do not edit by hand.")
    out.append("% SR@d = success rate (%) = fraction of images whose final-step displacement")
    out.append("% (vs. true GPS) exceeds d km; FSD = mean final-step displacement (km).")
    out.append("\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}")
    out.append("\\begin{tabular}{l" + "".join("|" + "r" * n for n in per_run) + "}")
    out.append("\\toprule")
    # dataset header row
    hdr = [""]
    for r, n in zip(runs, per_run):
        hdr.append(f"\\multicolumn{{{n}}}{{c}}{{{DATASET_LABELS.get(r['dataset'], r['dataset'])}}}")
    out.append(" & ".join(hdr) + " \\\\")
    # budget header row
    hdr = [""]
    for r, n in zip(runs, per_budget):
        for b in r["budgets"]:
            hdr.append(f"\\multicolumn{{{n}}}{{c}}{{$\\epsilon={budget_label(b)}$}}")
    out.append(" & ".join(hdr) + " \\\\")
    # metric header row
    hdr = ["Attack"]
    for d, b, k, t, _ in cols:
        hdr.append(f"SR@{t:.0f}" if k == "sr" else "FSD")
    out.append(" & ".join(hdr) + " \\\\\n\\midrule")
    prev_group = None
    for a in attacks:
        group = "geofa" if a in GEOFA_FAMILY else "baseline"
        if prev_group == "baseline" and group == "geofa":
            out.append("\\midrule")
        prev_group = group
        cells = []
        for (d, b, k, t, g), bmax in zip(cols, best):
            v = g(a)
            s = _fmt_pct(v) if k == "sr" else _fmt_km(v).replace(",", "\\,")
            cells.append(f"\\textbf{{{s}}}" if v is not None and bmax is not None and v >= bmax else s)
        name = _display_attack_name(a).replace("\u00b2", "$^{2}$")
        if a == "dtd":
            name = f"\\textbf{{{name}}}"
        out.append(" & ".join([name] + cells) + " \\\\")
    out.append("\\bottomrule\n\\end{tabular}")
    out.append(f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table*}}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", metavar="DATASET=PLOTS_DIR",
                    help="plots dir holding a dataset's sidecars; repeatable (default: yfcc4k_full + osv_full)")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(HERE), "results", "tables"))
    args = ap.parse_args()

    root = os.path.dirname(HERE)
    specs = args.run or [
        f"yfcc={os.path.join(root, 'results', 'cluster_eval', 'yfcc4k_full', 'plots')}",
        f"osv={os.path.join(root, 'results', 'cluster_eval_2', 'osv_full', 'plots')}",
    ]
    runs = []
    for spec in specs:
        ds, _, plots_dir = spec.partition("=")
        runs.append(load_run(plots_dir, ds))
        n = runs[-1]["n"]
        print(f"[{ds}] {plots_dir}: attacks={list(n)} n_images={sorted(set(v for vs in n.values() for v in vs))}")

    os.makedirs(args.out_dir, exist_ok=True)
    written = []

    def emit(stem, sel_runs, caption):
        attacks = order_attacks(set().union(*(set(r["success"]) for r in sel_runs)))
        md = render_markdown(sel_runs, attacks)
        tex = render_latex(sel_runs, attacks, f"tab:{stem}", caption)
        for ext, body in (("md", md), ("tex", tex)):
            path = os.path.join(args.out_dir, f"{stem}.{ext}")
            open(path, "w").write(body + "\n")
            written.append(path)
        print(f"\n### {stem}\n{md}")

    for r in runs:
        ds = r["dataset"]
        n_img = max(v for vs in r["n"].values() for v in vs)
        emit(f"success_rate_{ds}", [r],
             f"Attack success rate (\\%, higher is better) at 200/750/2500\\,km and mean final-step "
             f"displacement (km) on {DATASET_LABELS.get(ds, ds)} ({n_img} images), per budget $\\epsilon$.")
    if len(runs) > 1:
        emit("success_rate_all", runs,
             "Attack success rate (\\%, higher is better) at 200/750/2500\\,km and mean final-step "
             "displacement (km) on " + " and ".join(DATASET_LABELS.get(r["dataset"], r["dataset"]) for r in runs)
             + ", per budget $\\epsilon$. Best value per column in bold.")
        # Compact headline table: 2500 km success rate only.
        compact = []
        for r in runs:
            rr = dict(r); rr["thresholds"] = [t for t in r["thresholds"] if t == 2500.0] or r["thresholds"][-1:]
            rr["fsd_mean"] = {}
            compact.append(rr)
        emit("success_rate_2500km", compact,
             "Attack success rate (\\%) at the 2500\\,km threshold, per dataset and budget $\\epsilon$.")

    print("\nWritten:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
