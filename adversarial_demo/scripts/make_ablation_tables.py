#!/usr/bin/env python
"""Build the ablation tables (attack success rate, every budget) from a run's ablation JSONs.

Companion of ``make_success_rate_table.py`` (main results) for the five ablations that
``merge-shards`` writes next to the merged results:

  restarts         <ds>_restarts_results.json        best-of-k, k = 1..max_restarts
  cfg              <ds>_cfg_results.json             guidance scale w at evaluation
  sampling_steps   <ds>_sampling_steps_results.json  sampler steps at evaluation
  model_transfer   <ds>_model_transfer_results.json  RFM (attacked) -> diffusion / flow
  jpeg, blur       <ds>_robustness_results.json      JPEG quality / Gaussian blur sigma
                                                     (displacement vs. the true GPS)

For every ablation and every success-rate threshold it writes one table
``<ds>_<ablation>_sr<thr>.{md,tex}``: one row per attack, one column group per budget,
one column per ablation setting, values = success rate (%), best of each column in
bold. Attack order and labels follow ``make_success_rate_table.py``.

Metric. The robustness tables use the displacement to the TRUE location. The cfg /
sampling-steps / model-transfer JSONs only hold the paired displacement between the
attacked and the CLEAN prediction (``metrics["final_step_displacement"]``), so those
tables are labelled "vs. clean prediction"; a ground-truth version needs the per-image
``ablation_gps`` batches that only ``save_deltas=true`` runs keep.

The restart table is rebuilt from the per-restart ``final_step_displacement_true`` in
``<ds>_<attack>_results.pt`` (the metric restarts are selected on) when those files are
present, else from the JSON's ``best_after_k`` curves (vs. clean prediction). Only images
with the full restart depth count; GeoShield has no restarts and is absent.
``--drop-restarts 1`` additionally writes a ``restarts_distinct`` table without the given
restart indices (e.g. a restart that duplicates restart 0 because of a shared RNG state).

Usage:
  python scripts/make_ablation_tables.py --results-dir <run dir> --dataset yfcc --out-dir <dir>
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from make_success_rate_table import GEOFA_FAMILY, budget_label, order_attacks  # noqa: E402
from utils.plots.common import _display_attack_name  # noqa: E402

TITLES = {
    "restarts": ("Restarts $k$", "best of $k$ restarts"),
    "restarts_distinct": ("Restarts $k$", "best of $k$ distinct restarts"),
    "cfg": ("Guidance $w$", "classifier-free guidance scale $w$ at evaluation"),
    "sampling_steps": ("Steps", "number of sampling steps at evaluation"),
    "model_transfer": ("Model", "evaluated model (attacks optimised on the RFM model)"),
    "jpeg": ("JPEG $q$", "JPEG compression quality $q$ applied to the attacked image"),
    "blur": ("Blur $\\sigma$", "Gaussian blur $\\sigma$ applied to the attacked image"),
}


def _key(x) -> str:
    """Setting key as stored by utils/ablations.py (ints without a trailing '.0')."""
    return str(int(x)) if float(x).is_integer() else str(x)


def _bkey(b: float) -> str:
    return f"budget_{b:.6f}"


# Displacement each ablation's success rate is computed from.
METRIC_TRUE = "true location"
METRIC_CLEAN = "clean prediction"


def _best_so_far(xs):
    out, best = [], -math.inf
    for x in xs:
        best = max(best, x)
        out.append(best)
    return out


def _restart_curves_from_pt(results_dir: str, dataset: str, drop=()) -> tuple[dict, list, int]:
    """{(attack, budget): [best-of-k curve per image]} from the per-restart true-GPS FSD."""
    import torch

    curves, budgets, depth = {}, None, 0
    for path in sorted(glob.glob(os.path.join(results_dir, f"{dataset}_*_results.pt"))):
        attack = os.path.basename(path)[len(dataset) + 1:-len("_results.pt")]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = torch.load(path, map_location="cpu", weights_only=False)
        if budgets is None:
            args_path = os.path.join(results_dir, f"{dataset}_run_config_merged.yaml")
            import yaml
            budgets = [float(b) for b in yaml.safe_load(open(args_path))["attack_budgets"][dataset]]
        for bi, per_image in enumerate(res.get("restart_results", [])):
            rows = []
            for restarts in per_image:
                disps = [float(r["final_step_displacement_true"]) for r in (restarts or [])
                         if r is not None and r.get("final_step_displacement_true") is not None]
                if len(disps) < 2:  # single-shot attacks (GeoShield) have no restart axis
                    continue
                disps = [d for i, d in enumerate(disps) if i not in drop]
                rows.append(disps)
            if rows:
                depth = max(depth, max(len(r) for r in rows))
                curves[(attack, budgets[bi])] = rows
        print(f"  restarts: {attack} loaded", flush=True)
    return curves, budgets or [], depth


def _restart_ablation(curves: dict, budgets: list, kmax: int, thresholds: list, metric: str) -> dict:
    rate, n_used = {}, {}
    for (a, b), rows in curves.items():
        full = [_best_so_far(r[:kmax]) for r in rows if len(r) >= kmax]
        n_used[(a, b)] = len(full)
        if not full:
            continue
        for k in range(1, kmax + 1):
            for t in thresholds:
                rate[(a, b, str(k), t)] = sum(c[k - 1] > t for c in full) / len(full)
    return {"settings": [str(k) for k in range(1, kmax + 1)], "labels": [str(k) for k in range(1, kmax + 1)],
            "budgets": budgets, "thresholds": thresholds, "rate": rate, "n_images": n_used,
            "nan_mean": [], "metric": metric}


def load_ablations(results_dir: str, dataset: str, drop_restarts=()) -> dict:
    """{ablation: {"settings": [...], "budgets": [...], "thresholds": [...],
                   "rate": {(attack, budget, setting, thr): float}}}"""
    out = {}

    def jload(name):
        path = os.path.join(results_dir, f"{dataset}_{name}_results.json")
        return json.load(open(path)) if os.path.exists(path) else None

    def sweep(name, d, settings, getter, labels=None, metric=METRIC_CLEAN):
        rate = {}
        for a in d["attack_types"]:
            for b in d["attack_budgets"]:
                node = d["results"].get(a, {}).get(_bkey(b))
                if not node:
                    continue
                for s in settings:
                    cell = getter(node, s)
                    if not cell:
                        continue
                    for t, v in cell["success_rates"].items():
                        rate[(a, b, s, float(t))] = v
        out[name] = {
            "settings": settings, "labels": labels or [str(s) for s in settings],
            "budgets": d["attack_budgets"],
            "thresholds": [float(t) for t in d["success_rate_thresholds_km"]], "rate": rate,
            "nan_mean": sorted({(a, s) for a in d["attack_types"] for b in d["attack_budgets"]
                                for s in settings
                                if (c := getter(d["results"].get(a, {}).get(_bkey(b)) or {}, s))
                                and isinstance(c.get("mean_displacement_km"), float)
                                and math.isnan(c["mean_displacement_km"])}),
            "metric": metric,
        }

    if d := jload("cfg"):
        sweep("cfg", d, [_key(w) for w in d["eval_cfgs"]], lambda n, s: n.get(s))
    if d := jload("sampling_steps"):
        sweep("sampling_steps", d, [_key(s) for s in d["eval_num_steps"]], lambda n, s: n.get(s))
    if d := jload("model_transfer"):
        labels = [f"{m} (attacked)" if m == d.get("attacked_model_label") else m for m in d["model_labels"]]
        sweep("model_transfer", d, d["model_labels"], lambda n, s: n.get(s), labels)
    if d := jload("robustness"):
        thr = d["success_rate_thresholds_km"]
        sweep("jpeg", {**d, "success_rate_thresholds_km": thr}, [_key(q) for q in d["jpeg_quality_factors"]],
              lambda n, s: (n.get("jpeg", {}).get(s) or {}).get("true"), metric=METRIC_TRUE)
        sweep("blur", {**d, "success_rate_thresholds_km": thr}, [_key(s) for s in d["gaussian_blur_sigmas"]],
              lambda n, s: (n.get("blur", {}).get(s) or {}).get("true"), metric=METRIC_TRUE)
    thresholds = [200.0, 750.0, 2500.0]
    if (c := jload("cfg")) is not None:
        thresholds = [float(t) for t in c["success_rate_thresholds_km"]]
    if glob.glob(os.path.join(results_dir, f"{dataset}_*_results.pt")):
        curves, budgets, depth = _restart_curves_from_pt(results_dir, dataset)
        out["restarts"] = _restart_ablation(curves, budgets, depth, thresholds, METRIC_TRUE)
        if drop_restarts:
            kept = {key: [[d for i, d in enumerate(r) if i not in drop_restarts] for r in rows]
                    for key, rows in curves.items()}
            out["restarts_distinct"] = _restart_ablation(kept, budgets, depth - len(drop_restarts),
                                                         thresholds, METRIC_TRUE)
    elif d := jload("restarts"):
        kmax = int(d["max_restarts"])
        curves = {(a, b): [v["restart_displacements"] for v in d["results"][a][_bkey(b)].values()
                           if v and len(v.get("restart_displacements") or []) >= kmax]
                  for a in d["attack_types"] for b in d["attack_budgets"]}
        out["restarts"] = _restart_ablation(curves, d["attack_budgets"], kmax, thresholds, METRIC_CLEAN)
    return out


def _fmt(v) -> str:
    return "--" if v is None else f"{100 * v:.1f}"


def render(abl: dict, name: str, thr: float, dataset: str) -> tuple[str, str]:
    rate, settings, budgets = abl["rate"], abl["settings"], abl["budgets"]
    attacks = order_attacks({a for (a, *_rest) in rate})
    cols = [(b, s) for b in budgets for s in settings]
    get = lambda a, b, s: rate.get((a, b, s, thr))  # noqa: E731
    best = {c: max((v for a in attacks if (v := get(a, *c)) is not None), default=None) for c in cols}
    nan_cells = set(abl.get("nan_mean", []))

    def cell(a, b, s, bold, dagger):
        v = get(a, b, s)
        txt = _fmt(v) + (dagger if (a, s) in nan_cells and v is not None else "")
        return bold(txt) if v is not None and best[(b, s)] is not None and v >= best[(b, s)] else txt

    # Markdown
    head = ["Attack"] + [f"{budget_label(b)} · {lab}" for b in budgets for lab in abl["labels"]]
    md = [f"_ASR (%) at {thr:.0f} km; displacement measured to the **{abl['metric']}**._\n",
          "| " + " | ".join(head) + " |", "|" + "|".join(["---"] + ["---:"] * len(cols)) + "|"]
    for a in attacks:
        md.append("| " + " | ".join([_display_attack_name(a)] +
                                    [cell(a, b, s, lambda x: f"**{x}**", "†") for b, s in cols]) + " |")
    if nan_cells:
        md.append("\n† lower bound: some per-image samples are NaN and count as failures "
                  "(mean displacement is NaN for these cells).")

    # LaTeX
    col_title, desc = TITLES[name]
    n = len(settings)
    tex = ["% Auto-generated by scripts/make_ablation_tables.py -- do not edit by hand.",
           "\\begin{table*}[t]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}",
           "\\begin{tabular}{l" + ("|" + "r" * n) * len(budgets) + "}", "\\toprule",
           " & ".join([""] + [f"\\multicolumn{{{n}}}{{c}}{{$b={budget_label(b)}$}}" for b in budgets])
           + " \\\\",
           " & ".join([col_title] + [lab.replace("_", "\\_") for _ in budgets for lab in abl["labels"]])
           + " \\\\\n\\midrule"]
    prev = None
    for a in attacks:
        group = "geofa" if a in GEOFA_FAMILY else "baseline"
        if prev == "baseline" and group == "geofa":
            tex.append("\\midrule")
        prev = group
        label = _display_attack_name(a).replace("\u00b2", "$^{2}$")
        tex.append(" & ".join([label] + [cell(a, b, s, lambda x: f"\\textbf{{{x}}}", "$^\\dagger$")
                                         for b, s in cols]) + " \\\\")
    note = (" $^\\dagger$Lower bound: a few per-image samples are NaN and count as failures."
            if nan_cells else "")
    tex += ["\\bottomrule\n\\end{tabular}",
            f"\\caption{{Attack success rate (\\%) at {thr:.0f}\\,km vs.\\ {desc}, per budget "
            f"$b$ ({dataset}); displacement measured to the {abl['metric']}. "
            f"Best value per column in bold.{note}}}",
            f"\\label{{tab:{dataset}_{name}_sr{thr:.0f}}}\n\\end{{table*}}"]
    return "\n".join(md), "\n".join(tex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--dataset", default="yfcc")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--drop-restarts", type=int, nargs="*", default=[],
                    help="restart indices to leave out of an extra 'restarts_distinct' table")
    args = ap.parse_args()

    abls = load_ablations(args.results_dir, args.dataset, tuple(args.drop_restarts))
    os.makedirs(args.out_dir, exist_ok=True)
    written = []
    for name, abl in abls.items():
        for thr in abl["thresholds"]:
            md, tex = render(abl, name, thr, args.dataset)
            stem = f"{args.dataset}_{name}_sr{thr:.0f}"
            for ext, body in (("md", md), ("tex", tex)):
                path = os.path.join(args.out_dir, f"{stem}.{ext}")
                open(path, "w").write(body + "\n")
                written.append(path)
            print(f"\n### {stem}\n{md}")
        if name.startswith("restarts"):
            print("\nrestart curves used per (attack, budget):",
                  {f"{a}@{budget_label(b)}": n for (a, b), n in abl["n_images"].items()})
    print("\nWritten:\n  " + "\n  ".join(written))


if __name__ == "__main__":
    main()
