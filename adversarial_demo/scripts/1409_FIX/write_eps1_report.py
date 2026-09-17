"""1409_FIX: comprehensive 1/255 section (results/1409_FIX/eps1.md, inlined by write_report.py).

Sources: 1409_FIX_eps1 shard states (main metrics: sampling_v2, dtd_v3_old_forward, geoshield),
/Data/mathias.ollu/results/1409_FIX_offline_eps1/{transfer,steps,transforms}_<attack>.pt (offline
re-evaluations of the saved 1/255 deltas), the 4/255 clean-transformed control (attack- and
budget-independent) and results/1409_FIX/delta_budget.md (delta_budget_compare.py).
All GT-based: FSD = km between prediction and true location, mean over 128 samples per image.
"""
import glob, os, warnings
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "results/1409_FIX"); PLOT_DIR = os.path.join(OUT_DIR, "plots")
SHARDS = "/Data/mathias.ollu/results/1409_FIX_eps1/shards"
OFFLINE = "/Data/mathias.ollu/results/1409_FIX_offline_eps1"
CONTROL = "/Data/mathias.ollu/results/1409_FIX_offline/transforms_clean.pt"
ATTACKS = [("sampling_v2", "GeoFA-sample v2"), ("dtd_v3_old_forward", "GeoFA-TD v3 (old fwd.)")]
FSD_ATTACKS = ATTACKS + [("encoder_v3", "GeoFA-enc. v3"), ("training_loss_v3", "GeoFA-AdvDM v3"), ("diffusion_l2_v3", "GeoFA-L2 v3"), ("l2_target", "GeoFA-L2 targeted"), ("geoshield", "GeoShield")]
COLORS = {"sampling_v2": "tab:brown", "dtd_v3_old_forward": "tab:purple", "encoder_v3": "gold", "training_loss_v3": "tab:olive", "diffusion_l2_v3": "teal", "l2_target": "darkslategray", "geoshield": "tab:red"}
THR = (200, 750, 2500)
BINS = torch.tensor([1.0, 100.0, 1000.0, 5000.0, 2.2e4])
HDR = "| setting | attacked FSD: median \\| mean km \\| ASR@200 / 750 / 2500 | clean FSD: median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|"


def haversine_km(a, b):
    a = torch.as_tensor(a).float().reshape(-1, 2); b = torch.as_tensor(b).float().reshape(-1, 2)
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def stats(v):
    v = v[torch.isfinite(v)]
    return "—" if len(v) == 0 else f"{v.median():.0f} \\| {v.mean():.0f} \\| " + " / ".join(f"{(v > t).float().mean() * 100:.0f}" for t in THR)


def asr(v, t=2500):
    v = v[torch.isfinite(v)]; return float((v > t).float().mean() * 100) if len(v) else float("nan")


def pearson(x, y):
    x, y = x.float(), y.float()
    return float("nan") if len(x) < 2 or x.std() == 0 or y.std() == 0 else float(((x - x.mean()) * (y - y.mean())).mean() / (x.std(unbiased=False) * y.std(unbiased=False)))


def reduce(store, key, side):
    out = []
    for e in store.values():
        v = e.get(key, {}).get(side)
        out.append(float("nan") if v is None else haversine_km(torch.tensor(e["true_gps"]).expand_as(v), v).nanmean().item())
    return torch.tensor(out)


def load_store(mode, att):
    p = os.path.join(OFFLINE, f"{mode}_{att}.pt")
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else {}


def main_pairs(att):
    """(clean FSD, attacked FSD) per image from the 1/255 shard state."""
    suffix, key = ("_geoshield", "geoshield") if att == "geoshield" else ("", att)
    ps = sorted(glob.glob(f"{SHARDS}/{att}__w*/yfcc_seed42_eval_state{suffix}.pt"))
    if not ps:
        return None
    s = torch.load(ps[0], map_location="cpu", weights_only=False)
    loc = [l for l in s["location_results"][key][0] if l is not None and l.get("predicted_gps_perturbed") is not None]
    c = torch.tensor([haversine_km(l["true_gps"].expand_as(l["predicted_gps_source"].reshape(-1, 2)), l["predicted_gps_source"]).mean() for l in loc])
    a = torch.tensor([haversine_km(l["true_gps"].expand_as(l["predicted_gps_perturbed"].reshape(-1, 2)), l["predicted_gps_perturbed"]).mean() for l in loc])
    return c, a


def fsd_panel(ax, clean, true, name, color):
    x, y = clean.clamp(min=1), true.clamp(min=1)
    ax.scatter(x.numpy(), y.numpy(), s=9, alpha=0.35, color=color, lw=0)
    xs, med, q1, q3 = [], [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() < 3:
            continue
        xs.append(float((lo * hi).sqrt())); v = y[m]
        med.append(float(v.median())); q1.append(float(v.quantile(0.25))); q3.append(float(v.quantile(0.75)))
        ax.text(xs[-1], 1.25, f"n={int(m.sum())}", ha="center", fontsize=6, color="gray")
    if xs:
        ax.fill_between(xs, q1, q3, color=color, alpha=0.25, lw=0, label="IQR (25–75 %)"); ax.plot(xs, med, "o-", color=color, lw=1.6, ms=4, label="binned median")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 2.2e4); ax.set_ylim(1, 2.2e4)
    ax.plot([1, 2.2e4], [1, 2.2e4], "k--", lw=0.7, alpha=0.5)
    for t in THR: ax.axhline(t, color="gray", lw=0.5, ls=":", alpha=0.7)
    for b in BINS[1:-1]: ax.axvline(float(b), color="lightgray", lw=0.4)
    ax.set_title(f"{name} (r={pearson(clean, true):.2f})", fontsize=8.5)


md = ["# Budget 1/255 — comprehensive evaluation (GT-based, 100 YFCC images, single run, seed 42)\n",
      "Same protocol as the 4/255 study: attacks retrained at ε = 1/255 (`configs/1409_FIX/config_eps1.yaml`), evaluated with sigmoid/Euler, "
      "250 steps, cfg 10, 128 shared noises; every other number below is an offline re-evaluation of the saved 1/255 perturbations "
      "(`offline_eval.py --deltas-root 1409_FIX_eps1/shards`). GeoShield 1/255 appears in the FSD figure only (it is indistinguishable "
      "from no attack at this budget, see the budget table above).\n"]

# ---- attacked vs clean FSD
md.append("## Attacked FSD vs clean FSD (1/255)\n")
pairs = [(att, name, main_pairs(att)) for att, name in FSD_ATTACKS]
pairs = [(a, n, p) for a, n, p in pairs if p is not None]
if pairs:
    ncol = min(4, len(pairs)); nrow = (len(pairs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.3 * nrow), squeeze=False, sharey=True, sharex=True)
    for ax, (att, name, (c, a)) in zip(axes.flat, pairs):
        fsd_panel(ax, c, a, name, COLORS[att])
    for ax in axes.flat[len(pairs):]:
        ax.set_axis_off()
    for ax in axes[-1]: ax.set_xlabel("clean FSD (km)", fontsize=8)
    for ax in axes[:, 0]: ax.set_ylabel("attacked FSD (km)", fontsize=8)
    axes[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Per-image FSD, attacked vs clean, budget 1/255 — line: median, band: IQR per clean-FSD bin; dotted: ASR thresholds; dashed: y = x", fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "attacked_vs_clean_fsd_1.png"), dpi=150); plt.close(fig)
    md.append("![attacked vs clean FSD, 1/255](plots/attacked_vs_clean_fsd_1.png)\n")

# ---- transfer
md.append("## Cross-model transfer (1/255; δ trained on rfm, cfg 10, 250 steps)\n")
for att, name in ATTACKS:
    st = load_store("transfer", att)
    if not st:
        md.append(f"### {name}\n\nnot run yet\n"); continue
    keys = [k for k in next(iter(st.values())) if isinstance(next(iter(st.values()))[k], dict)]
    md.append(f"### {name} — n = {len(st)}" + ("" if len(st) == 100 else " (partial)") + "\n\n" + HDR.replace("| setting |", "| model |"))
    for k in keys:
        md.append(f"| {k} | {stats(reduce(st, k, 'gps_perturbed'))} | {stats(reduce(st, k, 'gps_source'))} |")
    md.append("")

# ---- steps
md.append("## Sampling steps 2–250 (1/255; sigmoid / Euler, cfg 10)\n")
fig, ax = plt.subplots(figsize=(6, 3.4)); plotted = False
for att, name in ATTACKS:
    st = load_store("steps", att)
    if not st:
        md.append(f"### {name}\n\nnot run yet\n"); continue
    keys = sorted([k for k in next(iter(st.values())) if isinstance(next(iter(st.values()))[k], dict)], key=lambda k: int(k.split()[0]))
    md.append(f"### {name} — n = {len(st)}" + ("" if len(st) == 100 else " (partial)") + "\n\n" + HDR.replace("| setting |", "| steps |"))
    xs, ya, yc = [], [], []
    for k in keys:
        a, c = reduce(st, k, "gps_perturbed"), reduce(st, k, "gps_source")
        md.append(f"| {k} | {stats(a)} | {stats(c)} |"); xs.append(int(k.split()[0])); ya.append(asr(a)); yc.append(asr(c))
    md.append("")
    ax.plot(xs, ya, "o-", color=COLORS[att], label=f"{name}: attacked FSD > 2500 km"); ax.plot(xs, yc, "x--", color=COLORS[att], alpha=0.6, label=f"{name}: clean FSD > 2500 km"); plotted = True
if plotted:
    ax.set_xscale("log"); ax.set_xlabel("sampling steps"); ax.set_ylabel("%"); ax.set_ylim(0, 102); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "steps_asr_eps1.png"), dpi=140); md.append("![steps, 1/255](plots/steps_asr_eps1.png)\n")
plt.close(fig)

# ---- transforms
md.append("## Robustness to JPEG / blur (1/255; protected image transformed, clean untouched)\n")
ctrl = torch.load(CONTROL, map_location="cpu", weights_only=False) if os.path.exists(CONTROL) else {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4)); plotted = False
for att, name in ATTACKS:
    st = load_store("transforms", att)
    if not st:
        md.append(f"### {name}\n\nnot run yet\n"); continue
    keys = [k for k in next(iter(st.values())) if isinstance(next(iter(st.values()))[k], dict)]
    md.append(f"### {name} — n = {len(st)}" + ("" if len(st) == 100 else " (partial)") + f"\n\n| transform | attacked FSD: median \\| mean km \\| ASR@200 / 750 / 2500 | control: clean image transformed (n = {len(ctrl)}): median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|")
    jq, jb = [], []
    for k in keys:
        a = reduce(st, k, "gps_perturbed"); cc = reduce(ctrl, k, "gps_source") if ctrl else torch.tensor([])
        md.append(f"| {k} | {stats(a)} | {stats(cc) if ctrl else 'pending'} |")
        if k.startswith("jpeg"): jq.append((int(k.split("q")[1]), asr(a)))
        if k.startswith("blur"): jb.append((float(k.split("s")[1]), asr(a)))
    md.append("")
    axes[0].plot([q for q, _ in jq], [v for _, v in jq], "o-", color=COLORS[att], label=name); axes[1].plot([s_ for s_, _ in jb], [v for _, v in jb], "o-", color=COLORS[att], label=name); plotted = True
if plotted:
    if ctrl:
        ck = [k for k in next(iter(ctrl.values())) if isinstance(next(iter(ctrl.values()))[k], dict)]
        axes[0].plot([int(k.split("q")[1]) for k in ck if k.startswith("jpeg")], [asr(reduce(ctrl, k, "gps_source")) for k in ck if k.startswith("jpeg")], "kx--", lw=0.8, label="control: clean image transformed")
        axes[1].plot([float(k.split("s")[1]) for k in ck if k.startswith("blur")], [asr(reduce(ctrl, k, "gps_source")) for k in ck if k.startswith("blur")], "kx--", lw=0.8, label="control: clean image transformed")
    axes[0].set_xlabel("JPEG quality"); axes[1].set_xlabel("Gaussian blur σ (px)")
    for pax in axes: pax.set_ylabel("ASR@2500 km (%)"); pax.set_ylim(0, 102); pax.grid(alpha=0.3)
    axes[0].legend(fontsize=7); fig.suptitle("1/255", fontsize=9); fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "transforms_asr_eps1.png"), dpi=140)
    md.append("![robustness, 1/255](plots/transforms_asr_eps1.png)\n")
plt.close(fig)

# ---- delta comparison
db = os.path.join(OUT_DIR, "delta_budget.md")
if os.path.exists(db):
    md.append(open(db).read())
open(os.path.join(OUT_DIR, "eps1.md"), "w").write("\n".join(md))
print("\n".join(md))
