"""1409_FIX sampler / cfg / cross-model ablations, ground-truth based.

Reads the per-image delta files written by the ablation runs
(/Data/mathias.ollu/results/1409_FIX_ablations/shards/<attack>__w*/deltas/*.pt, key ``ablation_gps``:
{"steps"|"cfg"|"transfer": {key: {"gps_source", "gps_perturbed"}}} = the 128 clean / attacked GPS
samples of each eval-time re-evaluation) and reports, per axis, the displacement of the attacked
prediction to the TRUE location (mean over the 128 samples, then median / mean over images) and
ASR@200/750/2500 km, alongside each setting's own clean-vs-GT baseline. Writes
results/1409_FIX/ablations.md (+ plots/ablation_*.png); write_report.py inlines it.
"""
import glob, os, warnings
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARDS = "/Data/mathias.ollu/results/1409_FIX_ablations/shards"
OUT = os.path.join(ROOT, "results/1409_FIX/ablations.md")
PLOT_DIR = os.path.join(ROOT, "results/1409_FIX/plots")
ATTACKS = [("sampling_v2", "GeoFA-sample v2"), ("dtd_v3_old_forward", "GeoFA-TD v3 (old fwd.)")]
AXES = [("steps", "Sampling steps (cfg 10)", "steps"), ("cfg", "Guidance scale (250 steps)", "cfg"),
        ("transfer", "Attacked model (cfg 10, 250 steps; δ trained on rfm)", "model")]
THR = (200, 750, 2500)
N_TOTAL = 100


def haversine_km(a, b):
    a = torch.as_tensor(a).float().reshape(-1, 2); b = torch.as_tensor(b).float().reshape(-1, 2)
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def stats(v):
    asr = " / ".join(f"{(v > t).float().mean() * 100:.0f}" for t in THR)
    return f"{v.median():.0f} \\| {v.mean():.0f} \\| {asr}"


def load(att):
    """{axis: {key: (attacked_vs_gt[n], clean_vs_gt[n])}} over the finished images of one attack."""
    files = sorted(glob.glob(f"{SHARDS}/{att}__w*/deltas/*.pt"))
    per = {ax: {} for ax, _, _ in AXES}
    for f in files:
        d = torch.load(f, map_location="cpu", weights_only=False)
        tg = torch.as_tensor(d["true_gps"]).float()
        for ax, _, _ in AXES:
            for k, v in d.get("ablation_gps", {}).get(ax, {}).items():
                per[ax].setdefault(k, ([], []))
                # nanmean: the Euclidean-flow head returns NaN coordinates for a few samples on some
                # images; an image whose 128 samples are all NaN is dropped from that row (counted).
                per[ax][k][0].append(haversine_km(tg.expand_as(v["gps_perturbed"]), v["gps_perturbed"]).nanmean())
                per[ax][k][1].append(haversine_km(tg.expand_as(v["gps_source"]), v["gps_source"]).nanmean())
                NAN.setdefault((att, ax, k), [0, 0])
                NAN[(att, ax, k)][0] += int(v["gps_perturbed"].isnan().any())
                NAN[(att, ax, k)][1] += int(v["gps_source"].isnan().any())
    return len(files), {ax: {k: (torch.tensor(a), torch.tensor(c)) for k, (a, c) in kv.items()} for ax, kv in per.items()}


NAN = {}  # (attack, axis, key) -> [#images with >=1 NaN attacked sample, #images with >=1 NaN clean sample]


def valid(a, c):
    m = torch.isfinite(a) & torch.isfinite(c)
    return a[m], c[m], int((~m).sum())


os.makedirs(PLOT_DIR, exist_ok=True)
md = ["## Sampler / guidance / cross-model ablations (GT-based, 4/255)\n",
      "Each trained perturbation (δ from a fresh single run, seed 42, same 100 images) is re-evaluated without retraining while ONE evaluation "
      "knob varies: number of sampling steps (cfg 10), guidance scale (250 steps), or the PLONK backbone the attacked image is fed to "
      "(cfg 10, 250 steps; the three YFCC variants share the DINOv2 encoder and differ in the generative head: Riemannian flow matching = "
      "attacked model, Euclidean diffusion, Euclidean flow matching). Values: **attacked FSD** (protected image's prediction vs the TRUE location, mean over 128 samples "
      "per image; median | mean km over images | ASR@200 / 750 / 2500 km) next to the **clean FSD** of the same setting (median | mean | >200 / >750 / >2500), "
      "so a low ASR can be told apart from a model that is simply inaccurate at that setting.\n"]
for att, name in ATTACKS:
    n, per = load(att)
    if n == 0:
        md.append(f"### {name}\n\nnot run yet\n"); continue
    tag = "" if n == N_TOTAL else f" (first {n} imgs, partial)"
    md.append(f"### {name} — n = {n}{tag}\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for (ax, title, col), pax in zip(AXES, axes):
        kv = per[ax]
        if not kv:
            md.append(f"**{title}**: not recorded\n"); continue
        keys = sorted(kv, key=lambda k: (isinstance(k, str), k))
        md.append(f"**{title}**\n\n| {col} | attacked FSD: median \\| mean km \\| ASR@200 / 750 / 2500 | clean FSD: median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|")
        notes = []
        for k in keys:
            a, c, dropped = valid(*kv[k])
            kl = k if not isinstance(k, float) else f"{k:g}"
            md.append(f"| {kl}{'*' if dropped else ''} | {stats(a)} | {stats(c)} |")
            if dropped:
                na, nc = NAN[(att, ax, k)]
                notes.append(f"\\* {kl}: n = {len(a)} — {dropped} image(s) dropped because all 128 **attacked** samples are NaN (the Euclidean-flow head returns NaN coordinates when the attacked image drives it off the valid range: >=1 NaN sample on {na} attacked vs {nc} clean images; partial NaNs ignored via nanmean). Counting the dropped images as successes gives ASR@2500 = {((a > 2500).sum() + dropped) / (len(a) + dropped) * 100:.0f} %, as failures {(a > 2500).sum() / (len(a) + dropped) * 100:.0f} %.")
        md.append("")
        md.extend(n + "\n" for n in notes)
        xs = [str(k) if isinstance(k, str) else f"{k:g}" for k in keys]
        for ti, t in enumerate(THR):
            pax.plot(xs, [(valid(*kv[k])[0] > t).float().mean() * 100 for k in keys], "o-", label=f"attacked FSD > {t} km (ASR)")
        pax.plot(xs, [(valid(*kv[k])[1] > 2500).float().mean() * 100 for k in keys], "x--", color="gray", label="clean FSD > 2500 km")
        pax.set_title(title, fontsize=9); pax.set_ylim(0, 102); pax.set_ylabel("%"); pax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{name}, 4/255, n={n}", fontsize=10); fig.tight_layout()
    fn = f"ablation_{att}.png"; fig.savefig(os.path.join(PLOT_DIR, fn), dpi=140); plt.close(fig)
    md.append(f"![{name} ablations](plots/{fn})\n")
open(OUT, "w").write("\n".join(md))
print("\n".join(md))
