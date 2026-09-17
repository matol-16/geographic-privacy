"""1409_FIX report: pixel-space fix, single run (no restarts), GeoFA-TD / GeoFA-enc. / GeoFA-AdvDM /
GeoFA-TD v2 (geodesic RFM forward process) / GeoShield (lossless PNG, 336 px).

Reads the shard state files under /Data/mathias.ollu/results/1409_FIX/shards (or the copies in
results/1409_FIX/data) and writes results/1409_FIX/report.md. Only finished (attack, budget)
cells are tabulated; partial runs are reported on the finished image subset.
"""
import datetime, glob, math, os, warnings
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHARDS = [os.path.join(ROOT, "results/1409_FIX/data"), "/Data/mathias.ollu/results/1409_FIX/shards"]
OUT = os.path.join(ROOT, "results/1409_FIX/report.md")
PLOT_DIR = os.path.join(ROOT, "results/1409_FIX/plots")
COLORS = {"dtd": "tab:blue", "dtd_v2": "tab:cyan", "dtd_v3": "navy", "dtd_v3_old_forward": "tab:purple", "encoder": "tab:orange", "training_loss": "tab:green", "training_loss_v3": "tab:olive", "encoder_v3": "gold", "sampling_v2": "tab:brown", "diffusion_l2_v2": "lightseagreen", "diffusion_l2_v3": "teal", "l2_target": "darkslategray", "geoshield": "tab:red", "geoshield_v1_cropshift": "tab:gray"}
# geoshield_v1_cropshift = the first fixed GeoShield run (PNG, 336 px) whose clean copies were the raw
# files: GeoShield's Resize->CenterCrop vs PLONK's CenterCrop->Resize left a sub-pixel crop shift
# between clean and attacked (flips ~8 % of predictions with delta = 0). Kept for comparison.
ATTACKS = [("dtd", "GeoFA-TD"), ("dtd_v2", "GeoFA-TD v2"), ("dtd_v3", "GeoFA-TD v3"), ("dtd_v3_old_forward", "GeoFA-TD v3 (old fwd.)"), ("encoder", "GeoFA-enc."), ("training_loss", "GeoFA-AdvDM"), ("training_loss_v3", "GeoFA-AdvDM v3"), ("encoder_v3", "GeoFA-enc. v3"), ("sampling_v2", "GeoFA-sample v2"), ("diffusion_l2_v2", "GeoFA-L2 v2"), ("diffusion_l2_v3", "GeoFA-L2 v3"), ("l2_target", "GeoFA-L2 targeted"), ("geoshield", "GeoShield"), ("geoshield_v1_cropshift", "GeoShield (prev., sub-px crop shift)")]
RES_KEY = lambda key: "geoshield" if key.startswith("geoshield") else key  # key used inside the state file
THR = (200, 750, 2500)
L = lambda p: torch.load(p, map_location="cpu", weights_only=False)


def find_state(att):
    # GeoShield (precomputed-pair runner) names its state file with the attack suffix.
    suffix = "_geoshield" if att.startswith("geoshield") else ""
    for d in SHARDS:
        for p in sorted(glob.glob(f"{d}/{att}__w*/yfcc_seed42_eval_state{suffix}.pt")):
            return p
    return None


def norm_ids(state):
    # GeoShield ids are file names ("<id>.png"); trainable attacks store the bare id.
    return [str(x).split(".")[0] for x in state["source_image_ids"]]


def haversine_km(a, b):
    a = torch.as_tensor(a).float().reshape(-1, 2); b = torch.as_tensor(b).float().reshape(-1, 2)
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def stats(v):
    v = v.float()
    asr = " / ".join(f"{(v > t).float().mean() * 100:.0f}" for t in THR)
    return f"{v.median():.0f} \\| {v.mean():.0f} \\| {asr}"


def pearson(x, y):
    x, y = x.float(), y.float()
    if len(x) < 2 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


states, budgets, ids, args_seen = {}, None, None, {}
for key, _ in ATTACKS:
    p = find_state(key)
    if p is None:
        continue
    s = L(p); states[key] = s
    sid = norm_ids(s)
    if key.startswith("geoshield"):
        # the GeoShield shard walks the same seeded window but in file-name order: build a
        # permutation onto the trainable attacks' order (set by the first trainable state).
        continue
    ids = sid if ids is None else ids
    assert sid == ids, f"image ids differ for {key}"
    ap = os.path.join(os.path.dirname(p), "yfcc_attack_args.pt")
    if os.path.exists(ap):
        a = L(ap); args_seen[key] = a["attack_kwargs"][0]
# Budgets to tabulate come from the experiment config (4/255 only since 2026-09-14 22:30); each
# state carries its own budget list in signature["attack_budgets"], so a budget is looked up per
# state (older states also hold 8/255, which is simply not reported).
budgets = list(yaml.safe_load(open(os.path.join(ROOT, "configs/1409_FIX/config.yaml")))["attack_budgets"]["yfcc"])
def budget_index(state, eps):
    for i, b in enumerate(state["signature"]["attack_budgets"]):
        if abs(float(b) - float(eps)) < 1e-6:
            return i
    return None
for gkey in [k for k in states if k.startswith("geoshield")]:
    gs = states[gkey]; gid = norm_ids(gs)
    if ids is None:
        ids = gid
    assert sorted(gid) == sorted(ids), f"{gkey} image set differs from the trainable runs"
    pos = {g: i for i, g in enumerate(gid)}
    perm = torch.tensor([pos[i] for i in ids])  # geoshield row -> trainable ordering
    for m in list(gs["results"]["geoshield"].keys()):
        gs["results"]["geoshield"][m] = gs["results"]["geoshield"][m][:, perm]
    gs["location_results"]["geoshield"] = [[bl[j] for j in perm.tolist()] for bl in gs["location_results"]["geoshield"]]
n_img = len(ids) if ids is not None else 100


def cell(key, eps):
    """Return dict of per-image tensors for finished images of (attack, budget); None if the
    state was not run at this budget."""
    s = states[key]; rk = RES_KEY(key)
    bi = budget_index(s, eps)
    if bi is None:
        return None
    pred = s["results"][rk]["final_step_displacement_predicted"][bi].float()
    true = s["results"][rk]["final_step_displacement_true"][bi].float()
    idx = torch.nonzero(torch.isfinite(pred) & (pred > 0)).flatten()
    loc = s["location_results"][rk][bi]
    idx = torch.tensor([i for i in idx.tolist() if loc[i] is not None and loc[i].get("predicted_gps_source") is not None], dtype=torch.long)
    # predicted_gps_source is the clean prediction batch (num_samples, 2); states written before
    # the core.py single-sample fix hold one (1, 2) sample -- the mean covers both.
    clean = torch.tensor([haversine_km(loc[i]["true_gps"], loc[i]["predicted_gps_source"]).mean().item() for i in idx.tolist()])
    return dict(idx=idx, paired=pred[idx], true=true[idx], clean=clean)


hdr_true = "| Attack | n | attacked FSD: median km \\| mean km \\| ASR@200 / 750 / 2500 km | clean FSD: median km \\| mean km \\| >200 / >750 / >2500 km |\n|---|---|---|---|"
os.makedirs(PLOT_DIR, exist_ok=True)
BINS = torch.tensor([1.0, 100.0, 1000.0, 5000.0, 2.2e4])  # clean-FSD bins (km): <100 (incl. <1 km clipped), 100-1k, 1k-5k, >5k; 10-100 km holds only 3 images so it is merged


def fsd_panel(ax, clean, true, name, color):
    """One attack: per-image attacked FSD vs clean FSD (scatter) + binned median with IQR band."""
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
        ax.fill_between(xs, q1, q3, color=color, alpha=0.25, lw=0, label="IQR (25–75 %)")
        ax.plot(xs, med, "o-", color=color, lw=1.6, ms=4, label="binned median")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1, 2.2e4); ax.set_ylim(1, 2.2e4)
    ax.plot([1, 2.2e4], [1, 2.2e4], "k--", lw=0.7, alpha=0.5)
    for t in THR:
        ax.axhline(t, color="gray", lw=0.5, ls=":", alpha=0.7)
    for b in BINS[1:-1]:
        ax.axvline(float(b), color="lightgray", lw=0.4)
    ax.set_title(f"{name} (r={pearson(clean, true):.2f})", fontsize=8.5); ax.grid(False)


secs, status = [], []
for bi, eps in enumerate(budgets):
    e255 = round(eps * 255)
    rows_t = []
    panels = []
    for key, name in ATTACKS:
        if key not in states:
            rows_t.append(f"| {name} | — | not run yet | |"); status.append(f"{name}: not started"); continue
        c = cell(key, eps)
        if c is None:
            rows_t.append(f"| {name} | — | not run at this budget | |"); status.append(f"{name} {e255}/255: not run"); continue
        n = len(c["idx"])
        tag = "" if n == n_img else f" (first {n} imgs, partial)"
        if n == 0:
            rows_t.append(f"| {name} | 0 | in progress | |")
        else:
            rows_t.append(f"| {name}{tag} | {n} | {stats(c['true'])} | {stats(c['clean'])} |")
            panels.append((key, name, c))
        status.append(f"{name} {e255}/255: {n}/{n_img}")
    secs.append(f"### Budget {e255}/255 (eps_max = {eps})\n\n**Attacked FSD vs clean FSD** (both against the ground-truth location; the clean column is the model's own error on the same images)\n\n{hdr_true}\n" + "\n".join(rows_t))
    plot_rel = f"plots/attacked_vs_clean_fsd_{e255}.png"
    if panels:
        # One panel per attack: attacked FSD (attacked prediction vs GT) against clean FSD (clean
        # prediction vs GT), per image, with the median and interquartile range of the attacked FSD
        # in clean-FSD bins (<100, 100-1k, 1k-5k, >5k km). Above the diagonal =
        # the attack moved the prediction further from the truth than the model's own error.
        ncol = 4; nrow = (len(panels) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow), squeeze=False, sharex=True, sharey=True)
        for ax, (key, name, c) in zip(axes.flat, panels):
            fsd_panel(ax, c["clean"], c["true"], name, COLORS.get(key))
        for ax in axes.flat[len(panels):]:
            ax.set_axis_off()
        for ax in axes[-1]:
            ax.set_xlabel("clean FSD (km)", fontsize=8)
        for ax in axes[:, 0]:
            ax.set_ylabel("attacked FSD (km)", fontsize=8)
        axes[0, 0].legend(fontsize=6.5, loc="lower right")
        fig.suptitle(f"Per-image FSD, attacked vs clean, budget {e255}/255 — line: median, band: IQR per clean-FSD bin; dotted: ASR thresholds; dashed: y = x", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, plot_rel.split("/")[-1]), dpi=150)
        plt.close(fig)
        secs[-1] += f"\n\n![attacked vs clean FSD, budget {e255}/255]({plot_rel})"

inline = lambda f: open(os.path.join(ROOT, "results/1409_FIX", f)).read() if os.path.exists(os.path.join(ROOT, "results/1409_FIX", f)) else ""
kw = next(iter(args_seen.values()), {})
num_restarts = kw.get("num_restarts", "1")
kw_str = ", ".join(f"{k} {v}" for k, v in kw.items() if k not in ("device",))
cfg_path = os.path.join(ROOT, "configs/1409_FIX/config.yaml")
md = f"""# 1409_FIX — pixel-space fix, single run (no restarts) — {datetime.date.today().isoformat()}

## Main hyperparameters

- Attacks: `dtd` = GeoFA-TD, `dtd_v2` = GeoFA-TD v2 (geodesic RFM forward process), `dtd_v3` = GeoFA-TD v3 (v2 + x_t noised from the **true** location x₀ instead of 512 clean-model samples; reference still the clean predicted velocity), `dtd_v3_old_forward` = GeoFA-TD v1 (Gaussian x_t) + true x₀ (isolates the x₀ effect: v3_old_forward − v1 = x₀, v3 − v3_old_forward = geodesic path), `encoder` = GeoFA-enc., `training_loss` = GeoFA-AdvDM, `training_loss_v3` = GeoFA-AdvDM v3 (x_t and training target v(x_t) built from the true x₀, Eq. 9), `encoder_v3` = GeoFA-enc. optimised with the shared sign-SGD PGD loop (squared L2 of Eq. 7, tiny init, encoder in eval mode; v1 used Adam, unsquared norm, full-budget init, encoder in train mode), `sampling_v2` = GeoFA-Sampling maximising the haversine distance to the **true** location (Eq. 8; v1 deviated from the clean prediction), `diffusion_l2_v2` / `_v3` = GeoFA-L² with the geodesic x_t and tangent L² (Eq. 12), clean-model / true x₀ (in-process, pixel-space fix), `l2_target` = GeoFA-L² targeted (DP-Attacker-style targeted denoising loss, minimised towards the antipode of the true location; geodesic x_t), `geoshield` = GeoShield (external, 100 FGSM steps, α = 1/255, ensemble CLIP B/16 + B/32 + LAION; fair-comparison changes: lossless PNG in/out, `model.input_res=336`, clean copies = PLONK's CenterCrop(1:1)→Resize(336) square so GeoShield's own Resize→CenterCrop is a no-op and clean/attacked differ by δ only). `geoshield_v1_cropshift` = previous GeoShield run (PNG, 336 px, raw clean copies): a sub-pixel crop shift between clean and attacked that alone (δ = 0) moves 4/25 control images > 200 km and 2/25 > 2500 km; shown for comparison only.
- **`num_restarts` = {num_restarts}, `--max-restarts 1`: single run per (image, budget), no restart selection**, for every attack including GeoShield.
- Budgets (L∞): {", ".join(f"{round(b*255)}/255 ({b})" for b in budgets)}.
- GeoFA attack args: {kw_str}.
- Model: PLONK_YFCC (DINOv2 ViT-L/14-reg). Dataset: YFCC4k, first 100 images (seed 42), identical ids across attacks. Evaluation: cfg 10, batch 128, 250 DDIM steps.
- **Metric: FSD (final-step displacement, km)** = haversine distance between the model's final prediction and the **ground-truth** location, averaged over the 128 samples of an image. *Attacked FSD* is measured on the protected image (`final_step_displacement_true`), *clean FSD* on the clean image (`predicted_gps_source` vs GT); every table puts the two side by side, so the clean column is the model's own error on exactly the same images and sampler. ASR@τ = fraction of images with attacked FSD > τ km; the clean column's >τ figures are the same fractions for the clean FSD. The paired clean-vs-attacked displacement is stored (`final_step_displacement_predicted`) but not reported. NB: GeoFA states written before the `core.py` fix of 2026-09-14 hold **one** of the 128 samples for the attacked FSD (`_normalize_restart_result` kept row `[-1]` of the sample batch); the relaunched GeoShield and all offline / ablation re-evaluations use the mean over 128. At cfg 10 the batch is usually collapsed so the two coincide except on multimodal images.
- Run status: {"; ".join(status)}.

## Results

{(chr(10)*2).join(secs)}

{inline("ablations.md")}

{inline("budget.md")}

{inline("transforms.md")}

{inline("low_steps.md")}

{inline("eps1.md")}

## Varying schedulers

Standalone report: [schedulers.md](schedulers.md) — which schedule / integrator is used at training, in each attack and at evaluation, and how GeoFA-sample v2 and GeoFA-TD v3 (old fwd.) behave when only the inference sampler is changed (linear / cosine schedules, exp. map / Heun / stochastic integrators, 250 and 8 steps).
"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write(md)
print(md)
