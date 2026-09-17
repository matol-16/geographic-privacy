"""1409_FIX: reports for the offline re-evaluations and the 1/255 budget runs (all GT-based).

Writes
  results/1409_FIX/schedulers.md  standalone "varying schedulers" report: which schedule /
                                  integrator is used where (training, attacks, evaluation) and
                                  how the two attacks behave when only the inference sampler is
                                  changed (offline_eval.py --mode schedulers)
  results/1409_FIX/transforms.md  JPEG / blur robustness of the protected image (--mode transforms)
  results/1409_FIX/budget.md      1/255 vs 4/255 (retrained shards in 1409_FIX_eps1)
plus plots/schedulers_*.png, plots/transforms_*.png. write_report.py inlines the last two.
Metric everywhere: FSD = haversine km between the prediction and the TRUE location (attacked FSD: protected image; clean FSD: clean image), mean over
the 128 samples of an image (nanmean), then median | mean over images and ASR@200/750/2500; the
clean prediction under the same setting is reported next to it as the baseline.
"""
import glob, os, warnings
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OFFLINE = "/Data/mathias.ollu/results/1409_FIX_offline"
EPS1_SHARDS = "/Data/mathias.ollu/results/1409_FIX_eps1/shards"
MAIN_SHARDS = "/Data/mathias.ollu/results/1409_FIX/shards"
OUT_DIR = os.path.join(ROOT, "results/1409_FIX")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
ATTACKS = [("sampling_v2", "GeoFA-sample v2"), ("dtd_v3_old_forward", "GeoFA-TD v3 (old fwd.)")]
COLORS = {"sampling_v2": "tab:brown", "dtd_v3_old_forward": "tab:purple"}
THR = (200, 750, 2500)
os.makedirs(PLOT_DIR, exist_ok=True)


def haversine_km(a, b):
    a = torch.as_tensor(a).float().reshape(-1, 2); b = torch.as_tensor(b).float().reshape(-1, 2)
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, (a[:, 0], a[:, 1], b[:, 0], b[:, 1]))
    h = torch.sin((lat2 - lat1) / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * torch.asin(torch.sqrt(h.clamp(0, 1)))


def stats(v):
    v = v[torch.isfinite(v)]
    if len(v) == 0:
        return "—"
    return f"{v.median():.0f} \\| {v.mean():.0f} \\| " + " / ".join(f"{(v > t).float().mean() * 100:.0f}" for t in THR)


def asr(v, t=2500):
    v = v[torch.isfinite(v)]
    return float((v > t).float().mean() * 100) if len(v) else float("nan")


def reduce(store, key, side):
    """Per-image mean distance to GT of one variant (``side`` = gps_perturbed | gps_source)."""
    out = []
    for e in store.values():
        v = e.get(key, {}).get(side)
        if v is None:
            out.append(float("nan")); continue
        tg = torch.tensor(e["true_gps"]).expand_as(v)
        out.append(haversine_km(tg, v).nanmean().item())
    return torch.tensor(out)


def load_store(mode, att):
    p = os.path.join(OFFLINE, f"{mode}_{att}.pt")
    return torch.load(p, map_location="cpu", weights_only=False) if os.path.exists(p) else {}


HDR = "| setting | attacked FSD: median \\| mean km \\| ASR@200 / 750 / 2500 | clean FSD: median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|"

# ----------------------------------------------------------------------------- schedulers --
sched_md = ["# Varying schedulers — 1409_FIX, 4/255, 100 YFCC images\n",
"## Which schedule / sampler is used where\n",
"PLONK_YFCC is a Riemannian flow-matching model on S². A *noise schedule* γ(t) ∈ [0, 1] sets how far along the "
"noise→data path a state is (γ = 1 data, γ = 0 noise; the paper's κ = 1 − γ); the network predicts the dγ-velocity "
"and a *sampler* integrates it from a uniform point on S² (γ = 0) to γ = 1. Exactly one schedule appears in the "
"whole pipeline — the **sigmoid** one the model was trained with — and every 1409_FIX number so far used the same "
"sampler as PLONK ships: projected Euler, 250 steps, cfg 10.\n",
"| stage | schedule γ(t) | sampler / integrator | steps | cfg | source |",
"|---|---|---|---|---|---|",
"| PLONK training (`train_noise_scheduler`) | sigmoid: γ(t) = [σ(3) − σ(10t − 7)] / [σ(3) − σ(−7)] (start −7, end 3, τ 1) | — (x_t = geodesic(noise, x₀) at γ) | — | cond-drop 0.1 | `plonk/configs/exp/YFCC100M_…_sigmoid_flow_riemann.yaml` |",
"| PLONK inference default (`inference_noise_scheduler`, `PlonkPipeline`) | same sigmoid (−7, 3, 1) | `riemannian_flow_sampler`: x ← proj_S²(x + Δγ·v) | 250 | user | `plonk/pipe.py`, `plonk/models/samplers/riemannian_flow_sampler.py` |",
"| GeoFA-TD (all versions) | t ~ U(0, 1), γ = `pipeline.scheduler`(t) = sigmoid — the schedule only picks which γ the velocities are compared at | none (no unroll; one network call per PGD step at (x_t, γ)) | — | 0 | `attacks/trajectory_deviation.py` |",
"| GeoFA-sample v2 | sigmoid (through `pipeline.scheduler`) | `pipeline.sampler` = the shipped projected Euler, unrolled and differentiated end-to-end; fresh x_N ~ N(0, I) each PGD step | 16 | 0 | `attacks/diffusion_attack_salman.py` |",
"| 1409_FIX evaluation (all tables) | sigmoid | shipped projected Euler, shared x_N (seed 42) | 250 | 10 | `utils/adversarial_utils.py::run_paired_pipeline_with_shared_noise` |",
"| **this ablation** | sigmoid / **linear** γ = 1 − t / **cosine** γ = cos²(πt/2) | shipped Euler / **exp. map** x ← exp_x(Δγ·proj_TₓS²(v)) / **Heun** (predictor–corrector, 2 NFE/step) / **stochastic** (Euler + tangent noise 3·√Δγ·(1 − γ), re-projected) | 250 and **8** | 10 | `scripts/1409_FIX/offline_eval.py` |",
"",
"The saved 4/255 perturbations (one per image, from the `1409_FIX_ablations` runs) are re-evaluated with only the "
"inference schedule and/or integrator changed; the model, δ, guidance, batch (128) and shared initial noise are fixed. "
"The sigmoid profile spends most of its 250 steps near the data end (γ > 0.88 for t < 0.5) and crosses the noisy "
"half in a few large steps; linear steps uniformly in γ; cosine concentrates steps at both ends. At 250 steps all "
"integrators have converged (baseline vs exp. map vs Heun agree to < 1 km on the clean model), so the 8-step block "
"is where the discretisation genuinely changes the trajectory. The stochastic sampler is deliberately strong: churn 3 "
"already moves the *clean* prediction (see its baseline column).\n"]

# gamma(t) profiles figure
fig, ax = plt.subplots(figsize=(4.2, 3))
t = torch.linspace(0, 1, 400)
sig = lambda x: torch.sigmoid(torch.as_tensor(x, dtype=torch.float32))
ax.plot(t, (sig(3) - sig(10 * t - 7)) / (sig(3) - sig(-7)), label="sigmoid (−7, 3, 1): PLONK")
ax.plot(t, 1 - t, label="linear")
ax.plot(t, torch.cos(torch.pi * t / 2) ** 2, label="cosine")
ax.set_xlabel("sampler time t (1 = noise → 0 = data)"); ax.set_ylabel("γ(t)  (1 = data)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "schedulers_gamma.png"), dpi=140); plt.close(fig)
sched_md.append("![gamma profiles](plots/schedulers_gamma.png)\n")

sched_md.append("## Results (attacked FSD | clean FSD, both under the same sampler)\n")
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
for (att, name), pax in zip(ATTACKS, axes):
    store = load_store("schedulers", att)
    n = len(store)
    if n == 0:
        sched_md.append(f"### {name}\n\nnot run yet\n"); continue
    keys = [k for k in next(iter(store.values())).keys() if isinstance(next(iter(store.values()))[k], dict)]
    sched_md.append(f"### {name} — n = {n}" + ("" if n == 100 else f" (first {n} imgs, partial)") + "\n\n" + HDR)
    xs, ya, yc = [], [], []
    for k in keys:
        a, c = reduce(store, k, "gps_perturbed"), reduce(store, k, "gps_source")
        sched_md.append(f"| {k} | {stats(a)} | {stats(c)} |")
        xs.append(k.replace(" steps", "").replace("sigmoid + ", "sig+").replace("linear + ", "lin+").replace("cosine + ", "cos+").replace(" (baseline)", "*")); ya.append(asr(a)); yc.append(asr(c))
    sched_md.append("")
    pax.bar(range(len(xs)), ya, color=COLORS[att], alpha=0.85, label="attacked FSD > 2500 km (ASR)")
    pax.plot(range(len(xs)), yc, "kx--", lw=0.8, label="clean FSD > 2500 km")
    pax.set_xticks(range(len(xs))); pax.set_xticklabels(xs, rotation=45, ha="right", fontsize=7)
    pax.set_title(f"{name} (n={n})", fontsize=9); pax.set_ylim(0, 102); pax.grid(axis="y", alpha=0.3)
axes[0].set_ylabel("%"); axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "schedulers_asr.png"), dpi=140); plt.close(fig)
sched_md.append("![ASR under each sampler](plots/schedulers_asr.png)\n")
open(os.path.join(OUT_DIR, "schedulers.md"), "w").write("\n".join(sched_md))

# ----------------------------------------------------------------------------- transforms --
tr_md = ["## Robustness of the protected image to JPEG / blur (GT-based, 4/255)\n",
"The saved 4/255 protected image (clean + δ, 8-bit PNG) is JPEG-compressed or Gaussian-blurred (PIL, radius = σ px at "
"336 px) and re-evaluated with the standard sampler (sigmoid, Euler, 250 steps, cfg 10, shared noise); the clean image "
"is left untouched, as in GeoShield Fig. 6 (the defender post-processes what it receives). `clean` = no transform "
"(control, equals the main table up to the fresh single run). Attacked FSD of the transformed protected image; the "
"second column is the control: the CLEAN image put through the same transform (no δ), so a rise of the attacked FSD "
"under strong blur / compression can be told apart from the transform itself destroying the model's accuracy "
"(offline_eval.py --mode transforms_clean; attack-independent).\n"]
ctrl_path = os.path.join(OFFLINE, "transforms_clean.pt")
ctrl = torch.load(ctrl_path, map_location="cpu", weights_only=False) if os.path.exists(ctrl_path) else {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
for att, name in ATTACKS:
    store = load_store("transforms", att)
    n = len(store)
    if n == 0:
        tr_md.append(f"### {name}\n\nnot run yet\n"); continue
    c = reduce(store, "clean", "gps_source")
    keys = [k for k in next(iter(store.values())).keys() if isinstance(next(iter(store.values()))[k], dict)]
    tr_md.append(f"### {name} — n = {n}" + ("" if n == 100 else f" (first {n} imgs, partial)") + "\n\n| transform | attacked FSD (protected image transformed): median \\| mean km \\| ASR@200 / 750 / 2500 | control: clean FSD (clean image transformed, n = " + str(len(ctrl)) + "): median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|")
    jq, jb = [], []
    for k in keys:
        a = reduce(store, k, "gps_perturbed")
        cc = reduce(ctrl, k, "gps_source") if ctrl else torch.tensor([])
        tr_md.append(f"| {k} | {stats(a)} | {stats(cc) if ctrl else 'pending'} |")
        if k.startswith("jpeg"): jq.append((int(k.split("q")[1]), asr(a)))
        if k.startswith("blur"): jb.append((float(k.split("s")[1]), asr(a)))
    tr_md.append(f"\nclean FSD (untransformed clean image): {stats(c)}\n")
    if jq: axes[0].plot([q for q, _ in jq], [v for _, v in jq], "o-", color=COLORS[att], label=name)
    if jb: axes[1].plot([s for s, _ in jb], [v for _, v in jb], "o-", color=COLORS[att], label=name)
    axes[0].axhline(asr(c), color=COLORS[att], ls=":", lw=0.8); axes[1].axhline(asr(c), color=COLORS[att], ls=":", lw=0.8)
if ctrl:
    ck = [k for k in next(iter(ctrl.values())).keys() if isinstance(next(iter(ctrl.values()))[k], dict)]
    axes[0].plot([int(k.split("q")[1]) for k in ck if k.startswith("jpeg")], [asr(reduce(ctrl, k, "gps_source")) for k in ck if k.startswith("jpeg")], "kx--", lw=0.8, label="control: clean image transformed")
    axes[1].plot([float(k.split("s")[1]) for k in ck if k.startswith("blur")], [asr(reduce(ctrl, k, "gps_source")) for k in ck if k.startswith("blur")], "kx--", lw=0.8, label="control: clean image transformed")
axes[0].set_xlabel("JPEG quality"); axes[1].set_xlabel("Gaussian blur σ (px)")
for pax in axes:
    pax.set_ylabel("ASR@2500 km (%)"); pax.set_ylim(0, 102); pax.grid(alpha=0.3)
axes[0].legend(fontsize=7); axes[0].set_title("dotted: clean FSD > 2500 km", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "transforms_asr.png"), dpi=140); plt.close(fig)
tr_md.append("![robustness](plots/transforms_asr.png)\n")
open(os.path.join(OUT_DIR, "transforms.md"), "w").write("\n".join(tr_md))

# ---------------------------------------------------------------------------- low steps --
st_md = ["## Few sampling steps: 2 to 16 (GT-based, 4/255)\n",
"Saved 4/255 perturbations re-evaluated with the shipped sampler (sigmoid schedule, projected Euler, cfg 10, 128 shared "
"initial noises) at 2-16 steps; the in-run steps ablation above covers 8-250. GeoFA-sample v2 was trained through a 16-step "
"unroll (cfg 0), GeoFA-TD through none. Attacked FSD next to the clean FSD of the same coarse sampler (which degrades "
"quickly below ~16 steps, so read the ASR against its baseline column).\n"]
fig, ax = plt.subplots(figsize=(6, 3.4))
for att, name in ATTACKS:
    store = load_store("steps", att)
    n = len(store)
    if n == 0:
        st_md.append(f"### {name}\n\nnot run yet\n"); continue
    keys = sorted([k for k in next(iter(store.values())).keys() if isinstance(next(iter(store.values()))[k], dict)], key=lambda k: int(k.split()[0]))
    st_md.append(f"### {name} — n = {n}" + ("" if n == 100 else f" (first {n} imgs, partial)") + "\n\n" + HDR.replace("| setting |", "| steps |"))
    xs, ya, yc = [], [], []
    for k in keys:
        a, c = reduce(store, k, "gps_perturbed"), reduce(store, k, "gps_source")
        st_md.append(f"| {k} | {stats(a)} | {stats(c)} |")
        xs.append(int(k.split()[0])); ya.append(asr(a)); yc.append(asr(c))
    st_md.append("")
    ax.plot(xs, ya, "o-", color=COLORS[att], label=f"{name}: attacked FSD > 2500 km")
    ax.plot(xs, yc, "x--", color=COLORS[att], alpha=0.6, label=f"{name}: clean FSD > 2500 km")
ax.set_xlabel("sampling steps (sigmoid / Euler, cfg 10)"); ax.set_ylabel("%"); ax.set_ylim(0, 102); ax.set_xticks(LOW := [2, 3, 4, 5, 6, 8, 10, 12, 16]); ax.grid(alpha=0.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(PLOT_DIR, "low_steps_asr.png"), dpi=140); plt.close(fig)
st_md.append("![few sampling steps](plots/low_steps_asr.png)\n")
open(os.path.join(OUT_DIR, "low_steps.md"), "w").write("\n".join(st_md))

# --------------------------------------------------------------------------------- budget --
def shard_stats(shards, att):
    # GeoShield (precomputed-pair runner) suffixes its state file and stores under the key "geoshield".
    suffix, key = ("_geoshield", "geoshield") if att.startswith("geoshield") else ("", att)
    ps = sorted(glob.glob(f"{shards}/{att}__w*/yfcc_seed42_eval_state{suffix}.pt"))
    if not ps:
        return None
    s = torch.load(ps[0], map_location="cpu", weights_only=False)
    eps = float(s["signature"]["attack_budgets"][0])
    true = s["results"][key]["final_step_displacement_true"][0].float()
    loc = s["location_results"][key][0]
    idx = [i for i in range(len(loc)) if loc[i] is not None and loc[i].get("predicted_gps_source") is not None and torch.isfinite(true[i])]
    clean = torch.tensor([haversine_km(loc[i]["true_gps"], loc[i]["predicted_gps_source"]).mean().item() for i in idx])
    return eps, true[idx], clean

BUDGET_ATTACKS = ATTACKS + [("encoder_v3", "GeoFA-enc. v3"), ("training_loss_v3", "GeoFA-AdvDM v3"), ("diffusion_l2_v3", "GeoFA-L2 v3"), ("l2_target", "GeoFA-L2 targeted"), ("geoshield", "GeoShield")]
bd_md = ["## Budget: 1/255 vs 4/255 (GT-based)\n",
"Both attacks retrained from scratch at ε = 1/255 (`configs/1409_FIX/config_eps1.yaml`, otherwise identical: 60 PGD steps, "
"single run, same 100 images, seed 42); 4/255 rows are the main-table runs. GeoShield at 1/255 = the external "
"GeoShield generator with epsilon 1/255 (100 sign-gradient steps with step size alpha = 1/255 = epsilon, i.e. every pixel sits at +-epsilon and only its sign is updated per step), same "
"PNG / 336 px / model-crop protocol as its 4/255 run. Attacked FSD | clean FSD.\n",
"| attack | ε | n | attacked FSD: median \\| mean km \\| ASR@200 / 750 / 2500 | clean FSD: median \\| mean km \\| >200 / >750 / >2500 |\n|---|---|---|---|---|"]
for att, name in BUDGET_ATTACKS:
    for shards in (EPS1_SHARDS, MAIN_SHARDS):
        r = shard_stats(shards, att)
        if r is None:
            bd_md.append(f"| {name} | {'1/255' if shards == EPS1_SHARDS else '4/255'} | — | not run yet | |"); continue
        eps, a, c = r
        n = len(a)
        bd_md.append(f"| {name} | {round(eps * 255)}/255 | {n}{'' if n == 100 else ' (partial)'} | {stats(a)} | {stats(c)} |")
bd_md.append("")
open(os.path.join(OUT_DIR, "budget.md"), "w").write("\n".join(bd_md))
print("\n".join(sched_md[-40:])); print("\n".join(tr_md)); print("\n".join(bd_md))
