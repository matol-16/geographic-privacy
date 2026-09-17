"""1409_FIX: how do the perturbations differ between budgets 1/255 and 4/255?

For GeoFA-sample v2 and GeoFA-TD v3 (old fwd.), pairs the saved deltas of the 4/255 run
(1409_FIX_ablations) and the 1/255 run (1409_FIX_eps1) image by image and reports
  - size: L_inf, L2, mean |delta| (in /255 units), fraction of pixels at the budget (saturated),
    PSNR of the protected image vs the clean one;
  - structure: cosine similarity and sign agreement between delta_1 and delta_4 of the same
    image (1 = the 4/255 perturbation is a scaled-up copy of the 1/255 one), and the fraction of
    the perturbation energy above the Nyquist/4 spatial frequency (high-frequency share);
plus an image grid (clean | delta_1 x255/1 | protected 1/255 | delta_4 x255/4 | protected 4/255)
for a few images. Writes results/1409_FIX/delta_budget.md + plots/delta_budget_<attack>.png.
"""
import glob, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.getcwd())
import yaml
from main import get_pipeline_for_model_type
from utils.adversarial_utils import add_perturbation_to_image, conditional_preprocessing, normalized_to_pixels, pixels_to_pil
from utils.datasets import retrieve_yfcc_images

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PLOT_DIR = os.path.join(ROOT, "results/1409_FIX/plots")
OUT = os.path.join(ROOT, "results/1409_FIX/delta_budget.md")
RUNS = {"4/255": "/Data/mathias.ollu/results/1409_FIX_ablations/shards", "1/255": "/Data/mathias.ollu/results/1409_FIX_eps1/shards"}
ATTACKS = [("sampling_v2", "GeoFA-sample v2"), ("dtd_v3_old_forward", "GeoFA-TD v3 (old fwd.)")]
SHOW = 4  # images in the grid


def load_deltas(att, shards):
    out = {}
    for f in glob.glob(f"{shards}/{att}__w*/deltas/*.pt"):
        r = torch.load(f, map_location="cpu", weights_only=False)
        out[str(r["image_id"])] = (r["delta"].squeeze(0), float(r["eps"]))
    return out


def hf_share(delta):
    """Share of |FFT|^2 energy at spatial frequencies above N/4 (either axis), mean over channels."""
    f = torch.fft.fftshift(torch.fft.fft2(delta), dim=(-2, -1)).abs() ** 2
    n = delta.shape[-1]; c = n // 2; k = n // 4
    inner = f[..., c - k:c + k, c - k:c + k].sum()
    return float(1 - inner / f.sum())


config = yaml.safe_load(open("configs/1409_FIX/config.yaml"))
pipeline = get_pipeline_for_model_type(config, "yfcc", "")
images, gps, ids = retrieve_yfcc_images(n_images_to_eval=4000, seed=42, use_real_gps=True,
                                        local_dir=config["data_dirs"]["yfcc"], window_start=0, window_end=100)
by_id = {str(i): img for i, img in zip(ids, images)}

md = ["## Perturbations at 1/255 vs 4/255\n",
      "Same images, same attack, the only change is ε. `δ` is the saved best perturbation (pixel units); saturated = fraction of "
      "pixel values at ±ε; PSNR = protected vs clean 336² square; cos / sign = cosine similarity / sign agreement between δ₁ and δ₄ "
      "of the same image (a scaled copy would give 1 / 1, independent random signs 0 / 0.5); HF share = fraction of δ's spectral "
      "energy above a quarter of the Nyquist frequency (white noise ≈ 0.75). Means over the 100 images.\n",
      "| attack | ε | L∞ (/255) | L2 | mean \\|δ\\| (/255) | saturated | PSNR (dB) | cos(δ₁, δ₄) | sign agreement | HF share |\n|---|---|---|---|---|---|---|---|---|---|"]
figs = []
for att, name in ATTACKS:
    d = {k: load_deltas(att, v) for k, v in RUNS.items()}
    common = sorted(set(d["4/255"]) & set(d["1/255"]))
    stats = {k: dict(linf=[], l2=[], mean=[], sat=[], psnr=[], hf=[]) for k in RUNS}
    cos, sign = [], []
    for iid in common:
        clean = normalized_to_pixels(conditional_preprocessing(by_id[iid], pipeline, device="cuda"), pipeline).squeeze(0).cpu()
        for k in RUNS:
            delta, eps = d[k][iid]
            prot = torch.tensor(np.asarray(add_perturbation_to_image(by_id[iid], delta.cuda(), pipeline)), dtype=torch.float32).permute(2, 0, 1) / 255
            clean_q = torch.tensor(np.asarray(pixels_to_pil(clean)), dtype=torch.float32).permute(2, 0, 1) / 255
            mse = ((prot - clean_q) ** 2).mean()
            s = stats[k]
            s["linf"].append(delta.abs().max().item() * 255); s["l2"].append(delta.norm().item()); s["mean"].append(delta.abs().mean().item() * 255)
            s["sat"].append((delta.abs() >= eps * 0.999).float().mean().item()); s["psnr"].append(float(10 * torch.log10(1 / mse))); s["hf"].append(hf_share(delta))
        a, b = d["1/255"][iid][0].flatten(), d["4/255"][iid][0].flatten()
        cos.append(torch.nn.functional.cosine_similarity(a, b, dim=0).item()); sign.append((a.sign() == b.sign()).float().mean().item())
    m = lambda v: float(np.mean(v))
    for k in ("1/255", "4/255"):
        s = stats[k]
        md.append(f"| {name} | {k} | {m(s['linf']):.2f} | {m(s['l2']):.2f} | {m(s['mean']):.2f} | {m(s['sat']) * 100:.0f} % | {m(s['psnr']):.1f} | "
                  + (f"{m(cos):.2f} | {m(sign):.2f} |" if k == "1/255" else "(same pair) | |") + f" {m(s['hf']):.2f} |")
    # grid
    fig, axes = plt.subplots(SHOW, 5, figsize=(12.5, 2.6 * SHOW))
    for r, iid in enumerate(common[:SHOW]):
        clean_pil = pixels_to_pil(normalized_to_pixels(conditional_preprocessing(by_id[iid], pipeline, device="cuda"), pipeline).squeeze(0).cpu())
        cols = [(clean_pil, "clean")]
        for k in ("1/255", "4/255"):
            delta, eps = d[k][iid]
            amp = ((delta / eps).clamp(-1, 1) * 0.5 + 0.5).permute(1, 2, 0).numpy()
            cols += [(amp, f"δ at {k} (×255/{round(eps * 255)}, grey = 0)"), (add_perturbation_to_image(by_id[iid], delta.cuda(), pipeline), f"protected {k}")]
        for c, (im, title) in enumerate(cols):
            axes[r, c].imshow(im); axes[r, c].set_axis_off()
            if r == 0: axes[r, c].set_title(title, fontsize=9)
        axes[r, 0].text(-0.02, 0.5, iid, transform=axes[r, 0].transAxes, rotation=90, va="center", ha="right", fontsize=7)
    fig.suptitle(f"{name}: perturbation at 1/255 vs 4/255", fontsize=10); fig.tight_layout()
    os.makedirs(PLOT_DIR, exist_ok=True); fn = f"delta_budget_{att}.png"; fig.savefig(os.path.join(PLOT_DIR, fn), dpi=110); plt.close(fig)
    figs.append(f"![{name} δ at 1/255 vs 4/255](plots/{fn})\n")
md.append("")
md.extend(figs)
open(OUT, "w").write("\n".join(md))
print("\n".join(md))
