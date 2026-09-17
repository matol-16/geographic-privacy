#!/usr/bin/env python
"""Pre/post-processing exactness checks for the attacked pipelines (YFCC/DINOv2, OSV/StreetCLIP).

For every requested dataset pipeline and a few images this verifies that

1. the protected image built by ``add_perturbation_to_image`` (clean pixels + delta, clamped,
   8-bit) re-enters the pipeline as exactly ``clean_tensor + delta`` (up to 8-bit rounding):
   the pipeline's own preprocessing must be a no-op on the exported square, otherwise the
   evaluated perturbation would not be the trained one (crop shift / resize / JPEG errors);
2. a zero perturbation reproduces the clean tensor bit-exactly (clean == protected(0));
3. the exported perturbation respects the l_inf budget on the 8-bit scale
   (``|protected - clean| <= round(eps * 255)`` grey levels);
4. the GeoShield "model-exact" clean copy (utils.geoshield with a pipeline) equals the
   evaluator's clean tensor;
5. ``pipeline.cond_preprocessing`` on the raw image and on the exported clean copy give the
   same embedding (max abs diff 0).

Usage:
  python scripts/tests/check_preprocessing_roundtrip.py --datasets yfcc osv --images img1.jpg ...
It needs no dataset: any images work (non-square ones are the interesting case).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
os.chdir(os.path.dirname(os.path.dirname(HERE)))

from utils.adversarial_utils import (  # noqa: E402
    add_perturbation_to_image,
    compute_embedding,
    conditional_preprocessing,
    normalized_to_pixels,
    pixels_to_pil,
)
from utils.pipe_trajectory import PlonkPipelineTrajectory  # noqa: E402

PIPELINES = {"yfcc": "nicolas-dufour/PLONK_YFCC", "osv": "nicolas-dufour/PLONK_OSV_5M"}


def check(pipeline, image_path: str, eps: float, device: str) -> list[str]:
    errors: list[str] = []
    with Image.open(image_path) as f:
        image = f.convert("RGB")
    clean = conditional_preprocessing(image, pipeline, device=device)           # [1,3,H,W] normalised
    clean_px = normalized_to_pixels(clean, pipeline)                             # [0,1]
    H, W = clean.shape[-2:]
    if (H, W) != (336, 336):
        errors.append(f"model input is {H}x{W}, expected 336x336")

    # 2. zero perturbation -> identical clean tensor (crop/resize must be a no-op on the square)
    protected0 = add_perturbation_to_image(image, torch.zeros_like(clean), pipeline)
    re0 = conditional_preprocessing(protected0, pipeline, device=device)
    d0 = (re0 - clean).abs().max().item()
    if d0 > 1e-5:
        errors.append(f"zero-delta round trip differs from the clean tensor: max|diff|={d0:.3e} (normalised units)")

    # 1./3. random +-eps perturbation -> protected pixels == clamp(clean + delta) quantised
    g = torch.Generator(device="cpu").manual_seed(0)
    delta = (torch.rand(clean.shape, generator=g) * 2 - 1).to(device) * eps
    protected = add_perturbation_to_image(image, delta, pipeline)
    re = conditional_preprocessing(protected, pipeline, device=device)
    re_px = normalized_to_pixels(re, pipeline)
    expected_px = (clean_px + delta).clamp(0, 1)
    d1 = ((re_px - expected_px).abs().max() * 255).item()
    if d1 > 0.5 + 1e-3:
        errors.append(f"protected image differs from clean+delta by {d1:.3f} grey levels (>0.5 = beyond 8-bit rounding)")
    levels = ((re_px - clean_px).abs().max() * 255).round().item()
    if levels > round(eps * 255):
        errors.append(f"exported perturbation reaches {levels:.0f} grey levels > budget {round(eps * 255)}")

    # 4. GeoShield model-exact clean copy == evaluator clean tensor
    copy_pil = pixels_to_pil(normalized_to_pixels(conditional_preprocessing(image, pipeline, device="cpu"), pipeline).squeeze(0))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "clean.png")
        copy_pil.save(p, format="PNG")
        with Image.open(p) as f:
            reloaded = f.convert("RGB")
    re_copy = conditional_preprocessing(reloaded, pipeline, device=device)
    d4 = (re_copy - clean).abs().max().item()
    if d4 > 1e-5:
        errors.append(f"GeoShield model-exact clean copy differs from the clean tensor: max|diff|={d4:.3e}")

    # 5. embeddings of raw image vs exported clean copy
    with torch.no_grad():
        e_raw = compute_embedding(clean, 1, pipeline, device=device, track_grad=False)
        e_copy = compute_embedding(re_copy, 1, pipeline, device=device, track_grad=False)
    d5 = (e_raw - e_copy).abs().max().item()
    if d5 > 1e-4:
        errors.append(f"embedding of the exported clean copy differs from the raw image's: max|diff|={d5:.3e}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["yfcc", "osv"], choices=list(PIPELINES))
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0039, 0.0157, 0.0314])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    failures = 0
    for ds in args.datasets:
        print(f"\n=== {ds}: {PIPELINES[ds]}")
        pipeline = PlonkPipelineTrajectory(PIPELINES[ds]).to(args.device)
        for path in args.images:
            with Image.open(path) as f:
                size = f.size
            for eps in args.eps:
                errs = check(pipeline, path, eps, args.device)
                status = "OK " if not errs else "FAIL"
                print(f"  [{status}] {os.path.basename(path)} {size} eps={eps}")
                for e in errs:
                    print(f"         - {e}")
                failures += len(errs)
        del pipeline
        torch.cuda.empty_cache()
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
