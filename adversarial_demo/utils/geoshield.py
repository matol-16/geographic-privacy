"""Fold GeoShield (an out-of-process attack) into the evaluate-dataset pipeline.

GeoShield is not a trainable attack type in this repo: its perturbations are produced
by the separate Geoshield repository (``Geoshield/geoshield.py``). This module ports
the former ``scripts/geoshield_common.sh`` orchestration into Python so a single
``evaluate-dataset`` run can train the in-process attacks AND generate + evaluate
GeoShield, storing its results under the same ``<dataset>_geoshield_results.pt`` naming
as every other attack (so the combined plots overlay everything, no bash glue).

Pipeline (yfcc or osv, mirroring the seeded backbone selection):
  1. Select the same seeded, prefix-stable clean images the trainable attacks use
     (``utils.datasets.select_image_metadata``) and copy them into a clean dir, keeping
     a filename -> ground-truth GPS map so the precomputed evaluator can report the
     true-position metric just like the in-process attacks.
  2. Run ``Geoshield/geoshield.py`` once per budget (epsilon on the 0-255 scale,
     derived from the L-inf budget unless overridden in the config).
  3. Return the per-budget (clean_dir, attacked_dir) pairs and the GPS map for the
     existing precomputed-pair evaluator (``PrecomputedPairEvaluationRunner``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.datasets import GpsCoord, select_image_metadata

GEOSHIELD_ATTACK_NAME = "geoshield"


def geoshield_epsilon_for_budget(budget: float) -> int:
    """GeoShield epsilon (0-255 scale) matching an L-inf budget on the 0-1 scale.

    e.g. 0.0157 -> 4, 0.0314 -> 8. Keeps ``attack_budgets`` the single source of truth
    so the GeoShield run is automatically aligned with the trainable attacks.
    """
    return int(round(float(budget) * 255))


def default_repos_root() -> Path:
    """Directory holding both the plonk repo and the Geoshield repo.

    ``utils/geoshield.py`` -> utils -> adversarial_demo -> plonk -> repos_root.
    """
    return Path(__file__).resolve().parents[3]


def _select_clean_images(
    dataset: str,
    n_images: int,
    seed: int,
    local_dir: Optional[str],
    clean_dir: str,
    window_start: int = 0,
    window_end: Optional[int] = None,
    clean_image_format: str = "png",
    clean_preprocess: str = "none",
    input_res: Optional[int] = None,
    pipeline=None,
) -> Dict[str, GpsCoord]:
    """Copy the seeded, prefix-stable backbone selection into ``clean_dir``.

    The clean dir is repopulated from scratch each run so stale images from a previous
    selection cannot leak into the GeoShield evaluation. Returns a map from each copied
    file's name to its ground-truth GPS, so the precomputed evaluator can compute the
    true-position metric (the pair keys are the file names). ``window_start``/``window_end``
    select the same image slice a trainable-attack shard would use.

    ``clean_preprocess="model"`` writes the *model-preprocessed* square instead of the raw
    file: PLONK's ``CenterCrop(ratio="1:1") -> Resize(input_res, bicubic)``. GeoShield
    preprocesses with ``Resize(input_res) -> CenterCrop(input_res)``: the same square, but
    cropped at an integer offset *after* the resize, i.e. shifted by a fraction of a pixel
    w.r.t. PLONK's crop-then-resize. That sub-pixel shift alone (no perturbation) differs from
    the evaluator's clean tensor by ~4/255 on average, and flips the prediction on ~8 % of
    YFCC images. On an ``input_res`` square both chains are exact no-ops, so GeoShield's
    perturbation is then measured against precisely the clean tensor the trainable attacks
    are evaluated on. PIL's resize already yields uint8, so the PNG is bit-identical to what
    the pipeline computes from the raw file. Requires ``clean_image_format="png"``.

    When ``pipeline`` is given, the square is produced by the pipeline's OWN conditional
    preprocessing (``conditional_preprocessing`` -> pixels -> 8 bit), so it is exact for every
    encoder: DINOv2 (YFCC: CenterCrop(1:1) -> Resize(336)) and StreetCLIP (OSV: the CLIP
    processor's Resize(shortest=336) -> CenterCrop(336), whose crop offset differs from the
    DINOv2 chain by up to a pixel on odd sizes). Without a pipeline the DINOv2 chain is used
    (only correct for the DINOv2-encoded datasets).
    """
    metadata = select_image_metadata(
        dataset, int(n_images), seed=int(seed), local_dir=local_dir,
        window_start=int(window_start), window_end=window_end,
    )
    if not metadata:
        raise RuntimeError("GeoShield: no clean images selected; check dataset root / config.")

    clean_root = Path(clean_dir)
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)

    # GeoShield saves its protected image with the *same extension* as the clean file
    # (``torchvision.utils.save_image`` -> PIL picks the format from the suffix). With the
    # datasets' ``.jpg`` files that means the perturbation goes through JPEG (quality 75)
    # before evaluation, while the in-process attacks are evaluated losslessly. Copying the
    # clean images as PNG (lossless re-encode of the decoded pixels) keeps the pipeline
    # lossless end to end. ``clean_image_format="jpg"`` restores the old (lossy) behaviour.
    fmt = str(clean_image_format).lower().lstrip(".")
    if fmt not in ("png", "jpg", "jpeg"):
        raise ValueError(f"geoshield.clean_image_format must be png or jpg, got {clean_image_format!r}")
    preprocess = str(clean_preprocess or "none").lower()
    if preprocess not in ("none", "model"):
        raise ValueError(f"geoshield.clean_preprocess must be 'none' or 'model', got {clean_preprocess!r}")
    if preprocess == "model":
        if fmt != "png":
            raise ValueError("geoshield.clean_preprocess='model' requires clean_image_format='png' (lossless).")
        if input_res is None:
            raise ValueError("geoshield.clean_preprocess='model' requires geoshield.input_res (the model's input size).")
        if pipeline is not None:
            from utils.adversarial_utils import conditional_preprocessing, normalized_to_pixels, pixels_to_pil

            def model_transform(im):
                # Exactly the clean tensor the evaluator builds, de-normalised and quantised
                # (the preprocessed pixels are integers/255, so the round trip is lossless).
                tensor = conditional_preprocessing(im, pipeline, device="cpu")
                pil = pixels_to_pil(normalized_to_pixels(tensor, pipeline).squeeze(0))
                if pil.size != (int(input_res), int(input_res)):
                    raise RuntimeError(
                        f"geoshield.input_res={input_res} does not match the model input {pil.size}"
                    )
                return pil
        else:
            if dataset != "yfcc":
                print(
                    "GeoShield WARNING: clean_preprocess='model' without a pipeline falls back to the "
                    f"DINOv2 chain, which is not the {dataset} encoder's preprocessing."
                )
            from torchvision import transforms
            from plonk.utils.image_processing import CenterCrop  # PLONK's own crop (see plonk/pipe.py)
            model_transform = transforms.Compose([
                CenterCrop(ratio="1:1"),
                transforms.Resize(int(input_res), interpolation=transforms.InterpolationMode.BICUBIC),
            ])
    gps_by_filename: Dict[str, GpsCoord] = {}
    for path, gps, _ in metadata:
        if fmt == "png":
            from PIL import Image
            filename = f"{Path(path).stem}.png"
            with Image.open(path) as im:
                im = im.convert("RGB")
                if preprocess == "model":
                    im = model_transform(im)
                im.save(clean_root / filename, format="PNG")
        else:
            filename = Path(path).name
            shutil.copy(path, clean_root / filename)
        gps_by_filename[filename] = gps
    print(f"GeoShield: copied {len(metadata)} clean images to {clean_root} (format={fmt}, preprocess={preprocess}).")
    return gps_by_filename


def _run_geoshield_generation(
    repos_root: str,
    script: str,
    python_exe: str,
    clean_dir: str,
    output_path: str,
    n_images: int,
    epsilon: int,
    steps: int,
    input_res: Optional[int] = None,
    reuse_existing: bool = False,
) -> Path:
    """Run ``Geoshield/geoshield.py`` once and return the dir of attacked images.

    ``reuse_existing``: if an attacked dir for this output path already holds ``n_images``
    files, return it without re-running GeoShield (the clean copies are deterministic, so
    outputs of an interrupted run stay valid).

    ``input_res`` overrides GeoShield's ``model.input_res`` (default 640 in its config). The
    attacked model takes 336x336 inputs, so running GeoShield at 336 applies its l_inf
    budget at the evaluated resolution instead of at 640 px followed by a bicubic downscale.

    GeoShield writes to ``<output>/img/<config_hash>+geoshield/<clean_basename>/``; the
    most recently written matching dir is returned (mirroring the former bash ``find``).
    """
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_basename = Path(clean_dir).name

    if reuse_existing:
        done = [d for d in (output_root / "img").glob(f"*+geoshield/{clean_basename}")
                if d.is_dir() and sum(1 for f in d.iterdir() if f.is_file()) >= int(n_images)]
        if done:
            attacked_dir = max(done, key=lambda d: d.stat().st_mtime)
            print(f"GeoShield: reusing existing attacked images (eps={epsilon}): {attacked_dir}")
            return attacked_dir

    cmd = [
        python_exe,
        script,
        f"data.cle_data_path={clean_dir}",
        f"data.tgt_data_path={clean_dir}",
        f"data.output={output_root}",
        f"data.num_samples={int(n_images)}",
        f"optim.epsilon={int(epsilon)}",
        f"optim.steps={int(steps)}",
    ]
    if input_res is not None:
        cmd.append(f"model.input_res={int(input_res)}")
    print(f"GeoShield: running attack (epsilon={epsilon}/255, steps={steps}, input_res={input_res or 'default'})...")
    subprocess.run(cmd, cwd=str(repos_root), check=True)

    img_root = output_root / "img"
    candidates = [d for d in img_root.glob(f"*+geoshield/{clean_basename}") if d.is_dir()]
    if not candidates:
        raise RuntimeError(
            f"GeoShield: could not locate attacked images under {img_root} "
            f"(expected <hash>+geoshield/{clean_basename})."
        )
    attacked_dir = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"GeoShield: attacked images (eps={epsilon}): {attacked_dir}")
    return attacked_dir


def generate_geoshield_pairs(
    config: Dict[str, Any],
    dataset: str,
    n_images: int,
    attack_budgets: Sequence[float],
    seed: int,
    geoshield_cfg: Optional[Dict[str, Any]] = None,
    window_start: int = 0,
    window_end: Optional[int] = None,
    run_tag: Optional[str] = None,
    pipeline=None,
) -> Tuple[List[str], List[str], Dict[str, GpsCoord]]:
    """Generate GeoShield-attacked images for every budget.

    ``pipeline`` (the attacked PLONK pipeline) makes ``clean_preprocess="model"`` use that
    model's own conditional preprocessing for the clean copies (see ``_select_clean_images``).

    Returns ``(clean_image_dirs, attacked_image_dirs, gps_by_filename)`` -- one dir entry
    per budget, ready for ``PrecomputedPairEvaluationRunner``, plus a filename -> GPS map
    so the evaluator can report the true-position metric. The clean dir is shared across
    budgets (same seeded selection); only the attacked dir changes with epsilon. Clean
    dir and output base are made dataset-specific so yfcc and osv runs never collide.

    For multi-node sharding, ``window_start``/``window_end`` restrict the selection to one
    image window and ``run_tag`` (e.g. ``"w000000_000100"``) is appended to the clean dir
    and output base so concurrent shards never clobber each other's images. ``n_images`` is
    the size of the full seeded pool the window is taken from.
    """
    geoshield_cfg = dict(geoshield_cfg or {})
    repos_root = geoshield_cfg.get("repos_root") or str(default_repos_root())
    script = geoshield_cfg.get("script", "Geoshield/geoshield.py")
    python_exe = geoshield_cfg.get("python") or sys.executable
    steps = int(geoshield_cfg.get("steps", 100))
    input_res = geoshield_cfg.get("input_res")  # None => GeoShield's own default (640)
    clean_image_format = geoshield_cfg.get("clean_image_format", "png")
    clean_preprocess = geoshield_cfg.get("clean_preprocess", "none")  # "model" => PLONK crop+resize square
    reuse_existing = bool(geoshield_cfg.get("reuse_existing", False))
    clean_dir_base = geoshield_cfg.get("clean_dir")
    output_base = geoshield_cfg.get("output_base")
    if not clean_dir_base or not output_base:
        raise ValueError("geoshield.clean_dir and geoshield.output_base must be set in the config.")

    script_path = Path(repos_root) / script
    if not script_path.exists():
        raise FileNotFoundError(
            f"GeoShield script not found: {script_path}. "
            "Set geoshield.repos_root / geoshield.script in the config."
        )

    attack_budgets = list(attack_budgets)
    epsilons = geoshield_cfg.get("epsilons")
    if epsilons is None:
        epsilons = [geoshield_epsilon_for_budget(b) for b in attack_budgets]
    if len(epsilons) != len(attack_budgets):
        raise ValueError(
            f"geoshield.epsilons ({len(epsilons)}) must match the number of attack_budgets "
            f"({len(attack_budgets)})."
        )

    # Keep yfcc and osv outputs separate so concurrent / successive runs never collide.
    # For sharded runs the run_tag (window range) further isolates each shard's dirs.
    clean_dir = str(Path(clean_dir_base) / dataset / run_tag) if run_tag else str(Path(clean_dir_base) / dataset)
    output_base = f"{output_base}_{dataset}_{run_tag}" if run_tag else f"{output_base}_{dataset}"

    local_dir = (config.get("data_dirs", {}) or {}).get(dataset)
    gps_by_filename = _select_clean_images(
        dataset, n_images, seed, local_dir, clean_dir, window_start, window_end,
        clean_image_format=clean_image_format,
        clean_preprocess=clean_preprocess,
        input_res=int(input_res) if input_res is not None else None,
        pipeline=pipeline,
    )
    num_images = len(gps_by_filename)  # actual images in this (windowed) selection

    clean_dirs: List[str] = []
    attacked_dirs: List[str] = []
    for epsilon in epsilons:
        output_path = f"{output_base}_e_{int(epsilon)}"
        attacked_dir = _run_geoshield_generation(
            repos_root=repos_root,
            script=script,
            python_exe=python_exe,
            clean_dir=clean_dir,
            output_path=output_path,
            n_images=num_images,
            epsilon=int(epsilon),
            steps=steps,
            input_res=int(input_res) if input_res is not None else None,
            reuse_existing=reuse_existing,
        )
        clean_dirs.append(str(clean_dir))
        attacked_dirs.append(str(attacked_dir))
    return clean_dirs, attacked_dirs, gps_by_filename
