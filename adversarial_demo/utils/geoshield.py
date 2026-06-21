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
) -> Dict[str, GpsCoord]:
    """Copy the seeded, prefix-stable backbone selection into ``clean_dir``.

    The clean dir is repopulated from scratch each run so stale images from a previous
    selection cannot leak into the GeoShield evaluation. Returns a map from each copied
    file's name to its ground-truth GPS, so the precomputed evaluator can compute the
    true-position metric (the pair keys are the file names).
    """
    metadata = select_image_metadata(dataset, int(n_images), seed=int(seed), local_dir=local_dir)
    if not metadata:
        raise RuntimeError("GeoShield: no clean images selected; check dataset root / config.")

    clean_root = Path(clean_dir)
    if clean_root.exists():
        shutil.rmtree(clean_root)
    clean_root.mkdir(parents=True, exist_ok=True)

    gps_by_filename: Dict[str, GpsCoord] = {}
    for path, gps, _ in metadata:
        filename = Path(path).name
        shutil.copy(path, clean_root / filename)
        gps_by_filename[filename] = gps
    print(f"GeoShield: copied {len(metadata)} clean images to {clean_root}.")
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
) -> Path:
    """Run ``Geoshield/geoshield.py`` once and return the dir of attacked images.

    GeoShield writes to ``<output>/img/<config_hash>+geoshield/<clean_basename>/``; the
    most recently written matching dir is returned (mirroring the former bash ``find``).
    """
    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)
    clean_basename = Path(clean_dir).name

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
    print(f"GeoShield: running attack (epsilon={epsilon}/255, steps={steps})...")
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
) -> Tuple[List[str], List[str], Dict[str, GpsCoord]]:
    """Generate GeoShield-attacked images for every budget.

    Returns ``(clean_image_dirs, attacked_image_dirs, gps_by_filename)`` -- one dir entry
    per budget, ready for ``PrecomputedPairEvaluationRunner``, plus a filename -> GPS map
    so the evaluator can report the true-position metric. The clean dir is shared across
    budgets (same seeded selection); only the attacked dir changes with epsilon. Clean
    dir and output base are made dataset-specific so yfcc and osv runs never collide.
    """
    geoshield_cfg = dict(geoshield_cfg or {})
    repos_root = geoshield_cfg.get("repos_root") or str(default_repos_root())
    script = geoshield_cfg.get("script", "Geoshield/geoshield.py")
    python_exe = geoshield_cfg.get("python") or sys.executable
    steps = int(geoshield_cfg.get("steps", 100))
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
    clean_dir = str(Path(clean_dir_base) / dataset)
    output_base = f"{output_base}_{dataset}"

    local_dir = (config.get("data_dirs", {}) or {}).get(dataset)
    gps_by_filename = _select_clean_images(dataset, n_images, seed, local_dir, clean_dir)

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
            n_images=n_images,
            epsilon=int(epsilon),
            steps=steps,
        )
        clean_dirs.append(str(clean_dir))
        attacked_dirs.append(str(attacked_dir))
    return clean_dirs, attacked_dirs, gps_by_filename
