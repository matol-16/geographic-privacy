"""
Test-set loading for adversarial evaluation.

Holds the dataset-retrieval helpers (OSV-5M and YFCC4k). These live in their own
module so that `core.py` can import them without pulling in `adversarial_eval.py`
(which itself imports `core`), avoiding a circular import.

Image selection is prefix-stable: with a fixed seed, the first N images of a large
run match those of a smaller run, so evaluations at different `n_images` stay comparable.
"""

from __future__ import annotations

import csv
import os
import random
import zipfile
from typing import Dict, List, Optional, Tuple

from PIL import Image
from huggingface_hub import hf_hub_download

GpsCoord = Tuple[float, float]
RetrievedImages = Tuple[List[Image.Image], List[GpsCoord], List[str]]
# (image path, (lat, lon), image id) for the seeded selection, without loading pixels.
ImageMetadata = Tuple[str, GpsCoord, str]


def load_osv5m_test(local_dir: str = "/Data/mathias.ollu/hf_cache/datasets/osv5m") -> None:
    """Download and extract the OSV-5M test split if it is not already present."""
    # only download if the data is not already present
    if os.path.exists(os.path.join(local_dir, "images", "test")) and \
       any(os.path.isdir(os.path.join(local_dir, "images", "test", d)) for d in os.listdir(os.path.join(local_dir, "images", "test"))):
        return
    for i in range(5):
        hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir=local_dir)
    # extract zip files
    img_dir = os.path.join(local_dir, "images", "test")
    for f in os.listdir(img_dir):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(img_dir, f), 'r') as z:
                z.extractall(img_dir)
    return


def _default_yfcc_dir() -> str:
    return "/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k"


def _load_yfcc_rows(local_dir: str) -> List[dict]:
    """Read info.txt and keep only rows whose image exists on disk."""
    info_path = os.path.join(local_dir, "info.txt")
    img_dir = os.path.join(local_dir, "images")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"YFCC4k info.txt not found at {info_path}. Run build_yfcc4k_from_revisiting_im2gps.py first.")

    rows = []
    with open(info_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            photo_id = parts[0]
            lon = float(parts[1])
            lat = float(parts[2])
            img_path = os.path.join(img_dir, f"{photo_id}.jpg")
            if os.path.exists(img_path):
                rows.append({"id": photo_id, "path": img_path, "latitude": lat, "longitude": lon})
    return rows


def _select_prefix_stable(rows: List[dict], n_images_to_eval: int, seed: int) -> List[dict]:
    """Deterministic, prefix-stable selection shared by every YFCC retrieval path.

    Sort by id, shuffle with a seeded RNG, then take the first N. With a fixed seed
    the first N images of a larger run match those of a smaller run, so evaluations
    at different ``n_images_to_eval`` stay comparable.
    """
    rows = sorted(rows, key=lambda r: str(r["id"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[: min(n_images_to_eval, len(rows))]


def select_yfcc_image_paths(
    n_images_to_eval: int = 100,
    seed: int = 0,
    local_dir: Optional[str] = None,
) -> List[str]:
    """Return the file paths of the seeded YFCC4k selection without loading images.

    Uses the exact same selection as ``retrieve_yfcc_images`` so external tooling
    (e.g. GeoShield) can attack/evaluate the same images as the attack backbone.
    """
    return [path for path, _, _ in select_yfcc_image_metadata(n_images_to_eval, seed, local_dir)]


def select_yfcc_image_metadata(
    n_images_to_eval: int = 100,
    seed: int = 0,
    local_dir: Optional[str] = None,
) -> List[ImageMetadata]:
    """(path, (lat, lon), id) for the seeded YFCC4k selection, without loading pixels."""
    if local_dir is None:
        local_dir = _default_yfcc_dir()
    samples = _select_prefix_stable(_load_yfcc_rows(local_dir), n_images_to_eval, seed)
    return [(s["path"], (s["latitude"], s["longitude"]), str(s["id"])) for s in samples]


def retrieve_yfcc_images(
    n_images_to_eval: int = 100,
    seed: int = 0,
    use_real_gps: bool = False,
    local_dir: Optional[str] = None,
    im_idx=None,  # If specified, retrieves only the image with this index in the dataset (after sorting by ID). Useful for debugging with a single image.
) -> RetrievedImages:
    """Load up to ``n_images_to_eval`` YFCC4k images with their ground-truth GPS labels."""
    if local_dir is None:
        local_dir = _default_yfcc_dir()
    info_path = os.path.join(local_dir, "info.txt")
    img_dir = os.path.join(local_dir, "images")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"YFCC4k info.txt not found at {info_path}. Run build_yfcc4k_from_revisiting_im2gps.py first.")

    if im_idx is not None:
        # load image corresponding to this index (im_idx.jpg)
        img_path = os.path.join(img_dir, f"{str(im_idx)}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image with index {im_idx} not found at {img_path}. Check if im_idx is correct and if images are properly stored.")
        img = Image.open(img_path).convert("RGB")
        # retrieve metadata for this image from info.txt
        with open(info_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                photo_id = parts[0]
                if photo_id == str(im_idx):
                    lon = float(parts[1])
                    lat = float(parts[2])
                    gps = (lat, lon)
                    return [img], [gps], [photo_id]
        raise ValueError(f"Metadata for image with index {im_idx} not found in info.txt. Check if im_idx is correct and if info.txt is properly formatted.")

    # Keep selection prefix-stable across different n_images_to_eval values:
    # with a fixed seed, first N images of a larger run match a smaller run.
    samples = _select_prefix_stable(_load_yfcc_rows(local_dir), n_images_to_eval, seed)

    source_images = [Image.open(s["path"]).convert("RGB") for s in samples]
    source_gps = [(s["latitude"], s["longitude"]) for s in samples]
    source_image_ids = [s["id"] for s in samples]
    print(f"Loaded {len(source_images)} images from YFCC4k.")
    return source_images, source_gps, source_image_ids


def _default_osv_dir() -> str:
    return "/Data/mathias.ollu/hf_cache/datasets/osv5m"


def _load_osv_rows(local_dir: str) -> List[dict]:
    """Read the OSV-5M test split and keep only rows whose image exists on disk.

    Returns rows shaped like the YFCC rows ({id, path, latitude, longitude}) so the
    shared ``_select_prefix_stable`` selection applies to both datasets identically.
    """
    load_osv5m_test(local_dir=local_dir)  # download & extract if needed

    csv_path = os.path.join(local_dir, "test.csv")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    img_dir = os.path.join(local_dir, "images", "test")
    subdirs = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
    id_to_path = {}
    for sd in subdirs:
        sd_path = os.path.join(img_dir, sd)
        for fname in os.listdir(sd_path):
            img_id = os.path.splitext(fname)[0]
            id_to_path[img_id] = os.path.join(sd_path, fname)

    return [
        {
            "id": r["id"],
            "path": id_to_path[r["id"]],
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
        }
        for r in rows
        if r["id"] in id_to_path
    ]


def retrieve_osv_images(
    n_images_to_eval: int = 100,
    seed: int = 0,
    use_real_gps: bool = False,
    local_dir: Optional[str] = None,
) -> RetrievedImages:
    """Load up to ``n_images_to_eval`` OSV-5M test images with their ground-truth GPS labels."""
    if local_dir is None:
        local_dir = _default_osv_dir()
    # Same seeded, prefix-stable selection as YFCC (and as select_osv_image_metadata).
    samples = _select_prefix_stable(_load_osv_rows(local_dir), n_images_to_eval, seed)

    source_images = [Image.open(s["path"]).convert("RGB") for s in samples]
    source_gps = [(s["latitude"], s["longitude"]) for s in samples]
    source_image_ids = [str(s["id"]) for s in samples]
    print(f"Loaded {len(source_images)} images from OSV-5M test set.")
    return source_images, source_gps, source_image_ids


def select_osv_image_metadata(
    n_images_to_eval: int = 100,
    seed: int = 0,
    local_dir: Optional[str] = None,
) -> List[ImageMetadata]:
    """(path, (lat, lon), id) for the seeded OSV-5M selection, without loading pixels."""
    if local_dir is None:
        local_dir = _default_osv_dir()
    samples = _select_prefix_stable(_load_osv_rows(local_dir), n_images_to_eval, seed)
    return [(s["path"], (s["latitude"], s["longitude"]), str(s["id"])) for s in samples]


def select_image_metadata(
    dataset: str,
    n_images_to_eval: int = 100,
    seed: int = 0,
    local_dir: Optional[str] = None,
) -> List[ImageMetadata]:
    """Dataset-agnostic seeded selection of (path, (lat, lon), id), without loading pixels.

    Matches the order/selection used by ``retrieve_{yfcc,osv}_images`` so external tooling
    (GeoShield) attacks/evaluates exactly the same images as the in-process attacks.
    """
    if dataset == "yfcc":
        return select_yfcc_image_metadata(n_images_to_eval, seed, local_dir)
    if dataset == "osv":
        return select_osv_image_metadata(n_images_to_eval, seed, local_dir)
    raise ValueError(f"Unknown dataset: {dataset}")
