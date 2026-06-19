import argparse
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


#urls retrieved from https://github.com/lugiavn/revisiting-im2gps?tab=readme-ov-file
IMAGES_PAGE_URL = "http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip"
METADATA_PAGE_URL = "http://www.mediafire.com/file/8v2j565997i5jed/0aaaa.r.imagedata.txt"

DEFAULT_CONFIG_PATH = "/users/eleves-b/2023/mathias.ollu/repos/plonk/adversarial_demo/config.yaml"


def load_config(config_path: str) -> dict:
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path(__file__).resolve().parent / config_file
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(value: str, base_dir: Path, label: str) -> Path:
    if not value:
        raise ValueError(f"Missing required config value for {label}")

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def build_default_paths(config: dict) -> dict:
    data_root = Path(config.get("data_root", "/Data/mathias.ollu/hf_cache/datasets")).expanduser()
    build_config = config.get("build_yfcc4k", {})

    yfcc_dirname = build_config.get("yfcc_dirname", "YFCC100M")
    downloads_dirname = build_config.get("downloads_dirname", "downloads")
    dataset_dirname = build_config.get("dataset_dirname", "yfcc4k")

    yfcc_root = data_root / yfcc_dirname
    return {
        "images_zip": yfcc_root / downloads_dirname / build_config.get("images_zip_name", "yfcc4k.zip"),
        "imagedata_txt": yfcc_root / downloads_dirname / build_config.get("imagedata_txt_name", "0aaaa.r.imagedata.txt"),
        "output_dir": yfcc_root / dataset_dirname,
    }


def download(url, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)


def extract_mediafire_direct_link(path, filename):
    text = path.read_text(encoding="utf-8", errors="ignore")
    pattern = rf"(https?://download.*?/{re.escape(filename)})"
    match = re.search(pattern, text, re.S)
    if match is None:
        return None
    return "".join(match.group(1).split())


def ensure_source_file(path, page_url, is_zip=False):
    if path.exists():
        return

    download(page_url, path)

    if is_zip and not zipfile.is_zipfile(path):
        direct_url = extract_mediafire_direct_link(path, path.name)
        if direct_url:
            download(direct_url, path)
    elif not is_zip:
        head = path.read_text(encoding="utf-8", errors="ignore")[:512].lower()
        if "<!doctype html" in head or "<html" in head:
            direct_url = extract_mediafire_direct_link(path, path.name)
            if direct_url:
                download(direct_url, path)


def iter_images(root):
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def write_jpg(src, dst):
    if src.suffix.lower() in {".jpg", ".jpeg"}:
        shutil.copy2(src, dst)
    else:
        Image.open(src).convert("RGB").save(dst, format="JPEG", quality=95)


def build_info_line(photo_id, lon, lat):
    cols = ["" for _ in range(14)]
    cols[1] = str(photo_id)
    cols[12] = str(float(lon))
    cols[13] = str(float(lat))
    return "\t".join(cols)


def parse_metadata_line(line):
    path_str, lat_str, lon_str = line.strip().split()[:3]
    photo_id = Path(path_str).stem
    return photo_id, float(lat_str), float(lon_str)


def main(args):
    config = load_config(args.config)
    build_config = config.get("build_yfcc4k", {})

    default_paths = build_default_paths(config)
    config_dir = Path(args.config).expanduser().resolve().parent if args.config else DEFAULT_CONFIG_PATH.parent

    images_zip = resolve_path(args.images_zip or str(default_paths["images_zip"]), config_dir, "build_yfcc4k.images_zip")
    imagedata_txt = resolve_path(args.imagedata_txt or str(default_paths["imagedata_txt"]), config_dir, "build_yfcc4k.imagedata_txt")
    output_dir = resolve_path(args.output_dir or str(default_paths["output_dir"]), config_dir, "build_yfcc4k.output_dir")

    images_dir = output_dir / "images"
    extract_dir = output_dir / "_tmp_extract"

    ensure_source_file(images_zip, IMAGES_PAGE_URL, is_zip=True)
    ensure_source_file(imagedata_txt, METADATA_PAGE_URL, is_zip=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    if extract_dir.exists() and args.clean_tmp:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(images_zip, "r") as zip_file:
        zip_file.extractall(extract_dir)

    image_by_stem = {image_path.stem: image_path for image_path in iter_images(extract_dir)}

    info_lines = []
    kept = 0

    with open(imagedata_txt, "r", encoding="utf-8", errors="ignore") as file:
        for line in tqdm(file, desc="Building yfcc4k"):
            photo_id, lat, lon = parse_metadata_line(line)
            source_image = image_by_stem[photo_id]

            destination_image = images_dir / f"{photo_id}.jpg"
            if args.overwrite or not destination_image.exists():
                write_jpg(source_image, destination_image)

            info_lines.append(build_info_line(photo_id, lon, lat))
            kept += 1

    info_path = output_dir / "info.txt"
    with open(info_path, "w", encoding="utf-8") as file:
        file.write("\n".join(info_lines) + "\n")

    if args.clean_tmp:
        shutil.rmtree(extract_dir, ignore_errors=True)

    print("Done.")
    print(f"Output dir: {output_dir}")
    print(f"Images written: {kept}")
    print(f"info.txt: {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build YFCC4k dataset in Baseline(yfcc4k) format.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--images_zip",
        type=str,
        default=None,
        help="Path to yfcc4k.zip.",
    )
    parser.add_argument(
        "--imagedata_txt",
        type=str,
        default=None,
        help="Path to 0aaaa.r.imagedata.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder compatible with Baseline(yfcc4k).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing jpg files.",
    )
    parser.add_argument(
        "--clean_tmp",
        action="store_true",
        help="Delete temporary extracted files once complete.",
    )

    main(parser.parse_args())
