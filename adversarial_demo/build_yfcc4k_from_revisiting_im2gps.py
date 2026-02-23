import argparse
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image
from tqdm import tqdm


#urls retrieved from https://github.com/lugiavn/revisiting-im2gps?tab=readme-ov-file
IMAGES_PAGE_URL = "http://www.mediafire.com/file/3og8y3o6c9de3ye/yfcc4k.zip"
METADATA_PAGE_URL = "http://www.mediafire.com/file/8v2j565997i5jed/0aaaa.r.imagedata.txt"


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
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    extract_dir = output_dir / "_tmp_extract"

    images_zip = Path(args.images_zip)
    imagedata_txt = Path(args.imagedata_txt)

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
        "--images_zip",
        type=str,
        default="datasets/YFCC100M/downloads/yfcc4k.zip",
        help="Path to yfcc4k.zip.",
    )
    parser.add_argument(
        "--imagedata_txt",
        type=str,
        default="datasets/YFCC100M/downloads/0aaaa.r.imagedata.txt",
        help="Path to 0aaaa.r.imagedata.txt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="datasets/YFCC100M/yfcc4k",
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
