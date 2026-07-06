"""TikZ/pgfplots teaser figure: source photo + attacked photos pinned on a world map.

Reproduces the spirit of the "Map: predicted locations under attack" cell at the end
of ``demo_notebooks/Show Image Examples.ipynb`` (true location, clean prediction, and
each attack's predicted location, connected by great-circle lines) as a standalone,
hand-editable pgfplots figure -- with the actual source/attacked image thumbnails
placed as real image nodes, pinned to their predicted location on the map. Meant for
a paper teaser: by default it only pins the two attacks a teaser cares about (DTD --
this paper's attack -- and GeoShield, the main baseline), not the full 8-attack sweep
the notebook's matplotlib map draws.

Unlike the rest of ``utils.plots_tikz``, this does not read ``results_dir`` .pt files:
its input is the per-image folder ``.media/images_examples/<image_id>/`` produced by
the notebook, which already caches each attack's predicted GPS in
``prediction_points.json`` alongside the actual clean/attacked PNGs.

The world map background is a raster (cartopy has no TikZ equivalent -- see
``utils.plots.maps``), generated once and cached, then placed at exact
(lon, lat) = (x, y) axis coordinates via pgfplots' ``\\addplot graphics``; every pin,
connector line, legend entry, and image thumbnail on top of it is real vector/TikZ
content.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from utils.plots.common import _display_attack_name
from utils.plots_tikz.common import SINGLE_MARK_LEGEND_IMAGE, attack_pgf_color, attack_marker, render_tikz, tex_escape

# "True"/"clean" aren't attacks (no canonical color/marker), so they keep
# their own ad hoc styling; every real attack now reuses attack_pgf_color()/
# attack_marker() from the shared tikz style module instead of an ad hoc
# lookup, so e.g. GeoShield is drawn with the exact same color+marker here as
# in every other figure in the paper.
_TRUE_STYLE = ("star", "trueGold", "True location")
_CLEAN_STYLE = ("*", "white", "Clean prediction")

_MAP_COLORS = {
    "ocean": "EAF1F7", "land": "EDEDED", "coastline": "888888", "border": "AAAAAA",
}
_EARTH_RADIUS_KM = 6371.0


def _to_xyz(lat_deg: float, lon_deg: float) -> np.ndarray:
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def _geodesic_km(p1, p2) -> float:
    v1, v2 = _to_xyz(*p1), _to_xyz(*p2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return _EARTH_RADIUS_KM * float(np.arccos(dot))


def _great_circle_lonlat(p1, p2, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """``n`` points along the shorter great-circle arc from p1=(lat,lon) to p2=(lat,lon)."""
    v1, v2 = _to_xyz(*p1), _to_xyz(*p2)
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-9:
        return np.array([p1[1], p2[1]]), np.array([p1[0], p2[0]])
    t = np.linspace(0.0, 1.0, n)
    sin_omega = np.sin(omega)
    pts = (np.sin((1 - t)[:, None] * omega) * v1 + np.sin(t[:, None] * omega) * v2) / sin_omega
    lat = np.degrees(np.arcsin(np.clip(pts[:, 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
    return lon, lat


def _split_at_dateline(lons: np.ndarray, lats: np.ndarray) -> list[tuple[list[float], list[float]]]:
    """Split a lon/lat polyline into segments wherever it jumps across +/-180."""
    segments = []
    seg_lon, seg_lat = [float(lons[0])], [float(lats[0])]
    for i in range(1, len(lons)):
        if abs(lons[i] - lons[i - 1]) > 180:
            segments.append((seg_lon, seg_lat))
            seg_lon, seg_lat = [], []
        seg_lon.append(float(lons[i]))
        seg_lat.append(float(lats[i]))
    segments.append((seg_lon, seg_lat))
    return segments


def _generate_world_map_background(out_path: Path, resolution: str = "110m") -> Path:
    """Render a plain ocean/land/coastline/border world map at exactly [-180,180]x[-90,90],
    no margins -- so it can be placed with pgfplots' ``\\addplot graphics`` at those exact
    axis coordinates. Cached: skipped if ``out_path`` already exists.
    """
    if out_path.exists():
        return out_path
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale(resolution), facecolor=f"#{_MAP_COLORS['ocean']}")
    ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor=f"#{_MAP_COLORS['land']}", edgecolor="none")
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), edgecolor=f"#{_MAP_COLORS['coastline']}", linewidth=0.5)
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), linestyle=":", edgecolor=f"#{_MAP_COLORS['border']}", linewidth=0.4)
    ax.axis("off")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def plot_geolocation_teaser_tikz(
    image_dir,
    plot_dir=None,
    attack_labels: Sequence[str] = ("dtd", "geoshield"),
    thumb_width_cm: float = 2.6,
    map_resolution: str = "110m",
) -> dict:
    """Standalone pgfplots teaser figure for one example image.

    ``image_dir`` is a per-image folder produced by "Show Image Examples.ipynb"
    (``.media/images_examples/<image_id>/``): must contain ``prediction_points.json``,
    ``clean.png``, and one PNG per requested attack in ``attack_labels``. Missing
    attack images are skipped with a printed warning (e.g. GeoShield generation failed
    for that image) rather than raising, so the figure still comes out for whichever
    attacks succeeded.

    Draws: the true GPS location (gold star), the unattacked model's prediction
    (small white circle), and -- for every attack in ``attack_labels`` that has
    both an image and a cached prediction -- a pin at its predicted location.
    Every prediction (clean and attacked alike) gets its own great-circle line
    back to the *true* location, not to the clean prediction -- what a teaser
    figure needs to show is how far off each prediction is from the truth, not
    how far an attack moved a prediction relative to the clean one. Each
    prediction's source/attacked photo is shown in a labeled row below the map
    (image identity + how far that prediction actually was from the truth),
    with no connecting line back to its pin -- the color-matched border and
    caption identify it instead, so the map itself stays readable.
    """
    image_dir = Path(image_dir)
    plot_dir = Path(plot_dir) if plot_dir is not None else image_dir
    with open(image_dir / "prediction_points.json") as f:
        rec = json.load(f)
    true_gps = tuple(rec["true"])
    clean_gps = tuple(rec["clean"])
    attacks_gps = rec.get("attacks", {})

    tikz_dir = plot_dir / "tikz"
    tikz_dir.mkdir(parents=True, exist_ok=True)
    stem = f"teaser_{image_dir.name}"

    # Background world map: generated once under a shared assets cache, then copied
    # alongside this figure's own .tex so the output folder stays self-contained.
    assets_dir = Path(__file__).resolve().parent.parent.parent / ".teaser_map_assets"
    master_bg = _generate_world_map_background(assets_dir / f"world_map_{map_resolution}.png", map_resolution)
    bg_name = f"{stem}_world_map.png"
    shutil.copyfile(master_bg, tikz_dir / bg_name)

    if not (image_dir / "clean.png").exists():
        raise FileNotFoundError(f"{image_dir / 'clean.png'} not found -- run the notebook's attack cells first.")
    shutil.copyfile(image_dir / "clean.png", tikz_dir / f"{stem}_source.png")

    # Resolve which attacks actually have both an image and a cached prediction.
    thumbs = [("source", true_gps, tikz_dir / f"{stem}_source.png", "Source")]
    pins = [("true", true_gps, *_TRUE_STYLE[:2], _TRUE_STYLE[2])]
    pins.append(("clean", clean_gps, *_CLEAN_STYLE[:2], _CLEAN_STYLE[2]))
    colors: dict[str, str] = {"trueGold": "FFD700"}

    for attack in attack_labels:
        img_path = image_dir / f"{attack}.png"
        gps = attacks_gps.get(attack)
        if gps is None or not img_path.exists():
            print(f"Skipping '{attack}' in the teaser for {image_dir.name}: "
                  f"missing {'prediction' if gps is None else img_path.name}.")
            continue
        color_name, hexcode = attack_pgf_color(attack)
        colors[color_name] = hexcode
        marker = attack_marker(attack)
        display_name = _display_attack_name(attack)
        label = f"{display_name} (ours)" if attack.lower() == "dtd" else display_name
        thumb_name = f"{stem}_{attack}.png"
        shutil.copyfile(img_path, tikz_dir / thumb_name)
        thumbs.append((attack, tuple(gps), tikz_dir / thumb_name, display_name))
        pins.append((attack, tuple(gps), marker, color_name, label))

    # --- pins (drawn as normal pgfplots addplots so the legend comes for free) ---
    pin_blocks = []
    for key, (lat, lon), marker, color_name, label in pins:
        size = "7pt" if key == "true" else ("6pt" if key == "dtd" else "4.5pt")
        extra = ", mark options={fill=white, draw=black, line width=0.9pt}" if key == "clean" else ", mark options={line width=1.1pt}"
        pin_blocks.append(
            f"\\addplot[only marks, mark={marker}, {color_name}, mark size={size}{extra}] "
            f"coordinates {{({lon:.4f},{lat:.4f})}};\n\\addlegendentry{{{tex_escape(label)}}}"
        )

    # --- great-circle connectors, true location -> every prediction (clean and
    #     each attack alike) -- a teaser needs to show how far off a prediction
    #     landed from the truth, not how far an attack moved it relative to the
    #     unattacked prediction, so every line originates at "true", not "clean". ---
    connector_blocks = []
    for key, gps, marker, color_name, _label in pins:
        if key == "true":
            continue
        lon_lat_segments = _split_at_dateline(*_great_circle_lonlat(true_gps, gps))
        width = "1.3pt" if key == "dtd" else "0.8pt"
        style = "dashed" if key == "clean" else "solid"
        for seg_lon, seg_lat in lon_lat_segments:
            if len(seg_lon) < 2:
                continue
            pts = " ".join(f"({lo:.3f},{la:.3f})" for lo, la in zip(seg_lon, seg_lat))
            connector_blocks.append(
                f"\\addplot[{color_name}, {style}, line width={width}, opacity=0.65, mark=none, forget plot] "
                f"coordinates {{{pts}}};"
            )

    # --- image thumbnails in a row below the map -- a color-matched border and a
    #     caption (name + great-circle distance from the truth) identify each one;
    #     no leader line back to its pin, which would visually tie a *predicted
    #     point* to its *source image* rather than showing prediction error. ---
    n = len(thumbs)
    thumb_xs = np.linspace(-150, 150, n) if n > 1 else np.array([0.0])
    thumb_y = -108.0
    thumb_blocks = []
    for (key, (lat, lon), img_path, label), tx in zip(thumbs, thumb_xs):
        color_name = {"source": "trueGold"}.get(key) or next(
            (c for k, _, m, c, _ in pins if k == key), "black"
        )
        rel_path = os.path.relpath(img_path, tikz_dir)
        if key == "source":
            caption = tex_escape(label)
        else:
            dist_km = _geodesic_km(true_gps, (lat, lon))
            caption = f"{tex_escape(label)} \\\\ {dist_km:,.0f} km off"
        thumb_blocks.append(
            f"\\node[anchor=north, inner sep=0pt, draw={color_name}, line width=1.1pt] at "
            f"(axis cs:{tx:.2f},{thumb_y:.2f}) "
            f"{{\\includegraphics[width={thumb_width_cm}cm]{{{rel_path}}}}};\n"
            f"\\node[anchor=north, align=center, font=\\small, yshift=-{thumb_width_cm}cm-8pt] at "
            f"(axis cs:{tx:.2f},{thumb_y:.2f}) {{{caption}}};"
        )

    # Put the legend in whichever map corner sits farthest (in plain lon/lat degrees --
    # not geodesic, just a cheap heuristic) from every pin, so it doesn't happen to land
    # on top of a marker for this particular image (attacks can relocate a prediction
    # anywhere on the globe, including right under a fixed corner).
    corner_candidates = [
        ("north west", -175.0, 85.0), ("north east", 175.0, 85.0),
        ("south west", -175.0, -85.0), ("south east", 175.0, -85.0),
    ]
    pin_lonlat = [(lon, lat) for _key, (lat, lon), *_rest in pins]
    def _min_dist(clon, clat):
        return min(((clon - lo) ** 2 + (clat - la) ** 2) ** 0.5 for lo, la in pin_lonlat)
    legend_anchor, legend_lon, legend_lat = max(corner_candidates, key=lambda c: _min_dist(c[1], c[2]))

    axis_opts = ", ".join([
        "axis equal image", "width=14cm",
        "xmin=-180", "xmax=180", "ymin=-128", "ymax=98",
        "hide axis",
        f"legend style={{at={{(axis cs:{legend_lon:.1f},{legend_lat:.1f})}}, anchor={legend_anchor}, "
        "draw=gray, fill=white, fill opacity=0.88, text opacity=1, font=\\footnotesize, "
        "legend columns=1}",
        SINGLE_MARK_LEGEND_IMAGE,
        "clip=false",
    ])
    bg_rel = os.path.relpath(tikz_dir / bg_name, tikz_dir)
    body = (
        f"\\begin{{axis}}[{axis_opts}]\n"
        f"\\addplot[forget plot] graphics[xmin=-180,ymin=-90,xmax=180,ymax=90] {{{bg_rel}}};\n"
        + "\n".join(connector_blocks) + "\n"
        + "\n".join(pin_blocks) + "\n"
        + "\n".join(thumb_blocks) + "\n"
        "\\end{axis}"
    )
    from utils.plots_tikz.common import color_definitions
    return render_tikz(body, str(plot_dir), stem, color_definitions(colors))
