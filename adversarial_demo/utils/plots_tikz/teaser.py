"""TikZ/pgfplots teaser figure: perturbation diagram + every attack's prediction on a world map.

Reproduces the spirit of the "Map: predicted locations under attack" cell at the end
of ``demo_notebooks/Show Image Examples.ipynb`` (true location, clean prediction, and
each attack's predicted location, connected by great-circle lines) as a standalone,
hand-editable pgfplots figure. Left of the map, a compact diagram shows the source
photo, the GeoFA-TD perturbation being added, and the protected photo; a zoom inset
resolves the predictions that stay near the true location.

Unlike the rest of ``utils.plots_tikz``, this does not read ``results_dir`` .pt files:
its input is the per-image folder ``.media/images_examples/<image_id>/`` produced by
the notebook, which already caches each attack's predicted GPS in
``prediction_points.json`` alongside the actual clean/attacked PNGs.

The world map background is a raster (cartopy has no TikZ equivalent -- see
``utils.plots.maps``), generated once and cached, then placed at exact
(lon, lat) = (x, y) axis coordinates via pgfplots' ``\\addplot graphics``; every pin,
connector line, legend entry, and diagram node on top of it is real vector/TikZ
content.
"""

from __future__ import annotations

import hashlib
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
_TRUE_STYLE = ("filledstar", "trueGold", "True location")
_CLEAN_STYLE = ("o", "black", "Clean prediction")

_MAP_COLORS = {
    "ocean": "D9E7F1", "land": "F6F3EC", "coastline": "9DB2C2", "border": "FFFFFF",
}
# At zoom level white borders on cream land disappear; the inset draws them darker.
_ZOOM_BORDER = "C4BBA8"
# Antarctica / the high Arctic are empty for every image we show; cropping them gives the
# populated latitudes more room. A pin outside this band falls back to the full globe.
_MAP_LAT_RANGE = (-58.0, 84.0)
_EARTH_RADIUS_KM = 6371.0
_PIXEL_UPSCALE = 4  # 336 px model-input squares -> 1344 px, nearest-neighbour

# Filled five-pointed star (pgfplots' own "star" mark is line-only and vanishes on the map).
_EXTRA_PREAMBLE = (
    "\\usepackage{times}\n"  # ICLR body font (math stays Computer Modern, as in the template)
    "\\usetikzlibrary{shapes.geometric, calc, arrows.meta}\n"
    "\\pgfdeclareplotmark{filledstar}{\\node[star, star points=5, star point ratio=2.4, "
    "inner sep=0pt, minimum size=2.6\\pgfplotmarksize, draw=black, line width=0.5pt, "
    "fill=trueGold] {};}"
)


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


def _pin_plot(key: str, lat: float, lon: float, marker: str, color_name: str, scale: float = 1.0,
              legend: bool = True) -> str:
    size = {"true": 4.2, "clean": 4.6, "dtd": 6.0}.get(key, 3.6) * scale
    extra = {
        "true": "",
        "clean": ", mark options={line width=1.1pt}",
    }.get(key, ", mark options={line width=1.5pt}")
    halo = ""
    if key not in ("true", "clean"):
        halo = (f"\\addplot[only marks, mark={marker}, white, mark size={size:.2f}pt, "
                f"mark options={{draw=white, fill=white, line width=3.2pt}}, forget plot] "
                f"coordinates {{({lon:.4f},{lat:.4f})}};\n")
    return (halo + f"\\addplot[only marks, mark={marker}, {color_name}, mark size={size:.2f}pt{extra}"
            f"{'' if legend else ', forget plot'}] coordinates {{({lon:.4f},{lat:.4f})}};")


def _connector_plots(pts: str, key: str, color_name: str, scale: float = 1.0) -> str:
    width = {"dtd": 1.6, "clean": 1.0}.get(key, 0.75) * scale
    opacity = 1.0 if key in ("dtd", "clean") else 0.8
    style = "dashed" if key == "clean" else "solid"
    line_color = "black!70" if key == "clean" else color_name
    return (f"\\addplot[white, line width={width + 1.2:.2f}pt, opacity={0.85 * opacity:.2f}, mark=none, forget plot, "
            f"line cap=round] coordinates {{{pts}}};\n"
            f"\\addplot[{line_color}, {style}, line width={width:.2f}pt, opacity={opacity:.2f}, mark=none, forget plot, "
            f"line cap=round] coordinates {{{pts}}};")


def _save_delta_view(clean_png: Path, attacked_png: Path, out_png: Path, pct: float = 99.5,
                     budget_levels: Optional[float] = None) -> float:
    """delta = attacked - clean, centred on mid-grey and amplified. With ``budget_levels``
    (the L-inf budget in grey levels) the gain maps +-budget to full contrast, so deltas of
    different methods are shown on the same scale; otherwise the gain comes from a high
    percentile of |delta|. Returns the gain."""
    from PIL import Image
    c = np.asarray(Image.open(clean_png).convert("RGB"), dtype=float)
    a = np.asarray(Image.open(attacked_png).convert("RGB"), dtype=float)
    diff = a - c
    ref = budget_levels if budget_levels else float(np.percentile(np.abs(diff), pct))
    gain = 127.5 / max(1.0, ref)
    view = Image.fromarray(np.clip(127.5 + diff * gain, 0, 255).astype("uint8"))
    view.resize((view.width * _PIXEL_UPSCALE, view.height * _PIXEL_UPSCALE), Image.NEAREST).save(out_png)
    return gain


def _save_pixel_exact(src_png: Path, out_png: Path) -> None:
    """Nearest-neighbour upscale: identical pixel data, but PDF viewers (most ignore the
    no-interpolation flag) show crisp pixels instead of a blur when the reader zooms in."""
    from PIL import Image
    im = Image.open(src_png).convert("RGB")
    im.resize((im.width * _PIXEL_UPSCALE, im.height * _PIXEL_UPSCALE), Image.NEAREST).save(out_png)


def _world_map_cache_name(resolution: str, lat_range, lon_range=(-180.0, 180.0), *extra) -> str:
    style = hashlib.md5(json.dumps([_MAP_COLORS, resolution, list(lat_range), list(lon_range), *extra]).encode()).hexdigest()[:8]
    return f"world_map_{resolution}_{style}.pdf"


def _zoom_window(true_gps, points, cluster_km: float = 2500.0, min_lon_span: float = 18.0,
                 aspect: float = 1.4) -> Optional[tuple[float, float, float, float]]:
    """(lon0, lon1, lat0, lat1) framing the true location and every prediction within
    ``cluster_km`` of it -- the cluster that is unreadable at world scale. None when no
    prediction is that close (nothing to zoom on)."""
    near = [p for p in points if _geodesic_km(true_gps, p) < cluster_km]
    if not near:
        return None
    lats = [true_gps[0]] + [p[0] for p in near]
    lons = [true_gps[1]] + [p[1] for p in near]
    if max(lons) - min(lons) > 180:
        return None
    lon_c, lat_c = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    lon_span = max(min_lon_span, 1.7 * (max(lons) - min(lons)), 1.7 * aspect * (max(lats) - min(lats)))
    lat_span = lon_span / aspect
    return lon_c - lon_span / 2, lon_c + lon_span / 2, lat_c - lat_span / 2, lat_c + lat_span / 2


def _place_inset(map_png: Path, lat_range, zoom, box_w: float, box_h: float, avoid_pts) -> tuple[float, float]:
    """South-west corner (lon, lat) for a box_w x box_h (degrees) inset over the map: mostly
    ocean, covering no pin or connector, not overlapping the zoom window, and as close to it
    as possible (short leader lines)."""
    from PIL import Image
    lat_min, lat_max = lat_range
    nlat = int(round(lat_max - lat_min))
    rgb = np.asarray(Image.open(map_png).convert("RGB").resize((360, nlat), Image.BILINEAR), dtype=float)
    land_rgb = np.array([int(_MAP_COLORS["land"][i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
    ocean_rgb = np.array([int(_MAP_COLORS["ocean"][i:i + 2], 16) for i in (0, 2, 4)], dtype=float)
    land = (np.linalg.norm(rgb - land_rgb, axis=2) < np.linalg.norm(rgb - ocean_rgb, axis=2))[::-1]  # row 0 = lat_min
    integral = np.pad(land.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    pts = np.asarray(avoid_pts, dtype=float).reshape(-1, 2)  # (lon, lat)
    z0, z1, zl0, zl1 = zoom
    zc = np.array([(z0 + z1) / 2, (zl0 + zl1) / 2])
    best, best_score = None, np.inf
    margin = 2.0
    for lon0 in np.arange(-180 + margin, 180 - margin - box_w + 1e-9, 2.0):
        for lat0 in np.arange(lat_min + margin, lat_max - margin - box_h + 1e-9, 2.0):
            lon1, lat1 = lon0 + box_w, lat0 + box_h
            if lon0 < z1 + 4 and lon1 > z0 - 4 and lat0 < zl1 + 4 and lat1 > zl0 - 4:
                continue
            c0, c1 = int(lon0 + 180), int(lon1 + 180)
            r0, r1 = int(lat0 - lat_min), int(lat1 - lat_min)
            land_frac = (integral[r1, c1] - integral[r0, c1] - integral[r1, c0] + integral[r0, c0]) / max(1, (r1 - r0) * (c1 - c0))
            inside = np.sum((pts[:, 0] > lon0 - 3) & (pts[:, 0] < lon1 + 3) & (pts[:, 1] > lat0 - 3) & (pts[:, 1] < lat1 + 3))
            dist = np.linalg.norm(np.array([lon0 + box_w / 2, lat0 + box_h / 2]) - zc) / 360.0
            score = 4.0 * land_frac + 100.0 * inside + dist
            if score < best_score:
                best, best_score = (float(lon0), float(lat0)), score
    return best


def _leader_corners(zoom, inset) -> list[tuple[str, str]]:
    """Which zoom-window corner connects to which inset corner (the two outer lines of the
    classic zoom trapezoid), picked from the side the inset sits on."""
    z0, z1, zl0, zl1 = zoom
    i0, i1, il0, il1 = inset
    gaps = {"below": zl0 - il1, "above": il0 - zl1, "left": z0 - i1, "right": i0 - z1}
    side = max(gaps, key=gaps.get)
    return {
        "below": [("SW", "NW"), ("SE", "NE")], "above": [("NW", "SW"), ("NE", "SE")],
        "left": [("NW", "NE"), ("SW", "SE")], "right": [("NE", "NW"), ("SE", "SW")],
    }[side]


def _generate_world_map_background(out_path: Path, resolution: str = "50m",
                                   lat_range: tuple[float, float] = (-90.0, 90.0),
                                   lon_range: tuple[float, float] = (-180.0, 180.0),
                                   width_in: float = 12.0, border: Optional[str] = None,
                                   border_lw: float = 0.45) -> Path:
    """Render a plain ocean/land/coastline/border map of exactly lon_range x lat_range, no
    margins -- so it can be placed with pgfplots' ``\\addplot graphics`` at those exact axis
    coordinates. Written as a vector PDF (sharp at any zoom) plus a small PNG twin used only
    to locate land when placing the inset. Cached: skipped if both already exist (the cache
    name encodes the style, resolution and extent, so restyling never reuses a stale map).
    """
    if out_path.exists() and out_path.with_suffix(".png").exists():
        return out_path
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    fig = plt.figure(figsize=(width_in, width_in * (lat_max - lat_min) / (lon_max - lon_min)), dpi=400)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.set_facecolor(f"#{_MAP_COLORS['ocean']}")
    ax.add_feature(cfeature.LAND.with_scale(resolution), facecolor=f"#{_MAP_COLORS['land']}", edgecolor="none")
    ax.add_feature(cfeature.LAKES.with_scale(resolution), facecolor=f"#{_MAP_COLORS['ocean']}", edgecolor="none")
    ax.add_feature(cfeature.BORDERS.with_scale(resolution), edgecolor=f"#{border or _MAP_COLORS['border']}", linewidth=border_lw)
    ax.add_feature(cfeature.COASTLINE.with_scale(resolution), edgecolor=f"#{_MAP_COLORS['coastline']}", linewidth=0.3)
    ax.spines["geo"].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, pad_inches=0)
    fig.savefig(out_path.with_suffix(".png"), dpi=100, pad_inches=0)
    plt.close(fig)
    return out_path


_DEFAULT_ATTACKS = ("dtd", "diffusion_l2", "targeted_l2", "sampling", "encoder",
                    "training_loss", "unidef", "geoshield")
# Near-full-globe window: two stacked photos next to a map need a tall map to be legible.
_PANEL_LAT_WINDOW = (-80.0, 90.0)
ICLR_TEXTWIDTH_CM = 13.97  # 5.5 in
# Drawable width: the standalone class adds a 2pt border on each side, and pgfplots lines stick
# out half their width, so the finished PDF comes out at (just under) \textwidth.
_CONTENT_WIDTH_CM = ICLR_TEXTWIDTH_CM - 0.2


def _load_panel(image_dir: Path, attack_labels: Sequence[str]):
    with open(image_dir / "prediction_points.json") as f:
        rec = json.load(f)
    for name in ("clean.png", "dtd.png"):
        if not (image_dir / name).exists():
            raise FileNotFoundError(f"{image_dir / name} not found -- run the notebook's attack cells first.")
    true_gps, clean_gps = tuple(rec["true"]), tuple(rec["clean"])
    attacks_gps = rec.get("attacks", {})
    pins = [("true", true_gps, *_TRUE_STYLE[:2], _TRUE_STYLE[2]),
            ("clean", clean_gps, *_CLEAN_STYLE[:2], _CLEAN_STYLE[2])]
    colors = {"trueGold": "FFD700"}
    for attack in attack_labels:
        gps = attacks_gps.get(attack)
        if gps is None or not (image_dir / f"{attack}.png").exists():
            print(f"Skipping '{attack}' for {image_dir}: missing prediction or image.")
            continue
        color_name, hexcode = attack_pgf_color(attack)
        colors[color_name] = hexcode
        name = _display_attack_name(attack)
        pins.append((attack, tuple(gps), attack_marker(attack), color_name,
                     f"{name} (ours)" if attack == "dtd" else name))
    needed = [true_gps[0]]
    for _k, g, *_ in pins[1:]:
        needed += list(_great_circle_lonlat(true_gps, g)[1])
    needed_range = (max(-90.0, float(np.floor(min(needed))) - 3), min(90.0, float(np.ceil(max(needed))) + 3))
    return rec, pins, colors, needed_range


def _fit_lat_range(needed, span):
    """A latitude window of ``span`` degrees (>= the needed range) as close as possible to
    _PANEL_LAT_WINDOW, containing ``needed``, clamped to [-90, 90]."""
    span = max(span, needed[1] - needed[0])
    lo = min(max(_PANEL_LAT_WINDOW[0], needed[1] - span), needed[0])
    lo = min(max(lo, -90.0), 90.0 - span)
    return lo, lo + span


def _teaser_panel(image_dir: Path, rec, pins, lat_range, tag: str, tikz_dir: Path, stem: str,
                  x_off: float, map_w: float, img: float, col_gap: float, map_resolution: str,
                  zoom_inset: bool, inset_width_cm: float, pin_scale: float, legend_opts: Optional[str]) -> str:
    """TikZ for one panel: clean photo above the GeoFA-TD photo (joined by a "+delta" arrow),
    each tagged with its distance to the truth, next to a world map with every attack's
    prediction. Lengths in cm; the panel's south-west corner is at (x_off, 0)."""
    true_gps, clean_gps = tuple(rec["true"]), tuple(rec["clean"])
    attacks_gps = rec["attacks"]
    lat_min, lat_max = lat_range
    map_h = map_w * (lat_max - lat_min) / 360.0
    deg_per_cm = 360.0 / map_w
    assets_dir = Path(__file__).resolve().parent.parent.parent / ".teaser_map_assets"
    master_bg = _generate_world_map_background(
        assets_dir / _world_map_cache_name(map_resolution, lat_range), map_resolution, lat_range)
    bg_name = f"{stem}_{tag}_world_map.pdf"
    shutil.copyfile(master_bg, tikz_dir / bg_name)
    _save_pixel_exact(image_dir / "clean.png", tikz_dir / f"{stem}_{tag}_clean.png")
    _save_pixel_exact(image_dir / "dtd.png", tikz_dir / f"{stem}_{tag}_dtd.png")

    pin_blocks = []
    for key, (lat, lon), marker, color_name, label in pins:
        plot = _pin_plot(key, lat, lon, marker, color_name, scale=pin_scale, legend=legend_opts is not None)
        pin_blocks.append(plot + (f"\n\\addlegendentry{{{tex_escape(label)}}}" if legend_opts is not None else ""))
    pin_blocks += [_pin_plot(k, g[0], g[1], m, c, scale=pin_scale, legend=False)
                   for k, g, m, c, _l in pins if k in ("dtd", "true")]

    connector_blocks, connector_pts = [], []
    for key, gps, _m, color_name, _l in sorted(pins, key=lambda p: p[0] == "dtd"):
        if key == "true":
            continue
        for seg_lon, seg_lat in _split_at_dateline(*_great_circle_lonlat(true_gps, gps)):
            if len(seg_lon) < 2:
                continue
            connector_pts += list(zip(seg_lon, seg_lat))
            pts = " ".join(f"({lo:.3f},{la:.3f})" for lo, la in zip(seg_lon, seg_lat))
            connector_blocks.append(_connector_plots(pts, key, color_name, scale=pin_scale))

    inset_main, inset_after = "", ""
    zoom = _zoom_window(true_gps, [g for k, g, *_ in pins if k != "true"]) if zoom_inset else None
    sw = None
    if zoom is not None:
        z0, z1, zl0, zl1 = zoom
        box_w = inset_width_cm * deg_per_cm
        box_h = box_w * (zl1 - zl0) / (z1 - z0)
        avoid = [(g[1], g[0]) for _k, g, *_ in pins] + connector_pts
        sw = _place_inset(master_bg.with_suffix(".png"), lat_range, zoom, box_w, box_h, avoid)
    if zoom is not None and sw is not None:
        i0, il0 = sw
        i1, il1 = i0 + box_w, il0 + box_h
        zoom_bg = _generate_world_map_background(
            assets_dir / _world_map_cache_name(map_resolution, (zl0, zl1), (z0, z1), _ZOOM_BORDER),
            map_resolution, (zl0, zl1), (z0, z1), width_in=5.0, border=_ZOOM_BORDER, border_lw=0.8)
        zoom_bg_name = f"{stem}_{tag}_zoom_map.pdf"
        shutil.copyfile(zoom_bg, tikz_dir / zoom_bg_name)
        corner = {"SW": lambda a, b, c, d: (a, c), "SE": lambda a, b, c, d: (b, c),
                  "NW": lambda a, b, c, d: (a, d), "NE": lambda a, b, c, d: (b, d)}
        leader = []
        for zc, ic in _leader_corners(zoom, (i0, i1, il0, il1)):
            zx, zy = corner[zc](z0, z1, zl0, zl1)
            ix, iy = corner[ic](i0, i1, il0, il1)
            leader.append(f"\\draw[black!55, line width=0.4pt] (axis cs:{zx:.2f},{zy:.2f}) -- (axis cs:{ix:.2f},{iy:.2f});")
        inset_main = (
            f"\\draw[black!70, line width=0.5pt] (axis cs:{z0:.2f},{zl0:.2f}) rectangle (axis cs:{z1:.2f},{zl1:.2f});\n"
            + "\n".join(leader) + "\n"
            f"\\coordinate (insetSW{tag}) at (axis cs:{i0:.2f},{il0:.2f});"
        )
        inset_conn = []
        for key, gps, _m, color_name, _l in sorted(pins, key=lambda p: p[0] == "dtd"):
            if key == "true":
                continue
            lons, lats = _great_circle_lonlat(true_gps, gps, n=400)
            for seg_lon, seg_lat in _split_at_dateline(lons, lats):
                if len(seg_lon) >= 2:
                    pts = " ".join(f"({lo:.3f},{la:.3f})" for lo, la in zip(seg_lon, seg_lat))
                    inset_conn.append(_connector_plots(pts, key, color_name, scale=pin_scale))
        inset_pins = [_pin_plot(k, g[0], g[1], m, c, scale=pin_scale * 1.1, legend=False)
                      for k, g, m, c, _l in sorted(pins, key=lambda p: (p[0] == "dtd") + 2 * (p[0] == "true"))]
        width_km = (z1 - z0) * 111.32 * np.cos(np.radians((zl0 + zl1) / 2))
        bar_km = max([b for b in (100.0, 200.0, 500.0, 1000.0, 2000.0) if b <= 0.35 * width_km] or [100.0])
        bar_lat = zl0 + 0.1 * (zl1 - zl0)
        bar_x1 = z1 - 0.08 * (z1 - z0)
        bar_x0 = bar_x1 - bar_km / (111.32 * np.cos(np.radians(bar_lat)))
        inset_opts = ", ".join([
            f"name=inset{tag}", f"at={{(insetSW{tag})}}", "anchor=south west", "scale only axis",
            f"width={inset_width_cm:.3f}cm", f"height={box_h / deg_per_cm:.3f}cm",
            f"xmin={z0:.4f}", f"xmax={z1:.4f}", f"ymin={zl0:.4f}", f"ymax={zl1:.4f}",
            "xtick=\\empty", "ytick=\\empty", "axis line style={black!70, line width=0.5pt}",
            "axis on top", "clip=true",
        ])
        inset_after = (
            f"\n\\begin{{axis}}[{inset_opts}]\n"
            f"\\addplot[forget plot] graphics[xmin={z0:.4f},ymin={zl0:.4f},xmax={z1:.4f},ymax={zl1:.4f}] {{{zoom_bg_name}}};\n"
            + "\n".join(inset_conn) + "\n" + "\n".join(inset_pins) + "\n"
            f"\\draw[black!75, line width=0.6pt] (axis cs:{bar_x0:.3f},{bar_lat:.3f}) -- (axis cs:{bar_x1:.3f},{bar_lat:.3f});\n"
            f"\\node[anchor=south, font=\\tiny, inner sep=1pt, text=black!75] "
            f"at (axis cs:{(bar_x0 + bar_x1) / 2:.3f},{bar_lat:.3f}) {{\\scalebox{{0.8}}{{{bar_km:.0f} km}}}};\n"
            "\\end{axis}"
        )

    dtd_color = attack_pgf_color("dtd")[0]
    budget = rec.get("budget_over_255")
    xc = x_off + img / 2

    def _photo(name, file, y, frame, tag_text, tag_color, dist):
        band = f"{dist:,.0f} km off"
        band_scale = min(0.85, (img - 0.1) / (0.092 * len(band)))
        tag_scale = min(0.85, (img - 0.1) / (0.092 * len(tag_text)))
        return (
            f"\\node[inner sep=0pt, draw={frame}, line width=1.1pt] ({name}) at ({xc:.3f}cm,{y:.3f}cm) "
            f"{{\\includegraphics[width={img:.3f}cm]{{{file}}}}};\n"
            f"\\node[anchor=north west, inner sep=1pt, fill=white, fill opacity=0.85, text opacity=1, "
            f"text={tag_color}, font=\\tiny\\bfseries] at ({name}.north west) {{\\scalebox{{{tag_scale:.2f}}}{{{tag_text}}}}};\n"
            f"\\node[anchor=south, inner sep=1pt, minimum width={img:.3f}cm, fill=black, fill opacity=0.6, "
            f"text=white, text opacity=1, font=\\tiny\\bfseries] at ({name}.south) "
            f"{{\\scalebox{{{band_scale:.2f}}}{{{band}}}}};"
        )

    photos = "\n".join([
        _photo(f"clean{tag}", f"{stem}_{tag}_clean.png", map_h - img / 2, "black", "clean", "black",
               _geodesic_km(true_gps, clean_gps)),
        _photo(f"adv{tag}", f"{stem}_{tag}_dtd.png", img / 2, dtd_color, "GeoFA-TD", f"{dtd_color}!65!black",
               _geodesic_km(true_gps, tuple(attacks_gps["dtd"]))),
        f"\\draw[-{{Stealth[length=3.5pt]}}, line width=0.7pt, {dtd_color}] (clean{tag}.south) -- "
        f"node[right, font=\\tiny, inner sep=1pt, text=black] {{$+\\delta$}} "
        + (f"node[left, font=\\tiny, inner sep=1pt, text=black!70] {{\\scalebox{{0.8}}{{{budget}/255}}}} " if budget else "")
        + f"(adv{tag}.north);",
    ])

    axis_opts = [
        f"name=map{tag}", f"at={{({x_off + img + col_gap:.3f}cm,0cm)}}", "anchor=south west",
        "axis equal image", f"width={map_w:.3f}cm", "scale only axis",
        "xmin=-180", "xmax=180", f"ymin={lat_min:g}", f"ymax={lat_max:g}", "hide axis", "clip=false",
    ]
    if legend_opts is not None:
        axis_opts += [legend_opts, SINGLE_MARK_LEGEND_IMAGE]
    return (
        f"\\begin{{axis}}[{', '.join(axis_opts)}]\n"
        f"\\addplot[forget plot] graphics[xmin=-180,ymin={lat_min:g},xmax=180,ymax={lat_max:g}] {{{bg_name}}};\n"
        f"\\draw[black!35, line width=0.4pt] (axis cs:-180,{lat_min:g}) rectangle (axis cs:180,{lat_max:g});\n"
        + "\n".join(connector_blocks) + "\n" + "\n".join(pin_blocks) + "\n" + inset_main + "\n"
        "\\end{axis}" + inset_after + "\n" + photos
    )


def _legend_opts(n_entries: int, center_lon: float, lat_max: float, columns: Optional[int] = None) -> str:
    cols = columns or (-1 if n_entries <= 6 else (n_entries + 1) // 2)
    return (f"legend style={{at={{(axis cs:{center_lon:.2f},{lat_max:g})}}, anchor=south, yshift=2pt, "
            "draw=none, fill=none, font=\\scriptsize, legend cell align=left, inner sep=1pt, "
            f"legend columns={cols}, /tikz/every even column/.append style={{column sep=6pt}}}}")


def _render(body: str, out_dir: Path, stem: str, colors: dict) -> dict:
    from utils.plots_tikz.common import color_definitions
    result = render_tikz(body, str(out_dir), stem, color_definitions(colors) + "\n" + _EXTRA_PREAMBLE)
    result["png_600dpi"] = None
    if result.get("pdf") and shutil.which("pdftoppm"):
        import subprocess
        hi = out_dir / "tikz" / f"{stem}_600dpi"
        subprocess.run(["pdftoppm", "-r", "600", "-png", "-singlefile", result["pdf"], str(hi)], check=False)
        if hi.with_suffix(".png").exists():
            result["png_600dpi"] = str(hi.with_suffix(".png"))
    return result


def _panel_geometry(needed_ranges, panel_w: float, col_gap: float, gap: float):
    """Map width, photo size and shared latitude span such that two stacked photos exactly
    match the map height and photo + gap + map fill ``panel_w``."""
    span = max([_PANEL_LAT_WINDOW[1] - _PANEL_LAT_WINDOW[0]] + [n[1] - n[0] for n in needed_ranges])
    r = span / 360.0
    # panel_w = img + col_gap + map_w, with img = (map_w * r - gap) / 2
    map_w = (panel_w - col_gap + gap / 2) / (1 + r / 2)
    img = (map_w * r - gap) / 2
    return map_w, img, span


def plot_geolocation_teaser_tikz(
    image_dir,
    plot_dir=None,
    attack_labels: Sequence[str] = _DEFAULT_ATTACKS,
    map_resolution: str = "50m",
    stem: Optional[str] = None,
    width_cm: float = _CONTENT_WIDTH_CM,
    zoom_inset: bool = True,
) -> dict:
    """Single-panel teaser at ``width_cm`` (default: ICLR text width), legend on top.

    ``image_dir`` is a per-image folder of "Show Image Examples.ipynb" (prediction_points.json,
    clean.png, dtd.png and one PNG per attack). The map shows the true location, the clean
    prediction and every attack's prediction, each linked to the truth by a great-circle arc;
    left of it, the clean photo above the GeoFA-TD photo, each tagged with its distance to
    the truth."""
    image_dir = Path(image_dir)
    out_dir = Path(plot_dir) if plot_dir is not None else image_dir
    (out_dir / "tikz").mkdir(parents=True, exist_ok=True)
    stem = stem or f"teaser_{image_dir.name}"
    rec, pins, colors, needed = _load_panel(image_dir, attack_labels)
    col_gap, gap = 0.2, 0.45
    map_w, img, span = _panel_geometry([needed], width_cm, col_gap, gap)
    lat_range = _fit_lat_range(needed, span)
    center_lon = -180 + (width_cm / 2 - img - col_gap) * 360.0 / map_w
    body = _teaser_panel(image_dir, rec, pins, lat_range, "a", out_dir / "tikz", stem, 0.0, map_w, img, col_gap,
                         map_resolution, zoom_inset, 0.3 * map_w, map_w / 10.0,
                         _legend_opts(len(pins), center_lon, lat_range[1]))
    return _render(body, out_dir, stem, colors)


def plot_geolocation_teaser_pair_tikz(
    image_dirs,
    out_dir,
    stem: str = "teaser_pair",
    attack_labels: Sequence[str] = _DEFAULT_ATTACKS,
    map_resolution: str = "50m",
    width_cm: float = _CONTENT_WIDTH_CM,
    panel_sep_cm: float = 0.35,
    zoom_inset: bool = True,
) -> dict:
    """Two teaser panels side by side at ``width_cm`` (default: ICLR text width, so the PDF
    goes in with ``\\includegraphics[width=\\textwidth]``) with one shared legend on top.
    Both maps share the same latitude window height, so the panels align."""
    image_dirs = [Path(d) for d in image_dirs]
    assert len(image_dirs) == 2, "a pair needs exactly two image folders"
    out_dir = Path(out_dir)
    (out_dir / "tikz").mkdir(parents=True, exist_ok=True)
    loaded = [_load_panel(d, attack_labels) for d in image_dirs]
    col_gap, gap = 0.15, 0.38
    panel_w = (width_cm - panel_sep_cm) / 2
    map_w, img, span = _panel_geometry([l[3] for l in loaded], panel_w, col_gap, gap)
    colors: dict = {}
    # The legend belongs to the panel with the most entries (normally both have all attacks).
    legend_idx = max(range(2), key=lambda i: len(loaded[i][1]))
    bodies = []
    for i, (d, (rec, pins, cols, needed)) in enumerate(zip(image_dirs, loaded)):
        colors.update(cols)
        lat_range = _fit_lat_range(needed, span)
        x_off = i * (panel_w + panel_sep_cm)
        legend = None
        if i == legend_idx:
            center_lon = -180 + (width_cm / 2 - x_off - img - col_gap) * 360.0 / map_w
            legend = _legend_opts(len(pins), center_lon, lat_range[1])
        bodies.append(_teaser_panel(d, rec, pins, lat_range, "ab"[i], out_dir / "tikz", stem, x_off, map_w, img,
                                    col_gap, map_resolution, zoom_inset, 0.33 * map_w, 0.62, legend))
    result = _render("\n".join(bodies), out_dir, stem, colors)
    snippet = out_dir / "tikz" / f"{stem}_figure.tex"
    snippet.write_text(
        "% ICLR: the PDF is exactly \\textwidth wide (5.5in); no scaling needed.\n"
        "\\begin{figure}[t]\n  \\centering\n"
        f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}\n"
        "  \\caption{TODO}\n  \\label{fig:teaser}\n\\end{figure}\n")
    result["figure_snippet"] = str(snippet)
    return result
