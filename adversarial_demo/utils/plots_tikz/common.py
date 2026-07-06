"""Shared helpers for the TikZ/pgfplots plotting backend.

Mirrors ``utils.plots.style``/``utils.plots.common`` but emits standalone
pgfplots ``.tex`` sources -- paper-ready vector figures, no whole-figure
titles (same "no titles, captions live in LaTeX" convention as the
matplotlib backend) -- instead of matplotlib figures.

Best-effort compile: if ``pdflatex``/``pdftoppm`` are on PATH, every plot is
also compiled to a PDF and rasterized to a PNG preview so it can be viewed
without leaving Python. If either tool -- or a LaTeX package the source
needs (pgfplots) -- is missing, the ``.tex`` is still written (it will
compile wherever pgfplots is available, e.g. Overleaf) and a short warning
is printed instead of raising.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Optional, Sequence

import numpy as np

from utils.plots.style import _ATTACK_COLOR_ORDER, attack_color

_PDFLATEX = shutil.which("pdflatex")
_LUALATEX = shutil.which("lualatex")
_PDFTOPPM = shutil.which("pdftoppm")
_PREVIEW_DPI = 150

_PREAMBLE = r"""\documentclass[tikz,border=2pt]{{standalone}}
\usepackage[T1]{{fontenc}}
\usepackage{{graphicx}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.17}}
\usepgfplotslibrary{{fillbetween}}
\usepgfplotslibrary{{groupplots}}
\usepgfplotslibrary{{statistics}}
\usetikzlibrary{{patterns}}
{extra_preamble}
\begin{{document}}
\begin{{tikzpicture}}
{body}
\end{{tikzpicture}}
\end{{document}}
"""

# Same canonical order as ``attack_color`` (position, not call order, decides
# the assignment) so that color, marker shape, and bar hatch pattern for a
# given attack always agree across every figure -- an attack should be
# identifiable by color *or* shape alone (colorblind-safe, grayscale-safe).
_ATTACK_MARKERS = ["*", "square*", "triangle*", "star", "diamond*",
                   "pentagon*", "+", "10-pointed star", "x", "asterisk"]
# "oplus*"/"otimes*" (circle + plus/cross) render as plain filled circles in
# this pgfplots version -- indistinguishable from "*" -- so plain "+"/"x" are
# used instead, which actually show their shape at these small mark sizes.
_ATTACK_PATTERNS = ["north east lines", "horizontal lines", "vertical lines", "crosshatch", "dots",
                    "north west lines", "grid", "crosshatch dots", "bricks", "fivepointed stars"]


def attack_marker(attack_type: str) -> str:
    """Deterministic pgfplots mark shape for an attack, stable across every figure
    (mirrors ``attack_color``'s canonical-position scheme)."""
    normalized = str(attack_type).lower()
    if normalized in _ATTACK_COLOR_ORDER:
        return _ATTACK_MARKERS[_ATTACK_COLOR_ORDER.index(normalized)]
    return _ATTACK_MARKERS[hash(normalized) % len(_ATTACK_MARKERS)]


def attack_pattern(attack_type: str) -> str:
    """Deterministic tikz fill pattern (for bar charts) for an attack, stable
    across every figure (mirrors ``attack_color``'s canonical-position scheme)."""
    normalized = str(attack_type).lower()
    if normalized in _ATTACK_COLOR_ORDER:
        return _ATTACK_PATTERNS[_ATTACK_COLOR_ORDER.index(normalized)]
    return _ATTACK_PATTERNS[hash(normalized) % len(_ATTACK_PATTERNS)]


# Thin/line-based marks read as smaller than filled shapes at the same nominal
# size -- bump them up so every attack's marker is about equally visible.
_THIN_MARKERS = {"star", "asterisk", "10-pointed star", "+", "x"}


def attack_mark_size(attack_type: str, base_pt: float = 1.3) -> str:
    size = base_pt * 1.6 if attack_marker(attack_type) in _THIN_MARKERS else base_pt
    # DTD is this paper's own attack -- give it a further size bump on top of
    # the thin-marker boost so it visibly stands out from the baselines.
    if str(attack_type).lower() == "dtd":
        size *= 1.2
    return f"{size:.2g}pt"


def attack_line_width(attack_type: str, base_pt: float = 1.4) -> str:
    """Line width for a line-plot series -- DTD gets a bit thicker so it reads
    as this paper's own attack among the baselines."""
    size = base_pt * 1.25 if str(attack_type).lower() == "dtd" else base_pt
    return f"{size:.2g}pt"


def sequential_colormap_def(colormap_name: str, hex_color: str) -> str:
    """A pgfplots colormap from white to ``hex_color``, for a density heatmap
    that should read as "this attack's own hue, more of it where mass
    concentrates" rather than a generic sequential palette."""
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"\\pgfplotsset{{colormap={{{colormap_name}}}{{rgb255=(255,255,255) rgb255=({r},{g},{b})}}}}"


def contrasting_ink(hex_color: str) -> str:
    """"black" or "white", whichever reads better on top of ``hex_color``
    (perceptual luma threshold) -- used to pick a bar chart's pattern-overlay
    color so the hatch is visible regardless of how light/dark the fill is."""
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return "black" if luma > 140 else "white"


# pgfplots' default legend image for a marked line samples the mark at BOTH
# ends of the little line swatch (an intentional "this is a marked line" cue),
# which reads as "the symbol is shown twice" once markers are the primary way
# to tell attacks apart. Overriding "legend image code" to sample the marker
# at only the middle of a 3-point line keeps the line-plus-mark look with a
# single, centered symbol. Verified via isolated pgfplots test (mark repeat=3,
# mark phase=2 marks only the 2nd of 3 plotted points).
SINGLE_MARK_LEGEND_IMAGE = (
    "legend image code/.code={"
    "\\draw[mark repeat=3, mark phase=2, #1] "
    "plot coordinates {(0cm,0cm) (0.3cm,0cm) (0.6cm,0cm)};"
    "}"
)


def bar_legend_image_code(color_name: str, pattern: str, pattern_color: str) -> str:
    """Single-swatch pgfplots legend image for a ``ybar`` bar using
    ``postaction={pattern=...}`` for its hatch.

    Without this, pgfplots' default "area legend" swatch renders as two
    adjacent rectangles once a pattern postaction is involved (verified via
    an isolated test: postaction on a plain ybar addplot alone is fine, but
    the *default* legend image duplicates itself into a wide-then-narrow pair
    once a pattern is added) -- this draws the fill, then the pattern, as one
    explicit rectangle instead of relying on that default.
    """
    return (
        "legend image code/.code={"
        f"\\path[fill={color_name}] (0cm,-0.11cm) rectangle (0.62cm,0.24cm);"
        f"\\path[pattern={pattern}, pattern color={pattern_color}, draw=black, line width=0.3pt] "
        "(0cm,-0.11cm) rectangle (0.62cm,0.24cm);"
        "}"
    )


def attacks_dtd_last(attack_types: Sequence[str]) -> list[str]:
    """Reorder so DTD (this paper's attack) is always last -- the rightmost column,
    bar, or facet in every figure that lays attacks out spatially -- while every
    other attack keeps its relative order."""
    attack_types = list(attack_types)
    dtd = [a for a in attack_types if str(a).lower() == "dtd"]
    others = [a for a in attack_types if str(a).lower() != "dtd"]
    return others + dtd


def sanitize_pgfname(name: str) -> str:
    """A pgf-safe identifier (letters only) for use as a color/path name."""
    cleaned = re.sub(r"[^a-zA-Z]", "", str(name))
    if not cleaned or not cleaned[0].isalpha():
        cleaned = "n" + cleaned
    return cleaned


def tex_escape(text: str) -> str:
    """Escape LaTeX special characters in a plain-text label/legend entry."""
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def hex_to_pgf(hex_color: str) -> str:
    return hex_color.lstrip("#").upper()


def darken_hex(hex_color: str, factor: float) -> str:
    """Multiply RGB by ``factor`` (clipped to [0, 255]).

    Matches the matplotlib backend's ``_darken_rgba`` exactly, so threshold
    shading in ``plot_attack_success_rate`` looks identical across backends.
    """
    rgb = np.array([int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)
    rgb = np.clip(rgb * factor, 0, 255)
    return "".join(f"{int(round(c)):02X}" for c in rgb)


def nice_linear_ticks(vmax: float, n: int = 4) -> list[float]:
    """~``n`` round-numbered ticks spanning ``[0, vmax]`` (e.g. 0/5000/10000/...).

    Standard "nice numbers" step selection: round the raw step up to the
    nearest 1/2/2.5/5/10 x its order of magnitude.
    """
    if vmax <= 0:
        return [0.0]
    raw_step = vmax / n
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = magnitude * 10
    for mult in (1, 2, 2.5, 5, 10):
        if mult * magnitude >= raw_step:
            step = mult * magnitude
            break
    n_ticks = int(np.ceil(vmax / step)) + 1
    return [float(i * step) for i in range(n_ticks)]


def format_km_tick(value: float) -> str:
    """``5000 -> "5{,}000 km"`` -- comma needs escaping since it's a pgfplots list separator."""
    return f"{value:,.0f}".replace(",", "{,}") + " km"


def attack_pgf_color(attack_type: str) -> tuple[str, str]:
    """Return ``(pgf color name, hex-without-#)`` for an attack.

    Colors match the matplotlib backend's ``attack_color`` exactly.
    """
    name = "attack" + sanitize_pgfname(attack_type).capitalize()
    return name, hex_to_pgf(attack_color(attack_type))


def groupplot_legend_center_x(n_cols: int, panel_width_cm: float, h_sep_cm: float) -> float:
    """X-fraction (in column-0's own axis frame) that centers a legend over an
    entire ``groupplot`` row of ``n_cols`` equal-width panels.

    Pair with ``legend style={at={(<this>,y)}, anchor=south, ...}`` on column 0
    -- pgfplots' ``at`` coordinate isn't clipped to [0,1], so a fraction >1 is
    the normal way to place a legend relative to a *different* axis's frame.
    Without this, ``at={(0.5,y)}`` only centers over column 0 itself, leaving
    the legend visibly off-center once there's more than one column.
    """
    total_width = n_cols * panel_width_cm + (n_cols - 1) * h_sep_cm
    return total_width / (2 * panel_width_cm)


def color_definitions(colors: dict[str, str]) -> str:
    """``colors``: ``{pgf_name: hex_without_#}``."""
    return "\n".join(f"\\definecolor{{{name}}}{{HTML}}{{{hexcode}}}" for name, hexcode in colors.items())


def boxplot_stats(samples: np.ndarray) -> Optional[dict]:
    """Tukey five-number summary matching matplotlib's default ``boxplot`` (whis=1.5).

    Whiskers extend to the most extreme data point still within 1.5x IQR of
    the box -- not to the theoretical 1.5xIQR bound itself -- exactly like
    matplotlib, so a box drawn from these numbers matches the PNG/PDF twin.
    Returns ``None`` for empty input.
    """
    finite = np.asarray(samples, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    q1, median, q3 = np.percentile(finite, [25, 50, 75])
    iqr = q3 - q1
    lo_bound, hi_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    within = finite[(finite >= lo_bound) & (finite <= hi_bound)]
    whisker_lo = float(within.min()) if within.size else float(q1)
    whisker_hi = float(within.max()) if within.size else float(q3)
    return {"median": float(median), "q1": float(q1), "q3": float(q3),
            "whisker_lo": whisker_lo, "whisker_hi": whisker_hi}


def boxplot_prepared(stats: dict, position: Optional[int] = None) -> str:
    """``position`` sets an explicit x-slot (``boxplot prepared/draw position``).

    Without it, pgfplots auto-increments an internal position counter whose
    starting value/step isn't guaranteed to line up with a hand-written
    ``xtick={0,1,2,...}`` -- pass the box's index explicitly so it always
    lands under the right tick label.
    """
    pos = f", draw position={position}" if position is not None else ""
    return (f"boxplot prepared={{median={stats['median']:.6g}, "
            f"upper quartile={stats['q3']:.6g}, lower quartile={stats['q1']:.6g}, "
            f"upper whisker={stats['whisker_hi']:.6g}, lower whisker={stats['whisker_lo']:.6g}{pos}}}")


def coordinates(xs: Sequence[float], ys: Sequence[float]) -> str:
    pts = [f"({float(x):.6g},{float(y):.6g})" for x, y in zip(xs, ys)
           if np.isfinite(x) and np.isfinite(y)]
    return "coordinates {" + " ".join(pts) + "}"


def write_dat_file(path: str, columns: dict[str, np.ndarray]) -> None:
    """Write an external pgfplots data table (used for large scatter series)."""
    names = list(columns.keys())
    arrays = [np.asarray(columns[n], dtype=np.float64) for n in names]
    n = len(arrays[0]) if arrays else 0
    with open(path, "w") as f:
        f.write(" ".join(names) + "\n")
        for i in range(n):
            row = [arrays[j][i] for j in range(len(arrays))]
            if all(np.isfinite(v) for v in row):
                f.write(" ".join(f"{v:.6g}" for v in row) + "\n")


def render_tikz(body: str, plot_dir: str, filename_stem: str, extra_preamble: str = "") -> dict:
    """Write ``body`` (the contents of a ``tikzpicture``) as a standalone .tex file
    under ``<plot_dir>/tikz/``, then best-effort compile it to a PDF and rasterize a
    PNG preview. Returns the paths actually produced (missing ones are ``None``)."""
    tikz_dir = os.path.abspath(os.path.join(plot_dir, "tikz"))
    os.makedirs(tikz_dir, exist_ok=True)
    tex_path = os.path.join(tikz_dir, f"{filename_stem}.tex")
    with open(tex_path, "w") as f:
        f.write(_PREAMBLE.format(extra_preamble=extra_preamble, body=body))
    print(f"TikZ source saved to: {tex_path}")

    result = {"tex": tex_path, "pdf": None, "png": None}
    if _PDFLATEX is None:
        print("  (pdflatex not found on PATH -- skipping local compile; the .tex is "
              "still portable to any TeX install with pgfplots, e.g. Overleaf)")
        return result

    # cwd=tikz_dir so relative external-data ``\addplot table {foo.dat}`` references
    # (written alongside the .tex by the scatter plots) resolve; pass just the
    # basename as the input file for the same reason.
    tex_basename = os.path.basename(tex_path)
    pdf_path = os.path.join(tikz_dir, f"{filename_stem}.pdf")

    def _run(compiler):
        return subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", tikz_dir, tex_basename],
            cwd=tikz_dir, capture_output=True, text=True, timeout=180,
        )

    try:
        proc = _run(_PDFLATEX)
    except subprocess.TimeoutExpired:
        print(f"  pdflatex timed out compiling {tex_path}")
        return result

    if (proc.returncode != 0 or not os.path.exists(pdf_path)) and _LUALATEX is not None:
        # pdftex has a fixed memory pool and can run out on dense scatter plots
        # (thousands of \addplot marks); lualatex allocates memory dynamically and
        # handles the exact same .tex source without this limit.
        print("  pdflatex failed (often a fixed-memory limit on dense scatter data) -- retrying with lualatex")
        try:
            proc = _run(_LUALATEX)
        except subprocess.TimeoutExpired:
            print(f"  lualatex timed out compiling {tex_path}")
            return result

    if proc.returncode != 0 or not os.path.exists(pdf_path):
        tail = "\n".join(proc.stdout.splitlines()[-15:])
        print(f"  Failed to compile {tex_path} (exit {proc.returncode}):\n{tail}")
        return result
    result["pdf"] = pdf_path
    print(f"  Compiled to: {pdf_path}")

    if _PDFTOPPM is not None:
        png_stem = os.path.join(tikz_dir, filename_stem)
        try:
            subprocess.run(
                [_PDFTOPPM, "-r", str(_PREVIEW_DPI), "-png", "-singlefile", pdf_path, png_stem],
                cwd=tikz_dir, capture_output=True, text=True, timeout=60, check=False,
            )
        except subprocess.TimeoutExpired:
            return result
        png_path = png_stem + ".png"
        if os.path.exists(png_path):
            result["png"] = png_path
            print(f"  Preview PNG: {png_path}")
    return result
