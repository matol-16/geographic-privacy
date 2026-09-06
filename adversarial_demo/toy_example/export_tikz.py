"""Emit a clean standalone TikZ sphere of the toy *result* from
``out/toy_geometry.npz``.

Produces ``out/toy_sphere.tex`` -- a self-contained ``standalone`` document with a
single shaded ball (TikZ ``ball color`` shading, no raster) showing, from a shared
set of starts that nominally commit to basin A:
  - the **magnitude-deviated** ensemble (blue) staying at mode A,
  - the **direction-deviated** ensemble (orange) migrating to mode B,
with endpoints coloured by the basin they land in, plus the three velocity arrows
(nominal / rotated / rescaled) on one representative trajectory. Compile with:

    cd out && pdflatex toy_sphere.tex

Only needs numpy (the 3D->2D orthographic projection is done here), so it runs in
any environment.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

# orthographic camera (bisector of the two modes; front iff p.camera_dir > 0) ---
ELEV, AZIM = np.radians(30.0), np.radians(-6.0)


def _basis():
    ce, se = np.cos(ELEV), np.sin(ELEV)
    ca, sa = np.cos(AZIM), np.sin(AZIM)
    fwd = np.array([ce * ca, ce * sa, se])
    right = np.array([-sa, ca, 0.0])
    up = np.cross(right, fwd)
    return right, up, fwd


RIGHT, UP, FWD = _basis()


def project(p):
    p = np.asarray(p, float)
    return np.array([p @ RIGHT, p @ UP]), float(p @ FWD)


def poly(points, min_depth=0.0):
    """3D polyline -> list of front-hemisphere TikZ coordinate segments."""
    segs, cur = [], []
    for p in points:
        (u, v), d = project(p)
        if d < min_depth:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
            continue
        cur.append(f"({u:.4f},{v:.4f})")
    if len(cur) > 1:
        segs.append(cur)
    return segs


def emit_path(segs, style):
    return "\n".join(r"  \draw[" + style + "] " + " -- ".join(s) + ";"
                     for s in segs)


def emit_dot(p, style, r=1.6, min_depth=-0.05):
    (u, v), d = project(p)
    if d < min_depth:
        return ""
    return fr"  \fill[{style}] ({u:.4f},{v:.4f}) circle ({r}pt);"


def emit_arrow(base, vec, style, scale, halo=True):
    (u0, v0), _ = project(base)
    tip = base + vec * scale
    (u1, v1), _ = project(tip)
    out = ""
    if halo:  # white casing so the arrow reads over the sphere/threads
        out += (fr"  \draw[white,line width=2.4pt,-{{Latex[length=2.4mm]}}] "
                fr"({u0:.4f},{v0:.4f}) -- ({u1:.4f},{v1:.4f});" + "\n")
    return out + (fr"  \draw[{style},-{{Latex[length=2.2mm]}}] "
                  fr"({u0:.4f},{v0:.4f}) -- ({u1:.4f},{v1:.4f});")


def graticule(n_lat=5, n_lon=12, n=120):
    lines = []
    for lat in np.linspace(-60, 60, n_lat):
        la = np.radians(lat)
        t = np.linspace(0, 2 * np.pi, n)
        c = np.stack([np.cos(la) * np.cos(t), np.cos(la) * np.sin(t),
                      np.sin(la) * np.ones_like(t)], axis=1)
        lines += poly(c, min_depth=0.03)
    for lon in np.linspace(0, 2 * np.pi, n_lon, endpoint=False):
        t = np.linspace(-np.pi / 2, np.pi / 2, n)
        c = np.stack([np.cos(t) * np.cos(lon), np.cos(t) * np.sin(lon),
                      np.sin(t)], axis=1)
        lines += poly(c, min_depth=0.03)
    return lines


def basin(p, mode_a, mode_b):
    return 0 if np.dot(p, mode_a) >= np.dot(p, mode_b) else 1


TEMPLATE = r"""\documentclass[tikz,border=3pt]{standalone}
\usepackage{amsmath}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\definecolor{cnom}{HTML}{3A3A3A}
\definecolor{cA}{HTML}{D62728}   % basin A + magnitude deviation (stays at A)
\definecolor{cB}{HTML}{1F6FB2}   % basin B + direction deviation (flips to B)
\definecolor{cwire}{HTML}{AFC0D2}
\definecolor{cflow}{HTML}{6B7683} % faint true-flow streamlines
\definecolor{cball}{HTML}{F4F8FC} % clean, light (not grey) sphere fill
\begin{document}
\begin{tikzpicture}[scale=4.6]
  % --- clean light sphere (flat fill + soft highlight, no grey ball shading) ---
  \fill[cball] (0,0) circle (1);
  \fill[white,opacity=0.6] (-0.32,0.34) circle (0.40);
  \draw[cwire!85!black,line width=0.7pt] (0,0) circle (1);
  % --- graticule (front) ---
@@GRAT@@
  % --- true (nominal) velocity field: faint streamlines under everything ---
@@FLOW@@
  % --- separatrix ---
@@SEP@@
  % --- magnitude ensemble (stays at A) ---
@@MAG@@
  % --- direction ensemble (flips to B) ---
@@DIR@@
  % --- starts + endpoints ---
@@DOTS@@
  % --- velocity arrows on one representative trajectory ---
@@ARROWS@@
  % --- modes ---
@@MODES@@
  % --- legend (boxed, stands out) ---
  \node[draw=black!60, fill=white, rounded corners=2pt, inner sep=6pt,
        anchor=north west, font=\footnotesize, align=left] at (-1.42,1.42)
    {\tikz{\draw[cB,line width=1.6pt] (0,0)--(0.42,0);}\,~direction dev.\ (rotate $v$)\,$\to$\,B\\[3pt]
     \tikz{\draw[cA,line width=1.6pt] (0,0)--(0.42,0);}\,~magnitude dev.\ (rescale $\lVert v\rVert$)\,$\to$\,A\\[3pt]
     \tikz{\draw[cflow,line width=1.0pt,opacity=0.55] (0,0)--(0.42,0);}\,~true flow (unperturbed field)};
\end{tikzpicture}
\end{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--n-traj", type=int, default=16,
                    help="ensemble trajectories to draw (subsampled)")
    args = ap.parse_args()

    g = dict(np.load(os.path.join(args.out, "toy_geometry.npz")))
    mode_a, mode_b = g["mode_a"], g["mode_b"]

    ens_dir = g["ens_direction"]
    ens_mag = g["ens_magnitude"]
    k = len(ens_dir)
    idx = np.linspace(0, k - 1, min(args.n_traj, k)).round().astype(int)

    grat = emit_path(graticule(), "cwire, line width=0.25pt, opacity=0.55")
    flow = "\n".join(emit_path(poly(tr), "cflow, line width=0.3pt, opacity=0.28")
                     for tr in g["bg_streamlines"])
    sep = emit_path(poly(g["separatrix"], min_depth=0.03),
                    "cwire!55!black, dashed, line width=0.5pt")

    # magnitude ensemble -> basin A (red); direction ensemble -> basin B (blue)
    mag = "\n".join(emit_path(poly(ens_mag[i]),
                              "cA, line width=0.55pt, opacity=0.55") for i in idx)
    dirc = "\n".join(emit_path(poly(ens_dir[i]),
                               "cB, line width=0.55pt, opacity=0.55") for i in idx)

    dots = []
    for i in idx:
        dots.append(emit_dot(ens_dir[i][0], "cnom", r=0.7))           # start
    for i in idx:
        dots.append(emit_dot(ens_mag[i][-1], "cA", r=1.3))            # ends at A
        dots.append(emit_dot(ens_dir[i][-1], "cB", r=1.3))            # ends at B
    dots = "\n".join(d for d in dots if d)

    scale = 0.42 / (np.linalg.norm(g["v_nominal"]) + 1e-9)
    arrows = "\n".join([
        emit_arrow(g["arrow_base"], g["v_norm"], "cA, line width=1.3pt", scale),
        emit_arrow(g["arrow_base"], g["v_angle"], "cB, line width=1.3pt", scale),
        emit_arrow(g["arrow_base"], g["v_nominal"], "cnom, line width=1.3pt", scale),
        emit_dot(g["arrow_base"], "cnom", r=1.1),
    ])

    modes = "\n".join([
        emit_dot(mode_a, "white", r=3.6),
        emit_dot(mode_a, "cA", r=2.9),
        fr"  \node[font=\bfseries\small,white] at "
        fr"({project(mode_a)[0][0]:.3f},{project(mode_a)[0][1]:.3f}) {{A}};",
        emit_dot(mode_b, "white", r=3.6),
        emit_dot(mode_b, "cB", r=2.9),
        fr"  \node[font=\bfseries\small,white] at "
        fr"({project(mode_b)[0][0]:.3f},{project(mode_b)[0][1]:.3f}) {{B}};",
    ])

    tex = TEMPLATE
    for token, value in {
        "@@GRAT@@": grat, "@@FLOW@@": flow, "@@SEP@@": sep, "@@MAG@@": mag,
        "@@DIR@@": dirc, "@@DOTS@@": dots, "@@ARROWS@@": arrows, "@@MODES@@": modes,
    }.items():
        tex = tex.replace(token, value)

    p = os.path.join(args.out, "toy_sphere.tex")
    with open(p, "w") as f:
        f.write(tex)
    print(f"[write] {p}")
    print(f"Compile with:  cd {args.out} && pdflatex toy_sphere.tex")


if __name__ == "__main__":
    main()
