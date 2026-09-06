"""Render the toy result on a clean sphere.

Reads ``out/toy_geometry.npz`` + ``out/toy_results.json`` (from
``run_toy_experiment.py``) and writes ``out/toy_figure.pdf/.png``.

Top row -- three orthographic globes showing *ensembles* of trajectories from many
starts that nominally commit to basin A, integrated under:
  (1) nominal, (2) direction-deviated (rotate v), (3) magnitude-deviated
      (rescale ||v||), at a matched L2 budget.
Endpoints are coloured by the basin they land in (green = A, purple = B): the
direction ensemble migrates to mode B, the magnitude ensemble stays at mode A.

Bottom row -- endpoint displacement and basin-flip rate vs the matched L2 budget.

The globe is drawn with a normal-shaded disc + faint graticule (orthographic
projection computed here), so it reads as a clean 3D ball rather than matplotlib's
grey ``plot_surface``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.patches import Circle    # noqa: E402

# palette (unified with the TikZ sphere): red = basin A / magnitude (stays at A),
# blue = basin B / direction (flips to B).
C_NOM = "#3a3a3a"      # nominal flow
C_DIR = "#1f6fb2"      # direction / cosine deviation  (-> basin B)
C_MAG = "#d62728"      # magnitude / L2 deviation      (-> basin A)
C_A = "#d62728"        # mode A / basin A  (red)
C_B = "#1f6fb2"        # mode B / basin B  (blue)
SPHERE_RGB = np.array([0.94, 0.96, 0.99])   # light, clean ball (not grey)
LIGHT_DIR = np.array([-0.4, 0.6, 0.72])     # screen-space light

# orthographic camera (matches export_tikz) -----------------------------------
ELEV, AZIM = np.radians(30.0), np.radians(-6.0)


def _basis():
    ce, se = np.cos(ELEV), np.sin(ELEV)
    ca, sa = np.cos(AZIM), np.sin(AZIM)
    fwd = np.array([ce * ca, ce * sa, se])   # camera direction; front iff p.fwd>0
    right = np.array([-sa, ca, 0.0])
    up = np.cross(right, fwd)
    return right, up, fwd


RIGHT, UP, FWD = _basis()


def project(P):
    """(...,3) unit points -> (...,2) screen coords, (...,) depth (front if >0)."""
    P = np.asarray(P, float)
    u = P @ RIGHT
    v = P @ UP
    d = P @ FWD
    return np.stack([u, v], axis=-1), d


# --- clean globe -------------------------------------------------------------
def draw_globe(ax, res_px=520):
    """A normal-shaded unit ball + faint graticule, front hemisphere."""
    xs = np.linspace(-1, 1, res_px)
    gx, gy = np.meshgrid(xs, xs)
    r2 = gx ** 2 + gy ** 2
    inside = r2 <= 1.0
    gz = np.sqrt(np.clip(1 - r2, 0, 1))
    n = np.stack([gx, gy, gz], axis=-1)              # screen-space normal
    diff = np.clip(n @ (LIGHT_DIR / np.linalg.norm(LIGHT_DIR)), 0, 1)
    shade = 0.78 + 0.22 * diff ** 0.9   # gentle shading, stays light/clean
    rgb = SPHERE_RGB[None, None, :] * shade[..., None]
    rgba = np.concatenate([rgb, inside[..., None].astype(float)], axis=-1)
    ax.imshow(rgba, extent=(-1, 1, -1, 1), origin="lower", zorder=0,
              interpolation="bilinear")
    ax.add_patch(Circle((0, 0), 1.0, fill=False, ec="#9aa7b4", lw=1.0, zorder=6))

    # graticule (faint), front only
    t = np.linspace(0, 2 * np.pi, 160)
    for lat in np.linspace(-60, 60, 5):
        la = np.radians(lat)
        c = np.stack([np.cos(la) * np.cos(t), np.cos(la) * np.sin(t),
                      np.sin(la) * np.ones_like(t)], axis=1)
        _draw_world_curve(ax, c, color="#b7c2cd", lw=0.5, alpha=0.5, zorder=1)
    for lon in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        tt = np.linspace(-np.pi / 2, np.pi / 2, 100)
        c = np.stack([np.cos(tt) * np.cos(lon), np.cos(tt) * np.sin(lon),
                      np.sin(tt)], axis=1)
        _draw_world_curve(ax, c, color="#b7c2cd", lw=0.5, alpha=0.5, zorder=1)


def _split_front(P):
    """Yield contiguous front-hemisphere (depth>0) screen-space segments."""
    xy, d = project(P)
    seg = []
    for i in range(len(P)):
        if d[i] > 0.0:
            seg.append(xy[i])
        elif seg:
            yield np.array(seg)
            seg = []
    if seg:
        yield np.array(seg)


def _draw_world_curve(ax, P, **kw):
    for seg in _split_front(P):
        if len(seg) > 1:
            ax.plot(seg[:, 0], seg[:, 1], **kw)


def _basin_of(p, mode_a, mode_b):
    return 0 if np.dot(p, mode_a) >= np.dot(p, mode_b) else 1


def _globe_panel(ax, g, trajs, line_color, title, subtitle):
    mode_a, mode_b = g["mode_a"], g["mode_b"]
    draw_globe(ax)
    _draw_world_curve(ax, g["separatrix"], color="#8a97a4", lw=1.1,
                      alpha=0.9, ls=(0, (5, 3)), zorder=2)

    # trajectories (front segments), thin + translucent
    for tr in trajs:
        for seg in _split_front(tr):
            if len(seg) > 1:
                ax.plot(seg[:, 0], seg[:, 1], color=line_color, lw=0.9,
                        alpha=0.45, zorder=3, solid_capstyle="round")
    # start points (small) and endpoints (coloured by basin)
    starts = np.array([tr[0] for tr in trajs])
    ends = np.array([tr[-1] for tr in trajs])
    for P, s, z, ec in [(starts, 9, 4, "none"), (ends, 34, 5, "white")]:
        xy, d = project(P)
        front = d > -0.02
        if P is starts:
            ax.scatter(xy[front, 0], xy[front, 1], s=s, c="#2b2b2b",
                       zorder=z, linewidths=0)
        else:
            cols = [C_A if _basin_of(p, mode_a, mode_b) == 0 else C_B
                    for p in P[front]]
            ax.scatter(xy[front, 0], xy[front, 1], s=s, c=cols, zorder=z,
                       edgecolors=ec, linewidths=0.5)

    # modes
    for m, col, lab in [(mode_a, C_A, "A"), (mode_b, C_B, "B")]:
        (u, v), d = project(m)
        if d > -0.05:
            ax.scatter([u], [v], s=150, c=col, edgecolors="white",
                       linewidths=1.3, zorder=7)
            ax.text(u, v, lab, color="white", ha="center", va="center",
                    fontsize=8.5, fontweight="bold", zorder=8)

    ax.set_title(title, fontsize=10.5, pad=2)
    ax.text(0.5, -0.06, subtitle, transform=ax.transAxes, ha="center",
            va="top", fontsize=8.7, color=line_color)
    ax.set_xlim(-1.13, 1.13)
    ax.set_ylim(-1.13, 1.13)
    ax.set_aspect("equal")
    ax.axis("off")


def _flip_frac(trajs, g):
    ends = np.array([tr[-1] for tr in trajs])
    return np.mean([_basin_of(p, g["mode_a"], g["mode_b"]) == 1 for p in ends])


# --- quantitative panels -----------------------------------------------------
def plot_displacement(ax, res):
    sw = res["sweep"]
    b = np.array(sw["budgets"])
    ax.axhline(res["hero"]["mode_separation_rad"], color="#c2c2c2", ls=":", lw=1.0)
    ax.text(b[0], res["hero"]["mode_separation_rad"], "  inter-mode distance",
            va="bottom", ha="left", fontsize=7.5, color="#8a8a8a")
    ax.plot(b, sw["angle_displacement_mean"], "-o", color=C_DIR, ms=4,
            label="direction (rotate $v$)")
    ax.plot(b, sw["norm_displacement_mean"], "-s", color=C_MAG, ms=4,
            label="magnitude (rescale $\\|v\\|$)")
    ax.set_xlabel("matched $L^2$ budget  $s$")
    ax.set_ylabel("endpoint disp. (rad)")
    ax.set_title("Displacement vs equal-$L^2$ perturbation", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(bottom=0)


def plot_flip(ax, res):
    sw = res["sweep"]
    b = np.array(sw["budgets"])
    ax.plot(b, sw["angle_basin_flip_frac"], "-o", color=C_DIR, ms=4,
            label="direction (rotate $v$)")
    ax.plot(b, sw["norm_basin_flip_frac"], "-s", color=C_MAG, ms=4,
            label="magnitude (rescale $\\|v\\|$)")
    ax.set_xlabel("matched $L^2$ budget  $s$")
    ax.set_ylabel("basin-flip fraction")
    ax.set_title("How often the predicted mode changes", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper left", fontsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    args = ap.parse_args()

    g = dict(np.load(os.path.join(args.out, "toy_geometry.npz")))
    with open(os.path.join(args.out, "toy_results.json")) as f:
        res = json.load(f)

    ens_nom = list(g["ens_nominal"])
    ens_dir = list(g["ens_direction"])
    ens_mag = list(g["ens_magnitude"])
    n = len(ens_nom)
    s = float(g["ens_s"])
    fd, fm = _flip_frac(ens_dir, g), _flip_frac(ens_mag, g)

    fig = plt.figure(figsize=(11.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 6, height_ratios=[1.25, 1.0])
    axg = [fig.add_subplot(gs[0, i:i + 2]) for i in (0, 2, 4)]
    _globe_panel(axg[0], g, ens_nom, C_NOM, f"Nominal  ($n={n}$ runs)",
                 "all commit to basin A")
    _globe_panel(axg[1], g, ens_dir, C_DIR,
                 f"Direction-deviated  ($s={s:g}$)",
                 f"{fd*100:.0f}% flip to basin B")
    _globe_panel(axg[2], g, ens_mag, C_MAG,
                 f"Magnitude-deviated  ($s={s:g}$)",
                 f"{fm*100:.0f}% flip  (stays in A)")

    ax_disp = fig.add_subplot(gs[1, 0:3])
    ax_flip = fig.add_subplot(gs[1, 3:6])
    plot_displacement(ax_disp, res)
    plot_flip(ax_flip, res)

    fig.suptitle("Geolocation sampling is basin selection: direction chooses the "
                 "basin, magnitude only the speed within it", fontsize=12)

    for ext in ("pdf", "png"):
        p = os.path.join(args.out, f"toy_figure.{ext}")
        fig.savefig(p, dpi=200)
        print(f"[write] {p}")


if __name__ == "__main__":
    main()
