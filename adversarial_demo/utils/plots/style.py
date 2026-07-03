"""Shared style helpers for paper-ready (ECCV) figures.

Two things live here:
- A fixed attack -> color mapping, keyed by canonical position rather than by
  the order attacks happen to be passed to a given plot call, so e.g. "encoder"
  is always the same color whether it appears alone or alongside seven other
  attacks in a different figure.
- ``apply_paper_style()``, a couple of rcParams needed for camera-ready output
  (real font embedding in the PDF, legible base font sizes once figures are
  shrunk to column width in the paper).
"""

from __future__ import annotations

import matplotlib.pyplot as plt

# Canonical ordering used to assign colors -- position in this list, not the
# order a given plot call happens to receive attacks in, decides the color.
_ATTACK_COLOR_ORDER = [
    "encoder", "sampling", "diffusion_l2", "dtd", "diffusion",
    "ace", "unidef", "unidef_nofdje", "geoshield", "training_loss",
]
# Okabe-Ito colorblind-safe qualitative palette (standard choice for CV papers).
_ATTACK_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
    "#56B4E9", "#F0E442", "#000000", "#999999", "#8B4513",
]


def attack_color(attack_type: str) -> str:
    """Deterministic color for an attack type, stable across every figure."""
    normalized = str(attack_type).lower()
    if normalized in _ATTACK_COLOR_ORDER:
        return _ATTACK_PALETTE[_ATTACK_COLOR_ORDER.index(normalized)]
    # Unseen attack name: stable fallback so repeated calls still agree.
    return _ATTACK_PALETTE[hash(normalized) % len(_ATTACK_PALETTE)]


_STYLE_APPLIED = False


def apply_paper_style() -> None:
    """Set rcParams once for camera-ready figures (idempotent, safe to re-call)."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    plt.rcParams.update({
        # Embed real (Type 1 / TrueType) fonts in vector output -- most CV
        # venues (incl. ECCV/CVF) reject Type 3 fonts in submitted PDFs.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    _STYLE_APPLIED = True
