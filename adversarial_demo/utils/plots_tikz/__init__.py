"""TikZ/pgfplots backend for the plotting package.

Standalone-.tex twins of the matplotlib plots in ``utils.plots``, sharing the
same data-loading code so numbers/colors match exactly. See
``utils.plots_tikz.plots`` for the individual functions and
``utils.plots_tikz.common`` for the shared doc/compile helpers.
"""

from utils.plots_tikz.plots import (
    plot_attack_dtd_variance_tikz,
    plot_attack_success_rate_tikz,
    plot_clean_vs_attacked_displacement_tikz,
    plot_loss_vs_fsd_tikz,
    plot_localizability_results_tikz,
    plot_localizability_vs_attacks_tikz,
    plot_model_transfer_success_rate_tikz,
    plot_results_tikz,
    plot_robustness_results_tikz,
    plot_sampling_steps_success_rate_tikz,
)
from utils.plots_tikz.teaser import plot_geolocation_teaser_tikz

__all__ = [
    "plot_results_tikz",
    "plot_attack_success_rate_tikz",
    "plot_attack_dtd_variance_tikz",
    "plot_loss_vs_fsd_tikz",
    "plot_clean_vs_attacked_displacement_tikz",
    "plot_localizability_results_tikz",
    "plot_localizability_vs_attacks_tikz",
    "plot_sampling_steps_success_rate_tikz",
    "plot_model_transfer_success_rate_tikz",
    "plot_robustness_results_tikz",
    "plot_geolocation_teaser_tikz",
]
