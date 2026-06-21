"""
Plotting package for adversarial evaluation.

Split by concern:
- ``common``    : shared data helpers + JSON dumping (no matplotlib/cartopy)
- ``maps``      : world-map scatter / trajectory plots
- ``results``   : displacement and success-rate plots (also dump JSON)
- ``ablations`` : restart, sampling-steps, localizability, transferability plots

``utils.plots_adversarial_attacks`` re-exports everything here so existing imports
keep working.
"""

from utils.plots.common import (
    _display_attack_name,
    _get_metric_samples_by_budget,
    _get_metric_tensor,
    _load_attack_results,
    _metric_aliases,
    _plot_valid_path,
    _sanitize_lon_lat,
    _select_displacement_metric,
    _summarize_samples,
    save_plot_json,
)
from utils.plots.maps import (
    plot_gps_samples_on_map,
    plot_gps_trajectories_clean,
    plot_gps_trajectories_on_map,
)
from utils.plots.results import (
    plot_attack_success_rate,
    plot_results,
)
from utils.plots.ablations import (
    plot_localizability_results,
    plot_restarts_success,
    plot_sampling_steps_success_rate,
    plot_transferability_results,
)

__all__ = [
    "plot_gps_samples_on_map",
    "plot_gps_trajectories_on_map",
    "plot_gps_trajectories_clean",
    "plot_results",
    "plot_attack_success_rate",
    "plot_restarts_success",
    "plot_sampling_steps_success_rate",
    "plot_localizability_results",
    "plot_transferability_results",
    "save_plot_json",
]
