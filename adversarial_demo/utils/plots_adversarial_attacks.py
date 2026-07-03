"""
Backward-compatibility facade for the plotting package.

The plotting code now lives in the ``utils.plots`` package (split into ``maps``,
``results``, ``ablations``, and shared ``common`` helpers). This module re-exports
every public name so existing imports such as
``from utils.plots_adversarial_attacks import plot_results`` keep working.
"""

from utils.plots import (  # noqa: F401
    plot_attack_dtd_variance,
    plot_attack_success_rate,
    plot_clean_vs_attacked_displacement,
    plot_gps_samples_on_map,
    plot_gps_trajectories_clean,
    plot_gps_trajectories_on_map,
    plot_loss_vs_fsd,
    plot_localizability_results,
    plot_model_transfer_success_rate,
    plot_restarts_success,
    plot_results,
    plot_robustness_results,
    plot_sampling_steps_success_rate,
    plot_transferability_results,
    save_plot_json,
)

# Shared helpers, re-exported for the rare external caller that imported them here.
from utils.plots.common import (  # noqa: F401
    _display_attack_name,
    _get_metric_samples_by_budget,
    _get_metric_tensor,
    _load_attack_results,
    _metric_aliases,
    _plot_valid_path,
    _sanitize_lon_lat,
    _select_displacement_metric,
    _summarize_samples,
)
