"""
Main-result plots: displacement vs. budget and attack success rate vs. budget.

Both functions also persist the plotted numbers as JSON next to the PNG (success
rates and final-step-displacement summaries) so figures can be reproduced or
re-styled without re-running the evaluation.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator

from utils.plots.common import (
    _display_attack_name,
    _get_metric_samples_by_budget,
    _load_attack_results,
    _select_displacement_metric,
    _summarize_samples,
    save_plot_json,
)


def plot_results(
    results_dir,
    attack_budgets,
    plot_dir,
    dataset_name,
    attack_types=None,
    attack_type=None,
    all_results=None,
    results=None,
    stored_metrics=None,
    gps_true: bool = False,
):
    # Support both old single-attack and new multi-attack signatures
    if attack_types is None:
        attack_types = [attack_type] if attack_type is not None else []
    if stored_metrics is None:
        stored_metrics = [_select_displacement_metric(gps_true)]
    if all_results is None:
        if results is not None:
            all_results = {attack_types[0]: results}
        elif results_dir is not None:
            all_results = _load_attack_results(results_dir, dataset_name, attack_types)

    for metric in stored_metrics:
        plt.figure(figsize=(10, 8))
        plotted_any = False
        json_payload: dict[str, Any] = {
            "dataset": dataset_name,
            "metric": metric,
            "attack_budgets": [float(b) for b in attack_budgets],
            "attacks": {},
        }
        for at, res in all_results.items():
            budget_samples = _get_metric_samples_by_budget(res, metric)
            if budget_samples is None:
                continue

            summary = np.asarray([
                _summarize_samples(samples, attack_name=at, budget=attack_budgets[i])
                for i, samples in enumerate(budget_samples)
            ], dtype=np.float64)
            mean_metric = summary[:, 0]
            median_metric = summary[:, 1]
            q25_metric = summary[:, 2]
            q75_metric = summary[:, 3]

            print(q25_metric, mean_metric, q75_metric)

            json_payload["attacks"][at] = {
                "mean": mean_metric.tolist(),
                "median": median_metric.tolist(),
                "q25": q25_metric.tolist(),
                "q75": q75_metric.tolist(),
                "n_samples_per_budget": [int(np.isfinite(s).sum()) for s in budget_samples],
            }

            attack_name = _display_attack_name(at)
            plt.plot(attack_budgets, mean_metric, linestyle='--', alpha=1.0, linewidth=3.0, label=f"{attack_name} mean")
            # plt.plot(attack_budgets, median_metric, label=f"{at} median")
            plt.fill_between(attack_budgets, q25_metric, q75_metric, alpha=0.3, label=f"{at} IQR (25-75%)")
            plotted_any = True
        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Attack budget (out of 255)")
        # plt.ylabel("Final step displacement (km)")

        # add note about log scale on y axis + quantiles

        # x and y axis should be log scale
        plt.xscale("log")
        plt.yscale("log")

        ax = plt.gca()
        tick_labels = [f"{eps*255:.0f}" for eps in attack_budgets]
        ax.xaxis.set_major_locator(FixedLocator(attack_budgets))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        ax.xaxis.set_minor_locator(NullLocator())
        # replace y ticks with nice values (1km, 10km, 100km, 1000km, 10000km)
        plt.yticks([1000, 2500, 5000, 10000], ["1,000 km", "2500 km", "5,000 km", "10,000 km"], rotation=90, va='center')  # Rotate y-tick labels for better readability

        # add grid
        plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        plt.title(f"{dataset_name} dataset")
        plt.legend(fontsize="x-large", markerscale=2)
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            suffix = '_'.join(attack_types)
            plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{suffix}_{metric}.png"))
            save_plot_json(plot_dir, f"{dataset_name}_{suffix}_{metric}", json_payload)
        else:
            plt.show()
        plt.close()


def plot_attack_success_rate(
    results_dir,
    attack_budgets,
    plot_dir,
    dataset_name,
    threshold_km: Any = 2500,
    attack_types=None,
    attack_type=None,
    all_results=None,
    results=None,
    gps_true: bool = False,
):
    """
    Takes same input as plot_results, but plots attack success rate instead of metrics. Attack success is defined as the fraction of samples for which the final step displacement is above a certain threshold (e.g. 100km), which indicates a successful attack that significantly changes the predicted location.

    """
    if attack_types is None:
        attack_types = [attack_type] if attack_type is not None else []
    if all_results is None:
        if results is not None:
            all_results = {attack_types[0]: results}
        elif results_dir is not None:
            all_results = _load_attack_results(results_dir, dataset_name, attack_types)
    if all_results is None:
        raise ValueError("No results available to plot. Provide results/all_results or a valid results_dir.")

    if isinstance(threshold_km, (int, float, np.integer, np.floating)):
        thresholds = [float(threshold_km)]
    else:
        thresholds = np.asarray(threshold_km, dtype=np.float64).reshape(-1).tolist()
    if len(thresholds) == 0:
        raise ValueError("threshold_km list cannot be empty")

    # Ensure higher thresholds are rendered darker, regardless of input order.
    thresholds = sorted(set(thresholds))

    linewidths = np.linspace(0.5, 3, len(thresholds))

    def _darken_rgba(color_rgba, factor):
        import matplotlib.colors as mcolors
        rgb = np.asarray(mcolors.to_rgb(color_rgba))
        rgb = np.clip(rgb * factor, 0.0, 1.0)
        return (rgb[0], rgb[1], rgb[2], 1.0)

    plt.figure(figsize=(10, 8))
    # Use fixed qualitative colors so the first two attacks are clearly distinct (blue, red).
    attack_palette = [
        '#9467bd',  # purple
        '#ff7f0e',  # orange
        '#d62728',  # red
        '#2ca02c',  # green
        '#9467bd',  # purple
        '#8c564b',  # brown
    ]

    metric_name = _select_displacement_metric(gps_true)
    json_payload: dict[str, Any] = {
        "dataset": dataset_name,
        "metric": metric_name,
        "thresholds_km": [float(t) for t in thresholds],
        "attack_budgets": [float(b) for b in attack_budgets],
        "attacks": {},
    }

    for attack_index, (at, res) in enumerate(all_results.items()):
        base_color = attack_palette[attack_index % len(attack_palette)]
        budget_samples = _get_metric_samples_by_budget(res, metric_name)
        if budget_samples is None:
            continue

        # filter out nan values:
        for i, samples in enumerate(budget_samples):
            l_before = len(samples)
            samples = samples[~np.isnan(samples)]
            nb_dropped = l_before - samples.size
            if nb_dropped > 0:
                print(f"Budget {attack_budgets[i]:.3f}, attack {at}: Dropped {nb_dropped} samples out of {l_before} total samples for success rate computation")
            budget_samples[i] = samples

        success_curves = []
        for t in thresholds:
            success_rate = np.asarray([
                float((samples > t).mean()) if samples.size > 0 else np.nan
                for samples in budget_samples
            ])
            success_curves.append(success_rate)

        success_curves = np.asarray(success_curves)

        print(f"Attack {at}: Success rates at thresholds {thresholds} are:\n{success_curves}")

        json_payload["attacks"][at] = {
            "success_rate": {
                str(t): success_curves[threshold_index].tolist()
                for threshold_index, t in enumerate(thresholds)
            },
            "n_samples_per_budget": [int(samples.size) for samples in budget_samples],
        }

        if success_curves.shape[0] > 1:
            low_curve = np.min(success_curves, axis=0)
            high_curve = np.max(success_curves, axis=0)
            plt.fill_between(
                attack_budgets,
                low_curve,
                high_curve,
                color=base_color,
                alpha=0.30,
                linewidth=0,
            )

        n_thresholds = len(thresholds)
        threshold_linestyles = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1, 1, 1))]
        for threshold_index, t in enumerate(thresholds):
            if n_thresholds == 1:
                shade_factor = 0.85
            else:
                # Highest threshold gets the darkest shade.
                shade_factor = 1.0 - 0.55 * (threshold_index / (n_thresholds - 1))
            line_color = _darken_rgba(base_color, shade_factor)
            linestyle = threshold_linestyles[threshold_index % len(threshold_linestyles)]

            attack_name = _display_attack_name(at)

            plt.plot(
                attack_budgets,
                success_curves[threshold_index],
                color=line_color,
                linewidth=linewidths[threshold_index],
                linestyle=linestyle,
                alpha=0.95,
                label=f"{attack_name} > {t:.0f} km",
            )

    plt.xlabel("Attack budget out of 255 (log scale))")
    # if len(thresholds) == 1:
    #     plt.ylabel(f"Attack Success Rate (disp > {thresholds[0]:.0f} km)")
    # else:
    #     plt.ylabel("Attack Success Rate")
    plt.title(f"Attack Success Rate on {dataset_name} dataset")
    plt.ylim(0.4, 1.0)
    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    plt.legend(ncol=2, fontsize='x-large')
    plt.xscale("log")
    ax = plt.gca()
    tick_labels = [f"{eps*255:.0f}" for eps in attack_budgets]
    ax.xaxis.set_major_locator(FixedLocator(attack_budgets))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    ax.xaxis.set_minor_locator(NullLocator())

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        suffix = '_'.join(attack_types)
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{suffix}_attack_success_rate.png"))
        save_plot_json(plot_dir, f"{dataset_name}_{suffix}_attack_success_rate", json_payload)
    else:
        plt.show()
    plt.close()
