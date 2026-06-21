"""Ablation and breakdown plots: restarts, sampling steps, localizability, transferability."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.patches import Patch

from utils.plots.common import _display_attack_name, _get_metric_tensor


def plot_transferability_results(results_dir, attack_budgets, plot_dir, dataset_name, metric, results=None):
    # For each attack budget, plot a boxplot of the metric for each attack type, to
    # better show the distribution of the metric across samples and attacks (more
    # informative for transferability evaluation).
    if results is None and results_dir is not None:
        results = torch.load(os.path.join(results_dir, f"{dataset_name}_results_transferability.pt"))
    data_to_plot = []
    attack_labels = []
    for attack, res in results.items():
        for i, budget in enumerate(attack_budgets):
            data_to_plot.append(res[i].cpu().numpy())
            attack_labels.append(f"{attack} (eps={budget:.3f})")
    plt.figure(figsize=(12, 6))
    plt.boxplot(data_to_plot, showfliers=False)
    plt.xticks(range(1, len(attack_labels) + 1), attack_labels, rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f"Attack transferability evaluation on {dataset_name} dataset")
    plt.tight_layout()

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_transferability_boxplot.png"))
    else:
        plt.show()


def plot_localizability_results(attack_budgets, plot_dir, all_datasets_results, results_attack_budgets):
    """Produces a 2×2 grid: rows = datasets, columns = attacks.

    Args:
        attack_budgets: single float (or length-1 iterable) — the budget to plot.
        plot_dir: directory to save the figure, or None to show interactively.
        all_datasets_results: dict[dataset_name -> results] with two entries.
        results_attack_budgets: list of budgets used when running evaluate_localizability,
            used to map ``attack_budgets`` to the correct row index in the results tensors.
    """

    if isinstance(attack_budgets, (int, float, np.integer, np.floating)):
        selected_budget = float(attack_budgets)
    else:
        budgets_arr = np.asarray(attack_budgets, dtype=np.float64).reshape(-1)
        if budgets_arr.size != 1:
            raise ValueError("plot_localizability_results expects a single budget (float)")
        selected_budget = float(budgets_arr[0])

    budget_arr = np.asarray(results_attack_budgets, dtype=np.float64).reshape(-1)
    budget_idx = int(np.argmin(np.abs(budget_arr - selected_budget)))
    if not np.isclose(budget_arr[budget_idx], selected_budget, rtol=1e-4):
        raise ValueError(
            f"Budget {selected_budget:.6f} not found in results_attack_budgets "
            f"{[f'{b:.6f}' for b in budget_arr]}. Closest is {budget_arr[budget_idx]:.6f}."
        )

    bucket_names = ["Low", "Medium", "High"]
    colors = colormaps['Set1'](np.linspace(0, 1, 3))
    y_tick_values = [1000, 2500, 5000, 10000]
    y_tick_labels = ["1,000 km", "2,500 km", "5,000 km", "10,000 km"]

    def _get_buckets(res):
        loc = res["localizability"].detach().cpu()
        low_t = torch.quantile(loc, 0.33)
        high_t = torch.quantile(loc, 0.66)
        b = torch.zeros_like(loc, dtype=torch.long)
        b[(loc > low_t) & (loc <= high_t)] = 1
        b[loc > high_t] = 2
        return b

    def _get_attack_strength(res, attack):
        attack_result = res["attack_results"][attack]
        if isinstance(attack_result, dict):
            t = _get_metric_tensor(attack_result, "final_step_displacement_predicted")
            if t is None:
                t = _get_metric_tensor(attack_result, "final_step_displacement")
            if t is None:
                tensor_values = [value for value in attack_result.values() if isinstance(value, torch.Tensor)]
                if not tensor_values:
                    raise ValueError(f"No tensor-valued attack results found for attack '{attack}'")
                t = tensor_values[0]
        else:
            t = attack_result
        t = t.detach().cpu()
        if t.ndim == 1:
            return t
        if t.ndim == 2:
            return t[budget_idx]
        raise ValueError(f"Unexpected tensor shape {tuple(t.shape)} for attack '{attack}'")

    def _fill_ax(ax, res, attack, row_label=None):
        buckets = _get_buckets(res)
        strength = _get_attack_strength(res, attack)
        data = [strength[buckets == k].numpy() for k in range(3)]
        box = ax.boxplot(
            data,
            positions=np.arange(3),
            widths=0.9,
            showfliers=False,
            patch_artist=True,
            medianprops=dict(color='black', linewidth=1.5),
        )
        for i, patch in enumerate(box['boxes']):
            patch.set_facecolor(colors[i])
            patch.set_edgecolor(colors[i])
            patch.set_alpha(0.5)
        attack_name = _display_attack_name(attack)
        ax.set_title(f"{attack_name} attack")
        # ax.set_xlabel(f"Budget = {selected_budget * 255:.0f}/255")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(bucket_names)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        if row_label is not None:
            ax.set_ylabel(f"{row_label}")
        return np.concatenate([v for v in data if len(v) > 0]) if any(len(v) > 0 for v in data) else np.array([])

    dataset_names = list(all_datasets_results.keys())
    attacks = list(next(iter(all_datasets_results.values()))["attack_results"].keys())
    n_rows = len(dataset_names)
    n_cols = len(attacks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharey='row')
    axes = np.atleast_2d(axes)

    all_values = []
    for r, ds in enumerate(dataset_names):
        res = all_datasets_results[ds]
        ds_name = "YFCC4K" if ds == "yfcc" else "OSV-5M"

        for c, attack in enumerate(attacks):
            ax = axes[r, c]
            vals = _fill_ax(ax, res, attack, row_label=ds_name if c == 0 else None)
            all_values.append(vals)

    flat = np.concatenate([v for v in all_values if v.size > 0]) if any(v.size > 0 for v in all_values) else np.array([])
    if flat.size > 0 and np.nanmax(flat) >= y_tick_values[0]:
        for r in range(n_rows):
            axes[r, 0].set_yticks(y_tick_values)
            axes[r, 0].set_yticklabels(y_tick_labels, rotation=90, va='center')
            axes[r, 0].set_yscale('log')
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor=colors[i], alpha=0.5, label=f"{bucket_names[i]} localizability")
        for i in range(3)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.04))
    # fig.suptitle(f"Attack strength vs localizability — budget = {selected_budget * 255:.0f}/255", y=1.05)
    fig.tight_layout()

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(
            os.path.join(plot_dir, f"localizability_budget_{selected_budget * 255:.0f}.png"),
            bbox_inches='tight',
        )
    else:
        plt.show()
    plt.close(fig)


def plot_restarts_success(
    json_results: dict,
    plot_dir: str,
    per_image: bool = True,
) -> None:
    """
    Plot best displacement vs. number of restarts.

    For each image (when ``per_image``): scatter of individual restart displacements +
    best-so-far line, one line per (attack_type, budget) combination.
    When n_images > 1, also produces a summary plot with mean ± std across images.

    Set ``per_image=False`` for full-dataset runs (e.g. the restart ablation folded
    into evaluate-dataset) to skip the per-image figures and keep only the summary.
    """
    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    max_restarts = json_results["max_restarts"]
    n_images = json_results["n_images"]
    image_ids = json_results.get("image_ids", [str(i) for i in range(n_images)])
    colors = plt.cm.tab10.colors

    os.makedirs(plot_dir, exist_ok=True)

    # When per-image plots are disabled and there is no summary (single image),
    # still emit the one image so the call produces a figure.
    per_image_indices = range(n_images) if (per_image or n_images == 1) else range(0)
    for img_idx in per_image_indices:
        fig, ax = plt.subplots(figsize=(7, 5))
        color_idx = 0
        for attack_type in attack_types:
            for budget_idx, budget in enumerate(attack_budgets):
                bkey = f"budget_{budget:.6f}"
                ikey = f"image_{img_idx}"
                color = colors[color_idx % len(colors)]
                label = f"{attack_type} eps={budget:.3f}"
                disps = json_results["results"][attack_type][bkey][ikey]["restart_displacements"]
                best_k = json_results["results"][attack_type][bkey][ikey]["best_after_k"]
                xs = list(range(1, len(disps) + 1))
                ax.scatter(xs, disps, color=color, alpha=0.4, s=30, zorder=2)
                ax.plot(range(1, len(best_k) + 1), best_k, color=color, label=label, linewidth=2, zorder=3)
                color_idx += 1
        ax.set_xlabel("Number of restarts")
        ax.set_ylabel("Displacement (km)")
        img_label = image_ids[img_idx] if img_idx < len(image_ids) else str(img_idx)
        ax.set_title(f"Best displacement vs. restarts — {dataset.upper()} — image {img_label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        suffix = f"_image_{img_idx}" if n_images > 1 else ""
        path = os.path.join(plot_dir, f"{dataset}_restarts_success{suffix}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot saved to: {path}")

    if n_images > 1:
        fig, ax = plt.subplots(figsize=(7, 5))
        color_idx = 0
        for attack_type in attack_types:
            for budget_idx, budget in enumerate(attack_budgets):
                bkey = f"budget_{budget:.6f}"
                color = colors[color_idx % len(colors)]
                label = f"{attack_type} eps={budget:.3f}"
                curves = np.array([
                    json_results["results"][attack_type][bkey][f"image_{i}"]["best_after_k"]
                    for i in range(n_images)
                ])
                mean = curves.mean(axis=0)
                std = curves.std(axis=0)
                xs = list(range(1, max_restarts + 1))
                ax.plot(xs, mean, color=color, label=label, linewidth=2)
                ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
                color_idx += 1
        ax.set_xlabel("Number of restarts")
        ax.set_ylabel("Displacement (km)")
        ax.set_title(f"Best displacement vs. restarts — {dataset.upper()} — {n_images} images (mean ± std)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(plot_dir, f"{dataset}_restarts_success_summary.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Summary plot saved to: {path}")


def plot_sampling_steps_success_rate(
    json_results: dict,
    plot_dir: str,
) -> None:
    """
    Plot attack success rate vs. number of sampling steps.

    One subplot per distance threshold; one line per (attack_type, budget) pair.
    json_results is the dict returned by evaluate_sampling_steps().
    """
    thresholds = json_results["success_rate_thresholds_km"]
    eval_num_steps = json_results["eval_num_steps"]
    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    # Per-type budget lists are set by merge_sampling_steps_results when combining
    # results from different commands (e.g. encoder vs GeoShield).
    budgets_per_type = json_results.get("attack_budgets_per_type", {})

    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(6 * n_thresholds, 5), squeeze=False)
    colors = plt.cm.tab10.colors

    color_idx = 0
    for attack_type in attack_types:
        budgets = budgets_per_type.get(attack_type, attack_budgets)
        for budget in budgets:
            bkey = f"budget_{budget:.6f}"
            label = f"{attack_type} eps={budget:.3f}"
            color = colors[color_idx % len(colors)]
            for ax_idx, thr in enumerate(thresholds):
                ax = axes[0][ax_idx]
                rates = [
                    json_results["results"][attack_type][bkey][str(ns)]["success_rates"][str(thr)]
                    for ns in eval_num_steps
                ]
                ax.plot(eval_num_steps, rates, marker="o", label=label, color=color)
            color_idx += 1

    for ax_idx, thr in enumerate(thresholds):
        ax = axes[0][ax_idx]
        ax.set_xlabel("Sampling steps")
        ax.set_ylabel("Attack success rate")
        ax.set_title(f"Displacement > {thr} km")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Attack success vs. sampling steps — {dataset.upper()}")
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{dataset}_sampling_steps_success_rate.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")


def plot_robustness_results(
    json_results: dict,
    plot_dir: str,
) -> None:
    """Plot attack robustness to JPEG compression and Gaussian blur (GeoShield Fig. 6).

    Two columns (JPEG quality factor | Gaussian blur sigma). The top row shows mean
    final-step displacement vs. transform strength; one row per distance threshold
    follows with the attack success rate. One coloured line per (attack_type, budget),
    with the displacement-from-clean-prediction metric drawn solid ("predicted") and
    the displacement-from-ground-truth metric dashed ("true", the GeoShield metric).
    ``json_results`` is the dict produced by ``build_robustness_json``.
    """
    from matplotlib.lines import Line2D

    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    thresholds = json_results["success_rate_thresholds_km"]
    num_steps = json_results.get("num_steps")
    metrics = json_results.get("metrics", ["predicted", "true"])
    metric_styles = {"predicted": "-", "true": "--"}

    # (x-axis label, JSON sub-dict key)
    transforms = [
        ("JPEG quality factor", "jpeg"),
        ("Gaussian blur σ", "blur"),
    ]
    colors = plt.cm.tab10.colors

    n_rows = 1 + len(thresholds)  # row 0: mean displacement; then one per threshold
    fig, axes = plt.subplots(n_rows, 2, figsize=(7 * 2, 4 * n_rows), squeeze=False)

    def _sorted_levels(level_dict: dict):
        items = sorted(level_dict.items(), key=lambda kv: float(kv[0]))
        return [float(k) for k, _ in items], [v for _, v in items]

    drawn_metrics: set = set()
    for col_idx, (xlabel, json_key) in enumerate(transforms):
        color_idx = 0
        for attack_type in attack_types:
            for budget in attack_budgets:
                bkey = f"budget_{budget:.6f}"
                level_dict = json_results["results"][attack_type][bkey][json_key]
                if not level_dict:
                    continue
                xs, entries = _sorted_levels(level_dict)
                color = colors[color_idx % len(colors)]
                label = f"{_display_attack_name(attack_type)} eps={budget:.3f}"
                for metric in metrics:
                    style = metric_styles.get(metric, "-")
                    means = [entry[metric]["mean_displacement_km"] for entry in entries]
                    if all(np.isnan(m) for m in means):
                        continue  # e.g. "true" unavailable when the dataset has no GPS
                    drawn_metrics.add(metric)
                    # Only the predicted (solid) line carries the colour legend label.
                    line_label = label if metric == "predicted" else None
                    axes[0][col_idx].plot(xs, means, marker="o", linestyle=style, label=line_label, color=color)
                    for row_offset, thr in enumerate(thresholds, start=1):
                        rates = [entry[metric]["success_rates"][str(thr)] for entry in entries]
                        axes[row_offset][col_idx].plot(xs, rates, marker="o", linestyle=style, color=color)
                color_idx += 1

        axes[0][col_idx].set_xlabel(xlabel)
        axes[0][col_idx].set_ylabel("Mean displacement (km)")
        axes[0][col_idx].set_title("Mean displacement")
        axes[0][col_idx].grid(True, alpha=0.3)
        axes[0][col_idx].legend(fontsize=7)
        for row_offset, thr in enumerate(thresholds, start=1):
            ax = axes[row_offset][col_idx]
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Attack success rate")
            ax.set_title(f"Displacement > {thr} km")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

    # Linestyle legend (predicted vs true) on the top-left axis, kept alongside the
    # colour/attack legend by re-adding the latter as a separate artist.
    style_handles = [
        Line2D([0], [0], color="black", linestyle=metric_styles.get(m, "-"),
               label={"predicted": "predicted (vs clean)", "true": "true (vs GT)"}.get(m, m))
        for m in metrics if m in drawn_metrics
    ]
    if style_handles:
        color_legend = axes[0][0].get_legend()
        axes[0][0].legend(handles=style_handles, fontsize=7, loc="lower left")
        if color_legend is not None:
            axes[0][0].add_artist(color_legend)

    steps_note = f" (eval steps = {num_steps})" if num_steps is not None else ""
    fig.suptitle(f"Robustness to JPEG / blur — {dataset.upper()}{steps_note}")
    fig.tight_layout()
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"{dataset}_robustness_results.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to: {plot_path}")
