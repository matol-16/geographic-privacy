"""Ablation and breakdown plots: restarts, sampling steps, localizability, transferability."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.patches import Patch

from utils.adversarial_metrics import trajectory_displacement
from utils.plots.common import (
    _display_attack_name,
    _get_metric_samples_by_budget,
    _get_metric_tensor,
    _load_attack_results,
    _select_displacement_metric,
    save_plot_json,
    select_closest_budget,
)
from utils.plots.style import apply_paper_style, attack_color

apply_paper_style()

# Paper-ready figures: high-dpi raster + vector PDF side by side, saved together.
_FIGURE_DPI = 300
# The ablation figures (robustness, model-transfer, sampling-steps, DTD-variance)
# fix the attack budget to this value by default so each panel shows one line/box
# per attack instead of one per (attack, budget) pair.
DEFAULT_ABLATION_EPS = 0.0314


def _savefig(fig, plot_dir: str, filename_stem: str) -> None:
    """Save ``fig`` as both PNG (high-dpi raster) and PDF (vector, paper quality)."""
    os.makedirs(plot_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(plot_dir, f"{filename_stem}.{ext}")
        fig.savefig(path, dpi=_FIGURE_DPI, bbox_inches="tight")
        print(f"Plot saved to: {path}")


def _finalize_figure(fig, handles=None, labels=None, ncol=None) -> None:
    """Place one legend shared by every subplot, anchored above the axes.

    Centralising the legend (instead of repeating it per panel) guarantees it never
    lands on top of plotted data -- the standard layout for multi-panel ablation
    figures in CV papers. Titles are omitted here: captions are written in LaTeX.
    """
    handles = handles or []
    labels = labels or []
    fig_height_in = fig.get_size_inches()[1]
    if handles:
        ncol = ncol or min(len(labels), 4)
        legend_y = 1.0 + 0.14 / fig_height_in
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, legend_y),
                   ncol=ncol, frameon=False)


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
        ax.set_title(f"{attack_name}")
        # ax.set_xlabel(f"Budget = {selected_budget * 255:.0f}/255")
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(bucket_names)
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
        #crop y axis at 10^-1
        ax.set_ylim(bottom=10**-1)
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
        ds_name = "FSD (YFCC4K)" if ds == "yfcc" else "FSD (OSV-5M)"

        for c, attack in enumerate(attacks):
            ax = axes[r, c]
            vals = _fill_ax(ax, res, attack, row_label=ds_name if c == 0 else None)
            all_values.append(vals)

    flat = np.concatenate([v for v in all_values if v.size > 0]) if any(v.size > 0 for v in all_values) else np.array([])
    if flat.size > 0 and np.nanmax(flat) >= y_tick_values[0]:
        for r in range(n_rows):
            axes[r, 0].set_yticks(y_tick_values)
            axes[r, 0].set_yticklabels(y_tick_labels)
            axes[r, 0].set_yscale('log')
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor=colors[i], alpha=0.5, label=f"{bucket_names[i]} localizability")
        for i in range(3)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.52, 1.04))
    fig.tight_layout()

    if plot_dir is not None:
        _savefig(fig, plot_dir, f"localizability_budget_{selected_budget * 255:.0f}")
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
                # ax.scatter(xs, disps, color=color, alpha=0.4, s=30, zorder=2)
                ax.plot(range(1, len(best_k) + 1), best_k, color=color, label=label, linewidth=2, zorder=3)
                color_idx += 1
        ax.set_xlabel("Number of restarts")
        ax.set_ylabel("FSD (km)")
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
    eps: float = DEFAULT_ABLATION_EPS,
) -> None:
    """
    Plot attack success rate vs. number of sampling steps.

    One subplot per distance threshold; one line per attack, all pinned to the
    single attack budget closest to ``eps`` (default 8/255) so each panel isn't
    cluttered with one line per (attack, budget) pair.
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
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5.5 * n_thresholds, 4.5), squeeze=False)

    legend_handles, legend_labels = [], []
    for attack_type in attack_types:
        budget = select_closest_budget(budgets_per_type.get(attack_type, attack_budgets), eps)
        bkey = f"budget_{budget:.6f}"
        color = attack_color(attack_type)
        line = None
        for ax_idx, thr in enumerate(thresholds):
            ax = axes[0][ax_idx]
            rates = [
                json_results["results"][attack_type][bkey][str(ns)]["success_rates"][str(thr)]
                for ns in eval_num_steps
            ]
            line, = ax.plot(eval_num_steps, rates, marker="o", color=color)
        legend_handles.append(line)
        legend_labels.append(_display_attack_name(attack_type))

    for ax_idx, thr in enumerate(thresholds):
        ax = axes[0][ax_idx]
        ax.set_xlabel("Sampling steps")
        ax.set_ylabel("Attack success rate")
        ax.set_title(f"Displacement > {thr} km")
        ax.set_ylim(0.3, 1)
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, legend_handles, legend_labels)
    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset}_sampling_steps_success_rate")
    plt.close(fig)


def plot_cfg_success_rate(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> None:
    """
    Plot attack success rate vs. the classifier-free guidance scale (cfg).

    One subplot per distance threshold; one line per attack, all pinned to the
    single attack budget closest to ``eps`` (default 8/255) so each panel isn't
    cluttered with one line per (attack, budget) pair. ``json_results`` is the dict
    produced by ``build_cfg_json``.
    """
    thresholds = json_results["success_rate_thresholds_km"]
    eval_cfgs = json_results["eval_cfgs"]
    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    # Per-type budget lists are set when merging results from different commands.
    budgets_per_type = json_results.get("attack_budgets_per_type", {})

    def _cfg_key(cfg: float) -> str:
        return f"{float(cfg):g}"

    n_thresholds = len(thresholds)
    fig, axes = plt.subplots(1, n_thresholds, figsize=(5.5 * n_thresholds, 4.5), squeeze=False)

    legend_handles, legend_labels = [], []
    for attack_type in attack_types:
        budget = select_closest_budget(budgets_per_type.get(attack_type, attack_budgets), eps)
        bkey = f"budget_{budget:.6f}"
        color = attack_color(attack_type)
        line = None
        for ax_idx, thr in enumerate(thresholds):
            ax = axes[0][ax_idx]
            rates = [
                json_results["results"][attack_type][bkey][_cfg_key(c)]["success_rates"][str(thr)]
                for c in eval_cfgs
            ]
            line, = ax.plot(eval_cfgs, rates, marker="o", color=color)
        legend_handles.append(line)
        legend_labels.append(_display_attack_name(attack_type))

    for ax_idx, thr in enumerate(thresholds):
        ax = axes[0][ax_idx]
        ax.set_xlabel("Guidance scale (cfg)")
        ax.set_ylabel("Attack success rate")
        ax.set_title(f"Displacement > {thr} km")
        ax.set_ylim(0.3, 1)
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, legend_handles, legend_labels)
    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset}_cfg_success_rate")
    plt.close(fig)


def plot_model_transfer_success_rate(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> None:
    """Plot cross-model transfer: attack success rate against each generative backbone.

    The attacked image is trained against one PLONK variant and re-evaluated against
    every configured variant (diffusion / flow / RFM). One subplot per distance
    threshold; one line per attack, all pinned to the single attack budget closest to
    ``eps`` (default 8/255). The variant the attack was trained on is annotated in the
    title. ``json_results`` is the dict produced by ``build_model_transfer_json``.
    """
    model_labels = json_results["model_labels"]
    #models labels in capital letters:
    model_labels_capital = [m.upper() for m in model_labels]
    thresholds = json_results["success_rate_thresholds_km"]
    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    attacked_model = json_results.get("attacked_model_label")
    # Per-type budget lists are set when merging results from different commands.
    budgets_per_type = json_results.get("attack_budgets_per_type", {})

    x = list(range(len(model_labels)))
    n_cols = len(thresholds)
    fig, axes = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4.5), squeeze=False)

    legend_handles, legend_labels = [], []
    for attack_type in attack_types:
        budget = select_closest_budget(budgets_per_type.get(attack_type, attack_budgets), eps)
        bkey = f"budget_{budget:.6f}"
        color = attack_color(attack_type)
        cells = [json_results["results"][attack_type][bkey][m] for m in model_labels]
        line = None
        for col_idx, thr in enumerate(thresholds):
            rates = [c["success_rates"][str(thr)] for c in cells]
            line, = axes[0][col_idx].plot(x, rates, marker="o", color=color)
        legend_handles.append(line)
        legend_labels.append(_display_attack_name(attack_type))
    axes[0][0].set_ylabel("Attack success rate")

    for col_idx, thr in enumerate(thresholds):
        axes[0][col_idx].set_title(f"Displacement > {thr} km")
        axes[0][col_idx].set_ylim(0, 1)
    for col_idx in range(n_cols):
        ax = axes[0][col_idx]
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels_capital, rotation=30, ha="right")
        ax.grid(True, alpha=0.3)

    _finalize_figure(fig, legend_handles, legend_labels)
    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset}_model_transfer_success_rate")
    plt.close(fig)


def plot_robustness_results(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> None:
    """Plot attack robustness to JPEG compression and Gaussian blur (GeoShield Fig. 6).

    Two columns (JPEG quality factor | Gaussian blur sigma), one row per distance
    threshold, showing attack success rate only. One coloured line per attack, all
    pinned to the single attack budget closest to ``eps`` (default 8/255); the
    displacement-from-clean-prediction metric is drawn solid ("predicted") and the
    displacement-from-ground-truth metric dashed ("true", the GeoShield metric).
    ``json_results`` is the dict produced by ``build_robustness_json``.
    """
    from matplotlib.lines import Line2D

    attack_types = json_results["attack_types"]
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    thresholds = [t for t in json_results["success_rate_thresholds_km"] if float(t) == 2500.0]
    num_steps = json_results.get("num_steps")
    metrics = json_results.get("metrics", ["true"])
    metric_styles = {"true": "-"}
    budget = select_closest_budget(attack_budgets, eps)
    bkey = f"budget_{budget:.6f}"

    # (x-axis label, JSON sub-dict key)
    transforms = [
        ("JPEG quality factor", "jpeg"),
        ("Gaussian blur σ", "blur"),
    ]

    n_rows = len(thresholds)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.5 * 2, 4 * n_rows), squeeze=False)

    def _sorted_levels(level_dict: dict):
        items = sorted(level_dict.items(), key=lambda kv: float(kv[0]))
        return [float(k) for k, _ in items], [v for _, v in items]

    drawn_metrics: set = set()
    legend_handles, legend_labels = [], []
    for col_idx, (xlabel, json_key) in enumerate(transforms):
        for attack_type in attack_types:
            level_dict = json_results["results"][attack_type][bkey][json_key]
            if not level_dict:
                continue
            xs, entries = _sorted_levels(level_dict)
            color = attack_color(attack_type)
            line = None
            for metric in metrics:
                style = metric_styles.get(metric, "-")
                any_finite = any(
                    not np.isnan(entry[metric]["success_rates"][str(thr)])
                    for entry in entries for thr in thresholds
                )
                if not any_finite:
                    continue  # e.g. "true" unavailable when the dataset has no GPS
                drawn_metrics.add(metric)
                for row_idx, thr in enumerate(thresholds):
                    rates = [entry[metric]["success_rates"][str(thr)] for entry in entries]
                    line, = axes[row_idx][col_idx].plot(xs, rates, marker="o", linestyle=style, color=color)
            if col_idx == 0 and line is not None:
                legend_handles.append(line)
                legend_labels.append(_display_attack_name(attack_type))

        for row_idx, thr in enumerate(thresholds):
            ax = axes[row_idx][col_idx]
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Attack success rate")
            if len(thresholds) > 1:
                ax.set_title(f"Displacement > {thr} km")
            ax.set_ylim(0.3, 1)
            ax.grid(True, alpha=0.3)
            if json_key == "jpeg":
                # Lower quality factor = more compression; invert so both columns
                # read left-to-right as "less" -> "more" transform strength.
                ax.invert_xaxis()

    # Linestyle legend (predicted vs true) alongside the attack-colour legend, both
    # centered above the whole figure so neither can land on top of plotted data.
    style_handles = [
        Line2D([0], [0], color="black", linestyle=metric_styles.get(m, "-"),
               label={"predicted": "predicted (vs clean)", "true": "true (vs GT)"}.get(m, m))
        for m in metrics if m in drawn_metrics
    ]
    style_labels = [h.get_label() for h in style_handles]

    # _finalize_figure(fig, legend_handles + style_handles, legend_labels + style_labels)
    _finalize_figure(fig, legend_handles, legend_labels + style_labels)
    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset}_robustness_results")
    plt.close(fig)


def plot_attack_dtd_variance(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=None,
    all_results=None,
    gps_true: bool = True,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """Box plot of each attack's spread in final-step displacement ("DTD") across images.

    Where the other plots show *mean* attack strength, this one shows how *consistent*
    each attack is: for every attack, at the single budget closest to ``eps`` (default
    8/255), the per-image final-step displacement metric (km) — the core
    geolocation-deviation quantity used throughout this evaluation, which we refer to
    as "DTD" following the paper's naming — is summarized as a box (median, IQR,
    whiskers) plus mean/std/variance in the JSON sidecar. Attacks with a tall box move
    some images far and others barely at all; attacks with a short box are uniformly
    (in)effective.

    Args:
        results_dir/attack_budgets/dataset_name/attack_types: same as ``plot_results``.
        all_results: optional pre-loaded ``{attack_type: results_dict}`` to skip disk I/O.
        gps_true: use the true-GPS displacement metric instead of the vs-clean-prediction one.
        eps: attack budget to plot (the closest available budget is used).
    """
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)

    metric_name = _select_displacement_metric(gps_true)
    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    box_data: list[np.ndarray] = []
    box_colors: list[Any] = []
    tick_labels: list[str] = []
    json_payload: dict[str, Any] = {
        "dataset": dataset_name,
        "metric": metric_name,
        "budget": float(budget),
        "attacks": {},
    }

    for attack_type in attack_types:
        res = all_results.get(attack_type)
        if res is None:
            continue
        budget_samples = _get_metric_samples_by_budget(res, metric_name)
        if budget_samples is None:
            continue

        samples = budget_samples[budget_idx] if budget_idx < len(budget_samples) else np.array([])
        samples = np.asarray(samples, dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        box_data.append(samples)
        box_colors.append(attack_color(attack_type))
        tick_labels.append(_display_attack_name(attack_type))
        json_payload["attacks"][attack_type] = {
            "mean_km": float(np.mean(samples)) if samples.size else None,
            "std_km": float(np.std(samples)) if samples.size else None,
            "variance_km2": float(np.var(samples)) if samples.size else None,
            "median_km": float(np.median(samples)) if samples.size else None,
            "n_samples": int(samples.size),
        }

    if not box_data:
        raise ValueError("No attack results available to plot DTD variance for.")

    fig, ax = plt.subplots(figsize=(max(8.0, 1.3 * len(tick_labels)), 5.5))
    bp = ax.boxplot(
        box_data,
        positions=list(range(len(box_data))),
        widths=0.6,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(color)
        patch.set_alpha(0.65)

    ax.set_xticks(range(len(tick_labels)))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right")
    ax.set_ylabel("Final-step displacement — \"DTD\" (km)")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset_name}_attack_dtd_variance")
    plt.close(fig)

    save_plot_json(plot_dir, f"{dataset_name}_attack_dtd_variance", json_payload)


def plot_loss_vs_fsd(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=("dtd", "training_loss", "sampling"),
    all_results=None,
    gps_true: bool = True,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """Scatter each attack's optimization loss against its achieved FSD, per image.

    Relates how well an attack drove down its own training objective to how much
    geolocation displacement that actually bought. Only meaningful for loss-based
    attacks (dtd, training_loss/"AdvDM", sampling) — their loss functions are on
    incompatible scales/units (cosine similarity, MSE-in-velocity-space, negative
    km respectively), so each attack gets its own panel (shared FSD y-axis, free
    x-axis) rather than one shared scatter; overlaying raw loss values across
    attacks would be comparing unrelated units.

    Args:
        results_dir/attack_budgets/dataset_name/attack_types: same as ``plot_results``.
        all_results: optional pre-loaded ``{attack_type: results_dict}`` to skip disk I/O.
        gps_true: use the true-GPS displacement metric instead of the vs-clean-prediction one.
        eps: attack budget to plot (the closest available budget is used).
    """
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)

    metric_name = _select_displacement_metric(gps_true)
    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    panels: list[tuple[str, np.ndarray, np.ndarray]] = []
    json_payload: dict[str, Any] = {
        "dataset": dataset_name,
        "metric": metric_name,
        "budget": float(budget),
        "attacks": {},
    }

    for attack_type in attack_types:
        res = all_results.get(attack_type)
        if res is None:
            continue
        loss_samples = _get_metric_samples_by_budget(res, "final_loss")
        fsd_samples = _get_metric_samples_by_budget(res, metric_name)
        if loss_samples is None or fsd_samples is None:
            continue

        loss = np.asarray(loss_samples[budget_idx], dtype=np.float64) if budget_idx < len(loss_samples) else np.array([])
        fsd = np.asarray(fsd_samples[budget_idx], dtype=np.float64) if budget_idx < len(fsd_samples) else np.array([])
        n = min(loss.size, fsd.size)
        loss, fsd = loss[:n], fsd[:n]
        valid = np.isfinite(loss) & np.isfinite(fsd)
        loss, fsd = loss[valid], fsd[valid]
        if loss.size == 0:
            continue

        pearson_r = float(np.corrcoef(loss, fsd)[0, 1]) if loss.size > 1 else None
        json_payload["attacks"][attack_type] = {
            "loss": loss.tolist(),
            "fsd_km": fsd.tolist(),
            "pearson_r": pearson_r,
            "n_samples": int(loss.size),
        }
        panels.append((attack_type, loss, fsd))

    if not panels:
        raise ValueError(
            "No paired (loss, FSD) samples available — this requires results saved with "
            "the 'final_loss' field, which older runs may not have (re-run evaluate-dataset "
            "to backfill)."
        )

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.5), squeeze=False)
    axes = axes[0]

    for ax, (attack_type, loss, fsd) in zip(axes, panels):
        color = attack_color(attack_type)
        ax.scatter(loss, fsd, s=14, color=color, alpha=0.5, edgecolors="none")
        ax.set_xlabel("Final training loss")
        ax.set_yscale("log")
        r = json_payload["attacks"][attack_type]["pearson_r"]
        r_label = f"r = {r:.2f}" if r is not None else "r = n/a"
        ax.set_title(f"{_display_attack_name(attack_type)} ({r_label})")
        ax.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Final-step displacement (km, log scale)")

    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset_name}_loss_vs_fsd")
    plt.close(fig)

    save_plot_json(plot_dir, f"{dataset_name}_loss_vs_fsd", json_payload)


def _get_clean_vs_attacked_true_samples(res, budget_idx):
    """Per-image (clean-prediction, perturbed-prediction) displacement to true GPS.

    The clean-vs-true distance isn't stored as a top-level tensor (only the
    perturbed-vs-true one is); it's recovered here from the raw ``gps_source``/
    ``true_gps`` coordinates kept in ``restart_results`` (averaged over restarts,
    since the clean prediction can vary slightly across restarts due to sampling
    stochasticity). All restart pairs across all images are batched into a single
    vectorized haversine call rather than one Python-level call per restart, since a
    full-dataset run can have tens of thousands of (image, restart) pairs per attack.
    Returns ``None`` if the results don't have restart-level data.
    """
    attacked_samples = _get_metric_samples_by_budget(res, "final_step_displacement_true")
    restart_results = res.get("restart_results")
    if attacked_samples is None or restart_results is None or budget_idx >= len(restart_results):
        return None

    attacked = np.asarray(attacked_samples[budget_idx], dtype=np.float64)

    budget_restarts = restart_results[budget_idx]
    image_idx_list = []
    gps_source_list = []
    true_gps_list = []
    for img_idx, image_results in enumerate(budget_restarts):
        for restart_result in (image_results or []):
            gps_source = restart_result.get("gps_source")
            true_gps = restart_result.get("true_gps")
            if gps_source is None or true_gps is None:
                continue
            image_idx_list.append(img_idx)
            gps_source_list.append(gps_source.detach().cpu().float().reshape(2))
            true_gps_list.append(torch.as_tensor(true_gps, dtype=torch.float32).detach().cpu().reshape(2))

    n_images = len(budget_restarts)
    baseline_sum = np.zeros(n_images, dtype=np.float64)
    baseline_count = np.zeros(n_images, dtype=np.int64)

    if gps_source_list:
        gps_source_batch = torch.stack(gps_source_list).unsqueeze(0)  # (1, N, 2)
        true_gps_batch = torch.stack(true_gps_list).unsqueeze(0)  # (1, N, 2)
        dists = trajectory_displacement(true_gps_batch, gps_source_batch)[0].numpy()  # (N,)
        image_idx_arr = np.asarray(image_idx_list)
        np.add.at(baseline_sum, image_idx_arr, dists)
        np.add.at(baseline_count, image_idx_arr, 1)

    with np.errstate(invalid="ignore"):
        baseline = np.where(baseline_count > 0, baseline_sum / np.maximum(baseline_count, 1), np.nan)

    n = min(baseline.size, attacked.size)
    return baseline[:n], attacked[:n]


def plot_clean_vs_attacked_displacement(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=None,
    all_results=None,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """Scatter clean-prediction-vs-truth displacement against perturbed-vs-truth displacement.

    Both axes are the same haversine distance to the ground-truth GPS (km), so unlike
    ``plot_loss_vs_fsd`` this doesn't need per-attack facets: one shared log-log panel,
    one colour per attack, with a y=x reference line marking "attack had no effect"
    (points above it were pushed further from the truth than the clean prediction
    already was).

    Args:
        results_dir/attack_budgets/dataset_name/attack_types: same as ``plot_results``.
        all_results: optional pre-loaded ``{attack_type: results_dict}`` to skip disk I/O.
        eps: attack budget to plot (the closest available budget is used).
    """
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)

    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    fig, ax = plt.subplots(figsize=(7, 7))
    json_payload: dict[str, Any] = {
        "dataset": dataset_name,
        "budget": float(budget),
        "attacks": {},
    }

    all_vals = []
    for attack_type in attack_types:
        res = all_results.get(attack_type)
        if res is None:
            continue
        pairs = _get_clean_vs_attacked_true_samples(res, budget_idx)
        if pairs is None:
            continue
        baseline, attacked = pairs
        valid = np.isfinite(baseline) & np.isfinite(attacked)
        baseline, attacked = baseline[valid], attacked[valid]
        if baseline.size == 0:
            continue

        color = attack_color(attack_type)
        ax.scatter(baseline, attacked, s=14, color=color, alpha=0.5, edgecolors="none",
                   label=_display_attack_name(attack_type))

        pearson_r = float(np.corrcoef(baseline, attacked)[0, 1]) if baseline.size > 1 else None
        json_payload["attacks"][attack_type] = {
            "clean_vs_true_km": baseline.tolist(),
            "attacked_vs_true_km": attacked.tolist(),
            "pearson_r": pearson_r,
            "n_samples": int(baseline.size),
        }
        all_vals.append(baseline)
        all_vals.append(attacked)

    if not all_vals:
        raise ValueError(
            "No paired (clean, attacked) true-GPS displacement samples available — this "
            "requires results saved with restart-level 'gps_source'/'true_gps' data."
        )

    combined = np.concatenate(all_vals)
    positive = combined[combined > 0]
    lo = float(positive.min()) if positive.size else 1.0
    hi = float(combined.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1, alpha=0.6,
            label="y = x (no attack effect)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Clean prediction FSD to true GPS (km)")
    ax.set_ylabel("Perturbed prediction FSD to true GPS (km)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", frameon=True, framealpha=0.85)

    fig.tight_layout()
    _savefig(fig, plot_dir, f"{dataset_name}_clean_vs_attacked_displacement")
    plt.close(fig)

    save_plot_json(plot_dir, f"{dataset_name}_clean_vs_attacked_displacement", json_payload)
