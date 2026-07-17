"""TikZ/pgfplots twins of the matplotlib plots in ``utils.plots``.

Every function here reads exactly the same raw results (``.pt``/JSON files
under ``results_dir``) as its matplotlib counterpart -- reusing the same
data-extraction helpers from ``utils.plots.common``/``utils.plots.ablations``
so the numbers and colors are identical -- but emits a standalone pgfplots
``.tex`` source instead of a matplotlib figure (see ``utils.plots_tikz.common``
for the shared doc-wrapping/compile helpers).

World-map plots (``utils.plots.maps``) are not reproduced here: they draw
real coastlines/borders via cartopy, which has no straightforward TikZ
equivalent, and they are not part of the ``plot_yfcc4k_full*.sh`` pipeline
this module mirrors.
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np

from utils.plots.ablations import DEFAULT_ABLATION_EPS, _get_clean_vs_attacked_true_samples
from utils.plots.common import (
    _display_attack_name,
    _get_metric_samples_by_budget,
    _get_metric_tensor,
    _load_attack_results,
    _select_displacement_metric,
    _summarize_samples,
    select_closest_budget,
)
from utils.plots_tikz.common import (
    SINGLE_MARK_LEGEND_IMAGE,
    attack_line_width,
    attack_mark_size,
    attack_marker,
    attack_pgf_color,
    attacks_dtd_last,
    bar_legend_image_code,
    boxplot_prepared,
    boxplot_stats,
    color_definitions,
    contrasting_ink,
    coordinates,
    darken_hex,
    format_km_tick,
    groupplot_legend_center_x,
    nice_linear_ticks,
    render_tikz,
    sanitize_pgfname,
    sequential_colormap_def,
    tex_escape,
    write_dat_file,
)

_BUCKET_NAMES = ["Low", "Medium", "High"]

# Shared per-panel geometry for the single-row "success rate vs. X, one panel
# per threshold" groupplots (sampling-steps, model-transfer, robustness): same
# width/height/sep everywhere so fonts, markers, and line widths land at the
# same visual scale across those figures instead of each picking its own.
SUCCESS_RATE_PANEL_W = 7.0
SUCCESS_RATE_PANEL_H = 5.5
SUCCESS_RATE_H_SEP = 1.6


# --------------------------------------------------------------------------- #
# Main results: displacement + success rate vs. budget
# --------------------------------------------------------------------------- #


def plot_results_tikz(
    results_dir,
    attack_budgets,
    plot_dir,
    dataset_name,
    attack_types=None,
    all_results=None,
    stored_metrics=None,
    gps_true: bool = True,
):
    """TikZ twin of ``utils.plots.results.plot_results``."""
    if stored_metrics is None:
        stored_metrics = [_select_displacement_metric(gps_true)]
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)

    outputs = []
    for metric in stored_metrics:
        colors: dict[str, str] = {}
        blocks = []
        for at in attack_types:
            res = all_results.get(at)
            if res is None:
                continue
            budget_samples = _get_metric_samples_by_budget(res, metric)
            if budget_samples is None:
                continue

            summary = np.asarray([
                _summarize_samples(samples, attack_name=at, budget=attack_budgets[i])
                for i, samples in enumerate(budget_samples)
            ], dtype=np.float64)
            mean_metric, _median, q25_metric, q75_metric = summary.T

            color_name, hexcode = attack_pgf_color(at)
            colors[color_name] = hexcode
            q25_path, q75_path = f"q25{color_name}", f"q75{color_name}"
            blocks.append(
                f"\\addplot[name path={q25_path}, draw=none, forget plot] "
                f"{coordinates(attack_budgets, q25_metric)};\n"
                f"\\addplot[name path={q75_path}, draw=none, forget plot] "
                f"{coordinates(attack_budgets, q75_metric)};\n"
                f"\\addplot[{color_name}, opacity=0.25, forget plot] "
                f"fill between[of={q25_path} and {q75_path}];\n"
                f"\\addplot[{color_name}, dashed, thick, mark=none] "
                f"{coordinates(attack_budgets, mean_metric)};\n"
                f"\\addlegendentry{{{tex_escape(_display_attack_name(at))}}}"
            )
        if not blocks:
            continue

        tick_positions = ",".join(f"{b:.6g}" for b in attack_budgets)
        tick_labels = ",".join(f"{b * 255:.0f}" for b in attack_budgets)
        axis_opts = ", ".join([
            "xmode=log", "ymode=log",
            f"xtick={{{tick_positions}}}", f"xticklabels={{{tick_labels}}}",
            "minor tick num=0",
            "xlabel={Attack budget (out of 255)}",
            "ytick={1000,2500,5000,10000}",
            "yticklabels={1{,}000 km,2{,}500 km,5{,}000 km,10{,}000 km}",
            "grid=both", "grid style={dashed, gray!30}",
            "legend style={at={(1.02,1)}, anchor=north west, draw=none}",
            "width=12cm", "height=9cm",
        ])
        body = f"\\begin{{axis}}[{axis_opts}]\n" + "\n".join(blocks) + "\n\\end{axis}"
        suffix = "_".join(attack_types)
        stem = f"{dataset_name}_{suffix}_{metric}"
        outputs.append(render_tikz(body, plot_dir, stem, color_definitions(colors)))
    return outputs


def plot_attack_success_rate_tikz(
    results_dir,
    attack_budgets,
    plot_dir,
    dataset_name,
    threshold_km: Any = 2500,
    attack_types=None,
    all_results=None,
    gps_true: bool = True,
):
    """TikZ twin of ``utils.plots.results.plot_attack_success_rate``."""
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)
    if all_results is None:
        raise ValueError("No results available to plot. Provide all_results or a valid results_dir.")

    if isinstance(threshold_km, (int, float, np.integer, np.floating)):
        thresholds = [float(threshold_km)]
    else:
        thresholds = sorted(set(np.asarray(threshold_km, dtype=np.float64).reshape(-1).tolist()))
    if not thresholds:
        raise ValueError("threshold_km list cannot be empty")

    n_thresholds = len(thresholds)
    linewidths = np.linspace(0.5, 3, n_thresholds)
    linestyles = ["solid", "dashed", "dashdotted", "dotted", "dashdotdotted"]
    metric_name = _select_displacement_metric(gps_true)

    colors: dict[str, str] = {}
    blocks = []
    for at in attack_types:
        res = all_results.get(at)
        if res is None:
            continue
        budget_samples = _get_metric_samples_by_budget(res, metric_name)
        if budget_samples is None:
            continue
        budget_samples = [np.asarray(s)[~np.isnan(np.asarray(s, dtype=np.float64))] for s in budget_samples]

        success_curves = np.asarray([
            [float((s > t).mean()) if s.size > 0 else np.nan for s in budget_samples]
            for t in thresholds
        ])

        color_name, hexcode = attack_pgf_color(at)
        colors[color_name] = hexcode

        if n_thresholds > 1:
            low_path, high_path = f"lo{color_name}", f"hi{color_name}"
            blocks.append(
                f"\\addplot[name path={low_path}, draw=none, forget plot] "
                f"{coordinates(attack_budgets, success_curves.min(axis=0))};\n"
                f"\\addplot[name path={high_path}, draw=none, forget plot] "
                f"{coordinates(attack_budgets, success_curves.max(axis=0))};\n"
                f"\\addplot[{color_name}, opacity=0.30, forget plot] "
                f"fill between[of={low_path} and {high_path}];"
            )

        for ti, t in enumerate(thresholds):
            shade = 0.85 if n_thresholds == 1 else 1.0 - 0.55 * (ti / (n_thresholds - 1))
            shade_name = f"{color_name}Thr{ti}"
            colors[shade_name] = darken_hex(hexcode, shade)
            style = linestyles[ti % len(linestyles)]
            label = f"{_display_attack_name(at)} > {t:.0f} km"
            blocks.append(
                f"\\addplot[{shade_name}, {style}, line width={linewidths[ti]:.2f}pt, mark=none] "
                f"{coordinates(attack_budgets, success_curves[ti])};\n"
                f"\\addlegendentry{{{tex_escape(label)}}}"
            )

    if not blocks:
        raise ValueError("No attack results available to plot success rate for.")

    tick_positions = ",".join(f"{b:.6g}" for b in attack_budgets)
    tick_labels = ",".join(f"{b * 255:.0f}" for b in attack_budgets)
    axis_opts = ", ".join([
        "xmode=log",
        f"xtick={{{tick_positions}}}", f"xticklabels={{{tick_labels}}}", "minor tick num=0",
        "xlabel={Attack budget out of 255 (log scale)}",
        "ylabel={Attack success rate}", "ymin=0.4", "ymax=1.0",
        "grid=major", "grid style={dashed, gray!25}",
        "legend style={at={(1.02,1)}, anchor=north west, draw=none}",
        "width=12cm", "height=9cm",
    ])
    body = f"\\begin{{axis}}[{axis_opts}]\n" + "\n".join(blocks) + "\n\\end{axis}"
    suffix = "_".join(attack_types)
    stem = f"{dataset_name}_{suffix}_attack_success_rate"
    return render_tikz(body, plot_dir, stem, color_definitions(colors))


# --------------------------------------------------------------------------- #
# Ablation / breakdown plots
# --------------------------------------------------------------------------- #


def plot_attack_dtd_variance_tikz(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=None,
    all_results=None,
    gps_true: bool = True,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """TikZ twin of ``utils.plots.ablations.plot_attack_dtd_variance``."""
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)
    attack_types = attacks_dtd_last(attack_types)

    metric_name = _select_displacement_metric(gps_true)
    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    colors: dict[str, str] = {}
    blocks = []
    tick_labels = []
    for attack_type in attack_types:
        res = all_results.get(attack_type)
        if res is None:
            continue
        budget_samples = _get_metric_samples_by_budget(res, metric_name)
        if budget_samples is None:
            continue
        samples = budget_samples[budget_idx] if budget_idx < len(budget_samples) else np.array([])
        stats = boxplot_stats(samples)
        if stats is None:
            continue
        color_name, hexcode = attack_pgf_color(attack_type)
        colors[color_name] = hexcode
        blocks.append(
            f"\\addplot+[{boxplot_prepared(stats, position=len(tick_labels))}, fill={color_name}, draw={color_name}, "
            f"fill opacity=0.65, solid, mark=none] coordinates {{}};"
        )
        tick_labels.append(tex_escape(_display_attack_name(attack_type)))

    if not blocks:
        raise ValueError("No attack results available to plot DTD variance for.")

    ticks = ",".join(str(i) for i in range(len(tick_labels)))
    labels = ",".join(tick_labels)
    width_cm = max(8.0, 1.3 * len(tick_labels))
    axis_opts = ", ".join([
        "boxplot/draw direction=y",
        f"xtick={{{ticks}}}", f"xticklabels={{{labels}}}",
        "x tick label style={rotate=30, anchor=east}",
        "ylabel={Final-step displacement --- ``DTD'' (km)}",
        "grid=major", "grid style={gray!25}", "ymajorgrids",
        f"width={width_cm:.1f}cm", "height=6.5cm",
    ])
    body = f"\\begin{{axis}}[{axis_opts}]\n" + "\n".join(blocks) + "\n\\end{axis}"
    return render_tikz(body, plot_dir, f"{dataset_name}_attack_dtd_variance", color_definitions(colors))


def plot_loss_vs_fsd_tikz(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=("dtd", "training_loss", "sampling"),
    all_results=None,
    gps_true: bool = True,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """TikZ twin of ``utils.plots.ablations.plot_loss_vs_fsd``.

    Scatter data is written to external ``.dat`` files next to the ``.tex``
    (one per attack) rather than inlined -- a full-dataset run has thousands
    of points per panel, which would otherwise bloat the ``.tex`` and slow
    down its compilation.
    """
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)
    attack_types = attacks_dtd_last(attack_types)

    metric_name = _select_displacement_metric(gps_true)
    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    panels = []
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
        panels.append((attack_type, loss, fsd, pearson_r))

    if not panels:
        raise ValueError(
            "No paired (loss, FSD) samples available -- this requires results saved "
            "with the 'final_loss' field, which older runs may not have."
        )

    tikz_dir = os.path.join(plot_dir, "tikz")
    os.makedirs(tikz_dir, exist_ok=True)
    stem = f"{dataset_name}_loss_vs_fsd"

    colors: dict[str, str] = {}
    group_axes = []
    for i, (attack_type, loss, fsd, r) in enumerate(panels):
        color_name, hexcode = attack_pgf_color(attack_type)
        colors[color_name] = hexcode
        marker, mark_size = attack_marker(attack_type), attack_mark_size(attack_type, base_pt=1.0)
        dat_name = f"{stem}_{attack_type}.dat"
        write_dat_file(os.path.join(tikz_dir, dat_name), {"loss": loss, "fsd": fsd})
        r_label = f"r = {r:.2f}" if r is not None else "r = n/a"
        title = tex_escape(f"{_display_attack_name(attack_type)} ({r_label})")
        opts = [f"title={{{title}}}", "xlabel={Final training loss}", "ymode=log",
                "grid=both", "grid style={dashed, gray!25}"]
        if i == 0:
            opts.append("ylabel={Final-step displacement (km, log scale)}")
        group_axes.append(
            f"\\nextgroupplot[{', '.join(opts)}]\n"
            f"\\addplot[only marks, mark={marker}, mark size={mark_size}, {color_name}, "
            f"mark options={{fill opacity=0.5, draw opacity=0.5}}] table[x=loss,y=fsd] {{{dat_name}}};"
        )

    body = (
        f"\\begin{{groupplot}}[group style={{group size={len(panels)} by 1, "
        "horizontal sep=1.8cm}, width=6.5cm, height=6cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    return render_tikz(body, plot_dir, stem, color_definitions(colors))


def plot_clean_vs_attacked_displacement_tikz(
    results_dir=None,
    attack_budgets=None,
    plot_dir=None,
    dataset_name=None,
    attack_types=None,
    all_results=None,
    eps: float = DEFAULT_ABLATION_EPS,
):
    """TikZ twin of ``utils.plots.ablations.plot_clean_vs_attacked_displacement``.

    Faceted small multiples (one panel per attack) instead of every attack
    overlaid on one axis, and each panel is a 2D log-log density heatmap
    rather than a raw scatter: with ~4000 points per attack, an overlaid
    scatter is dominated by overplotting exactly where it matters most (the
    diagonal ridge), hiding where the mass actually concentrates. Each panel
    keeps its own attack color as a white-to-hue sequential ramp (density,
    not identity, is the encoded variable here -- every panel is already
    single-attack) so the figure still reads as "this is Encoder's panel,
    this is DTD's panel" at a glance. A shared log-log axis range across
    panels keeps them directly comparable.
    """
    if all_results is None:
        all_results = _load_attack_results(results_dir, dataset_name, attack_types)
    attack_types = attacks_dtd_last(attack_types)

    budget = select_closest_budget(attack_budgets, eps)
    budget_idx = int(np.argmin(np.abs(np.asarray(attack_budgets, dtype=np.float64) - budget)))

    tikz_dir = os.path.join(plot_dir, "tikz")
    os.makedirs(tikz_dir, exist_ok=True)
    stem = f"{dataset_name}_clean_vs_attacked_displacement"

    panels = []
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
        pearson_r = float(np.corrcoef(baseline, attacked)[0, 1]) if baseline.size > 1 else None
        panels.append((attack_type, baseline, attacked, pearson_r))
        all_vals.append(baseline)
        all_vals.append(attacked)

    if not panels:
        raise ValueError(
            "No paired (clean, attacked) true-GPS displacement samples available -- this "
            "requires results saved with restart-level 'gps_source'/'true_gps' data."
        )

    combined = np.concatenate(all_vals)
    positive = combined[combined > 0]
    # Anchored to the 2nd percentile (not the raw minimum): a handful of
    # near-zero outliers used to stretch the log axis down to ~1m, wasting
    # most of the panel's height on an almost-empty 10^-3..10^0 km band while
    # the actual mass of the data sits above ~1km.
    lo = float(np.percentile(positive, 2)) * 0.6 if positive.size else 1.0
    hi = float(combined.max()) * 1.3

    n_cols = min(4, len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))

    # Shared log-space bin grid across every panel -- same reasoning as the
    # shared xmin/xmax/ymin/ymax: panels stay directly comparable.
    n_bins = 45
    bin_edges = np.linspace(np.log10(lo), np.log10(hi), n_bins + 1)
    bin_centers = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)

    colors: dict[str, str] = {}
    colormap_defs: list[str] = []
    group_axes = []
    for i, (attack_type, baseline, attacked, r) in enumerate(panels):
        row, col = divmod(i, n_cols)
        color_name, hexcode = attack_pgf_color(attack_type)
        colors[color_name] = hexcode
        colormap_name = f"{color_name}Map"
        colormap_defs.append(sequential_colormap_def(colormap_name, hexcode))

        positive = (baseline > 0) & (attacked > 0)
        counts, _, _ = np.histogram2d(
            np.log10(baseline[positive]), np.log10(attacked[positive]), bins=[bin_edges, bin_edges]
        )
        # log1p (not raw counts): the diagonal ridge bins vastly outnumber
        # the scattered off-diagonal ones, and a linear color scale would
        # make everything but that ridge read as blank white.
        density = np.log1p(counts.T)  # rows follow y (attacked), cols follow x (clean)
        density_max = max(float(density.max()), 1e-6)

        xs_flat = np.tile(bin_centers, n_bins)
        ys_flat = np.repeat(bin_centers, n_bins)
        zs_flat = density.reshape(-1)
        dat_name = f"{stem}_{attack_type}.dat"
        write_dat_file(os.path.join(tikz_dir, dat_name),
                        {"clean": xs_flat, "attacked": ys_flat, "density": zs_flat})

        r_label = f"r = {r:.2f}" if r is not None else "r = n/a"
        title = tex_escape(f"{_display_attack_name(attack_type)} ({r_label})")
        opts = [
            f"title={{{title}}}", "xmode=log", "ymode=log", "view={0}{90}", "axis on top",
            f"xmin={lo:.4g}", f"xmax={hi:.4g}", f"ymin={lo:.4g}", f"ymax={hi:.4g}",
            "grid=both", "grid style={dashed, gray!25}",
            f"colormap name={colormap_name}", "point meta min=0", f"point meta max={density_max:.6g}",
        ]
        if col == 0:
            opts.append("ylabel={Attacked FSD (km)}")
        if row == n_rows - 1:
            opts.append("xlabel={Clean FSD (km)}")
        entries = [
            f"\\addplot3[surf, mesh/rows={n_bins}, mesh/cols={n_bins}, shader=flat corner, "
            f"point meta=explicit, forget plot] table[x=clean,y=attacked,meta=density] {{{dat_name}}};",
            f"\\addplot[black, dashed, thick, mark=none, forget plot] "
            f"{coordinates([lo, hi], [lo, hi])};",
        ]
        group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(entries))

    # One shared density legend for the whole grid (not per-panel, since every
    # panel uses the same light-to-dark convention, just tinted with its own
    # attack hue): a plain gray gradient swatch is enough to say what shade
    # means what, without implying it's tied to any single panel's color.
    density_legend = (
        "\\begin{scope}[shift={($(current bounding box.north)+(0,0.4cm)$)}]\n"
        "\\shade[left color=white, right color=black, draw=black, line width=0.3pt] "
        "(-1.4,0) rectangle (1.4,0.28);\n"
        "\\node[anchor=east, font=\\small] at (-1.55,0.14) {Low density};\n"
        "\\node[anchor=west, font=\\small] at (1.55,0.14) {High density};\n"
        "\\end{scope}"
    )

    body = (
        f"\\begin{{groupplot}}[group style={{group size={n_cols} by {n_rows}, "
        "horizontal sep=0.5cm, vertical sep=1.1cm, "
        "xticklabels at=edge bottom, yticklabels at=edge left}, width=5cm, height=5cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}\n"
        + density_legend
    )
    extra_preamble = color_definitions(colors) + "\n" + "\n".join(colormap_defs)
    return render_tikz(body, plot_dir, stem, extra_preamble)


# --------------------------------------------------------------------------- #
# Localizability grid (joins localizability scores with attack results)
# --------------------------------------------------------------------------- #


def _localizability_bucket_colors() -> list[str]:
    from matplotlib import colormaps
    cmap_colors = colormaps["Set1"](np.linspace(0, 1, 3))
    return ["".join(f"{int(round(c * 255)):02X}" for c in rgba[:3]) for rgba in cmap_colors]


def _get_localizability_buckets(res) -> np.ndarray:
    import torch
    loc = res["localizability"].detach().cpu()
    low_t = torch.quantile(loc, 0.33)
    high_t = torch.quantile(loc, 0.66)
    b = torch.zeros_like(loc, dtype=torch.long)
    b[(loc > low_t) & (loc <= high_t)] = 1
    b[loc > high_t] = 2
    return b.numpy()


def _get_localizability_attack_strength(res, attack, budget_idx) -> np.ndarray:
    import torch
    attack_result = res["attack_results"][attack]
    if isinstance(attack_result, dict):
        t = _get_metric_tensor(attack_result, "final_step_displacement_predicted")
        if t is None:
            t = _get_metric_tensor(attack_result, "final_step_displacement")
        if t is None:
            tensor_values = [v for v in attack_result.values() if isinstance(v, torch.Tensor)]
            if not tensor_values:
                raise ValueError(f"No tensor-valued attack results found for attack '{attack}'")
            t = tensor_values[0]
    else:
        t = attack_result
    t = t.detach().cpu()
    if t.ndim == 1:
        return t.numpy()
    if t.ndim == 2:
        return t[budget_idx].numpy()
    raise ValueError(f"Unexpected tensor shape {tuple(t.shape)} for attack '{attack}'")


def plot_localizability_results_tikz(attack_budgets, plot_dir, all_datasets_results, results_attack_budgets):
    """TikZ twin of ``utils.plots.ablations.plot_localizability_results``.

    Produces the same rows-x-columns (dataset x attack) grid of box plots
    bucketed by localizability tertile.
    """
    if isinstance(attack_budgets, (int, float, np.integer, np.floating)):
        selected_budget = float(attack_budgets)
    else:
        budgets_arr = np.asarray(attack_budgets, dtype=np.float64).reshape(-1)
        if budgets_arr.size != 1:
            raise ValueError("plot_localizability_results_tikz expects a single budget (float)")
        selected_budget = float(budgets_arr[0])

    budget_arr = np.asarray(results_attack_budgets, dtype=np.float64).reshape(-1)
    budget_idx = int(np.argmin(np.abs(budget_arr - selected_budget)))
    if not np.isclose(budget_arr[budget_idx], selected_budget, rtol=1e-4):
        raise ValueError(
            f"Budget {selected_budget:.6f} not found in results_attack_budgets "
            f"{[f'{b:.6f}' for b in budget_arr]}. Closest is {budget_arr[budget_idx]:.6f}."
        )

    bucket_hex = _localizability_bucket_colors()
    bucket_color_names = [f"bucket{name}" for name in _BUCKET_NAMES]
    colors = dict(zip(bucket_color_names, bucket_hex))

    dataset_names = list(all_datasets_results.keys())
    attacks = attacks_dtd_last(next(iter(all_datasets_results.values()))["attack_results"].keys())
    n_rows, n_cols = len(dataset_names), len(attacks)

    panel_w, panel_h, h_sep, v_sep = 4.4, 4.4, 0.4, 0.9
    legend_x = groupplot_legend_center_x(n_cols, panel_w, h_sep)

    group_axes = []
    legend_pending = True
    for r, ds in enumerate(dataset_names):
        res = all_datasets_results[ds]
        ds_label = "FSD (YFCC4K)" if ds == "yfcc" else "FSD (OSV-5M)"
        buckets = _get_localizability_buckets(res)

        row_cells = []
        row_values = []
        for attack in attacks:
            strength = _get_localizability_attack_strength(res, attack, budget_idx)
            data = [strength[buckets == k] for k in range(3)]
            stats = [boxplot_stats(d) for d in data]
            row_cells.append((attack, stats))
            nonempty = [d for d in data if d.size > 0]
            row_values.append(np.concatenate(nonempty) if nonempty else np.array([]))

        flat_row = np.concatenate([v for v in row_values if v.size > 0]) if any(v.size > 0 for v in row_values) else np.array([])
        has_data = flat_row.size > 0
        # Linear, not log: most displacement values sit above ~1000 km, so a log
        # axis wastes most of the panel's height on the near-empty 0-1000 km band
        # and compresses the actually-informative median/IQR comparisons into a
        # thin sliver near the top. A shared linear 0..row_ymax range (same ymax
        # on every column, not just the same tick labels) makes the buckets'
        # differences directly readable as height, at the cost of not resolving
        # the handful of near-zero whiskers -- an acceptable trade for this data.
        # n=5 (not the default 4), and no headroom multiplier: with n=4 the raw
        # step for this data (max ~20,000km) lands just above the 5,000 "nice"
        # step, so the algorithm jumps a full order of magnitude to 10,000 and
        # overshoots to a 30,000km ceiling -- wasting a third of every panel on
        # empty space above the actual data.
        row_ticks = nice_linear_ticks(float(np.nanmax(flat_row)), n=5) if has_data else []
        row_ymax = row_ticks[-1] if has_data else None

        for c, (attack, stats) in enumerate(row_cells):
            opts = [
                "boxplot/draw direction=y",
                "xtick={0,1,2}", f"xticklabels={{{','.join(_BUCKET_NAMES)}}}",
                "xticklabel style={font=\\scriptsize}",
                "ymin=0",
                # Otherwise pgfplots still shows a "\times 10^4"-style common-factor
                # label based on the underlying tick values even though the actual
                # text is fully overridden by our own "X,000 km" -- confusing/wrong
                # since nothing is actually scaled.
                "scaled y ticks=false",
                f"title={{{tex_escape(_display_attack_name(attack))}}}",
                "grid=major", "grid style={dashed, gray!25}", "ymajorgrids",
            ]
            if has_data:
                # Every column gets the same ``ytick`` (so gridlines line up across
                # the shared row), but only column 0 gets the label text -- an
                # *explicit* ``yticklabels`` on every axis (needed for the km
                # formatting) beats the groupplot's automatic "yticklabels at=edge
                # left" suppression, so the other columns get empty labels by hand.
                ytick_str = ",".join(f"{t:.6g}" for t in row_ticks)
                opts += [f"ymax={row_ymax:.6g}", f"ytick={{{ytick_str}}}"]
                if c == 0:
                    labels_str = ",".join(format_km_tick(t) for t in row_ticks)
                    opts += [f"yticklabels={{{labels_str}}}", "yticklabel style={font=\\tiny}"]
                else:
                    opts.append(f"yticklabels={{{',' * (len(row_ticks) - 1)}}}")
            if c == 0:
                opts.append(f"ylabel={{{tex_escape(ds_label)}}}")
            if legend_pending:
                opts.append(f"legend style={{at={{({legend_x:.4g},1.28)}}, anchor=south, "
                            "legend columns=3, draw=none, column sep=14pt, "
                            "/tikz/every node/.append style={inner xsep=3pt}}")

            boxes = []
            for k, s in enumerate(stats):
                if s is None:
                    continue
                boxes.append(
                    f"\\addplot+[{boxplot_prepared(s, position=k)}, fill={bucket_color_names[k]}, "
                    f"draw={bucket_color_names[k]}, fill opacity=0.5, solid, mark=none, forget plot] coordinates {{}};"
                )
            if legend_pending and boxes:
                # Plain rectangle swatches (like the matplotlib backend's legend
                # ``Patch``) instead of pgfplots' default boxplot-shaped legend
                # icon, which renders oversized/tall in a 3-column legend.
                for k in range(3):
                    cname = bucket_color_names[k]
                    boxes.append(
                        "\\addlegendimage{legend image code/.code={"
                        f"\\fill[{cname}, fill opacity=0.5] (0cm,-0.12cm) rectangle (0.7cm,0.12cm);"
                        f"\\draw[{cname}] (0cm,-0.12cm) rectangle (0.7cm,0.12cm);"
                        "}}\n"
                        f"\\addlegendentry{{{_BUCKET_NAMES[k]} localizability}}"
                    )
                legend_pending = False
            group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(boxes))

    body = (
        f"\\begin{{groupplot}}[group style={{group size={n_cols} by {n_rows}, "
        f"horizontal sep={h_sep}cm, vertical sep={v_sep}cm}}, "
        f"width={panel_w}cm, height={panel_h}cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    stem = f"localizability_budget_{selected_budget * 255:.0f}"
    return render_tikz(body, plot_dir, stem, color_definitions(colors))


def plot_localizability_vs_attacks_tikz(
    datasets: Sequence[str],
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    results_dir: str,
    plot_dir: str,
    plot_budgets: Optional[Sequence[float]] = None,
):
    """TikZ twin of ``utils.adversarial_eval.plot_localizability_vs_attacks``."""
    from utils.adversarial_eval import build_localizability_dataset_result

    all_datasets_results: dict[str, Any] = {}
    for dataset_name in datasets:
        try:
            all_datasets_results[dataset_name] = build_localizability_dataset_result(
                dataset_name, attack_types, results_dir,
            )
        except FileNotFoundError as exc:
            print(f"Skipping '{dataset_name}': {exc}")
    if not all_datasets_results:
        raise FileNotFoundError(
            "No dataset had both localizability scores and attack results; nothing to plot."
        )

    budgets_to_plot = list(plot_budgets) if plot_budgets else list(attack_budgets)
    os.makedirs(plot_dir, exist_ok=True)
    outputs = []
    for budget in budgets_to_plot:
        print(f"[tikz] Plotting localizability vs attack strength at budget {budget:.4f} "
              f"({round(budget * 255)}/255)...")
        outputs.append(plot_localizability_results_tikz(
            attack_budgets=budget,
            plot_dir=plot_dir,
            all_datasets_results=all_datasets_results,
            results_attack_budgets=list(attack_budgets),
        ))
    return outputs


# --------------------------------------------------------------------------- #
# JSON-backed ablations: sampling steps, model transfer, robustness
# --------------------------------------------------------------------------- #


def plot_sampling_steps_success_rate_tikz(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> dict:
    """TikZ twin of ``utils.plots.ablations.plot_sampling_steps_success_rate``."""
    thresholds = json_results["success_rate_thresholds_km"]
    eval_num_steps = json_results["eval_num_steps"]
    attack_types = attacks_dtd_last(json_results["attack_types"])
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    budgets_per_type = json_results.get("attack_budgets_per_type", {})

    colors: dict[str, str] = {}
    panels: list[list[str]] = [[] for _ in thresholds]
    for attack_type in attack_types:
        budget = select_closest_budget(budgets_per_type.get(attack_type, attack_budgets), eps)
        bkey = f"budget_{budget:.6f}"
        color_name, hexcode = attack_pgf_color(attack_type)
        colors[color_name] = hexcode
        marker, mark_size = attack_marker(attack_type), attack_mark_size(attack_type, base_pt=2.8)
        line_width = attack_line_width(attack_type, base_pt=1.4)
        for ax_idx, thr in enumerate(thresholds):
            rates = [
                json_results["results"][attack_type][bkey][str(ns)]["success_rates"][str(thr)]
                for ns in eval_num_steps
            ]
            entry = (f"\\addplot[{color_name}, solid, line width={line_width}, "
                     f"mark={marker}, mark size={mark_size}, mark options={{line width=1.1pt}}] "
                     f"{coordinates(eval_num_steps, rates)};")
            if ax_idx == 0:
                entry += f"\n\\addlegendentry{{{tex_escape(_display_attack_name(attack_type))}}}"
            panels[ax_idx].append(entry)

    # Panel size/sep shared with plot_model_transfer_success_rate_tikz and
    # plot_robustness_results_tikz -- all three are single-row groupplots of
    # success-rate-vs-something, so keeping their per-panel geometry identical
    # (rather than each picking its own numbers) is what makes fonts, markers,
    # and line widths read at the same visual scale when the figures sit next
    # to each other in the paper.
    panel_w, panel_h, h_sep = SUCCESS_RATE_PANEL_W, SUCCESS_RATE_PANEL_H, SUCCESS_RATE_H_SEP
    legend_x = groupplot_legend_center_x(len(thresholds), panel_w, h_sep)

    # eval_num_steps roughly doubles each step (e.g. 8/16/32/64) then jumps to
    # a much larger final value (250) -- on a linear x-axis that last jump
    # dwarfs the spacing between the earlier points, squeezing all the actual
    # variation into a sliver on the left while the right half of the panel
    # sits empty. A log x-axis (explicit ticks at the actual step values, not
    # log-scale's default decade ticks) spaces every point by its ratio
    # instead of its raw distance, so the informative 8->64 range gets as
    # much width as the flat tail out to 250.
    xtick_str = ",".join(str(s) for s in eval_num_steps)
    group_axes = []
    for ax_idx, thr in enumerate(thresholds):
        opts = [
            "xmode=log", "log basis x=2", "log ticks with fixed point",
            "xminorticks=false", f"xtick={{{xtick_str}}}",
            "xlabel={Sampling steps}",
            f"title={{Displacement > {thr} km}}", "ymin=0.3", "ymax=1",
            "grid=major", "grid style={gray!25}",
        ]
        if ax_idx == 0:
            opts.append("ylabel={Attack success rate}")
            opts.append(f"legend style={{at={{({legend_x:.4g},1.35)}}, anchor=south, "
                        f"legend columns=-1, draw=none, font=\\large}}")
            opts.append(SINGLE_MARK_LEGEND_IMAGE)
        group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(panels[ax_idx]))

    body = (
        f"\\begin{{groupplot}}[group style={{group size={len(thresholds)} by 1, "
        f"horizontal sep={h_sep}cm}}, width={panel_w}cm, height={panel_h}cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    return render_tikz(body, plot_dir, f"{dataset}_sampling_steps_success_rate", color_definitions(colors))


def plot_model_transfer_success_rate_tikz(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> dict:
    """TikZ twin of ``utils.plots.ablations.plot_model_transfer_success_rate``.

    Grouped bar chart, one cluster per attack, one color+pattern coded bar
    per backbone model (RFM/diffusion/flow): the question this figure answers
    is "does this attack transfer across backbones", which reads directly off
    a single cluster once the attack is the group, whereas grouping by model
    (the original layout) scattered that comparison across separate clusters.
    Restricted to the headline attacks -- UniDef, DTD, Sampling, Encoder, and
    GeoShield when the dataset has it -- so the chart isn't cluttered with
    every ablation baseline.
    """
    model_labels = json_results["model_labels"]

    # "RFM" (an acronym) stays all-caps; "flow" is displayed as "FM" (its
    # common short name in this paper); other backbones get Title case, not
    # SHOUTING case -- "Diffusion", not "DIFFUSION".
    def _model_label(m: str) -> str:
        normalized = m.lower()
        if normalized == "rfm":
            return m.upper()
        if normalized == "flow":
            return "FM"
        return m.capitalize()

    model_labels_capital = [tex_escape(_model_label(m)) for m in model_labels]
    thresholds = json_results["success_rate_thresholds_km"]
    _HEADLINE_ATTACKS = ["unidef", "dtd", "sampling", "encoder", "geoshield"]
    attack_types = sorted(
        (a for a in json_results["attack_types"] if a.lower() in _HEADLINE_ATTACKS),
        key=lambda a: _HEADLINE_ATTACKS.index(a.lower()),
    )
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    budgets_per_type = json_results.get("attack_budgets_per_type", {})

    # Fixed qualitative palette for the 3 backbones (independent of the
    # per-attack palette used everywhere else -- here the model is the
    # series, not the attack).
    _MODEL_COLORS = {"rfm": "1F77B4", "diffusion": "FF7F0E", "flow": "2CA02C"}
    _MODEL_PATTERNS = {"rfm": "north east lines", "diffusion": "horizontal lines", "flow": "dots"}

    xcoords_labels = [tex_escape(_display_attack_name(a)) for a in attack_types]
    colors: dict[str, str] = {}
    panels: list[list[str]] = [[] for _ in thresholds]
    for m_idx, model in enumerate(model_labels):
        color_name = "model" + sanitize_pgfname(model).capitalize()
        hexcode = _MODEL_COLORS.get(model.lower(), _MODEL_COLORS["rfm"])
        colors[color_name] = hexcode
        pattern = _MODEL_PATTERNS.get(model.lower(), "crosshatch")
        pattern_color = contrasting_ink(hexcode)
        for col_idx, thr in enumerate(thresholds):
            rates = []
            for attack_type in attack_types:
                budget = select_closest_budget(budgets_per_type.get(attack_type, attack_budgets), eps)
                bkey = f"budget_{budget:.6f}"
                cell = json_results["results"][attack_type][bkey][model]
                rates.append(cell["success_rates"][str(thr)])
            pts = " ".join(f"({lbl},{r:.6g})" for lbl, r in zip(xcoords_labels, rates))
            # Single-swatch legend image only matters for the addplot that
            # actually contributes a legend entry (col_idx == 0); the default
            # (unset) legend image on the other columns' addplots is never
            # rendered since they have no \addlegendentry.
            legend_img = f", {bar_legend_image_code(color_name, pattern, pattern_color)}" if col_idx == 0 else ""
            entry = (f"\\addplot[fill={color_name}, draw=black, line width=0.3pt, "
                     f"postaction={{pattern={pattern}, pattern color={pattern_color}}}{legend_img}] "
                     f"coordinates {{{pts}}};")
            if col_idx == 0:
                entry += f"\n\\addlegendentry{{{model_labels_capital[m_idx]}}}"
            panels[col_idx].append(entry)

    xcoords = ",".join(xcoords_labels)
    panel_w, panel_h, h_sep = SUCCESS_RATE_PANEL_W, SUCCESS_RATE_PANEL_H, SUCCESS_RATE_H_SEP
    enlarge = 0.12
    legend_x = groupplot_legend_center_x(len(thresholds), panel_w, h_sep)
    # Bar width is an absolute pt size, so it must fit within one category's
    # actual on-canvas slot (panel width minus the enlarge margins, divided
    # among the attack categories) or adjacent clusters overlap -- the old
    # formula sized bars only off the bar count, ignoring how many categories
    # they had to fit next to, which is what caused the overlap.
    _CM_TO_PT = 28.4527559
    usable_pt = panel_w * _CM_TO_PT * (1 - 2 * enlarge)
    slot_pt = usable_pt / max(len(attack_types), 1)
    bar_width = max(3.0, slot_pt / max(len(model_labels), 1) * 0.8)

    group_axes = []
    for col_idx, thr in enumerate(thresholds):
        opts = [
            "ybar", f"bar width={bar_width:.3g}pt",
            f"symbolic x coords={{{xcoords}}}", "xtick=data",
            "x tick label style={rotate=30, anchor=east}",
            f"enlarge x limits={{{enlarge}}}",
            f"title={{Displacement > {thr} km}}", "ymin=0", "ymax=1",
            "grid=major", "grid style={gray!25}",
        ]
        if col_idx == 0:
            opts.append("ylabel={Attack success rate}")
            opts.append(f"legend style={{at={{({legend_x:.4g},1.35)}}, anchor=south, "
                        f"legend columns=-1, draw=none, font=\\large}}")
        group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(panels[col_idx]))

    body = (
        f"\\begin{{groupplot}}[group style={{group size={len(thresholds)} by 1, "
        f"horizontal sep={h_sep}cm}}, width={panel_w}cm, height={panel_h}cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    return render_tikz(body, plot_dir, f"{dataset}_model_transfer_success_rate", color_definitions(colors))


def plot_robustness_results_tikz(
    json_results: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> dict:
    """TikZ twin of ``utils.plots.ablations.plot_robustness_results`` -- only
    the 200km and 2500km distance thresholds, laid out as a single row, and a
    JPEG quality=100 point synthesized from the Gaussian-blur-sigma=0 level
    (quality=100 / sigma=0 are both "no transform applied", so they share the
    same success rate -- the robustness sweep just never evaluated
    quality=100 explicitly under the JPEG branch).
    """
    attack_types = attacks_dtd_last(json_results["attack_types"])
    attack_budgets = json_results["attack_budgets"]
    dataset = json_results["dataset"]
    _HEADLINE_THRESHOLDS = [200, 2500]
    thresholds = [t for t in _HEADLINE_THRESHOLDS if t in json_results["success_rate_thresholds_km"]]
    # Success rate against the *true* GPS (not the clean prediction) -- the
    # primary metric used everywhere else in this evaluation. The JSON may
    # also carry a "predicted" metric, but it renders with identical styling
    # (same color/marker/linestyle), so overlaying both just doubles every
    # line with no visual differentiation -- draw the one metric that matters.
    metric = "true"
    budget = select_closest_budget(attack_budgets, eps)
    bkey = f"budget_{budget:.6f}"

    transforms = [("JPEG quality factor", "jpeg"), ("Gaussian blur $\\sigma$", "blur")]
    n_cols = len(thresholds) * len(transforms)

    def _sorted_levels(level_dict: dict):
        items = sorted(level_dict.items(), key=lambda kv: float(kv[0]))
        return [float(k) for k, _ in items], [v for _, v in items]

    colors: dict[str, str] = {}
    cell_blocks: dict[tuple[int, int], list[str]] = {}
    for col_idx, (_, json_key) in enumerate(transforms):
        for attack_type in attack_types:
            level_dict = json_results["results"][attack_type][bkey][json_key]
            if not level_dict:
                continue
            if json_key == "jpeg" and "100" not in level_dict:
                blur_dict = json_results["results"][attack_type][bkey].get("blur", {})
                if "0" in blur_dict:
                    level_dict = {**level_dict, "100": blur_dict["0"]}
            xs, entries = _sorted_levels(level_dict)
            any_finite = any(
                not np.isnan(entry[metric]["success_rates"][str(thr)])
                for entry in entries for thr in thresholds
            )
            if not any_finite:
                continue
            color_name, hexcode = attack_pgf_color(attack_type)
            colors[color_name] = hexcode
            marker, mark_size = attack_marker(attack_type), attack_mark_size(attack_type, base_pt=2.8)
            line_width = attack_line_width(attack_type, base_pt=1.4)
            for row_idx, thr in enumerate(thresholds):
                rates = [entry[metric]["success_rates"][str(thr)] for entry in entries]
                key = (col_idx, row_idx)
                lines = cell_blocks.setdefault(key, [])
                is_legend_line = (col_idx == 0 and row_idx == 0)
                forget = "" if is_legend_line else ", forget plot"
                line = (f"\\addplot[{color_name}, solid, line width={line_width}, "
                        f"mark={marker}, mark size={mark_size}, "
                        f"mark options={{line width=1.1pt}}{forget}] "
                        f"{coordinates(xs, rates)};")
                if is_legend_line:
                    line += f"\n\\addlegendentry{{{tex_escape(_display_attack_name(attack_type))}}}"
                lines.append(line)

    panel_w, panel_h, h_sep = SUCCESS_RATE_PANEL_W, SUCCESS_RATE_PANEL_H, SUCCESS_RATE_H_SEP
    v_sep = h_sep
    # The single shared legend spans the *entire* row (every panel), so it's
    # centered relative to the very first axis.
    legend_x = groupplot_legend_center_x(n_cols, panel_w, h_sep)

    group_axes = []
    for col_idx, (xlabel, json_key) in enumerate(transforms):
        for thr_idx, thr in enumerate(thresholds):
            opts = [
                "ymin=0.3", "ymax=1", "grid=major", "grid style={gray!25}",
                f"title={{Displacement > {thr} km}}",
            ]
            # Every panel sits on the same row, so every panel gets its own
            # transform-type xlabel (no bottom-row-only special case needed).
            opts.append(f"xlabel={{{xlabel}}}")
            if col_idx == 0 and thr_idx == 0:
                opts.append("ylabel={Attack success rate}")
            if json_key == "jpeg":
                opts.append("x dir=reverse")
            if col_idx == 0 and thr_idx == 0:
                opts.append(f"legend style={{at={{({legend_x:.4g},1.35)}}, anchor=south, "
                            f"legend columns=-1, draw=none, font=\\large}}")
                opts.append(SINGLE_MARK_LEGEND_IMAGE)
            blocks = cell_blocks.get((col_idx, thr_idx), [])
            group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(blocks))

    body = (
        f"\\begin{{groupplot}}[group style={{group size={n_cols} by 1, "
        f"horizontal sep={h_sep}cm, vertical sep={v_sep}cm}}, width={panel_w}cm, height={panel_h}cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    return render_tikz(body, plot_dir, f"{dataset}_robustness_results", color_definitions(colors))


def plot_robustness_and_sampling_steps_tikz(
    robustness_json: dict,
    sampling_steps_json: dict,
    plot_dir: str,
    eps: float = DEFAULT_ABLATION_EPS,
) -> dict:
    """Combined robustness figure: rows are the 200km/2500km success-rate
    thresholds, columns are the three robustness axes -- Gaussian blur, JPEG
    compression, and sampling steps. The blur/jpeg columns only cover the
    attacks common to both ablations (geoshield/dtd/encoder/sampling), since
    L2 and AdvDM never got a robustness sweep (see
    plot_robustness_results_tikz's docstring) -- but they *did* get a
    sampling-steps sweep, so they still appear in that column and in the
    shared legend (registered there via a legend-only ``\\addlegendimage``,
    since they have no line in the panel the legend is physically attached
    to). Panel geometry matches the other single-row success-rate groupplots
    (SUCCESS_RATE_PANEL_W/H/H_SEP) so this figure sits at the same visual
    scale as those.
    """
    _HEADLINE_THRESHOLDS = [200, 2500]
    thresholds = [t for t in _HEADLINE_THRESHOLDS if t in robustness_json["success_rate_thresholds_km"]]
    metric = "true"
    dataset = robustness_json["dataset"]

    common_attack_types = [a for a in robustness_json["attack_types"] if a in sampling_steps_json["attack_types"]]
    # L2 (diffusion_l2) / AdvDM (training_loss): sampling-steps-only attacks,
    # requested in addition to the common set even though they can't appear
    # in the blur/jpeg columns.
    extra_attack_types = [
        a for a in ("diffusion_l2", "training_loss")
        if a in sampling_steps_json["attack_types"] and a not in common_attack_types
    ]
    attack_types = attacks_dtd_last(common_attack_types + extra_attack_types)
    common_set = set(common_attack_types)

    rob_budget = select_closest_budget(robustness_json["attack_budgets"], eps)
    rob_bkey = f"budget_{rob_budget:.6f}"
    steps_budgets_per_type = sampling_steps_json.get("attack_budgets_per_type", {})
    steps_attack_budgets = sampling_steps_json["attack_budgets"]
    eval_num_steps = sampling_steps_json["eval_num_steps"]

    # (xlabel, json_key) -- json_key is None for the sampling-steps column,
    # which reads from a different results file than the blur/jpeg columns.
    transforms = [
        ("Gaussian blur $\\sigma$", "blur"),
        ("JPEG quality factor", "jpeg"),
        ("Sampling steps", None),
    ]
    n_cols = len(transforms)
    n_rows = len(thresholds)

    def _sorted_levels(level_dict):
        items = sorted(level_dict.items(), key=lambda kv: float(kv[0]))
        return [float(k) for k, _ in items], [v for _, v in items]

    colors: dict[str, str] = {}
    cell_blocks: dict[tuple[int, int], list[str]] = {}

    for attack_type in attack_types:
        color_name, hexcode = attack_pgf_color(attack_type)
        colors[color_name] = hexcode
        marker, mark_size = attack_marker(attack_type), attack_mark_size(attack_type, base_pt=2.8)
        line_width = attack_line_width(attack_type, base_pt=1.4)
        has_robustness_data = attack_type in common_set

        def _add_line(col_idx: int, row_idx: int, xs, rates, want_legend: bool) -> None:
            key = (col_idx, row_idx)
            lines = cell_blocks.setdefault(key, [])
            forget = "" if want_legend else ", forget plot"
            line = (f"\\addplot[{color_name}, solid, line width={line_width}, "
                    f"mark={marker}, mark size={mark_size}, "
                    f"mark options={{line width=1.1pt}}{forget}] "
                    f"{coordinates(xs, rates)};")
            if want_legend:
                line += f"\n\\addlegendentry{{{tex_escape(_display_attack_name(attack_type))}}}"
            lines.append(line)

        if has_robustness_data:
            for col_idx, (_, json_key) in enumerate(transforms[:2]):
                level_dict = robustness_json["results"][attack_type][rob_bkey][json_key]
                if not level_dict:
                    continue
                if json_key == "jpeg" and "100" not in level_dict:
                    blur_dict = robustness_json["results"][attack_type][rob_bkey].get("blur", {})
                    if "0" in blur_dict:
                        level_dict = {**level_dict, "100": blur_dict["0"]}
                xs, entries = _sorted_levels(level_dict)
                for row_idx, thr in enumerate(thresholds):
                    rates = [entry[metric]["success_rates"][str(thr)] for entry in entries]
                    _add_line(col_idx, row_idx, xs, rates, want_legend=(col_idx == 0 and row_idx == 0))
        else:
            # No blur/jpeg data -- register the legend entry directly on the
            # (0, 0) axis (where the shared legend lives) without a real plot.
            legend_only = (f"\\addlegendimage{{{color_name}, solid, line width={line_width}, "
                           f"mark={marker}, mark size={mark_size}, mark options={{line width=1.1pt}}}}\n"
                           f"\\addlegendentry{{{tex_escape(_display_attack_name(attack_type))}}}")
            cell_blocks.setdefault((0, 0), []).append(legend_only)

        budget = select_closest_budget(steps_budgets_per_type.get(attack_type, steps_attack_budgets), eps)
        bkey = f"budget_{budget:.6f}"
        for row_idx, thr in enumerate(thresholds):
            rates = [
                sampling_steps_json["results"][attack_type][bkey][str(ns)]["success_rates"][str(thr)]
                for ns in eval_num_steps
            ]
            _add_line(2, row_idx, eval_num_steps, rates, want_legend=False)

    panel_w, panel_h, h_sep = SUCCESS_RATE_PANEL_W, SUCCESS_RATE_PANEL_H, SUCCESS_RATE_H_SEP
    # Wider than h_sep: each row-to-row gap has to fit the bottom row's xlabel
    # *and* the next row's title stacked in it, unlike the purely-horizontal
    # single-row figures where h_sep only ever separates two plot frames.
    v_sep = 2.2
    legend_x = groupplot_legend_center_x(n_cols, panel_w, h_sep)
    xtick_str = ",".join(str(s) for s in eval_num_steps)

    group_axes = []
    for row_idx, thr in enumerate(thresholds):
        for col_idx, (xlabel, json_key) in enumerate(transforms):
            opts = [
                "ymin=0.3", "ymax=1", "grid=major", "grid style={gray!25}",
                f"title={{Displacement > {thr} km}}",
            ]
            if col_idx == 0:
                opts.append("ylabel={Attack success rate}")
            # The x-axis quantity, on the other hand, does need to appear on
            # every panel: each row is a separate axis, so a bottom-row-only
            # xlabel would leave the top row's columns unlabeled.
            opts.append(f"xlabel={{{xlabel}}}")
            if json_key == "jpeg":
                opts.append("x dir=reverse")
            elif json_key is None:
                opts += ["xmode=log", "log basis x=2", "log ticks with fixed point",
                         "xminorticks=false", f"xtick={{{xtick_str}}}"]
            if row_idx == 0 and col_idx == 0:
                opts.append(f"legend style={{at={{({legend_x:.4g},1.35)}}, anchor=south, "
                            f"legend columns=-1, draw=none, font=\\large}}")
                opts.append(SINGLE_MARK_LEGEND_IMAGE)
            blocks = cell_blocks.get((col_idx, row_idx), [])
            group_axes.append(f"\\nextgroupplot[{', '.join(opts)}]\n" + "\n".join(blocks))

    body = (
        f"\\begin{{groupplot}}[group style={{group size={n_cols} by {n_rows}, "
        f"horizontal sep={h_sep}cm, vertical sep={v_sep}cm}}, width={panel_w}cm, height={panel_h}cm]\n"
        + "\n".join(group_axes) + "\n\\end{groupplot}"
    )
    return render_tikz(body, plot_dir, f"{dataset}_robustness_and_sampling_steps", color_definitions(colors))
