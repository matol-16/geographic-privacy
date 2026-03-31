import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from torchvision import transforms
import torch
from PIL import Image
import numpy as np

from adversarial_metrics import trajectory_displacement


def _sanitize_lon_lat(coords):
    """Return valid [lat, lon] rows only, wrapping lon to [-180, 180]."""
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected coords shape [N, 2] with [lat, lon]")

    arr = arr.copy()
    arr[:, 1] = ((arr[:, 1] + 180.0) % 360.0) - 180.0
    arr[:, 0] = np.clip(arr[:, 0], -90.0, 90.0)

    valid = np.isfinite(arr).all(axis=1)
    return arr[valid], valid


def _plot_valid_path(ax, lat_lon_traj, **plot_kwargs):
    """Plot only contiguous valid trajectory segments to avoid Shapely warnings."""
    traj = np.asarray(lat_lon_traj, dtype=np.float64)
    if traj.ndim != 2 or traj.shape[1] != 2:
        return

    traj = traj.copy()
    traj[:, 1] = ((traj[:, 1] + 180.0) % 360.0) - 180.0
    traj[:, 0] = np.clip(traj[:, 0], -90.0, 90.0)
    valid = np.isfinite(traj).all(axis=1)

    start = None
    for i, is_valid in enumerate(valid):
        if is_valid and start is None:
            start = i
        if (not is_valid or i == len(valid) - 1) and start is not None:
            end = i if not is_valid else i + 1
            if end - start >= 2:
                seg = traj[start:end]
                ax.plot(seg[:, 1], seg[:, 0], **plot_kwargs)
            start = None


def plot_gps_samples_on_map(gps_coords_source, gps_coords_target, gps_coords_perturbed, perturb_budget = None, cfg=None, point_size=100):
    plt.figure(figsize=(8,6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

    # Higher-contrast map colors for better point visibility
    ax.set_facecolor('#1f2a38')
    ax.add_feature(cfeature.OCEAN, facecolor='#1f2a38')
    ax.add_feature(cfeature.LAND, facecolor='#d9d2b6', edgecolor='none')
    ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white', linewidth=0.6)

    gps_coords_source, _ = _sanitize_lon_lat(gps_coords_source)
    gps_coords_perturbed, _ = _sanitize_lon_lat(gps_coords_perturbed)
    if gps_coords_target is not None and len(gps_coords_target) > 0:
        gps_coords_target, _ = _sanitize_lon_lat(gps_coords_target)

    # Pipeline outputs arrays shaped [N, 2] as [latitude, longitude]
    if len(gps_coords_source) > 0:
        ax.scatter(
            gps_coords_source[:, 1],
            gps_coords_source[:, 0],
            color='deepskyblue',
            marker='o',
            s=200,
            alpha=0.95,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            label='Source Locations',
            zorder=5,
        )
    if gps_coords_target is not None and len(gps_coords_target) > 0:
        ax.scatter(
            gps_coords_target[:, 1],
            gps_coords_target[:, 0],
            color='lime',
            marker='o',
            s=point_size,
            alpha=0.95,
            linewidths=0.5,
            transform=ccrs.PlateCarree(),
            label='Target Locations',
            zorder=6,
        )
    if len(gps_coords_perturbed) > 0:
        ax.scatter(
            gps_coords_perturbed[:, 1],
            gps_coords_perturbed[:, 0],
            color='red',
            marker='X',
            s=point_size,
            alpha=0.95,
            edgecolors='white',
            linewidths=0.8,
            transform=ccrs.PlateCarree(),
            label='Perturbed Predicted Locations',
            zorder=7,
        )

    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.45, color='white', alpha=0.25, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False

    # Add title and legend
    title = 'Geolocation Samples (Global View)' if perturb_budget is None else f'Geolocation Samples (Budget: {perturb_budget:.3f})'
    
    if cfg is not None:
        title += f", CFG: {cfg}"
    
    plt.title(title, fontsize=16, color='black', pad=20)
    plt.legend(loc='upper left', fontsize='large', frameon=True, facecolor='white', edgecolor='black')

    plt.tight_layout()
    plt.show()
    
    

def plot_gps_trajectories_on_map(
    gps_traj_source,
    gps_traj_perturbed,
    perturb_budget=None,
    cfg=None,
    max_trajectories=16,
    show_map=True,
    show_paths=True,
    show_connectors=True,
    show_displacement=True,
    point_size=8,
    metric="haversine",
):
    if not show_map and not show_displacement:
        raise ValueError("At least one of show_map or show_displacement must be True")

    map_ax = None
    disp_ax = None

    if show_map and show_displacement:
        fig = plt.figure(figsize=(12, 5))
        map_ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        disp_ax = fig.add_subplot(1, 2, 2)
    elif show_map:
        fig = plt.figure(figsize=(8, 6))
        map_ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    else:
        fig = plt.figure(figsize=(7, 5))
        disp_ax = fig.add_subplot(1, 1, 1)

    if show_map:
        assert map_ax is not None
        map_ax.set_global()
        map_ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Higher-contrast map colors for better point visibility
        map_ax.set_facecolor('#1f2a38')
        map_ax.add_feature(cfeature.OCEAN, facecolor='#1f2a38')
        map_ax.add_feature(cfeature.LAND, facecolor='#d9d2b6', edgecolor='none')
        map_ax.add_feature(cfeature.COASTLINE, edgecolor='white', linewidth=0.7)
        map_ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white', linewidth=0.6)

    if isinstance(gps_traj_source, torch.Tensor):
        gps_traj_source = gps_traj_source.detach().cpu().numpy()
    if isinstance(gps_traj_perturbed, torch.Tensor):
        gps_traj_perturbed = gps_traj_perturbed.detach().cpu().numpy()

    gps_traj_source = np.asarray(gps_traj_source)
    gps_traj_perturbed = np.asarray(gps_traj_perturbed)

    if gps_traj_source.ndim != 3 or gps_traj_source.shape[-1] != 2:
        raise ValueError("gps_traj_source must have shape [num_steps, batch_size, 2]")
    if gps_traj_perturbed.ndim != 3 or gps_traj_perturbed.shape[-1] != 2:
        raise ValueError("gps_traj_perturbed must have shape [num_steps, batch_size, 2]")
    if gps_traj_perturbed.shape[1] != gps_traj_source.shape[1]:
        raise ValueError("gps_traj_source and gps_traj_perturbed must have the same batch_size")

    num_steps, batch_size, _ = gps_traj_source.shape
    n_plot = min(batch_size, max_trajectories)
    colors = colormaps['tab20'](np.linspace(0, 1, max(n_plot, 2)))

    for trajectory_index in range(n_plot):
        c = colors[trajectory_index % len(colors)]

        source_traj = gps_traj_source[:, trajectory_index, :]  # [num_steps, 2]
        perturbed_traj = gps_traj_perturbed[:, trajectory_index, :]  # [num_steps, 2]

        if show_map:
            if show_paths:
                _plot_valid_path(
                    map_ax,
                    source_traj,
                    color=c,
                    linewidth=0.7,
                    alpha=0.35,
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            source_traj_sanitized, source_valid = _sanitize_lon_lat(source_traj)
            map_ax.scatter(
                source_traj_sanitized[:, 1],
                source_traj_sanitized[:, 0],
                color=c,
                marker='o',
                s=point_size,
                alpha=0.85,
                linewidths=0,
                transform=ccrs.PlateCarree(),
                zorder=5,
                label='Source trajectories' if trajectory_index == 0 else None,
            )

            if show_paths:
                _plot_valid_path(
                    map_ax,
                    perturbed_traj,
                    color=c,
                    linewidth=0.7,
                    alpha=0.35,
                    linestyle='--',
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )
            perturbed_traj_sanitized, perturbed_valid = _sanitize_lon_lat(perturbed_traj)
            map_ax.scatter(
                perturbed_traj_sanitized[:, 1],
                perturbed_traj_sanitized[:, 0],
                color=c,
                marker='x',
                s=point_size,
                alpha=0.85,
                linewidths=0.8,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label='Perturbed trajectories' if trajectory_index == 0 else None,
            )

            if show_connectors:
                valid_steps = source_valid & perturbed_valid
                for step_index in range(num_steps):
                    if not valid_steps[step_index]:
                        continue
                    map_ax.plot(
                        [source_traj[step_index, 1], perturbed_traj[step_index, 1]],
                        [source_traj[step_index, 0], perturbed_traj[step_index, 0]],
                        color=c,
                        linewidth=0.4,
                        alpha=0.2,
                        transform=ccrs.PlateCarree(),
                        zorder=3,
                    )

    if show_map:
        # Add gridlines
        gl = map_ax.gridlines(draw_labels=True, linewidth=0.45, color='white', alpha=0.25, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        # Add title and legend
        title = 'Geolocation Trajectories (Global View)' if perturb_budget is None else f'Geolocation Trajectories (Budget: {perturb_budget:.3f})'

        if cfg is not None:
            title += f", CFG: {cfg}"
        if n_plot < batch_size:
            title += f" (showing {n_plot}/{batch_size})"

        map_ax.set_title(title, fontsize=13, color='black', pad=12)
        map_ax.legend(loc='upper left', fontsize='small', frameon=True, facecolor='white', edgecolor='black')

    if show_displacement:
        assert disp_ax is not None
        src = gps_traj_source[:, :n_plot, :].astype(np.float64)
        per = gps_traj_perturbed[:, :n_plot, :].astype(np.float64)
        #convert to tensors for metrics, which expect tensors
        src_tensor = torch.from_numpy(src)
        per_tensor = torch.from_numpy(per)
        displacement = trajectory_displacement(src_tensor, per_tensor, metric=metric)
        displacement_np = displacement.detach().cpu().numpy()
        mean_disp = np.nanmean(displacement_np, axis=1)
        median_disp = np.nanmedian(displacement_np, axis=1)
        q25_disp = np.nanquantile(displacement_np, 0.25, axis=1)
        q75_disp = np.nanquantile(displacement_np, 0.75, axis=1)
        steps = np.arange(num_steps)
        disp_ax.plot(steps, mean_disp, color='black', linewidth=1.2, linestyle='--', alpha=0.8, label='Mean displacement')
        disp_ax.plot(steps, median_disp, color='crimson', linewidth=1.8, label='Median displacement')
        disp_ax.fill_between(steps, q25_disp, q75_disp, color='crimson', alpha=0.2, label='IQR (25-75%)')
        disp_ax.set_title('Perturbation effect over steps', fontsize=12)
        disp_ax.set_xlabel('Step')
        disp_ax.set_ylabel(f'Displacement {metric}')
        disp_ax.grid(alpha=0.3)
        disp_ax.legend(fontsize='small')

    fig.tight_layout()
    plt.show()


def plot_gps_trajectories_clean(gps_traj_source, gps_traj_perturbed, perturb_budget=None, cfg=None):
    return plot_gps_trajectories_on_map(
        gps_traj_source=gps_traj_source,
        gps_traj_perturbed=gps_traj_perturbed,
        perturb_budget=perturb_budget,
        cfg=cfg,
        max_trajectories=12,
        show_paths=True,
        show_connectors=False,
        show_displacement=True,
        point_size=6,
    )
    
    
    
#comprehensive evaluation plots:

import os

def plot_transferability_results(results_dir, attack_budgets, plot_dir, dataset_name, metric, results=None):
    # if results is None and results_dir is not None:
    # 	results = torch.load(os.path.join(results_dir, f"{dataset_name}_results_transferability.pt"))

    # plt.figure(figsize=(10,6))
    # for attack, res in results.items():
    # 	mean_metric = res.mean(dim=1)
    # 	std_metric = res.std(dim=1)
    # 	plt.plot(attack_budgets, mean_metric, label=attack)
    # 	plt.fill_between(attack_budgets, mean_metric-std_metric, mean_metric+std_metric, alpha=0.2)

    # plt.xlabel("Attack budget (eps)")
    # plt.ylabel(metric)
    # plt.title(f"Attack transferability evaluation on {dataset_name} dataset")
    # plt.legend()
    # if plot_dir is not None:
    # 	os.makedirs(plot_dir, exist_ok=True)
    # 	plt.savefig(os.path.join(plot_dir, f"{dataset_name}_transferability.png"))
    # else:
    # 	plt.show()
 
    #instead, for each attack budget, plot a boxplot of the metric for each attack type, to better show the distribution of the metric across samples and attacks, which is more informative for transferability evaluation
    if results is None and results_dir is not None:
        results = torch.load(os.path.join(results_dir, f"{dataset_name}_results_transferability.pt"))
    data_to_plot = []
    attack_labels = []
    for attack, res in results.items():
        for i, budget in enumerate(attack_budgets):
            data_to_plot.append(res[i].cpu().numpy())
            attack_labels.append(f"{attack} (eps={budget:.3f})")
    plt.figure(figsize=(12,6))
    plt.boxplot(data_to_plot, labels=attack_labels, showfliers=False)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric)
    plt.title(f"Attack transferability evaluation on {dataset_name} dataset")
    plt.tight_layout()
    plt.xticks(attack_budgets, [f"{eps*255:.0f}" for eps in attack_budgets])

    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_transferability_boxplot.png"))
    else:
        plt.show()
 
  
def plot_results(results_dir, attack_budgets, plot_dir, dataset_name, attack_types=None, attack_type=None, all_results=None, results=None, stored_metrics=["final_step_displacement", "final_loss"]):
    # Support both old single-attack and new multi-attack signatures
    if attack_types is None:
        attack_types = [attack_type] if attack_type is not None else []
    if all_results is None:
        if results is not None:
            all_results = {attack_types[0]: results}
        elif results_dir is not None:
            all_results = {}
            for at in attack_types:
                all_results[at] = torch.load(os.path.join(results_dir, f"{dataset_name}_{at}_results.pt"))

    for metric in stored_metrics:
        plt.figure(figsize=(10,8))
        for at, res in all_results.items():
            metric_values = res[metric]
            mean_metric = metric_values.mean(dim=1)
            median_metric = torch.median(metric_values, dim=1).values
            q25_metric = torch.quantile(metric_values, 0.25, dim=1)
            q75_metric = torch.quantile(metric_values, 0.75, dim=1)
            attack_name = "DTD" if at.lower() == "diffusion" else "encoder"
            plt.plot(attack_budgets, mean_metric, linestyle='--', alpha=1.0,linewidth=3.0, label=f"{attack_name} mean")
            # plt.plot(attack_budgets, median_metric, label=f"{at} median")
            plt.fill_between(attack_budgets, q25_metric, q75_metric, alpha=0.3, label= f"{at} IQR (25-75%)")
        plt.xlabel("Attack budget (out of 255)")
        # plt.ylabel("Final step displacement (km)")
        
        #add note about log scale on y axis + quantiles
        
        #x and y axis should be log scale
        plt.xscale("log")
        plt.yscale("log")
        
        
        plt.xticks(attack_budgets, [f"{eps*255:.0f}" for eps in attack_budgets])
        #replace y ticks with nice values (1km, 10km, 100km, 1000km, 10000km)
        plt.yticks([1000, 2500,5000,10000], ["1,000 km", "2500 km","5,000 km", "10,000 km"], rotation = 90, va='center')  # Rotate y-tick labels for better readability
                
        #add grid
        plt.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        
        plt.title(f"{dataset_name} dataset")
        plt.legend(fontsize="x-large", markerscale=2)
        if plot_dir is not None:
            os.makedirs(plot_dir, exist_ok=True)
            suffix = '_'.join(attack_types)
            plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{suffix}_{metric}.png"))
        else:
            plt.show()
        plt.close()
        
        
def plot_attack_success_rate(results_dir, attack_budgets, plot_dir, dataset_name, threshold_km=2500,attack_types=None, attack_type=None, all_results=None, results=None):
    """
    Takes same input as plot_results, but plots attack success rate instead of metrics. Attack success is defined as the fraction of samples for which the final step displacement is above a certain threshold (e.g. 100km), which indicates a successful attack that significantly changes the predicted location.
    
    """
    if attack_types is None:
        attack_types = [attack_type] if attack_type is not None else []
    if all_results is None:
        if results is not None:
            all_results = {attack_types[0]: results}
        elif results_dir is not None:
            all_results = {}
            for at in attack_types:
                all_results[at] = torch.load(os.path.join(results_dir, f"{dataset_name}_{at}_results.pt"))
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
    
    linewidths= np.linspace(0.5,3, len(thresholds))

    def _darken_rgba(color_rgba, factor):
        import matplotlib.colors as mcolors
        rgb = np.asarray(mcolors.to_rgb(color_rgba))
        rgb = np.clip(rgb * factor, 0.0, 1.0)
        return (rgb[0], rgb[1], rgb[2], 1.0)

    plt.figure(figsize=(10, 8))
    attack_names = list(all_results.keys())
    # Use fixed qualitative colors so the first two attacks are clearly distinct (blue, red).
    attack_palette = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#d62728',  # red
        '#2ca02c',  # green
        '#9467bd',  # purple
        '#8c564b',  # brown
    ]

    for attack_index, (at, res) in enumerate(all_results.items()):
        base_color = attack_palette[attack_index % len(attack_palette)]
        final_disp = res["final_step_displacement"]

        success_curves = []
        for t in thresholds:
            success_rate = (final_disp > t).float().mean(dim=1).cpu().numpy()
            success_curves.append(success_rate)

        success_curves = np.asarray(success_curves)
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
            
            attack_name = "DTD" if at.lower() == "diffusion" else "encoder"

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
    # plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25, linestyle='--', linewidth=0.6)
    plt.legend(ncol=2, fontsize='x-large')
    plt.xscale("log")
    plt.xticks(attack_budgets, [f"{eps*255:.0f}" for eps in attack_budgets])


    if plot_dir is not None:
        os.makedirs(plot_dir, exist_ok=True)
        suffix = '_'.join(attack_types)
        plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{suffix}_attack_success_rate.png"))
    else:
        plt.show()
    plt.close()

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
        t = res["attack_results"][attack].detach().cpu()
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
        attack_name = "DTD" if attack.lower() == "diffusion" else "Encoder"
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

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows), sharey='row')
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
            axes[r,0].set_yscale('log')
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor=colors[i], alpha=0.5, label=f"{bucket_names[i]} localizability")
        for i in range(3)
    ]
    fig.legend(handles=legend_handles, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
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


    