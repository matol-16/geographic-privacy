import matplotlib.pyplot as plt
from matplotlib import colormaps
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
    ax.set_extent([-20, 180, -10, 90], crs=ccrs.PlateCarree())

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
        mean_disp = np.nanmean(displacement, axis=1)
        var_disp = np.nanvar(displacement, axis=1)
        std_disp = np.sqrt(var_disp)
        steps = np.arange(num_steps)
        disp_ax.plot(steps, mean_disp, color='crimson', linewidth=1.8, label='Mean displacement')
        disp_ax.fill_between(steps, np.maximum(mean_disp - std_disp, 0), mean_disp + std_disp, color='crimson', alpha=0.2, label='±1 std')
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
	if results is None and results_dir is not None:
		results = torch.load(os.path.join(results_dir, f"{dataset_name}_results_transferability.pt"))

	plt.figure(figsize=(10,6))
	for attack, res in results.items():
		mean_metric = res.mean(dim=1)
		std_metric = res.std(dim=1)
		plt.plot(attack_budgets, mean_metric, label=attack)
		plt.fill_between(attack_budgets, mean_metric-std_metric, mean_metric+std_metric, alpha=0.2)

	plt.xlabel("Attack budget (eps)")
	plt.ylabel(metric)
	plt.title(f"Attack transferability evaluation on {dataset_name} dataset")
	plt.legend()
	if plot_dir is not None:
		os.makedirs(plot_dir, exist_ok=True)
		plt.savefig(os.path.join(plot_dir, f"{dataset_name}_transferability.png"))
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
		plt.figure(figsize=(10,6))
		for at, res in all_results.items():
			mean_metric = res[metric].mean(dim=1)
			std_metric = res[metric].std(dim=1)
			plt.plot(attack_budgets, mean_metric, label=at)
			plt.fill_between(attack_budgets, mean_metric-std_metric, mean_metric+std_metric, alpha=0.2)
		plt.xlabel("Attack budget (eps)")
		plt.ylabel(metric)
		plt.title(f"{metric} on {dataset_name} dataset")
		plt.legend()
		if plot_dir is not None:
			os.makedirs(plot_dir, exist_ok=True)
			suffix = '_'.join(attack_types)
			plt.savefig(os.path.join(plot_dir, f"{dataset_name}_{suffix}_{metric}.png"))
		else:
			plt.show()
		plt.close()