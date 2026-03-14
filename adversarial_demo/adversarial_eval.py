"""Evaluation utilities for adversarial attacks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from plonk.metrics.utils import haversine

def trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric = "haversine") -> torch.Tensor:
    """
    Computes distances per step and batch element.
    """
    # Pipeline outputs degrees; haversine expects radians
    gps_traj_source = torch.deg2rad(gps_traj_source.float())
    gps_traj_perturbed = torch.deg2rad(gps_traj_perturbed.float())
    distance = torch.zeros(gps_traj_source.shape[0], gps_traj_source.shape[1])
    for step in range(gps_traj_source.shape[0]):
        distance[step] = haversine(gps_traj_source[step], gps_traj_perturbed[step])
        if metric == "geoscore":
            distance[step] = 5000 * torch.exp(-distance[step] / 1492.7)
    return distance

def mean_trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric = "haversine") -> float:
	"""Mean displacement over all diffusion steps and batch elements."""
	distance = trajectory_displacement(gps_traj_source, gps_traj_perturbed, metric=metric)	
	return float(distance.mean().mean().item())


def mean_final_prediction_distance(gps_coords_source, gps_coords_perturbed, metric="haversine") -> float:
	"""Mean Euclidean distance in degree space between final predictions."""
	distance = trajectory_displacement(gps_coords_source, gps_coords_perturbed, metric=metric)
	return float(distance[-1].mean().item())

def select_displacement_score(
	mean_step_disp: float,
	final_step_disp: float,
	metric_name: str,
) -> float:
	"""Select scalar score from explicit displacement metric choice."""
	if metric_name == "mean_step_displacement":
		return float(mean_step_disp)
	if metric_name == "final_step_displacement":
		return float(final_step_disp)
	raise ValueError(
		f"Unknown displacement metric: {metric_name}. "
		"Expected one of ['mean_step_displacement', 'final_step_displacement']"
	)


def evaluate_displacement_metrics(gps_traj_source, gps_traj_perturbed) -> Dict[str, float]:
	"""Return both canonical displacement metrics for a pair of trajectories"""
	#convert to tensor if numpy (metrics requires tensors)
	if isinstance(gps_traj_source, np.ndarray):
		gps_traj_source = torch.from_numpy(gps_traj_source)
	if isinstance(gps_traj_perturbed, np.ndarray):
		gps_traj_perturbed = torch.from_numpy(gps_traj_perturbed)
	
	mean_step = mean_trajectory_displacement(gps_traj_source, gps_traj_perturbed)
	final_step = mean_final_prediction_distance(gps_traj_source, gps_traj_perturbed)
	return {
		"mean_step_displacement": float(mean_step),
		"final_step_displacement": float(final_step),
	}

def evaluate_source_perturbation(
	pipeline,
	source_image,
	perturbed_image,
	batch_size: int = 256,
	cfg: float = 10.0,
	num_steps: Optional[int] = None,
	seed: Optional[int] = 1234,
	device: str = "cuda",
) -> Dict[str, Any]:
	"""Run source and perturbed images from the same initial noise and compute metrics."""
	generator = torch.Generator(device=device)
	if seed is not None:
		generator.manual_seed(int(seed))
	x_n = torch.randn(int(batch_size), 3, device=device, generator=generator)

	kwargs = {
		"batch_size": int(batch_size),
		"cfg": float(cfg),
		"x_N": x_n,
		"return_trajectories": True,
	}
	if num_steps is not None:
		kwargs["num_steps"] = int(num_steps)

	gps_source, traj_source = pipeline(source_image, **kwargs)
	gps_perturbed, traj_perturbed = pipeline(perturbed_image, **kwargs)

	return {
		"gps_source": gps_source,
		"gps_perturbed": gps_perturbed,
		"traj_source": traj_source,
		"traj_perturbed": traj_perturbed,
		"mean_trajectory_displacement": mean_trajectory_displacement(traj_source, traj_perturbed),
		"mean_final_prediction_distance": mean_final_prediction_distance(gps_source, gps_perturbed),
	}
