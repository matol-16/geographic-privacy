from __future__ import annotations

from typing import Any, Dict, Optional, cast

import numpy as np
from PIL import Image

from adversarial_utils import filter_kwargs_for
from encoder_attacks import train_encoder_attack


def train_diffusion_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	log_every: int = 20,
	target_pure_noise: bool = False,
	dot_product_loss: str = "squared",
	reconstruction_loss_weight: float = 0.0,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
) -> Dict[str, Any]:
	"""Thin wrapper around trajectory_deviation diffusion attack training."""
	import trajectory_deviation as td

	delta, history, source_tensor, best_metrics = td.train_diffusion_perturbation(
		source_image=source_image,
		pipeline=pipeline,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		log_every=log_every,
		target_pure_noise=target_pure_noise,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		print_restart_results=print_restart_results,
		show_progress=show_progress,
		device=device,
	)

	final_loss = float(history[-1]) if len(history) > 0 else float("inf")
	min_loss = float(np.min(history)) if len(history) > 0 else float("inf")

	return {
		"attack_type": "diffusion",
		"delta": delta,
		"history": history,
		"source_tensor": source_tensor,
		"final_loss": final_loss,
		"min_loss": min_loss,
		"best_metrics": best_metrics,
		"config": {
			"n_steps": int(n_steps),
			"train_batch_size": int(train_batch_size),
			"lr": float(lr),
			"eps_max": float(eps_max),
			"anchor_samples": int(anchor_samples),
			"clean_num_steps": int(clean_num_steps),
			"target_pure_noise": bool(target_pure_noise),
			"dot_product_loss": dot_product_loss,
			"reconstruction_loss_weight": float(reconstruction_loss_weight),
			"num_restarts": int(num_restarts),
			"restart_selection_metric": restart_selection_metric,
		},
	}


def run_attack(
	attack_type: str,
	source_image: Image.Image,
	pipeline,
  	target_image = None,
	silent: bool = False,
	**kwargs,
) -> Dict[str, Any]:
	"""
	Run any supported attack from one API.

	You can pass a superset of hyperparameters: only those accepted by the
	chosen attack function will be forwarded (unrecognised keys are ignored).

	Args:
		attack_type: "encoder" or "diffusion".
		source_image: PIL source image to attack.
		pipeline: PLONK pipeline instance.
		silent: If True, suppress all prints and progress bars from the attack.
		**kwargs: forwarded to the corresponding attack function.
	"""
	if silent:
		kwargs.setdefault("print_restart_results", False)
		kwargs.setdefault("show_progress", False)

	aliases = {
		"encoder": "encoder",
		"enc": "encoder",
		"diffusion": "diffusion",
		"diff": "diffusion",
	}
	normalized_type = aliases.get(str(attack_type).lower())
	if normalized_type is None:
		raise ValueError(
			f"Unknown attack_type={attack_type}. Expected one of {sorted(aliases.keys())}"
		)

	if normalized_type == "encoder":
		filtered = filter_kwargs_for(train_encoder_attack, kwargs)
		return train_encoder_attack(source_image=source_image, target_image=target_image, pipeline=pipeline, **filtered)

	filtered = filter_kwargs_for(train_diffusion_attack, kwargs)
	return train_diffusion_attack(source_image=source_image, pipeline=pipeline, **filtered)


def run_attack_and_build_image(
	attack_type: str,
	source_image: Image.Image,
	pipeline,
	**kwargs,
) -> Dict[str, Any]:
	"""
	Run an attack and directly return the perturbed PIL image.

	Returns a dict with:
	  - attack_result: output of run_attack(...)
	  - perturbed_image: PIL image built from source_image + learned delta
	"""
	from adversarial_utils import add_perturbation_to_image

	attack_result = run_attack(
		attack_type=attack_type,
		source_image=source_image,
		pipeline=pipeline,
		**kwargs,
	)
	perturbed_image = add_perturbation_to_image(cast(Any, source_image), attack_result["delta"], pipeline)

	return {
		"attack_result": attack_result,
		"perturbed_image": perturbed_image,
	}

