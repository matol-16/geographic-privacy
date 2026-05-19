"""Unified attack dispatcher and utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import torch
from PIL import Image

from adversarial_utils import filter_kwargs_for, add_perturbation_to_image, resolve_torch_device
from encoder_attacks import EncoderAttack
from trajectory_deviation import DiffusionAttack


def run_attack(
	attack_type: str,
	source_image: Image.Image,
	pipeline,
	target_image: Optional[Image.Image] = None,
	silent: bool = False,
	**kwargs,
) -> Dict[str, Any]:
	"""
	Run any supported attack from one API.

	You can pass a superset of hyperparameters: only those accepted by the
	chosen attack class will be forwarded (unrecognised keys are ignored).

	Args:
		attack_type: "encoder" or "diffusion".
		source_image: PIL source image to attack.
		pipeline: PLONK pipeline instance.
		target_image: Target image for targeted encoder attacks.
		silent: If True, suppress all prints and progress bars from the attack.
		**kwargs: forwarded to the attack implementation.
	
	Returns:
		Standardized attack result dict with attack_type, delta, history, source_tensor, 
		best_metrics, config, etc.
	"""
	if silent:
		kwargs.setdefault("print_restart_results", False)
		kwargs.setdefault("show_progress", False)

	# Resolve device to handle CUDA fallback
	if "device" in kwargs:
		kwargs["device"] = resolve_torch_device(kwargs["device"])
	else:
		kwargs["device"] = resolve_torch_device("cuda")

	# Normalize attack type
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

	# Run the appropriate attack
	if normalized_type == "encoder":
		return _run_encoder_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			**kwargs
		)
	else:
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)


def _run_encoder_attack(
	source_image: Image.Image,
	pipeline,
	target_image: Optional[Image.Image] = None,
	n_steps: int = 100,
	lr: float = 0.1,
	eps_max: float = 0.1,
	device: str = "cuda",
	criterion_name: str = "MSE",
	l_z: float = 1.0,
	l_x: float = 1.0,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run encoder-space attack with optional optimizations."""
	attack = EncoderAttack(
		pipeline=pipeline,
		source_image=source_image,
		target_image=target_image,
		n_steps=n_steps,
		lr=lr,
		eps_max=eps_max,
		criterion_name=criterion_name,
		l_z=l_z,
		l_x=l_x,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
	)
	attack.restart_manager.print_results = print_restart_results
	
	def run_single_restart(restart_idx):
		"""Helper: run training and evaluation for a single restart."""
		delta, history = attack.train_restart(
			restart_idx=restart_idx,
			n_steps=n_steps,
			optimizer_fn=lambda params: torch.optim.Adam(params, lr=lr),
			show_progress=show_progress,
			early_stopping_patience=early_stopping_patience,
		)
		
		eval_result = attack.evaluate_restart(
			delta=delta,
			restart_idx=restart_idx,
			batch_size=restart_eval_batch_size,
			cfg=restart_eval_cfg,
			num_steps=restart_eval_num_steps,
			seed=restart_eval_seed,
		)
		
		return restart_idx, delta, history, eval_result["metrics"]
	
	# Run restarts (parallel or sequential based on num_restart_workers)
	if num_restart_workers > 1:
		# Parallel execution
		with ThreadPoolExecutor(max_workers=num_restart_workers) as executor:
			results = list(executor.map(run_single_restart, range(attack.restart_manager.num_restarts)))
		
		# Update restart_manager sequentially with results (maintains order and thread safety)
		for restart_idx, delta, history, metrics in results:
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=metrics,
			)
	else:
		# Sequential execution
		for restart_idx in range(attack.restart_manager.num_restarts):
			restart_idx, delta, history, metrics = run_single_restart(restart_idx)
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=metrics,
			)
	
	return attack.finalize_result(
		attack_type="encoder",
		attack_mode=attack.attack_mode,
		z_source=attack.z_source,
		z_target=attack.z_target,
	)


def _run_diffusion_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	target_pure_noise: bool = False,
	dot_product_loss: str = "absolute",
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
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run diffusion attack with optimizations: shared x0_bank, early stopping, and parallel restarts."""
	from trajectory_deviation import build_x0_bank_from_clean_model
	
	# Build x0_bank once and reuse across all restarts (KEY OPTIMIZATION)
	if show_progress:
		print("Building x0 bank (shared across restarts)...")
	x0_bank = build_x0_bank_from_clean_model(
		pipeline,
		source_image,
		n_samples=anchor_samples,
		num_steps=clean_num_steps,
		cfg=0.0,
		device=device,
	)
	
	# Create attack with shared x0_bank
	attack = DiffusionAttack(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		target_pure_noise=target_pure_noise,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,  # Pass shared x0_bank
	)
	attack.restart_manager.print_results = print_restart_results
	
	def run_single_restart(restart_idx):
		"""Helper: run training and evaluation for a single restart."""
		delta, history = attack.train_restart(
			restart_idx=restart_idx,
			n_steps=n_steps,
			optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
			show_progress=show_progress,
			early_stopping_patience=early_stopping_patience,
		)
		
		eval_result = attack.evaluate_restart(
			delta=delta,
			restart_idx=restart_idx,
			batch_size=restart_eval_batch_size,
			cfg=restart_eval_cfg,
			num_steps=restart_eval_num_steps,
			seed=restart_eval_seed,
		)
		
		return restart_idx, delta, history, eval_result["metrics"]
	
	# Run restarts (parallel or sequential based on num_restart_workers)
	if num_restart_workers > 1:
		# Parallel execution
		with ThreadPoolExecutor(max_workers=num_restart_workers) as executor:
			results = list(executor.map(run_single_restart, range(attack.restart_manager.num_restarts)))
		
		# Update restart_manager sequentially with results (maintains order and thread safety)
		for restart_idx, delta, history, metrics in results:
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=metrics,
			)
	else:
		# Sequential execution
		for restart_idx in range(attack.restart_manager.num_restarts):
			restart_idx, delta, history, metrics = run_single_restart(restart_idx)
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=metrics,
			)
	
	return attack.finalize_result(attack_type="diffusion")


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
	attack_result = run_attack(
		attack_type=attack_type,
		source_image=source_image,
		pipeline=pipeline,
		**kwargs,
	)
	perturbed_image = add_perturbation_to_image(source_image, attack_result["delta"], pipeline)

	return {
		"attack_result": attack_result,
		"perturbed_image": perturbed_image,
	}

