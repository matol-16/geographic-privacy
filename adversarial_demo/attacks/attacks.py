"""Unified attack dispatcher and utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image

from utils.adversarial_utils import resolve_torch_device
from utils.adversarial_metrics import final_prediction_distance_to_point
from attacks.encoder_attacks import EncoderAttack
from attacks.trajectory_deviation import DiffusionAttack, ACE, UniDef, TrainingLossAttack, TargetedL2, DTDBlur, CosineTargeted
from attacks.diffusion_attack_salman import SamplingAttack
from attacks.diffusion_attack_truncated import TruncatedSamplingAttack


def _fsd_to_point(gps, point) -> float:
	"""Mean haversine distance (km) from a batch of final predictions ``gps`` [(N, 2) degrees] to ``point``."""
	import numpy as np
	coords = torch.as_tensor(np.asarray(gps), dtype=torch.float32).reshape(-1, 2)
	return final_prediction_distance_to_point(coords.unsqueeze(0), point)


def _run_restartable_attack(
	attack,
	attack_type: str,
	optimizer_fn: Callable[[list], torch.optim.Optimizer],
	show_progress: bool,
	early_stopping_patience: int,
	restart_eval_batch_size: int,
	restart_eval_cfg: float,
	restart_eval_num_steps: Optional[int],
	restart_eval_seed: int,
	num_restart_workers: int,
	finalize_kwargs: Optional[Dict[str, Any]] = None,
	main_num_restarts: Optional[int] = None,
	true_gps: Optional[Any] = None,
) -> Dict[str, Any]:
	"""Train, evaluate, and select the best restart for a restartable attack.

	``true_gps`` (the image's (lat, lon)) adds ``final_step_displacement_true`` -- the FSD, i.e.
	the mean haversine distance between the attacked prediction and the true location -- to
	every restart's metrics, so restarts can be selected by it
	(``restart_selection_metric="final_step_displacement_true"``, the paper's protocol).
	"""
	finalize_kwargs = finalize_kwargs or {}
	if true_gps is not None:
		true_gps = tuple(float(v) for v in true_gps)
	elif attack.restart_manager.selection_metric == "final_step_displacement_true":
		raise ValueError(
			"restart_selection_metric='final_step_displacement_true' needs the image's true GPS "
			"(pass true_gps=(lat, lon))."
		)
	# Restrict the reported best to the first ``main_num_restarts`` restarts; the rest are
	# still evaluated to populate the restart ablation (see RestartManager.update_best).
	if main_num_restarts is not None:
		attack.restart_manager.main_num_restarts = int(main_num_restarts)

	def run_single_restart(restart_idx: int):
		"""Run one restart end-to-end."""
		delta, history = attack.train_restart(
			restart_idx=restart_idx,
			n_steps=attack.n_steps,
			optimizer_fn=optimizer_fn,
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
		if true_gps is not None:
			# Same quantity MetricsCollector reports as the FSD (final step, mean over the batch,
			# haversine to the true location) for the attacked and the clean prediction.
			eval_result["metrics"]["final_step_displacement_true"] = _fsd_to_point(eval_result["gps_perturbed"], true_gps)
			eval_result["metrics"]["final_step_displacement_source_true"] = _fsd_to_point(eval_result["gps_source"], true_gps)

		return restart_idx, delta, history, eval_result

	if num_restart_workers > 1:
		with ThreadPoolExecutor(max_workers=num_restart_workers) as executor:
			results = list(executor.map(run_single_restart, range(attack.restart_manager.num_restarts)))
		for restart_idx, delta, history, eval_result in results:
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=eval_result["metrics"],
				eval_result=eval_result,
			)
	else:
		for restart_idx in range(attack.restart_manager.num_restarts):
			restart_idx, delta, history, eval_result = run_single_restart(restart_idx)
			attack.restart_manager.update_best(
				restart_idx=restart_idx,
				delta=delta,
				history=history,
				metrics=eval_result["metrics"],
				eval_result=eval_result,
			)

	return attack.finalize_result(attack_type=attack_type, **finalize_kwargs)


def _build_shared_x0_bank(
	pipeline,
	source_image: Image.Image,
	anchor_samples: int,
	clean_num_steps: int,
	device: str,
	show_progress: bool,
	x0_source: str = "clean_model",
	true_gps=None,
) -> torch.Tensor:
	"""Build the x0 bank once so it can be reused across all restarts (KEY OPTIMIZATION).

	``x0_source="clean_model"``: ``anchor_samples`` samples of the clean model (legacy).
	``x0_source="true"``: the image's true location only (paper Eq. 1 / 9 / 15); needs ``true_gps``.
	"""
	from attacks.trajectory_deviation import build_x0_bank_from_clean_model, build_x0_bank_from_true_gps

	if x0_source == "true":
		return build_x0_bank_from_true_gps(true_gps, device=device)
	if show_progress:
		print("Building x0 bank (shared across restarts)...")
	return build_x0_bank_from_clean_model(
		pipeline,
		source_image,
		n_samples=anchor_samples,
		num_steps=clean_num_steps,
		cfg=0.0,
		device=device,
	)


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

	Standard specification of each attack key (the settings of the paper's runs; every
	legacy behaviour stays reachable through the explicit options named in brackets):

	  dtd            GeoFA-TD: min E cos(psi(x_t|c+d), psi(x_t|c)); x_t = sqrt(g) x0 + sqrt(1-g) n
	                 with x0 = the image's TRUE location, Euclidean cosine
	                 [forward_process="model" -> geodesic x_t + tangent cosine; x0_source="clean_model"]
	  diffusion_l2   GeoFA-L2: max E ||psi(x_t|c+d) - psi(x_t|c)||^2, same x_t as dtd (metric-only difference)
	  targeted_l2    GeoFA-DPA / L2-targeted: min RFM training loss towards the antipode of the true location
	  sampling       GeoFA-Sampling: max E haversine(x0_hat(x_N), true location) through the 16-step
	                 sampler run WITHOUT guidance (w = 0) [reference="clean_prediction"]
	  sampling_cfg10 GeoFA-Sampling trained through the sampler at the evaluation guidance w = 10
	  encoder        GeoFA-enc.: max ||E(c+d) - E(c)||^2 with the shared sign-SGD PGD loop
	                 [update_rule="adam", delta_init=None -> the original Adam/full-budget-init version]
	  training_loss  GeoFA-AdvDM: max RFM training loss, x_t and target from the TRUE location
	                 [x0_source="clean_model"]
	  unidef         GeoFA-UniDef: training_loss with the finite-difference (FDJE) gradient estimator
	                 [use_fdje=False, x0_source="clean_model"]
	  dtd_blur       GeoFA-TD-blur: dtd + the same loss on a Gaussian-blurred copy (blur EOT)
	  ace            GeoFA-ACE (image-space target), diffusion / truncated_sampling / cosine_targeted:
	                 other baselines and ablations.

	You can pass a superset of hyperparameters: only those accepted by the chosen attack
	are forwarded (unrecognised keys are ignored).

	Args:
		attack_type: one of the keys above (a few short aliases are accepted, see ``aliases``).
		source_image: PIL source image to attack.
		pipeline: PLONK pipeline instance.
		target_image: Target image for targeted attacks ("encoder" targeted mode and "ace").
		silent: If True, suppress all prints and progress bars from the attack.
		**kwargs: forwarded to the attack implementation (incl. ``true_gps`` = the image's
			(lat, lon), which the true-location attacks require).

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

	aliases = {
		"dtd": "dtd",
		"diffusion_cosine_neg": "dtd",  # historical name of GeoFA-TD
		"diffusion_l2": "diffusion_l2",
		"l2": "diffusion_l2",
		"targeted_l2": "targeted_l2",
		"l2_target": "targeted_l2",
		"dpa": "targeted_l2",
		"sampling": "sampling",
		"diffusion_salman": "sampling",  # historical name
		"sampling_cfg10": "sampling_cfg10",
		"encoder": "encoder",
		"enc": "encoder",
		"training_loss": "training_loss",
		"advdm": "training_loss",
		"unidef": "unidef",
		"dtd_blur": "dtd_blur",
		"ace": "ace",
		"diffusion": "diffusion",
		"truncated_sampling": "truncated_sampling",
		"cosine_targeted": "cosine_targeted",
	}
	normalized_type = aliases.get(str(attack_type).lower())
	if normalized_type is None:
		raise ValueError(
			f"Unknown attack_type={attack_type}. Expected one of {sorted(aliases.keys())}"
		)

	if normalized_type == "encoder":
		return _run_encoder_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "dtd":
		kwargs["dot_product_loss"] = "cosine_similarity_negative"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd",
			**kwargs
		)
	if normalized_type == "diffusion_l2":
		kwargs["dot_product_loss"] = "l2"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="diffusion_l2",
			**kwargs
		)
	if normalized_type == "targeted_l2":
		return _run_targeted_l2_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "sampling":
		return _run_diffusion_salman_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "sampling_cfg10":
		# Same objective, but the 16-step sampler that is back-propagated through runs with the
		# evaluation guidance (w = 10; the batch is doubled for the unconditional branch).
		kwargs["sampling_cfg"] = 10.0
		return _run_diffusion_salman_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="sampling_cfg10",
			**kwargs
		)
	if normalized_type == "training_loss":
		return _run_training_loss_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "unidef":
		return _run_unidef_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "dtd_blur":
		kwargs.setdefault("dot_product_loss", "cosine_similarity_negative")
		return _run_dtd_blur_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "ace":
		return _run_ace_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "diffusion":
		# Historical baseline (absolute-dot deviation on the model's own forward process, clean-model x0).
		kwargs.setdefault("forward_process", "model")
		kwargs.setdefault("x0_source", "clean_model")
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "truncated_sampling":
		return _run_truncated_sampling_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "cosine_targeted":
		return _run_cosine_targeted_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	raise ValueError(f"Unhandled normalized attack type: {normalized_type}")


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
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	attack_type_label: str = "encoder",
	update_rule: str = "sign_sgd",  # standard: the shared sign-SGD PGD loop ("adam" = the original Adam version)
	delta_init: Optional[float] = 1e-4,  # standard: tiny init (None => uniform over the whole budget, the Adam version)
	true_gps: Optional[Any] = None,  # (lat, lon) of the image, for the FSD-based restart selection
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
		update_rule=update_rule,
		delta_init=delta_init,
	)
	attack.restart_manager.print_results = print_restart_results
	if update_rule == "sign_sgd":
		optimizer_fn = lambda params: torch.optim.SGD(params, lr=lr)
	else:
		optimizer_fn = lambda params: torch.optim.Adam(params, lr=lr)
	
	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=optimizer_fn,
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
		finalize_kwargs={
			"attack_mode": attack.attack_mode,
			"z_source": attack.z_source,
			"z_target": attack.z_target,
		},
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
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	attack_type_label: str = "diffusion",  # Label stored in result dict
	forward_process: str = "gaussian",  # standard: Gaussian interpolant sqrt(g) x0 + sqrt(1-g) n ("model": the attacked model's own process -- geodesic + tangent metric for RFM)
	x0_source: str = "true",  # standard: the image's true location ("clean_model": a bank of clean-model samples)
	true_gps: Optional[Any] = None,  # (lat, lon) in degrees; required for x0_source="true"
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run diffusion attack with optimizations: shared x0_bank, early stopping, and parallel restarts."""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
		x0_source=x0_source,
		true_gps=true_gps,
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
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,  # Pass shared x0_bank
		forward_process=forward_process,
		x0_source=x0_source,
		true_gps=true_gps,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)


def _run_dtd_blur_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	dot_product_loss: str = "cosine_similarity_negative",
	reconstruction_loss_weight: float = 0.0,
	blur_sigma_min: float = 0.0,
	blur_sigma_max: float = 4.0,
	blur_loss_weight: float = 0.5,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "final_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	attack_type_label: str = "dtd_blur",  # Label stored in result dict
	forward_process: str = "gaussian",  # as dtd (see _run_diffusion_attack)
	x0_source: str = "true",  # as dtd
	true_gps: Optional[Any] = None,  # (lat, lon) in degrees; required for x0_source="true"
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run the blur-robust DTD attack (expectation over Gaussian blur, half clean/half blurred)."""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
		x0_source=x0_source,
		true_gps=true_gps,
	)

	attack = DTDBlur(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		blur_sigma_min=blur_sigma_min,
		blur_sigma_max=blur_sigma_max,
		blur_loss_weight=blur_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,  # Pass shared x0_bank
		forward_process=forward_process,
		x0_source=x0_source,
		true_gps=true_gps,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)


def _run_ace_attack(
	source_image: Image.Image,
	pipeline,
	target_image: Optional[Image.Image] = None,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	dot_product_loss: str = "l2_target",
	reconstruction_loss_weight: float = 0.0,
	alpha: float = 100.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	diagnose_grad_balance: bool = True,  # Print per-term grad norms once to check alpha
	true_gps: Optional[Any] = None,  # (lat, lon) of the image, for the FSD-based restart selection
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run the ACE attack: pull the score and encoding towards a target image.

	Combines a target score-alignment term with an encoder l2 term weighted by ``alpha``.
	"""
	if target_image is None:
		raise ValueError("attack_type='ace' requires a target_image.")

	# Allow the target image to be supplied as a filesystem path (e.g. from config
	# overrides), loading it into a PIL image for preprocessing.
	if isinstance(target_image, str):
		target_image = Image.open(target_image).convert("RGB")

	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
	)

	# Create attack with shared x0_bank
	attack = ACE(
		pipeline=pipeline,
		source_image=source_image,
		target_image=target_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		alpha=alpha,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,  # Pass shared x0_bank
		diagnose_grad_balance=diagnose_grad_balance,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type="ace",
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
		finalize_kwargs={"attack_mode": "targeted"},
	)


def _run_unidef_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	reconstruction_loss_weight: float = 0.0,
	use_fdje: bool = True,
	fd: float = 0.01,
	fdje_direction: str = "embedding",
	fdje_num_samples: int = 1,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "final_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,  # 0=disabled, >0=stop if no improvement for N steps
	num_restart_workers: int = 1,  # Number of parallel workers for restarts; 1=sequential
	attack_type_label: str = "unidef",  # Label stored in result dict
	x0_source: str = "true",  # standard: the image's true location ("clean_model": a bank of clean-model samples)
	true_gps: Optional[Any] = None,  # (lat, lon) in degrees; required for x0_source="true"
	**kwargs,  # Absorb unused kwargs
) -> Dict[str, Any]:
	"""Run the UniDef attack: global trajectory deviation (CDD) with optional FDJE."""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
		x0_source=x0_source,
		true_gps=true_gps,
	)

	attack = UniDef(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		reconstruction_loss_weight=reconstruction_loss_weight,
		use_fdje=use_fdje,
		fd=fd,
		fdje_direction=fdje_direction,
		fdje_num_samples=fdje_num_samples,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,  # Pass shared x0_bank
		x0_source=x0_source,
		true_gps=true_gps,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)


def _run_training_loss_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	reconstruction_loss_weight: float = 0.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "final_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	attack_type_label: str = "training_loss",
	x0_source: str = "true",  # standard: the image's true location ("clean_model": a bank of clean-model samples)
	true_gps: Optional[Any] = None,
	**kwargs,
) -> Dict[str, Any]:
	"""Run the training-loss attack: maximize the model's own training loss."""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
		x0_source=x0_source,
		true_gps=true_gps,
	)

	attack = TrainingLossAttack(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		reconstruction_loss_weight=reconstruction_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,
		x0_source=x0_source,
		true_gps=true_gps,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)


def _run_targeted_l2_attack(
	source_image: Image.Image,
	pipeline,
	true_gps: Optional[Any] = None,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	reconstruction_loss_weight: float = 0.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "final_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	**kwargs,
) -> Dict[str, Any]:
	"""Run the targeted-L2 attack: minimize the training loss towards the antipodal target.

	Steers the model towards the point on Earth farthest from the image's real coordinates
	(``true_gps``, its antipode). When ``true_gps`` is omitted the attack falls back to the
	antipode of the clean model's mean predicted location.
	"""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
	)

	attack = TargetedL2(
		pipeline=pipeline,
		source_image=source_image,
		true_gps=true_gps,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		reconstruction_loss_weight=reconstruction_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type="targeted_l2",
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
		finalize_kwargs={"attack_mode": "targeted"},
	)


def _run_cosine_targeted_attack(
	source_image: Image.Image,
	pipeline,
	true_gps: Optional[Any] = None,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	clean_num_steps: int = 200,
	reconstruction_loss_weight: float = 0.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "final_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	**kwargs,
) -> Dict[str, Any]:
	"""Run the cosine-targeted attack: direction-only steering towards the antipodal target.

	Same target as ``targeted_l2`` (the point farthest from ``true_gps``, its antipode), but
	aligns only the DIRECTION of the model's prediction with the target velocity (negative
	cosine similarity) instead of matching it in L2. Falls back to the antipode of the clean
	model's mean prediction when ``true_gps`` is omitted.
	"""
	x0_bank = _build_shared_x0_bank(
		pipeline,
		source_image,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		device=device,
		show_progress=show_progress,
	)

	attack = CosineTargeted(
		pipeline=pipeline,
		source_image=source_image,
		true_gps=true_gps,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		clean_num_steps=clean_num_steps,
		reconstruction_loss_weight=reconstruction_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		x0_bank=x0_bank,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type="cosine_targeted",
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
		finalize_kwargs={"attack_mode": "targeted"},
	)


def _run_diffusion_salman_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	sampling_steps_salman: int = 16,
	clean_num_steps: int = 100,
	target_pure_noise: bool = False,
	dot_product_loss: str = "absolute",
	reconstruction_loss_weight: float = 0.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	attack_type_label: str = "sampling",
	reference: str = "true_location",  # standard: deviate from the TRUE location (paper Eq. 8); "clean_prediction": from the clean model's prediction
	true_gps: Optional[Any] = None,
	sampling_cfg: float = 0.0,  # guidance inside the attacked 16-step sampler: 0 for "sampling", 10 for "sampling_cfg10"
	**kwargs,
) -> Dict[str, Any]:
	"""Run the Salman diffusion attack through the full sampling trajectory."""
	attack = SamplingAttack(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		sampling_steps_salman=sampling_steps_salman,
		clean_num_steps=clean_num_steps,
		target_pure_noise=target_pure_noise,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
		reference=reference,
		true_gps=true_gps,
		sampling_cfg=sampling_cfg,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type=attack_type_label,
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)


def _run_truncated_sampling_attack(
	source_image: Image.Image,
	pipeline,
	n_steps: int = 400,
	train_batch_size: int = 64,
	lr: float = 2e-2,
	eps_max: float = 1.0,
	anchor_samples: int = 256,
	total_sampling_steps: int = 250,
	backprop_steps: int = 16,
	clean_num_steps: int = 100,
	target_pure_noise: bool = False,
	dot_product_loss: str = "absolute",
	reconstruction_loss_weight: float = 0.0,
	delta_init: float = 1e-4,
	num_restarts: int = 1,
	restart_selection_metric: str = "mean_step_displacement",
	restart_eval_batch_size: int = 256,
	restart_eval_cfg: float = 10.0,
	restart_eval_num_steps: Optional[int] = None,
	main_num_restarts: Optional[int] = None,
	restart_eval_seed: int = 1234,
	print_restart_results: bool = True,
	show_progress: bool = True,
	device: str = "cuda",
	early_stopping_patience: int = 0,
	num_restart_workers: int = 1,
	true_gps: Optional[Any] = None,  # (lat, lon) of the image, for the FSD-based restart selection
	**kwargs,
) -> Dict[str, Any]:
	"""Run the sampling attack with truncated backprop through the tail of a long trajectory.

	Runs a ``total_sampling_steps`` (default 250) trajectory in the forward pass but keeps
	the autograd graph only for the final ``backprop_steps`` (default 16) steps, bounding
	activation memory while preserving a high-fidelity sampling path.
	"""
	attack = TruncatedSamplingAttack(
		pipeline=pipeline,
		source_image=source_image,
		n_steps=n_steps,
		train_batch_size=train_batch_size,
		lr=lr,
		eps_max=eps_max,
		anchor_samples=anchor_samples,
		total_sampling_steps=total_sampling_steps,
		backprop_steps=backprop_steps,
		clean_num_steps=clean_num_steps,
		target_pure_noise=target_pure_noise,
		dot_product_loss=dot_product_loss,
		reconstruction_loss_weight=reconstruction_loss_weight,
		delta_init=delta_init,
		num_restarts=num_restarts,
		restart_selection_metric=restart_selection_metric,
		device=device,
	)
	attack.restart_manager.print_results = print_restart_results

	return _run_restartable_attack(
		attack=attack,
		attack_type="truncated_sampling",
		optimizer_fn=lambda params: torch.optim.SGD(params, lr=lr),
		show_progress=show_progress,
		early_stopping_patience=early_stopping_patience,
		restart_eval_batch_size=restart_eval_batch_size,
		restart_eval_cfg=restart_eval_cfg,
		restart_eval_num_steps=restart_eval_num_steps,
		restart_eval_seed=restart_eval_seed,
		num_restart_workers=num_restart_workers,
		main_num_restarts=main_num_restarts,
		true_gps=true_gps,
	)

