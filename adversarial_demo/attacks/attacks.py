"""Unified attack dispatcher and utilities."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

import torch
from PIL import Image

from utils.adversarial_utils import resolve_torch_device
from attacks.encoder_attacks import EncoderAttack
from attacks.trajectory_deviation import DiffusionAttack, ACE, UniDef, TrainingLossAttack, TargetedL2, DTDBlur, CosineTargeted
from attacks.diffusion_attack_salman import SamplingAttack
from attacks.diffusion_attack_truncated import TruncatedSamplingAttack


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
) -> Dict[str, Any]:
	"""Train, evaluate, and select the best restart for a restartable attack."""
	finalize_kwargs = finalize_kwargs or {}
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

	You can pass a superset of hyperparameters: only those accepted by the
	chosen attack class will be forwarded (unrecognised keys are ignored).

	Args:
		attack_type: "encoder", "diffusion", "ace", "sampling", or "diffusion_salman" (alias).
		source_image: PIL source image to attack.
		pipeline: PLONK pipeline instance.
		target_image: Target image for targeted attacks ("encoder" and "ace").
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

	# Optional version pin: ``attack_variant`` (config ``per_attack.<key>.attack_variant``)
	# runs the attack under a canonical key (``dtd``, ``diffusion_l2``, ...) with the
	# implementation of another key (``dtd_v3_old_forward``, ``diffusion_l2_v3``, ...).
	# Results / plots keep the canonical name; the variant is recorded in the result
	# (``attack_type_label`` / ``config``) and in the saved run config.
	attack_variant = kwargs.pop("attack_variant", None)
	dispatch_type = attack_variant if attack_variant else attack_type

	# Normalize attack type
	aliases = {
		"encoder": "encoder",
		"enc": "encoder",
		"diffusion": "diffusion",
		"diff": "diffusion",
		"sampling": "sampling",
		"diffusion_salman": "sampling",  # backward-compatible alias for Sampling
		"salman": "sampling",            # backward-compatible alias for Sampling
		"truncated_sampling": "truncated_sampling",
		"sampling_truncated": "truncated_sampling",
		"salman_truncated": "truncated_sampling",
		"truncated": "truncated_sampling",
		"diffusion_l2": "diffusion_l2",
		"l2": "diffusion_l2",
		"diffusion_l2_v2": "diffusion_l2_v2",  # GeoFA-L2 with the geodesic x_t / tangent L2 (paper Eq. 12)
		"l2_v2": "diffusion_l2_v2",
		"diffusion_l2_v3": "diffusion_l2_v3",  # l2_v2 + x_t noised from the TRUE location
		"l2_v3": "diffusion_l2_v3",
		"diffusion_l2_v3_old_forward": "diffusion_l2_v3_old_forward",  # v1 (Gaussian x_t, Euclidean L2) + TRUE x0: the L2 twin of dtd_v3_old_forward
		"l2_v3_old_forward": "diffusion_l2_v3_old_forward",
		"sampling_v2": "sampling_v2",  # GeoFA-Sampling deviating from the TRUE location (paper Eq. 8)
		"encoder_v3": "encoder_v3",  # GeoFA-enc. with the shared sign-SGD PGD loop (paper Sec. 4.1 / Eq. 7)
		"enc_v3": "encoder_v3",
		"dtd": "dtd",
		"diffusion_cosine_neg": "dtd",  # backward-compatible alias for DTD
		"dtd_v2": "dtd_v2",  # GeoFA-TD with the model's own (geodesic) forward process
		"dtdv2": "dtd_v2",
		"dtd_v3": "dtd_v3",  # dtd_v2 + x_t noised from the TRUE location (paper Eq. 1 / 15)
		"dtdv3": "dtd_v3",
		"dtd_v3_old_forward": "dtd_v3_old_forward",  # v1 (Gaussian x_t) + TRUE x0: isolates the x0 change
		"dtdv3oldforward": "dtd_v3_old_forward",
		"training_loss_v3": "training_loss_v3",  # GeoFA-AdvDM + x_t / target from the TRUE location (Eq. 9)
		"traininglossv3": "training_loss_v3",
		"advdm_v3": "training_loss_v3",
		"dtd_blur": "dtd_blur",
		"dtdblur": "dtd_blur",
		"dtd_blur_v3_old_forward": "dtd_blur_v3_old_forward",  # dtd_blur on top of dtd_v3_old_forward (Gaussian x_t + TRUE x0)
		"dtdblurv3oldforward": "dtd_blur_v3_old_forward",
		"ace": "ace",
		"diffusion_target": "ace",
		"dtd_target": "ace",
		"diff_target": "ace",
		"unidef": "unidef",
		"uni_def": "unidef",
		"cdd": "unidef",
		"unidef_nofdje": "unidef_nofdje",
		"unidef_v3": "unidef_v3",  # UniDef with x_t / target from the TRUE location (like training_loss_v3)
		"unidefv3": "unidef_v3",
		"training_loss": "training_loss",
		"targeted_l2": "targeted_l2",
		"targetedl2": "targeted_l2",
		"l2_target": "targeted_l2",
		"cosine_targeted": "cosine_targeted",
		"cosinetargeted": "cosine_targeted",
		"cosine_target": "cosine_targeted",
	}
	normalized_type = aliases.get(str(dispatch_type).lower())
	if normalized_type is None:
		raise ValueError(
			f"Unknown attack_type={dispatch_type}. Expected one of {sorted(aliases.keys())}"
		)
	if attack_variant and aliases.get(str(attack_type).lower()) is None:
		raise ValueError(f"Unknown attack_type={attack_type} (pinned to variant {attack_variant!r})")

	# Run the appropriate attack
	if normalized_type == "encoder":
		return _run_encoder_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "diffusion":
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "diffusion_l2":
		# GeoFA-L2 v1 (legacy): Gaussian x_t regardless of the model; see diffusion_l2_v2 for Eq. 12.
		kwargs["dot_product_loss"] = "l2"
		kwargs.setdefault("forward_process", "gaussian")
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="diffusion_l2",
			**kwargs
		)
	if normalized_type == "diffusion_l2_v2":
		# GeoFA-L2 v2: paper Eq. 12 as written -- (x_t, gamma) from the model's own geodesic
		# forward process and the L2 distance taken between tangent-projected velocities.
		kwargs["dot_product_loss"] = "l2"
		kwargs["forward_process"] = "model"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="diffusion_l2_v2",
			**kwargs
		)
	if normalized_type == "diffusion_l2_v3":
		# GeoFA-L2 v3: l2_v2 with x_t noised from the image's TRUE location (Eq. 1 x0).
		kwargs["dot_product_loss"] = "l2"
		kwargs["forward_process"] = "model"
		kwargs["x0_source"] = "true"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="diffusion_l2_v3",
			**kwargs
		)
	if normalized_type == "diffusion_l2_v3_old_forward":
		# GeoFA-L2 with exactly the forward process of dtd_v3_old_forward: legacy Gaussian
		# interpolant sqrt(gamma) x0 + sqrt(1-gamma) n with x0 = the TRUE location, Euclidean L2
		# between the raw velocity predictions (no tangent projection). Differs from that TD pin
		# by the metric only.
		kwargs["dot_product_loss"] = "l2"
		kwargs["forward_process"] = "gaussian"
		kwargs["x0_source"] = "true"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="diffusion_l2_v3_old_forward",
			**kwargs
		)
	if normalized_type == "sampling_v2":
		# GeoFA-Sampling v2: maximise the haversine distance between the perturbed prediction and
		# the image's TRUE location (paper Eq. 8); v1 deviates from the clean prediction instead.
		kwargs["reference"] = "true_location"
		return _run_diffusion_salman_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="sampling_v2",
			**kwargs
		)
	if normalized_type == "encoder_v3":
		# GeoFA-enc. v3: same untargeted objective, but optimised with the PGD loop every other
		# GeoFA attack uses (sign-SGD, squared L2 of Eq. 7, tiny init, encoder in eval mode) --
		# v1 used Adam on the unsquared distance with a full-budget uniform init.
		kwargs["update_rule"] = "sign_sgd"
		kwargs.setdefault("delta_init", 1e-4)
		return _run_encoder_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			attack_type_label="encoder_v3",
			**kwargs
		)
	if normalized_type == "dtd":
		# DTD (Diffusion Trajectory Deviation): raw (signed) cosine-similarity
		# objective between the clean and perturbed eps predictions.
		# v1 (legacy): Gaussian x_t regardless of the model; dtd_v2 / dtd_v3 use the model's own.
		kwargs["dot_product_loss"] = "cosine_similarity_negative"
		kwargs.setdefault("forward_process", "gaussian")
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd",
			**kwargs
		)
	if normalized_type == "dtd_v2":
		# GeoFA-TD v2: same cosine objective, but (x_t, gamma) are drawn from the model's own
		# forward process (geodesic on S^2 for RFM) and velocities are compared in the tangent
		# space at x_t, as written in the paper (Eq. 1 / Eq. 15). v1 ("dtd") uses the legacy
		# Gaussian interpolant sqrt(gamma) x0 + sqrt(1-gamma) n, which leaves the sphere.
		kwargs["dot_product_loss"] = "cosine_similarity_negative"
		kwargs["forward_process"] = "model"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd_v2",
			**kwargs
		)
	if normalized_type == "dtd_v3":
		# GeoFA-TD v3 = v2 (geodesic x_t, tangent cosine) with x_t noised from the image's TRUE
		# location instead of samples of the clean model (paper Eq. 1: x_t = exp_{x0}(k(t) log_{x0} eps)
		# with x0 the true location). The reference is still the clean predicted velocity psi(x_t|c).
		kwargs["dot_product_loss"] = "cosine_similarity_negative"
		kwargs["forward_process"] = "model"
		kwargs["x0_source"] = "true"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd_v3",
			**kwargs
		)
	if normalized_type == "dtd_v3_old_forward":
		# Ablation isolating the x0 change: GeoFA-TD v1 (legacy Gaussian interpolant
		# sqrt(gamma) x0 + sqrt(1-gamma) n, Euclidean cosine) with x0 = the TRUE location.
		# v3 - v3_old_forward = effect of the geodesic path; v3_old_forward - v1 = effect of x0.
		kwargs["dot_product_loss"] = "cosine_similarity_negative"
		kwargs["forward_process"] = "gaussian"
		kwargs["x0_source"] = "true"
		return _run_diffusion_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd_v3_old_forward",
			**kwargs
		)
	if normalized_type == "training_loss_v3":
		# GeoFA-AdvDM v3: the RFM training loss with x_t and the target velocity v(x_t) built from
		# the image's TRUE location (paper Eq. 9), not from clean-model samples.
		kwargs["x0_source"] = "true"
		return _run_training_loss_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="training_loss_v3",
			**kwargs
		)
	if normalized_type == "dtd_blur":
		# Blur-robust DTD: cosine-neg trajectory deviation trained with an
		# expectation-over-Gaussian-blur objective (half clean, half blurred).
		kwargs.setdefault("dot_product_loss", "cosine_similarity_negative")
		return _run_dtd_blur_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "dtd_blur_v3_old_forward":
		# GeoFA-TD-blur on top of dtd_v3_old_forward: the same blur-EOT objective, but with
		# the legacy Gaussian interpolant and x0 = the image's TRUE location (Eq. 1 x0), so it
		# differs from dtd_v3_old_forward by the blur term only.
		kwargs.setdefault("dot_product_loss", "cosine_similarity_negative")
		kwargs["forward_process"] = "gaussian"
		kwargs["x0_source"] = "true"
		return _run_dtd_blur_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="dtd_blur_v3_old_forward",
			**kwargs
		)
	if normalized_type == "sampling":
		return _run_diffusion_salman_attack(
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
	if normalized_type == "ace":
		return _run_ace_attack(
			source_image=source_image,
			target_image=target_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "unidef":
		return _run_unidef_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "unidef_v3":
		# GeoFA-UniDef with x_t and the target velocity built from the TRUE location (as
		# training_loss_v3), so it differs from GeoFA-AdvDM by the gradient estimator only.
		kwargs["x0_source"] = "true"
		return _run_unidef_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="unidef_v3",
			**kwargs
		)
	if normalized_type == "unidef_nofdje":
		kwargs["use_fdje"] = False
		return _run_unidef_attack(
			source_image=source_image,
			pipeline=pipeline,
			attack_type_label="unidef_nofdje",
			**kwargs
		)
	if normalized_type == "training_loss":
		return _run_training_loss_attack(
			source_image=source_image,
			pipeline=pipeline,
			**kwargs
		)
	if normalized_type == "targeted_l2":
		return _run_targeted_l2_attack(
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
	update_rule: str = "adam",  # "adam" (v1) or "sign_sgd" (the shared PGD loop, v3)
	delta_init: Optional[float] = None,  # None => uniform over the whole budget (v1)
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
	forward_process: str = "model",  # "model": the attacked model's own forward process (geodesic for RFM, DDPM / linear for diffusion / flow); "gaussian": legacy v1
	x0_source: str = "clean_model",  # "clean_model" (legacy bank) or "true" (the image's true location)
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
	forward_process: str = "model",  # see DiffusionAttack ("model" or legacy "gaussian")
	x0_source: str = "clean_model",  # "clean_model" (legacy bank) or "true" (the image's true location)
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
	x0_source: str = "clean_model",  # "clean_model" (legacy bank) or "true" (the image's true location)
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
	x0_source: str = "clean_model",  # "clean_model" (legacy bank) or "true" (the image's true location)
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
	reference: str = "clean_prediction",  # "clean_prediction" (v1) or "true_location" (paper Eq. 8, v2)
	true_gps: Optional[Any] = None,
	sampling_cfg: float = 0.0,  # guidance inside the attacked 16-step sampler (0 = v1; cluster_full uses 10)
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
	)

