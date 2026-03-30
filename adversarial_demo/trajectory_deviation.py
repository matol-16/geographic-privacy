

from plonk.pipe import PlonkPipeline
from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import tqdm as tqdm
import numpy as np
from itertools import product
from adversarial_metrics import (
    evaluate_displacement_metrics,
    select_displacement_score,
    mean_final_prediction_distance,
)
from adversarial_utils import (
    add_perturbation_to_image,
    conditional_preprocessing,
    compute_embedding,
    run_paired_pipeline_with_shared_noise,
)

############################################################################################
# Attempt 1 training pipeline: universal perturbation on a single source image
# Objective: min E_{x0, eps, t} <psi(x_t | c+delta), eps>^2
# where x_t = sqrt(gamma_t) * x0 + sqrt(1-gamma_t) * eps


def _compute_dot_alignment_loss(eps_reference, eps_prediction, dot_product_loss="squared"):
    """Compute alignment loss from dot products between reference and predicted directions."""
    metric_aliases = {
        "square": "squared",
        "squared": "squared",
        "squared_dot": "squared",
        "abs": "absolute",
        "absolute": "absolute",
        "absolute_dot": "absolute",
    }
    normalized_metric = metric_aliases.get(str(dot_product_loss).lower())
    if normalized_metric is None:
        raise ValueError(
            f"Unknown dot_product_loss: {dot_product_loss}. Expected one of ['squared', 'absolute']"
        )

    dot = torch.sum(eps_reference * eps_prediction, dim=-1)
    if normalized_metric == "squared":
        return (dot ** 2).mean()
    return torch.abs(dot).mean()

def build_x0_bank_from_clean_model(
    pipeline,
    source_image,
    n_samples=256,
    num_steps=200,
    cfg=0.0,
    device="cuda",
):
    """
    Build a bank of plausible x0 states by sampling the clean model on the source image.
    This approximates expectation over x0 in the objective.
    """
    with torch.no_grad():
        gps_samples = pipeline(
            source_image,
            batch_size=n_samples,
            num_steps=num_steps,
            cfg=cfg,
        )
    if isinstance(gps_samples, tuple):
        gps_samples = gps_samples[0]
    return _gps_degrees_to_cartesian(gps_samples, device=device)


def train_diffusion_perturbation(
    source_image,
    pipeline,
    n_steps=400,
    train_batch_size=64,
    lr=2e-2,
    eps_max=1.0,
    anchor_samples=256,
    clean_num_steps=200,
    log_every=20,
    target_pure_noise = False,
    dot_product_loss="absolute",
    reconstruction_loss_weight=0.0,
    num_restarts=1,
    restart_selection_metric="mean_step_displacement",
    restart_eval_batch_size=256,
    restart_eval_cfg=10.0,
    restart_eval_num_steps=None,
    restart_eval_seed=1234,
    print_restart_results=True,
    show_progress=True,
    device="cuda",
):
    # Freeze PLONK denoiser and embedding model parameters (we only optimize delta)
    pipeline.network.eval().requires_grad_(False)
    pipeline.cond_preprocessing.emb_model.eval().requires_grad_(False)

    # Preprocess once; perturbation is optimized in normalized-image space
    #Preprocess depends on the embedder
    source_tensor = conditional_preprocessing(source_image, pipeline, device=device)

    # Approximate x0 distribution for this source image
    x0_bank = build_x0_bank_from_clean_model(
        pipeline,
        source_image,
        n_samples=anchor_samples,
        num_steps=clean_num_steps,
        cfg=0.0,
    )  # [N, 3]

    if int(num_restarts) < 1:
        raise ValueError("num_restarts must be >= 1")

    restart_summaries = []
    best_score = -float("inf")
    best_delta = None
    best_history = None
    best_restart = None
    best_metrics=None

    for restart_idx in range(int(num_restarts)):
        # Universal perturbation parameter (fresh start at each restart)
        delta = torch.zeros_like(source_tensor, requires_grad=True)
        #we use sign sgd
        optimizer = torch.optim.SGD([delta], lr=lr)

        history = []
        if show_progress:
            pbar = tqdm.trange(n_steps, desc=f"PGD attack training (restart {restart_idx + 1}/{int(num_restarts)})")
        else:
            pbar = range(n_steps)

        for step in pbar:
            optimizer.zero_grad(set_to_none=True)

            # Sample x0 from bank, noise eps, and continuous t ~ U(0,1)
            idx = torch.randint(0, x0_bank.shape[0], (train_batch_size,), device=device)
            x0 = x0_bank[idx]
            eps = torch.randn_like(x0)

            t = torch.rand(train_batch_size, device=device)
            gamma = pipeline.scheduler(t)  # [B]

            x_t = (
                torch.sqrt(gamma).unsqueeze(-1) * x0
                + torch.sqrt(1.0 - gamma).unsqueeze(-1) * eps
            )

            # Compute conditional embedding of perturbed source image
            perturbed_source = source_tensor + delta
            
            #Conditional embedding depends on the embedder
            emb = compute_embedding(
                perturbed_source,
                train_batch_size,
                pipeline,
                device=device,
                track_grad=True,
            )

            # Denoiser epsilon prediction, using PLONK expected batch keys
            model_batch_perturbed = {
                "y": x_t,
                "emb": emb,
                "gamma": gamma,
            }
            eps_pred_perturbed = pipeline.model(model_batch_perturbed)

            if not target_pure_noise:
                #compute conditional embedding of unperturbed source image
                emb_source = compute_embedding(
                    source_tensor,
                    train_batch_size,
                    pipeline,
                    device=device,
                    track_grad=False,
                )

                model_batch = {
                    "y": x_t,
                    "emb": emb_source,
                    "gamma": gamma,
                }
                eps_pred = pipeline.model(model_batch)
                eps= eps_pred


            # Alignment objective on dot product between reference and predicted directions.
            loss = _compute_dot_alignment_loss(eps, eps_pred_perturbed, dot_product_loss=dot_product_loss)

            if reconstruction_loss_weight > 0:
                # Add image reconstruction loss to ensure perturbation does not degrade image quality too much
                loss_x = torch.nn.functional.l1_loss(perturbed_source, source_tensor)
                loss= loss + reconstruction_loss_weight*loss_x


            loss.backward()

            #perform sign sgd in the l inf ball of radius eps_max:
            with torch.no_grad():
                delta.grad = torch.sign(delta.grad)
                optimizer.step()
                delta.data = torch.clamp(delta.data, -eps_max, eps_max)
                delta.grad.zero_()


            history.append(loss.item())
            if show_progress and hasattr(pbar, 'set_postfix') and (step + 1) % log_every == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        # Evaluate restart quality on trajectory displacement with a shared initial noise.
        perturbed_image = add_perturbation_to_image(source_image, delta.detach(), pipeline)
        eval_result = run_paired_pipeline_with_shared_noise(
            pipeline=pipeline,
            source_image=source_image,
            perturbed_image=perturbed_image,
            batch_size=int(restart_eval_batch_size),
            cfg=float(restart_eval_cfg),
            num_steps=restart_eval_num_steps,
            seed=int(restart_eval_seed) + restart_idx,
            device=device,
        )
        metrics = eval_result["metrics"]
        mean_displacement = metrics["mean_step_displacement"]
        final_displacement = metrics["final_step_displacement"]
        score = select_displacement_score(
            mean_step_disp=mean_displacement,
            final_step_disp=final_displacement,
            metric_name=restart_selection_metric,
        )

        summary = {
            "restart": restart_idx,
            "mean_step_displacement": float(mean_displacement),
            "final_step_displacement": float(final_displacement),
            "score": float(score),
            "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
            "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
        }
        restart_summaries.append(summary)

        if print_restart_results:
            print(
                f"[restart {restart_idx + 1}/{int(num_restarts)}] "
                f"final_loss={summary['final_loss']:.6f}, "
                f"min_loss={summary['min_loss']:.6f}, "
                f"mean_step_disp={summary['mean_step_displacement']:.6f}, "
                f"final_step_disp={summary['final_step_displacement']:.6f}, "
                f"selection_score={summary['score']:.6f}"
            )

        if score > best_score:
            best_score = float(score)
            best_delta = delta.detach().clone()
            best_history = list(history)
            best_restart = restart_idx
            best_metrics=metrics

    if best_delta is None or best_history is None or best_restart is None:
        raise RuntimeError("No restart produced a valid perturbation")

    if print_restart_results and int(num_restarts) > 1:
        print(
            f"Selected restart {best_restart + 1}/{int(num_restarts)} using "
            f"{restart_selection_metric} (score={best_score:.6f})"
        )

    return best_delta, best_history, source_tensor, best_metrics


###########################################################################################################


def build_diffusion_hparam_grid(
    lrs,
    train_batch_sizes,
    anchor_samples_list,
    clean_num_steps_list,
    eps_max_list=None,
    dot_product_loss_list=None,
    reconstruction_loss_weight_list=None,
):
    """
    Build cartesian-product hyperparameter configurations for diffusion attack training.

    Returns a list of dicts with keys:
      - lr
      - train_batch_size
      - anchor_samples
      - clean_num_steps
      - eps_max (optional; included when eps_max_list is provided)
            - dot_product_loss (optional; included when dot_product_loss_list is provided)
      - reconstruction_loss_weight (optional; included when reconstruction_loss_weight_list is provided)
    """
    eps_max_values = [None] if eps_max_list is None else list(eps_max_list)
    dot_product_loss_values = [None] if dot_product_loss_list is None else list(dot_product_loss_list)
    reconstruction_loss_weight_values = (
        [None]
        if reconstruction_loss_weight_list is None
        else list(reconstruction_loss_weight_list)
    )

    grid = []
    for (
        lr,
        train_batch_size,
        anchor_samples,
        clean_num_steps,
        eps_max_value,
        dot_product_loss_value,
        reconstruction_loss_weight_value,
    ) in product(
        lrs,
        train_batch_sizes,
        anchor_samples_list,
        clean_num_steps_list,
        eps_max_values,
        dot_product_loss_values,
        reconstruction_loss_weight_values,
    ):
        if lr is None or train_batch_size is None or anchor_samples is None or clean_num_steps is None:
            raise ValueError("lrs, train_batch_sizes, anchor_samples_list and clean_num_steps_list cannot contain None")
        config = {
            "lr": float(lr),
            "train_batch_size": int(train_batch_size),
            "anchor_samples": int(anchor_samples),
            "clean_num_steps": int(clean_num_steps),
        }
        if eps_max_value is not None:
            config["eps_max"] = float(eps_max_value)
        if dot_product_loss_value is not None:
            config["dot_product_loss"] = str(dot_product_loss_value)
        if reconstruction_loss_weight_value is not None:
            config["reconstruction_loss_weight"] = float(reconstruction_loss_weight_value)
        grid.append(config)
    return grid


def _score_history(history, tail_k=25):
    """Lower is better: mean of the last k losses (or all if shorter)."""
    if len(history) == 0:
        return float("inf")
    k = min(int(tail_k), len(history))
    return float(np.mean(history[-k:]))


def compute_mean_perturbation_effect_over_steps(gps_traj_source, gps_traj_perturbed):
    """
    Compute the perturbation-effect metric used in plotting utilities.

    This reproduces the displacement computation in `plot_gps_trajectories_on_map`:
      displacement(step, sample) = sqrt((dlat)^2 + (dlon)^2)
    and returns the mean perturbation effect over all steps.
    """
    metrics = evaluate_displacement_metrics(gps_traj_source, gps_traj_perturbed)
    return float(metrics["mean_step_displacement"])


def run_diffusion_hparam_search(
    source_image,
    pipeline,
    hparam_grid,
    n_steps=400,
    eps_max=1,
    target_pure_noise=False,
    dot_product_loss="squared",
    reconstruction_loss_weight=0,
    device="cuda",
    log_every=20,
    score_tail_k=25,
    keep_deltas=False,
    show_progress=True,
    num_restarts=1,
    eval_batch_size=256,
    eval_cfg=10.0,
    eval_num_steps=None,
    eval_seed=1234,
    eval_metric="mean_perturbation_effect_over_steps",
):
    """
    Run a hyperparameter search over `train_diffusion_perturbation`.

    Args:
        source_image: PIL image used for training.
        pipeline: PLONK pipeline.
        hparam_grid: iterable of dicts with keys
            {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
            and optionally "eps_max", "dot_product_loss" and "reconstruction_loss_weight".
        n_steps, eps_max, target_pure_noise, dot_product_loss,
        reconstruction_loss_weight, device, log_every:
            forwarded to train_diffusion_perturbation.
        score_tail_k: score = mean of the last k training losses (kept for logging).
        keep_deltas: if True, each trial stores trained perturbation tensor.
        show_progress: tqdm over trials.
        num_restarts: number of independent training runs per config.
        eval_batch_size, eval_cfg, eval_num_steps, eval_seed:
            parameters used to evaluate trajectory perturbation effect.
        eval_metric: one of
            - "mean_perturbation_effect_over_steps"
            - "mean_final_prediction_distance"

    Returns:
        dict with keys:
          - best_index
          - best_config
          - best_score
          - trials (list of per-trial dicts)
                each trial has:
                  config, score, metric_name, final_loss, min_loss, history,
                  best_restart, restart_summaries, (optional) delta
    """
    trials = []
    best_index = None
    best_score = -float("inf")

    iterator = enumerate(hparam_grid)
    if show_progress:
        iterator = tqdm.tqdm(iterator, total=len(hparam_grid), desc="Hyperparameter search")

    valid_metrics = {
        "mean_perturbation_effect_over_steps",
        "mean_final_prediction_distance",
    }
    if eval_metric not in valid_metrics:
        raise ValueError(
            f"Unknown eval_metric: {eval_metric}. Expected one of {sorted(valid_metrics)}"
        )

    for trial_index, config in iterator:
        required = {"lr", "train_batch_size", "anchor_samples", "clean_num_steps"}
        missing = required.difference(config.keys())
        if missing:
            raise ValueError(f"Missing keys in config {trial_index}: {sorted(missing)}")

        trial_eps_max = float(config["eps_max"]) if "eps_max" in config else float(eps_max)
        trial_reconstruction_loss_weight = (
            float(config["reconstruction_loss_weight"])
            if "reconstruction_loss_weight" in config
            else float(reconstruction_loss_weight)
        )
        trial_dot_product_loss = str(config.get("dot_product_loss", dot_product_loss))

        restart_summaries = []
        best_restart_effect = -float("inf")
        best_restart_idx = None
        best_restart_payload = None

        for restart_idx in range(int(num_restarts)):
            delta, history, _ = train_diffusion_perturbation(
                source_image=source_image,
                pipeline=pipeline,
                n_steps=n_steps,
                train_batch_size=int(config["train_batch_size"]),
                lr=float(config["lr"]),
                eps_max=trial_eps_max,
                anchor_samples=int(config["anchor_samples"]),
                clean_num_steps=int(config["clean_num_steps"]),
                log_every=log_every,
                target_pure_noise=target_pure_noise,
                dot_product_loss=trial_dot_product_loss,
                reconstruction_loss_weight=trial_reconstruction_loss_weight,
                device=device,
            )

            # Build a perturbed image and evaluate perturbation effect on trajectories.
            perturbed_image = add_perturbation_to_image(source_image, delta, pipeline)
            eval_result = run_paired_pipeline_with_shared_noise(
                pipeline=pipeline,
                source_image=source_image,
                perturbed_image=perturbed_image,
                batch_size=eval_batch_size,
                cfg=eval_cfg,
                num_steps=eval_num_steps,
                seed=int(eval_seed) + trial_index * 1000 + restart_idx,
                device=device,
            )
            gps_source_eval = eval_result["gps_source"]
            gps_perturbed_eval = eval_result["gps_perturbed"]
            traj_source = eval_result["traj_source"]
            traj_perturbed = eval_result["traj_perturbed"]

            if eval_metric == "mean_perturbation_effect_over_steps":
                effect_score = compute_mean_perturbation_effect_over_steps(traj_source, traj_perturbed)
            else:
                effect_score = mean_final_prediction_distance(gps_source_eval, gps_perturbed_eval)
            history_score = _score_history(history, tail_k=score_tail_k)
            restart_summary = {
                "restart": restart_idx,
                "effect_score": effect_score,
                "history_score": history_score,
                "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
                "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
            }
            restart_summaries.append(restart_summary)

            if effect_score > best_restart_effect:
                best_restart_effect = effect_score
                best_restart_idx = restart_idx
                best_restart_payload = {
                    "delta": delta,
                    "history": history,
                    "history_score": history_score,
                }

        if best_restart_payload is None:
            raise RuntimeError("No restart completed successfully for this hyperparameter config")
        if best_restart_idx is None:
            raise RuntimeError("Unable to select best restart for this hyperparameter config")

        score = float(best_restart_effect)
        history = best_restart_payload["history"]
        trial = {
            "config": {
                "lr": float(config["lr"]),
                "train_batch_size": int(config["train_batch_size"]),
                "anchor_samples": int(config["anchor_samples"]),
                "clean_num_steps": int(config["clean_num_steps"]),
            },
            "score": score,
            "metric_name": eval_metric,
            "best_restart": best_restart_idx,
            "restart_summaries": restart_summaries,
            "history_score": float(best_restart_payload["history_score"]),
            "final_loss": float(history[-1]) if len(history) > 0 else float("inf"),
            "min_loss": float(np.min(history)) if len(history) > 0 else float("inf"),
            "history": history,
        }
        if "eps_max" in config:
            trial["config"]["eps_max"] = trial_eps_max
        if "dot_product_loss" in config:
            trial["config"]["dot_product_loss"] = trial_dot_product_loss
        if "reconstruction_loss_weight" in config:
            trial["config"]["reconstruction_loss_weight"] = trial_reconstruction_loss_weight
        if keep_deltas:
            trial["delta"] = best_restart_payload["delta"]

        trials.append(trial)

        if score > best_score:
            best_score = score
            best_index = trial_index

    best_config = trials[best_index]["config"] if best_index is not None else None
    return {
        "best_index": best_index,
        "best_config": best_config,
        "best_score": best_score,
        "trials": trials,
    }