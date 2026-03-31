

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
