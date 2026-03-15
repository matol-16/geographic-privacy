from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import tqdm as tqdm
from PIL import Image
from adversarial_metrics import (
    evaluate_displacement_metrics,
    select_displacement_score,
)

from adversarial_utils import (conditional_preprocessing, model_dependent_embedding)

def _project_linf_(delta: torch.Tensor, eps_max: float) -> None:
    """In-place projection on the l_inf ball."""
    delta.clamp_(-float(eps_max), float(eps_max))


def train_encoder_attack(
    source_image: Image.Image,
    pipeline,
    z_target: Optional[torch.Tensor] = None,
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
    log_every: int = 20,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """
    Train an encoder-space perturbation using projected gradient descent.

    If both z_target and target_image are None, the attack is untargeted and
    maximizes encoder-space L2 distance to the source image embedding.

    Supported losses:
      - "MSE"
      - "MSE+Reconstruction"
    """
    valid_criteria = {"MSE", "MSE+Reconstruction"}
    if criterion_name not in valid_criteria:
        raise ValueError(f"Unknown criterion_name={criterion_name}. Expected one of {sorted(valid_criteria)}")
    if int(num_restarts) < 1:
        raise ValueError("num_restarts must be >= 1")

    pipeline.network.eval().requires_grad_(False)
    pipeline.cond_preprocessing.emb_model.train().requires_grad_(False)

    # source_tensor = _prepare_source_tensor(source_image, pipeline, device)
    source_tensor = conditional_preprocessing(source_image, pipeline, device)

    z_source= model_dependent_embedding(source_tensor, pipeline, track_grad=False).detach()
    
    target_tensor = conditional_preprocessing(target_image, pipeline, device) if target_image is not None else None
    z_target = model_dependent_embedding(target_tensor, pipeline, track_grad=False).detach() if target_tensor is not None else None
   
    attack_mode = "untargeted" if z_target is None else "targeted"

    best_delta = None
    best_history: List[float] = []
    best_restart = None
    best_score = -float("inf")
    restart_summaries = []
    best_metrics=None

    from adversarial_utils import add_perturbation_to_image

    for restart_idx in range(int(num_restarts)):
        delta = torch.empty_like(source_tensor).uniform_(-float(eps_max), float(eps_max))
        delta.requires_grad_(True)
        optimizer = torch.optim.Adam([delta], lr=float(lr))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=int(n_steps))

        history: List[float] = []
        step_iter = range(int(n_steps))
        pbar = None
        if show_progress:
            pbar = tqdm.trange(
                int(n_steps),
                desc=f"Encoder attack (restart {restart_idx + 1}/{int(num_restarts)})",
            )
            step_iter = pbar

        for step in step_iter:
            optimizer.zero_grad(set_to_none=True)

            perturbed_tensor = source_tensor + delta
            z_perturbed = model_dependent_embedding(perturbed_tensor, pipeline, track_grad=True)

            if attack_mode == "targeted":
                loss_embed = torch.nn.functional.mse_loss(z_perturbed, z_target)
                signed_embed_term = float(l_z) * loss_embed
            else:
                embed_l2 = torch.norm(z_perturbed - z_source, p=2, dim=-1).mean()
                loss_embed = embed_l2
                signed_embed_term = -float(l_z) * loss_embed

            if criterion_name == "MSE+Reconstruction":
                loss_recon = torch.nn.functional.l1_loss(perturbed_tensor, source_tensor)
                loss = signed_embed_term + float(l_x) * loss_recon
            else:
                loss_recon = torch.zeros((), device=loss_embed.device)
                loss = signed_embed_term

            loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                _project_linf_(delta, eps_max)

            history.append(float(loss.item()))

            if pbar is not None and (step + 1) % int(log_every) == 0:
                pbar.set_postfix(loss=f"{loss.item():.6f}")

        perturbed_image = add_perturbation_to_image(source_image, delta.detach(), pipeline)
        generator = torch.Generator(device=device)
        generator.manual_seed(int(restart_eval_seed) + restart_idx)
        x_n = torch.randn(int(restart_eval_batch_size), 3, device=device, generator=generator)

        eval_kwargs = {
            "batch_size": int(restart_eval_batch_size),
            "cfg": float(restart_eval_cfg),
            "x_N": x_n,
            "return_trajectories": True,
        }
        if restart_eval_num_steps is not None:
            eval_kwargs["num_steps"] = int(restart_eval_num_steps)

        _, traj_source = pipeline(source_image, **eval_kwargs)
        _, traj_perturbed = pipeline(perturbed_image, **eval_kwargs)
        metrics = evaluate_displacement_metrics(traj_source, traj_perturbed)
        mean_step_displacement = metrics["mean_step_displacement"]
        final_step_displacement = metrics["final_step_displacement"]
        score = select_displacement_score(
            mean_step_disp=mean_step_displacement,
            final_step_disp=final_step_displacement,
            metric_name=restart_selection_metric,
        )

        final_loss = float(history[-1]) if history else float("inf")
        min_loss = float(min(history)) if history else float("inf")
        summary = {
            "restart": restart_idx,
            "mean_step_displacement": float(mean_step_displacement),
            "final_step_displacement": float(final_step_displacement),
            "score": float(score),
            "final_loss": final_loss,
            "min_loss": min_loss,
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
            best_history = history
            best_restart = restart_idx
            best_metrics = metrics

    if best_delta is None or best_restart is None:
        raise RuntimeError("No restart produced a valid perturbation")

    if print_restart_results and int(num_restarts) > 1:
        print(
            f"Selected restart {best_restart + 1}/{int(num_restarts)} using "
            f"{restart_selection_metric} (score={best_score:.6f})"
        )

    return {
        "attack_type": "encoder",
        "attack_mode": attack_mode,
        "delta": best_delta,
        "history": best_history,
        "source_tensor": source_tensor,
        "z_target": z_target,
        "z_source": z_source,
        "best_restart": int(best_restart),
        "restart_summaries": restart_summaries,
        "best_metrics": best_metrics,
        "config": {
            "num_steps": int(n_steps),
            "lr": float(lr),
            "attack_size": float(eps_max),
            "criterion_name": criterion_name,
            "l_z": float(l_z),
            "l_x": float(l_x),
            "num_restarts": int(num_restarts),
            "restart_selection_metric": restart_selection_metric,
            "restart_eval_batch_size": int(restart_eval_batch_size),
            "restart_eval_cfg": float(restart_eval_cfg),
        },
    }
