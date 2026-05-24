"""Diffusion trajectory deviation attacks."""

from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from attacks_core import AttackBase
from adversarial_utils import (
    conditional_preprocessing,
    compute_embedding,
)




############################################################################################
# Diffusion attack helpers
############################################################################################


def _compute_dot_alignment_loss(eps_reference, eps_prediction, dot_product_loss="squared"):
    """Compute alignment loss from dot products between reference and predicted directions."""
    metric_aliases = {
        "square": "squared",
        "squared": "squared",
        "squared_dot": "squared",
        "abs": "absolute",
        "absolute": "absolute",
        "absolute_dot": "absolute",
        "cosine_similarity": "cosine_similarity",
    }
    normalized_metric = metric_aliases.get(str(dot_product_loss).lower())
    if normalized_metric is None:
        raise ValueError(
            f"Unknown dot_product_loss: {dot_product_loss}. Expected one of ['squared', 'absolute']"
        )

    dot = torch.sum(eps_reference * eps_prediction, dim=-1)
    if normalized_metric == "squared":
        return (dot ** 2).mean()
    if normalized_metric == "cosine_similarity":
        eps_reference_norm = torch.norm(eps_reference, dim=-1)
        eps_prediction_norm = torch.norm(eps_prediction, dim=-1)
        cosine_sim = dot / (eps_reference_norm * eps_prediction_norm + 1e-8)
        return cosine_sim.mean()
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


############################################################################################
# Diffusion attack using AttackBase
############################################################################################


class DiffusionAttack(AttackBase):
    """Diffusion-space universal perturbation attack."""

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
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
        device: str = "cuda",
        x0_bank: Optional[torch.Tensor] = None,  # Shared x0_bank across restarts
    ):
        super().__init__(
            pipeline=pipeline,
            source_image=source_image,
            num_restarts=num_restarts,
            selection_metric=restart_selection_metric,
            device=device,
        )
        
        # Freeze models
        pipeline.network.eval().requires_grad_(False)
        pipeline.cond_preprocessing.emb_model.eval().requires_grad_(False)
        
        # Build x0 bank for stochastic training (or reuse provided one)
        if x0_bank is None:
            self.x0_bank = build_x0_bank_from_clean_model(
                pipeline,
                source_image,
                n_samples=anchor_samples,
                num_steps=clean_num_steps,
                cfg=0.0,
                device=device,
            )
        else:
            self.x0_bank = x0_bank  # Reuse shared x0_bank from caller
        
        # Attack hyperparameters
        self.n_steps = int(n_steps)
        self.train_batch_size = int(train_batch_size)
        self.lr = float(lr)
        self.eps_max = float(eps_max)
        self.target_pure_noise = target_pure_noise
        self.dot_product_loss = dot_product_loss
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)

    def initialize_delta(self, restart_idx: int) -> torch.Tensor:
        """Initialize delta as zeros (universal perturbation)."""
        delta = torch.zeros_like(self.source_tensor, requires_grad=True)
        return delta

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute one sign-SGD step."""
        optimizer.zero_grad(set_to_none=True)
        
        # Sample from x0 bank and diffusion process
        idx = torch.randint(0, self.x0_bank.shape[0], (self.train_batch_size,), device=self.device)
        x0 = self.x0_bank[idx]
        eps = torch.randn_like(x0)
        
        t = torch.rand(self.train_batch_size, device=self.device)
        gamma = self.pipeline.scheduler(t)
        
        x_t = (
            torch.sqrt(gamma).unsqueeze(-1) * x0
            + torch.sqrt(1.0 - gamma).unsqueeze(-1) * eps
        )
        
        # Compute embeddings
        perturbed_source = self.source_tensor + delta
        emb_perturbed = compute_embedding(
            perturbed_source,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=True,
        )
        
        # Denoiser prediction on perturbed
        model_batch_perturbed = {
            "y": x_t,
            "emb": emb_perturbed,
            "gamma": gamma,
        }
        eps_pred_perturbed = self.pipeline.model(model_batch_perturbed)
        
        # Reference noise prediction
        if not self.target_pure_noise:
            emb_source = compute_embedding(
                self.source_tensor,
                self.train_batch_size,
                self.pipeline,
                device=self.device,
                track_grad=False,
            )
            model_batch = {
                "y": x_t,
                "emb": emb_source,
                "gamma": gamma,
            }
            eps_pred = self.pipeline.model(model_batch)
        else:
            eps_pred = eps
        
        # Alignment loss
        loss = _compute_dot_alignment_loss(eps_pred, eps_pred_perturbed, dot_product_loss=self.dot_product_loss)
        
        # Reconstruction loss
        if self.reconstruction_loss_weight > 0:
            loss_x = torch.nn.functional.l1_loss(perturbed_source, self.source_tensor)
            loss = loss + self.reconstruction_loss_weight * loss_x
        
        loss.backward()
        
        # Sign-SGD update in l_inf ball
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad)
            optimizer.step()
            delta.data = torch.clamp(delta.data, -self.eps_max, self.eps_max)
            delta.grad.zero_()
        
        return float(loss.item())

    def get_config(self) -> Dict[str, Any]:
        """Return attack configuration."""
        return {
            "n_steps": self.n_steps,
            "train_batch_size": self.train_batch_size,
            "lr": self.lr,
            "eps_max": self.eps_max,
            "target_pure_noise": self.target_pure_noise,
            "dot_product_loss": self.dot_product_loss,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "num_restarts": self.restart_manager.num_restarts,
        }
