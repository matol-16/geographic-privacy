"""Encoder-space adversarial attacks using projected gradient descent."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from PIL import Image

from attacks_core import AttackBase
from adversarial_utils import (
    conditional_preprocessing,
    model_dependent_embedding,
)


def _project_linf_(delta: torch.Tensor, eps_max: float) -> None:
    """In-place projection on the l_inf ball."""
    delta.clamp_(-float(eps_max), float(eps_max))


class EncoderAttack(AttackBase):
    """Encoder-space attack using projected gradient descent."""

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
        target_image: Optional[Image.Image] = None,
        n_steps: int = 100,
        lr: float = 0.1,
        eps_max: float = 0.1,
        criterion_name: str = "MSE",
        l_z: float = 1.0,
        l_x: float = 1.0,
        num_restarts: int = 1,
        restart_selection_metric: str = "mean_step_displacement",
        device: str = "cuda",
    ):
        super().__init__(
            pipeline=pipeline,
            source_image=source_image,
            num_restarts=num_restarts,
            selection_metric=restart_selection_metric,
            device=device,
        )
        
        # Validation
        valid_criteria = {"MSE", "MSE+Reconstruction"}
        if criterion_name not in valid_criteria:
            raise ValueError(f"Unknown criterion_name={criterion_name}. Expected one of {sorted(valid_criteria)}")
        
        # Setup models
        pipeline.network.eval().requires_grad_(False)
        pipeline.cond_preprocessing.emb_model.train().requires_grad_(False)
        
        # Prepare embeddings
        self.z_source = model_dependent_embedding(self.source_tensor, pipeline, track_grad=False).detach()
        
        target_tensor = conditional_preprocessing(target_image, pipeline, device=device) if target_image is not None else None
        self.z_target = model_dependent_embedding(target_tensor, pipeline, track_grad=False).detach() if target_tensor is not None else None
        
        self.attack_mode = "untargeted" if self.z_target is None else "targeted"
        
        # Attack hyperparameters
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.eps_max = float(eps_max)
        self.criterion_name = criterion_name
        self.l_z = float(l_z)
        self.l_x = float(l_x)

    def initialize_delta(self, restart_idx: int) -> torch.Tensor:
        """Initialize delta uniformly in [-eps_max, eps_max]."""
        delta = torch.empty_like(self.source_tensor).uniform_(-self.eps_max, self.eps_max)
        delta.requires_grad_(True)
        return delta

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute one PGD step."""
        optimizer.zero_grad(set_to_none=True)
        
        perturbed_tensor = self.source_tensor + delta
        z_perturbed = model_dependent_embedding(perturbed_tensor, self.pipeline, track_grad=True)
        
        # Embedding loss
        if self.attack_mode == "targeted":
            loss_embed = torch.nn.functional.mse_loss(z_perturbed, self.z_target)
            signed_embed_term = self.l_z * loss_embed
        else:
            embed_l2 = torch.norm(z_perturbed - self.z_source, p=2, dim=-1).mean()
            signed_embed_term = -self.l_z * embed_l2
        
        # Optional reconstruction loss
        if self.criterion_name == "MSE+Reconstruction":
            loss_recon = torch.nn.functional.l1_loss(perturbed_tensor, self.source_tensor)
            loss = signed_embed_term + self.l_x * loss_recon
        else:
            loss = signed_embed_term
        
        loss.backward()
        optimizer.step()
        
        # Project to l_inf ball
        with torch.no_grad():
            _project_linf_(delta, self.eps_max)
        
        return float(loss.item())

    def get_config(self) -> Dict[str, Any]:
        """Return attack configuration."""
        return {
            "n_steps": self.n_steps,
            "lr": self.lr,
            "eps_max": self.eps_max,
            "criterion_name": self.criterion_name,
            "l_z": self.l_z,
            "l_x": self.l_x,
            "num_restarts": self.restart_manager.num_restarts,
        }
