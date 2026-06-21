"""Sampling attack (Salman et al.): backpropagates through the full DDIM trajectory."""

from typing import Any, Dict, Optional

from PIL import Image
import torch

from plonk.metrics.utils import haversine

from attacks.attacks_core import AttackBase
from utils.adversarial_utils import compute_embedding


class SamplingAttack(AttackBase):
    """Directly optimizes haversine distance by backpropagating through the full DDIM sampling trajectory."""

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
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
        device: str = "cuda",
        x0_bank: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            pipeline=pipeline,
            source_image=source_image,
            num_restarts=num_restarts,
            selection_metric=restart_selection_metric,
            device=device,
        )

        pipeline.network.eval().requires_grad_(False)
        pipeline.cond_preprocessing.emb_model.eval().requires_grad_(False)

        # Attack hyperparameters
        self.n_steps = int(n_steps)
        self.train_batch_size = int(train_batch_size)
        self.lr = float(lr)
        self.eps_max = float(eps_max)
        self.sampling_steps_salman = int(sampling_steps_salman)
        self.clean_num_steps = int(clean_num_steps)
        self.target_pure_noise = bool(target_pure_noise)
        self.dot_product_loss = dot_product_loss
        # CFG is intentionally 0 during training: using CFG would double the batch inside the
        # DDIM loop and make backprop through the full trajectory prohibitively expensive.
        self.sampling_cfg = 0.0
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)
        self.delta_init = float(delta_init)

        self._clean_embedding = compute_embedding(
            self.source_tensor,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=False,
        )

    def _sample_cartesian(self, embedding: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Run the DDIM sampler without postprocessing so gradients reach the image embedding."""
        model_batch = {
            "y": x_t,
            "emb": embedding,
        }
        return self.pipeline.sampler(
            self.pipeline.model,
            model_batch,
            conditioning_keys="emb",
            scheduler=self.pipeline.scheduler,
            num_steps=self.sampling_steps_salman,
            cfg_rate=self.sampling_cfg,
            generator=None,
            return_trajectories=False,
        )

    def initialize_delta(self, restart_idx: int) -> torch.Tensor:
        """Initialize delta with tiny random noise to avoid zero-gradient dead start."""
        if self.delta_init > 0:
            delta = torch.empty_like(self.source_tensor).uniform_(-self.delta_init, self.delta_init)
        else:
            delta = torch.zeros_like(self.source_tensor)
        delta.requires_grad_(True)
        return delta

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute one sign-SGD step."""
        optimizer.zero_grad(set_to_none=True)

        x_t = torch.randn(self.train_batch_size, 3, device=self.device)

        perturbed_source = self.source_tensor + delta
        emb_perturbed = compute_embedding(
            perturbed_source,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=True,
        )

        pred_perturbed = self._sample_cartesian(emb_perturbed, x_t)
        with torch.no_grad():
            pred_clean = self._sample_cartesian(self._clean_embedding, x_t)

        pred_perturbed_gps = self.pipeline.postprocessing(pred_perturbed)
        with torch.no_grad():
            pred_clean_gps = self.pipeline.postprocessing(pred_clean)

        loss = -haversine(pred_perturbed_gps, pred_clean_gps).mean()

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
            "sampling_steps_salman": self.sampling_steps_salman,
            "clean_num_steps": self.clean_num_steps,
            "target_pure_noise": self.target_pure_noise,
            "dot_product_loss": self.dot_product_loss,
            "sampling_cfg": self.sampling_cfg,
            "reconstruction_loss_weight": self.reconstruction_loss_weight,
            "delta_init": self.delta_init,
            "num_restarts": self.restart_manager.num_restarts,
        }