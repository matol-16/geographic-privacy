"""Diffusion trajectory deviation attacks."""

from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from attacks.attacks_core import AttackBase
from utils.adversarial_utils import (
    conditional_preprocessing,
    compute_embedding,
)




############################################################################################
# Diffusion attack helpers
############################################################################################


def _compute_dot_alignment_loss(eps_reference, eps_prediction, dot_product_loss="squared"):
    """Compute alignment loss from dot products between reference and predicted directions."""
    metric_aliases = {
        "l2": "l2",
        "l2_target": "l2_target",
        "square": "squared",
        "squared": "squared",
        "squared_dot": "squared",
        "abs": "absolute",
        "absolute": "absolute",
        "absolute_dot": "absolute",
        "cosine_similarity": "cosine_similarity",
        "cosine_similarity_negative": "cosine_similarity_negative",
        "cosine_similarity_target": "cosine_similarity_target",
    }
    normalized_metric = metric_aliases.get(str(dot_product_loss).lower())
    if normalized_metric is None:
        raise ValueError(
            f"Unknown dot_product_loss: {dot_product_loss}. Expected one of ['squared', 'absolute']"
        )
    if normalized_metric == "l2":
        return -1*torch.nn.functional.mse_loss(eps_prediction, eps_reference)
    if normalized_metric == "l2_target":
        # Loss to be *minimized*: pulls the prediction towards the (target) reference.
        return torch.nn.functional.mse_loss(eps_prediction, eps_reference)
    dot = torch.sum(eps_reference * eps_prediction, dim=-1)
    if normalized_metric == "squared":
        return (dot ** 2).mean()
    if normalized_metric == "absolute":
        abs_dot= torch.abs(dot)
        return abs_dot.mean()
    if normalized_metric == "cosine_similarity":
        abs_dot=torch.abs(dot)
        eps_reference_norm = torch.norm(eps_reference, dim=-1)
        eps_prediction_norm = torch.norm(eps_prediction, dim=-1)
        cosine_sim = abs_dot / (eps_reference_norm * eps_prediction_norm + 1e-8)
        return cosine_sim.mean()
    if normalized_metric == "cosine_similarity_negative":
        eps_reference_norm = torch.norm(eps_reference, dim=-1)
        eps_prediction_norm = torch.norm(eps_prediction, dim=-1)
        cosine_sim = dot / (eps_reference_norm * eps_prediction_norm + 1e-8)
        return cosine_sim.mean()
    if normalized_metric == "cosine_similarity_target":
        # Loss to be *minimized*: minimizing -cosine_sim maximizes the (signed)
        # cosine similarity between the prediction and the target reference.
        eps_reference_norm = torch.norm(eps_reference, dim=-1)
        eps_prediction_norm = torch.norm(eps_prediction, dim=-1)
        cosine_sim = dot / (eps_reference_norm * eps_prediction_norm + 1e-8)
        return -cosine_sim.mean()


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
        delta_init: float = 1e-4,
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
        self.delta_init = float(delta_init)

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
            "delta_init": self.delta_init,
            "num_restarts": self.restart_manager.num_restarts,
        }


############################################################################################
# ACE: targeted diffusion trajectory deviation + encoder alignment
############################################################################################


class ACE(DiffusionAttack):
    """Targeted diffusion-space perturbation attack with encoder alignment (ACE).

    Instead of pushing the perturbed score *away* from the clean source score
    (as in :class:`DiffusionAttack`), this attack pulls the perturbed prediction
    *towards* a chosen ``target_image``. The objective combines two terms:

    1. A score-alignment term that pulls the perturbed score prediction towards
       the score the model predicts for the target image (steering the
       conditioned diffusion model towards the target's geographic
       distribution). Defaults to an l2 (MSE) loss between the two predictions;
       other metrics such as ``cosine_similarity_target`` are also supported.
    2. An encoder term (the encoder-attack objective): the l2 distance between
       the encoded target image and the encoded perturbed image, which directly
       drives the perturbed embedding towards the target embedding.

    The two terms are combined as ``score_loss + alpha * encoder_l2``, where
    ``alpha`` weighs the encoder term. Both terms are minimized.
    """

    # Class-level flag so the gradient-balance diagnostic prints only once per
    # process (on the first step of the first instance), not once per image.
    _grad_diag_done: bool = False

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
        target_image: Image.Image,
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
        device: str = "cuda",
        x0_bank: Optional[torch.Tensor] = None,  # Shared x0_bank across restarts
        diagnose_grad_balance: bool = True,  # Print per-term grad norms once to check alpha
    ):
        if target_image is None:
            raise ValueError("ACE requires a target_image.")

        super().__init__(
            pipeline=pipeline,
            source_image=source_image,
            n_steps=n_steps,
            train_batch_size=train_batch_size,
            lr=lr,
            eps_max=eps_max,
            anchor_samples=anchor_samples,
            clean_num_steps=clean_num_steps,
            target_pure_noise=False,
            dot_product_loss=dot_product_loss,
            reconstruction_loss_weight=reconstruction_loss_weight,
            delta_init=delta_init,
            num_restarts=num_restarts,
            restart_selection_metric=restart_selection_metric,
            device=device,
            x0_bank=x0_bank,
        )

        # Weight of the encoder l2 alignment term.
        self.alpha = float(alpha)

        # One-time diagnostic of the score vs (alpha * encoder) gradient balance.
        self.diagnose_grad_balance = bool(diagnose_grad_balance)

        # Preprocess the target image once; its embedding is used as the
        # (frozen) reference the perturbed prediction is pulled towards.
        self.target_image = target_image
        self.target_tensor = conditional_preprocessing(target_image, pipeline, device=device)

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Execute one sign-SGD step combining score alignment and encoder alignment."""
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

        # Denoiser prediction on the perturbed source (tracks gradient)
        perturbed_source = self.source_tensor + delta
        emb_perturbed = compute_embedding(
            perturbed_source,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=True,
        )
        model_batch_perturbed = {
            "y": x_t,
            "emb": emb_perturbed,
            "gamma": gamma,
        }
        eps_pred_perturbed = self.pipeline.model(model_batch_perturbed)

        # Reference noise prediction conditioned on the target image (frozen)
        emb_target = compute_embedding(
            self.target_tensor,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=False,
        )
        model_batch_target = {
            "y": x_t,
            "emb": emb_target,
            "gamma": gamma,
        }
        eps_pred_target = self.pipeline.model(model_batch_target)

        # Term 1: score-alignment loss pulling the perturbed score towards the target
        # prediction (default l2_target = MSE; configurable via dot_product_loss).
        score_loss = _compute_dot_alignment_loss(
            eps_pred_target, eps_pred_perturbed, dot_product_loss=self.dot_product_loss
        )
        loss = score_loss

        # Term 2: encoder l2 distance between encoded perturbed and encoded target image.
        # Reuse the embeddings already computed above (rows are identical copies, so a
        # single row carries the per-image embedding); target side is detached.
        encoder_l2 = None
        if self.alpha != 0:
            z_perturbed = emb_perturbed[:1]
            z_target = emb_target[:1].detach()
            encoder_l2 = torch.norm(z_perturbed - z_target, p=2, dim=-1).mean()
            loss = loss + self.alpha * encoder_l2

        # One-time diagnostic: compare the magnitude of each term's gradient w.r.t. the
        # perturbation, *before* the sign is taken. Since the update is sign-SGD, alpha is
        # well chosen when the score and (alpha * encoder) gradient norms are comparable.
        if self.diagnose_grad_balance and not ACE._grad_diag_done and encoder_l2 is not None:
            ACE._grad_diag_done = True
            g_score = torch.autograd.grad(score_loss, delta, retain_graph=True)[0]
            g_encoder = torch.autograd.grad(encoder_l2, delta, retain_graph=True)[0]
            score_norm = float(g_score.norm())
            encoder_norm = float(g_encoder.norm())
            weighted_encoder_norm = self.alpha * encoder_norm
            ratio = weighted_encoder_norm / (score_norm + 1e-12)
            print(
                "[ACE grad balance] "
                f"||g_score||={score_norm:.4e}, "
                f"||g_encoder||={encoder_norm:.4e}, "
                f"alpha={self.alpha:g}, "
                f"||alpha*g_encoder||={weighted_encoder_norm:.4e}, "
                f"ratio(alpha*encoder/score)={ratio:.3f} "
                "(>>1 encoder-dominated, <<1 score-dominated, ~1 balanced)"
            )

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
        config = super().get_config()
        config["alpha"] = self.alpha
        config["attack_mode"] = "targeted"
        return config
