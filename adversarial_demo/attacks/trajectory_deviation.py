"""Diffusion trajectory deviation attacks."""

from plonk.pipe import _gps_degrees_to_cartesian
from PIL import Image
import torch
import numpy as np
from typing import Any, Dict, List, Optional

from torch.func import jvp as _func_jvp, vmap as _func_vmap
from plonk.utils.manifolds import geodesic as _sphere_geodesic

from attacks.attacks_core import AttackBase
from utils.adversarial_utils import (
    conditional_preprocessing,
    compute_embedding,
    model_dependent_embedding,
)




############################################################################################
# Diffusion attack helpers
############################################################################################


def _compute_alignment_loss(eps_reference, eps_prediction, dot_product_loss="squared"):
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


def detect_model_kind(pipeline) -> str:
    """Identify the generative parameterization of a PLONK pipeline.

    Returns one of:
      - ``"diffusion"``: DDIM sampler; the network predicts the noise ``eps``.
      - ``"rfm"``: Riemannian flow matching on the sphere; predicts a *velocity*
        and samples by projecting onto the manifold each step.
      - ``"flow"``: Euclidean flow matching; predicts a *velocity*.

    Detection is based on the sampler bound to the pipeline (the source of truth
    in ``plonk.pipe.MODELS``), with the model path as a fallback. Note the order
    of the checks: ``"riemannian_flow_sampler"`` also contains ``"flow"``.
    """
    sampler_name = getattr(getattr(pipeline, "sampler", None), "__name__", "") or ""
    model_path = str(getattr(pipeline, "model_path", "")).lower()
    if "ddim" in sampler_name or "diffusion" in model_path:
        return "diffusion"
    if "riemannian" in sampler_name:
        return "rfm"
    if "flow" in sampler_name or "flow" in model_path:
        return "flow"
    # Default: the base PLONK models (no suffix) are Riemannian flow matching.
    return "rfm"


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
        dot_product_loss: str = "l2",
        reconstruction_loss_weight: float = 0.0,
        delta_init: float = 1e-4,
        num_restarts: int = 1,
        restart_selection_metric: str = "final_step_displacement",
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
        loss = _compute_alignment_loss(eps_pred, eps_pred_perturbed, dot_product_loss=self.dot_product_loss)
        
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
        restart_selection_metric: str = "final_step_displacement",
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
        score_loss = _compute_alignment_loss(
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


############################################################################################
# UniDef: Consistent Distribution Deviation + Finite-Difference Jacobian Estimation
############################################################################################


class UniDef(DiffusionAttack):
    """UniDef universal-defense attack adapted to the image-conditioned PLONK model.

    Port of *UniDef: Universal Defense Against Unauthorized Image Manipulation*
    (Shao et al., CVPR) to this geographic diffusion model, whose denoiser
    ``eps_theta(y_t, emb, gamma)`` diffuses GPS coordinates ``y_t`` conditioned on
    an image embedding ``emb``. The protected variable is the *conditioning
    image*, so the perturbation enters only through ``emb``.

    Two components are implemented:

    1. **Consistent Distribution Deviation (CDD).** Rather than perturbing a
       single/local denoising step, the objective maximizes the trajectory bias
       ``||v_theta(y_t, emb', gamma) - v_theta(y_t, emb_clean, gamma)||^2``
       integrated over the whole sampling trajectory. The trajectory integral is
       realised exactly as in :class:`DiffusionAttack`: ``y_t`` is built from a
       bank of plausible clean states (sampled from the clean model) with
       timesteps ``t`` drawn uniformly, so every noise level contributes.
       Maximizing this deviation over all steps steers the sampling ODE away from
       the clean geographic distribution (paper Eq. 5/9).

       The reference subtracted from the perturbed prediction adapts to the
       model parameterization (auto-detected by :func:`detect_model_kind`):

       - **diffusion (DDIM):** the network regresses the noise ``eps``, so the
         reference is the *true sampled noise* and the objective is the faithful
         UniDef ``max ||eps_theta(x_t) - eps||^2``.
       - **flow / RFM:** the network regresses a *velocity field*, for which
         ``eps`` is not a valid target, so the reference is the *clean conditional
         velocity field* ``v_theta(x_t, emb_clean, gamma)`` (the best available
         proxy for the clean data distribution).

       For the manifold-native RFM model, ``x_t`` is additionally projected onto
       the sphere so the network is queried on-manifold, as it is during sampling.

    2. **Finite-Difference Jacobian Estimation (FDJE).** To avoid overfitting to
       one denoiser's gradient (improving transfer across the RFM / diffusion /
       flow backbones), the denoiser Jacobian is estimated by symmetric finite
       differences instead of backpropagation. In this conditional model the
       relevant Jacobian is ``J_e = d eps_theta / d emb``; using a direction
       ``z`` (the clean image's embedding, UniDef's "latent z"),

           J_e z ~= (eps_theta(y_t, emb' + fd*z) - eps_theta(y_t, emb' - fd*z)) / (2 fd),

       and the Hutchinson identity gives ``grad_emb ||residual||^2 ~= 2 <J_e z,
       residual> z`` (paper Eqs. 13-16). The denoiser is therefore only ever
       evaluated forward (no autograd through it); the gradient is propagated to
       the perturbation through the *encoder only* via a vector-Jacobian product.
       Setting ``use_fdje=False`` recovers exact backprop (the paper's "w/o FDJE"
       ablation).

    Optimization is sign-SGD in an l_inf ball (projected gradient ascent with the
    sign of the gradient), matching paper Eq. 17 and the rest of this module.
    """

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
        reconstruction_loss_weight: float = 0.0,
        use_fdje: bool = True,
        fd: float = 0.01,
        fdje_direction: str = "embedding",  # "embedding" (UniDef latent z) or "gaussian"
        fdje_num_samples: int = 1,
        cdd_reference: str = "clean_velocity",  # "auto" | "noise" | "clean_velocity"
        project_to_manifold: Optional[bool] = None,  # None => auto from model kind
        delta_init: float = 1e-4,
        num_restarts: int = 1,
        restart_selection_metric: str = "final_step_displacement",
        device: str = "cuda",
        x0_bank: Optional[torch.Tensor] = None,  # Shared x0_bank across restarts
    ):
        super().__init__(
            pipeline=pipeline,
            source_image=source_image,
            n_steps=n_steps,
            train_batch_size=train_batch_size,
            lr=lr,
            eps_max=eps_max,
            anchor_samples=anchor_samples,
            clean_num_steps=clean_num_steps,
            # CDD maximizes the l2 trajectory bias (-MSE, minimized == bias
            # maximized). The reference subtracted from the perturbed prediction is
            # set per model kind below (see self.cdd_reference).
            target_pure_noise=False,
            dot_product_loss="l2",
            reconstruction_loss_weight=reconstruction_loss_weight,
            delta_init=delta_init,
            num_restarts=num_restarts,
            restart_selection_metric=restart_selection_metric,
            device=device,
            x0_bank=x0_bank,
        )

        self.use_fdje = bool(use_fdje)
        self.fd = float(fd)
        self.fdje_direction = str(fdje_direction).lower()
        self.fdje_num_samples = int(fdje_num_samples)
        if self.fdje_direction not in ("embedding", "gaussian"):
            raise ValueError(
                f"Unknown fdje_direction={fdje_direction}. Expected 'embedding' or 'gaussian'."
            )

        # Adapt the objective to the model's parameterization (RFM / flow / DDIM).
        self.model_kind = detect_model_kind(pipeline)

        # CDD reference: what the perturbed prediction is pushed away from.
        #   - "noise": the true sampled epsilon (faithful UniDef; valid for DDIM,
        #     whose network regresses noise).
        #   - "clean_velocity": the clean conditional velocity field (for flow /
        #     RFM, whose network regresses a velocity, so epsilon is not a target).
        reference = str(cdd_reference).lower()
        if reference == "auto":
            reference = "noise" if self.model_kind == "diffusion" else "clean_velocity"
        if reference not in ("noise", "clean_velocity"):
            raise ValueError(
                f"Unknown cdd_reference={cdd_reference}. "
                "Expected 'auto', 'noise', or 'clean_velocity'."
            )
        self.cdd_reference = reference

        # Project the noisy state onto the sphere for the manifold-native RFM model,
        # so the network is queried on-manifold as it is during RFM sampling.
        if project_to_manifold is None:
            project_to_manifold = self.model_kind == "rfm"
        self.project_to_manifold = bool(project_to_manifold)

        # Cache the clean-image embedding direction used as UniDef's latent z.
        with torch.no_grad():
            z = model_dependent_embedding(self.source_tensor, pipeline, track_grad=False)
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        self.z_dir = z  # (1, D), unit norm

    def _fdje_grad_emb(self, x_t, gamma, emb_det, residual):
        """Estimate grad_emb ||residual||^2 by finite-difference Jacobian (no denoiser autograd)."""
        grad_emb = torch.zeros_like(emb_det)
        for _ in range(self.fdje_num_samples):
            if self.fdje_direction == "gaussian":
                z = torch.randn_like(emb_det)
            else:
                z = self.z_dir.expand_as(emb_det)
            e_plus = self.pipeline.model({"y": x_t, "emb": emb_det + self.fd * z, "gamma": gamma})
            e_minus = self.pipeline.model({"y": x_t, "emb": emb_det - self.fd * z, "gamma": gamma})
            jz = (e_plus - e_minus) / (2.0 * self.fd)  # (B, d) ~= J_e z
            coeff = (jz * residual).sum(dim=-1, keepdim=True)  # <J_e z, residual> (B, 1)
            grad_emb = grad_emb + coeff * z
        return grad_emb / self.fdje_num_samples

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One sign-SGD ascent step on the CDD objective (optionally FDJE-estimated)."""
        optimizer.zero_grad(set_to_none=True)

        # Sample from x0 bank and the forward diffusion process (trajectory integral).
        idx = torch.randint(0, self.x0_bank.shape[0], (self.train_batch_size,), device=self.device)
        x0 = self.x0_bank[idx]
        eps = torch.randn_like(x0)

        t = torch.rand(self.train_batch_size, device=self.device)
        gamma = self.pipeline.scheduler(t)

        x_t = (
            torch.sqrt(gamma).unsqueeze(-1) * x0
            + torch.sqrt(1.0 - gamma).unsqueeze(-1) * eps
        )
        # For the manifold-native RFM model, query the network on-sphere (as during
        # sampling). No-op for the Euclidean flow / diffusion models.
        if self.project_to_manifold:
            x_t = self.pipeline.manifold.projx(x_t)

        # CDD reference, adapted to the model parameterization (see __init__):
        #   - "noise": the true sampled epsilon (DDIM, whose network regresses noise),
        #     giving the faithful UniDef objective max ||eps_theta(x_t) - eps||^2.
        #   - "clean_velocity": the clean conditional velocity field v_theta(x_t,
        #     emb_clean, gamma) (flow / RFM, whose network regresses a velocity).
        # Maximizing the deviation from this reference at every (x_t, gamma) along the
        # trajectory steers the sampling ODE away from the clean geographic
        # distribution (paper Eq. 5/9).
        if self.cdd_reference == "noise":
            reference = eps
        else:
            with torch.no_grad():
                emb_clean = compute_embedding(
                    self.source_tensor,
                    self.train_batch_size,
                    self.pipeline,
                    device=self.device,
                    track_grad=False,
                )
                reference = self.pipeline.model(
                    {"y": x_t, "emb": emb_clean, "gamma": gamma}
                )

        # Embedding of the perturbed conditioning image (tracks grad through the encoder).
        perturbed_source = self.source_tensor + delta
        emb_perturbed = compute_embedding(
            perturbed_source,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=True,
        )

        if not self.use_fdje:
            # Exact backprop through the denoiser ("w/o FDJE" ablation).
            pred_perturbed = self.pipeline.model(
                {"y": x_t, "emb": emb_perturbed, "gamma": gamma}
            )
            # Maximize ||pred_perturbed - reference||^2  <=>  minimize -MSE.
            loss = -torch.nn.functional.mse_loss(pred_perturbed, reference)
            if self.reconstruction_loss_weight > 0:
                loss = loss + self.reconstruction_loss_weight * torch.nn.functional.l1_loss(
                    perturbed_source, self.source_tensor
                )
            loss.backward()
            loss_value = float(loss.item())
        else:
            # FDJE: estimate the gradient w.r.t. the embedding without autograd through
            # the denoiser, then propagate to delta through the encoder via a VJP.
            with torch.no_grad():
                emb_det = emb_perturbed.detach()
                pred_perturbed = self.pipeline.model({"y": x_t, "emb": emb_det, "gamma": gamma})
                residual = pred_perturbed - reference  # trajectory bias to be maximized
                grad_emb = self._fdje_grad_emb(x_t, gamma, emb_det, residual)
            # We maximize ||residual||^2; the sign-SGD step descends, so feed the
            # negated estimated gradient as the upstream grad of the encoder VJP.
            emb_perturbed.backward(gradient=-grad_emb)
            if self.reconstruction_loss_weight > 0:
                recon = self.reconstruction_loss_weight * torch.nn.functional.l1_loss(
                    perturbed_source, self.source_tensor
                )
                recon.backward()
            loss_value = float(-(residual ** 2).mean().item())

        # Sign-SGD update in the l_inf ball (projected gradient ascent, paper Eq. 17).
        with torch.no_grad():
            delta.grad = torch.sign(delta.grad)
            optimizer.step()
            delta.data = torch.clamp(delta.data, -self.eps_max, self.eps_max)
            delta.grad.zero_()

        return loss_value

    def get_config(self) -> Dict[str, Any]:
        """Return attack configuration."""
        config = super().get_config()
        config["use_fdje"] = self.use_fdje
        config["fd"] = self.fd
        config["fdje_direction"] = self.fdje_direction
        config["fdje_num_samples"] = self.fdje_num_samples
        config["model_kind"] = self.model_kind
        config["cdd_reference"] = self.cdd_reference
        config["project_to_manifold"] = self.project_to_manifold
        config["attack_mode"] = "untargeted"
        return config


############################################################################################
# TrainingLossAttack: maximize the model's own training loss
############################################################################################


class TrainingLossAttack(DiffusionAttack):
    """Maximize the model's training loss w.r.t. the conditioning image.

    Objective: max_delta E_{t, noise}[||network(x_t, emb(I+delta), gamma) - v_true||^2]

    where the forward process and true target v_true adapt to the model parameterization:

    - **diffusion (DDPM)**: x_t = sqrt(gamma)*x0 + sqrt(1-gamma)*n,  v_true = n
    - **flow matching**:    x_t = gamma*x0 + (1-gamma)*n,            v_true = x0 - n
    - **Riemannian FM**:    x_t = geodesic(x0_sphere, x0_data, gamma), v_true = d(geodesic)/d(gamma)

    Unlike the other DiffusionAttack variants (which compare the perturbed network output
    against the *clean model's* prediction), this attack uses the true training target, so
    it directly maximises the loss that the model was trained to minimise.
    """

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
        reconstruction_loss_weight: float = 0.0,
        delta_init: float = 1e-4,
        num_restarts: int = 1,
        restart_selection_metric: str = "final_step_displacement",
        device: str = "cuda",
        x0_bank: Optional[torch.Tensor] = None,
    ):
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
            dot_product_loss="l2",  # not used; run_step is fully overridden
            reconstruction_loss_weight=reconstruction_loss_weight,
            delta_init=delta_init,
            num_restarts=num_restarts,
            restart_selection_metric=restart_selection_metric,
            device=device,
            x0_bank=x0_bank,
        )
        self.model_kind = detect_model_kind(pipeline)
        self.manifold = getattr(pipeline, "manifold", None) if self.model_kind == "rfm" else None

    def run_step(
        self,
        delta: torch.Tensor,
        step: int,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """One sign-SGD step maximizing the model's training loss."""
        optimizer.zero_grad(set_to_none=True)

        idx = torch.randint(0, self.x0_bank.shape[0], (self.train_batch_size,), device=self.device)
        x0 = self.x0_bank[idx]
        n = torch.randn_like(x0)
        t = torch.rand(self.train_batch_size, device=self.device)
        gamma = self.pipeline.scheduler(t)

        if self.model_kind == "diffusion":
            # DDPM forward process; network predicts noise n
            x_t = torch.sqrt(gamma).unsqueeze(-1) * x0 + torch.sqrt(1.0 - gamma).unsqueeze(-1) * n
            v_true = n

        elif self.model_kind == "flow":
            # Linear flow matching forward process; network predicts velocity (x0 - n)
            x_t = gamma.unsqueeze(-1) * x0 + (1.0 - gamma).unsqueeze(-1) * n
            v_true = x0 - n

        else:  # "rfm"
            # Riemannian flow matching: geodesic from a random sphere point to the data point.
            # x0_bank holds data (GPS as Cartesian sphere points) = x1 in training notation;
            # x0_sphere is the noise end (uniform random on sphere) = x0 in training notation.
            x0_sphere = self.manifold.random_base(self.train_batch_size, x0.shape[-1]).to(
                device=self.device, dtype=x0.dtype
            )
            gamma_exp = gamma.unsqueeze(-1)  # (B, 1) — vmap maps over batch dimension

            def _cond_u(x0_s, x1, g):
                path = _sphere_geodesic(self.manifold, x0_s, x1)
                x_t_i, u_t_i = _func_jvp(path, (g,), (torch.ones_like(g),))
                return x_t_i.squeeze(-2), u_t_i.squeeze(-2)

            x_t_batch, v_true_batch = _func_vmap(_cond_u)(x0_sphere, x0, gamma_exp)
            x_t = x_t_batch.reshape(self.train_batch_size, -1)
            v_true = v_true_batch.reshape(self.train_batch_size, -1)

        # Perturbed embedding (gradient tracked through the encoder)
        perturbed_source = self.source_tensor + delta
        emb_perturbed = compute_embedding(
            perturbed_source,
            self.train_batch_size,
            self.pipeline,
            device=self.device,
            track_grad=True,
        )

        pred = self.pipeline.model({"y": x_t, "emb": emb_perturbed, "gamma": gamma})

        # Maximize ||pred - v_true||^2 by minimizing the negated MSE
        if self.model_kind == "rfm":
            diff = pred - v_true
            loss = -(self.manifold.inner(x_t, diff, diff).mean() / x_t.shape[-1])
        else:
            loss = -torch.nn.functional.mse_loss(pred, v_true)

        if self.reconstruction_loss_weight > 0:
            loss = loss + self.reconstruction_loss_weight * torch.nn.functional.l1_loss(
                perturbed_source, self.source_tensor
            )

        loss.backward()

        with torch.no_grad():
            delta.grad = torch.sign(delta.grad)
            optimizer.step()
            delta.data = torch.clamp(delta.data, -self.eps_max, self.eps_max)
            delta.grad.zero_()

        return float(loss.item())

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config["model_kind"] = self.model_kind
        config.pop("target_pure_noise", None)
        config.pop("dot_product_loss", None)
        return config
