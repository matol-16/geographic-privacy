"""Truncated-backprop sampling attack.

Like :class:`SamplingAttack` (Salman et al.), this attack optimizes the haversine
distance between the clean and perturbed model predictions by differentiating
*through the sampler*. The difference is purely how much of the sampling trajectory
is kept in the autograd graph:

  - ``SamplingAttack`` runs a short ``sampling_steps_salman`` (default 16) trajectory
    and backpropagates through **every** step. The forward trajectory is therefore a
    coarse 16-step approximation of the model.

  - ``TruncatedSamplingAttack`` runs the full-fidelity ``total_sampling_steps``
    (default 250) trajectory in the forward pass, but only keeps the autograd graph
    for the **last** ``backprop_steps`` (default 16) steps. The earlier steps run under
    ``torch.no_grad()``, so activation memory is bounded by ``backprop_steps`` rather
    than by the full trajectory length.

This is *truncated backpropagation through time*: the gradient is the exact gradient
of the final ``backprop_steps`` denoising steps with respect to the image embedding,
treating the state that enters the differentiable window as a constant. Because the
image embedding conditions the network at every step, those final steps still carry a
useful gradient signal, while the expensive 234-step prefix is evaluated cheaply
without storing activations.

Note the window must be the *tail* of the trajectory: the loss is a function of the
final sample ``x_0``, and a ``no_grad`` suffix would sever the gradient path back to
any earlier window. Backpropagating through the last ``backprop_steps`` steps is the
only truncation that yields a non-zero gradient here.

The default 16-of-250 configuration is the requested baseline.
"""

import contextlib
from typing import Any, Dict, Optional

from PIL import Image
import torch

from plonk.utils.manifolds import Sphere

from attacks.diffusion_attack_salman import SamplingAttack
from attacks.trajectory_deviation import detect_model_kind


class TruncatedSamplingAttack(SamplingAttack):
    """Sampling attack that backpropagates through only the tail of a long trajectory."""

    def __init__(
        self,
        pipeline,
        source_image: Image.Image,
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
            # The parent's `sampling_steps_salman` is unused here (we override the
            # sampler), but keep it aligned with `backprop_steps` for consistency.
            sampling_steps_salman=backprop_steps,
            clean_num_steps=clean_num_steps,
            target_pure_noise=target_pure_noise,
            dot_product_loss=dot_product_loss,
            reconstruction_loss_weight=reconstruction_loss_weight,
            delta_init=delta_init,
            num_restarts=num_restarts,
            restart_selection_metric=restart_selection_metric,
            device=device,
            x0_bank=x0_bank,
        )

        self.total_sampling_steps = int(total_sampling_steps)
        self.backprop_steps = int(backprop_steps)
        if self.backprop_steps < 1:
            raise ValueError("backprop_steps must be >= 1 (otherwise no gradient flows).")
        if self.backprop_steps > self.total_sampling_steps:
            raise ValueError(
                "backprop_steps cannot exceed total_sampling_steps "
                f"({self.backprop_steps} > {self.total_sampling_steps})."
            )

        # The sampler math depends on the generative parameterization; auto-detect it.
        self.model_kind = detect_model_kind(pipeline)
        self.manifold = getattr(pipeline, "manifold", None) or Sphere()

    def _integrate_step(
        self,
        x_cur: torch.Tensor,
        denoised: torch.Tensor,
        gamma_now: torch.Tensor,
        gamma_next: torch.Tensor,
    ) -> torch.Tensor:
        """One sampler update, mirroring the library sampler for the detected model kind."""
        if self.model_kind == "diffusion":
            # DDIM update (see plonk/models/samplers/ddim.py).
            x_pred = (x_cur - torch.sqrt(1 - gamma_now) * denoised) / torch.sqrt(gamma_now)
            x_pred = torch.clamp(x_pred, -1, 1)
            noise_pred = (x_cur - torch.sqrt(gamma_now) * x_pred) / torch.sqrt(1 - gamma_now)
            return torch.sqrt(gamma_next) * x_pred + torch.sqrt(1 - gamma_next) * noise_pred

        # (Riemannian) flow matching: Euler step along the predicted velocity.
        dt = gamma_next - gamma_now
        x_next = x_cur + dt * denoised
        if self.model_kind == "rfm":
            x_next = self.manifold.projx(x_next)
        return x_next

    def _sample_cartesian(self, embedding: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Run a ``total_sampling_steps`` trajectory, keeping grad only for the last ``backprop_steps``.

        The forward pass is identical to running the library sampler with
        ``num_steps=total_sampling_steps`` and ``cfg_rate=0``; the only change is that
        the leading ``total_sampling_steps - backprop_steps`` steps are wrapped in
        ``torch.no_grad()`` so their activations are never stored.
        """
        num_steps = self.total_sampling_steps
        split = num_steps - self.backprop_steps  # steps [0, split) run without grad

        net = self.pipeline.model
        scheduler = self.pipeline.scheduler

        x_cur = x_t.to(torch.float32)
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)
        steps = 1 - step_indices / num_steps
        gammas = scheduler(steps)

        # Match the AMP dtype each library sampler uses for its network evaluation.
        if self.model_kind == "diffusion":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            amp_dtype = torch.float32

        batch = {"emb": embedding}
        for step in range(num_steps):
            gamma_now = gammas[step]
            gamma_next = gammas[step + 1]
            differentiable = step >= split
            # nullcontext keeps the ambient grad mode: this leaves the clean pass
            # (called under torch.no_grad in run_step) fully grad-free, while the
            # perturbed pass tracks grad only once we cross into the tail window.
            grad_ctx = contextlib.nullcontext() if differentiable else torch.no_grad()
            with grad_ctx:
                with torch.cuda.amp.autocast(dtype=amp_dtype):
                    batch["y"] = x_cur
                    batch["gamma"] = gamma_now.expand(x_cur.shape[0])
                    denoised = net(batch)
                x_cur = self._integrate_step(x_cur, denoised, gamma_now, gamma_next)

        return x_cur.to(torch.float32)

    def get_config(self) -> Dict[str, Any]:
        """Return attack configuration (parent config + truncated-backprop provenance)."""
        config = super().get_config()
        config.update(
            {
                "objective": "truncated_backprop_sampling",
                "total_sampling_steps": self.total_sampling_steps,
                "backprop_steps": self.backprop_steps,
                "model_kind": self.model_kind,
            }
        )
        return config
