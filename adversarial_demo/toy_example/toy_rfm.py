"""Tiny Riemannian flow-matching model on S^2 for the direction-vs-magnitude toy.

This is the controlled experiment behind the paper claim:

    "Under high CFG, geolocation sampling is essentially basin selection on a
     compact manifold: direction chooses the basin; magnitude merely chooses the
     speed within it."

We build a *two-mode* velocity field on the sphere and fit a small RFM to it,
reusing the exact geometry and training objective of the real PLONK models:

  - ``plonk.utils.manifolds.Sphere``            -- the manifold (logmap/expmap/projx)
  - ``plonk.models.losses.RiemannianFlowMatchingLoss`` -- the training objective
  - ``plonk.models.schedulers.SigmoidScheduler``       -- the time schedule
  - ``plonk.models.samplers.riemannian_flow_sampler``  -- the Euler-on-manifold sampler

The only new pieces are (a) a tiny MLP velocity network and (b) an integrator
that mirrors ``riemannian_flow_sampler`` but lets us perturb the velocity at each
step -- either by *rotating* it in the tangent plane (a direction/cosine
deviation) or by *rescaling* its norm (an L2/magnitude deviation) -- so we can
compare endpoint displacement under perturbations of identical L2 size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from plonk.utils.manifolds import Sphere
from plonk.models.schedulers import SigmoidScheduler
from plonk.models.losses import RiemannianFlowMatchingLoss

MANIFOLD = Sphere()


# --------------------------------------------------------------------------- #
# Geometry helpers (thin wrappers around the shared Sphere manifold)
# --------------------------------------------------------------------------- #
def normalize(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Project ``v`` onto the tangent space T_x S^2 (remove the radial part)."""
    return v - (v * x).sum(dim=-1, keepdim=True) * x


def geodesic_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Great-circle (angular) distance between unit vectors, in radians."""
    dot = (normalize(x) * normalize(y)).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(dot)


def rotate_in_tangent(x: torch.Tensor, v: torch.Tensor, theta: float) -> torch.Tensor:
    """Rotate the tangent vector ``v`` by angle ``theta`` within T_x S^2.

    The rotation axis is the outward normal ``x``; ``x cross v`` is ``v`` turned a
    quarter-turn inside the tangent plane with the same norm, so Rodrigues gives a
    pure in-plane rotation that preserves ``||v||`` and stays tangent.
    """
    x = normalize(x)
    v = project_tangent(x, v)
    axis_cross_v = torch.cross(x, v, dim=-1)
    return math.cos(theta) * v + math.sin(theta) * axis_cross_v


# --------------------------------------------------------------------------- #
# Two-mode target distribution on the sphere
# --------------------------------------------------------------------------- #
@dataclass
class TwoModeTarget:
    """A two-mode data distribution on S^2: pick a mode, then a tight blob."""

    # Two well-separated modes (~104 deg apart), symmetric about the y=0 plane so
    # the separatrix is a clean meridian great circle, and tilted toward the
    # viewer (+x) and up (+z) so both basins are visible in the figure.
    mode_a: torch.Tensor = field(
        default_factory=lambda: normalize(torch.tensor([0.6, 1.0, 0.5]))
    )
    mode_b: torch.Tensor = field(
        default_factory=lambda: normalize(torch.tensor([0.6, -1.0, 0.5]))
    )
    spread: float = 0.05  # angular std of each blob (tight = sharp, contracting basins)
    weight_a: float = 0.5

    def sample(self, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        modes = torch.stack([self.mode_a, self.mode_b], dim=0)
        pick = (torch.rand(n, generator=generator) >= self.weight_a).long()
        base = modes[pick]
        noise = torch.randn(n, 3, generator=generator) * self.spread
        return normalize(base + project_tangent(base, noise))


# --------------------------------------------------------------------------- #
# Tiny velocity network v_theta(x, gamma) : S^2 x [0,1] -> T_x S^2
# --------------------------------------------------------------------------- #
def gamma_features(gamma: torch.Tensor, n_freq: int = 6) -> torch.Tensor:
    gamma = gamma.reshape(-1, 1)
    freqs = 2.0 ** torch.arange(n_freq, device=gamma.device, dtype=gamma.dtype)
    ang = gamma * freqs * math.pi
    return torch.cat([gamma, torch.sin(ang), torch.cos(ang)], dim=-1)


class ToyRFM(nn.Module):
    """Small MLP predicting a tangent velocity on S^2, conditioned on gamma."""

    def __init__(self, hidden: int = 128, n_freq: int = 6):
        super().__init__()
        in_dim = 3 + (1 + 2 * n_freq)
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        feats = torch.cat([x, gamma_features(gamma, self.n_freq)], dim=-1)
        raw = self.net(feats)
        return project_tangent(x, raw)  # enforce tangency, matching u_t


def make_batch_forward(model: ToyRFM):
    """Adapter so the network fits both the RFM loss and the reference sampler.

    - ``preconditioning(network, batch)`` -> velocity  (used by RiemannianFlowMatchingLoss)
    - ``net(batch)`` -> velocity                        (used by riemannian_flow_sampler)
    """

    def preconditioning(network, batch):
        return network(batch["y"], batch["gamma"])

    def net(batch):  # matches the sampler's ``net(batch)`` call for cfg_rate=0
        return model(batch["y"], batch["gamma"])

    return preconditioning, net


# --------------------------------------------------------------------------- #
# Training (reuses the real RiemannianFlowMatchingLoss)
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    steps: int = 8000
    batch_size: int = 512
    lr: float = 2e-3
    hidden: int = 128
    seed: int = 0
    sched_start: float = -7.0  # PLONK sigmoid-scheduler defaults (see pipe.py)
    sched_end: float = 3.0
    sched_tau: float = 1.0


def train_toy_rfm(
    target: TwoModeTarget,
    cfg: TrainConfig,
    device: str = "cpu",
    log_every: int = 500,
    silent: bool = False,
) -> ToyRFM:
    torch.manual_seed(cfg.seed)
    scheduler = SigmoidScheduler(cfg.sched_start, cfg.sched_end, cfg.sched_tau)
    loss_fn = RiemannianFlowMatchingLoss(
        scheduler=scheduler, cond_drop_rate=0.0, conditioning_key="label"
    )
    model = ToyRFM(hidden=cfg.hidden).to(device)
    preconditioning, _ = make_batch_forward(model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed + 1)

    model.train()
    for step in range(cfg.steps):
        x0 = target.sample(cfg.batch_size, generator=gen).to(device)
        # The loss reads batch["x_0"] as the *data* endpoint and, with a None
        # conditioning entry + cond_drop_rate=0, trains the unconditional field.
        batch = {"x_0": x0, "label": None}
        loss = loss_fn(preconditioning, model, batch).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if not silent and (step % log_every == 0 or step == cfg.steps - 1):
            print(f"  [train] step {step:5d}/{cfg.steps}  loss {loss.item():.5f}")
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Integrators: reference (no perturbation) and perturbed
# --------------------------------------------------------------------------- #
def gamma_grid(
    num_steps: int, cfg: TrainConfig, device: str = "cpu"
) -> torch.Tensor:
    """The exact gamma schedule used by ``riemannian_flow_sampler``."""
    scheduler = SigmoidScheduler(cfg.sched_start, cfg.sched_end, cfg.sched_tau)
    idx = torch.arange(num_steps + 1, dtype=torch.float32, device=device)
    steps = 1 - idx / num_steps
    return scheduler(steps)


PerturbFn = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


@torch.no_grad()
def integrate(
    model: ToyRFM,
    x_start: torch.Tensor,
    num_steps: int,
    cfg: TrainConfig,
    perturb: Optional[PerturbFn] = None,
    device: str = "cpu",
    return_traj: bool = False,
):
    """Euler-on-manifold integration, mirroring ``riemannian_flow_sampler``.

    ``perturb(x, v, step)`` (optional) maps the model velocity to a perturbed
    velocity before the Euler step. Endpoint (and optional trajectory) returned.
    """
    x_cur = normalize(x_start.to(device).clone())
    if x_cur.ndim == 1:
        x_cur = x_cur.unsqueeze(0)
    gammas = gamma_grid(num_steps, cfg, device=device)
    traj = [x_cur.clone()] if return_traj else None
    for step, (g_now, g_next) in enumerate(zip(gammas[:-1], gammas[1:])):
        g = g_now.expand(x_cur.shape[0])
        v = model(x_cur, g)
        if perturb is not None:
            v = perturb(x_cur, v, step)
        dt = (g_next - g_now)
        x_cur = MANIFOLD.projx(x_cur + dt * v)
        if return_traj:
            traj.append(x_cur.clone())
    if return_traj:
        return x_cur, torch.stack(traj, dim=0)  # [T+1, B, 3]
    return x_cur


def angle_perturbation(theta: float) -> PerturbFn:
    """Rotate the velocity by a fixed angle in the tangent plane (cosine deviation)."""

    def fn(x, v, step):
        return rotate_in_tangent(x, v, theta)

    return fn


def windowed_angle_perturbation(theta: float, num_steps: int,
                                frac: float = 0.4) -> PerturbFn:
    """Rotate the velocity only during the *early* (high-uncertainty) phase, then
    follow the clean field. This is the basin-selection picture: the direction
    choice is made near the separatrix, after which the (unperturbed) flow carries
    the trajectory into the chosen basin -- so it lands cleanly on the far mode
    instead of spiralling under a rotation that never switches off."""
    cutoff = int(frac * num_steps)

    def fn(x, v, step):
        return rotate_in_tangent(x, v, theta) if step < cutoff else v

    return fn


def norm_perturbation(scale: float) -> PerturbFn:
    """Rescale the velocity norm (L2/magnitude deviation), direction unchanged."""

    def fn(x, v, step):
        return (1.0 + scale) * v

    return fn


# --------------------------------------------------------------------------- #
# Matched-L2 perturbation strength
# --------------------------------------------------------------------------- #
def theta_for_strength(s: float) -> float:
    """Rotation angle whose per-step perturbation has relative L2 size ``s``.

    Rotating a tangent vector v by theta changes it by ||v||*2*sin(theta/2);
    rescaling by (1+c) changes it by ||v||*|c|. Setting |c| = s and
    2*sin(theta/2) = s makes the two perturbations identical in per-step (hence
    total-trajectory) L2 size -- the "equal L2" comparison in the paper.
    """
    s = min(max(s, 0.0), 2.0)
    return 2.0 * math.asin(s / 2.0)
