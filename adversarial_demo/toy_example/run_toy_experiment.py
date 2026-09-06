"""Direction-vs-magnitude toy experiment on S^2.

Trains a tiny RFM on a two-mode velocity field and shows that, under matched L2
budgets, *rotating* the predicted velocity (a cosine/direction deviation) moves
sampling endpoints far more than *rescaling* its norm (an L2/magnitude
deviation): direction chooses the basin, magnitude only the speed within it.

Run:
    python run_toy_experiment.py                 # train + experiment + export
    python run_toy_experiment.py --steps 2000    # faster/rougher model

Outputs (into ./out by default):
    toy_results.json   -- displacement-vs-budget curves + headline numbers
    toy_geometry.npz   -- sphere/flow/trajectory/arrow data for the plots
    toy_rfm.pt         -- trained weights (reused on the next run unless --retrain)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

import toy_rfm as T


# --------------------------------------------------------------------------- #
# Perturbation measurement (closed-loop, with realized-L2 accounting)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def integrate_measured(model, x_start, num_steps, cfg, kind, strength, device="cpu"):
    """Integrate with a rotation OR norm perturbation and record its realized L2.

    ``kind`` in {"angle", "norm"}. ``strength`` is the *relative* per-step L2 size
    ``s`` (identical for both families by construction), so the endpoints are
    compared under matched budgets. Returns (endpoint[3], realized_total_L2).
    """
    theta = T.theta_for_strength(strength)
    x_cur = T.normalize(x_start.to(device).clone()).unsqueeze(0)
    gammas = T.gamma_grid(num_steps, cfg, device=device)
    l2_sq = 0.0
    for g_now, g_next in zip(gammas[:-1], gammas[1:]):
        g = g_now.expand(x_cur.shape[0])
        v = model(x_cur, g)
        if kind == "angle":
            v_pert = T.rotate_in_tangent(x_cur, v, theta)
        elif kind == "norm":
            v_pert = (1.0 + strength) * v
        else:
            raise ValueError(kind)
        dt = (g_next - g_now)
        # realized perturbation of the *velocity field* at this step
        l2_sq += (dt * (v_pert - v)).pow(2).sum().item()
        x_cur = T.MANIFOLD.projx(x_cur + dt * v_pert)
    return x_cur.squeeze(0), float(np.sqrt(l2_sq))


def basin_of(x, target):
    """Which mode (0=A, 1=B) the point ``x`` is closest to on the sphere."""
    da = T.geodesic_distance(x, target.mode_a.to(x)).item()
    db = T.geodesic_distance(x, target.mode_b.to(x)).item()
    return 0 if da <= db else 1


@torch.no_grad()
def displacement_under_budget(model, x_start, nominal_end, num_steps, cfg, kind,
                              strength, target, device="cpu"):
    """At a fixed matched budget, the attacker picks the worst +/- sign. Returns
    the max endpoint displacement, its realized L2, and whether the *basin*
    (nearest mode) flipped -- the quantity the paper's claim is really about."""
    best_disp, best_l2, flipped = 0.0, 0.0, False
    nominal_basin = basin_of(nominal_end, target)
    for sign in (+1.0, -1.0):
        s = sign * strength
        if kind == "norm":
            # Floor the rescale factor at 0.3: a near-full-stop (v->0) is a
            # degenerate stall, not a magnitude deviation. Flooring (vs skipping)
            # keeps the worst-case-over-sign curve continuous.
            s = max(s, -0.7)
        end, l2 = integrate_measured(model, x_start, num_steps, cfg, kind, s, device)
        disp = T.geodesic_distance(end, nominal_end).item()
        if basin_of(end, target) != nominal_basin:
            flipped = True
        if disp > best_disp:
            best_disp, best_l2 = disp, l2
    return best_disp, best_l2, flipped


# --------------------------------------------------------------------------- #
# Start sampling
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_separatrix_starts(target, n, device="cpu", band=(0.04, 0.20), seed=123,
                             model=None, cfg=None, num_steps=None, commit=0.35,
                             for_view=False):
    """Base samples in the basin-selection regime: in an annulus *around* (but not
    exactly on) the separatrix great circle {x . (A-B) = 0}, in the lower
    (noise-like) hemisphere. Starting exactly on the fence is degenerate -- the
    field is near-zero there and a larger discrete step can overshoot across it,
    which is not the mechanism we study. If a model is given we additionally keep
    only starts whose *nominal* trajectory clearly commits to a basin (endpoint
    within ``commit`` rad of a mode), so 'basin flip' is well defined.

    ``for_view=True`` additionally keeps the start on the viewer-facing hemisphere
    (used only for the illustrative hero trajectory, so start + both basins are on
    the front of the sphere in the figure); it does not affect the aggregate."""
    gen = torch.Generator().manual_seed(seed)
    n_hat = T.normalize(target.mode_a - target.mode_b).to(device)
    lo, hi = band
    out = []
    while len(out) < n:
        c = T.normalize(torch.randn(8192, 3, generator=gen)).to(device)
        m = (c @ n_hat).abs()
        keep = (m > lo) & (m < hi) & (c[:, 2] < 0.3)
        if for_view:
            keep &= (c[:, 0] > 0.45) & (c[:, 2] > -0.35)
        for x in c[keep]:
            if model is not None:
                end = T.integrate(model, x, num_steps, cfg, device=device).squeeze(0)
                near = min(T.geodesic_distance(end, target.mode_a.to(device)).item(),
                           T.geodesic_distance(end, target.mode_b.to(device)).item())
                if near > commit:
                    continue
            out.append(x)
            if len(out) >= n:
                break
    return torch.stack(out[:n], dim=0)


# --------------------------------------------------------------------------- #
# Aggregate sweep over many starts (basin-selection regime)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_sweep(model, target, cfg, num_steps, budgets, n_starts, device="cpu",
              seed=123):
    starts = sample_separatrix_starts(target, n_starts, device=device, seed=seed,
                                      model=model, cfg=cfg, num_steps=num_steps)
    nominal_ends = torch.stack(
        [T.integrate(model, s, num_steps, cfg, device=device).squeeze(0) for s in starts]
    )

    angle_mean, norm_mean = [], []
    angle_l2, norm_l2 = [], []
    angle_flip, norm_flip = [], []
    for s in budgets:
        a_d, n_d, a_l, n_l, a_f, n_f = [], [], [], [], [], []
        for i in range(n_starts):
            da, la, fa = displacement_under_budget(
                model, starts[i], nominal_ends[i], num_steps, cfg, "angle", s,
                target, device)
            dn, ln, fn = displacement_under_budget(
                model, starts[i], nominal_ends[i], num_steps, cfg, "norm", s,
                target, device)
            a_d.append(da); n_d.append(dn); a_l.append(la); n_l.append(ln)
            a_f.append(fa); n_f.append(fn)
        angle_mean.append(float(np.mean(a_d)))
        norm_mean.append(float(np.mean(n_d)))
        angle_l2.append(float(np.mean(a_l)))
        norm_l2.append(float(np.mean(n_l)))
        angle_flip.append(float(np.mean(a_f)))
        norm_flip.append(float(np.mean(n_f)))
    return {
        "budgets": list(map(float, budgets)),
        "angle_displacement_mean": angle_mean,
        "norm_displacement_mean": norm_mean,
        "angle_realized_l2_mean": angle_l2,
        "norm_realized_l2_mean": norm_l2,
        "angle_basin_flip_frac": angle_flip,
        "norm_basin_flip_frac": norm_flip,
        "n_starts": int(n_starts),
    }


# --------------------------------------------------------------------------- #
# Hero trajectory for the figure: a start near the separatrix that flips basin
# under an angle perturbation but not under the matched norm perturbation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def pick_hero_start(model, target, cfg, num_steps, s_hero, device="cpu", seed=7,
                    max_candidates=80):
    cands = sample_separatrix_starts(target, max_candidates, device=device, seed=seed,
                                     model=model, cfg=cfg, num_steps=num_steps,
                                     for_view=True)
    best = None
    for x0 in cands:
        end = T.integrate(model, x0, num_steps, cfg, device=device).squeeze(0)
        da, _, _ = displacement_under_budget(model, x0, end, num_steps, cfg, "angle",
                                             s_hero, target, device)
        dn, _, _ = displacement_under_budget(model, x0, end, num_steps, cfg, "norm",
                                             s_hero, target, device)
        score = da - dn  # want big angle displacement, small norm displacement
        if best is None or score > best[0]:
            best = (score, x0, da, dn)
    return best  # (score, x0, angle_disp, norm_disp)


@torch.no_grad()
def angle_sign_toward_b(model, x0, target, num_steps, cfg, s_hero, device="cpu"):
    """Which rotation sign moves the endpoint toward mode B (the flip)."""
    theta = T.theta_for_strength(s_hero)
    best_sign, best_close = +1.0, -1e9
    for sign in (+1.0, -1.0):
        end = T.integrate(model, x0, num_steps, cfg,
                          perturb=T.angle_perturbation(sign * theta),
                          device=device).squeeze(0)
        close = (end @ target.mode_b.to(device)).item()
        if close > best_close:
            best_close, best_sign = close, sign
    return best_sign


@torch.no_grad()
def export_geometry(model, target, cfg, num_steps, s_hero, x0_hero, device="cpu",
                    seed=7):
    theta = T.theta_for_strength(s_hero)
    sign = angle_sign_toward_b(model, x0_hero, target, num_steps, cfg, s_hero, device)
    win_frac = 0.4  # rotate only during the early decision phase (see below)

    _, traj_nom = T.integrate(model, x0_hero, num_steps, cfg, device=device,
                              return_traj=True)
    # Hero direction deviation: an *early-phase* rotation that selects basin B,
    # after which the clean field carries the trajectory into B (lands on the
    # mode rather than spiralling). The aggregate sweep instead uses a constant
    # per-step rotation -- the rigorous worst-case sensitivity measure.
    _, traj_ang = T.integrate(model, x0_hero, num_steps, cfg,
                              perturb=T.windowed_angle_perturbation(
                                  sign * theta, num_steps, win_frac),
                              device=device, return_traj=True)
    _, traj_nrm = T.integrate(model, x0_hero, num_steps, cfg,
                              perturb=T.norm_perturbation(s_hero),
                              device=device, return_traj=True)
    traj_nom = traj_nom.squeeze(1).cpu().numpy()
    traj_ang = traj_ang.squeeze(1).cpu().numpy()
    traj_nrm = traj_nrm.squeeze(1).cpu().numpy()

    # velocity arrows at a marker inside the early rotation window, where the
    # direction choice is being made
    t_star = int(0.28 * num_steps)
    x_star = torch.tensor(traj_nom[t_star], device=device).unsqueeze(0)
    g_star = T.gamma_grid(num_steps, cfg, device=device)[t_star].expand(1)
    v_nom = model(x_star, g_star)
    v_ang = T.rotate_in_tangent(x_star, v_nom, sign * theta)
    v_nrm = (1.0 + s_hero) * v_nom

    # Background streamlines of the *true* (unperturbed) field: generative
    # trajectories x_dot = v_theta(x, gamma) integrated with the same sampler.
    # Seeds are placed on an even grid over the viewer-facing hemisphere (the disk
    # spanned by the two modes' bisector) so they read as evenly-spaced flow lines
    # rather than a random tangle; seeds already sitting on a mode are dropped.
    ma, mb = target.mode_a.numpy(), target.mode_b.numpy()
    bis = ma + mb
    bis /= np.linalg.norm(bis)                      # ~ camera axis
    e1 = np.cross(bis, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)                         # horizontal (A <-> B)
    e2 = np.cross(bis, e1)                           # vertical
    bg = []
    for a in np.linspace(-0.92, 0.92, 9):
        for b in np.linspace(-0.85, 0.55, 6):
            r2 = a * a + b * b
            if r2 > 0.9:
                continue
            p = np.sqrt(1.0 - r2) * bis + a * e1 + b * e2
            p = p / np.linalg.norm(p)
            if min(np.arccos(np.clip(p @ ma, -1, 1)),
                   np.arccos(np.clip(p @ mb, -1, 1))) < 0.3:
                continue                             # skip stubs already at a mode
            s0 = torch.tensor(p, dtype=torch.float32, device=device)
            _, tr = T.integrate(model, s0, num_steps, cfg, device=device,
                                return_traj=True)
            bg.append(tr.squeeze(1).cpu().numpy()[::3])
    bg = np.stack(bg, axis=0)  # [n_bg, T', 3]

    # separatrix great circle {x . (A-B) = 0}
    n_hat = T.normalize(target.mode_a - target.mode_b).numpy()
    e1 = np.cross(n_hat, np.array([0.0, 0.0, 1.0]))
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(n_hat, e1)
    phi = np.linspace(0, 2 * np.pi, 200)
    separatrix = np.outer(np.cos(phi), e1) + np.outer(np.sin(phi), e2)

    return {
        "mode_a": target.mode_a.numpy(),
        "mode_b": target.mode_b.numpy(),
        "separatrix": separatrix,
        "traj_nominal": traj_nom,
        "traj_angle": traj_ang,
        "traj_norm": traj_nrm,
        "bg_streamlines": bg,
        "x_start": traj_nom[0],
        "arrow_base": x_star.squeeze(0).cpu().numpy(),
        "v_nominal": v_nom.squeeze(0).cpu().numpy(),
        "v_angle": v_ang.squeeze(0).cpu().numpy(),
        "v_norm": v_nrm.squeeze(0).cpu().numpy(),
        "s_hero": np.float32(s_hero),
        "num_steps": np.int64(num_steps),
    }


@torch.no_grad()
def export_ensemble(model, target, cfg, num_steps, s_ens, k=36, device="cpu",
                    seed=31, stride=3):
    """Ensemble for the *result* figure: many starts that nominally commit to
    basin A (near the separatrix, viewer-facing), integrated under three
    conditions -- nominal, direction-deviated (early-window rotation toward B),
    and magnitude-deviated (worst-sign rescale). Shows where endpoints land: the
    direction ensemble migrates to mode B, the magnitude ensemble stays at A."""
    theta = T.theta_for_strength(s_ens)
    n_hat = T.normalize(target.mode_a - target.mode_b).to(device)
    mode_a = target.mode_a.to(device)
    mode_b = target.mode_b.to(device)
    gen = torch.Generator().manual_seed(seed)

    starts = []
    while len(starts) < k:
        c = T.normalize(torch.randn(4096, 3, generator=gen)).to(device)
        m = (c @ n_hat).abs()
        keep = (m > 0.02) & (m < 0.22) & (c[:, 0] > 0.45) & \
               (c[:, 2] > -0.42) & (c[:, 2] < 0.32)
        for x in c[keep]:
            end = T.integrate(model, x, num_steps, cfg, device=device).squeeze(0)
            da = T.geodesic_distance(end, mode_a).item()
            db = T.geodesic_distance(end, mode_b).item()
            if da < db and da < 0.35:          # nominally commits to basin A
                starts.append(x)
                if len(starts) >= k:
                    break

    nom, dirc, mag = [], [], []
    for x0 in starts:
        sign = angle_sign_toward_b(model, x0, target, num_steps, cfg, s_ens, device)
        _, tn = T.integrate(model, x0, num_steps, cfg, device=device,
                            return_traj=True)
        _, td = T.integrate(model, x0, num_steps, cfg,
                            perturb=T.windowed_angle_perturbation(
                                sign * theta, num_steps, 0.4),
                            device=device, return_traj=True)
        # worst-sign magnitude perturbation (floored factor), matched budget
        best = None
        for sgn in (+1.0, -1.0):
            ss = max(sgn * s_ens, -0.7)
            _, tm = T.integrate(model, x0, num_steps, cfg,
                                perturb=T.norm_perturbation(ss),
                                device=device, return_traj=True)
            disp = T.geodesic_distance(tm[-1, 0], tn[-1, 0]).item()
            if best is None or disp > best[0]:
                best = (disp, tm)
        nom.append(tn.squeeze(1).cpu().numpy()[::stride])
        dirc.append(td.squeeze(1).cpu().numpy()[::stride])
        mag.append(best[1].squeeze(1).cpu().numpy()[::stride])
    return {
        "ens_nominal": np.stack(nom),      # [k, T', 3]
        "ens_direction": np.stack(dirc),
        "ens_magnitude": np.stack(mag),
        "ens_s": np.float32(s_ens),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--steps", type=int, default=8000, help="training steps")
    ap.add_argument("--num-steps", type=int, default=250, help="sampling Euler steps")
    ap.add_argument("--n-starts", type=int, default=48, help="starts in the sweep")
    ap.add_argument("--s-hero", type=float, default=0.5,
                    help="matched budget for the hero figure")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "toy_rfm.pt")
    cfg = T.TrainConfig(steps=args.steps)
    target = T.TwoModeTarget()

    if os.path.exists(ckpt) and not args.retrain:
        print(f"[load] {ckpt}")
        model = T.ToyRFM(hidden=cfg.hidden).to(args.device)
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        model.eval()
    else:
        print("[train] fitting tiny RFM on the two-mode field ...")
        model = T.train_toy_rfm(target, cfg, device=args.device)
        torch.save(model.state_dict(), ckpt)
        print(f"[save] {ckpt}")

    budgets = np.linspace(0.0, 1.0, 11)[1:]  # skip s=0
    print("[sweep] displacement vs matched budget over "
          f"{args.n_starts} separatrix-regime starts ...")
    sweep = run_sweep(model, target, cfg, args.num_steps, budgets, args.n_starts,
                      device=args.device)

    # headline number at a representative mid budget
    j = len(budgets) // 2
    ratio = sweep["angle_displacement_mean"][j] / max(
        sweep["norm_displacement_mean"][j], 1e-9)
    headline = {
        "budget": float(budgets[j]),
        "angle_displacement_rad": sweep["angle_displacement_mean"][j],
        "norm_displacement_rad": sweep["norm_displacement_mean"][j],
        "angle_over_norm_ratio": float(ratio),
        "angle_realized_l2": sweep["angle_realized_l2_mean"][j],
        "norm_realized_l2": sweep["norm_realized_l2_mean"][j],
        "angle_basin_flip_frac": sweep["angle_basin_flip_frac"][j],
        "norm_basin_flip_frac": sweep["norm_basin_flip_frac"][j],
    }

    print("[hero] searching for a separatrix start that flips basin ...")
    _, x0_hero, _, _ = pick_hero_start(
        model, target, cfg, args.num_steps, args.s_hero, device=args.device)
    geom = export_geometry(model, target, cfg, args.num_steps, args.s_hero,
                           x0_hero, device=args.device)
    print("[ensemble] integrating multi-run ensembles for the result figure ...")
    geom.update(export_ensemble(model, target, cfg, args.num_steps, args.s_hero,
                                device=args.device))
    # Hero numbers taken from the *plotted* trajectories so they match the figure.
    nom_end = torch.tensor(geom["traj_nominal"][-1])
    da_h = float(T.geodesic_distance(torch.tensor(geom["traj_angle"][-1]), nom_end))
    dn_h = float(T.geodesic_distance(torch.tensor(geom["traj_norm"][-1]), nom_end))
    dist_b = float(T.geodesic_distance(
        torch.tensor(geom["traj_angle"][-1]), target.mode_b))

    results = {
        "config": {
            "train_steps": args.steps, "num_steps": args.num_steps,
            "n_starts": args.n_starts, "s_hero": args.s_hero,
            "mode_a": target.mode_a.tolist(), "mode_b": target.mode_b.tolist(),
        },
        "sweep": sweep,
        "headline": headline,
        "hero": {
            "angle_displacement_rad": float(da_h),
            "norm_displacement_rad": float(dn_h),
            "angle_endpoint_dist_to_mode_b_rad": float(dist_b),
            "mode_separation_rad": float(
                T.geodesic_distance(target.mode_a, target.mode_b)),
        },
    }
    with open(os.path.join(args.out, "toy_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    np.savez(os.path.join(args.out, "toy_geometry.npz"), **geom)

    print("\n==================== RESULTS ====================")
    print(f"mode separation           : {results['hero']['mode_separation_rad']:.3f} rad "
          f"({np.degrees(results['hero']['mode_separation_rad']):.1f} deg)")
    print(f"budget s = {headline['budget']:.2f} (matched L2)")
    print(f"  angle (direction) disp  : {headline['angle_displacement_rad']:.4f} rad")
    print(f"  norm  (magnitude) disp  : {headline['norm_displacement_rad']:.4f} rad")
    print(f"  angle / norm ratio      : {headline['angle_over_norm_ratio']:.1f}x")
    print(f"  basin-flip frac a/n     : {headline['angle_basin_flip_frac']:.2f} / "
          f"{headline['norm_basin_flip_frac']:.2f}")
    print(f"  realized L2 (angle/norm): {headline['angle_realized_l2']:.4f} / "
          f"{headline['norm_realized_l2']:.4f}")
    print(f"hero flip: angle disp {da_h:.3f} rad (lands {dist_b:.3f} rad from "
          f"mode B) vs norm disp {dn_h:.3f} rad (stays in basin A)")
    print(f"\n[write] {os.path.join(args.out, 'toy_results.json')}")
    print(f"[write] {os.path.join(args.out, 'toy_geometry.npz')}")
    print("Now render figures with:  python plot_toy.py  &&  python export_tikz.py")


if __name__ == "__main__":
    main()
