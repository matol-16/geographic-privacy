"""1409_FIX: re-evaluate SAVED perturbations offline (no retraining).

Each 1409_FIX_ablations run stores every best delta under ``<shard>/deltas/`` (config
``save_deltas``), so any evaluation-time question can be answered by rebuilding the protected
image and running the pipeline again. Two modes:

``schedulers``  vary the INFERENCE schedule / ODE integrator only (the attacked model, the
                perturbation, the step count (250), the guidance (10) and the shared initial
                noise are all held fixed). This is the test of "is a sampler-unrolled attack
                tied to the sampler it was unrolled through?".
``steps``       the shipped sampler at 2..16 steps (cfg 10): is the attack tied to a fine discretisation?
``transfer``    the protected image fed to the other PLONK_YFCC heads (diffusion, flow) and the attacked
                rfm model (cfg 10, 250 steps) -- cross-model transfer, GT-based
``transforms``  degrade the protected image (JPEG quality / Gaussian blur) and re-evaluate it
                against the untransformed clean image, as in GeoShield Fig. 6.

Every metric is ground-truth based: the raw (128, 2) clean / attacked GPS sample batches of
each evaluation are stored, and the report scripts reduce them against the true location.

Usage: offline_eval.py --mode schedulers --attacks sampling_v2 dtd_v3_old_forward
Output: <out-dir>/<mode>_<attack>.pt = {image_id: {"true_gps": (lat, lon),
                                                   variant: {"gps_source", "gps_perturbed"}}}
Resumable: images already present in the output file are skipped.
"""
import argparse, glob, os, random, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plonk.pipe import scheduler_fn
from plonk.utils.manifolds import Sphere
from utils.ablations import apply_gaussian_blur, apply_jpeg_compression
from utils.adversarial_utils import add_perturbation_to_image, make_shared_initial_noise
from utils.datasets import retrieve_yfcc_images

DELTAS_ROOT = "/Data/mathias.ollu/results/1409_FIX_ablations/shards"
OUT_DIR = "/Data/mathias.ollu/results/1409_FIX_offline"
CFG, NUM_STEPS, BATCH, SEED = 10.0, 250, 128, 42
JPEG_QUALITIES = [20, 40, 60, 80, 100]
BLUR_SIGMAS = [0, 2, 4, 6, 8, 10]
LOW_STEPS = [2, 3, 4, 5, 6, 8, 10, 12, 16]  # --mode steps: coarse samplers (shipped sigmoid/Euler, cfg 10)


def make_sampler(kind: str, churn: float = 0.3):
    """A drop-in replacement for ``riemannian_flow_sampler`` with a different integrator.

    ``euler``      the shipped step: x <- projx(x + dt * v)  (reimplemented, for control)
    ``expmap``     the exact Riemannian step: x <- exp_x(dt * proj_tangent(v))
    ``heun``       2nd-order (predictor/corrector), i.e. 2 network evaluations per step
    ``stochastic`` Euler + tangent Gaussian noise of scale ``churn * sqrt(dt) * (1 - gamma)``
                   (vanishes as the sample lands, so the SDE is a perturbed version of the ODE)
    """
    manifold = Sphere()

    def sampler(net, batch, manifold=manifold, conditioning_keys=None, scheduler=None,
                num_steps=250, cfg_rate=0, generator=None, return_trajectories=False):
        x_cur = batch["y"].to(torch.float32)
        if kind == "expmap":
            # PLONK's initial noise is an unnormalised randn, so the shipped sampler starts off
            # the sphere (its first projx fixes that). expmap is only defined on the manifold.
            x_cur = manifold.projx(x_cur)
        traj = [x_cur.detach()] if return_trajectories else None
        step_indices = torch.arange(num_steps + 1, dtype=torch.float32, device=x_cur.device)
        gammas = scheduler(1 - step_indices / num_steps)
        stacked = {}
        use_cfg = cfg_rate > 0 and conditioning_keys is not None
        if use_cfg:
            stacked[conditioning_keys] = torch.cat(
                [batch[conditioning_keys], torch.zeros_like(batch[conditioning_keys])], dim=0
            )

        def velocity(x, gamma):
            if use_cfg:
                stacked["y"] = torch.cat([x, x], dim=0)
                stacked["gamma"] = gamma.expand(x.shape[0] * 2)
                cond, uncond = net(stacked).chunk(2, dim=0)
                return cond * (1 + cfg_rate) - uncond * cfg_rate
            batch["y"] = x
            batch["gamma"] = gamma.expand(x.shape[0])
            return net(batch)

        for gamma_now, gamma_next in zip(gammas[:-1], gammas[1:]):
            dt = gamma_next - gamma_now
            v = velocity(x_cur, gamma_now)
            if kind == "expmap":
                x_next = manifold.expmap(x_cur, dt * manifold.proju(x_cur, v))
            elif kind == "heun":
                v2 = velocity(manifold.projx(x_cur + dt * v), gamma_next)
                x_next = manifold.projx(x_cur + dt * 0.5 * (v + v2))
            else:
                x_next = manifold.projx(x_cur + dt * v)
                if kind == "stochastic":
                    noise = manifold.proju(x_next, torch.randn_like(x_next))
                    x_next = manifold.projx(x_next + churn * dt.abs().sqrt() * (1 - gamma_next) * noise)
            x_cur = x_next
            if return_trajectories:
                traj.append(x_cur.detach().to(torch.float32))
        return (x_cur.to(torch.float32), traj) if return_trajectories else x_cur.to(torch.float32)

    return sampler


def scheduler_variants(pipeline, churn: float = 3.0, shipped=None, sigmoid=None):
    """[(label, scheduler, sampler, num_steps)] — the inference-time knobs, model and delta fixed.

    The baseline is what every other 1409_FIX number uses: PLONK's sigmoid schedule
    (start -7, end 3, tau 1) integrated by the shipped projected-Euler sampler at 250 steps.
    The 8-step block repeats the schedule/integrator comparison on a COARSE sampler, where the
    discretisation actually changes the trajectory (at 250 steps every integrator has converged).
    """
    shipped = shipped if shipped is not None else pipeline.sampler
    sigmoid = sigmoid if sigmoid is not None else pipeline.scheduler
    linear, cosine = scheduler_fn("linear", 1, 0, 1.0), scheduler_fn("cosine", 1, 0, 1.0)
    heun, expmap = make_sampler("heun"), make_sampler("expmap")
    sde = make_sampler("stochastic", churn=churn)
    return [
        ("sigmoid + Euler, 250 steps (baseline)", sigmoid, shipped, 250),
        ("linear + Euler, 250 steps", linear, shipped, 250),
        ("cosine + Euler, 250 steps", cosine, shipped, 250),
        ("sigmoid + exp. map, 250 steps", sigmoid, expmap, 250),
        ("sigmoid + Heun, 250 steps", sigmoid, heun, 250),
        (f"sigmoid + stochastic (churn {churn:g}), 250 steps", sigmoid, sde, 250),
        ("sigmoid + Euler, 8 steps", sigmoid, shipped, 8),
        ("linear + Euler, 8 steps", linear, shipped, 8),
        ("cosine + Euler, 8 steps", cosine, shipped, 8),
        ("sigmoid + Heun, 8 steps", sigmoid, heun, 8),
        (f"sigmoid + stochastic (churn {churn:g}), 8 steps", sigmoid, sde, 8),
    ]


def predict(pipeline, image, x_n, seed=SEED, num_steps=NUM_STEPS):
    """One pipeline run on ``image`` with the shared initial noise; returns (128, 2) GPS degrees."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        gps, _ = pipeline(image, batch_size=BATCH, cfg=CFG, x_N=x_n, num_steps=int(num_steps),
                          return_trajectories=True)
    return torch.as_tensor(np.asarray(gps), dtype=torch.float32).clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["schedulers", "transforms", "transforms_clean", "steps", "transfer"], required=True)
    ap.add_argument("--steps", nargs="+", type=int, default=None, help="step counts for --mode steps (default LOW_STEPS)")
    ap.add_argument("--attacks", nargs="*", default=[],
                    help="attack keys (delta sources); ignored by transforms_clean, which is attack-independent")
    ap.add_argument("--deltas-root", default=DELTAS_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images (pilot runs)")
    ap.add_argument("--churn", type=float, default=3.0, help="noise scale of the stochastic sampler")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from main import get_pipeline_for_model_type
    import yaml
    config = yaml.safe_load(open("configs/1409_FIX/config.yaml"))
    pipeline = get_pipeline_for_model_type(config, "yfcc", "")
    images, gps, ids = retrieve_yfcc_images(
        n_images_to_eval=4000, seed=42, use_real_gps=True,
        local_dir=config["data_dirs"]["yfcc"], window_start=0, window_end=100,
    )
    by_id = {str(i): (img, g) for i, img, g in zip(ids, images, gps)}
    os.makedirs(args.out_dir, exist_ok=True)
    x_n = make_shared_initial_noise(batch_size=BATCH, device="cuda", seed=SEED)
    transfer = {}
    if args.mode == "transfer":
        # The three YFCC variants share the DINOv2 encoder and differ in the generative head.
        transfer = {"rfm": pipeline}
        for mt in ("diffusion", "flow"):
            transfer[mt] = get_pipeline_for_model_type(config, "yfcc", mt)
    steps_list = args.steps or LOW_STEPS
    shipped_sampler, sigmoid_scheduler = pipeline.sampler, pipeline.scheduler
    variants = scheduler_variants(pipeline, churn=args.churn, shipped=shipped_sampler, sigmoid=sigmoid_scheduler)

    if args.mode == "transforms_clean":
        # Control for the transforms mode: the CLEAN image degraded the same way (no delta), so a
        # rise of the attacked-vs-GT displacement under strong blur / compression can be told
        # apart from the transform itself destroying the model's accuracy.
        out_path = os.path.join(args.out_dir, "transforms_clean.pt")
        store = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
        todo = [i for i in ids if str(i) not in store][: args.limit]
        print(f"[transforms_clean] {len(ids)} images, {len(store)} already done", flush=True)
        for n, image_id in enumerate(map(str, todo), 1):
            image, true_gps = by_id[image_id]
            t0 = time.time()
            entry = {"true_gps": tuple(float(v) for v in true_gps), "clean": {"gps_source": predict(pipeline, image, x_n)}}
            for q in JPEG_QUALITIES:
                entry[f"jpeg q{q}"] = {"gps_source": predict(pipeline, apply_jpeg_compression(image, int(q)), x_n)}
            for sg in BLUR_SIGMAS:
                entry[f"blur s{sg}"] = {"gps_source": predict(pipeline, apply_gaussian_blur(image, float(sg)), x_n)}
            store[image_id] = entry
            torch.save(store, out_path)
            print(f"  clean image={n}/{len(todo)} id={image_id} ({time.time() - t0:.0f}s)", flush=True)
        print("OFFLINE_EVAL_DONE", flush=True)
        return

    for attack in args.attacks:
        out_path = os.path.join(args.out_dir, f"{args.mode}_{attack}.pt")
        store = torch.load(out_path, weights_only=False) if os.path.exists(out_path) else {}
        files = sorted(glob.glob(f"{args.deltas_root}/{attack}__w*/deltas/*.pt"))
        if args.limit:
            files = files[: args.limit]
        print(f"[{args.mode}] {attack}: {len(files)} deltas, {len(store)} already done", flush=True)
        for n, f in enumerate(files, 1):
            rec = torch.load(f, map_location="cpu", weights_only=False)
            image_id = str(rec["image_id"])
            if image_id in store:
                continue
            image, true_gps = by_id[image_id]
            attacked = add_perturbation_to_image(image, rec["delta"].to("cuda"), pipeline)
            t0 = time.time()
            entry = {"true_gps": tuple(float(v) for v in true_gps), "eps": float(rec["eps"])}
            if args.mode == "schedulers":
                for label, scheduler, sampler, steps in variants:
                    pipeline.scheduler, pipeline.sampler = scheduler, sampler
                    entry[label] = {"gps_source": predict(pipeline, image, x_n, num_steps=steps),
                                    "gps_perturbed": predict(pipeline, attacked, x_n, num_steps=steps)}
                pipeline.scheduler, pipeline.sampler = sigmoid_scheduler, shipped_sampler
            elif args.mode == "transfer":
                for label, pipe in transfer.items():
                    entry[label] = {"gps_source": predict(pipe, image, x_n),
                                    "gps_perturbed": predict(pipe, attacked, x_n)}
            elif args.mode == "steps":
                for k in steps_list:
                    entry[f"{k} steps"] = {"gps_source": predict(pipeline, image, x_n, num_steps=k),
                                           "gps_perturbed": predict(pipeline, attacked, x_n, num_steps=k)}
            else:
                entry["clean"] = {"gps_source": predict(pipeline, image, x_n),
                                  "gps_perturbed": predict(pipeline, attacked, x_n)}
                for q in JPEG_QUALITIES:
                    entry[f"jpeg q{q}"] = {"gps_perturbed": predict(pipeline, apply_jpeg_compression(attacked, int(q)), x_n)}
                for s in BLUR_SIGMAS:
                    entry[f"blur s{s}"] = {"gps_perturbed": predict(pipeline, apply_gaussian_blur(attacked, float(s)), x_n)}
            store[image_id] = entry
            torch.save(store, out_path)
            print(f"  {attack} image={n}/{len(files)} id={image_id} ({time.time() - t0:.0f}s)", flush=True)
    print("OFFLINE_EVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
