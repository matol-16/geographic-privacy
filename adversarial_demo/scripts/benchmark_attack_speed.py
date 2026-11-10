"""Per-optimisation-step wall time and peak GPU memory of every attack (main-table "speed" column).

Fairness protocol (every attack gets exactly the same treatment):
  * the attacks run through the evaluation code path itself: ``run_attack`` with the
    cluster_full config (same n_steps, train_batch_size, lr, per-attack overrides, budget);
    GeoShield runs its own ``attack_imgpair`` / ``fgsm_attack_masked`` in-process, with the
    options the evaluation passes to it (336 px model-preprocessed PNGs, epsilon, steps);
  * ONE attack per fresh process on an otherwise idle GPU, so neither the CUDA caching
    allocator nor cuDNN/cuBLAS autotuning state is shared between attacks, and each process
    holds only the models its attack needs (PLONK pipeline, or GeoShield's CLIP ensemble);
  * the same seeded images for every attack (first N of the 4000-image YFCC4k pool);
    image 0 is a warm-up and is dropped (lazy CUDA init, allocator growth);
  * a step is timed from a ``cuda.synchronize()`` before it to one after it, so the queued
    kernels are counted; the peak-memory counter is reset at the start of every step;
  * only the optimisation steps are timed: the one-off per-image setup (x0 bank, clean
    embedding, target encoding) and the restart evaluation (sampling at w=10) are recorded
    separately and are identical or negligible across attacks;
  * identical backend flags for every attack (PyTorch defaults: cudnn.benchmark off,
    no TF32 matmul) -- GeoShield's own ``set_environment`` (cudnn.deterministic) is not
    applied, only its seeds;
  * two rounds, the second in reverse attack order, to expose drift (clock / thermal);
  * exclusive GPU: a worker waits until no other process is on the GPU, nvidia-smi is polled
    every 2 s during the run, and a run that overlapped with any other process is discarded
    (moved to contaminated/) and retried.

    python scripts/benchmark_attack_speed.py run --out /Data/mathias.ollu/results/attack_speed
    python scripts/benchmark_attack_speed.py summarize --out /Data/mathias.ollu/results/attack_speed

Cost curves for iso-time / iso-memory comparisons: ``--dataset osv`` and ``--variants``, a list of
attack-kwarg overrides each timed as its own configuration (``""`` = the config as is), e.g.

    python scripts/benchmark_attack_speed.py run --dataset osv --out <dir> --attacks dtd sampling \
        --variants train_batch_size=64 train_batch_size=256 train_batch_size=64,sampling_steps_salman=8

Results go to ``round<r>/<attack>__<variant>.json``. ``--restart-eval-steps`` shortens the restart
evaluation, which is not part of the timed optimisation steps.

``run`` is resumable: a (round, attack) whose JSON exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_DIR / "configs" / "cluster_full" / "config.yaml"
DEFAULT_DATA_DIRS = {
    "yfcc": "/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k",
    "osv": "/Data/mathias.ollu/hf_cache/datasets/osv5m",
}
DEFAULT_ATTACKS = [
    "geoshield", "encoder", "sampling", "training_loss", "ace", "targeted_l2",
    "unidef", "diffusion_l2", "dtd", "sampling_cfg10", "dtd_blur",
]
GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# Timing helpers
# --------------------------------------------------------------------------- #

class StepRecorder:
    """Collects (image, step, seconds, peak bytes, bytes resident at step start)."""

    def __init__(self):
        self.records = []
        self.image = -1

    def start(self):
        import torch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        return time.perf_counter(), torch.cuda.memory_allocated()

    def stop(self, t0, mem0):
        import torch
        torch.cuda.synchronize()
        self.records.append({
            "image": self.image,
            "seconds": time.perf_counter() - t0,
            "peak_bytes": torch.cuda.max_memory_allocated(),
            "resident_bytes": mem0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        })


def _patch_run_steps(recorder: StepRecorder) -> None:
    """Time every ``run_step`` of every attack class (outermost call only: subclasses may call super)."""
    import inspect
    from attacks import attacks_core, trajectory_deviation, encoder_attacks
    from attacks import diffusion_attack_salman, diffusion_attack_truncated

    depth = [0]

    def wrap(orig):
        def timed(self, *args, **kwargs):
            if depth[0] > 0:
                return orig(self, *args, **kwargs)
            depth[0] += 1
            try:
                t0, mem0 = recorder.start()
                out = orig(self, *args, **kwargs)
                recorder.stop(t0, mem0)
                return out
            finally:
                depth[0] -= 1
        return timed

    for mod in (attacks_core, trajectory_deviation, encoder_attacks, diffusion_attack_salman, diffusion_attack_truncated):
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__ and "run_step" in cls.__dict__ and not inspect.isabstract(cls):
                cls.run_step = wrap(cls.__dict__["run_step"])


def _foreign_gpu_processes() -> list:
    """Compute processes on the GPU other than this one (the benchmark must own the GPU)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    procs = [line.strip() for line in out.splitlines() if line.strip()]
    return [p for p in procs if p.split(",")[0].strip() != str(os.getpid())]


class GpuWatcher:
    """Polls nvidia-smi while the attack runs; any foreign process seen invalidates the timings.

    The GPU is shared (another user's job on it roughly doubled our step times once), so a
    result is only kept if the GPU was ours alone for its whole duration.
    """

    def __init__(self, interval_s: float = 2.0):
        import threading
        self.interval_s = interval_s
        self.seen = []
        self.samples = []  # (sm clock MHz, power W, throttle reasons) -- context for the caption
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            foreign = _foreign_gpu_processes()
            if foreign:
                self.seen.append({"t": time.strftime("%H:%M:%S"), "procs": foreign})
            try:
                q = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,clocks_throttle_reasons.active",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
                self.samples.append([x.strip() for x in q.strip().split(",")])
            except Exception:
                pass

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


def _wait_for_free_gpu(poll_s: int = 60) -> None:
    waited = 0
    while True:
        foreign = _foreign_gpu_processes()
        if not foreign:
            return
        if waited % 600 == 0:
            print(f"GPU busy with other processes {foreign}; waiting ({waited // 60} min so far)", flush=True)
        time.sleep(poll_s)
        waited += poll_s


def _gpu_info() -> dict:
    import torch
    return {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "hostname": os.uname().nodename,
        "other_gpu_processes": "; ".join(_foreign_gpu_processes()),
    }


# --------------------------------------------------------------------------- #
# Workers (one attack per process)
# --------------------------------------------------------------------------- #

def parse_variant(variant: str) -> dict:
    """``"train_batch_size=64,sampling_steps_salman=8"`` -> {key: int/float/str}; ``""`` -> {}."""
    import yaml
    out = {}
    for item in filter(None, (variant or "").split(",")):
        key, value = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(value)
    return out


def variant_tag(variant: str) -> str:
    """File-name-safe label of a variant ("" -> "default")."""
    return variant.replace("=", "").replace(",", "__") if variant else "default"


def worker_plonk(args) -> dict:
    import torch
    import yaml
    sys.path.insert(0, str(PROJECT_DIR))
    os.chdir(PROJECT_DIR)  # config paths (ACE target image) are relative to PROJECT_DIR
    from main import get_pipeline, get_attack_kwargs, get_attack_type_kwargs
    from attacks.attacks import run_attack
    from attacks.attacks_core import AttackBase
    from utils.datasets import retrieve_yfcc_images, retrieve_osv_images

    ds = args.dataset
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config["data_dirs"][ds] = args.data_dir
    torch.manual_seed(int(config.get("seed", 42)))

    retrieve = retrieve_yfcc_images if ds == "yfcc" else retrieve_osv_images
    images, gps, ids = retrieve(
        n_images_to_eval=4000, seed=int(config.get("seed", 42)), use_real_gps=True,
        local_dir=args.data_dir, window_start=0, window_end=args.n_images,
    )

    torch.cuda.reset_peak_memory_stats()
    pipeline = get_pipeline(config, ds)
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated()

    kwargs = dict(get_attack_kwargs(config, ds)[0])
    kwargs.update(get_attack_type_kwargs(config, ds).get(args.attack, {}))
    kwargs.update(num_restarts=1, main_num_restarts=1)  # step cost does not depend on the restart count
    kwargs.update(parse_variant(args.variant))
    if args.restart_eval_steps:
        kwargs["restart_eval_num_steps"] = int(args.restart_eval_steps)  # untimed; only shortens the run

    recorder = StepRecorder()
    _patch_run_steps(recorder)
    eval_seconds = []
    orig_eval = AttackBase.evaluate_restart

    def timed_eval(self, *a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        out = orig_eval(self, *a, **k)
        torch.cuda.synchronize(); eval_seconds.append(time.perf_counter() - t0)
        return out
    AttackBase.evaluate_restart = timed_eval

    per_image = []
    for i, (img, g, image_id) in enumerate(zip(images, gps, ids)):
        recorder.image = i
        n_before, n_eval_before = len(recorder.records), len(eval_seconds)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        run_attack(attack_type=args.attack, source_image=img, pipeline=pipeline,
                   eps_max=args.eps, silent=True, true_gps=tuple(g), **kwargs)
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        steps = recorder.records[n_before:]
        train = sum(r["seconds"] for r in steps)
        ev = sum(eval_seconds[n_eval_before:])
        per_image.append({"image": i, "image_id": image_id, "n_steps": len(steps), "total_s": total,
                          "train_s": train, "restart_eval_s": ev, "setup_s": total - train - ev})
        print(f"[{args.attack}] image {i}: {len(steps)} steps, {1e3 * train / max(len(steps), 1):.1f} ms/step, "
              f"total {total:.1f}s", flush=True)

    return {"weights_bytes": weights_bytes, "steps": recorder.records, "per_image": per_image,
            "attack_kwargs": {k: v for k, v in kwargs.items() if isinstance(v, (int, float, str, bool, type(None)))}}


def worker_geoshield(args) -> dict:
    import torch
    import yaml
    from omegaconf import OmegaConf

    with open(args.config) as f:
        gs = yaml.safe_load(f)["geoshield"]
    geoshield_dir = Path(args.geoshield_repo).resolve()
    os.chdir(geoshield_dir)
    sys.path.insert(0, str(geoshield_dir))  # GeoShield's own ``utils`` (clashes with ours: never import both)
    import geoshield as G
    G.wandb.init(mode="disabled")  # G.main's setup_wandb, with the WANDB_MODE=disabled it forces

    cfg = OmegaConf.load(geoshield_dir / "config" / "ensemble_3models.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([
        f"data.cle_data_path={args.clean_dir}", f"data.tgt_data_path={args.clean_dir}",
        f"data.output={args.work_dir}", f"data.num_samples={args.n_images}",
        f"optim.epsilon={int(round(args.eps * 255))}", f"optim.steps={int(gs['steps'])}",
        f"model.input_res={int(gs['input_res'])}",
    ]))
    # Seeds only (as G.set_environment) -- the cudnn flags stay at the defaults every attack uses.
    import random, numpy as np
    random.seed(2023); np.random.seed(2023); torch.manual_seed(2023); torch.cuda.manual_seed(2023)

    torch.cuda.reset_peak_memory_stats()
    ensemble_extractor, models = G.get_models(cfg)
    ensemble_loss = G.get_ensemble_loss(cfg, models)
    torch.cuda.synchronize()
    weights_bytes = torch.cuda.memory_allocated()

    # Same data pipeline as G.main.
    tfm = G.transforms.Compose([
        G.transforms.Resize(cfg.model.input_res, interpolation=G.torchvision.transforms.InterpolationMode.BICUBIC),
        G.transforms.CenterCrop(cfg.model.input_res),
        G.transforms.Lambda(lambda img: img.convert("RGB")),
        G.transforms.Lambda(lambda img: G.to_tensor(img)),
    ])
    data = G.ImageFolderWithPaths(args.clean_dir, transform=tfm)
    loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
    source_crop = G.transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_source_crop else torch.nn.Identity()
    target_crop = G.transforms.RandomResizedCrop(cfg.model.input_res, scale=cfg.model.crop_scale) if cfg.model.use_target_crop else torch.nn.Identity()

    recorder = StepRecorder()

    class TimedSteps:
        """Stands in for GeoShield's ``tqdm(range(steps))``: times each loop-body execution."""
        def __init__(self, iterable, *a, **k):
            self.iterable = iterable
        def __iter__(self):
            for x in self.iterable:
                t0, mem0 = recorder.start()
                yield x
                recorder.stop(t0, mem0)
        def set_postfix(self, *a, **k):
            pass
    G.tqdm = TimedSteps

    per_image = []
    for i, (image_org, _, path_org) in enumerate(loader):
        if i >= args.n_images:
            break
        recorder.image = i
        n_before = len(recorder.records)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        # source == target, as in the evaluation (tgt_data_path = cle_data_path).
        G.attack_imgpair(cfg=cfg, ensemble_extractor=ensemble_extractor, ensemble_loss=ensemble_loss,
                         source_crop=source_crop, target_crop=target_crop, img_index=i,
                         image_org=image_org, path_org=path_org, image_tgt=image_org.clone(),
                         bbox_dict={})
        torch.cuda.synchronize(); total = time.perf_counter() - t0
        steps = recorder.records[n_before:]
        train = sum(r["seconds"] for r in steps)
        per_image.append({"image": i, "image_id": Path(path_org[0]).stem, "n_steps": len(steps), "total_s": total,
                          "train_s": train, "restart_eval_s": 0.0, "setup_s": total - train})
        print(f"[geoshield] image {i}: {len(steps)} steps, {1e3 * train / max(len(steps), 1):.1f} ms/step, "
              f"total {total:.1f}s", flush=True)

    return {"weights_bytes": weights_bytes, "steps": recorder.records, "per_image": per_image,
            "attack_kwargs": OmegaConf.to_container(cfg, resolve=True)}


def cmd_worker(args) -> None:
    import torch
    _wait_for_free_gpu()
    info = _gpu_info()
    t0 = time.time()
    with GpuWatcher() as watcher:
        res = worker_geoshield(args) if args.attack == "geoshield" else worker_plonk(args)
    res.update(attack=args.attack, round=args.round, eps=args.eps, n_images=args.n_images,
               dataset=args.dataset, variant=args.variant, variant_kwargs=parse_variant(args.variant),
               env=info, wall_s=time.time() - t0, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
               foreign_gpu_processes_seen=watcher.seen, gpu_samples=watcher.samples)
    out = Path(args.result)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(out)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _export_clean_pngs(args, clean_dir: Path) -> None:
    """Model-preprocessed 336 px PNGs of the same images -- what the evaluation feeds GeoShield."""
    if clean_dir.exists() and len(list(clean_dir.glob("*.png"))) >= args.n_images:
        return
    clean_dir.parent.mkdir(parents=True, exist_ok=True)
    import yaml
    sys.path.insert(0, str(PROJECT_DIR))
    from utils.geoshield import _select_clean_images
    with open(args.config) as f:
        config = yaml.safe_load(f)
    gs = config["geoshield"]
    _select_clean_images("yfcc", 4000, int(config.get("seed", 42)), args.data_dir, str(clean_dir),
                         window_start=0, window_end=args.n_images, clean_image_format="png",
                         clean_preprocess=gs.get("clean_preprocess", "model"), input_res=int(gs["input_res"]))


def _result_stem(attack: str, variant: str, variants: list) -> str:
    """``<attack>`` for the plain single-config run (backward compatible), else ``<attack>__<tag>``."""
    return attack if variants == [""] else f"{attack}__{variant_tag(variant)}"


def cmd_run(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clean_root = out / "geoshield_clean" / "clean"  # flat folder, as GeoShield reads it
    if "geoshield" in args.attacks:
        _export_clean_pngs(args, clean_root)
    for rnd in range(args.rounds):
        order = args.attacks if rnd % 2 == 0 else list(reversed(args.attacks))
        for attack, variant in [(a, v) for a in order for v in args.variants]:
            result = out / f"round{rnd}" / f"{_result_stem(attack, variant, args.variants)}.json"
            if result.exists():
                print(f"skip {result} (exists)", flush=True)
                continue
            cmd = [sys.executable, __file__, "worker", "--attack", attack, "--round", str(rnd),
                   "--result", str(result), "--config", str(args.config), "--data-dir", args.data_dir,
                   "--n-images", str(args.n_images), "--eps", str(args.eps),
                   "--geoshield-repo", args.geoshield_repo, "--clean-dir", str(clean_root),
                   "--work-dir", str(out / "geoshield_work"), "--dataset", args.dataset,
                   "--variant", variant, "--restart-eval-steps", str(args.restart_eval_steps or 0)]
            for attempt in range(args.max_attempts):
                print(f"== round {rnd}: {attack} [{variant or 'default'}] (attempt {attempt + 1})", flush=True)
                subprocess.run(cmd, check=True)
                r = json.loads(result.read_text())
                if not r.get("foreign_gpu_processes_seen"):
                    break
                bad = out / "contaminated" / f"round{rnd}_{result.stem}_attempt{attempt}.json"
                bad.parent.mkdir(parents=True, exist_ok=True)
                result.replace(bad)
                print(f"   another process used the GPU during the run -> discarded ({bad}), retrying", flush=True)
            else:
                raise RuntimeError(f"{attack} round {rnd}: GPU never free for a whole run ({args.max_attempts} attempts)")
    cmd_summarize(args)


def _pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def cmd_summarize(args) -> None:
    out = Path(args.out)
    summary = {}
    for attack, variant in [(a, v) for a in args.attacks for v in args.variants]:
        key = _result_stem(attack, variant, args.variants)
        rounds = []
        for f in sorted(out.glob(f"round*/{key}.json")):
            r = json.loads(f.read_text())
            if r.get("foreign_gpu_processes_seen"):
                print(f"WARNING: skipping {f}: another process used the GPU during it")
                continue
            steps = [s for s in r["steps"] if s["image"] >= args.warmup_images]
            if not steps:
                continue
            ms = [1e3 * s["seconds"] for s in steps]
            per_img = [p for p in r["per_image"] if p["image"] >= args.warmup_images]
            rounds.append({
                "round": r["round"], "n_step_samples": len(ms),
                "median_ms": statistics.median(ms), "mean_ms": statistics.fmean(ms),
                "p10_ms": _pct(ms, 0.10), "p90_ms": _pct(ms, 0.90),
                "peak_alloc_gib": max(s["peak_bytes"] for s in steps) / GIB,
                "peak_reserved_gib": max(s["peak_reserved_bytes"] for s in steps) / GIB,
                "weights_gib": r["weights_bytes"] / GIB,
                "peak_above_weights_gib": (max(s["peak_bytes"] for s in steps) - r["weights_bytes"]) / GIB,
                "n_steps_per_restart": statistics.median(p["n_steps"] for p in per_img),
                "train_s_per_restart": statistics.fmean(p["train_s"] for p in per_img),
                "setup_s_per_image": statistics.fmean(p["setup_s"] for p in per_img),
                "restart_eval_s": statistics.fmean(p["restart_eval_s"] for p in per_img),
                "gpu": r["env"]["gpu"], "other_gpu_processes": r["env"]["other_gpu_processes"],
            })
        if not rounds:
            continue
        med = [x["median_ms"] for x in rounds]
        summary[key] = {
            "attack": attack, "variant": variant, "variant_kwargs": parse_variant(variant),
            "median_ms_per_step": statistics.median(med),
            "round_spread_pct": 100 * (max(med) - min(med)) / min(med),
            "peak_alloc_gib": max(x["peak_alloc_gib"] for x in rounds),
            "peak_above_weights_gib": max(x["peak_above_weights_gib"] for x in rounds),
            "weights_gib": rounds[0]["weights_gib"],
            "n_steps_per_restart": rounds[0]["n_steps_per_restart"],
            "rounds": rounds,
        }
    (out / "speed_summary.json").write_text(json.dumps(summary, indent=2))

    hdr = f"{'attack':44s} {'ms/step':>9s} {'p10-p90':>15s} {'spread':>7s} {'peak GiB':>9s} {'>weights':>9s} {'steps':>6s} {'s/restart':>10s} {'setup s':>8s}"
    lines = [hdr, "-" * len(hdr)]
    for attack, s in summary.items():
        r0 = s["rounds"][0]
        lines.append(f"{attack:44s} {s['median_ms_per_step']:9.1f} {r0['p10_ms']:7.1f}-{r0['p90_ms']:<7.1f} "
                     f"{s['round_spread_pct']:6.1f}% {s['peak_alloc_gib']:9.2f} {s['peak_above_weights_gib']:9.2f} "
                     f"{s['n_steps_per_restart']:6.0f} {r0['train_s_per_restart']:10.1f} {r0['setup_s_per_image']:8.2f}")
    text = "\n".join(lines)
    (out / "speed_summary.txt").write_text(text + "\n")
    print(text)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "summarize", "worker"):
        s = sub.add_parser(name)
        s.add_argument("--out", default="/Data/mathias.ollu/results/attack_speed")
        s.add_argument("--config", default=str(DEFAULT_CONFIG))
        s.add_argument("--dataset", choices=sorted(DEFAULT_DATA_DIRS), default="yfcc")
        s.add_argument("--data-dir", default=None, help="default: DEFAULT_DATA_DIRS[--dataset]")
        s.add_argument("--variants", nargs="+", default=[""],
                       help='attack-kwarg overrides, one config each ("k=v,k2=v2"; "" = config as is)')
        s.add_argument("--restart-eval-steps", type=int, default=0,
                       help="sampling steps of the (untimed) restart evaluation; 0 = config value")
        s.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
        s.add_argument("--n-images", type=int, default=11, help="images per attack, incl. the warm-up image(s)")
        s.add_argument("--warmup-images", type=int, default=1)
        s.add_argument("--rounds", type=int, default=2)
        s.add_argument("--max-attempts", type=int, default=20, help="retries of a run another GPU process overlapped with")
        s.add_argument("--eps", type=float, default=0.0314, help="l_inf budget (8/255); step cost is budget-independent")
        s.add_argument("--geoshield-repo", default=str(PROJECT_DIR.parents[1] / "Geoshield"))
        if name == "worker":
            s.add_argument("--attack", required=True)
            s.add_argument("--round", type=int, required=True)
            s.add_argument("--result", required=True)
            s.add_argument("--clean-dir", required=True)
            s.add_argument("--work-dir", required=True)
            s.add_argument("--variant", default="")
    args = p.parse_args()
    args.data_dir = args.data_dir or DEFAULT_DATA_DIRS[args.dataset]
    {"run": cmd_run, "summarize": cmd_summarize, "worker": cmd_worker}[args.cmd](args)


if __name__ == "__main__":
    main()
