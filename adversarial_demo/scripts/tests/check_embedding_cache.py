#!/usr/bin/env python
"""Check that caching the frozen embeddings changes nothing (GeoFA-ACE target, GeoFA-TD-blur clean).

Runs each attack twice on 2 OSV images from the same seed, with deterministic kernels: once
with the cached embeddings, once with them recomputed every step (the old behaviour, restored
by monkeypatching the cache helpers). Prints EQUAL <attack> <image> True/False on the deltas.

    CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/tests/check_embedding_cache.py
"""
import os
import sys

import torch
import yaml
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

from main import get_pipeline, get_attack_kwargs, get_attack_type_kwargs  # noqa: E402
from attacks.attacks import run_attack  # noqa: E402
from attacks import trajectory_deviation as T  # noqa: E402
from utils.datasets import retrieve_osv_images  # noqa: E402
from utils.adversarial_utils import seed_everything, compute_embedding  # noqa: E402


def main() -> None:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    cfg = yaml.safe_load(open("configs/cluster_full/config.yaml"))
    cfg["data_dirs"]["osv"] = "/Data/mathias.ollu/hf_cache/datasets/osv5m"
    imgs, gps, _ = retrieve_osv_images(4000, 42, True, cfg["data_dirs"]["osv"], 0, 2)
    pipe = get_pipeline(cfg, "osv")

    def recompute_clean(self):
        return compute_embedding(self.source_tensor, self.train_batch_size, self.pipeline,
                                 device=self.device, track_grad=False)

    def recompute_target(self):
        return compute_embedding(self.target_tensor, self.train_batch_size, self.pipeline,
                                 device=self.device, track_grad=False)

    cached = (T.DiffusionAttack._clean_embedding_batch, T.ACE._target_embedding_batch)
    deltas = {}
    for mode in ("cached", "recomputed"):
        T.DiffusionAttack._clean_embedding_batch = cached[0] if mode == "cached" else recompute_clean
        T.ACE._target_embedding_batch = cached[1] if mode == "cached" else recompute_target
        for attack in ("ace", "dtd_blur"):
            kw = dict(get_attack_kwargs(cfg, "osv")[0])
            kw.update(get_attack_type_kwargs(cfg, "osv").get(attack, {}))
            kw.update(n_steps=5, num_restarts=1, restart_eval_num_steps=8)
            if attack == "ace":
                kw["target_image"] = Image.open(os.path.join(PROJECT_DIR, "..", ".media", "MIST.png")).convert("RGB")
            for i in range(2):
                seed_everything(1234 + i)
                r = run_attack(attack_type=attack, source_image=imgs[i], pipeline=pipe, eps_max=0.0039,
                               silent=True, true_gps=tuple(gps[i]), **kw)
                deltas[(mode, attack, i)] = r["delta"].detach().cpu()
    for attack in ("ace", "dtd_blur"):
        for i in range(2):
            print("EQUAL", attack, i, torch.equal(deltas[("cached", attack, i)], deltas[("recomputed", attack, i)]))


if __name__ == "__main__":
    main()
