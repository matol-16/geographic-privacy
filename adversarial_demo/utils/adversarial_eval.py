from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np
import torch

from utils.adversarial_metrics import evaluate_displacement_metrics, select_displacement_score
from utils.plots_adversarial_attacks import plot_results, plot_transferability_results, plot_localizability_results, plot_attack_success_rate

####################### Methods to evaluate a an attack across a dataset of source and perturbed images.

#We evaluate first on OSV-5M's test set. We may also evaluate on YFCC4k

from huggingface_hub import hf_hub_download
import os
import csv
import random
import zipfile
from PIL import Image
import tqdm as tqdm_module

import matplotlib.pyplot as plt

from utils.pipe_trajectory import PlonkPipelineTrajectory

from utils.adversarial_utils import (
    add_perturbation_to_image,
    expand_per_budget_kwargs,
    resolve_torch_device,
    run_paired_pipeline_with_shared_noise,
    seed_everything,
)



def load_osv5m_test(local_dir: str = "/Data/mathias.ollu/hf_cache/datasets/osv5m"):
    #only download if the data is not already present
    if os.path.exists(os.path.join(local_dir, "images", "test")) and \
       any(os.path.isdir(os.path.join(local_dir, "images", "test", d)) for d in os.listdir(os.path.join(local_dir, "images", "test"))):
        return
    for i in range(5):
        hf_hub_download(repo_id="osv5m/osv5m", filename=str(i).zfill(2)+'.zip', subfolder="images/test", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="README.md", repo_type='dataset', local_dir=local_dir)
    hf_hub_download(repo_id="osv5m/osv5m", filename="test.csv", repo_type='dataset', local_dir=local_dir)
    # extract zip files
    img_dir = os.path.join(local_dir, "images", "test")
    for f in os.listdir(img_dir):
        if f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(img_dir, f), 'r') as z:
                z.extractall(img_dir)
    return

def retrieve_yfcc_images(
    n_images_to_eval: int = 100,
    seed: int = 0,
    use_real_gps: bool = False,
    local_dir: Optional[str] = None,
    im_idx=None, # If specified, retrieves only the image with this index in the dataset (after sorting by ID). Useful for debugging with a single image.
):
    if local_dir is None:
        local_dir = "/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k"
    info_path = os.path.join(local_dir, "info.txt")
    img_dir = os.path.join(local_dir, "images")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"YFCC4k info.txt not found at {info_path}. Run build_yfcc4k_from_revisiting_im2gps.py first.")

    if im_idx is not None:
        #load image corresponding to this index (im_idx.jpg)
        img_path = os.path.join(img_dir, f"{str(im_idx)}.jpg")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image with index {im_idx} not found at {img_path}. Check if im_idx is correct and if images are properly stored.")
        img = Image.open(img_path).convert("RGB")
        #retrieve metadata for this image from info.txt
        with open(info_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                photo_id = parts[0]
                if photo_id == str(im_idx):
                    lon = float(parts[1])
                    lat = float(parts[2])
                    gps = (lat, lon)
                    return [img], [gps], [photo_id]
        raise ValueError(f"Metadata for image with index {im_idx} not found in info.txt. Check if im_idx is correct and if info.txt is properly formatted.")

    rows = []
    with open(info_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            photo_id = parts[0]
            lon = float(parts[1])
            lat = float(parts[2])
            img_path = os.path.join(img_dir, f"{photo_id}.jpg")
            if os.path.exists(img_path):
                rows.append({"id": photo_id, "path": img_path, "latitude": lat, "longitude": lon})

    # Keep selection prefix-stable across different n_images_to_eval values:
    # with a fixed seed, first N images of a larger run match a smaller run.
    rows = sorted(rows, key=lambda r: str(r["id"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    samples = rows[: min(n_images_to_eval, len(rows))]

    source_images = [Image.open(s["path"]).convert("RGB") for s in samples]
    source_gps = [(s["latitude"], s["longitude"]) for s in samples]
    source_image_ids = [s["id"] for s in samples]
    print(f"Loaded {len(source_images)} images from YFCC4k.")
    return source_images, source_gps, source_image_ids


def retrieve_osv_images(
    n_images_to_eval: int = 100,
    seed: int = 0,
    use_real_gps: bool = False,
    local_dir: Optional[str] = None,
):
    if local_dir is None:
        local_dir = "/Data/mathias.ollu/hf_cache/datasets/osv5m"
    load_osv5m_test(local_dir=local_dir)  # download & extract if needed

    # Load test metadata from CSV
    csv_path = os.path.join(local_dir, "test.csv")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    img_dir = os.path.join(local_dir, "images", "test")
    subdirs = sorted(d for d in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, d)))
    # Build a lookup: image_id -> file path
    id_to_path = {}
    for sd in subdirs:
        sd_path = os.path.join(img_dir, sd)
        for fname in os.listdir(sd_path):
            img_id = os.path.splitext(fname)[0]
            id_to_path[img_id] = os.path.join(sd_path, fname)

    # Keep only rows whose image exists on disk
    rows = [r for r in rows if r["id"] in id_to_path]

    # Keep selection prefix-stable across different n_images_to_eval values.
    rows = sorted(rows, key=lambda r: str(r["id"]))
    rng = random.Random(seed)
    rng.shuffle(rows)
    samples = rows[: min(n_images_to_eval, len(rows))]

    source_images = [Image.open(id_to_path[s["id"]]).convert("RGB") for s in samples]
    source_gps = [(float(s["latitude"]), float(s["longitude"])) for s in samples]
    source_image_ids = [s["id"] for s in samples]
    print(f"Loaded {len(source_images)} images from OSV-5M test set.")
    return source_images, source_gps, source_image_ids


def evaluate_attack_on_dataset(
    attack_types,
    pipeline,
    dataset_name,
    source_image=None, 
    seed: int = 0,
    use_real_gps: bool = False,
    n_images_to_eval: int = 100,
    plot_dir: Optional[str] = "/plots",
    results_dir: Optional[str] = "/results",
    attack_budgets: list[float] = [2/255, 15/255, 50/255],
    stored_metrics = ["final_step_displacement_predicted", "final_step_displacement_true"],
    attack_kwargs: list[Dict[str, Any]] = [{}],
    parallel_workers: int = 1,
    use_cuda_streams: bool = True,
    dataset_roots: Optional[Dict[str, str]] = None,
    plot_success_rate: bool = False,
    plot_success_rate_thresholds: Optional[list[float]] = None,
    plot_gps_true: bool = False,
    config_dump: Optional[Dict[str, Any]] = None,
):
    """
        Evaluate one or more attacks on images from a test dataset.

        Args:
            attack_types: A single attack name (str) or a list of attack names, e.g. ["encoder", "diffusion"].
            pipeline: Plonk pipeline.
            dataset_name: Name of the dataset to evaluate on. "osv" or "yfcc".
            use_real_gps: Whether use real gps coords from dataset as source trajectory, instead of the clean predicted one for evaluation.
            parallel_workers: Number of concurrent evaluations to run. Values > 1 enable parallel execution.
            use_cuda_streams: If True and running on CUDA, each worker uses its own CUDA stream.
            **kwargs: forwarded to the corresponding attack function.
    """
    from core import EvaluationConfig, EvaluationRunner, run_evaluation

    seed_everything(seed)

    dataset_roots = dataset_roots or {}
    results_dir = results_dir or "/results"
    plot_dir = plot_dir or "/plots"
    
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    
    # Handle custom source image
    if source_image is not None:
        # For single source image, use a custom path
        raise NotImplementedError("Custom source image not yet supported in refactored code")
    
    # Normalize attack_kwargs
    attack_kwargs = expand_per_budget_kwargs(attack_kwargs, len(attack_budgets))
    
    # Create configuration
    config = EvaluationConfig(
        dataset=dataset_name,
        seed=seed,
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        n_images=n_images_to_eval,
        results_dir=results_dir,
        plots_dir=plot_dir,
        stored_metrics=stored_metrics,
        parallel_workers=parallel_workers,
        use_cuda_streams=use_cuda_streams,
        use_real_gps=use_real_gps,
        dataset_roots=dataset_roots,
    )
    
    # Run evaluation
    runner = EvaluationRunner(config, pipeline)

    if config_dump is not None:
        runner.save_run_config(config_dump)
    run_evaluation(runner)
    runner.save_results()
    
    # Plot results
    all_results = runner.metrics_collector.get_results()
    plot_results(
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        plot_dir=plot_dir,
        dataset_name=dataset_name,
        attack_types=attack_types,
        all_results=all_results,
        stored_metrics=stored_metrics,
    )

    if plot_success_rate:
        plot_attack_success_rate(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plot_dir,
            dataset_name=dataset_name,
            attack_types=attack_types,
            all_results=all_results,
            threshold_km=plot_success_rate_thresholds or [2500],
            gps_true=plot_gps_true,
        )


def evaluate_localizability(
    attack_types,
    pipeline,
    dataset_name,
    seed: int = 0,
    n_images_to_eval: int = 100,
    plot_dir: Optional[str] = "/plots",
    results_dir: Optional[str] = "/results",
    attack_budgets: list[float] = [2/255, 15/255, 50/255],
    attack_kwargs: list[Dict[str, Any]] = [{}],
    dataset_roots: Optional[Dict[str, str]] = None,
    config_dump: Optional[Dict[str, Any]] = None,
):
    """ 
    Evaluates how strong an attack is depending on the localizability of the source image.
    
    Localizability is computed using the clean image. Attack strength is measured via final step displacement. We can bucket images into low/med/high localizability and plot the average attack strength in each bucket, for different attack budgets and attack types.
    
    This evaluation is done for each attack type and attack budget, to see if some attacks are more effective on low-localizability images than others, and if this trend is stronger for higher attack budgets.
    """
    from core import EvaluationConfig, EvaluationRunner, sequential_evaluate_attacks, ImageLoader

    seed_everything(seed)

    dataset_roots = dataset_roots or {}
    
    # Load images
    print(f"Loading {n_images_to_eval} images from {dataset_name} dataset...")
    source_images, source_gps, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )
    
    # Normalize attack_kwargs
    attack_kwargs = expand_per_budget_kwargs(attack_kwargs, len(attack_budgets))
    
    # Compute localizability for all images
    print("Computing localizability scores...")
    localizability = torch.zeros(len(source_images))
    pbar = tqdm_module.tqdm(total=len(source_images), desc="Computing localizability")
    for i, img in enumerate(source_images):
        localizability[i] = pipeline.compute_localizability(img, number_monte_carlo_samples=256).item()
        pbar.update(1)
    pbar.close()
    
    # Run evaluation using standard config
    results_dir = results_dir or "/results"
    plot_dir = plot_dir or "/plots"
    config = EvaluationConfig(
        dataset=dataset_name,
        seed=seed,
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        n_images=n_images_to_eval,
        results_dir=results_dir,
        plots_dir=plot_dir,
        stored_metrics=["final_step_displacement_predicted"],
        parallel_workers=1,  # Use sequential for localizability
        use_cuda_streams=False,
        dataset_roots=dataset_roots,
        state_suffix="_localizability",
    )
    
    runner = EvaluationRunner(config, pipeline)
    if config_dump is not None:
        runner.save_run_config(config_dump, suffix="_localizability")
    sequential_evaluate_attacks(runner)
    
    # Combine results with localizability
    results = {
        "attack_results": runner.metrics_collector.get_results(),
        "localizability": localizability,
    }
    
    # Save results
    runner.results_manager.save_metrics(results, dataset_name, suffix="_localizability")
    
    

def merge_sampling_steps_results(json_paths):
    """
    Load and merge multiple sampling-steps JSON result files for joint plotting.

    All files must share the same eval_num_steps and success_rate_thresholds_km.
    Attack types and their per-type budget lists are combined so that
    plot_sampling_steps_success_rate can overlay lines from different commands
    (e.g. encoder trained on-the-fly vs GeoShield precomputed pairs).

    Args:
        json_paths: iterable of file paths to JSON result files produced by
                    evaluate_sampling_steps or evaluate_sampling_steps_precomputed.

    Returns:
        Merged json_results dict compatible with plot_sampling_steps_success_rate.
    """
    import json

    results_list = []
    for path in json_paths:
        with open(path) as f:
            results_list.append(json.load(f))

    if not results_list:
        raise ValueError("No result files provided to merge")

    ref = results_list[0]
    for other in results_list[1:]:
        if other["eval_num_steps"] != ref["eval_num_steps"]:
            raise ValueError(
                f"Cannot merge: eval_num_steps differ ({ref['eval_num_steps']} vs {other['eval_num_steps']})"
            )
        if other["success_rate_thresholds_km"] != ref["success_rate_thresholds_km"]:
            raise ValueError(
                f"Cannot merge: success_rate_thresholds_km differ "
                f"({ref['success_rate_thresholds_km']} vs {other['success_rate_thresholds_km']})"
            )

    merged_attack_types = []
    merged_results = {}
    budgets_per_type = {}

    for r in results_list:
        for attack_type in r["attack_types"]:
            if attack_type in merged_results:
                raise ValueError(
                    f"Duplicate attack type '{attack_type}' across result files. "
                    "Rename one of them with a different --attack-name or --attack-types value."
                )
            merged_attack_types.append(attack_type)
            merged_results[attack_type] = r["results"][attack_type]
            budgets_per_type[attack_type] = list(r["attack_budgets"])

    return {
        "dataset": ref["dataset"],
        "attack_types": merged_attack_types,
        "attack_budgets": ref["attack_budgets"],  # fallback for single-file usage
        "attack_budgets_per_type": budgets_per_type,
        "eval_num_steps": ref["eval_num_steps"],
        "success_rate_thresholds_km": ref["success_rate_thresholds_km"],
        "n_images": ref["n_images"],
        "results": merged_results,
    }


def evaluate_sampling_steps_precomputed(
    attack_name: str,
    pipeline,
    dataset_name: str,
    clean_image_dirs: list,
    attacked_image_dirs: list,
    attack_budgets: list,
    seed: int = 0,
    eval_num_steps: list = (10, 25, 50, 100, 250),
    n_images: Optional[int] = None,
    results_dir: str = "./results",
    batch_size: int = 256,
    cfg: float = 10.0,
    success_rate_thresholds: list = (200, 750, 2500),
    device: str = "cuda",
    config_dump: dict = None,
):
    """
    Evaluate how attack success of a precomputed attack (e.g. GeoShield) varies
    with the number of sampling steps used at evaluation time.

    Loads clean/attacked image pairs from folders (one pair of dirs per budget),
    runs run_paired_pipeline_with_shared_noise at each num_steps value, and
    returns a JSON-serialisable results dict in the same format as
    evaluate_sampling_steps (so plot_sampling_steps_success_rate can be reused).
    """
    import json
    import numpy as np
    from PIL import Image as PILImage
    from utils.adversarial_utils import collect_common_image_pairs, run_paired_pipeline_with_shared_noise

    seed_everything(seed)
    eval_num_steps = list(eval_num_steps)

    if len(clean_image_dirs) != len(attack_budgets) or len(attacked_image_dirs) != len(attack_budgets):
        raise ValueError(
            "clean_image_dirs, attacked_image_dirs, and attack_budgets must have the same length"
        )

    # Load and match pairs per budget; optionally truncate
    pairs_by_budget = []
    for budget_idx, (clean_dir, attacked_dir) in enumerate(zip(clean_image_dirs, attacked_image_dirs)):
        pairs = collect_common_image_pairs(clean_dir, attacked_dir)
        if n_images is not None:
            pairs = pairs[: max(0, int(n_images))]
        if not pairs:
            raise ValueError(f"No matched image pairs found for budget index {budget_idx}")
        pairs_by_budget.append(pairs)

    n_images_actual = min(len(p) for p in pairs_by_budget)

    # Evaluate each pair at every num_steps value
    # raw[budget_idx][num_steps] = [displacement_km, ...]
    raw = {bi: {ns: [] for ns in eval_num_steps} for bi in range(len(attack_budgets))}

    for budget_idx, budget in enumerate(attack_budgets):
        pairs = pairs_by_budget[budget_idx]
        for num_steps in tqdm_module.tqdm(
            eval_num_steps,
            desc=f"  {attack_name} eps={budget:.4f} eval steps",
        ):
            for image_idx, (clean_path, attacked_path, _) in enumerate(pairs):
                with PILImage.open(clean_path) as f:
                    clean_image = f.convert("RGB")
                with PILImage.open(attacked_path) as f:
                    attacked_image = f.convert("RGB")
                eval_result = run_paired_pipeline_with_shared_noise(
                    pipeline=pipeline,
                    source_image=clean_image,
                    perturbed_image=attacked_image,
                    batch_size=batch_size,
                    cfg=cfg,
                    num_steps=int(num_steps),
                    seed=int(seed) + budget_idx * 100_000 + image_idx,
                    device=device,
                )
                raw[budget_idx][num_steps].append(
                    float(eval_result["metrics"]["final_step_displacement"])
                )

    # Build JSON in the same format as evaluate_sampling_steps
    json_results = {
        "dataset": dataset_name,
        "attack_types": [attack_name],
        "attack_budgets": list(attack_budgets),
        "eval_num_steps": eval_num_steps,
        "success_rate_thresholds_km": list(success_rate_thresholds),
        "n_images": n_images_actual,
        "results": {attack_name: {}},
    }
    for budget_idx, budget in enumerate(attack_budgets):
        bkey = f"budget_{budget:.6f}"
        json_results["results"][attack_name][bkey] = {}
        for num_steps in eval_num_steps:
            disps = raw[budget_idx][num_steps]
            success_rates = {
                str(thr): float(np.mean([d > thr for d in disps]))
                for thr in success_rate_thresholds
            }
            json_results["results"][attack_name][bkey][str(num_steps)] = {
                "mean_displacement_km": float(np.mean(disps)),
                "success_rates": success_rates,
            }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{dataset_name}_{attack_name}_sampling_steps_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved sampling-steps results to: {json_path}")

    return json_results


def evaluate_restarts(
    attack_types,
    pipeline,
    dataset_name: str,
    seed: int = 0,
    n_images_to_eval: int = 1,
    max_restarts: int = 10,
    results_dir: str = "./results",
    attack_budgets: list = (2/255, 15/255, 50/255),
    attack_kwargs: list = (),
    dataset_roots: dict = None,
    config_dump: dict = None,
):
    """
    Train attacks with max_restarts restarts and measure how the best displacement
    found evolves as the restart count increases.

    For each (attack_type, budget, image), runs the attack once with max_restarts
    restarts and extracts per-restart displacements from restart_summaries.
    Returns a JSON-serialisable results dict (also saved to results_dir).
    """
    import json
    import numpy as np
    from core import ImageLoader
    from attacks.attacks import run_attack
    from utils.adversarial_utils import expand_per_budget_kwargs

    seed_everything(seed)
    dataset_roots = dataset_roots or {}

    if isinstance(attack_types, str):
        attack_types = [attack_types]

    source_images, source_gps, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))

    # Train each attack with max_restarts; record per-restart displacement from summaries.
    # raw[attack_type][budget_idx][image_idx] = [disp_restart_0, disp_restart_1, ...]
    raw = {
        at: {bi: {} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            print(f"Training {attack_type} attacks with {max_restarts} restarts (eps={budget:.4f})...")
            for image_idx, image in enumerate(tqdm_module.tqdm(source_images, desc="  images")):
                kw = dict(attack_kwargs[budget_idx])
                kw["num_restarts"] = max_restarts
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=pipeline,
                    eps_max=budget,
                    silent=True,
                    **kw,
                )
                raw[attack_type][budget_idx][image_idx] = [
                    float(s["final_step_displacement"])
                    for s in result.get("restart_summaries", [])
                ]

    def _best_after_k(displacements):
        best, current = [], float("-inf")
        for d in displacements:
            current = max(current, d)
            best.append(current)
        return best

    json_results = {
        "dataset": dataset_name,
        "attack_types": attack_types,
        "attack_budgets": list(attack_budgets),
        "max_restarts": max_restarts,
        "n_images": n_images_to_eval,
        "image_ids": list(source_image_ids),
        "results": {},
    }
    for attack_type in attack_types:
        json_results["results"][attack_type] = {}
        for budget_idx, budget in enumerate(attack_budgets):
            bkey = f"budget_{budget:.6f}"
            json_results["results"][attack_type][bkey] = {}
            for image_idx in range(n_images_to_eval):
                disps = raw[attack_type][budget_idx][image_idx]
                json_results["results"][attack_type][bkey][f"image_{image_idx}"] = {
                    "image_id": source_image_ids[image_idx],
                    "restart_displacements": disps,
                    "best_after_k": _best_after_k(disps),
                }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{dataset_name}_restarts_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved restart ablation results to: {json_path}")

    return json_results


def evaluate_sampling_steps(
    attack_types,
    pipeline,
    dataset_name: str,
    seed: int = 0,
    n_images_to_eval: int = 20,
    eval_num_steps: list = (10, 25, 50, 100, 250),
    results_dir: str = "./results",
    attack_budgets: list = (2/255, 15/255, 50/255),
    attack_kwargs: list = (),
    success_rate_thresholds: list = (200, 750, 2500),
    dataset_roots: dict = None,
    config_dump: dict = None,
):
    """
    Train attacks on a set of images, then re-evaluate the trained perturbations
    at multiple sampling step counts to measure how attack success evolves with
    inference budget.

    Returns the JSON-serialisable results dict (also saved to results_dir).
    """
    import json
    import numpy as np
    from core import ImageLoader
    from attacks.attacks import run_attack
    from utils.adversarial_utils import (
        add_perturbation_to_image,
        expand_per_budget_kwargs,
        run_paired_pipeline_with_shared_noise,
    )

    seed_everything(seed)
    dataset_roots = dataset_roots or {}
    eval_num_steps = list(eval_num_steps)

    if isinstance(attack_types, str):
        attack_types = [attack_types]

    source_images, source_gps, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))
    device = str(attack_kwargs[0].get("device", "cuda"))
    eval_cfg = float(attack_kwargs[0].get("restart_eval_cfg", 10.0))
    eval_batch_size = int(attack_kwargs[0].get("restart_eval_batch_size", 128))

    # Phase 1: train attacks, collect best deltas
    # deltas[attack_type][budget_idx][image_idx] = CPU tensor
    deltas = {
        at: {bi: {} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            print(f"Training {attack_type} attacks (eps={budget:.4f})...")
            for image_idx, image in enumerate(tqdm_module.tqdm(source_images, desc="  images")):
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=pipeline,
                    eps_max=budget,
                    silent=True,
                    **dict(attack_kwargs[budget_idx]),
                )
                deltas[attack_type][budget_idx][image_idx] = result["delta"].detach().cpu()

    # Phase 2: re-evaluate each delta at every num_steps
    # raw[attack_type][budget_idx][num_steps] = [displacement_km, ...]
    raw = {
        at: {bi: {ns: [] for ns in eval_num_steps} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            for num_steps in tqdm_module.tqdm(
                eval_num_steps,
                desc=f"  {attack_type} eps={budget:.4f} eval steps",
            ):
                for image_idx, image in enumerate(source_images):
                    delta = deltas[attack_type][budget_idx][image_idx].to(device)
                    perturbed = add_perturbation_to_image(image, delta, pipeline)
                    eval_result = run_paired_pipeline_with_shared_noise(
                        pipeline=pipeline,
                        source_image=image,
                        perturbed_image=perturbed,
                        batch_size=eval_batch_size,
                        cfg=eval_cfg,
                        num_steps=int(num_steps),
                        seed=seed,
                        device=device,
                    )
                    raw[attack_type][budget_idx][num_steps].append(
                        float(eval_result["metrics"]["final_step_displacement"])
                    )

    # Build JSON-serialisable results dict
    json_results = {
        "dataset": dataset_name,
        "attack_types": attack_types,
        "attack_budgets": list(attack_budgets),
        "eval_num_steps": eval_num_steps,
        "success_rate_thresholds_km": list(success_rate_thresholds),
        "n_images": n_images_to_eval,
        "results": {},
    }
    for attack_type in attack_types:
        json_results["results"][attack_type] = {}
        for budget_idx, budget in enumerate(attack_budgets):
            bkey = f"budget_{budget:.6f}"
            json_results["results"][attack_type][bkey] = {}
            for num_steps in eval_num_steps:
                disps = raw[attack_type][budget_idx][num_steps]
                success_rates = {
                    str(thr): float(np.mean([d > thr for d in disps]))
                    for thr in success_rate_thresholds
                }
                json_results["results"][attack_type][bkey][str(num_steps)] = {
                    "mean_displacement_km": float(np.mean(disps)),
                    "success_rates": success_rates,
                }

    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, f"{dataset_name}_sampling_steps_results.json")
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved sampling-steps results to: {json_path}")

    return json_results


if __name__ == "__main__":
    # download_osv5m_test()
 
    device = "cuda"
    # attack_budgets = [1/255,2/255,5/255,10/255,20/255,30/255, 50/255] #yfcc
    attack_budgets = [1/255,2/255,5/255,10/255,15/255,20/255,25/255,30/255, 50/255]
    # attack_budgets = [2/255, 20/255, 50/255]
    train_args = [{"n_steps":80,
        "train_batch_size":256,
        "lr":1e-3,
        "anchor_samples":512,
        "clean_num_steps":100,
        "target_pure_noise": False,
        "dot_product_loss":"absolute",
        "reconstruction_loss_weight": 0.0,
        "num_restarts" : 6,
        "restart_selection_metric": "final_step_displacement",
        "restart_eval_cfg": 10.0,
        "device": device} for _ in range(len(attack_budgets))]
    
    # pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_OSV_5M_diffusion").to(device)	
    # pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_YFCC_diffusion").to(device)

    # evaluate_attack_on_dataset(
    #     attack_types=["encoder", "diffusion"],
    #     dataset_name="yfcc",
    #       source_image=None, 
    #     pipeline=pipeline, 
    #     n_images_to_eval=100,
    #     attack_budgets=attack_budgets,
    #     attack_kwargs=train_args,
    #     results_dir="./results",
    #     plot_dir="./plots",
    #     use_real_gps=False,
    # )
    
    # evaluate_localizability(
    #     attack_types=["encoder", "diffusion"],
    #     dataset_name="osv",
    #     pipeline=pipeline, 
    #     n_images_to_eval=100,
    #     attack_budgets=attack_budgets,
    #     attack_kwargs=train_args,
    #     results_dir="./results",
    #     plot_dir="./plots",
    # )
    
    # plot_localizability_results(
    #     results_dir="./results",
    #     attack_budgets=attack_budgets[0:2],
    #     plot_dir="./plots",
    #     dataset_name="osv",
    #     results=None
    # )
    
    # evaluate_attack_transferability(
    #     source_image=None,
    #     pipeline=pipeline,
    #     dataset_name="osv",
    #     attacks=["diffusion", "encoder"],
    #     n_images_to_eval=100,
    #     attack_kwargs=train_args,
    #     metric="final_step_displacement",
    #     results_dir="./results",
    #     plot_dir="./plots",
    # )
    
    plot_attack_success_rate(
        results_dir="./results",
        attack_budgets=attack_budgets,
        plot_dir="./plots",
        dataset_name="osv",
        attack_types=["encoder", "diffusion"],
        threshold_km=2500,
    )
 
