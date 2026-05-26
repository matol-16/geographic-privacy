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
):
    if local_dir is None:
        local_dir = "/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k"
    info_path = os.path.join(local_dir, "info.txt")
    img_dir = os.path.join(local_dir, "images")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"YFCC4k info.txt not found at {info_path}. Run build_yfcc4k_from_revisiting_im2gps.py first.")

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

    rng = random.Random(seed)
    samples = rng.sample(rows, min(n_images_to_eval, len(rows)))

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

    # Sample n_images_to_eval random images
    rng = random.Random(seed)
    samples = rng.sample(rows, min(n_images_to_eval, len(rows)))

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
 
