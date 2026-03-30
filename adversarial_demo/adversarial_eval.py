from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import numpy as np
import torch


from attacks import run_attack
from adversarial_metrics import evaluate_displacement_metrics, select_displacement_score
from plots_adversarial_attacks import plot_results, plot_transferability_results, plot_localizability_results, plot_attack_success_rate

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

from pipe_trajectory import PlonkPipelineTrajectory

from adversarial_utils import (
    add_perturbation_to_image,
    expand_per_budget_kwargs,
    resolve_torch_device,
    run_paired_pipeline_with_shared_noise,
)



def load_osv5m_test(local_dir="datasets/osv5m"):
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

def retrieve_yfcc_images(n_images_to_eval: int = 100, use_real_gps: bool = False, local_dir: str = "datasets/YFCC100M/yfcc4k"):
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

    rng = random.Random(42)
    samples = rng.sample(rows, min(n_images_to_eval, len(rows)))

    source_images = [Image.open(s["path"]) for s in samples]
    source_gps = None
    if use_real_gps:
        source_gps = [(s["latitude"], s["longitude"]) for s in samples]
    print(f"Loaded {len(source_images)} images from YFCC4k.")
    return source_images, source_gps


def retrieve_osv_images(n_images_to_eval: int = 100, use_real_gps: bool = False):
    local_dir = "datasets/osv5m"
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
    rng = random.Random(42)
    samples = rng.sample(rows, min(n_images_to_eval, len(rows)))

    source_images = [Image.open(id_to_path[s["id"]]) for s in samples]
    source_gps = None
    if use_real_gps:
        source_gps = [(float(s["latitude"]), float(s["longitude"])) for s in samples]
    print(f"Loaded {len(source_images)} images from OSV-5M test set.")
    return source_images, source_gps


def _evaluate_single_attack_task(
    attack_type,
    budget_index: int,
    image_index: int,
    image,
    pipeline,
    attack_budgets,
    attack_kwargs,
    stored_metrics,
    use_cuda_streams: bool,
):
    eps = attack_budgets[budget_index]
    kwargs = dict(attack_kwargs[budget_index])
    worker_device = str(kwargs.get("device", getattr(pipeline, "device", "cpu")))

    if use_cuda_streams and worker_device.startswith("cuda") and torch.cuda.is_available():
        stream = torch.cuda.Stream(device=worker_device)
        with torch.cuda.stream(stream):
            attack_result = run_attack(
                attack_type=attack_type,
                source_image=image,
                pipeline=pipeline,
                eps_max=eps,
                silent=True,
                **kwargs,
            )
        stream.synchronize()
    else:
        attack_result = run_attack(
            attack_type=attack_type,
            source_image=image,
            pipeline=pipeline,
            eps_max=eps,
            silent=True,
            **kwargs,
        )

    metric_values = {metric: float(attack_result["best_metrics"][metric]) for metric in stored_metrics}
    return attack_type, budget_index, image_index, metric_values
  
def evaluate_attack_on_dataset(
    attack_types,
    pipeline,
    dataset_name,
    source_image=None, 
    use_real_gps: bool = False,
    n_images_to_eval: int = 100,
    plot_dir: Optional[str] = "/plots",
    results_dir: Optional[str] = "/results",
    attack_budgets: list[float] = [2/255, 15/255, 50/255],
    stored_metrics = ["final_step_displacement"],
    attack_kwargs: list[Dict[str, Any]] = [{}],
    parallel_workers: int = 1,
    use_cuda_streams: bool = True,
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
    if isinstance(attack_types, str):
        attack_types = [attack_types]

    if source_image is not None:
        source_images = [source_image]
    elif dataset_name == "osv":
        source_images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval, use_real_gps=use_real_gps)
    elif dataset_name == "yfcc":
        source_images, source_gps = retrieve_yfcc_images(n_images_to_eval=n_images_to_eval, use_real_gps=use_real_gps)
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv', 'yfcc']")

    total = len(attack_types) * len(attack_budgets) * len(source_images)
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")
    all_results = {
        attack_type: {metric: torch.zeros((len(attack_budgets), len(source_images))) for metric in stored_metrics}
        for attack_type in attack_types
    }

    run_in_parallel = parallel_workers > 1

    if run_in_parallel:
        tasks = [
            (attack_type, j, i, source_image)
            for attack_type in attack_types
            for j in range(len(attack_budgets))
            for i, source_image in enumerate(source_images)
        ]

        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [
                executor.submit(
                    _evaluate_single_attack_task,
                    attack_type,
                    j,
                    i,
                    source_image,
                    pipeline,
                    attack_budgets,
                    attack_kwargs,
                    stored_metrics,
                    use_cuda_streams,
                )
                for attack_type, j, i, source_image in tasks
            ]
            for future in as_completed(futures):
                attack_type, j, i, metric_values = future.result()
                for metric in stored_metrics:
                    all_results[attack_type][metric][j, i] = metric_values[metric]
                pbar.set_postfix(attack=attack_type, eps=f"{attack_budgets[j]:.4f}", image=f"{i+1}/{len(source_images)}")
                pbar.update(1)
    else:
        for attack_type in attack_types:
            for j, eps in enumerate(attack_budgets):
                for i, source_image in enumerate(source_images):
                    attack_result = run_attack(
                        attack_type=attack_type,
                        source_image=source_image,
                        pipeline=pipeline,
                        eps_max=eps,
                        silent=True,
                        **attack_kwargs[j], #hyperparameters are specific to each attack budget
                    )
                    for metric in stored_metrics:
                        all_results[attack_type][metric][j, i] = attack_result["best_metrics"][metric]
                    pbar.set_postfix(attack=attack_type, eps=f"{eps:.4f}", image=f"{i+1}/{len(source_images)}")
                    pbar.update(1)

    for attack_type in attack_types:
        #store metrics separately per attack
        if results_dir is not None:
            os.makedirs(results_dir, exist_ok=True)
            torch.save(all_results[attack_type], os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt"))
    pbar.close()
    
    #also save attack args
    attack_args = {"attack_budgets": attack_budgets, "attack_kwargs": attack_kwargs}
    if results_dir is not None:
        torch.save(attack_args, os.path.join(results_dir, f"{dataset_name}_attack_args.pt"))

    #plot all attacks together
    plot_results(
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        plot_dir=plot_dir,
        dataset_name=dataset_name,
        attack_types=attack_types,
        all_results=all_results,
        stored_metrics=stored_metrics)
 
 
 
def evaluate_attack_transferability(
    source_image, #can be None; in that case the first image from the dataset is used
    pipeline,
    dataset_name,
    attacks: list[str] = ["encoder", "diffusion"],
    n_images_to_eval: int = 100,
    plot_dir: Optional[str] = "/plots",
    results_dir: Optional[str] = "/results",
    attack_budgets: list[float] = [2/255, 15/255, 50/255],
    metric="final_step_displacement",
    attack_kwargs: list[Dict[str, Any]] = [{}],
    eval_batch_size: int = 256,
    eval_cfg: float = 10.0,
    eval_num_steps: Optional[int] = None,
    eval_seed: int = 1234,
):
    """
        Evaluates the transferability of an attack. It trains a perturbation on a source_image,
        then applies the same perturbation to a set of source images from a dataset and evaluates
        the attack success metrics on the perturbed images.
    """
    if dataset_name == "osv":
        images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval)
    elif dataset_name == "yfcc":
        images, source_gps = retrieve_yfcc_images(n_images_to_eval=n_images_to_eval)
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv', 'yfcc']")


    if source_image is None:
        source_image = images[0]

    attack_kwargs = expand_per_budget_kwargs(attack_kwargs, len(attack_budgets))

    results = {attack: torch.zeros((len(attack_budgets), len(images))) for attack in attacks}
    total = len(attacks) * len(attack_budgets) * (1 + len(images))  # 1 train + N evals per (attack, budget)
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating transferability")
    for attack_idx, attack in enumerate(attacks):
        for j, eps in enumerate(attack_budgets):
            kwargs_j = dict(attack_kwargs[j])
            attack_result = run_attack(
                attack_type=attack,
                source_image=source_image,
                pipeline=pipeline,
                eps_max=eps,
                silent=True,
                **kwargs_j,
            )
            pbar.update(1)
            best_delta = attack_result["delta"]
            #evaluate on images:
            for i, img in enumerate(images):
                perturbed_image = add_perturbation_to_image(img, best_delta, pipeline)

                eval_device = resolve_torch_device(kwargs_j.get("device", getattr(pipeline, "device", "cpu")))
                eval_result = run_paired_pipeline_with_shared_noise(
                    pipeline=pipeline,
                    source_image=img,
                    perturbed_image=perturbed_image,
                    batch_size=int(eval_batch_size),
                    cfg=float(eval_cfg),
                    num_steps=eval_num_steps,
                    seed=int(eval_seed) + attack_idx * 1_000_000 + j * 10_000 + i,
                    device=eval_device,
                )
                metrics = eval_result["metrics"]
                results[attack][j, i] = metrics[metric]
                pbar.set_postfix(attack=attack, eps=f"{eps:.4f}", image=f"{i+1}/{len(images)}")
                pbar.update(1)
    pbar.close()

    #store results
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        torch.save(results, os.path.join(results_dir, f"{dataset_name}_results_transferability.pt"))
        

    #plot results
    plot_transferability_results(
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        plot_dir=plot_dir,
        dataset_name=dataset_name,
        metric=metric,
        results=results)
 
def evaluate_localizability(
    attack_types,
    pipeline,
    dataset_name,
    n_images_to_eval: int = 100,
    plot_dir: Optional[str] = "/plots",
    results_dir: Optional[str] = "/results",
    attack_budgets: list[float] = [2/255, 15/255, 50/255],
    attack_kwargs: list[Dict[str, Any]] = [{}],):
    """ 
    Evaluates how strong an attack is depending on the localizability of the source image.
    
    Localizability is computed using the clean image. Attack strength is measured via final step displacement. We can bucket images into low/med/high localizability and plot the average attack strength in each bucket, for different attack budgets and attack types.
    
    This evaluation is done for each attack type and attack budget, to see if some attacks are more effective on low-localizability images than others, and if this trend is stronger for higher attack budgets.
    """
    
    if dataset_name == "osv":
        images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval)

    elif dataset_name == "yfcc":
        images, source_gps = retrieve_yfcc_images(n_images_to_eval=n_images_to_eval)
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv', 'yfcc']")
    
    results = {"attack_results" : {attack: torch.zeros((len(attack_budgets), len(images))) for attack in attack_types}, "localizability" : torch.zeros(len(images))}

    pbar = tqdm_module.tqdm(total=len(images), desc="Computing localizability")    
    for i, img in enumerate(images):
        results["localizability"][i] = pipeline.compute_localizability(img, number_monte_carlo_samples=256).item()
        pbar.update(1)
    pbar.close()
    
    total = len(attack_types) * len(attack_budgets) * len(images)
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating localizability")
    for attack in attack_types:
        for j, eps in enumerate(attack_budgets):
            for i, img in enumerate(images):
                attack_result = run_attack(
                    attack_type=attack,
                    source_image=img,
                    pipeline=pipeline,
                    eps_max=eps,
                    silent=True,
                    **attack_kwargs[j],
                )
                results["attack_results"][attack][j, i] = attack_result["best_metrics"]["final_step_displacement"]
                pbar.set_postfix(attack=attack, eps=f"{eps:.4f}", image=f"{i+1}/{len(images)}")
                pbar.update(1)
    pbar.close()
    
        #store results
    if results_dir is not None:
        os.makedirs(results_dir, exist_ok=True)
        torch.save(results, os.path.join(results_dir, f"{dataset_name}_results_localizability.pt"))

    #plot results
    plot_localizability_results(
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        plot_dir=plot_dir,
        dataset_name=dataset_name,
        results=results)
    
    

      
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
        threshold_km=[500,1000,1500,2500]
    )
 
