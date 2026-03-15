from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch


from attacks import run_attack
from adversarial_metrics import evaluate_displacement_metrics, select_displacement_score

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

from adversarial_utils import add_perturbation_to_image

from plots_adversarial_attacks import plot_results, plot_transferability_results



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
  
def evaluate_attack_on_dataset(
	attack_types,
	pipeline,
	dataset_name,
	use_real_gps: bool = False,
	n_images_to_eval: int = 100,
	plot_dir: Optional[str] = "/plots",
	results_dir: Optional[str] = "/results",
	attack_budgets: list[float] = [2/255, 15/255, 50/255],
	stored_metrics = ["final_step_displacement"],
	attack_kwargs: list[Dict[str, Any]] = [{}],):
	"""
		Evaluate one or more attacks on images from a test dataset.

		Args:
			attack_types: A single attack name (str) or a list of attack names, e.g. ["encoder", "diffusion"].
			pipeline: Plonk pipeline.
			dataset_name: Name of the dataset to evaluate on. "osv" or "yfcc".
			use_real_gps: Whether use real gps coords from dataset as source trajectory, instead of the clean predicted one for evaluation.
			**kwargs: forwarded to the corresponding attack function.
	"""
	if isinstance(attack_types, str):
		attack_types = [attack_types]

	if dataset_name == "osv":
		source_images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval, use_real_gps=use_real_gps)
	elif dataset_name == "yfcc":
		source_images, source_gps = retrieve_yfcc_images(n_images_to_eval=n_images_to_eval, use_real_gps=use_real_gps)
	else:
		raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv', 'yfcc']")

	all_results = {} # attack_type -> {metric: tensor}
	total = len(attack_types) * len(attack_budgets) * len(source_images)
	pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")
	for attack_type in attack_types:
		results = {metric: torch.zeros((len(attack_budgets), len(source_images))) for metric in stored_metrics}
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
					results[metric][j, i] = attack_result["best_metrics"][metric]
				pbar.set_postfix(attack=attack_type, eps=f"{eps:.4f}", image=f"{i+1}/{len(source_images)}")
				pbar.update(1)

		#store metrics separately per attack
		if results_dir is not None:
			os.makedirs(results_dir, exist_ok=True)
			torch.save(results, os.path.join(results_dir, f"{dataset_name}_{attack_type}_results.pt"))
		all_results[attack_type] = results
	pbar.close()

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
	attack_kwargs: list[Dict[str, Any]] = [{}],):
	"""
		Evaluates the transferability of an attack. It trains a perturbation on a source_image,
		then applies the same perturbation to a set of source images from a dataset and evaluates
		the attack success metrics on the perturbed images.
	"""

	if dataset_name == "osv":
		images, source_gps = retrieve_osv_images(n_images_to_eval=n_images_to_eval)
	else:
		raise ValueError(f"Unknown dataset_name={dataset_name}. Expected one of ['osv']")

	if source_image is None:
		source_image = images[0]

	results = {attack: torch.zeros((len(attack_budgets), len(images))) for attack in attacks}
	total = len(attacks) * len(attack_budgets) * (1 + len(images))  # 1 train + N evals per (attack, budget)
	pbar = tqdm_module.tqdm(total=total, desc="Evaluating transferability")
	for attack in attacks:
		for j, eps in enumerate(attack_budgets):
			attack_result = run_attack(
				attack_type=attack,
				source_image=source_image,
				pipeline=pipeline,
				eps_max=eps,
				silent=True,
				**attack_kwargs[j],
			)
			pbar.update(1)
			best_delta = attack_result["delta"]
			#evaluate on images:
			for i, img in enumerate(images):
				perturbed_image = add_perturbation_to_image(img, best_delta, pipeline)
				_, traj_source = pipeline(img, batch_size=256, cfg=10.0, return_trajectories=True)
				_, traj_perturbed = pipeline(perturbed_image, batch_size=256, cfg=10.0, return_trajectories=True)
				metrics = evaluate_displacement_metrics(traj_source, traj_perturbed)
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

if __name__ == "__main__":
	# download_osv5m_test()
 
	device = "cuda"
 
	attack_budgets = [2/255,10/255,20/255, 50/255, 1.0]
 
	train_args = [{"n_steps":50,
		"train_batch_size":256,
		"lr":1e-3,
		"anchor_samples":512,
		"clean_num_steps":100,
		"target_pure_noise": False,
		"dot_product_loss":"absolute",
		"reconstruction_loss_weight": 0.0,
		"num_restarts" : 3,
		"restart_selection_metric": "final_step_displacement",
		"restart_eval_cfg": 10.0,
		"device": device} for _ in range(len(attack_budgets))]
	
	# pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_OSV_5M_diffusion").to(device)	
	pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_YFCC_diffusion").to(device)
	
	evaluate_attack_on_dataset(
		attack_types=["encoder", "diffusion"],
		dataset_name="yfcc",
		pipeline=pipeline, 
		n_images_to_eval=30,
		attack_budgets=attack_budgets,
		attack_kwargs=train_args,
		results_dir="./results",
		plot_dir="./plots",
		use_real_gps=False,
	)