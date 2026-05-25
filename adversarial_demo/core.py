"""
Core refactored components for adversarial attack evaluations.

Consolidates common patterns for:
- Image loading and management
- Results storage and loading
- Metrics collection
- Evaluation execution
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import numpy as np
from PIL import Image
import tqdm as tqdm_module

from utils.adversarial_metrics import trajectory_displacement
from utils.adversarial_eval import retrieve_yfcc_images, retrieve_osv_images

# Note: adversarial_eval imports and attacks imports are deferred to avoid circular imports


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    dataset: str
    attack_types: List[str]
    attack_budgets: List[float]
    attack_kwargs: List[Dict[str, Any]]
    n_images: int
    results_dir: str
    plots_dir: str
    stored_metrics: List[str]
    parallel_workers: int = 1
    use_cuda_streams: bool = True
    use_real_gps: bool = False


class ImageLoader:
    """Unified image loading interface for different datasets."""
    
    @staticmethod
    def load_images(
        dataset: str,
        n_images: int,
        use_real_gps: bool = False,
        dataset_roots: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Image.Image], Optional[List[Tuple[float, float]]]]:
        """Load images and optional GPS coordinates for a dataset."""

        dataset_roots = dataset_roots or {}
        
        if dataset == "yfcc":
            return retrieve_yfcc_images(
                n_images_to_eval=n_images,
                use_real_gps=use_real_gps,
                local_dir=dataset_roots.get("yfcc"),
            )
        elif dataset == "osv":
            return retrieve_osv_images(
                n_images_to_eval=n_images,
                use_real_gps=use_real_gps,
                local_dir=dataset_roots.get("osv"),
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")


class ResultsManager:
    """Manages saving and loading evaluation results."""
    
    def __init__(self, results_dir: str, plots_dir: str):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)
    
    def get_results_path(self, dataset: str, attack_type: str, suffix: str = "") -> str:
        """Get the path for saving results of a specific attack type."""
        filename = f"{dataset}_{attack_type}_results{suffix}.pt"
        return os.path.join(self.results_dir, filename)
    
    def get_attack_args_path(self, dataset: str) -> str:
        """Get the path for saving attack arguments."""
        return os.path.join(self.results_dir, f"{dataset}_attack_args.pt")
    
    def get_metrics_path(self, dataset: str, suffix: str = "") -> str:
        """Get the path for saving metrics."""
        filename = f"{dataset}_metrics{suffix}.pt"
        return os.path.join(self.results_dir, filename)

    def get_run_config_path(self, dataset: str, suffix: str = "") -> str:
        """Get the path for saving the resolved experiment config."""
        filename = f"{dataset}_run_config{suffix}.yaml"
        return os.path.join(self.results_dir, filename)
    
    def save_results(
        self,
        results: Dict[str, Any],
        dataset: str,
        attack_type: str,
        suffix: str = "",
    ) -> None:
        """Save evaluation results for an attack type."""
        path = self.get_results_path(dataset, attack_type, suffix)
        torch.save(results, path)
        print(f"Saved results to: {path}")
    
    def load_results(
        self,
        dataset: str,
        attack_type: str,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        """Load saved evaluation results for an attack type."""
        path = self.get_results_path(dataset, attack_type, suffix)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Results file not found: {path}")
        return torch.load(path)
    
    def save_attack_args(
        self,
        attack_budgets: List[float],
        attack_kwargs: List[Dict[str, Any]],
        dataset: str,
    ) -> None:
        """Save attack arguments for reproducibility."""
        path = self.get_attack_args_path(dataset)
        torch.save({
            "attack_budgets": attack_budgets,
            "attack_kwargs": attack_kwargs,
        }, path)
        print(f"Saved attack args to: {path}")
    
    def load_attack_args(self, dataset: str) -> Dict[str, Any]:
        """Load saved attack arguments."""
        path = self.get_attack_args_path(dataset)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Attack args file not found: {path}")
        return torch.load(path)
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        dataset: str,
        suffix: str = "",
    ) -> None:
        """Save metrics data."""
        path = self.get_metrics_path(dataset, suffix)
        torch.save(metrics, path)
        print(f"Saved metrics to: {path}")

    def save_run_config(
        self,
        run_config: Dict[str, Any],
        dataset: str,
        suffix: str = "",
    ) -> None:
        """Save the resolved experiment configuration used for a run."""
        path = self.get_run_config_path(dataset, suffix)
        with open(path, "w") as f:
            yaml.safe_dump(run_config, f, sort_keys=False)
        print(f"Saved run config to: {path}")
    
    def load_metrics(self, dataset: str, suffix: str = "") -> Dict[str, Any]:
        """Load saved metrics."""
        path = self.get_metrics_path(dataset, suffix)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Metrics file not found: {path}")
        return torch.load(path)


class MetricsCollector:
    """Unified metrics collection for evaluations."""
    
    def __init__(
        self,
        attack_types: List[str],
        attack_budgets: List[float],
        n_images: int,
        stored_metrics: List[str],
        source_gps: Optional[List[Tuple[float, float]]] = None,
    ):
        self.attack_types = attack_types
        self.attack_budgets = attack_budgets
        self.n_images = n_images
        self.stored_metrics = stored_metrics
        self.source_gps = source_gps
        
        # Initialize result tensors for each attack type and metric
        self.results = {
            attack_type: {
                metric: torch.full((len(attack_budgets), n_images), float("nan"))
                for metric in stored_metrics
            }
            for attack_type in attack_types
        }
        self.restart_results: Dict[str, List[List[Optional[List[Dict[str, Any]]]]]] = {
            attack_type: [[None for _ in range(n_images)] for _ in attack_budgets]
            for attack_type in attack_types
        }
        self.location_results: Dict[str, List[List[Optional[Dict[str, Any]]]]] = {
            attack_type: [[None for _ in range(n_images)] for _ in attack_budgets]
            for attack_type in attack_types
        }

    @staticmethod
    def _to_cpu_tensor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).cpu()
        return value

    def _normalize_restart_result(
        self,
        restart_result: Dict[str, Any],
        true_gps: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {
            "metrics": dict(restart_result.get("metrics", {})),
            "gps_source": self._to_cpu_tensor(restart_result.get("gps_source")),
            "gps_perturbed": self._to_cpu_tensor(restart_result.get("gps_perturbed")),
            "traj_source": self._to_cpu_tensor(restart_result.get("traj_source")),
            "traj_perturbed": self._to_cpu_tensor(restart_result.get("traj_perturbed")),
        }

        clean_metric = normalized["metrics"].get("final_step_displacement")
        if clean_metric is not None:
            normalized["final_step_displacement_predicted"] = float(clean_metric)

        if true_gps is not None and normalized["gps_perturbed"] is not None:
            perturbed = normalized["gps_perturbed"]
            if isinstance(perturbed, torch.Tensor) and perturbed.ndim == 2 and perturbed.shape[-1] == 2:
                source_coords = torch.tensor(true_gps, dtype=perturbed.dtype).view(1, 1, 2)
                source_coords = source_coords.expand(1, perturbed.shape[0], 2)
                perturbed_traj = perturbed.unsqueeze(0)
                true_disp = trajectory_displacement(source_coords, perturbed_traj)
                normalized["final_step_displacement_true"] = float(true_disp.mean().item())

        normalized["true_gps"] = torch.tensor(true_gps, dtype=torch.float32) if true_gps is not None else None

        return normalized
    
    def record_metric(
        self,
        attack_type: str,
        budget_index: int,
        image_index: int,
        metric_name: str,
        value: float,
    ) -> None:
        """Record a single metric value."""
        if metric_name not in self.stored_metrics:
            raise ValueError(f"Unknown metric: {metric_name}")
        self.results[attack_type][metric_name][budget_index, image_index] = value
    
    def record_attack_result(
        self,
        attack_type: str,
        budget_index: int,
        image_index: int,
        attack_result: Dict[str, Any],
    ) -> None:
        """Record all metrics from an attack result."""
        if "best_metrics" not in attack_result:
            raise ValueError("Attack result missing 'best_metrics' key")

        true_gps = self.source_gps[image_index] if self.source_gps is not None else None
        
        best_metrics = attack_result["best_metrics"]
        restart_results = [
            self._normalize_restart_result(restart_result, true_gps=true_gps)
            for restart_result in attack_result.get("restart_evaluations", [])
        ]
        self.restart_results[attack_type][budget_index][image_index] = restart_results

        best_restart = attack_result.get("best_restart")
        best_restart_result = None
        if isinstance(best_restart, int) and 0 <= best_restart < len(restart_results):
            best_restart_result = restart_results[best_restart]

        location_result: Dict[str, Any] = {
            "best_restart": int(best_restart) if isinstance(best_restart, int) else None,
            "true_gps": torch.tensor(true_gps, dtype=torch.float32) if true_gps is not None else None,
            "predicted_gps_source": None,
            "predicted_gps_perturbed": None,
        }
        if best_restart_result is not None:
            location_result["predicted_gps_source"] = best_restart_result.get("gps_source")
            location_result["predicted_gps_perturbed"] = best_restart_result.get("gps_perturbed")
        self.location_results[attack_type][budget_index][image_index] = location_result

        for metric in self.stored_metrics:
            if metric == "final_step_displacement_predicted" and "final_step_displacement" in best_metrics:
                self.record_metric(
                    attack_type,
                    budget_index,
                    image_index,
                    metric,
                    float(best_metrics["final_step_displacement"]),
                )
            elif metric == "final_step_displacement_true" and best_restart_result is not None:
                true_metric = best_restart_result.get("final_step_displacement_true")
                if true_metric is not None:
                    self.record_metric(attack_type, budget_index, image_index, metric, float(true_metric))
            elif metric in best_metrics:
                self.record_metric(
                    attack_type,
                    budget_index,
                    image_index,
                    metric,
                    float(best_metrics[metric]),
                )
            elif metric == "final_step_displacement" and "final_step_displacement" in best_metrics:
                self.record_metric(
                    attack_type,
                    budget_index,
                    image_index,
                    metric,
                    float(best_metrics["final_step_displacement"]),
                )
    
    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected results."""
        combined_results: Dict[str, Dict[str, Any]] = {}
        for attack_type in self.attack_types:
            attack_results: Dict[str, Any] = {
                metric: tensor for metric, tensor in self.results[attack_type].items()
            }
            attack_results["restart_results"] = self.restart_results[attack_type]
            attack_results["location_results"] = self.location_results[attack_type]
            combined_results[attack_type] = attack_results
        return combined_results
    
    def get_attack_type_results(self, attack_type: str) -> Dict[str, torch.Tensor]:
        """Get results for a specific attack type."""
        return self.results[attack_type]


class EvaluationRunner:
    """Base class for running evaluations with unified logic."""
    
    def __init__(self, config: EvaluationConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.results_manager = ResultsManager(config.results_dir, config.plots_dir)
        
        # Load images
        print(f"Loading {config.n_images} images from {config.dataset} dataset...")
        self.source_images, self.source_gps = ImageLoader.load_images(
            dataset=config.dataset,
            n_images=config.n_images,
            use_real_gps=config.use_real_gps,
        )

        self.metrics_collector = MetricsCollector(
            attack_types=config.attack_types,
            attack_budgets=config.attack_budgets,
            n_images=config.n_images,
            stored_metrics=config.stored_metrics,
            source_gps=self.source_gps,
        )
    
    def save_results(self) -> None:
        """Save collected results and attack arguments."""
        results = self.metrics_collector.get_results()
        for attack_type in self.config.attack_types:
            self.results_manager.save_results(
                results[attack_type],
                self.config.dataset,
                attack_type,
            )
        
        self.results_manager.save_attack_args(
            self.config.attack_budgets,
            self.config.attack_kwargs,
            self.config.dataset,
        )

    def save_run_config(self, run_config: Dict[str, Any], suffix: str = "") -> None:
        """Save the resolved experiment configuration used for the run."""
        self.results_manager.save_run_config(run_config, self.config.dataset, suffix=suffix)


def parallel_evaluate_attacks(
    runner: EvaluationRunner,
    attack_configs: List[Tuple[str, int, int, Image.Image]],  # (attack_type, budget_idx, image_idx, image)
) -> None:
    """Run attacks in parallel with thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from attacks.attacks import run_attack
    
    config = runner.config
    total = len(attack_configs)
    
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")
    
    def _evaluate_task(attack_type: str, budget_idx: int, image_idx: int, image: Image.Image):
        """Worker task for parallel evaluation."""
        eps = config.attack_budgets[budget_idx]
        kwargs = dict(config.attack_kwargs[budget_idx])
        
        # Handle CUDA streams if requested
        if (config.use_cuda_streams and 
            str(kwargs.get("device", "cpu")).startswith("cuda")):
            import torch
            stream = torch.cuda.Stream(device=kwargs.get("device", "cuda"))
            with torch.cuda.stream(stream):
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=runner.pipeline,
                    eps_max=eps,
                    silent=True,
                    **kwargs,
                )
            stream.synchronize()
        else:
            result = run_attack(
                attack_type=attack_type,
                source_image=image,
                pipeline=runner.pipeline,
                eps_max=eps,
                silent=True,
                **kwargs,
            )
        
        return attack_type, budget_idx, image_idx, result
    
    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = [
            executor.submit(_evaluate_task, at, bi, ii, img)
            for at, bi, ii, img in attack_configs
        ]
        
        for future in as_completed(futures):
            attack_type, budget_idx, image_idx, result = future.result()
            runner.metrics_collector.record_attack_result(
                attack_type,
                budget_idx,
                image_idx,
                result,
            )
            eps = config.attack_budgets[budget_idx]
            pbar.set_postfix(
                attack=attack_type,
                eps=f"{eps:.4f}",
                image=f"{image_idx+1}/{config.n_images}"
            )
            pbar.update(1)
    
    pbar.close()


def sequential_evaluate_attacks(
    runner: EvaluationRunner,
) -> None:
    """Run attacks sequentially."""
    from attacks.attacks import run_attack
    config = runner.config
    total = len(config.attack_types) * len(config.attack_budgets) * len(runner.source_images)
    
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")
    
    for attack_type in config.attack_types:
        for budget_idx, eps in enumerate(config.attack_budgets):
            for image_idx, image in enumerate(runner.source_images):
                kwargs = dict(config.attack_kwargs[budget_idx])
                
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=runner.pipeline,
                    eps_max=eps,
                    silent=True,
                    **kwargs,
                )
                runner.metrics_collector.record_attack_result(
                    attack_type,
                    budget_idx,
                    image_idx,
                    result,
                )
                
                pbar.set_postfix(
                    attack=attack_type,
                    eps=f"{eps:.4f}",
                    image=f"{image_idx+1}/{config.n_images}"
                )
                pbar.update(1)
    
    pbar.close()


def run_evaluation(runner: EvaluationRunner) -> None:
    """Execute evaluation with appropriate execution strategy."""
    if runner.config.parallel_workers > 1:
        # Build task list for parallel execution
        attack_configs = [
            (at, bi, ii, img)
            for at in runner.config.attack_types
            for bi in range(len(runner.config.attack_budgets))
            for ii, img in enumerate(runner.source_images)
        ]
        parallel_evaluate_attacks(runner, attack_configs)
    else:
        sequential_evaluate_attacks(runner)
