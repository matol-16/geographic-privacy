"""
Core refactored components for adversarial attack evaluations.

Consolidates common patterns for:
- Image loading and management
- Results storage and loading
- Metrics collection
- Evaluation execution
"""

from __future__ import annotations

import json
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
from utils.adversarial_utils import collect_common_image_pairs, run_paired_pipeline_with_shared_noise

# Note: adversarial_eval imports and attacks imports are deferred to avoid circular imports


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""
    dataset: str
    seed: int
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
    dataset_roots: Optional[Dict[str, str]] = None
    state_suffix: str = ""


@dataclass
class PrecomputedPairEvaluationConfig:
    """Configuration for evaluating precomputed clean/attacked image pairs."""
    dataset: str
    attack_name: str
    seed: int
    attack_budgets: List[float]
    clean_image_dirs: List[str]
    attacked_image_dirs: List[str]
    results_dir: str
    plots_dir: str
    stored_metrics: List[str]
    device: str = "cuda"
    batch_size: int = 256
    cfg: float = 10.0
    num_steps: Optional[int] = None
    n_images: Optional[int] = None
    state_suffix: str = ""


class ImageLoader:
    """Unified image loading interface for different datasets."""
    
    @staticmethod
    def load_images(
        dataset: str,
        n_images: int,
        seed: int = 0,
        use_real_gps: bool = False,
        dataset_roots: Optional[Dict[str, str]] = None,
    ) -> Tuple[List[Image.Image], Optional[List[Tuple[float, float]]], List[str]]:
        """Load images, optional GPS coordinates, and stable dataset image IDs."""

        dataset_roots = dataset_roots or {}
        
        if dataset == "yfcc":
            return retrieve_yfcc_images(
                n_images_to_eval=n_images,
                seed=seed,
                use_real_gps=use_real_gps,
                local_dir=dataset_roots.get("yfcc"),
            )
        elif dataset == "osv":
            return retrieve_osv_images(
                n_images_to_eval=n_images,
                seed=seed,
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

    def get_state_path(self, dataset: str, seed: int, suffix: str = "") -> str:
        """Get the path for saving incremental evaluation state."""
        filename = f"{dataset}_seed{seed}_eval_state{suffix}.pt"
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
        suffix: str = "",
    ) -> None:
        """Save attack arguments for reproducibility."""
        filename = f"{dataset}_attack_args{suffix}.pt"
        path = os.path.join(self.results_dir, filename)
        torch.save({
            "attack_budgets": attack_budgets,
            "attack_kwargs": attack_kwargs,
        }, path)
        print(f"Saved attack args to: {path}")
    
    def load_attack_args(self, dataset: str, suffix: str = "") -> Dict[str, Any]:
        """Load saved attack arguments."""
        filename = f"{dataset}_attack_args{suffix}.pt"
        path = os.path.join(self.results_dir, filename)
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

    def save_state(
        self,
        state: Dict[str, Any],
        dataset: str,
        seed: int,
        suffix: str = "",
    ) -> None:
        """Persist incremental evaluation state atomically."""
        path = self.get_state_path(dataset, seed, suffix)
        tmp_path = f"{path}.tmp"
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        print(f"Saved evaluation state to: {path}")

    def load_state(
        self,
        dataset: str,
        seed: int,
        suffix: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Load incremental evaluation state if it exists."""
        path = self.get_state_path(dataset, seed, suffix)
        if not os.path.exists(path):
            return None
        return torch.load(path, map_location="cpu")
    
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
        source_image_ids: Optional[List[str]] = None,
    ):
        self.attack_types = attack_types
        self.attack_budgets = attack_budgets
        self.n_images = n_images
        self.stored_metrics = stored_metrics
        self.source_gps = source_gps
        self.source_image_ids = source_image_ids
        
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
        def _final_step_tensor(val: Any) -> Any:
            v = self._to_cpu_tensor(val)
            # If we have a multi-step trajectory shaped (steps, 2), keep only final step
            if isinstance(v, torch.Tensor):
                # ensure final-step has shape (1, 2) for compatibility
                if v.ndim == 2 and v.shape[-1] == 2:
                    if v.shape[0] > 1:
                        v = v[-1].unsqueeze(0)
                    else:
                        v = v.view(1, 2)
                # If already a single coordinate (2,), convert to (1,2)
                elif v.ndim == 1 and v.shape[0] == 2:
                    v = v.view(1, 2)
            return v

        normalized: Dict[str, Any] = {
            "metrics": dict(restart_result.get("metrics", {})),
            # store only final-step coordinates (small) instead of full trajectories
            "gps_source": _final_step_tensor(restart_result.get("gps_source")),
            "gps_perturbed": _final_step_tensor(restart_result.get("gps_perturbed")),
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
            attack_results["image_indices"] = list(range(self.n_images))
            attack_results["image_ids"] = self.source_image_ids
            attack_results["restart_results"] = self.restart_results[attack_type]
            attack_results["location_results"] = self.location_results[attack_type]
            combined_results[attack_type] = attack_results
        return combined_results

    def is_task_complete(self, attack_type: str, budget_index: int, image_index: int) -> bool:
        """Check whether a specific attack/budget/image tuple has already been recorded."""
        return self.location_results[attack_type][budget_index][image_index] is not None
    
    def get_attack_type_results(self, attack_type: str) -> Dict[str, torch.Tensor]:
        """Get results for a specific attack type."""
        return self.results[attack_type]


class EvaluationRunner:
    """Base class for running evaluations with unified logic."""
    
    def __init__(self, config: EvaluationConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.results_manager = ResultsManager(config.results_dir, config.plots_dir)
        self.state_signature = self._build_state_signature()
        
        # Load images
        print(f"Loading {config.n_images} images from {config.dataset} dataset...")
        self.source_images, self.source_gps, self.source_image_ids = ImageLoader.load_images(
            dataset=config.dataset,
            n_images=config.n_images,
            seed=config.seed,
            use_real_gps=config.use_real_gps,
            dataset_roots=config.dataset_roots,
        )

        self.metrics_collector = MetricsCollector(
            attack_types=config.attack_types,
            attack_budgets=config.attack_budgets,
            n_images=config.n_images,
            stored_metrics=config.stored_metrics,
            source_gps=self.source_gps,
            source_image_ids=self.source_image_ids,
        )
        self._load_state_if_available()

    def _build_state_signature(self) -> Dict[str, Any]:
        """Create a compact signature that guards resume compatibility."""
        return {
            "dataset": self.config.dataset,
            "seed": self.config.seed,
            "attack_types": list(self.config.attack_types),
            "attack_budgets": list(self.config.attack_budgets),
            "attack_kwargs": self.config.attack_kwargs,
            "n_images": self.config.n_images,
            "stored_metrics": list(self.config.stored_metrics),
            "use_real_gps": self.config.use_real_gps,
            "dataset_roots": self.config.dataset_roots or {},
            "state_suffix": self.config.state_suffix,
        }

    def _build_state(self) -> Dict[str, Any]:
        """Collect the current incremental state for persistence."""
        results = {
            attack_type: {
                metric: tensor.detach().cpu()
                for metric, tensor in attack_results.items()
            }
            for attack_type, attack_results in self.metrics_collector.results.items()
        }
        return {
            "version": 1,
            "signature": self.state_signature,
            "source_image_ids": list(self.source_image_ids) if self.source_image_ids is not None else None,
            "image_indices": list(range(self.config.n_images)),
            "results": results,
            "restart_results": self.metrics_collector.restart_results,
            "location_results": self.metrics_collector.location_results,
        }

    def _load_state_if_available(self) -> None:
        """Restore progress from a previous interrupted run when compatible."""
        state = self.results_manager.load_state(
            self.config.dataset,
            self.config.seed,
            suffix=self.config.state_suffix,
        )
        if state is None:
            return

        saved_signature = state.get("signature")
        if not self._is_state_compatible(saved_signature):
            print("Existing evaluation state is incompatible with the current configuration; starting fresh.")
            return

        saved_image_ids = state.get("source_image_ids")
        if not isinstance(saved_image_ids, list):
            print("Existing evaluation state does not include image IDs; starting fresh.")
            return

        current_image_ids = list(self.source_image_ids)
        shared_n_images = min(len(saved_image_ids), len(current_image_ids))
        if saved_image_ids[:shared_n_images] != current_image_ids[:shared_n_images]:
            print("Existing evaluation state was built from a different image ordering; starting fresh.")
            return

        results = state.get("results")
        restart_results = state.get("restart_results")
        location_results = state.get("location_results")
        if results is None or restart_results is None or location_results is None:
            print("Existing evaluation state is incomplete; starting fresh.")
            return

        reusable_budget_pairs = self._get_reusable_budget_pairs(saved_signature)
        if not reusable_budget_pairs:
            print("Existing evaluation state has no reusable budget overlap; starting fresh.")
            return

        self._merge_saved_state(
            results=results,
            restart_results=restart_results,
            location_results=location_results,
            shared_n_images=shared_n_images,
            reusable_budget_pairs=reusable_budget_pairs,
        )
        print(
            f"Reused {len(reusable_budget_pairs)}/{len(self.config.attack_budgets)} budget rows from saved state."
        )
        print(f"Resumed evaluation state from {self.results_manager.get_state_path(self.config.dataset, self.config.seed, self.config.state_suffix)}")

    def _is_state_compatible(self, saved_signature: Any) -> bool:
        """Check whether a saved state can be partially merged into the current run."""
        if not isinstance(saved_signature, dict):
            return False

        keys_that_must_match = [
            "dataset",
            "seed",
            "stored_metrics",
            "use_real_gps",
            "dataset_roots",
            "state_suffix",
        ]
        for key in keys_that_must_match:
            if saved_signature.get(key) != self.state_signature.get(key):
                return False

        return True

    def _get_reusable_budget_pairs(self, saved_signature: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Return (current_budget_idx, saved_budget_idx) pairs that are safe to reuse."""
        saved_budgets = saved_signature.get("attack_budgets")
        saved_kwargs = saved_signature.get("attack_kwargs")
        if not isinstance(saved_budgets, list) or not isinstance(saved_kwargs, list):
            return []

        current_budgets = list(self.config.attack_budgets)
        current_kwargs = list(self.config.attack_kwargs)

        used_saved_indices: set[int] = set()
        reusable_pairs: List[Tuple[int, int]] = []
        for current_idx, budget in enumerate(current_budgets):
            current_kwarg = current_kwargs[current_idx] if current_idx < len(current_kwargs) else None
            match_idx: Optional[int] = None
            for saved_idx, saved_budget in enumerate(saved_budgets):
                if saved_idx in used_saved_indices:
                    continue
                if saved_budget != budget:
                    continue
                saved_kwarg = saved_kwargs[saved_idx] if saved_idx < len(saved_kwargs) else None
                if saved_kwarg != current_kwarg:
                    continue
                match_idx = saved_idx
                break

            if match_idx is None:
                continue
            used_saved_indices.add(match_idx)
            reusable_pairs.append((current_idx, match_idx))

        return reusable_pairs

    def _merge_saved_state(
        self,
        results: Any,
        restart_results: Any,
        location_results: Any,
        shared_n_images: int,
        reusable_budget_pairs: List[Tuple[int, int]],
    ) -> None:
        """Copy overlap from a compatible saved state into current in-memory buffers."""
        if not isinstance(results, dict):
            return
        if not isinstance(restart_results, dict):
            return
        if not isinstance(location_results, dict):
            return

        for attack_type in self.config.attack_types:
            if attack_type not in results:
                continue
            saved_attack_results = results.get(attack_type)
            if not isinstance(saved_attack_results, dict):
                continue

            for metric in self.config.stored_metrics:
                saved_metric = saved_attack_results.get(metric)
                if not isinstance(saved_metric, torch.Tensor):
                    continue
                current_metric = self.metrics_collector.results[attack_type][metric]
                image_limit = min(current_metric.shape[1], saved_metric.shape[1], shared_n_images)
                if image_limit == 0:
                    continue
                for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
                    if current_budget_idx >= current_metric.shape[0] or saved_budget_idx >= saved_metric.shape[0]:
                        continue
                    current_metric[current_budget_idx, :image_limit] = (
                        saved_metric[saved_budget_idx, :image_limit].detach().cpu()
                    )

            saved_restart_by_budget = restart_results.get(attack_type)
            saved_location_by_budget = location_results.get(attack_type)
            if not isinstance(saved_restart_by_budget, list) or not isinstance(saved_location_by_budget, list):
                continue

            current_restart_by_budget = self.metrics_collector.restart_results[attack_type]
            current_location_by_budget = self.metrics_collector.location_results[attack_type]
            for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
                if current_budget_idx >= len(current_restart_by_budget) or current_budget_idx >= len(current_location_by_budget):
                    continue
                if saved_budget_idx >= len(saved_restart_by_budget) or saved_budget_idx >= len(saved_location_by_budget):
                    continue

                current_restart_by_image = current_restart_by_budget[current_budget_idx]
                current_location_by_image = current_location_by_budget[current_budget_idx]
                saved_restart_by_image = saved_restart_by_budget[saved_budget_idx]
                saved_location_by_image = saved_location_by_budget[saved_budget_idx]
                if not isinstance(saved_restart_by_image, list) or not isinstance(saved_location_by_image, list):
                    continue

                image_limit = min(
                    len(current_restart_by_image),
                    len(current_location_by_image),
                    len(saved_restart_by_image),
                    len(saved_location_by_image),
                    shared_n_images,
                )
                for image_idx in range(image_limit):
                    if saved_restart_by_image[image_idx] is not None:
                        current_restart_by_image[image_idx] = saved_restart_by_image[image_idx]
                    if saved_location_by_image[image_idx] is not None:
                        current_location_by_image[image_idx] = saved_location_by_image[image_idx]

    def save_state(self) -> None:
        """Persist the current incremental evaluation state."""
        self.results_manager.save_state(
            self._build_state(),
            self.config.dataset,
            self.config.seed,
            suffix=self.config.state_suffix,
        )

    def get_attack_configs(self, pending_only: bool = True) -> List[Tuple[str, int, int, Image.Image]]:
        """Build the evaluation task list, optionally skipping completed tasks."""
        attack_configs: List[Tuple[str, int, int, Image.Image]] = []
        for attack_type in self.config.attack_types:
            for budget_idx, _ in enumerate(self.config.attack_budgets):
                for image_idx, image in enumerate(self.source_images):
                    if pending_only and self.metrics_collector.is_task_complete(attack_type, budget_idx, image_idx):
                        continue
                    attack_configs.append((attack_type, budget_idx, image_idx, image))
        return attack_configs
    
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


class PrecomputedPairEvaluationRunner:
    """Evaluation runner for precomputed clean/attacked image pairs."""

    def __init__(self, config: PrecomputedPairEvaluationConfig, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.results_manager = ResultsManager(config.results_dir, config.plots_dir)
        self.state_signature = self._build_state_signature()

        self.pairs_by_budget = [
            collect_common_image_pairs(clean_dir, attacked_dir)
            for clean_dir, attacked_dir in zip(config.clean_image_dirs, config.attacked_image_dirs)
        ]
        if len(self.pairs_by_budget) != len(config.attack_budgets):
            raise ValueError(
                "attack_budgets must have the same length as clean_image_dirs and attacked_image_dirs"
            )

        common_ids = [key for _, _, key in self.pairs_by_budget[0]]
        common_id_set = set(common_ids)
        for pairs in self.pairs_by_budget[1:]:
            common_id_set &= {key for _, _, key in pairs}
        ordered_ids = [image_id for image_id in common_ids if image_id in common_id_set]
        if config.n_images is not None:
            ordered_ids = ordered_ids[: max(0, int(config.n_images))]
        if not ordered_ids:
            raise ValueError("No matched image pairs available for precomputed evaluation")

        self.image_ids = ordered_ids
        self.pair_lookup = [
            {image_id: (clean_path, attacked_path) for clean_path, attacked_path, image_id in pairs}
            for pairs in self.pairs_by_budget
        ]
        self.metrics_collector = MetricsCollector(
            attack_types=[config.attack_name],
            attack_budgets=config.attack_budgets,
            n_images=len(self.image_ids),
            stored_metrics=config.stored_metrics,
            source_image_ids=self.image_ids,
        )
        self._load_state_if_available()

    def _build_state_signature(self) -> Dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "attack_name": self.config.attack_name,
            "seed": self.config.seed,
            "attack_budgets": list(self.config.attack_budgets),
            "clean_image_dirs": list(self.config.clean_image_dirs),
            "attacked_image_dirs": list(self.config.attacked_image_dirs),
            "stored_metrics": list(self.config.stored_metrics),
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "cfg": self.config.cfg,
            "num_steps": self.config.num_steps,
            "n_images": self.config.n_images,
            "state_suffix": self.config.state_suffix,
        }

    def _build_state(self) -> Dict[str, Any]:
        results = {
            attack_type: {
                metric: tensor.detach().cpu()
                for metric, tensor in attack_results.items()
            }
            for attack_type, attack_results in self.metrics_collector.results.items()
        }
        return {
            "version": 1,
            "signature": self.state_signature,
            "source_image_ids": list(self.image_ids),
            "image_indices": list(range(len(self.image_ids))),
            "results": results,
            "restart_results": self.metrics_collector.restart_results,
            "location_results": self.metrics_collector.location_results,
        }

    def _load_state_if_available(self) -> None:
        state = self.results_manager.load_state(
            self.config.dataset,
            self.config.seed,
            suffix=self.config.state_suffix,
        )
        if state is None:
            return

        saved_signature = state.get("signature")
        if not self._is_state_compatible(saved_signature):
            print("Existing precomputed evaluation state is incompatible with the current configuration; starting fresh.")
            return

        saved_image_ids = state.get("source_image_ids")
        if not isinstance(saved_image_ids, list):
            print("Existing precomputed evaluation state does not include image IDs; starting fresh.")
            return

        current_image_ids = list(self.image_ids)
        shared_n_images = min(len(saved_image_ids), len(current_image_ids))
        if saved_image_ids[:shared_n_images] != current_image_ids[:shared_n_images]:
            print("Existing precomputed evaluation state was built from a different image ordering; starting fresh.")
            return

        results = state.get("results")
        restart_results = state.get("restart_results")
        location_results = state.get("location_results")
        if results is None or restart_results is None or location_results is None:
            print("Existing precomputed evaluation state is incomplete; starting fresh.")
            return

        reusable_budget_pairs = self._get_reusable_budget_pairs(saved_signature)
        if not reusable_budget_pairs:
            print("Existing precomputed evaluation state has no reusable budget overlap; starting fresh.")
            return

        self._merge_saved_state(
            results=results,
            restart_results=restart_results,
            location_results=location_results,
            shared_n_images=shared_n_images,
            reusable_budget_pairs=reusable_budget_pairs,
        )
        print(
            f"Reused {len(reusable_budget_pairs)}/{len(self.config.attack_budgets)} precomputed budget rows from saved state."
        )
        print(
            f"Resumed precomputed evaluation state from {self.results_manager.get_state_path(self.config.dataset, self.config.seed, self.config.state_suffix)}"
        )

    def _is_state_compatible(self, saved_signature: Any) -> bool:
        """Check whether a saved precomputed state can be partially merged."""
        if not isinstance(saved_signature, dict):
            return False

        keys_that_must_match = [
            "dataset",
            "attack_name",
            "seed",
            "stored_metrics",
            "device",
            "batch_size",
            "cfg",
            "num_steps",
            "state_suffix",
        ]
        for key in keys_that_must_match:
            if saved_signature.get(key) != self.state_signature.get(key):
                return False

        return True

    def _get_reusable_budget_pairs(self, saved_signature: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Return (current_budget_idx, saved_budget_idx) pairs that are safe to reuse."""
        saved_budgets = saved_signature.get("attack_budgets")
        saved_clean_dirs = saved_signature.get("clean_image_dirs")
        saved_attacked_dirs = saved_signature.get("attacked_image_dirs")
        if not isinstance(saved_budgets, list) or not isinstance(saved_clean_dirs, list) or not isinstance(saved_attacked_dirs, list):
            return []

        current_budgets = list(self.config.attack_budgets)
        current_clean_dirs = list(self.config.clean_image_dirs)
        current_attacked_dirs = list(self.config.attacked_image_dirs)

        used_saved_indices: set[int] = set()
        reusable_pairs: List[Tuple[int, int]] = []
        for current_idx, budget in enumerate(current_budgets):
            current_clean = current_clean_dirs[current_idx] if current_idx < len(current_clean_dirs) else None
            current_attacked = current_attacked_dirs[current_idx] if current_idx < len(current_attacked_dirs) else None
            match_idx: Optional[int] = None

            for saved_idx, saved_budget in enumerate(saved_budgets):
                if saved_idx in used_saved_indices:
                    continue
                if saved_budget != budget:
                    continue
                saved_clean = saved_clean_dirs[saved_idx] if saved_idx < len(saved_clean_dirs) else None
                saved_attacked = saved_attacked_dirs[saved_idx] if saved_idx < len(saved_attacked_dirs) else None
                if saved_clean != current_clean or saved_attacked != current_attacked:
                    continue
                match_idx = saved_idx
                break

            if match_idx is None:
                continue
            used_saved_indices.add(match_idx)
            reusable_pairs.append((current_idx, match_idx))

        return reusable_pairs

    def _merge_saved_state(
        self,
        results: Any,
        restart_results: Any,
        location_results: Any,
        shared_n_images: int,
        reusable_budget_pairs: List[Tuple[int, int]],
    ) -> None:
        """Copy overlap from a compatible precomputed state into current buffers."""
        if not isinstance(results, dict):
            return
        if not isinstance(restart_results, dict):
            return
        if not isinstance(location_results, dict):
            return

        attack_type = self.config.attack_name
        saved_attack_results = results.get(attack_type)
        if not isinstance(saved_attack_results, dict):
            return

        for metric in self.config.stored_metrics:
            saved_metric = saved_attack_results.get(metric)
            if not isinstance(saved_metric, torch.Tensor):
                continue
            current_metric = self.metrics_collector.results[attack_type][metric]
            image_limit = min(current_metric.shape[1], saved_metric.shape[1], shared_n_images)
            if image_limit == 0:
                continue
            for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
                if current_budget_idx >= current_metric.shape[0] or saved_budget_idx >= saved_metric.shape[0]:
                    continue
                current_metric[current_budget_idx, :image_limit] = (
                    saved_metric[saved_budget_idx, :image_limit].detach().cpu()
                )

        saved_restart_by_budget = restart_results.get(attack_type)
        saved_location_by_budget = location_results.get(attack_type)
        if not isinstance(saved_restart_by_budget, list) or not isinstance(saved_location_by_budget, list):
            return

        current_restart_by_budget = self.metrics_collector.restart_results[attack_type]
        current_location_by_budget = self.metrics_collector.location_results[attack_type]
        for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
            if current_budget_idx >= len(current_restart_by_budget) or current_budget_idx >= len(current_location_by_budget):
                continue
            if saved_budget_idx >= len(saved_restart_by_budget) or saved_budget_idx >= len(saved_location_by_budget):
                continue

            current_restart_by_image = current_restart_by_budget[current_budget_idx]
            current_location_by_image = current_location_by_budget[current_budget_idx]
            saved_restart_by_image = saved_restart_by_budget[saved_budget_idx]
            saved_location_by_image = saved_location_by_budget[saved_budget_idx]
            if not isinstance(saved_restart_by_image, list) or not isinstance(saved_location_by_image, list):
                continue

            image_limit = min(
                len(current_restart_by_image),
                len(current_location_by_image),
                len(saved_restart_by_image),
                len(saved_location_by_image),
                shared_n_images,
            )
            for image_idx in range(image_limit):
                if saved_restart_by_image[image_idx] is not None:
                    current_restart_by_image[image_idx] = saved_restart_by_image[image_idx]
                if saved_location_by_image[image_idx] is not None:
                    current_location_by_image[image_idx] = saved_location_by_image[image_idx]

    def save_state(self) -> None:
        self.results_manager.save_state(
            self._build_state(),
            self.config.dataset,
            self.config.seed,
            suffix=self.config.state_suffix,
        )

    def get_attack_configs(self, pending_only: bool = True) -> List[Tuple[str, int, int, Image.Image, Image.Image]]:
        attack_configs: List[Tuple[str, int, int, Image.Image, Image.Image]] = []
        for budget_idx, _ in enumerate(self.config.attack_budgets):
            for image_idx, image_id in enumerate(self.image_ids):
                if pending_only and self.metrics_collector.is_task_complete(self.config.attack_name, budget_idx, image_idx):
                    continue
                clean_path, attacked_path = self.pair_lookup[budget_idx][image_id]
                attack_configs.append((self.config.attack_name, budget_idx, image_idx, clean_path, attacked_path))
        return attack_configs

    def save_results(self) -> None:
        results = self.metrics_collector.get_results()
        self.results_manager.save_results(
            results[self.config.attack_name],
            self.config.dataset,
            self.config.attack_name,
            suffix=self.config.state_suffix,
        )
        self.results_manager.save_attack_args(
            self.config.attack_budgets,
            [
                {
                    "clean_images_dir": clean_dir,
                    "attacked_images_dir": attacked_dir,
                }
                for clean_dir, attacked_dir in zip(self.config.clean_image_dirs, self.config.attacked_image_dirs)
            ],
            self.config.dataset,
            suffix=self.config.state_suffix,
        )

    def save_run_config(self, run_config: Dict[str, Any], suffix: str = "") -> None:
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
            runner.save_state()
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
    attack_configs: Optional[List[Tuple[str, int, int, Image.Image]]] = None,
) -> None:
    """Run attacks sequentially."""
    from attacks.attacks import run_attack
    config = runner.config
    if attack_configs is None:
        attack_configs = runner.get_attack_configs(pending_only=True)
    total = len(attack_configs)
    
    pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")
    
    for attack_type, budget_idx, image_idx, image in attack_configs:
        eps = config.attack_budgets[budget_idx]
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
        runner.save_state()

        pbar.set_postfix(
            attack=attack_type,
            eps=f"{eps:.4f}",
            image=f"{image_idx+1}/{config.n_images}"
        )
        pbar.update(1)
    
    pbar.close()


def run_evaluation(runner: EvaluationRunner) -> None:
    """Execute evaluation with appropriate execution strategy."""
    attack_configs = runner.get_attack_configs(pending_only=True)
    if len(attack_configs) == 0:
        print("No pending evaluation tasks. Using existing saved state/results.")
        return

    if runner.config.parallel_workers > 1:
        parallel_evaluate_attacks(runner, attack_configs)
    else:
        sequential_evaluate_attacks(runner, attack_configs)


def run_precomputed_pair_evaluation(runner: PrecomputedPairEvaluationRunner) -> None:
    """Evaluate precomputed clean/attacked pairs using the shared metrics architecture."""
    attack_configs = runner.get_attack_configs(pending_only=True)
    if len(attack_configs) == 0:
        print("No pending precomputed evaluation tasks. Using existing saved state/results.")
        return

    pbar = tqdm_module.tqdm(total=len(attack_configs), desc="Evaluating precomputed pairs")
    for attack_type, budget_idx, image_idx, clean_path, attacked_path in attack_configs:
        with Image.open(clean_path) as clean_image_file:
            clean_image = clean_image_file.convert("RGB")
        with Image.open(attacked_path) as attacked_image_file:
            attacked_image = attacked_image_file.convert("RGB")

        eval_result = run_paired_pipeline_with_shared_noise(
            pipeline=runner.pipeline,
            source_image=clean_image,
            perturbed_image=attacked_image,
            batch_size=runner.config.batch_size,
            cfg=runner.config.cfg,
            num_steps=runner.config.num_steps,
            seed=int(runner.config.seed) + budget_idx * 100_000 + image_idx,
            device=runner.config.device,
        )

        attack_result = {
            "best_metrics": eval_result["metrics"],
            "best_restart": 0,
            "restart_evaluations": [
                {
                    "metrics": eval_result["metrics"],
                    "gps_source": eval_result["gps_source"],
                    "gps_perturbed": eval_result["gps_perturbed"],
                }
            ],
        }
        runner.metrics_collector.record_attack_result(
            attack_type,
            budget_idx,
            image_idx,
            attack_result,
        )
        runner.save_state()
        pbar.set_postfix(
            attack=attack_type,
            eps=f"{runner.config.attack_budgets[budget_idx]:.4f}",
            image=f"{image_idx + 1}/{len(runner.image_ids)}",
        )
        pbar.update(1)

    pbar.close()
