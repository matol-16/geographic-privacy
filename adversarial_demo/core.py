"""
Core evaluation engine for adversarial attack experiments.

Responsibilities:
- Loading test images (``ImageLoader``)
- Saving/loading results, attack args, run config, and resumable state (``ResultsManager``)
- Collecting per-attack metrics, restart evaluations, and the optional
  sampling-steps ablation (``MetricsCollector``)
- Driving the evaluation loop, sequentially or in parallel, with resume support
  (``EvaluationRunner`` / ``PrecomputedPairEvaluationRunner``)

The two runners share all of their resume/state machinery through
``BaseEvaluationRunner``; each subclass only declares what makes a saved state
reusable (its signature keys and per-budget identity).
"""

from __future__ import annotations

import os
import yaml
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
import tqdm as tqdm_module

from utils.adversarial_metrics import trajectory_displacement
from utils.datasets import retrieve_yfcc_images, retrieve_osv_images
from utils.ablations import (
    evaluate_delta_across_models,
    evaluate_delta_at_steps,
    evaluate_delta_under_transforms,
)
from utils.adversarial_utils import collect_common_image_pairs, run_paired_pipeline_with_shared_noise

# Note: attacks imports are deferred inside the loop functions to avoid circular imports.


@dataclass
class EvaluationConfig:
    """Configuration for an attack evaluation run."""
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
    # Multi-node sharding: ``n_images`` is the size of the full seeded pool, and only
    # the ``[window_start:window_end]`` slice of that pool is loaded and evaluated by
    # this process. Defaults (0, None) load the whole pool, i.e. the single-GPU path.
    # The per-image buffers are sized to the loaded window; ``source_image_ids`` carry
    # the global identity so merge-shards can stitch windows back together.
    window_start: int = 0
    window_end: Optional[int] = None
    # Per-attack-type kwargs merged on top of the shared base only for the matching
    # attack (e.g. ACE's target_image / l2_target loss / alpha). Lets every attack
    # type run in a single evaluation without leaking settings into the others.
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None
    # Sampling-steps ablation (opt-in): re-evaluate the best perturbation of each
    # image at every step count in ``eval_num_steps`` during the same run.
    run_sampling_steps_ablation: bool = False
    eval_num_steps: Optional[List[int]] = None
    success_rate_thresholds: Optional[List[float]] = None
    # Robustness-to-transformation ablation (opt-in, per attack type): degrade the
    # best perturbation's protected image with JPEG compression / Gaussian blur and
    # re-evaluate at the baseline sampling-step count. Only runs for attack types in
    # ``robustness_attack_types`` (default: ["dtd"]); ``robustness_num_steps`` of None
    # falls back to each attack's ``restart_eval_num_steps`` (the baseline).
    run_robustness_ablation: bool = False
    robustness_attack_types: Optional[List[str]] = None
    robustness_jpeg_quality_factors: Optional[List[int]] = None
    robustness_gaussian_blur_sigmas: Optional[List[float]] = None
    robustness_num_steps: Optional[int] = None
    # Cross-model transfer ablation (opt-in): re-evaluate the best perturbation's
    # attacked image against every PLONK variant in ``model_transfer_types`` (model_type
    # config values, e.g. ["", "diffusion", "flow"]). The loaded pipelines are supplied
    # to the runner separately (``transfer_pipelines``); these fields only describe the
    # run for the state signature / JSON. ``model_transfer_num_steps`` of None falls back
    # to each attack's ``restart_eval_num_steps`` (the baseline), as for robustness.
    run_model_transfer_ablation: bool = False
    model_transfer_types: Optional[List[str]] = None
    model_transfer_num_steps: Optional[int] = None


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
    # Optional map {image_id -> (lat, lon)} of ground-truth GPS for the clean images.
    # When provided (e.g. GeoShield, which selects labelled dataset images), the runner
    # computes the true-position displacement metric just like the trainable attacks;
    # left None for arbitrary precomputed folders that have no labels (predicted-only).
    gps_by_id: Optional[Dict[str, Tuple[float, float]]] = None
    # Ablations the precomputed pairs (e.g. GeoShield) take part in. Both are eval-time
    # properties of the attacked image, so they apply unchanged to precomputed pairs.
    # (The restart ablation is intentionally absent: a precomputed attack has a single,
    # already-generated image, so there is nothing to restart.)
    run_sampling_steps_ablation: bool = False
    eval_num_steps: Optional[List[int]] = None
    run_robustness_ablation: bool = False
    robustness_jpeg_quality_factors: Optional[List[int]] = None
    robustness_gaussian_blur_sigmas: Optional[List[float]] = None
    robustness_num_steps: Optional[int] = None


class ImageLoader:
    """Unified image loading interface for different datasets."""

    @staticmethod
    def load_images(
        dataset: str,
        n_images: int,
        seed: int = 0,
        use_real_gps: bool = False,
        dataset_roots: Optional[Dict[str, str]] = None,
        window_start: int = 0,
        window_end: Optional[int] = None,
    ) -> Tuple[List[Image.Image], Optional[List[Tuple[float, float]]], List[str]]:
        """Load images, optional GPS coordinates, and stable dataset image IDs.

        ``n_images`` is the size of the full seeded pool; ``window_start``/``window_end``
        select the contiguous slice of that pool to load (defaults: the whole pool).
        """

        dataset_roots = dataset_roots or {}

        if dataset == "yfcc":
            return retrieve_yfcc_images(
                n_images_to_eval=n_images,
                seed=seed,
                use_real_gps=use_real_gps,
                local_dir=dataset_roots.get("yfcc"),
                window_start=window_start,
                window_end=window_end,
            )
        elif dataset == "osv":
            return retrieve_osv_images(
                n_images_to_eval=n_images,
                seed=seed,
                use_real_gps=use_real_gps,
                local_dir=dataset_roots.get("osv"),
                window_start=window_start,
                window_end=window_end,
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
    """Collects per-attack metrics, restart evaluations, and sampling-steps samples.

    Buffers are indexed ``[attack_type][budget_idx][image_idx]`` so a run can be
    resumed image-by-image. The main metric tensors are NaN-initialised and filled
    in as attacks complete.
    """

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
        # Best-restart training loss (attack-type-specific units/scale) per image,
        # used by the loss-vs-FSD ablation to relate optimization progress to
        # geolocation impact for loss-based attacks (dtd, training_loss, sampling).
        self.loss_results = {
            attack_type: torch.full((len(attack_budgets), n_images), float("nan"))
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
        # Optional sampling-steps ablation: per image, {num_steps: displacement_km}.
        self.sampling_steps_results: Dict[str, List[List[Optional[Dict[int, float]]]]] = {
            attack_type: [[None for _ in range(n_images)] for _ in attack_budgets]
            for attack_type in attack_types
        }
        # Optional robustness ablation: per image,
        # {"jpeg": {quality: disp_km}, "blur": {sigma: disp_km}}.
        self.robustness_results: Dict[str, List[List[Optional[Dict[str, Dict[float, float]]]]]] = {
            attack_type: [[None for _ in range(n_images)] for _ in attack_budgets]
            for attack_type in attack_types
        }
        # Optional cross-model transfer ablation: per image, {model_label: displacement_km}.
        self.model_transfer_results: Dict[str, List[List[Optional[Dict[str, float]]]]] = {
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

        final_loss = attack_result.get("final_loss")
        if final_loss is not None:
            self.loss_results[attack_type][budget_index, image_index] = float(final_loss)

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

    def record_sampling_steps(
        self,
        attack_type: str,
        budget_index: int,
        image_index: int,
        displacement_by_steps: Dict[int, float],
    ) -> None:
        """Record the sampling-steps ablation samples for one image."""
        self.sampling_steps_results[attack_type][budget_index][image_index] = dict(displacement_by_steps)

    def record_robustness(
        self,
        attack_type: str,
        budget_index: int,
        image_index: int,
        displacement_by_transform: Dict[str, Dict[float, float]],
    ) -> None:
        """Record the robustness ablation samples for one image."""
        self.robustness_results[attack_type][budget_index][image_index] = {
            "jpeg": dict(displacement_by_transform.get("jpeg", {})),
            "blur": dict(displacement_by_transform.get("blur", {})),
        }

    def record_model_transfer(
        self,
        attack_type: str,
        budget_index: int,
        image_index: int,
        displacement_by_model: Dict[str, float],
    ) -> None:
        """Record the cross-model transfer ablation samples for one image."""
        self.model_transfer_results[attack_type][budget_index][image_index] = dict(displacement_by_model)

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected results."""
        combined_results: Dict[str, Dict[str, Any]] = {}
        for attack_type in self.attack_types:
            attack_results: Dict[str, Any] = {
                metric: tensor for metric, tensor in self.results[attack_type].items()
            }
            attack_results["image_indices"] = list(range(self.n_images))
            attack_results["image_ids"] = self.source_image_ids
            attack_results["final_loss"] = self.loss_results[attack_type]
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


class BaseEvaluationRunner:
    """Shared resume/state/save machinery for evaluation runners.

    Subclasses set ``self.metrics_collector`` and the image accessors, then call
    ``self._load_state_if_available()`` at the end of their ``__init__``. They only
    need to declare what makes a saved state reusable: the signature dict
    (``_build_state_signature``), which keys must match exactly
    (``_signature_match_keys``), and how to identify a budget row for partial reuse
    (``_budget_identity_from_signature``).
    """

    state_kind: str = "evaluation"

    def __init__(self, config, pipeline):
        self.config = config
        self.pipeline = pipeline
        self.results_manager = ResultsManager(config.results_dir, config.plots_dir)
        self.state_signature = self._build_state_signature()

    # ---- Subclass hooks ---------------------------------------------------- #

    def _build_state_signature(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _signature_match_keys(self) -> List[str]:
        """Signature keys that must match exactly for a saved state to be reusable."""
        raise NotImplementedError

    def _budget_identity_from_signature(self, signature: Dict[str, Any], budget_idx: int) -> Any:
        """Identity of one budget row, used to match current vs saved budgets."""
        raise NotImplementedError

    @property
    def attack_types(self) -> List[str]:
        """Attack types tracked by the collector (used by save/merge)."""
        raise NotImplementedError

    @property
    def _state_image_ids(self) -> Optional[List[str]]:
        raise NotImplementedError

    @property
    def _n_images(self) -> int:
        raise NotImplementedError

    # ---- Shared state persistence ----------------------------------------- #

    def _build_state(self) -> Dict[str, Any]:
        """Collect the current incremental state for persistence."""
        mc = self.metrics_collector
        results = {
            attack_type: {
                metric: tensor.detach().cpu()
                for metric, tensor in attack_results.items()
            }
            for attack_type, attack_results in mc.results.items()
        }
        ids = self._state_image_ids
        return {
            "version": 1,
            "signature": self.state_signature,
            "source_image_ids": list(ids) if ids is not None else None,
            "image_indices": list(range(self._n_images)),
            "results": results,
            "restart_results": mc.restart_results,
            "location_results": mc.location_results,
            "sampling_steps_results": mc.sampling_steps_results,
            "robustness_results": mc.robustness_results,
            "model_transfer_results": mc.model_transfer_results,
        }

    def _state_file_suffix(self) -> str:
        """Suffix for the resumable-state filename (``..._eval_state<suffix>.pt``).

        Defaults to ``state_suffix``. Subclasses that may share a ``results_dir`` with
        another runner (e.g. the precomputed GeoShield runner sitting next to the
        trainable runner in the single-GPU ``evaluate-dataset``) override this to add the
        attack name, so the two never overwrite each other's state file.
        """
        return self.config.state_suffix

    def save_state(self) -> None:
        """Persist the current incremental evaluation state."""
        self.results_manager.save_state(
            self._build_state(),
            self.config.dataset,
            self.config.seed,
            suffix=self._state_file_suffix(),
        )

    def _load_state_if_available(self) -> None:
        """Restore progress from a previous interrupted run when compatible."""
        state = self.results_manager.load_state(
            self.config.dataset,
            self.config.seed,
            suffix=self._state_file_suffix(),
        )
        if state is None:
            return

        saved_signature = state.get("signature")
        if not self._is_state_compatible(saved_signature):
            print(f"Existing {self.state_kind} state is incompatible with the current configuration; starting fresh.")
            return

        saved_image_ids = state.get("source_image_ids")
        if not isinstance(saved_image_ids, list):
            print(f"Existing {self.state_kind} state does not include image IDs; starting fresh.")
            return

        current_image_ids = list(self._state_image_ids)
        shared_n_images = min(len(saved_image_ids), len(current_image_ids))
        if saved_image_ids[:shared_n_images] != current_image_ids[:shared_n_images]:
            print(f"Existing {self.state_kind} state was built from a different image ordering; starting fresh.")
            return

        results = state.get("results")
        restart_results = state.get("restart_results")
        location_results = state.get("location_results")
        if results is None or restart_results is None or location_results is None:
            print(f"Existing {self.state_kind} state is incomplete; starting fresh.")
            return

        reusable_budget_pairs = self._get_reusable_budget_pairs(saved_signature)
        if not reusable_budget_pairs:
            print(f"Existing {self.state_kind} state has no reusable budget overlap; starting fresh.")
            return

        self._merge_saved_state(
            results=results,
            restart_results=restart_results,
            location_results=location_results,
            sampling_steps_results=state.get("sampling_steps_results"),
            robustness_results=state.get("robustness_results"),
            model_transfer_results=state.get("model_transfer_results"),
            shared_n_images=shared_n_images,
            reusable_budget_pairs=reusable_budget_pairs,
        )
        print(
            f"Reused {len(reusable_budget_pairs)}/{len(self.config.attack_budgets)} budget rows from saved {self.state_kind} state."
        )
        print(
            f"Resumed {self.state_kind} state from "
            f"{self.results_manager.get_state_path(self.config.dataset, self.config.seed, self._state_file_suffix())}"
        )

    def _is_state_compatible(self, saved_signature: Any) -> bool:
        """Check whether a saved state can be partially merged into the current run."""
        if not isinstance(saved_signature, dict):
            return False
        for key in self._signature_match_keys():
            if saved_signature.get(key) != self.state_signature.get(key):
                return False
        return True

    def _get_reusable_budget_pairs(self, saved_signature: Dict[str, Any]) -> List[Tuple[int, int]]:
        """Return (current_budget_idx, saved_budget_idx) pairs that are safe to reuse."""
        saved_budgets = saved_signature.get("attack_budgets")
        if not isinstance(saved_budgets, list):
            return []

        current_budgets = list(self.config.attack_budgets)
        used_saved_indices: set[int] = set()
        reusable_pairs: List[Tuple[int, int]] = []
        for current_idx, budget in enumerate(current_budgets):
            current_id = self._budget_identity_from_signature(self.state_signature, current_idx)
            match_idx: Optional[int] = None
            for saved_idx, saved_budget in enumerate(saved_budgets):
                if saved_idx in used_saved_indices or saved_budget != budget:
                    continue
                if self._budget_identity_from_signature(saved_signature, saved_idx) != current_id:
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
        sampling_steps_results: Any,
        robustness_results: Any,
        model_transfer_results: Any,
        shared_n_images: int,
        reusable_budget_pairs: List[Tuple[int, int]],
    ) -> None:
        """Copy overlap from a compatible saved state into current in-memory buffers."""
        if not isinstance(results, dict) or not isinstance(restart_results, dict) or not isinstance(location_results, dict):
            return

        mc = self.metrics_collector
        for attack_type in self.attack_types:
            saved_attack_results = results.get(attack_type)
            if isinstance(saved_attack_results, dict):
                for metric in self.config.stored_metrics:
                    saved_metric = saved_attack_results.get(metric)
                    if not isinstance(saved_metric, torch.Tensor):
                        continue
                    current_metric = mc.results[attack_type][metric]
                    image_limit = min(current_metric.shape[1], saved_metric.shape[1], shared_n_images)
                    if image_limit == 0:
                        continue
                    for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
                        if current_budget_idx >= current_metric.shape[0] or saved_budget_idx >= saved_metric.shape[0]:
                            continue
                        current_metric[current_budget_idx, :image_limit] = (
                            saved_metric[saved_budget_idx, :image_limit].detach().cpu()
                        )

            self._merge_per_image_lists(
                attack_type=attack_type,
                restart_results=restart_results,
                location_results=location_results,
                sampling_steps_results=sampling_steps_results,
                robustness_results=robustness_results,
                model_transfer_results=model_transfer_results,
                shared_n_images=shared_n_images,
                reusable_budget_pairs=reusable_budget_pairs,
            )

    def _merge_per_image_lists(
        self,
        attack_type: str,
        restart_results: Any,
        location_results: Any,
        sampling_steps_results: Any,
        robustness_results: Any,
        model_transfer_results: Any,
        shared_n_images: int,
        reusable_budget_pairs: List[Tuple[int, int]],
    ) -> None:
        """Copy the per-image restart / location / sampling-steps / robustness / model-transfer buffers for one attack type."""
        mc = self.metrics_collector
        saved_restart_by_budget = restart_results.get(attack_type)
        saved_location_by_budget = location_results.get(attack_type)
        if not isinstance(saved_restart_by_budget, list) or not isinstance(saved_location_by_budget, list):
            return
        saved_steps_by_budget = (
            sampling_steps_results.get(attack_type) if isinstance(sampling_steps_results, dict) else None
        )
        saved_robustness_by_budget = (
            robustness_results.get(attack_type) if isinstance(robustness_results, dict) else None
        )
        saved_model_transfer_by_budget = (
            model_transfer_results.get(attack_type) if isinstance(model_transfer_results, dict) else None
        )

        current_restart_by_budget = mc.restart_results[attack_type]
        current_location_by_budget = mc.location_results[attack_type]
        current_steps_by_budget = mc.sampling_steps_results[attack_type]
        current_robustness_by_budget = mc.robustness_results[attack_type]
        current_model_transfer_by_budget = mc.model_transfer_results[attack_type]
        for current_budget_idx, saved_budget_idx in reusable_budget_pairs:
            if current_budget_idx >= len(current_restart_by_budget) or current_budget_idx >= len(current_location_by_budget):
                continue
            if saved_budget_idx >= len(saved_restart_by_budget) or saved_budget_idx >= len(saved_location_by_budget):
                continue

            current_restart_by_image = current_restart_by_budget[current_budget_idx]
            current_location_by_image = current_location_by_budget[current_budget_idx]
            current_steps_by_image = current_steps_by_budget[current_budget_idx]
            current_robustness_by_image = current_robustness_by_budget[current_budget_idx]
            current_model_transfer_by_image = current_model_transfer_by_budget[current_budget_idx]
            saved_restart_by_image = saved_restart_by_budget[saved_budget_idx]
            saved_location_by_image = saved_location_by_budget[saved_budget_idx]
            if not isinstance(saved_restart_by_image, list) or not isinstance(saved_location_by_image, list):
                continue
            saved_steps_by_image = (
                saved_steps_by_budget[saved_budget_idx]
                if isinstance(saved_steps_by_budget, list) and saved_budget_idx < len(saved_steps_by_budget)
                else None
            )
            saved_robustness_by_image = (
                saved_robustness_by_budget[saved_budget_idx]
                if isinstance(saved_robustness_by_budget, list) and saved_budget_idx < len(saved_robustness_by_budget)
                else None
            )
            saved_model_transfer_by_image = (
                saved_model_transfer_by_budget[saved_budget_idx]
                if isinstance(saved_model_transfer_by_budget, list) and saved_budget_idx < len(saved_model_transfer_by_budget)
                else None
            )

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
                if (
                    isinstance(saved_steps_by_image, list)
                    and image_idx < len(saved_steps_by_image)
                    and saved_steps_by_image[image_idx] is not None
                ):
                    current_steps_by_image[image_idx] = saved_steps_by_image[image_idx]
                if (
                    isinstance(saved_robustness_by_image, list)
                    and image_idx < len(saved_robustness_by_image)
                    and saved_robustness_by_image[image_idx] is not None
                ):
                    current_robustness_by_image[image_idx] = saved_robustness_by_image[image_idx]
                if (
                    isinstance(saved_model_transfer_by_image, list)
                    and image_idx < len(saved_model_transfer_by_image)
                    and saved_model_transfer_by_image[image_idx] is not None
                ):
                    current_model_transfer_by_image[image_idx] = saved_model_transfer_by_image[image_idx]

    def save_run_config(self, run_config: Dict[str, Any], suffix: str = "") -> None:
        """Save the resolved experiment configuration used for the run."""
        self.results_manager.save_run_config(run_config, self.config.dataset, suffix=suffix)


class EvaluationRunner(BaseEvaluationRunner):
    """Runs attacks over a dataset, with resume support and an optional ablation."""

    state_kind = "evaluation"

    def __init__(self, config: EvaluationConfig, pipeline, transfer_pipelines=None):
        super().__init__(config, pipeline)

        # Loaded PLONK variants for the cross-model transfer ablation, mapping label ->
        # pipeline (see utils.ablations.model_type_label). Includes the attack pipeline so
        # the attacked variant is part of the comparison. Empty unless the ablation is on.
        self.transfer_pipelines: Dict[str, Any] = transfer_pipelines or {}

        # Load images (only this process's window of the seeded pool; defaults to all).
        print(f"Loading {config.n_images} images from {config.dataset} dataset...")
        self.source_images, self.source_gps, self.source_image_ids = ImageLoader.load_images(
            dataset=config.dataset,
            n_images=config.n_images,
            seed=config.seed,
            use_real_gps=config.use_real_gps,
            dataset_roots=config.dataset_roots,
            window_start=config.window_start,
            window_end=config.window_end,
        )
        # The collector is sized to the loaded window, not the full pool. For the
        # single-GPU path (no window) this equals config.n_images.
        self.n_images = len(self.source_images)

        self.metrics_collector = MetricsCollector(
            attack_types=config.attack_types,
            attack_budgets=config.attack_budgets,
            n_images=self.n_images,
            stored_metrics=config.stored_metrics,
            source_gps=self.source_gps,
            source_image_ids=self.source_image_ids,
        )
        self._load_state_if_available()

    # ---- State hooks ------------------------------------------------------- #

    def _build_state_signature(self) -> Dict[str, Any]:
        return {
            "dataset": self.config.dataset,
            "seed": self.config.seed,
            "attack_types": list(self.config.attack_types),
            "attack_budgets": list(self.config.attack_budgets),
            "attack_kwargs": self.config.attack_kwargs,
            "n_images": self.config.n_images,
            "window_start": self.config.window_start,
            "window_end": self.config.window_end,
            "stored_metrics": list(self.config.stored_metrics),
            "use_real_gps": self.config.use_real_gps,
            "dataset_roots": self.config.dataset_roots or {},
            "state_suffix": self.config.state_suffix,
            "run_sampling_steps_ablation": self.config.run_sampling_steps_ablation,
            "eval_num_steps": list(self.config.eval_num_steps) if self.config.eval_num_steps else None,
            "attack_type_kwargs": self.config.attack_type_kwargs or {},
            "run_robustness_ablation": self.config.run_robustness_ablation,
            "robustness_attack_types": list(self.config.robustness_attack_types) if self.config.robustness_attack_types else None,
            "robustness_jpeg_quality_factors": list(self.config.robustness_jpeg_quality_factors) if self.config.robustness_jpeg_quality_factors else None,
            "robustness_gaussian_blur_sigmas": list(self.config.robustness_gaussian_blur_sigmas) if self.config.robustness_gaussian_blur_sigmas else None,
            "robustness_num_steps": self.config.robustness_num_steps,
            "run_model_transfer_ablation": self.config.run_model_transfer_ablation,
            "model_transfer_types": list(self.config.model_transfer_types) if self.config.model_transfer_types else None,
            "model_transfer_num_steps": self.config.model_transfer_num_steps,
        }

    def _signature_match_keys(self) -> List[str]:
        # window_start/window_end are intentionally NOT matched: a shard always uses its
        # own results_dir, and _load_state_if_available already rejects a state whose
        # source_image_ids prefix differs (which any different window would). Keeping
        # them out preserves resume compatibility with pre-window state files.
        return [
            "dataset",
            "seed",
            "stored_metrics",
            "use_real_gps",
            "dataset_roots",
            "state_suffix",
            "run_sampling_steps_ablation",
            "eval_num_steps",
            "attack_type_kwargs",
            "run_robustness_ablation",
            "robustness_attack_types",
            "robustness_jpeg_quality_factors",
            "robustness_gaussian_blur_sigmas",
            "robustness_num_steps",
            "run_model_transfer_ablation",
            "model_transfer_types",
            "model_transfer_num_steps",
        ]

    def _budget_identity_from_signature(self, signature: Dict[str, Any], budget_idx: int) -> Any:
        kwargs = signature.get("attack_kwargs")
        if isinstance(kwargs, list) and budget_idx < len(kwargs):
            return kwargs[budget_idx]
        return None

    @property
    def attack_types(self) -> List[str]:
        return list(self.config.attack_types)

    @property
    def _state_image_ids(self) -> Optional[List[str]]:
        return self.source_image_ids

    @property
    def _n_images(self) -> int:
        # The loaded window size (== config.n_images on the single-GPU path).
        return self.n_images

    # ---- Task scheduling --------------------------------------------------- #

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

    # ---- Per-attack kwargs ------------------------------------------------- #

    def merge_attack_kwargs(self, attack_type: str, budget_idx: int) -> Dict[str, Any]:
        """Shared per-budget kwargs with the per-attack-type overrides applied on top."""
        kwargs = dict(self.config.attack_kwargs[budget_idx])
        kwargs.update((self.config.attack_type_kwargs or {}).get(attack_type, {}))
        return kwargs

    # ---- Sampling-steps ablation ------------------------------------------ #

    def compute_sampling_steps_samples(
        self,
        attack_type: str,
        budget_idx: int,
        image: Image.Image,
        result: Dict[str, Any],
    ) -> Optional[Dict[int, float]]:
        """Re-evaluate the best perturbation at every ``eval_num_steps`` count.

        Returns ``{num_steps: displacement_km}`` or ``None`` when the ablation is
        disabled or no usable delta is available. GPU work only; recording into the
        collector is done separately so it can stay on the main thread.
        """
        if not self.config.run_sampling_steps_ablation:
            return None
        eval_num_steps = self.config.eval_num_steps or []
        if not eval_num_steps:
            return None
        delta = result.get("delta")
        if delta is None:
            return None

        kwargs = self.merge_attack_kwargs(attack_type, budget_idx)
        return evaluate_delta_at_steps(
            pipeline=self.pipeline,
            source_image=image,
            delta=delta,
            eval_num_steps=eval_num_steps,
            cfg=float(kwargs.get("restart_eval_cfg", 10.0)),
            batch_size=int(kwargs.get("restart_eval_batch_size", 128)),
            # Constant eval seed (matches the standalone evaluate_sampling_steps path):
            # the shared evaluation noise is identical for clean vs perturbed, so the
            # paired displacement is comparable across images and step counts.
            seed=int(self.config.seed),
            device=str(kwargs.get("device", "cuda")),
        )

    # ---- Robustness-to-transformation ablation ----------------------------- #

    def compute_robustness_samples(
        self,
        attack_type: str,
        budget_idx: int,
        image_idx: int,
        image: Image.Image,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, Dict[float, Dict[str, Optional[float]]]]]:
        """Degrade the best perturbation's protected image and re-evaluate.

        Returns ``{"jpeg": {quality: {"predicted": .., "true": ..}}, "blur": {...}}`` or
        ``None`` when the ablation is disabled, the attack type is not opted in, or no
        usable delta is available. The ``"true"`` displacement (vs ground-truth GPS) is
        only filled when the dataset provides labels. Re-evaluation uses the baseline
        sampling-step count (``robustness_num_steps``, defaulting to the attack's
        ``restart_eval_num_steps``) for every transform -- only the transform strength
        varies. GPU work only; recording into the collector is done separately so it can
        stay on the main thread (matching the sampling-steps ablation).
        """
        if not self.config.run_robustness_ablation:
            return None
        opted_in = self.config.robustness_attack_types or []
        if attack_type not in opted_in:
            return None
        jpeg_quality_factors = self.config.robustness_jpeg_quality_factors or []
        gaussian_blur_sigmas = self.config.robustness_gaussian_blur_sigmas or []
        if not jpeg_quality_factors and not gaussian_blur_sigmas:
            return None
        delta = result.get("delta")
        if delta is None:
            return None

        kwargs = self.merge_attack_kwargs(attack_type, budget_idx)
        # Baseline sampling steps: explicit override, else the attack's eval steps.
        num_steps = self.config.robustness_num_steps
        if num_steps is None:
            num_steps = kwargs.get("restart_eval_num_steps")
        true_gps = self.source_gps[image_idx] if self.source_gps is not None else None
        return evaluate_delta_under_transforms(
            pipeline=self.pipeline,
            source_image=image,
            delta=delta,
            jpeg_quality_factors=jpeg_quality_factors,
            gaussian_blur_sigmas=gaussian_blur_sigmas,
            cfg=float(kwargs.get("restart_eval_cfg", 10.0)),
            batch_size=int(kwargs.get("restart_eval_batch_size", 128)),
            # Constant eval seed (as in the sampling-steps ablation): identical shared
            # noise for clean vs perturbed keeps the paired displacement comparable.
            seed=int(self.config.seed),
            device=str(kwargs.get("device", "cuda")),
            num_steps=int(num_steps) if num_steps is not None else None,
            true_gps=true_gps,
        )

    # ---- Cross-model transfer ablation ------------------------------------- #

    def compute_model_transfer_samples(
        self,
        attack_type: str,
        budget_idx: int,
        image: Image.Image,
        result: Dict[str, Any],
    ) -> Optional[Dict[str, float]]:
        """Re-evaluate the best perturbation's attacked image against every model variant.

        Returns ``{model_label: displacement_km}`` or ``None`` when the ablation is
        disabled, no transfer pipelines were supplied, or no usable delta is available.
        The attacked image is reconstructed once (with the attack pipeline) and evaluated
        against each loaded variant at a single baseline step count
        (``model_transfer_num_steps``, defaulting to the attack's ``restart_eval_num_steps``)
        so the only thing that varies across the recorded values is the model. GPU work
        only; recording into the collector is done separately so it can stay on the main
        thread (matching the other ablations).
        """
        if not self.config.run_model_transfer_ablation:
            return None
        if not self.transfer_pipelines:
            return None
        delta = result.get("delta")
        if delta is None:
            return None

        kwargs = self.merge_attack_kwargs(attack_type, budget_idx)
        num_steps = self.config.model_transfer_num_steps
        if num_steps is None:
            num_steps = kwargs.get("restart_eval_num_steps")
        return evaluate_delta_across_models(
            attack_pipeline=self.pipeline,
            transfer_pipelines=self.transfer_pipelines,
            source_image=image,
            delta=delta,
            cfg=float(kwargs.get("restart_eval_cfg", 10.0)),
            batch_size=int(kwargs.get("restart_eval_batch_size", 128)),
            # Constant eval seed (as in the other ablations): identical shared noise for
            # clean vs perturbed keeps the paired displacement comparable across models.
            seed=int(self.config.seed),
            device=str(kwargs.get("device", "cuda")),
            num_steps=int(num_steps) if num_steps is not None else None,
        )


class PrecomputedPairEvaluationRunner(BaseEvaluationRunner):
    """Evaluation runner for precomputed clean/attacked image pairs."""

    state_kind = "precomputed evaluation"

    def __init__(self, config: PrecomputedPairEvaluationConfig, pipeline):
        super().__init__(config, pipeline)

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
        # Align ground-truth GPS to the matched image ids (None when no labels provided),
        # so the shared MetricsCollector can report the true-position displacement metric.
        self.source_gps = self._build_source_gps(config.gps_by_id, self.image_ids)
        self.metrics_collector = MetricsCollector(
            attack_types=[config.attack_name],
            attack_budgets=config.attack_budgets,
            n_images=len(self.image_ids),
            stored_metrics=config.stored_metrics,
            source_gps=self.source_gps,
            source_image_ids=self.image_ids,
        )
        self._load_state_if_available()

    @staticmethod
    def _build_source_gps(
        gps_by_id: Optional[Dict[str, Tuple[float, float]]],
        image_ids: List[str],
    ) -> Optional[List[Optional[Tuple[float, float]]]]:
        """Resolve per-image ground-truth GPS aligned to ``image_ids``.

        Pair keys may be full filenames ("123.jpg") while the GPS map may be keyed by
        filename, basename, or stem; we index by all three so GeoShield's filename keys
        match. Returns None when no GPS resolves (keeps the path predicted-only).
        """
        if not gps_by_id:
            return None
        normalized: Dict[str, Tuple[float, float]] = {}
        for key, gps in gps_by_id.items():
            k = str(key)
            normalized.setdefault(k, gps)
            normalized.setdefault(os.path.basename(k), gps)
            normalized.setdefault(os.path.splitext(os.path.basename(k))[0], gps)
        resolved: List[Optional[Tuple[float, float]]] = []
        for image_id in image_ids:
            s = str(image_id)
            resolved.append(
                normalized.get(s)
                or normalized.get(os.path.basename(s))
                or normalized.get(os.path.splitext(os.path.basename(s))[0])
            )
        if all(g is None for g in resolved):
            return None
        return resolved

    # ---- State hooks ------------------------------------------------------- #

    def _state_file_suffix(self) -> str:
        # Include the attack name so a precomputed run (e.g. GeoShield) never overwrites
        # the trainable run's ``..._eval_state.pt`` when they share a results_dir (the
        # single-GPU evaluate-dataset case).
        return f"_{self.config.attack_name}{self.config.state_suffix}"

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
            "run_sampling_steps_ablation": self.config.run_sampling_steps_ablation,
            "eval_num_steps": list(self.config.eval_num_steps) if self.config.eval_num_steps else None,
            "run_robustness_ablation": self.config.run_robustness_ablation,
            "robustness_jpeg_quality_factors": list(self.config.robustness_jpeg_quality_factors) if self.config.robustness_jpeg_quality_factors else None,
            "robustness_gaussian_blur_sigmas": list(self.config.robustness_gaussian_blur_sigmas) if self.config.robustness_gaussian_blur_sigmas else None,
            "robustness_num_steps": self.config.robustness_num_steps,
        }

    def _signature_match_keys(self) -> List[str]:
        return [
            "dataset",
            "attack_name",
            "seed",
            "stored_metrics",
            "device",
            "batch_size",
            "cfg",
            "num_steps",
            "state_suffix",
            "run_sampling_steps_ablation",
            "eval_num_steps",
            "run_robustness_ablation",
            "robustness_jpeg_quality_factors",
            "robustness_gaussian_blur_sigmas",
            "robustness_num_steps",
        ]

    def _budget_identity_from_signature(self, signature: Dict[str, Any], budget_idx: int) -> Any:
        clean_dirs = signature.get("clean_image_dirs")
        attacked_dirs = signature.get("attacked_image_dirs")
        clean = clean_dirs[budget_idx] if isinstance(clean_dirs, list) and budget_idx < len(clean_dirs) else None
        attacked = attacked_dirs[budget_idx] if isinstance(attacked_dirs, list) and budget_idx < len(attacked_dirs) else None
        return (clean, attacked)

    @property
    def attack_types(self) -> List[str]:
        return [self.config.attack_name]

    @property
    def _state_image_ids(self) -> Optional[List[str]]:
        return self.image_ids

    @property
    def _n_images(self) -> int:
        return len(self.image_ids)

    # ---- Task scheduling --------------------------------------------------- #

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


def parallel_evaluate_attacks(
    runner: EvaluationRunner,
    attack_configs: List[Tuple[str, int, int, Image.Image]],  # (attack_type, budget_idx, image_idx, image)
) -> None:
    """Run attacks in parallel with a thread pool (each worker on its own CUDA stream)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from attacks.attacks import run_attack

    config = runner.config
    total = len(attack_configs)

    pbar = tqdm_module.tqdm(total=total, desc="Evaluating attacks")

    def _evaluate_task(attack_type: str, budget_idx: int, image_idx: int, image: Image.Image):
        """Worker task: train the attack and (optionally) run the sampling-steps ablation."""
        eps = config.attack_budgets[budget_idx]
        kwargs = runner.merge_attack_kwargs(attack_type, budget_idx)

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
                steps_samples = runner.compute_sampling_steps_samples(attack_type, budget_idx, image, result)
                robustness_samples = runner.compute_robustness_samples(attack_type, budget_idx, image_idx, image, result)
                model_transfer_samples = runner.compute_model_transfer_samples(attack_type, budget_idx, image, result)
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
            steps_samples = runner.compute_sampling_steps_samples(attack_type, budget_idx, image, result)
            robustness_samples = runner.compute_robustness_samples(attack_type, budget_idx, image_idx, image, result)
            model_transfer_samples = runner.compute_model_transfer_samples(attack_type, budget_idx, image, result)

        return attack_type, budget_idx, image_idx, result, steps_samples, robustness_samples, model_transfer_samples

    with ThreadPoolExecutor(max_workers=config.parallel_workers) as executor:
        futures = [
            executor.submit(_evaluate_task, at, bi, ii, img)
            for at, bi, ii, img in attack_configs
        ]

        for future in as_completed(futures):
            attack_type, budget_idx, image_idx, result, steps_samples, robustness_samples, model_transfer_samples = future.result()
            runner.metrics_collector.record_attack_result(
                attack_type,
                budget_idx,
                image_idx,
                result,
            )
            if steps_samples is not None:
                runner.metrics_collector.record_sampling_steps(
                    attack_type, budget_idx, image_idx, steps_samples
                )
            if robustness_samples is not None:
                runner.metrics_collector.record_robustness(
                    attack_type, budget_idx, image_idx, robustness_samples
                )
            if model_transfer_samples is not None:
                runner.metrics_collector.record_model_transfer(
                    attack_type, budget_idx, image_idx, model_transfer_samples
                )
            runner.save_state()
            eps = config.attack_budgets[budget_idx]
            pbar.set_postfix(
                attack=attack_type,
                eps=f"{eps:.4f}",
                image=f"{image_idx+1}/{runner.n_images}"
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
        kwargs = runner.merge_attack_kwargs(attack_type, budget_idx)

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
        steps_samples = runner.compute_sampling_steps_samples(attack_type, budget_idx, image, result)
        if steps_samples is not None:
            runner.metrics_collector.record_sampling_steps(
                attack_type, budget_idx, image_idx, steps_samples
            )
        robustness_samples = runner.compute_robustness_samples(attack_type, budget_idx, image_idx, image, result)
        if robustness_samples is not None:
            runner.metrics_collector.record_robustness(
                attack_type, budget_idx, image_idx, robustness_samples
            )
        model_transfer_samples = runner.compute_model_transfer_samples(attack_type, budget_idx, image, result)
        if model_transfer_samples is not None:
            runner.metrics_collector.record_model_transfer(
                attack_type, budget_idx, image_idx, model_transfer_samples
            )
        runner.save_state()

        pbar.set_postfix(
            attack=attack_type,
            eps=f"{eps:.4f}",
            image=f"{image_idx+1}/{runner.n_images}"
        )
        pbar.update(1)

    pbar.close()


def run_evaluation(runner: EvaluationRunner) -> None:
    """Execute evaluation with the appropriate execution strategy."""
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

        # Eval-time ablations on the already-attacked image (same helpers as the
        # trainable path, but fed the perturbed image directly instead of a delta). Both
        # use a constant eval seed so the shared clean/perturbed noise stays comparable
        # across step counts / transforms, matching the trainable ablations.
        cfg = runner.config
        if cfg.run_sampling_steps_ablation and cfg.eval_num_steps:
            from utils.ablations import evaluate_pair_at_steps
            steps_samples = evaluate_pair_at_steps(
                pipeline=runner.pipeline,
                source_image=clean_image,
                perturbed_image=attacked_image,
                eval_num_steps=cfg.eval_num_steps,
                cfg=cfg.cfg,
                batch_size=cfg.batch_size,
                seed=int(cfg.seed),
                device=cfg.device,
            )
            runner.metrics_collector.record_sampling_steps(
                attack_type, budget_idx, image_idx, steps_samples
            )

        if cfg.run_robustness_ablation and (
            cfg.robustness_jpeg_quality_factors or cfg.robustness_gaussian_blur_sigmas
        ):
            from utils.ablations import evaluate_pair_under_transforms
            true_gps = runner.source_gps[image_idx] if runner.source_gps is not None else None
            robustness_samples = evaluate_pair_under_transforms(
                pipeline=runner.pipeline,
                source_image=clean_image,
                perturbed_image=attacked_image,
                jpeg_quality_factors=cfg.robustness_jpeg_quality_factors or [],
                gaussian_blur_sigmas=cfg.robustness_gaussian_blur_sigmas or [],
                cfg=cfg.cfg,
                batch_size=cfg.batch_size,
                seed=int(cfg.seed),
                device=cfg.device,
                num_steps=cfg.robustness_num_steps if cfg.robustness_num_steps is not None else cfg.num_steps,
                true_gps=true_gps,
            )
            runner.metrics_collector.record_robustness(
                attack_type, budget_idx, image_idx, robustness_samples
            )

        runner.save_state()
        pbar.set_postfix(
            attack=attack_type,
            eps=f"{runner.config.attack_budgets[budget_idx]:.4f}",
            image=f"{image_idx + 1}/{len(runner.image_ids)}",
        )
        pbar.update(1)

    pbar.close()
