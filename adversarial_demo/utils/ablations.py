"""
Shared building blocks for the restart and sampling-steps ablations.

Both ablations can be produced either standalone (the ``evaluate-restarts`` /
``evaluate-sampling-steps`` commands) or as a by-product of ``evaluate-dataset``.
To keep a single source of truth, the JSON assembly and the per-perturbation
re-evaluation live here and are reused by every caller.

JSON shapes are kept byte-compatible with the plotting helpers in
``utils.plots`` (``plot_restarts_success`` / ``plot_sampling_steps_success_rate``)
and with ``merge_sampling_steps_results``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from utils.adversarial_utils import (
    add_perturbation_to_image,
    run_paired_pipeline_with_shared_noise,
)


def best_after_k(displacements: Sequence[float]) -> List[float]:
    """Running maximum: best displacement found after the first k restarts."""
    best: List[float] = []
    current = float("-inf")
    for d in displacements:
        current = max(current, float(d))
        best.append(current)
    return best


def _budget_key(budget: float) -> str:
    """Stable per-budget dict key shared by every ablation JSON."""
    return f"budget_{budget:.6f}"


# --------------------------------------------------------------------------- #
# Restart ablation
# --------------------------------------------------------------------------- #


def restart_displacements_from_results(
    restart_results: Dict[str, List[List[Optional[List[Dict[str, Any]]]]]],
    attack_types: Sequence[str],
    n_budgets: int,
    n_images: int,
    metric_key: str = "final_step_displacement_predicted",
) -> Dict[str, Dict[int, Dict[int, List[float]]]]:
    """Pull per-restart displacements out of ``MetricsCollector.restart_results``.

    ``restart_results[attack_type][budget_idx][image_idx]`` is the list of
    normalized per-restart evaluations recorded during training, in restart
    order. We read ``metric_key`` from each so the restart ablation reuses the
    exact numbers the main evaluation already computed (no retraining).
    """
    extracted: Dict[str, Dict[int, Dict[int, List[float]]]] = {
        at: {bi: {} for bi in range(n_budgets)} for at in attack_types
    }
    for at in attack_types:
        for bi in range(n_budgets):
            for ii in range(n_images):
                per_restart = restart_results[at][bi][ii] or []
                extracted[at][bi][ii] = [
                    float(r[metric_key])
                    for r in per_restart
                    if r is not None and r.get(metric_key) is not None
                ]
    return extracted


def build_restart_ablation_json(
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    image_ids: Sequence[str],
    per_restart_disps: Dict[str, Dict[int, Dict[int, List[float]]]],
    max_restarts: int,
) -> Dict[str, Any]:
    """Assemble the restart-ablation JSON consumed by ``plot_restarts_success``."""
    n_images = len(image_ids)
    json_results: Dict[str, Any] = {
        "dataset": dataset_name,
        "attack_types": list(attack_types),
        "attack_budgets": list(attack_budgets),
        "max_restarts": int(max_restarts),
        "n_images": n_images,
        "image_ids": list(image_ids),
        "results": {},
    }
    for at in attack_types:
        json_results["results"][at] = {}
        for budget_idx, budget in enumerate(attack_budgets):
            bkey = _budget_key(budget)
            json_results["results"][at][bkey] = {}
            for image_idx in range(n_images):
                disps = per_restart_disps[at][budget_idx].get(image_idx, [])
                json_results["results"][at][bkey][f"image_{image_idx}"] = {
                    "image_id": image_ids[image_idx],
                    "restart_displacements": disps,
                    "best_after_k": best_after_k(disps),
                }
    return json_results


# --------------------------------------------------------------------------- #
# Sampling-steps ablation
# --------------------------------------------------------------------------- #


def evaluate_delta_at_steps(
    pipeline,
    source_image,
    delta,
    eval_num_steps: Sequence[int],
    cfg: float = 10.0,
    batch_size: int = 128,
    seed: int = 1234,
    device: str = "cuda",
) -> Dict[int, float]:
    """Re-evaluate a single trained perturbation at several sampling-step counts.

    Returns ``{num_steps: final_step_displacement_km}``. Reuses the exact paired,
    shared-noise evaluation used everywhere else, so the only thing that varies
    across calls is the number of inference steps.
    """
    perturbed_image = add_perturbation_to_image(source_image, delta.to(device), pipeline)
    displacement_by_steps: Dict[int, float] = {}
    for num_steps in eval_num_steps:
        eval_result = run_paired_pipeline_with_shared_noise(
            pipeline=pipeline,
            source_image=source_image,
            perturbed_image=perturbed_image,
            batch_size=batch_size,
            cfg=cfg,
            num_steps=int(num_steps),
            seed=seed,
            device=device,
        )
        displacement_by_steps[int(num_steps)] = float(
            eval_result["metrics"]["final_step_displacement"]
        )
    return displacement_by_steps


def build_sampling_steps_json(
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    eval_num_steps: Sequence[int],
    success_rate_thresholds: Sequence[float],
    n_images: int,
    samples: Dict[str, Dict[int, Dict[int, List[float]]]],
) -> Dict[str, Any]:
    """Assemble the sampling-steps JSON consumed by ``plot_sampling_steps_success_rate``.

    ``samples[attack_type][budget_idx][num_steps]`` is the list of per-image
    final-step displacements (km) measured at that step count.
    """
    eval_num_steps = list(eval_num_steps)
    json_results: Dict[str, Any] = {
        "dataset": dataset_name,
        "attack_types": list(attack_types),
        "attack_budgets": list(attack_budgets),
        "eval_num_steps": eval_num_steps,
        "success_rate_thresholds_km": list(success_rate_thresholds),
        "n_images": int(n_images),
        "results": {},
    }
    for at in attack_types:
        json_results["results"][at] = {}
        for budget_idx, budget in enumerate(attack_budgets):
            bkey = _budget_key(budget)
            json_results["results"][at][bkey] = {}
            for num_steps in eval_num_steps:
                disps = samples[at][budget_idx].get(num_steps, [])
                success_rates = {
                    str(thr): float(np.mean([d > thr for d in disps])) if disps else float("nan")
                    for thr in success_rate_thresholds
                }
                json_results["results"][at][bkey][str(num_steps)] = {
                    "mean_displacement_km": float(np.mean(disps)) if disps else float("nan"),
                    "success_rates": success_rates,
                }
    return json_results
