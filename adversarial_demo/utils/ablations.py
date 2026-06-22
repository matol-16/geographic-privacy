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

import io
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageFilter

from utils.adversarial_metrics import final_prediction_distance_to_point
from utils.adversarial_utils import (
    add_perturbation_to_image,
    run_paired_pipeline_with_shared_noise,
)

# Metric streams recorded by the robustness ablation: displacement of the
# (transformed) protected prediction from the clean prediction ("predicted") and
# from the ground-truth GPS ("true"), mirroring the two metrics the main eval keeps.
ROBUSTNESS_METRICS = ("predicted", "true")


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


def evaluate_pair_at_steps(
    pipeline,
    source_image,
    perturbed_image,
    eval_num_steps: Sequence[int],
    cfg: float = 10.0,
    batch_size: int = 128,
    seed: int = 1234,
    device: str = "cuda",
) -> Dict[int, float]:
    """Evaluate an already-built clean/perturbed image pair at several step counts.

    Returns ``{num_steps: final_step_displacement_km}``. The perturbed image is taken
    as-is, so this serves precomputed pairs (e.g. GeoShield, read from disk) where there
    is no ``delta`` tensor to reconstruct from -- the step count is the only thing that
    varies across calls, exactly as in the trainable path.
    """
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
    return evaluate_pair_at_steps(
        pipeline=pipeline,
        source_image=source_image,
        perturbed_image=perturbed_image,
        eval_num_steps=eval_num_steps,
        cfg=cfg,
        batch_size=batch_size,
        seed=seed,
        device=device,
    )


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


# --------------------------------------------------------------------------- #
# Robustness-to-transformation ablation (JPEG compression / Gaussian blur)
# --------------------------------------------------------------------------- #
#
# Mirrors GeoShield's Fig. 6: each trained perturbation's protected image is
# degraded by JPEG compression or Gaussian blur and re-evaluated, to check that
# the protection survives the lossy transforms common to real image sharing.
# Unlike the sampling-steps ablation, this re-evaluates at a single (baseline)
# sampling-step count -- only the transform strength varies.


def _jpeg_key(quality: float) -> str:
    """Stable per-JPEG-quality dict key (e.g. 30 -> ``"30"``)."""
    return str(int(quality))


def _blur_key(sigma: float) -> str:
    """Stable per-blur-sigma dict key (e.g. 2.0 -> ``"2"``, 2.5 -> ``"2.5"``)."""
    return f"{float(sigma):g}"


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    """Round-trip a PIL image through JPEG at the given quality factor."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """Gaussian-blur a PIL image. ``sigma <= 0`` returns an unmodified copy.

    PIL's ``GaussianBlur`` radius is the Gaussian standard deviation, matching
    GeoShield's blur-sigma axis directly.
    """
    if float(sigma) <= 0.0:
        return image.copy()
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


def evaluate_pair_under_transforms(
    pipeline,
    source_image,
    perturbed_image,
    jpeg_quality_factors: Sequence[int],
    gaussian_blur_sigmas: Sequence[float],
    cfg: float = 10.0,
    batch_size: int = 128,
    seed: int = 1234,
    device: str = "cuda",
    num_steps: Optional[int] = None,
    true_gps: Optional[Sequence[float]] = None,
) -> Dict[str, Dict[float, Dict[str, Optional[float]]]]:
    """Re-evaluate an already-built clean/perturbed pair after JPEG / blur degradation.

    Like :func:`evaluate_delta_under_transforms` but takes the protected image directly
    (read from disk), so it serves precomputed pairs such as GeoShield. The protected
    image is degraded at each level and compared against the untransformed clean image
    with the same paired, shared-noise evaluation. Returns
    ``{"jpeg": {quality: {"predicted": .., "true": ..}}, "blur": {...}}``.
    """

    def _displacements(transformed_image) -> Dict[str, Optional[float]]:
        eval_result = run_paired_pipeline_with_shared_noise(
            pipeline=pipeline,
            source_image=source_image,
            perturbed_image=transformed_image,
            batch_size=batch_size,
            cfg=cfg,
            num_steps=num_steps,
            seed=seed,
            device=device,
        )
        predicted = float(eval_result["metrics"]["final_step_displacement"])
        true = None
        if true_gps is not None:
            true = float(final_prediction_distance_to_point(eval_result["traj_perturbed"], true_gps))
        return {"predicted": predicted, "true": true}

    displacements: Dict[str, Dict[float, Dict[str, Optional[float]]]] = {"jpeg": {}, "blur": {}}
    for quality in jpeg_quality_factors:
        displacements["jpeg"][int(quality)] = _displacements(
            apply_jpeg_compression(perturbed_image, int(quality))
        )
    for sigma in gaussian_blur_sigmas:
        displacements["blur"][float(sigma)] = _displacements(
            apply_gaussian_blur(perturbed_image, float(sigma))
        )
    return displacements


def evaluate_delta_under_transforms(
    pipeline,
    source_image,
    delta,
    jpeg_quality_factors: Sequence[int],
    gaussian_blur_sigmas: Sequence[float],
    cfg: float = 10.0,
    batch_size: int = 128,
    seed: int = 1234,
    device: str = "cuda",
    num_steps: Optional[int] = None,
    true_gps: Optional[Sequence[float]] = None,
) -> Dict[str, Dict[float, Dict[str, Optional[float]]]]:
    """Re-evaluate a single trained perturbation after JPEG / blur degradation.

    The protected image (clean + delta) is JPEG-compressed and Gaussian-blurred at
    each configured level, then compared against the untransformed clean image with
    the exact paired, shared-noise evaluation used everywhere else. Sampling steps
    are held at ``num_steps`` (the baseline) for every transform.

    Each level records two displacements (km), matching the main eval's two metrics:
    ``"predicted"`` (transformed protected prediction vs the clean prediction) and,
    when ``true_gps`` is given, ``"true"`` (vs the ground-truth GPS, the GeoShield
    metric). Returns ``{"jpeg": {quality: {"predicted": .., "true": ..}}, "blur": {...}}``.
    """
    perturbed_image = add_perturbation_to_image(source_image, delta.to(device), pipeline)
    return evaluate_pair_under_transforms(
        pipeline=pipeline,
        source_image=source_image,
        perturbed_image=perturbed_image,
        jpeg_quality_factors=jpeg_quality_factors,
        gaussian_blur_sigmas=gaussian_blur_sigmas,
        cfg=cfg,
        batch_size=batch_size,
        seed=seed,
        device=device,
        num_steps=num_steps,
        true_gps=true_gps,
    )


def pred_true_from_cell(cell: Any) -> tuple[Optional[float], Optional[float]]:
    """Split a per-level robustness record into (predicted, true) displacements.

    Accepts the current ``{"predicted": .., "true": ..}`` dict and, defensively, a
    bare float (an older predicted-only record) or ``None``.
    """
    if isinstance(cell, dict):
        pred = cell.get("predicted")
        true = cell.get("true")
        return (float(pred) if pred is not None else None, float(true) if true is not None else None)
    if cell is None:
        return None, None
    return float(cell), None


def _transform_summary(
    disps: Sequence[float],
    success_rate_thresholds: Sequence[float],
) -> Dict[str, Any]:
    """Mean displacement + per-threshold success rate for one transform level."""
    disps = list(disps)
    success_rates = {
        str(thr): float(np.mean([d > thr for d in disps])) if disps else float("nan")
        for thr in success_rate_thresholds
    }
    return {
        "mean_displacement_km": float(np.mean(disps)) if disps else float("nan"),
        "success_rates": success_rates,
    }


def build_robustness_json(
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    jpeg_quality_factors: Sequence[int],
    gaussian_blur_sigmas: Sequence[float],
    success_rate_thresholds: Sequence[float],
    n_images: int,
    samples: Dict[str, Dict[int, Dict[str, Dict[float, Dict[str, List[float]]]]]],
    num_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Assemble the robustness JSON consumed by ``plot_robustness_results``.

    ``samples[attack_type][budget_idx]["jpeg"][quality][metric]`` (``metric`` in
    ``ROBUSTNESS_METRICS``; ``["blur"][sigma]`` likewise) is the list of per-image
    final-step displacements (km) measured at that level. Each emitted level entry is
    ``{"predicted": {mean, success_rates}, "true": {mean, success_rates}}``.
    """
    jpeg_quality_factors = list(jpeg_quality_factors)
    gaussian_blur_sigmas = [float(s) for s in gaussian_blur_sigmas]
    json_results: Dict[str, Any] = {
        "dataset": dataset_name,
        "attack_types": list(attack_types),
        "attack_budgets": list(attack_budgets),
        "jpeg_quality_factors": jpeg_quality_factors,
        "gaussian_blur_sigmas": gaussian_blur_sigmas,
        "num_steps": int(num_steps) if num_steps is not None else None,
        "success_rate_thresholds_km": list(success_rate_thresholds),
        "metrics": list(ROBUSTNESS_METRICS),
        "n_images": int(n_images),
        "results": {},
    }

    def _level_entry(metric_lists: Dict[str, List[float]]) -> Dict[str, Any]:
        return {
            metric: _transform_summary(metric_lists.get(metric, []), success_rate_thresholds)
            for metric in ROBUSTNESS_METRICS
        }

    for at in attack_types:
        json_results["results"][at] = {}
        for budget_idx, budget in enumerate(attack_budgets):
            bkey = _budget_key(budget)
            entry: Dict[str, Dict[str, Any]] = {"jpeg": {}, "blur": {}}
            for quality in jpeg_quality_factors:
                lists = samples[at][budget_idx]["jpeg"].get(int(quality), {})
                entry["jpeg"][_jpeg_key(quality)] = _level_entry(lists)
            for sigma in gaussian_blur_sigmas:
                lists = samples[at][budget_idx]["blur"].get(float(sigma), {})
                entry["blur"][_blur_key(sigma)] = _level_entry(lists)
            json_results["results"][at][bkey] = entry
    return json_results


def empty_robustness_samples(
    attack_types: Sequence[str],
    n_budgets: int,
    jpeg_quality_factors: Sequence[int],
    gaussian_blur_sigmas: Sequence[float],
) -> Dict[str, Dict[int, Dict[str, Dict[float, Dict[str, List[float]]]]]]:
    """Allocate the nested ``samples`` accumulator used by ``build_robustness_json``.

    Innermost level holds one list per metric in ``ROBUSTNESS_METRICS``.
    """
    def _levels(values, cast):
        return {cast(v): {metric: [] for metric in ROBUSTNESS_METRICS} for v in values}

    return {
        at: {
            bi: {
                "jpeg": _levels(jpeg_quality_factors, int),
                "blur": _levels(gaussian_blur_sigmas, float),
            }
            for bi in range(n_budgets)
        }
        for at in attack_types
    }
