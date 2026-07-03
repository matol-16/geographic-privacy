"""
Evaluation entry points for adversarial attacks on geolocation models.

Each public ``evaluate_*`` function is thin orchestration: it builds an
``EvaluationConfig``, runs the shared engine in ``core.py``, then saves results
and plots. The restart and sampling-steps ablations are produced from the shared
helpers in :mod:`utils.ablations`, so the standalone ``evaluate-restarts`` /
``evaluate-sampling-steps`` commands and the integrated ``evaluate-dataset`` path
emit byte-compatible JSON through one code path.

We evaluate on OSV-5M's test set and on YFCC4k. Dataset loading lives in
:mod:`utils.datasets` (re-exported here for backward compatibility).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch
import tqdm as tqdm_module

# Dataset retrieval moved to utils.datasets; re-exported so existing imports
# (`from utils.adversarial_eval import retrieve_yfcc_images`) keep working.
from utils.datasets import (  # noqa: F401
    load_osv5m_test,
    retrieve_yfcc_images,
    retrieve_osv_images,
)
from utils.ablations import (
    build_model_transfer_json,
    build_restart_ablation_json,
    build_robustness_json,
    build_sampling_steps_json,
    empty_robustness_samples,
    evaluate_delta_at_steps,
    evaluate_delta_under_transforms,
    model_type_label,
    pred_true_from_cell,
    restart_displacements_from_results,
)
from utils.adversarial_utils import (
    expand_per_budget_kwargs,
    run_paired_pipeline_with_shared_noise,
    seed_everything,
)
from utils.plots_adversarial_attacks import (
    plot_attack_success_rate,
    plot_model_transfer_success_rate,
    plot_restarts_success,
    plot_results,
    plot_robustness_results,
    plot_sampling_steps_success_rate,
)


# --------------------------------------------------------------------------- #
# Shared ablation persistence (used by evaluate-dataset and the standalone cmds)
# --------------------------------------------------------------------------- #


def _save_json(payload: Dict[str, Any], results_dir: str, filename: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {filename} to: {path}")
    return path


def build_and_save_restart_ablation(
    runner,
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    results_dir: str,
    max_restarts: Optional[int] = None,
) -> tuple[Dict[str, Any], int]:
    """Derive the restart ablation from already-collected per-restart evaluations.

    The per-restart displacements live in ``runner.metrics_collector.restart_results``
    and are recorded during training, so this adds no extra model calls. Returns the
    JSON dict plus the number of restarts actually observed (for plotting decisions).
    """
    mc = runner.metrics_collector
    per_restart = restart_displacements_from_results(
        mc.restart_results, attack_types, len(attack_budgets), mc.n_images
    )
    observed = max(
        (
            len(disps)
            for at in attack_types
            for bi in range(len(attack_budgets))
            for disps in per_restart[at][bi].values()
        ),
        default=0,
    )
    if max_restarts is None:
        max_restarts = observed
    image_ids = mc.source_image_ids or [str(i) for i in range(mc.n_images)]
    json_results = build_restart_ablation_json(
        dataset_name, attack_types, attack_budgets, image_ids, per_restart, max_restarts
    )
    _save_json(json_results, results_dir, f"{dataset_name}_restarts_results.json")
    return json_results, observed


def build_and_save_sampling_steps_ablation(
    runner,
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    eval_num_steps: Sequence[int],
    success_rate_thresholds: Sequence[float],
    results_dir: str,
    out_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate the per-image sampling-steps samples collected during the run.

    Saved to ``out_filename`` (default ``<dataset>_sampling_steps_results.json``). A caller
    overlaying a precomputed attack (e.g. GeoShield) onto the trainable ablation passes an
    attack-specific name so it does not clobber the trainable JSON before merging.
    """
    mc = runner.metrics_collector
    eval_num_steps = list(eval_num_steps)
    samples = {
        at: {bi: {ns: [] for ns in eval_num_steps} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for at in attack_types:
        for bi in range(len(attack_budgets)):
            for ii in range(mc.n_images):
                record = mc.sampling_steps_results[at][bi][ii]
                if not record:
                    continue
                for ns in eval_num_steps:
                    if ns in record:
                        samples[at][bi][ns].append(float(record[ns]))
    json_results = build_sampling_steps_json(
        dataset_name,
        attack_types,
        attack_budgets,
        eval_num_steps,
        success_rate_thresholds,
        mc.n_images,
        samples,
    )
    _save_json(json_results, results_dir, out_filename or f"{dataset_name}_sampling_steps_results.json")
    return json_results


def build_and_save_model_transfer_ablation(
    runner,
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    model_labels: Sequence[str],
    success_rate_thresholds: Sequence[float],
    results_dir: str,
    attacked_model_label: Optional[str] = None,
    num_steps: Optional[int] = None,
    out_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """Aggregate the per-image cross-model transfer samples collected during the run.

    Mirrors ``build_and_save_sampling_steps_ablation`` with the model variant as the
    swept axis. Saved to ``out_filename`` (default ``<dataset>_model_transfer_results.json``).
    """
    mc = runner.metrics_collector
    model_labels = list(model_labels)
    samples = {
        at: {bi: {label: [] for label in model_labels} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for at in attack_types:
        for bi in range(len(attack_budgets)):
            for ii in range(mc.n_images):
                record = mc.model_transfer_results[at][bi][ii]
                if not record:
                    continue
                for label in model_labels:
                    if label in record:
                        samples[at][bi][label].append(float(record[label]))
    json_results = build_model_transfer_json(
        dataset_name,
        attack_types,
        attack_budgets,
        model_labels,
        success_rate_thresholds,
        mc.n_images,
        samples,
        attacked_model_label=attacked_model_label,
        num_steps=num_steps,
    )
    _save_json(json_results, results_dir, out_filename or f"{dataset_name}_model_transfer_results.json")
    return json_results


def build_and_save_robustness_ablation(
    runner,
    dataset_name: str,
    robustness_attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    jpeg_quality_factors: Sequence[int],
    gaussian_blur_sigmas: Sequence[float],
    success_rate_thresholds: Sequence[float],
    results_dir: str,
    num_steps: Optional[int] = None,
    out_filename: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Aggregate the per-image robustness samples collected during the run.

    ``robustness_attack_types`` is the subset of evaluated attacks the ablation was
    run for (default: ["dtd"]). Returns ``None`` (and writes nothing) when none of
    those attacks were evaluated, so enabling the toggle is a no-op until an opted-in
    attack is part of the run. Saved to ``out_filename`` (default
    ``<dataset>_robustness_results.json``); a caller overlaying a precomputed attack
    passes an attack-specific name so it does not clobber the trainable JSON.
    """
    mc = runner.metrics_collector
    robustness_attack_types = [at for at in robustness_attack_types if at in mc.robustness_results]
    if not robustness_attack_types:
        return None

    samples = empty_robustness_samples(
        robustness_attack_types, len(attack_budgets), jpeg_quality_factors, gaussian_blur_sigmas
    )

    def _accumulate(target: dict, cell) -> None:
        pred, true = pred_true_from_cell(cell)
        if pred is not None:
            target["predicted"].append(pred)
        if true is not None:
            target["true"].append(true)

    for at in robustness_attack_types:
        for bi in range(len(attack_budgets)):
            for ii in range(mc.n_images):
                record = mc.robustness_results[at][bi][ii]
                if not record:
                    continue
                for quality in jpeg_quality_factors:
                    _accumulate(samples[at][bi]["jpeg"][int(quality)], record.get("jpeg", {}).get(int(quality)))
                for sigma in gaussian_blur_sigmas:
                    _accumulate(samples[at][bi]["blur"][float(sigma)], record.get("blur", {}).get(float(sigma)))

    json_results = build_robustness_json(
        dataset_name,
        robustness_attack_types,
        attack_budgets,
        jpeg_quality_factors,
        gaussian_blur_sigmas,
        success_rate_thresholds,
        mc.n_images,
        samples,
        num_steps=num_steps,
    )
    _save_json(json_results, results_dir, out_filename or f"{dataset_name}_robustness_results.json")
    return json_results


def _overlay_attack_into_ablation_json(
    canonical_path: str, attack_name: str, attack_json: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Merge one attack's entry from ``attack_json`` into the canonical on-disk ablation JSON.

    Sampling-steps and robustness JSONs share the ``{attack_types, results}`` shape, so the
    same union works for both: the precomputed attack's ``results[attack_name]`` is inserted
    into (and ``attack_name`` appended to) the trainable JSON already on disk, then re-saved.
    If that JSON is absent (no trainable attacks ran), ``attack_json`` becomes canonical.
    Returns the merged dict (for re-plotting), or ``None`` if there is nothing to overlay.
    """
    src_entry = (attack_json or {}).get("results", {}).get(attack_name)
    if src_entry is None:
        return None
    if os.path.exists(canonical_path):
        with open(canonical_path) as f:
            merged = json.load(f)
    else:
        merged = dict(attack_json)
    merged.setdefault("results", {})[attack_name] = src_entry
    if attack_name not in merged.setdefault("attack_types", []):
        merged["attack_types"].append(attack_name)
    with open(canonical_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Overlaid '{attack_name}' into: {canonical_path}")
    return merged


def overlay_precomputed_ablations(
    runner,
    dataset_name: str,
    attack_budgets: Sequence[float],
    results_dir: str,
    plots_dir: str,
    success_rate_thresholds: Sequence[float],
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[Sequence[int]] = None,
    run_robustness_ablation: bool = False,
    robustness_jpeg_quality_factors: Optional[Sequence[int]] = None,
    robustness_gaussian_blur_sigmas: Optional[Sequence[float]] = None,
    robustness_num_steps: Optional[int] = None,
) -> None:
    """Overlay a precomputed attack's ablations onto the trainable ones (single-GPU path).

    In ``evaluate-dataset`` the trainable attacks and GeoShield are evaluated by separate
    runners, so the trainable ablation JSONs/plots are already on disk when GeoShield finishes.
    This builds GeoShield's sampling-steps / robustness ablation from its own collector (saved to
    an attack-specific JSON), merges its entry into the canonical trainable JSON, and re-plots the
    combined figure -- mirroring how ``merge-shards`` overlays it in the multi-GPU path. The
    restart ablation is intentionally omitted (a precomputed attack has a single generated image).
    """
    attack_name = runner.attack_types[0]
    attack_budgets = list(attack_budgets)
    success_rate_thresholds = list(success_rate_thresholds)

    if run_sampling_steps_ablation and eval_num_steps:
        geo_json = build_and_save_sampling_steps_ablation(
            runner, dataset_name, [attack_name], attack_budgets, list(eval_num_steps),
            success_rate_thresholds, results_dir,
            out_filename=f"{dataset_name}_{attack_name}_sampling_steps_results.json",
        )
        merged = _overlay_attack_into_ablation_json(
            os.path.join(results_dir, f"{dataset_name}_sampling_steps_results.json"),
            attack_name, geo_json,
        )
        if merged is not None:
            plot_sampling_steps_success_rate(json_results=merged, plot_dir=plots_dir)

    if run_robustness_ablation:
        geo_json = build_and_save_robustness_ablation(
            runner, dataset_name, [attack_name], attack_budgets,
            robustness_jpeg_quality_factors or [], robustness_gaussian_blur_sigmas or [],
            success_rate_thresholds, results_dir, num_steps=robustness_num_steps,
            out_filename=f"{dataset_name}_{attack_name}_robustness_results.json",
        )
        merged = _overlay_attack_into_ablation_json(
            os.path.join(results_dir, f"{dataset_name}_robustness_results.json"),
            attack_name, geo_json,
        )
        if merged is not None:
            plot_robustness_results(json_results=merged, plot_dir=plots_dir)


# --------------------------------------------------------------------------- #
# Main dataset evaluation (+ integrated ablations)
# --------------------------------------------------------------------------- #


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
    stored_metrics=["final_step_displacement_predicted", "final_step_displacement_true"],
    attack_kwargs: list[Dict[str, Any]] = [{}],
    parallel_workers: int = 1,
    use_cuda_streams: bool = True,
    dataset_roots: Optional[Dict[str, str]] = None,
    plot_success_rate: bool = False,
    plot_success_rate_thresholds: Optional[list[float]] = None,
    plot_gps_true: bool = True,
    config_dump: Optional[Dict[str, Any]] = None,
    max_restarts: Optional[int] = None,
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[list[int]] = None,
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    run_robustness_ablation: bool = False,
    robustness_attack_types: Optional[list[str]] = None,
    robustness_jpeg_quality_factors: Optional[list[int]] = None,
    robustness_gaussian_blur_sigmas: Optional[list[float]] = None,
    robustness_num_steps: Optional[int] = None,
    run_model_transfer_ablation: bool = False,
    model_transfer_types: Optional[list[str]] = None,
    model_transfer_num_steps: Optional[int] = None,
    transfer_pipelines: Optional[Dict[str, Any]] = None,
    attacked_model_label: Optional[str] = None,
):
    """Evaluate one or more attacks on images from a test dataset.

    A single training pass yields the main results plus the ablations:

    - **Main results**: best-restart displacement per image → displacement and
      success-rate plots.
    - **Restart ablation** (always): best-displacement-after-k-restarts, derived
      for free from the per-restart evaluations collected during training. Reaches
      depth ``num_restarts`` (override per run with ``max_restarts``).
    - **Sampling-steps ablation** (opt-in via ``run_sampling_steps_ablation``):
      re-evaluates each image's best perturbation at every count in
      ``eval_num_steps`` (extra pipeline runs, no retraining).
    - **Robustness ablation** (opt-in via ``run_robustness_ablation``, restricted to
      ``robustness_attack_types`` -- default ["dtd"]): degrades each best
      perturbation's protected image with JPEG compression / Gaussian blur
      (GeoShield Fig. 6 levels) and re-evaluates at the baseline sampling-step count.

    Args:
        attack_types: A single attack name (str) or a list, e.g. ["encoder", "diffusion"].
        pipeline: PLONK pipeline.
        dataset_name: "osv" or "yfcc".
        use_real_gps: Use real GPS coords from the dataset as the source trajectory
            instead of the clean predicted one.
        parallel_workers: Concurrent evaluations; values > 1 enable parallel execution.
        use_cuda_streams: If True and on CUDA, each worker uses its own CUDA stream.
        max_restarts: If set, overrides ``num_restarts`` in every attack_kwargs so the
            restart ablation reaches the requested depth.
        run_sampling_steps_ablation / eval_num_steps: enable + configure the
            sampling-steps ablation.
        run_robustness_ablation / robustness_attack_types /
            robustness_jpeg_quality_factors / robustness_gaussian_blur_sigmas /
            robustness_num_steps: enable + configure the robustness ablation. The
            ablation only runs for attacks in ``robustness_attack_types`` (default
            ["dtd"]); ``robustness_num_steps`` of None uses each attack's baseline
            ``restart_eval_num_steps``.
        attack_type_kwargs: optional ``{attack_type: {kwarg: value}}`` overrides merged
            on top of the shared kwargs only for the matching attack (e.g. ACE's
            target_image / l2_target loss / alpha), so every attack type can run in one
            evaluation without leaking settings into the others.
    """
    from core import EvaluationConfig, EvaluationRunner, run_evaluation

    seed_everything(seed)

    dataset_roots = dataset_roots or {}
    results_dir = results_dir or "/results"
    plot_dir = plot_dir or "/plots"

    if isinstance(attack_types, str):
        attack_types = [attack_types]

    if source_image is not None:
        raise NotImplementedError("Custom source image not yet supported in refactored code")

    # Normalize per-budget kwargs and (optionally) force the restart depth.
    attack_kwargs = expand_per_budget_kwargs(attack_kwargs, len(attack_budgets))
    if max_restarts is not None:
        # Train ``max_restarts`` restarts so the restart ablation reaches that depth, but
        # keep the reported/"main" result selected over only the originally configured
        # ``num_restarts`` (see RestartManager.main_num_restarts). The extra restarts feed
        # the ablation without changing the reported best perturbation or its metrics.
        attack_kwargs = [
            {
                **kw,
                "num_restarts": int(max_restarts),
                "main_num_restarts": int(kw.get("num_restarts", max_restarts)),
            }
            for kw in attack_kwargs
        ]

    success_rate_thresholds = plot_success_rate_thresholds or [2500]

    # Robustness-ablation defaults (only consulted when run_robustness_ablation).
    # GeoShield Fig. 6 levels; restricted to ``robustness_attack_types`` (default dtd).
    robustness_attack_types = ["dtd"] if robustness_attack_types is None else list(robustness_attack_types)
    robustness_jpeg_quality_factors = (
        [10, 20, 30, 40, 50, 60]
        if robustness_jpeg_quality_factors is None
        else list(robustness_jpeg_quality_factors)
    )
    robustness_gaussian_blur_sigmas = (
        [0, 2, 4, 6, 8, 10]
        if robustness_gaussian_blur_sigmas is None
        else list(robustness_gaussian_blur_sigmas)
    )

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
        run_sampling_steps_ablation=run_sampling_steps_ablation,
        eval_num_steps=list(eval_num_steps) if eval_num_steps else None,
        success_rate_thresholds=list(success_rate_thresholds),
        attack_type_kwargs=attack_type_kwargs or {},
        run_robustness_ablation=run_robustness_ablation,
        robustness_attack_types=robustness_attack_types,
        robustness_jpeg_quality_factors=robustness_jpeg_quality_factors,
        robustness_gaussian_blur_sigmas=robustness_gaussian_blur_sigmas,
        robustness_num_steps=robustness_num_steps,
        run_model_transfer_ablation=run_model_transfer_ablation,
        model_transfer_types=list(model_transfer_types) if model_transfer_types else None,
        model_transfer_num_steps=model_transfer_num_steps,
    )

    runner = EvaluationRunner(config, pipeline, transfer_pipelines=transfer_pipelines)
    if config_dump is not None:
        runner.save_run_config(config_dump)
    run_evaluation(runner)
    runner.save_results()

    # ---- Main plots --------------------------------------------------------- #
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
            threshold_km=list(success_rate_thresholds),
            gps_true=plot_gps_true,
        )

    # ---- Restart ablation (always) ----------------------------------------- #
    restart_json, observed_restarts = build_and_save_restart_ablation(
        runner, dataset_name, attack_types, attack_budgets, results_dir
    )
    if observed_restarts >= 2:
        # Full-dataset run: summary plot only (avoid one figure per image).
        plot_restarts_success(json_results=restart_json, plot_dir=plot_dir, per_image=False)
    else:
        print("Restart ablation has < 2 restarts; JSON saved but plot skipped.")

    # ---- Sampling-steps ablation (opt-in) ---------------------------------- #
    if run_sampling_steps_ablation and eval_num_steps:
        steps_json = build_and_save_sampling_steps_ablation(
            runner,
            dataset_name,
            attack_types,
            attack_budgets,
            list(eval_num_steps),
            list(success_rate_thresholds),
            results_dir,
        )
        plot_sampling_steps_success_rate(json_results=steps_json, plot_dir=plot_dir)

    # ---- Robustness ablation (opt-in, per attack type) --------------------- #
    if run_robustness_ablation and (robustness_jpeg_quality_factors or robustness_gaussian_blur_sigmas):
        robustness_json = build_and_save_robustness_ablation(
            runner,
            dataset_name,
            robustness_attack_types,
            attack_budgets,
            robustness_jpeg_quality_factors,
            robustness_gaussian_blur_sigmas,
            list(success_rate_thresholds),
            results_dir,
            num_steps=robustness_num_steps,
        )
        if robustness_json is not None:
            plot_robustness_results(json_results=robustness_json, plot_dir=plot_dir)
        else:
            print(
                "Robustness ablation enabled but none of "
                f"{robustness_attack_types} were among the evaluated attacks; nothing to plot."
            )

    # ---- Cross-model transfer ablation (opt-in) ---------------------------- #
    if run_model_transfer_ablation and transfer_pipelines:
        model_labels = list(transfer_pipelines.keys())
        transfer_json = build_and_save_model_transfer_ablation(
            runner,
            dataset_name,
            attack_types,
            attack_budgets,
            model_labels,
            list(success_rate_thresholds),
            results_dir,
            attacked_model_label=attacked_model_label,
            num_steps=model_transfer_num_steps,
        )
        plot_model_transfer_success_rate(json_results=transfer_json, plot_dir=plot_dir)


# --------------------------------------------------------------------------- #
# Multi-node sharding: per-shard training + merge-and-plot
# --------------------------------------------------------------------------- #


def evaluate_attack_shard(
    attack_types,
    pipeline,
    dataset_name: str,
    seed: int,
    total_images: int,
    window_start: int,
    window_end: Optional[int],
    results_dir: str,
    attack_budgets: Sequence[float],
    attack_kwargs: Sequence[Dict[str, Any]],
    stored_metrics: Sequence[str],
    parallel_workers: int = 1,
    use_cuda_streams: bool = True,
    use_real_gps: bool = False,
    dataset_roots: Optional[Dict[str, str]] = None,
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    max_restarts: Optional[int] = None,
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[Sequence[int]] = None,
    success_rate_thresholds: Optional[Sequence[float]] = None,
    run_robustness_ablation: bool = False,
    robustness_attack_types: Optional[Sequence[str]] = None,
    robustness_jpeg_quality_factors: Optional[Sequence[int]] = None,
    robustness_gaussian_blur_sigmas: Optional[Sequence[float]] = None,
    robustness_num_steps: Optional[int] = None,
    run_model_transfer_ablation: bool = False,
    model_transfer_types: Optional[Sequence[str]] = None,
    model_transfer_num_steps: Optional[int] = None,
    transfer_pipelines: Optional[Dict[str, Any]] = None,
    config_dump: Optional[Dict[str, Any]] = None,
):
    """Train + evaluate one shard: the given attack(s) on a window of the seeded pool.

    A shard is one cluster job: it covers ``attack_types`` (typically a single attack)
    over the ``[window_start:window_end]`` slice of the deterministic ``total_images``
    selection. It does **not** plot or build ablation JSONs -- that is left to
    ``merge_shards``. The shard's resumable state file (which already contains the main
    metrics plus the restart / sampling-steps / robustness buffers, all keyed by global
    ``source_image_ids``) is the artifact ``merge_shards`` later stitches together.

    Writes its results into ``results_dir`` (which the CLI sets to a per-shard
    subdirectory so shards never collide). Returns the runner.
    """
    from core import EvaluationConfig, EvaluationRunner, run_evaluation

    seed_everything(seed)
    dataset_roots = dataset_roots or {}

    if isinstance(attack_types, str):
        attack_types = [attack_types]
    attack_types = list(attack_types)
    attack_budgets = list(attack_budgets)

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))
    if max_restarts is not None:
        # Train ``max_restarts`` restarts so the restart ablation reaches that depth, but
        # keep the reported/"main" result selected over only the originally configured
        # ``num_restarts`` (see RestartManager.main_num_restarts). The extra restarts feed
        # the ablation without changing the reported best perturbation or its metrics.
        attack_kwargs = [
            {
                **kw,
                "num_restarts": int(max_restarts),
                "main_num_restarts": int(kw.get("num_restarts", max_restarts)),
            }
            for kw in attack_kwargs
        ]

    success_rate_thresholds = list(success_rate_thresholds) if success_rate_thresholds else [2500]
    robustness_attack_types = ["dtd"] if robustness_attack_types is None else list(robustness_attack_types)
    robustness_jpeg_quality_factors = (
        [10, 20, 30, 40, 50, 60]
        if robustness_jpeg_quality_factors is None
        else list(robustness_jpeg_quality_factors)
    )
    robustness_gaussian_blur_sigmas = (
        [0, 2, 4, 6, 8, 10]
        if robustness_gaussian_blur_sigmas is None
        else list(robustness_gaussian_blur_sigmas)
    )

    config = EvaluationConfig(
        dataset=dataset_name,
        seed=seed,
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        n_images=int(total_images),
        window_start=int(window_start),
        window_end=None if window_end is None else int(window_end),
        results_dir=results_dir,
        plots_dir=results_dir,  # no plotting here, but ResultsManager needs a dir
        stored_metrics=list(stored_metrics),
        parallel_workers=parallel_workers,
        use_cuda_streams=use_cuda_streams,
        use_real_gps=use_real_gps,
        dataset_roots=dataset_roots,
        run_sampling_steps_ablation=run_sampling_steps_ablation,
        eval_num_steps=list(eval_num_steps) if eval_num_steps else None,
        success_rate_thresholds=success_rate_thresholds,
        attack_type_kwargs=attack_type_kwargs or {},
        run_robustness_ablation=run_robustness_ablation,
        robustness_attack_types=robustness_attack_types,
        robustness_jpeg_quality_factors=robustness_jpeg_quality_factors,
        robustness_gaussian_blur_sigmas=robustness_gaussian_blur_sigmas,
        robustness_num_steps=robustness_num_steps,
        run_model_transfer_ablation=run_model_transfer_ablation,
        model_transfer_types=list(model_transfer_types) if model_transfer_types else None,
        model_transfer_num_steps=model_transfer_num_steps,
    )

    runner = EvaluationRunner(config, pipeline, transfer_pipelines=transfer_pipelines)
    if config_dump is not None:
        runner.save_run_config(config_dump)
    run_evaluation(runner)
    runner.save_results()
    return runner


class _MergedCollectorRunner:
    """Minimal stand-in exposing ``.metrics_collector`` for the ablation builders."""

    def __init__(self, metrics_collector):
        self.metrics_collector = metrics_collector


def _copy_per_image_cell(src_by_attack, dst_by_attack, attack_type, budget_idx, local_idx, global_idx) -> None:
    """Copy one ``[attack][budget][image]`` per-image cell from a shard buffer to the merged one."""
    src = src_by_attack.get(attack_type) if isinstance(src_by_attack, dict) else None
    if not isinstance(src, list) or budget_idx >= len(src):
        return
    src_budget = src[budget_idx]
    if not isinstance(src_budget, list) or local_idx >= len(src_budget):
        return
    value = src_budget[local_idx]
    if value is None:
        return
    dst_by_attack[attack_type][budget_idx][global_idx] = value


def _attack_has_per_image_records(buffer_by_attack: Any, attack_type: str) -> bool:
    """True if any per-image cell for this attack holds data (vs. an all-None buffer).

    Used to drop attacks that carry no samples for a given ablation -- e.g. the
    out-of-process GeoShield, which records neither restarts nor sampling-steps -- so
    they don't leak into that ablation as empty lines. This matches the single-GPU path,
    where GeoShield is a separate runner and is never part of those ablations.
    """
    per_budget = buffer_by_attack.get(attack_type) if isinstance(buffer_by_attack, dict) else None
    if not isinstance(per_budget, list):
        return False
    for per_image in per_budget:
        if isinstance(per_image, list) and any(cell is not None for cell in per_image):
            return True
    return False


def merge_shards(
    dataset_name: str,
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    total_images: int,
    seed: int,
    results_dir: str,
    plots_dir: str,
    stored_metrics: Sequence[str],
    dataset_roots: Optional[Dict[str, str]] = None,
    plot_success_rate: bool = False,
    success_rate_thresholds: Optional[Sequence[float]] = None,
    plot_gps_true: bool = True,
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[Sequence[int]] = None,
    run_robustness_ablation: bool = False,
    robustness_attack_types: Optional[Sequence[str]] = None,
    robustness_jpeg_quality_factors: Optional[Sequence[int]] = None,
    robustness_gaussian_blur_sigmas: Optional[Sequence[float]] = None,
    robustness_num_steps: Optional[int] = None,
    run_model_transfer_ablation: bool = False,
    model_transfer_types: Optional[Sequence[str]] = None,
    model_transfer_num_steps: Optional[int] = None,
    attacked_model_label: Optional[str] = None,
    config_dump: Optional[Dict[str, Any]] = None,
    shards_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Stitch per-shard state files back into one full-dataset result set, then plot.

    Reads every shard state file under ``shards_dir`` (default ``results_dir/shards``),
    maps each shard's global ``source_image_ids`` onto the full seeded ordering of
    ``total_images``, and fills a combined ``MetricsCollector`` sized to the whole pool.
    Then it saves the combined per-attack results, the main + success-rate plots, and
    the restart / sampling-steps / robustness ablation JSONs and plots -- the exact
    artifacts a single-process ``evaluate-dataset`` would have produced.
    """
    from core import MetricsCollector, ResultsManager
    from utils.datasets import select_image_metadata

    attack_types = list(attack_types)
    attack_budgets = list(attack_budgets)
    stored_metrics = list(stored_metrics)
    success_rate_thresholds = list(success_rate_thresholds) if success_rate_thresholds else [2500]
    dataset_roots = dataset_roots or {}

    # 1. Full seeded ordering -> id -> global index. Keyed by the raw id and by its
    #    extension-stripped stem, so GeoShield's filename keys ("123.jpg") match the
    #    trainable attacks' photo-id keys ("123") in the same merge.
    metadata = select_image_metadata(
        dataset_name, int(total_images), int(seed), dataset_roots.get(dataset_name)
    )
    global_ids = [str(img_id) for _, _, img_id in metadata]
    id_to_global: Dict[str, int] = {}
    for i, gid in enumerate(global_ids):
        id_to_global[gid] = i
        id_to_global.setdefault(os.path.splitext(gid)[0], i)

    def _global_index(img_id: Any) -> Optional[int]:
        key = str(img_id)
        if key in id_to_global:
            return id_to_global[key]
        return id_to_global.get(os.path.splitext(key)[0])

    # 2. Combined collector for the whole pool.
    combined = MetricsCollector(
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        n_images=len(global_ids),
        stored_metrics=stored_metrics,
        source_gps=None,
        source_image_ids=global_ids,
    )

    # 3. Discover shard state files. The trailing ``*`` catches both the trainable shards
    #    (``..._eval_state.pt``) and the precomputed GeoShield shard, whose state file
    #    carries the attack name (``..._eval_state_geoshield.pt``, see
    #    PrecomputedPairEvaluationRunner._state_file_suffix).
    shards_dir = shards_dir or os.path.join(results_dir, "shards")
    state_paths = sorted(glob.glob(os.path.join(shards_dir, "*", f"{dataset_name}_seed{seed}_eval_state*.pt")))
    if not state_paths:
        state_paths = sorted(glob.glob(os.path.join(shards_dir, "*", "*_eval_state*.pt")))
    if not state_paths:
        raise FileNotFoundError(
            f"No shard state files found under {shards_dir}. Did the shard jobs run and finish?"
        )
    print(f"Merging {len(state_paths)} shard state file(s) from {shards_dir}")

    covered: set = set()
    unknown_ids = 0
    for state_path in state_paths:
        state = torch.load(state_path, map_location="cpu")
        shard_ids = state.get("source_image_ids") or []
        results = state.get("results") or {}
        restart_results = state.get("restart_results") or {}
        location_results = state.get("location_results") or {}
        sampling_steps_results = state.get("sampling_steps_results") or {}
        robustness_results = state.get("robustness_results") or {}
        model_transfer_results = state.get("model_transfer_results") or {}

        for attack_type in attack_types:
            saved = results.get(attack_type)
            if not isinstance(saved, dict):
                continue
            for budget_idx in range(len(attack_budgets)):
                for local_idx, img_id in enumerate(shard_ids):
                    global_idx = _global_index(img_id)
                    if global_idx is None:
                        unknown_ids += 1
                        continue
                    if location_results.get(attack_type) and \
                       _cell_value(location_results, attack_type, budget_idx, local_idx) is None:
                        # task not finished for this (attack, budget, image); skip cleanly
                        continue
                    for metric in stored_metrics:
                        tensor = saved.get(metric)
                        if isinstance(tensor, torch.Tensor) and budget_idx < tensor.shape[0] and local_idx < tensor.shape[1]:
                            combined.results[attack_type][metric][budget_idx, global_idx] = tensor[budget_idx, local_idx]
                    _copy_per_image_cell(restart_results, combined.restart_results, attack_type, budget_idx, local_idx, global_idx)
                    _copy_per_image_cell(location_results, combined.location_results, attack_type, budget_idx, local_idx, global_idx)
                    _copy_per_image_cell(sampling_steps_results, combined.sampling_steps_results, attack_type, budget_idx, local_idx, global_idx)
                    _copy_per_image_cell(robustness_results, combined.robustness_results, attack_type, budget_idx, local_idx, global_idx)
                    _copy_per_image_cell(model_transfer_results, combined.model_transfer_results, attack_type, budget_idx, local_idx, global_idx)
                    covered.add((attack_type, budget_idx, global_idx))

    expected = len(attack_types) * len(attack_budgets) * len(global_ids)
    print(f"Merged {len(covered)}/{expected} (attack, budget, image) cells.")
    if len(covered) < expected:
        print(f"  WARNING: {expected - len(covered)} cells missing (incomplete/failed shards). Plots use what is present.")
    if unknown_ids:
        print(f"  WARNING: {unknown_ids} shard rows had image ids absent from the seeded pool (total_images/seed mismatch?).")

    # 4. Save combined per-attack results.
    results_manager = ResultsManager(results_dir, plots_dir)
    all_results = combined.get_results()
    for attack_type in attack_types:
        results_manager.save_results(all_results[attack_type], dataset_name, attack_type)
    if config_dump is not None:
        results_manager.save_run_config(config_dump, dataset_name, suffix="_merged")

    # 5. Main plots.
    plot_results(
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        plot_dir=plots_dir,
        dataset_name=dataset_name,
        attack_types=attack_types,
        all_results=all_results,
        stored_metrics=stored_metrics,
    )
    if plot_success_rate:
        plot_attack_success_rate(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset_name,
            attack_types=attack_types,
            all_results=all_results,
            threshold_km=list(success_rate_thresholds),
            gps_true=plot_gps_true,
        )

    # 6. Ablations (shared builders, fed the merged collector). Each ablation is restricted
    #    to the attacks that actually carry its per-image samples, so out-of-process attacks
    #    like GeoShield (which record neither restarts nor sampling-steps) don't leak in as
    #    empty lines -- matching the single-GPU path, where they're never part of them.
    shim = _MergedCollectorRunner(combined)
    restart_attack_types = [at for at in attack_types if _attack_has_per_image_records(combined.restart_results, at)]
    restart_json, observed_restarts = build_and_save_restart_ablation(
        shim, dataset_name, restart_attack_types, attack_budgets, results_dir
    )
    if restart_attack_types and observed_restarts >= 2:
        plot_restarts_success(json_results=restart_json, plot_dir=plots_dir, per_image=False)
    else:
        print("Restart ablation has < 2 restarts; JSON saved but plot skipped.")

    if run_sampling_steps_ablation and eval_num_steps:
        steps_attack_types = [at for at in attack_types if _attack_has_per_image_records(combined.sampling_steps_results, at)]
        if steps_attack_types:
            steps_json = build_and_save_sampling_steps_ablation(
                shim,
                dataset_name,
                steps_attack_types,
                attack_budgets,
                list(eval_num_steps),
                list(success_rate_thresholds),
                results_dir,
            )
            plot_sampling_steps_success_rate(json_results=steps_json, plot_dir=plots_dir)
        else:
            print("Sampling-steps ablation: no merged attack carries sampling-steps samples; skipped.")

    if run_robustness_ablation:
        robustness_attack_types = ["dtd"] if robustness_attack_types is None else list(robustness_attack_types)
        robustness_jpeg_quality_factors = (
            [10, 20, 30, 40, 50, 60]
            if robustness_jpeg_quality_factors is None
            else list(robustness_jpeg_quality_factors)
        )
        robustness_gaussian_blur_sigmas = (
            [0, 2, 4, 6, 8, 10]
            if robustness_gaussian_blur_sigmas is None
            else list(robustness_gaussian_blur_sigmas)
        )
        robustness_json = build_and_save_robustness_ablation(
            shim,
            dataset_name,
            robustness_attack_types,
            attack_budgets,
            robustness_jpeg_quality_factors,
            robustness_gaussian_blur_sigmas,
            list(success_rate_thresholds),
            results_dir,
            num_steps=robustness_num_steps,
        )
        if robustness_json is not None:
            plot_robustness_results(json_results=robustness_json, plot_dir=plots_dir)
        else:
            print(
                "Robustness ablation enabled but none of "
                f"{robustness_attack_types} were among the merged attacks; nothing to plot."
            )

    if run_model_transfer_ablation and model_transfer_types:
        model_labels = [model_type_label(mt) for mt in model_transfer_types]
        transfer_attack_types = [
            at for at in attack_types if _attack_has_per_image_records(combined.model_transfer_results, at)
        ]
        if transfer_attack_types:
            transfer_json = build_and_save_model_transfer_ablation(
                shim,
                dataset_name,
                transfer_attack_types,
                attack_budgets,
                model_labels,
                list(success_rate_thresholds),
                results_dir,
                attacked_model_label=attacked_model_label,
                num_steps=model_transfer_num_steps,
            )
            plot_model_transfer_success_rate(json_results=transfer_json, plot_dir=plots_dir)
        else:
            print("Model-transfer ablation: no merged attack carries transfer samples; skipped.")

    return all_results


def _cell_value(src_by_attack, attack_type, budget_idx, local_idx):
    """Read one ``[attack][budget][image]`` per-image cell, or None if out of range."""
    src = src_by_attack.get(attack_type) if isinstance(src_by_attack, dict) else None
    if not isinstance(src, list) or budget_idx >= len(src):
        return None
    src_budget = src[budget_idx]
    if not isinstance(src_budget, list) or local_idx >= len(src_budget):
        return None
    return src_budget[local_idx]


# --------------------------------------------------------------------------- #
# Localizability: a two-stage pipeline decoupled from the attack evaluation.
#
# The localizability of an image (how confidently the RFM model can place it) is a
# property of the *clean* image alone, independent of any attack. On a full-dataset
# cluster run the attacks are evaluated separately (evaluate-dataset-shard / merge),
# so we precompute the localizability scores once, key them by image id, and store
# them. Stage 2 ("plot") then joins those scores to whatever attack results exist on
# disk by image id and produces the localizability-vs-attack-strength figure, without
# re-running any attack.
#
#   Stage 1 (compute):  compute_localizability_scores()  -> per-window shard files
#                       merge_localizability_scores()     -> {dataset}_localizability.pt
#   Stage 2 (plot):     plot_localizability_vs_attacks()  -> joins with attack results
# --------------------------------------------------------------------------- #

DEFAULT_LOCALIZABILITY_MC_SAMPLES = 256
LOCALIZABILITY_SHARDS_SUBDIR = "localizability_shards"


def _localizability_shard_path(results_dir: str, dataset_name: str, window_start: int, window_end: int) -> str:
    """Per-window shard file holding the localizability scores for one image window."""
    shards_dir = os.path.join(results_dir, LOCALIZABILITY_SHARDS_SUBDIR)
    return os.path.join(shards_dir, f"{dataset_name}_loc__w{window_start:06d}_{window_end:06d}.pt")


def _localizability_path(results_dir: str, dataset_name: str) -> str:
    """Merged, full-dataset localizability file (id -> score), the input to the plot stage."""
    return os.path.join(results_dir, f"{dataset_name}_localizability.pt")


def compute_localizability_scores(
    pipeline,
    dataset_name: str,
    seed: int = 0,
    n_images_to_eval: int = 100,
    results_dir: str = "./results",
    dataset_roots: Optional[Dict[str, str]] = None,
    window_start: int = 0,
    window_end: Optional[int] = None,
    num_monte_carlo_samples: int = DEFAULT_LOCALIZABILITY_MC_SAMPLES,
    config_dump: Optional[Dict[str, Any]] = None,
    save_every: int = 25,
) -> Dict[str, Any]:
    """Stage 1: compute the RFM localizability of each (clean) image in a seeded window.

    ``n_images_to_eval`` is the size of the full seeded pool; ``window_start`` /
    ``window_end`` select the contiguous slice to compute here (the whole pool by
    default), matching the windowing used by the sharded attack evaluation so the ids
    line up. Scores are written to a per-window shard file under
    ``<results_dir>/localizability_shards/`` and are resumable: an interrupted shard
    re-loads the ids it already finished and only computes the rest. ``merge_localizability_scores``
    later stitches every shard into ``<dataset>_localizability.pt`` for the plot stage.
    """
    from core import ImageLoader

    seed_everything(seed)
    dataset_roots = dataset_roots or {}

    print(f"Loading window [{window_start}:{window_end}] of the seeded {dataset_name} "
          f"pool (size {n_images_to_eval})...")
    source_images, _source_gps, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
        window_start=window_start,
        window_end=window_end,
    )
    resolved_window_end = window_end if window_end is not None else n_images_to_eval

    shard_path = _localizability_shard_path(results_dir, dataset_name, window_start, resolved_window_end)
    os.makedirs(os.path.dirname(shard_path), exist_ok=True)

    # Resume: keep any scores already computed for this exact window/seed/MC-sample config.
    scores_by_id: Dict[str, float] = {}
    if os.path.exists(shard_path):
        prev = torch.load(shard_path, map_location="cpu")
        if (prev.get("seed") == seed
                and prev.get("num_monte_carlo_samples") == num_monte_carlo_samples
                and prev.get("total_images") == n_images_to_eval):
            scores_by_id = dict(prev.get("scores_by_id", {}))
            print(f"Resuming from {shard_path}: {len(scores_by_id)} scores already computed.")

    def _save_shard() -> Dict[str, Any]:
        ids = [i for i in source_image_ids if i in scores_by_id]
        loc = torch.tensor([scores_by_id[i] for i in ids], dtype=torch.float32)
        payload = {
            "image_ids": ids,
            "localizability": loc,
            "scores_by_id": scores_by_id,
            "window_start": window_start,
            "window_end": resolved_window_end,
            "total_images": n_images_to_eval,
            "seed": seed,
            "num_monte_carlo_samples": num_monte_carlo_samples,
            "dataset": dataset_name,
        }
        tmp = f"{shard_path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, shard_path)
        return payload

    todo = [(i, img) for i, (img, iid) in enumerate(zip(source_images, source_image_ids))
            if iid not in scores_by_id]
    pbar = tqdm_module.tqdm(total=len(source_images), desc="Computing localizability")
    pbar.update(len(source_images) - len(todo))
    for done, (i, img) in enumerate(todo, start=1):
        with torch.inference_mode():
            score = pipeline.compute_localizability(
                img, number_monte_carlo_samples=num_monte_carlo_samples
            ).item()
        scores_by_id[source_image_ids[i]] = score
        pbar.update(1)
        if done % save_every == 0:
            _save_shard()
    pbar.close()

    payload = _save_shard()
    print(f"Saved {len(scores_by_id)} localizability scores to: {shard_path}")
    return payload


def merge_localizability_scores(
    dataset_name: str,
    results_dir: str = "./results",
    shards_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Stitch every localizability shard into the full-dataset ``<dataset>_localizability.pt``.

    Reads each per-window shard, unions the ``scores_by_id`` maps (later shards win on
    a clash, which only happens for overlapping windows of an identical score), and
    writes the merged file the plot stage consumes. The merged ``image_ids`` are sorted
    for a stable on-disk ordering; the plot stage re-aligns by id to each attack's order.
    """
    shards_dir = shards_dir or os.path.join(results_dir, LOCALIZABILITY_SHARDS_SUBDIR)
    shard_files = sorted(glob.glob(os.path.join(shards_dir, f"{dataset_name}_loc__w*.pt")))
    if not shard_files:
        raise FileNotFoundError(
            f"No localizability shards found under {shards_dir} for dataset '{dataset_name}'. "
            f"Run the compute stage first."
        )

    scores_by_id: Dict[str, float] = {}
    seeds, totals = set(), set()
    for path in shard_files:
        shard = torch.load(path, map_location="cpu")
        scores_by_id.update(shard.get("scores_by_id", {}))
        if shard.get("seed") is not None:
            seeds.add(shard["seed"])
        if shard.get("total_images") is not None:
            totals.add(shard["total_images"])

    if len(seeds) > 1:
        print(f"Warning: merging localizability shards with differing seeds {sorted(seeds)}.")

    ids = sorted(scores_by_id.keys())
    loc = torch.tensor([scores_by_id[i] for i in ids], dtype=torch.float32)
    payload = {
        "image_ids": ids,
        "localizability": loc,
        "scores_by_id": scores_by_id,
        "seed": next(iter(seeds)) if len(seeds) == 1 else sorted(seeds),
        "total_images": next(iter(totals)) if len(totals) == 1 else sorted(totals),
        "dataset": dataset_name,
        "n_shards": len(shard_files),
    }
    out_path = _localizability_path(results_dir, dataset_name)
    torch.save(payload, out_path)
    print(f"Merged {len(shard_files)} shards -> {len(ids)} localizability scores at: {out_path}")
    return payload


def _load_localizability_scores(dataset_name: str, results_dir: str) -> Dict[str, float]:
    """Return the merged ``id -> localizability`` map, merging shards on the fly if needed."""
    merged_path = _localizability_path(results_dir, dataset_name)
    if os.path.exists(merged_path):
        return dict(torch.load(merged_path, map_location="cpu")["scores_by_id"])
    # Fall back to merging shards (e.g. plot run directly after compute, no merge step).
    return dict(merge_localizability_scores(dataset_name, results_dir)["scores_by_id"])


def _reindex_attack_result(attack_result: Dict[str, Any], id_to_col: Dict[str, int], canonical_ids: List[str]) -> Dict[str, Any]:
    """Reorder an attack result's per-image tensors onto ``canonical_ids`` (by image id).

    Attack results store metric tensors with an image axis aligned to their own
    ``image_ids``. The plot reads ``res["attack_results"][attack]`` and indexes the image
    axis directly, so we slice each per-image tensor to the shared, ordered id list.
    """
    cols = [id_to_col[i] for i in canonical_ids]
    n_images = len(id_to_col)
    out: Dict[str, Any] = {}
    for key, value in attack_result.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[-1] == n_images:
            out[key] = value.detach().cpu()[..., cols]
        elif key in ("image_ids", "image_indices"):
            continue  # rewritten below
        else:
            out[key] = value
    out["image_ids"] = list(canonical_ids)
    out["image_indices"] = list(range(len(canonical_ids)))
    return out


def build_localizability_dataset_result(
    dataset_name: str,
    attack_types: Sequence[str],
    results_dir: str,
    attack_suffix: str = "",
) -> Dict[str, Any]:
    """Join precomputed localizability with on-disk attack results for one dataset.

    Loads ``<dataset>_localizability.pt`` (merging shards if absent) and each
    ``<dataset>_<attack>_results{suffix}.pt``, then restricts to the image ids common to
    the localizability scores and every requested attack, preserving the first attack's
    ordering. Returns ``{"attack_results": {attack: ...}, "localizability": tensor,
    "image_ids": [...]}`` in the shape ``plot_localizability_results`` expects.
    """
    from core import ResultsManager

    loc_by_id = _load_localizability_scores(dataset_name, results_dir)
    manager = ResultsManager(results_dir, results_dir)

    loaded: Dict[str, Dict[str, Any]] = {}
    for attack in attack_types:
        try:
            loaded[attack] = manager.load_results(dataset_name, attack, suffix=attack_suffix)
        except FileNotFoundError:
            print(f"Warning: no results for attack '{attack}' on '{dataset_name}'; skipping it.")
    if not loaded:
        raise FileNotFoundError(
            f"No attack results found for {dataset_name} in {results_dir} "
            f"(looked for {list(attack_types)})."
        )

    # Canonical ordering: the first available attack's ids, kept only where every attack
    # and the localizability scores all have the image.
    first_attack = next(iter(loaded))
    base_ids = list(loaded[first_attack]["image_ids"])
    common = set(loc_by_id)
    for res in loaded.values():
        common &= set(res["image_ids"])
    canonical_ids = [i for i in base_ids if i in common]
    if not canonical_ids:
        raise ValueError(
            f"No image ids shared between the localizability scores and the attack "
            f"results for '{dataset_name}'. Were they computed with the same seed/pool?"
        )
    dropped = len(base_ids) - len(canonical_ids)
    if dropped:
        print(f"[{dataset_name}] {len(canonical_ids)} images shared across localizability + "
              f"{len(loaded)} attacks ({dropped} dropped for missing scores/results).")

    attack_results: Dict[str, Any] = {}
    for attack, res in loaded.items():
        id_to_col = {iid: col for col, iid in enumerate(res["image_ids"])}
        attack_results[attack] = _reindex_attack_result(res, id_to_col, canonical_ids)

    localizability = torch.tensor([loc_by_id[i] for i in canonical_ids], dtype=torch.float32)
    return {
        "attack_results": attack_results,
        "localizability": localizability,
        "image_ids": canonical_ids,
    }


def plot_localizability_vs_attacks(
    datasets: Sequence[str],
    attack_types: Sequence[str],
    attack_budgets: Sequence[float],
    results_dir: str,
    plot_dir: str,
    plot_budgets: Optional[Sequence[float]] = None,
    attack_suffix: str = "",
) -> None:
    """Stage 2: join precomputed localizability with attack results and plot per budget.

    For each dataset that has both a localizability file and attack results on disk,
    builds the joined result and feeds the (one or two) datasets to
    ``plot_localizability_results`` (rows = datasets, columns = attacks). One figure is
    produced per budget in ``plot_budgets`` (default: every budget in ``attack_budgets``).
    """
    from utils.plots_adversarial_attacks import plot_localizability_results

    all_datasets_results: Dict[str, Any] = {}
    for dataset_name in datasets:
        try:
            all_datasets_results[dataset_name] = build_localizability_dataset_result(
                dataset_name, attack_types, results_dir, attack_suffix=attack_suffix
            )
        except FileNotFoundError as exc:
            print(f"Skipping '{dataset_name}': {exc}")
    if not all_datasets_results:
        raise FileNotFoundError(
            "No dataset had both localizability scores and attack results; nothing to plot."
        )

    budgets_to_plot = list(plot_budgets) if plot_budgets else list(attack_budgets)
    os.makedirs(plot_dir, exist_ok=True)
    for budget in budgets_to_plot:
        print(f"Plotting localizability vs attack strength at budget {budget:.4f} "
              f"({round(budget * 255)}/255)...")
        plot_localizability_results(
            attack_budgets=budget,
            plot_dir=plot_dir,
            all_datasets_results=all_datasets_results,
            results_attack_budgets=list(attack_budgets),
        )


# --------------------------------------------------------------------------- #
# Standalone ablations (share the helpers above)
# --------------------------------------------------------------------------- #


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
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Measure how the best displacement found grows with the number of restarts.

    Runs the standard evaluation engine with ``num_restarts = max_restarts`` and
    derives the ablation from the per-restart evaluations it already collects — the
    same code path used by ``evaluate_attack_on_dataset``. Returns the JSON dict
    (also written to ``results_dir``); the caller plots it.
    """
    from core import EvaluationConfig, EvaluationRunner, run_evaluation

    seed_everything(seed)
    dataset_roots = dataset_roots or {}
    if isinstance(attack_types, str):
        attack_types = [attack_types]
    attack_types = list(attack_types)
    attack_budgets = list(attack_budgets)

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))
    # Force max_restarts so the ablation reaches the requested depth.
    attack_kwargs = [{**kw, "num_restarts": int(max_restarts)} for kw in attack_kwargs]

    config = EvaluationConfig(
        dataset=dataset_name,
        seed=seed,
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        n_images=n_images_to_eval,
        results_dir=results_dir,
        plots_dir=results_dir,
        stored_metrics=["final_step_displacement_predicted"],
        parallel_workers=1,
        use_cuda_streams=False,
        dataset_roots=dataset_roots,
        state_suffix="_restarts",
        attack_type_kwargs=attack_type_kwargs or {},
    )

    runner = EvaluationRunner(config, pipeline)
    if config_dump is not None:
        runner.save_run_config(config_dump, suffix="_restarts")
    run_evaluation(runner)

    json_results, _ = build_and_save_restart_ablation(
        runner, dataset_name, attack_types, attack_budgets, results_dir, max_restarts=max_restarts
    )
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
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Train attacks once, then re-evaluate each perturbation at several step counts.

    Returns the JSON dict (also written to ``results_dir``); the caller plots it.
    """
    from core import ImageLoader
    from attacks.attacks import run_attack

    seed_everything(seed)
    dataset_roots = dataset_roots or {}
    eval_num_steps = list(eval_num_steps)

    if isinstance(attack_types, str):
        attack_types = [attack_types]
    attack_types = list(attack_types)
    attack_budgets = list(attack_budgets)

    source_images, _, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))
    device = str(attack_kwargs[0].get("device", "cuda"))
    eval_cfg = float(attack_kwargs[0].get("restart_eval_cfg", 10.0))
    eval_batch_size = int(attack_kwargs[0].get("restart_eval_batch_size", 128))

    # Phase 1: train attacks, keep the best delta per (attack, budget, image).
    deltas = {at: {bi: {} for bi in range(len(attack_budgets))} for at in attack_types}
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            print(f"Training {attack_type} attacks (eps={budget:.4f})...")
            for image_idx, image in enumerate(tqdm_module.tqdm(source_images, desc="  images")):
                kw = dict(attack_kwargs[budget_idx])
                kw.update((attack_type_kwargs or {}).get(attack_type, {}))
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=pipeline,
                    eps_max=budget,
                    silent=True,
                    **kw,
                )
                deltas[attack_type][budget_idx][image_idx] = result["delta"].detach().cpu()

    # Phase 2: re-evaluate every delta at each num_steps (shared helper).
    samples = {
        at: {bi: {ns: [] for ns in eval_num_steps} for bi in range(len(attack_budgets))}
        for at in attack_types
    }
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            for image_idx, image in enumerate(
                tqdm_module.tqdm(source_images, desc=f"  {attack_type} eps={budget:.4f} eval steps")
            ):
                disp_by_steps = evaluate_delta_at_steps(
                    pipeline=pipeline,
                    source_image=image,
                    delta=deltas[attack_type][budget_idx][image_idx],
                    eval_num_steps=eval_num_steps,
                    cfg=eval_cfg,
                    batch_size=eval_batch_size,
                    seed=seed,
                    device=device,
                )
                for ns in eval_num_steps:
                    samples[attack_type][budget_idx][ns].append(disp_by_steps[ns])

    json_results = build_sampling_steps_json(
        dataset_name,
        attack_types,
        attack_budgets,
        eval_num_steps,
        list(success_rate_thresholds),
        n_images_to_eval,
        samples,
    )
    _save_json(json_results, results_dir, f"{dataset_name}_sampling_steps_results.json")
    return json_results


def evaluate_robustness(
    attack_types,
    pipeline,
    dataset_name: str,
    seed: int = 0,
    n_images_to_eval: int = 20,
    jpeg_quality_factors: list = (10, 20, 30, 40, 50, 60),
    gaussian_blur_sigmas: list = (0, 2, 4, 6, 8, 10),
    robustness_num_steps: Optional[int] = None,
    results_dir: str = "./results",
    attack_budgets: list = (2/255, 15/255, 50/255),
    attack_kwargs: list = (),
    success_rate_thresholds: list = (200, 750, 2500),
    dataset_roots: dict = None,
    config_dump: dict = None,
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Train attacks once, then re-evaluate each perturbation under JPEG/blur.

    Standalone counterpart to the robustness ablation folded into
    ``evaluate_attack_on_dataset``. Unlike that path this trains every requested
    ``attack_types`` (no per-attack gating) since the caller chose them explicitly.
    Each protected image is degraded at the GeoShield levels and re-evaluated at the
    baseline sampling-step count. Returns the JSON dict (also written to
    ``results_dir``); the caller plots it.
    """
    from core import ImageLoader
    from attacks.attacks import run_attack

    seed_everything(seed)
    dataset_roots = dataset_roots or {}
    jpeg_quality_factors = list(jpeg_quality_factors)
    gaussian_blur_sigmas = list(gaussian_blur_sigmas)

    if isinstance(attack_types, str):
        attack_types = [attack_types]
    attack_types = list(attack_types)
    attack_budgets = list(attack_budgets)

    source_images, source_gps, _ = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )

    attack_kwargs = expand_per_budget_kwargs(list(attack_kwargs), len(attack_budgets))
    device = str(attack_kwargs[0].get("device", "cuda"))
    eval_cfg = float(attack_kwargs[0].get("restart_eval_cfg", 10.0))
    eval_batch_size = int(attack_kwargs[0].get("restart_eval_batch_size", 128))

    def _accumulate(target: dict, cell) -> None:
        pred, true = pred_true_from_cell(cell)
        if pred is not None:
            target["predicted"].append(pred)
        if true is not None:
            target["true"].append(true)

    samples = empty_robustness_samples(
        attack_types, len(attack_budgets), jpeg_quality_factors, gaussian_blur_sigmas
    )
    for attack_type in attack_types:
        for budget_idx, budget in enumerate(attack_budgets):
            kw_base = dict(attack_kwargs[budget_idx])
            kw_base.update((attack_type_kwargs or {}).get(attack_type, {}))
            # Baseline sampling steps: explicit override, else the attack's eval steps.
            num_steps = robustness_num_steps
            if num_steps is None:
                num_steps = kw_base.get("restart_eval_num_steps")
            print(f"Training {attack_type} attacks (eps={budget:.4f}) for robustness ablation...")
            for image_idx, image in enumerate(tqdm_module.tqdm(source_images, desc="  images")):
                result = run_attack(
                    attack_type=attack_type,
                    source_image=image,
                    pipeline=pipeline,
                    eps_max=budget,
                    silent=True,
                    **kw_base,
                )
                true_gps = source_gps[image_idx] if source_gps is not None else None
                disp_by_transform = evaluate_delta_under_transforms(
                    pipeline=pipeline,
                    source_image=image,
                    delta=result["delta"].detach().cpu(),
                    jpeg_quality_factors=jpeg_quality_factors,
                    gaussian_blur_sigmas=gaussian_blur_sigmas,
                    cfg=eval_cfg,
                    batch_size=eval_batch_size,
                    seed=seed,
                    device=device,
                    num_steps=int(num_steps) if num_steps is not None else None,
                    true_gps=true_gps,
                )
                for quality in jpeg_quality_factors:
                    _accumulate(
                        samples[attack_type][budget_idx]["jpeg"][int(quality)],
                        disp_by_transform["jpeg"][int(quality)],
                    )
                for sigma in gaussian_blur_sigmas:
                    _accumulate(
                        samples[attack_type][budget_idx]["blur"][float(sigma)],
                        disp_by_transform["blur"][float(sigma)],
                    )

    json_results = build_robustness_json(
        dataset_name,
        attack_types,
        attack_budgets,
        jpeg_quality_factors,
        gaussian_blur_sigmas,
        list(success_rate_thresholds),
        n_images_to_eval,
        samples,
        num_steps=robustness_num_steps,
    )
    _save_json(json_results, results_dir, f"{dataset_name}_robustness_results.json")
    return json_results


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
    """Sampling-steps ablation for a precomputed attack (e.g. GeoShield).

    Loads clean/attacked pairs from folders (one pair of dirs per budget), evaluates
    each pair at every num_steps, and writes the same JSON shape as
    ``evaluate_sampling_steps`` so the plots and merge helper are reused.
    """
    from PIL import Image as PILImage
    from utils.adversarial_utils import collect_common_image_pairs

    seed_everything(seed)
    eval_num_steps = list(eval_num_steps)
    attack_budgets = list(attack_budgets)

    if len(clean_image_dirs) != len(attack_budgets) or len(attacked_image_dirs) != len(attack_budgets):
        raise ValueError(
            "clean_image_dirs, attacked_image_dirs, and attack_budgets must have the same length"
        )

    pairs_by_budget = []
    for budget_idx, (clean_dir, attacked_dir) in enumerate(zip(clean_image_dirs, attacked_image_dirs)):
        pairs = collect_common_image_pairs(clean_dir, attacked_dir)
        if n_images is not None:
            pairs = pairs[: max(0, int(n_images))]
        if not pairs:
            raise ValueError(f"No matched image pairs found for budget index {budget_idx}")
        pairs_by_budget.append(pairs)

    n_images_actual = min(len(p) for p in pairs_by_budget)

    # samples[attack_name][budget_idx][num_steps] = [displacement_km, ...]
    samples = {attack_name: {bi: {ns: [] for ns in eval_num_steps} for bi in range(len(attack_budgets))}}
    for budget_idx, budget in enumerate(attack_budgets):
        pairs = pairs_by_budget[budget_idx]
        for image_idx, (clean_path, attacked_path, _) in enumerate(
            tqdm_module.tqdm(pairs, desc=f"  {attack_name} eps={budget:.4f}")
        ):
            with PILImage.open(clean_path) as f:
                clean_image = f.convert("RGB")
            with PILImage.open(attacked_path) as f:
                attacked_image = f.convert("RGB")
            for num_steps in eval_num_steps:
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
                samples[attack_name][budget_idx][num_steps].append(
                    float(eval_result["metrics"]["final_step_displacement"])
                )

    json_results = build_sampling_steps_json(
        dataset_name,
        [attack_name],
        attack_budgets,
        eval_num_steps,
        list(success_rate_thresholds),
        n_images_actual,
        samples,
    )
    _save_json(json_results, results_dir, f"{dataset_name}_{attack_name}_sampling_steps_results.json")
    return json_results


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
