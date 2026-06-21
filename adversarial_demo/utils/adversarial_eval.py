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

import json
import os
from typing import Any, Dict, Optional, Sequence

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
    build_restart_ablation_json,
    build_sampling_steps_json,
    evaluate_delta_at_steps,
    restart_displacements_from_results,
)
from utils.adversarial_utils import (
    expand_per_budget_kwargs,
    run_paired_pipeline_with_shared_noise,
    seed_everything,
)
from utils.plots_adversarial_attacks import (
    plot_attack_success_rate,
    plot_restarts_success,
    plot_results,
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
) -> Dict[str, Any]:
    """Aggregate the per-image sampling-steps samples collected during the run."""
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
    _save_json(json_results, results_dir, f"{dataset_name}_sampling_steps_results.json")
    return json_results


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
    plot_gps_true: bool = False,
    config_dump: Optional[Dict[str, Any]] = None,
    max_restarts: Optional[int] = None,
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[list[int]] = None,
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Evaluate one or more attacks on images from a test dataset.

    A single training pass yields the main results plus both ablations:

    - **Main results**: best-restart displacement per image → displacement and
      success-rate plots.
    - **Restart ablation** (always): best-displacement-after-k-restarts, derived
      for free from the per-restart evaluations collected during training. Reaches
      depth ``num_restarts`` (override per run with ``max_restarts``).
    - **Sampling-steps ablation** (opt-in via ``run_sampling_steps_ablation``):
      re-evaluates each image's best perturbation at every count in
      ``eval_num_steps`` (extra pipeline runs, no retraining).

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
        attack_kwargs = [{**kw, "num_restarts": int(max_restarts)} for kw in attack_kwargs]

    success_rate_thresholds = plot_success_rate_thresholds or [2500]

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
    )

    runner = EvaluationRunner(config, pipeline)
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
    attack_type_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """Evaluate how attack strength varies with the localizability of the source image.

    Localizability is computed on the clean image; attack strength is final-step
    displacement. Images can be bucketed into low/med/high localizability and the
    average attack strength compared per bucket, attack budget, and attack type.
    """
    from core import EvaluationConfig, EvaluationRunner, sequential_evaluate_attacks, ImageLoader

    seed_everything(seed)

    dataset_roots = dataset_roots or {}

    print(f"Loading {n_images_to_eval} images from {dataset_name} dataset...")
    source_images, source_gps, source_image_ids = ImageLoader.load_images(
        dataset=dataset_name,
        n_images=n_images_to_eval,
        seed=seed,
        dataset_roots=dataset_roots,
    )

    attack_kwargs = expand_per_budget_kwargs(attack_kwargs, len(attack_budgets))

    print("Computing localizability scores...")
    localizability = torch.zeros(len(source_images))
    pbar = tqdm_module.tqdm(total=len(source_images), desc="Computing localizability")
    for i, img in enumerate(source_images):
        localizability[i] = pipeline.compute_localizability(img, number_monte_carlo_samples=256).item()
        pbar.update(1)
    pbar.close()

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
        attack_type_kwargs=attack_type_kwargs or {},
    )

    runner = EvaluationRunner(config, pipeline)
    if config_dump is not None:
        runner.save_run_config(config_dump, suffix="_localizability")
    sequential_evaluate_attacks(runner)

    results = {
        "attack_results": runner.metrics_collector.get_results(),
        "localizability": localizability,
    }
    runner.results_manager.save_metrics(results, dataset_name, suffix="_localizability")


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
