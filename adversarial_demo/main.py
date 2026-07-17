#!/usr/bin/env python3
"""
Command-line interface for adversarial attack experiments.

Commands:
  - evaluate-dataset: Evaluate attacks on a dataset (main results + restart ablation,
    optional sampling-steps / robustness ablations, all from a single training pass).
    GeoShield can be folded in as just another attack by listing `geoshield` in
    --attack-types (generated out-of-process, then overlaid in the combined plots).
  - evaluate-dataset-shard: Run ONE shard (one attack x one image window) of a
    multi-node dataset evaluation. Used by the SLURM array in scripts/cluster/.
  - merge-shards: Stitch all per-shard outputs back into the full-dataset results,
    ablation JSONs, and plots (the artifacts evaluate-dataset would have produced).
  - evaluate-localizability: Attack effectiveness by image localizability, in two stages:
    --stage compute precomputes + stores per-image localizability scores (optionally one
    image window per cluster job), --stage merge stitches the window shards, and
    --stage plot joins the scores with the separately-produced attack results by image id
    and renders the localizability-vs-attack-strength figure.
  - evaluate-geoshield-vs-diffusion: Evaluate precomputed clean/attacked pairs (the
    standalone GeoShield path; the integrated path above is usually preferred)
  - evaluate-restarts / evaluate-sampling-steps / evaluate-robustness: standalone ablations
  - evaluate-sampling-steps-precomputed: sampling-steps ablation for precomputed pairs
  - plot: Plot saved results or attack success rates
  - list-configs: List available config parameters

Usage:
  python main.py evaluate-dataset --dataset yfcc --attack-types encoder diffusion
  python main.py evaluate-dataset --dataset yfcc --run-sampling-steps-ablation
  python main.py evaluate-dataset --dataset yfcc --attack-types dtd --run-robustness-ablation
  python main.py evaluate-dataset --dataset yfcc --attack-types dtd encoder geoshield
  python main.py evaluate-localizability --stage compute --dataset osv
  python main.py evaluate-localizability --stage plot --dataset yfcc --attack-types dtd encoder
  python main.py plot success-rate --dataset yfcc
"""

import argparse
import copy
import glob
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
import torch

from utils.pipe_trajectory import PlonkPipelineTrajectory
from utils.adversarial_eval import (
    evaluate_attack_on_dataset,
    evaluate_attack_shard,
    compute_localizability_scores,
    merge_localizability_scores,
    plot_localizability_vs_attacks,
    DEFAULT_LOCALIZABILITY_MC_SAMPLES,
    evaluate_restarts,
    evaluate_robustness,
    evaluate_sampling_steps,
    evaluate_sampling_steps_precomputed,
    merge_sampling_steps_results,
    merge_shards,
)
from utils.adversarial_utils import seed_everything, expand_to_budget_count
from utils.plots_adversarial_attacks import (
    plot_results,
    plot_attack_success_rate,
    plot_attack_dtd_variance,
    plot_clean_vs_attacked_displacement,
    plot_loss_vs_fsd,
    plot_restarts_success,
    plot_robustness_results,
    plot_model_transfer_success_rate,
    plot_sampling_steps_success_rate,
)
from utils.plots_tikz import (
    plot_results_tikz,
    plot_attack_success_rate_tikz,
    plot_attack_dtd_variance_tikz,
    plot_clean_vs_attacked_displacement_tikz,
    plot_loss_vs_fsd_tikz,
    plot_robustness_results_tikz,
    plot_model_transfer_success_rate_tikz,
    plot_sampling_steps_success_rate_tikz,
    plot_robustness_and_sampling_steps_tikz,
    plot_localizability_vs_attacks_tikz,
)
from core import (
    PrecomputedPairEvaluationConfig,
    PrecomputedPairEvaluationRunner,
    run_precomputed_pair_evaluation,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_PLOTS_DIR = Path(__file__).with_name("plots")
DEFAULT_ATTACK_TYPES = ["encoder", "diffusion"]
DEFAULT_STORED_METRICS = ["final_step_displacement_predicted", "final_step_displacement_true"]
DEFAULT_SUCCESS_RATE_THRESHOLDS = [200, 750, 2500]
DEFAULT_GEOSHIELD_ATTACK_NAME = "geoshield"
DEFAULT_EVAL_NUM_STEPS = [10, 25, 50, 100, 250]
# Guidance-scale (cfg) ablation: classifier-free guidance scales the attacked image is
# re-evaluated at (holding sampling steps at the baseline).
DEFAULT_EVAL_CFGS = [1.0, 2.0, 5.0, 10.0]
DEFAULT_MAX_RESTARTS = 10
DEFAULT_IMAGES_PER_SHARD = 100
# Robustness ablation defaults (GeoShield Fig. 6 levels; scoped to "dtd" by default).
DEFAULT_ROBUSTNESS_ATTACK_TYPES = ["dtd"]
DEFAULT_ROBUSTNESS_JPEG_QUALITY_FACTORS = [10, 20, 30, 40, 50, 60]
DEFAULT_ROBUSTNESS_GAUSSIAN_BLUR_SIGMAS = [0, 2, 4, 6, 8, 10]
# Cross-model transfer ablation: model_type config values for the PLONK variants the
# attacked image is re-evaluated against ("" -> Riemannian flow matching).
DEFAULT_MODEL_TRANSFER_TYPES = ["", "diffusion", "flow"]


# --------------------------------------------------------------------------- #
# Config loading and override handling
# --------------------------------------------------------------------------- #


def _parse_override_value(value_str: str) -> Any:
    """Parse override values using YAML rules, matching OmegaConf-style typing."""
    if value_str == "":
        return ""
    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        return value_str


def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a dotted key path inside a nested config mapping (OmegaConf dotlist style)."""
    parts = key_path.split(".")
    current: Any = config

    for part in parts[:-1]:
        if not isinstance(current, dict):
            raise TypeError(f"Cannot override nested key '{key_path}' because '{part}' is not a mapping")
        if part not in current or current[part] is None:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise TypeError(
                f"Cannot override nested key '{key_path}' because '{part}' is not a mapping"
            )
        current = current[part]

    if not isinstance(current, dict):
        raise TypeError(f"Cannot override key '{key_path}' because parent is not a mapping")
    current[parts[-1]] = value


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load YAML configuration file."""
    config_file = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if not config_file.is_absolute() and not config_file.exists():
        candidate = DEFAULT_CONFIG_PATH.parent / config_file
        if candidate.exists():
            config_file = candidate

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def merge_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge command-line overrides into config dictionary (dotted keys = nested)."""
    for key, value in overrides.items():
        if "." in key:
            _set_nested_value(config, key, value)
            continue
        config[key] = value
    return config


def parse_override_arg(arg: str) -> tuple[str, Any]:
    """Parse a single override argument of the form 'key=value' (value parsed as YAML)."""
    if "=" not in arg:
        raise ValueError(f"Invalid override format: {arg}. Expected 'key=value'")
    key, value_str = arg.split("=", 1)
    return key.strip(), _parse_override_value(value_str.strip())


def get_nested_config(config: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested configuration value with a fallback."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def pick_value(arg_value: Any, config_value: Any, default: Any) -> Any:
    """Prefer CLI input, then config, then a final fallback."""
    if arg_value is not None:
        return arg_value
    if config_value is not None:
        return config_value
    return default


# --------------------------------------------------------------------------- #
# Pipeline / device / attack-kwargs helpers
# --------------------------------------------------------------------------- #


def get_device(config: Dict[str, Any]) -> str:
    """Get device from config, defaulting to cuda if available."""
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    return device


def get_pipeline_for_model_type(
    config: Dict[str, Any], dataset: str, model_type: str
) -> PlonkPipelineTrajectory:
    """Load the PLONK pipeline for ``dataset`` and an explicit ``model_type``.

    ``model_type`` is the architecture suffix appended to the dataset's pipeline name
    ("" -> Riemannian flow matching, "diffusion", "flow"). Used both for the main run
    (the configured ``model_type``) and for the cross-model transfer ablation, which
    loads the other variants.
    """
    device = get_device(config)

    pipelines = config.get("pipelines", {})
    if dataset not in pipelines:
        raise ValueError(f"No pipeline configuration for dataset: {dataset}")

    model_name = pipelines[dataset]
    if model_type:
        suffix = f"_{model_type}"  # riemannian FM is just "" (no suffix)
        if not model_name.endswith(suffix):
            model_name = model_name + suffix
    print(f"Loading pipeline: {model_name}")
    return PlonkPipelineTrajectory(model_name).to(device)


def get_pipeline(config: Dict[str, Any], dataset: str) -> PlonkPipelineTrajectory:
    """Load and initialize the PLONK pipeline for the given dataset (configured model_type)."""
    return get_pipeline_for_model_type(config, dataset, config.get("model_type", "") or "")


def load_transfer_pipelines(
    config: Dict[str, Any],
    dataset: str,
    model_transfer_types: Sequence[str],
    attack_pipeline,
) -> Dict[str, Any]:
    """Load the PLONK variants for the cross-model transfer ablation.

    Returns ``{model_label: pipeline}`` over ``model_transfer_types`` (model_type config
    values). The attack's own variant is reused (``attack_pipeline``) rather than reloaded,
    so the attacked model is part of the comparison without doubling its memory. The other
    variants are loaded once and shared across the run. Loading several PLONK backbones at
    once raises GPU memory; reduce ``model_transfer_types`` if it does not fit.
    """
    from utils.ablations import model_type_label

    attack_model_type = config.get("model_type", "") or ""
    transfer_pipelines: Dict[str, Any] = {}
    for model_type in model_transfer_types:
        mt = model_type or ""
        label = model_type_label(mt)
        if label in transfer_pipelines:
            continue
        if mt == attack_model_type:
            transfer_pipelines[label] = attack_pipeline
        else:
            transfer_pipelines[label] = get_pipeline_for_model_type(config, dataset, mt)
    return transfer_pipelines


def normalize_model_transfer_types(model_transfer_types: Optional[Sequence[str]]) -> List[str]:
    """Map CLI/config model_type tokens to canonical suffixes, de-duplicating in order.

    Accepts ``rfm`` / ``riemannian`` as friendly aliases for the suffix-less Riemannian
    flow-matching variant ("") so users need not pass an empty string on the command line.
    """
    normalized: List[str] = []
    for token in (model_transfer_types or []):
        mt = "" if str(token).strip().lower() in {"rfm", "riemannian", "riemannian_flow_matching", ""} else str(token)
        if mt not in normalized:
            normalized.append(mt)
    return normalized


def get_attack_kwargs(config: Dict[str, Any], dataset: str) -> List[Dict[str, Any]]:
    """Return the shared base attack kwargs for the selected dataset.

    The optional ``per_attack`` sub-mapping is stripped out here and surfaced via
    ``get_attack_type_kwargs`` so it is applied per attack type, not globally.
    """
    device = get_device(config)
    global_seed = int(config.get("seed", 0))

    base_kwargs = dict(config.get("attack_train_args", {}).get(dataset, {}))
    base_kwargs.pop("per_attack", None)  # applied per attack type, not shared
    base_kwargs["device"] = device
    restart_eval_seed = base_kwargs.get("restart_eval_seed")
    if restart_eval_seed is None or restart_eval_seed == "seed":
        base_kwargs["restart_eval_seed"] = global_seed
    else:
        base_kwargs["restart_eval_seed"] = int(restart_eval_seed)

    return [base_kwargs]


def get_attack_type_kwargs(config: Dict[str, Any], dataset: str) -> Dict[str, Dict[str, Any]]:
    """Return per-attack-type kwarg overrides from ``attack_train_args.<dataset>.per_attack``.

    These are merged on top of the shared base only for the matching attack, so e.g.
    ACE's target_image / l2_target loss / alpha never leak into the untargeted attacks.
    """
    per_attack = config.get("attack_train_args", {}).get(dataset, {}).get("per_attack", {}) or {}
    return {attack_type: dict(overrides) for attack_type, overrides in per_attack.items()}


def get_attack_budgets(config: Dict[str, Any], dataset: str) -> List[float]:
    """Return configured attack budgets for the dataset, or raise if missing."""
    attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError(f"No attack budgets configured for dataset: {dataset}")
    return attack_budgets


def print_run_banner(title: str, **fields: Any) -> None:
    """Print a uniform header block for a command."""
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")
    for key, value in fields.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"{'='*60}\n")


@dataclass
class TrainingRunContext:
    """Resolved parameters shared by every attack-training command."""
    dataset: str
    attack_types: List[str]
    n_images: int
    attack_budgets: List[float]
    results_dir: str
    plots_dir: str
    seed: int
    pipeline: Any
    attack_kwargs: List[Dict[str, Any]]
    attack_type_kwargs: Dict[str, Dict[str, Any]]


def prepare_training_run(args, config: Dict[str, Any], default_n_images: int) -> TrainingRunContext:
    """Resolve the common parameters, seed, and load the pipeline for a training command."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), default_n_images)
    attack_budgets = get_attack_budgets(config, dataset)
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)
    attack_kwargs = get_attack_kwargs(config, dataset)
    attack_type_kwargs = get_attack_type_kwargs(config, dataset)

    return TrainingRunContext(
        dataset=dataset,
        attack_types=attack_types,
        n_images=n_images,
        attack_budgets=attack_budgets,
        results_dir=results_dir,
        plots_dir=plots_dir,
        seed=seed,
        pipeline=pipeline,
        attack_kwargs=attack_kwargs,
        attack_type_kwargs=attack_type_kwargs,
    )


def _ensure_dict(config: Dict[str, Any], key: str) -> Dict[str, Any]:
    """Return ``config[key]`` as a dict, replacing a missing/None value with a fresh one."""
    value = config.get(key)
    if not isinstance(value, dict):
        value = {}
        config[key] = value
    return value


def build_resolved_config_dump(
    config: Dict[str, Any],
    ctx: "TrainingRunContext",
    *,
    parallel_workers: Any = None,
    eval_num_steps: Any = None,
    max_restarts: Any = None,
    run_sampling_steps_ablation: Any = None,
    run_robustness_ablation: Any = None,
    run_model_transfer_ablation: Any = None,
    robustness_attack_types: Any = None,
    model_transfer_types: Any = None,
    run_cfg_ablation: Any = None,
    eval_cfgs: Any = None,
) -> Dict[str, Any]:
    """Return a deep copy of ``config`` with the CLI-resolved run parameters written back in.

    ``config`` only ever holds the file (and ``--override``) values; the named CLI flags
    (``--n-images``, ``--attack-types``, ``--results-dir``, ``--eval-num-steps`` ...) are
    resolved separately into ``ctx`` and local variables and are never merged back. Dumping
    ``config`` as the run config therefore misreports the parameters actually used. This
    rebuilds a dict that reflects the effective run so the saved ``*_run_config.yaml``
    matches what ran.
    """
    dump = copy.deepcopy(config)
    dump["dataset"] = ctx.dataset
    dump["attack_types"] = list(ctx.attack_types)
    dump["n_images_to_eval"] = ctx.n_images
    dump["results_dir"] = ctx.results_dir
    dump["plots_dir"] = ctx.plots_dir
    _ensure_dict(dump, "attack_budgets")[ctx.dataset] = list(ctx.attack_budgets)
    if parallel_workers is not None:
        dump["parallel_workers"] = parallel_workers
    if eval_num_steps is not None:
        dump["eval_num_steps"] = list(eval_num_steps)
    if eval_cfgs is not None:
        dump["eval_cfgs"] = list(eval_cfgs)
    if max_restarts is not None:
        dump["max_restarts"] = max_restarts
    plot = _ensure_dict(dump, "plot")
    if run_sampling_steps_ablation is not None:
        plot["run_sampling_steps_ablation"] = bool(run_sampling_steps_ablation)
    if run_cfg_ablation is not None:
        plot["run_cfg_ablation"] = bool(run_cfg_ablation)
    if run_robustness_ablation is not None:
        plot["run_robustness_ablation"] = bool(run_robustness_ablation)
    if run_model_transfer_ablation is not None:
        plot["run_model_transfer_ablation"] = bool(run_model_transfer_ablation)
    if robustness_attack_types is not None:
        _ensure_dict(dump, "robustness")["attack_types"] = list(robustness_attack_types)
    if model_transfer_types is not None:
        _ensure_dict(dump, "model_transfer")["model_types"] = list(model_transfer_types)
    return dump


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def cmd_evaluate_dataset(args, config: Dict[str, Any]) -> None:
    """Evaluate attacks on a dataset (main results + restart/sampling-steps ablations).

    GeoShield, an out-of-process attack, is folded in here as "just another attack":
    list ``geoshield`` in ``--attack-types`` (or set ``geoshield.enabled``) and it is
    generated + evaluated after the trainable attacks and overlaid in the combined plot.
    """
    ctx = prepare_training_run(args, config, default_n_images=100)

    # GeoShield is generated out-of-process, so split it off from the trainable attacks.
    geoshield_cfg = config.get("geoshield", {}) or {}
    geoshield_name = geoshield_cfg.get("attack_name", "geoshield")
    run_geoshield = bool(
        get_nested_config(config, "geoshield", "enabled", default=False)
    ) or (geoshield_name in ctx.attack_types)
    trainable_attack_types = [at for at in ctx.attack_types if at != geoshield_name]

    parallel_workers = pick_value(args.parallel_workers, config.get("parallel_workers"), 1)
    plot_gps_true = bool(get_nested_config(config, "plot", "gps_true", default=True))
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    # Ablation controls (CLI > config > default).
    run_sampling_steps_ablation = bool(pick_value(
        args.run_sampling_steps_ablation,
        get_nested_config(config, "plot", "run_sampling_steps_ablation", default=None),
        False,
    ))
    eval_num_steps = pick_value(args.eval_num_steps, config.get("eval_num_steps"), DEFAULT_EVAL_NUM_STEPS)
    max_restarts = pick_value(args.max_restarts, config.get("max_restarts"), None)

    # Robustness ablation controls (CLI > config > default). Scoped per attack type.
    run_robustness_ablation = bool(pick_value(
        args.run_robustness_ablation,
        get_nested_config(config, "plot", "run_robustness_ablation", default=None),
        False,
    ))
    robustness_attack_types = pick_value(
        args.robustness_attack_types,
        get_nested_config(config, "robustness", "attack_types", default=None),
        DEFAULT_ROBUSTNESS_ATTACK_TYPES,
    )
    robustness_jpeg_quality_factors = get_nested_config(
        config, "robustness", "jpeg_quality_factors", default=DEFAULT_ROBUSTNESS_JPEG_QUALITY_FACTORS
    )
    robustness_gaussian_blur_sigmas = get_nested_config(
        config, "robustness", "gaussian_blur_sigmas", default=DEFAULT_ROBUSTNESS_GAUSSIAN_BLUR_SIGMAS
    )
    robustness_num_steps = get_nested_config(config, "robustness", "num_steps", default=None)

    # Cross-model transfer ablation controls (CLI > config > default).
    from utils.ablations import model_type_label
    run_model_transfer_ablation = bool(pick_value(
        args.run_model_transfer_ablation,
        get_nested_config(config, "plot", "run_model_transfer_ablation", default=None),
        False,
    ))
    model_transfer_types = normalize_model_transfer_types(pick_value(
        args.model_transfer_types,
        get_nested_config(config, "model_transfer", "model_types", default=None),
        DEFAULT_MODEL_TRANSFER_TYPES,
    ))
    model_transfer_num_steps = pick_value(
        args.model_transfer_num_steps,
        get_nested_config(config, "model_transfer", "num_steps", default=None),
        None,
    )

    # Guidance-scale (cfg) ablation controls (CLI > config > default). Runs for every
    # evaluated attack (like sampling-steps), not scoped per attack type.
    run_cfg_ablation = bool(pick_value(
        args.run_cfg_ablation,
        get_nested_config(config, "plot", "run_cfg_ablation", default=None),
        False,
    ))
    eval_cfgs = pick_value(args.eval_cfgs, config.get("eval_cfgs"), DEFAULT_EVAL_CFGS)
    cfg_num_steps = pick_value(args.cfg_num_steps, config.get("cfg_num_steps"), None)
    # Load the other PLONK variants once and reuse the attack pipeline for its own variant.
    transfer_pipelines = (
        load_transfer_pipelines(config, ctx.dataset, model_transfer_types, ctx.pipeline)
        if run_model_transfer_ablation and trainable_attack_types
        else {}
    )
    attacked_model_label = model_type_label(config.get("model_type", "") or "")

    print_run_banner(
        f"Evaluating attacks on {ctx.dataset.upper()} dataset",
        attack_types=ctx.attack_types,
        attack_budgets=ctx.attack_budgets,
        images_to_evaluate=ctx.n_images,
        results_directory=ctx.results_dir,
        plots_directory=ctx.plots_dir,
        parallel_workers=parallel_workers,
        sampling_steps_ablation=run_sampling_steps_ablation,
        robustness_ablation=run_robustness_ablation,
        robustness_attack_types=robustness_attack_types if run_robustness_ablation else None,
        model_transfer_ablation=run_model_transfer_ablation,
        model_transfer_variants=list(transfer_pipelines.keys()) if run_model_transfer_ablation else None,
        geoshield=run_geoshield,
        max_restarts=max_restarts,
    )

    # Reflect the CLI-resolved parameters in the saved run config (config.yaml alone would
    # misreport --n-images / --attack-types / --results-dir / --eval-num-steps ...).
    resolved_config_dump = build_resolved_config_dump(
        config,
        ctx,
        parallel_workers=parallel_workers,
        eval_num_steps=eval_num_steps,
        max_restarts=max_restarts,
        run_sampling_steps_ablation=run_sampling_steps_ablation,
        run_robustness_ablation=run_robustness_ablation,
        run_model_transfer_ablation=run_model_transfer_ablation,
        robustness_attack_types=robustness_attack_types,
        model_transfer_types=model_transfer_types,
        run_cfg_ablation=run_cfg_ablation,
        eval_cfgs=eval_cfgs,
    )

    if trainable_attack_types:
        evaluate_attack_on_dataset(
            attack_types=trainable_attack_types,
            pipeline=ctx.pipeline,
            dataset_name=ctx.dataset,
            source_image=None,
            seed=ctx.seed,
            use_real_gps=pick_value(args.use_real_gps, config.get("use_real_gps"), False),
            n_images_to_eval=ctx.n_images,
            plot_dir=ctx.plots_dir,
            results_dir=ctx.results_dir,
            attack_budgets=ctx.attack_budgets,
            stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
            attack_kwargs=ctx.attack_kwargs,
            parallel_workers=parallel_workers,
            use_cuda_streams=bool(config.get("use_cuda_streams", True)),
            dataset_roots=config.get("data_dirs", {}),
            plot_success_rate=plot_success_rate,
            plot_success_rate_thresholds=success_rate_thresholds,
            plot_gps_true=plot_gps_true,
            config_dump=resolved_config_dump,
            max_restarts=max_restarts,
            run_sampling_steps_ablation=run_sampling_steps_ablation,
            eval_num_steps=eval_num_steps,
            attack_type_kwargs=ctx.attack_type_kwargs,
            run_robustness_ablation=run_robustness_ablation,
            robustness_attack_types=robustness_attack_types,
            robustness_jpeg_quality_factors=robustness_jpeg_quality_factors,
            robustness_gaussian_blur_sigmas=robustness_gaussian_blur_sigmas,
            robustness_num_steps=robustness_num_steps,
            run_model_transfer_ablation=run_model_transfer_ablation,
            model_transfer_types=model_transfer_types,
            model_transfer_num_steps=model_transfer_num_steps,
            transfer_pipelines=transfer_pipelines,
            attacked_model_label=attacked_model_label,
            run_cfg_ablation=run_cfg_ablation,
            eval_cfgs=eval_cfgs,
            cfg_num_steps=cfg_num_steps,
        )

    # GeoShield: generate out-of-process, evaluate the pairs, overlay in combined plots.
    if run_geoshield:
        _run_geoshield_step(
            config=config,
            config_dump=resolved_config_dump,
            ctx=ctx,
            geoshield_cfg=geoshield_cfg,
            geoshield_name=geoshield_name,
            trainable_attack_types=trainable_attack_types,
            plot_success_rate=plot_success_rate,
            success_rate_thresholds=success_rate_thresholds,
            plot_gps_true=plot_gps_true,
            ablation_controls=dict(
                run_sampling_steps_ablation=run_sampling_steps_ablation,
                eval_num_steps=eval_num_steps,
                run_robustness_ablation=run_robustness_ablation,
                robustness_attack_types=robustness_attack_types,
                robustness_jpeg_quality_factors=robustness_jpeg_quality_factors,
                robustness_gaussian_blur_sigmas=robustness_gaussian_blur_sigmas,
                robustness_num_steps=robustness_num_steps,
                run_cfg_ablation=run_cfg_ablation,
                eval_cfgs=eval_cfgs,
                cfg_num_steps=cfg_num_steps,
            ),
        )

    print(f"\nEvaluation complete! Results saved to: {ctx.results_dir}")
    print(f"Plots saved to: {ctx.plots_dir}")


def _run_geoshield_step(
    config: Dict[str, Any],
    ctx: "TrainingRunContext",
    geoshield_cfg: Dict[str, Any],
    geoshield_name: str,
    trainable_attack_types: List[str],
    plot_success_rate: bool,
    success_rate_thresholds: List[float],
    plot_gps_true: bool,
    ablation_controls: Optional[Dict[str, Any]] = None,
    config_dump: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate + evaluate GeoShield, then re-plot the combined results (trainable + GeoShield).

    GeoShield selects labelled dataset images, so the GPS map returned by the generator
    is threaded into the evaluator to also report the true-position displacement metric
    (like the trainable attacks). The combined plots honour ``plot.gps_true``. When the
    sampling-steps / robustness ablations are enabled, they are recorded for GeoShield's
    pairs and overlaid onto the trainable ablation plots (matching the multi-GPU merge).
    """
    from utils.geoshield import generate_geoshield_pairs
    from utils.adversarial_eval import overlay_precomputed_ablations

    ablation_controls = ablation_controls or {}
    # Robustness is opt-in per attack type (as for the trainable attacks): only for GeoShield
    # when it is listed in robustness.attack_types. Sampling-steps applies to every attack.
    run_robustness_geoshield = bool(ablation_controls.get("run_robustness_ablation")) and (
        geoshield_name in (ablation_controls.get("robustness_attack_types") or [])
    )
    run_sampling_steps = bool(ablation_controls.get("run_sampling_steps_ablation"))
    eval_num_steps = ablation_controls.get("eval_num_steps")
    run_cfg = bool(ablation_controls.get("run_cfg_ablation"))
    eval_cfgs = ablation_controls.get("eval_cfgs")
    cfg_num_steps = ablation_controls.get("cfg_num_steps")

    clean_dirs, attacked_dirs, gps_by_filename = generate_geoshield_pairs(
        config=config,
        dataset=ctx.dataset,
        n_images=ctx.n_images,
        attack_budgets=ctx.attack_budgets,
        seed=ctx.seed,
        geoshield_cfg=geoshield_cfg,
    )
    runner = run_precomputed_attack_eval(
        pipeline=ctx.pipeline,
        dataset=ctx.dataset,
        attack_name=geoshield_name,
        attack_budgets=ctx.attack_budgets,
        clean_image_dirs=clean_dirs,
        attacked_image_dirs=attacked_dirs,
        n_images=ctx.n_images,
        seed=ctx.seed,
        device=get_device(config),
        results_dir=ctx.results_dir,
        plots_dir=ctx.plots_dir,
        stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
        run_config=config_dump if config_dump is not None else config,
        gps_by_id=gps_by_filename,
        run_sampling_steps_ablation=run_sampling_steps,
        eval_num_steps=eval_num_steps,
        run_robustness_ablation=run_robustness_geoshield,
        robustness_jpeg_quality_factors=ablation_controls.get("robustness_jpeg_quality_factors"),
        robustness_gaussian_blur_sigmas=ablation_controls.get("robustness_gaussian_blur_sigmas"),
        robustness_num_steps=ablation_controls.get("robustness_num_steps"),
        run_cfg_ablation=run_cfg,
        eval_cfgs=eval_cfgs,
        cfg_num_steps=cfg_num_steps,
    )

    # Overlay GeoShield with the trainable attacks in one combined set of plots.
    combined_attack_types = trainable_attack_types + [geoshield_name]
    plot_results(
        results_dir=ctx.results_dir,
        attack_budgets=ctx.attack_budgets,
        plot_dir=ctx.plots_dir,
        dataset_name=ctx.dataset,
        attack_types=combined_attack_types,
        all_results=None,
        stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
    )
    if plot_success_rate:
        plot_attack_success_rate(
            results_dir=ctx.results_dir,
            attack_budgets=ctx.attack_budgets,
            plot_dir=ctx.plots_dir,
            dataset_name=ctx.dataset,
            attack_types=combined_attack_types,
            threshold_km=list(success_rate_thresholds),
            gps_true=plot_gps_true,
        )

    # Overlay GeoShield onto the trainable sampling-steps / robustness / cfg ablation plots.
    if run_sampling_steps or run_robustness_geoshield or run_cfg:
        overlay_precomputed_ablations(
            runner=runner,
            dataset_name=ctx.dataset,
            attack_budgets=ctx.attack_budgets,
            results_dir=ctx.results_dir,
            plots_dir=ctx.plots_dir,
            success_rate_thresholds=success_rate_thresholds,
            run_sampling_steps_ablation=run_sampling_steps,
            eval_num_steps=eval_num_steps,
            run_robustness_ablation=run_robustness_geoshield,
            robustness_jpeg_quality_factors=ablation_controls.get("robustness_jpeg_quality_factors"),
            robustness_gaussian_blur_sigmas=ablation_controls.get("robustness_gaussian_blur_sigmas"),
            robustness_num_steps=ablation_controls.get("robustness_num_steps"),
            run_cfg_ablation=run_cfg,
            eval_cfgs=eval_cfgs,
            cfg_num_steps=cfg_num_steps,
        )


def _resolve_ablation_controls(args, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the ablation controls shared by evaluate-dataset / -shard / merge (CLI > config > default)."""
    return dict(
        run_sampling_steps_ablation=bool(pick_value(
            args.run_sampling_steps_ablation,
            get_nested_config(config, "plot", "run_sampling_steps_ablation", default=None),
            False,
        )),
        eval_num_steps=pick_value(args.eval_num_steps, config.get("eval_num_steps"), DEFAULT_EVAL_NUM_STEPS),
        max_restarts=pick_value(args.max_restarts, config.get("max_restarts"), None),
        run_robustness_ablation=bool(pick_value(
            args.run_robustness_ablation,
            get_nested_config(config, "plot", "run_robustness_ablation", default=None),
            False,
        )),
        robustness_attack_types=pick_value(
            args.robustness_attack_types,
            get_nested_config(config, "robustness", "attack_types", default=None),
            DEFAULT_ROBUSTNESS_ATTACK_TYPES,
        ),
        robustness_jpeg_quality_factors=get_nested_config(
            config, "robustness", "jpeg_quality_factors", default=DEFAULT_ROBUSTNESS_JPEG_QUALITY_FACTORS
        ),
        robustness_gaussian_blur_sigmas=get_nested_config(
            config, "robustness", "gaussian_blur_sigmas", default=DEFAULT_ROBUSTNESS_GAUSSIAN_BLUR_SIGMAS
        ),
        robustness_num_steps=get_nested_config(config, "robustness", "num_steps", default=None),
        run_model_transfer_ablation=bool(pick_value(
            args.run_model_transfer_ablation,
            get_nested_config(config, "plot", "run_model_transfer_ablation", default=None),
            False,
        )),
        model_transfer_types=pick_value(
            args.model_transfer_types,
            get_nested_config(config, "model_transfer", "model_types", default=None),
            DEFAULT_MODEL_TRANSFER_TYPES,
        ),
        model_transfer_num_steps=pick_value(
            args.model_transfer_num_steps,
            get_nested_config(config, "model_transfer", "num_steps", default=None),
            None,
        ),
        run_cfg_ablation=bool(pick_value(
            args.run_cfg_ablation,
            get_nested_config(config, "plot", "run_cfg_ablation", default=None),
            False,
        )),
        eval_cfgs=pick_value(args.eval_cfgs, config.get("eval_cfgs"), DEFAULT_EVAL_CFGS),
        cfg_num_steps=pick_value(args.cfg_num_steps, config.get("cfg_num_steps"), None),
        success_rate_thresholds=get_nested_config(
            config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
        ),
        stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
    )


def _resolve_shard_window(args, total_images: int, config: Dict[str, Any]) -> tuple[int, Optional[int]]:
    """Resolve the [window_start:window_end] image slice for a shard from the CLI args.

    Either an explicit --window-start/--window-end, or --image-shard-index combined with
    --images-per-shard (window k = [k*ips, (k+1)*ips)). Both are clamped to total_images.
    """
    if args.window_start is not None or args.window_end is not None:
        window_start = int(args.window_start or 0)
        window_end = None if args.window_end is None else int(args.window_end)
    else:
        if args.image_shard_index is None:
            raise ValueError(
                "Provide either --image-shard-index (with --images-per-shard) or "
                "--window-start/--window-end to select the image window."
            )
        images_per_shard = int(pick_value(
            args.images_per_shard, config.get("images_per_shard"), DEFAULT_IMAGES_PER_SHARD
        ))
        window_start = int(args.image_shard_index) * images_per_shard
        window_end = window_start + images_per_shard

    window_start = min(window_start, total_images)
    if window_end is not None:
        window_end = min(window_end, total_images)
    return window_start, window_end


def _resolve_localizability_window(args, total_images: int, config: Dict[str, Any]) -> tuple[int, Optional[int]]:
    """Resolve the localizability compute window from the CLI args.

    Like ``_resolve_shard_window`` but optional: with no windowing flags the whole pool
    ``[0:total_images]`` is computed in one job. Otherwise an explicit
    ``--window-start``/``--window-end`` or ``--image-shard-index`` (+ ``--images-per-shard``)
    selects a contiguous slice of the seeded ordering for a cluster array task.
    """
    if args.window_start is not None or args.window_end is not None:
        window_start = int(args.window_start or 0)
        window_end = None if args.window_end is None else int(args.window_end)
    elif args.image_shard_index is not None:
        images_per_shard = int(pick_value(
            args.images_per_shard, config.get("images_per_shard"), DEFAULT_IMAGES_PER_SHARD
        ))
        window_start = int(args.image_shard_index) * images_per_shard
        window_end = window_start + images_per_shard
    else:
        window_start, window_end = 0, None  # whole pool in one job

    window_start = min(window_start, total_images)
    if window_end is not None:
        window_end = min(window_end, total_images)
    return window_start, window_end


def _run_geoshield_shard(
    config: Dict[str, Any],
    dataset: str,
    seed: int,
    total_images: int,
    window_start: int,
    window_end: Optional[int],
    window_end_label: int,
    base_results_dir: str,
    attack_budgets: List[float],
    geoshield_name: str,
    ctrls: Optional[Dict[str, Any]] = None,
) -> None:
    """Generate + evaluate GeoShield for one image window, saving a per-shard state file.

    Mirrors the trainable-attack shard, but GeoShield is produced out-of-process: each
    shard selects the same seeded image window, generates its perturbations into shard-
    isolated clean/attacked dirs (the ``run_tag`` keeps concurrent shards from clobbering
    one another), and evaluates the pairs with the shared precomputed-pair runner. The
    sampling-steps / robustness ablations (when enabled in ``ctrls``) are recorded on the
    attacked pairs too, so ``merge-shards`` overlays GeoShield in those ablation plots.
    The resulting state file is stitched in by ``merge-shards`` like any other attack.
    """
    from utils.geoshield import generate_geoshield_pairs

    geoshield_cfg = config.get("geoshield", {}) or {}
    ctrls = ctrls or {}
    # Robustness is opt-in per attack type (like the trainable path): only run it for
    # GeoShield when it is listed in robustness.attack_types. Sampling-steps applies to all.
    run_robustness_geoshield = bool(ctrls.get("run_robustness_ablation")) and (
        geoshield_name in (ctrls.get("robustness_attack_types") or [])
    )
    run_tag = f"w{window_start:06d}_{window_end_label:06d}"
    shard_results_dir = os.path.join(base_results_dir, "shards", f"{geoshield_name}__{run_tag}")

    print_run_banner(
        f"[shard] Generating + evaluating GeoShield on {dataset.upper()} dataset",
        attack_budgets=attack_budgets,
        image_window=f"[{window_start}:{window_end_label}] of {total_images}",
        results_directory=shard_results_dir,
    )

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)
    clean_dirs, attacked_dirs, gps_by_filename = generate_geoshield_pairs(
        config=config,
        dataset=dataset,
        n_images=total_images,
        attack_budgets=attack_budgets,
        seed=seed,
        geoshield_cfg=geoshield_cfg,
        window_start=window_start,
        window_end=window_end,
        run_tag=run_tag,
    )
    run_precomputed_attack_eval(
        pipeline=pipeline,
        dataset=dataset,
        attack_name=geoshield_name,
        attack_budgets=attack_budgets,
        clean_image_dirs=clean_dirs,
        attacked_image_dirs=attacked_dirs,
        n_images=None,  # the clean dir already contains exactly this window
        seed=seed,
        device=get_device(config),
        results_dir=shard_results_dir,
        plots_dir=shard_results_dir,
        # GeoShield has the dataset's true GPS, so keep the true-position metric too.
        stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
        run_config=config,
        gps_by_id=gps_by_filename,
        run_sampling_steps_ablation=bool(ctrls.get("run_sampling_steps_ablation")),
        eval_num_steps=ctrls.get("eval_num_steps"),
        run_robustness_ablation=run_robustness_geoshield,
        robustness_jpeg_quality_factors=ctrls.get("robustness_jpeg_quality_factors"),
        robustness_gaussian_blur_sigmas=ctrls.get("robustness_gaussian_blur_sigmas"),
        robustness_num_steps=ctrls.get("robustness_num_steps"),
        run_cfg_ablation=bool(ctrls.get("run_cfg_ablation")),
        eval_cfgs=ctrls.get("eval_cfgs"),
        cfg_num_steps=ctrls.get("cfg_num_steps"),
    )
    print(f"\nGeoShield shard complete! Results saved to: {shard_results_dir}")


def cmd_evaluate_dataset_shard(args, config: Dict[str, Any]) -> None:
    """Run ONE shard of a multi-node dataset evaluation: the given attack(s) on an image window.

    Writes only this shard's raw results + resumable state into a per-shard subdirectory
    (``<results_dir>/shards/<attacks>__w<start>_<end>/``); no plotting. ``merge-shards``
    later stitches every shard back into the full results, ablations, and plots.
    """
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)

    attack_budgets = get_attack_budgets(config, dataset)
    seed = int(config.get("seed", 0))
    total_images = int(pick_value(
        args.total_images, config.get("total_images"), config.get("n_images_to_eval", 100)
    ))
    window_start, window_end = _resolve_shard_window(args, total_images, config)
    window_end_label = window_end if window_end is not None else total_images
    if window_start >= window_end_label:
        print(f"Shard window [{window_start}:{window_end_label}] is empty (pool size {total_images}); nothing to do.")
        return

    base_results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    ctrls = _resolve_ablation_controls(args, config)

    # GeoShield is out-of-process: it gets its own shard branch (one attack per shard).
    geoshield_name = get_nested_config(config, "geoshield", "attack_name", default="geoshield")
    if geoshield_name in attack_types:
        if len(attack_types) > 1:
            raise ValueError(
                f"'{geoshield_name}' must be sharded on its own (one attack per shard); "
                f"got --attack-types {attack_types}."
            )
        _run_geoshield_shard(
            config=config,
            dataset=dataset,
            seed=seed,
            total_images=total_images,
            window_start=window_start,
            window_end=window_end,
            window_end_label=window_end_label,
            base_results_dir=base_results_dir,
            attack_budgets=attack_budgets,
            geoshield_name=geoshield_name,
            ctrls=ctrls,
        )
        return

    shard_tag = "-".join(attack_types)
    shard_results_dir = os.path.join(
        base_results_dir, "shards", f"{shard_tag}__w{window_start:06d}_{window_end_label:06d}"
    )

    parallel_workers = pick_value(args.parallel_workers, config.get("parallel_workers"), 1)
    use_real_gps = pick_value(args.use_real_gps, config.get("use_real_gps"), False)

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)
    attack_kwargs = get_attack_kwargs(config, dataset)
    attack_type_kwargs = get_attack_type_kwargs(config, dataset)

    # Cross-model transfer ablation: load the other PLONK variants once for this shard.
    run_model_transfer_ablation = bool(ctrls.get("run_model_transfer_ablation"))
    model_transfer_types = normalize_model_transfer_types(ctrls.get("model_transfer_types"))
    transfer_pipelines = (
        load_transfer_pipelines(config, dataset, model_transfer_types, pipeline)
        if run_model_transfer_ablation
        else {}
    )

    print_run_banner(
        f"[shard] Evaluating attacks on {dataset.upper()} dataset",
        attack_types=attack_types,
        attack_budgets=attack_budgets,
        image_window=f"[{window_start}:{window_end_label}] of {total_images}",
        results_directory=shard_results_dir,
        parallel_workers=parallel_workers,
        sampling_steps_ablation=ctrls["run_sampling_steps_ablation"],
        robustness_ablation=ctrls["run_robustness_ablation"],
        model_transfer_ablation=run_model_transfer_ablation,
        model_transfer_variants=list(transfer_pipelines.keys()) if run_model_transfer_ablation else None,
        max_restarts=ctrls["max_restarts"],
    )

    evaluate_attack_shard(
        attack_types=attack_types,
        pipeline=pipeline,
        dataset_name=dataset,
        seed=seed,
        total_images=total_images,
        window_start=window_start,
        window_end=window_end,
        results_dir=shard_results_dir,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        stored_metrics=ctrls["stored_metrics"],
        parallel_workers=parallel_workers,
        use_cuda_streams=bool(config.get("use_cuda_streams", True)),
        use_real_gps=use_real_gps,
        dataset_roots=config.get("data_dirs", {}),
        attack_type_kwargs=attack_type_kwargs,
        max_restarts=ctrls["max_restarts"],
        run_sampling_steps_ablation=ctrls["run_sampling_steps_ablation"],
        eval_num_steps=ctrls["eval_num_steps"],
        success_rate_thresholds=ctrls["success_rate_thresholds"],
        run_robustness_ablation=ctrls["run_robustness_ablation"],
        robustness_attack_types=ctrls["robustness_attack_types"],
        robustness_jpeg_quality_factors=ctrls["robustness_jpeg_quality_factors"],
        robustness_gaussian_blur_sigmas=ctrls["robustness_gaussian_blur_sigmas"],
        robustness_num_steps=ctrls["robustness_num_steps"],
        run_model_transfer_ablation=run_model_transfer_ablation,
        model_transfer_types=model_transfer_types,
        model_transfer_num_steps=ctrls.get("model_transfer_num_steps"),
        transfer_pipelines=transfer_pipelines,
        config_dump=config,
        run_cfg_ablation=ctrls["run_cfg_ablation"],
        eval_cfgs=ctrls["eval_cfgs"],
        cfg_num_steps=ctrls.get("cfg_num_steps"),
    )

    print(f"\nShard complete! Results saved to: {shard_results_dir}")


def cmd_merge_shards(args, config: Dict[str, Any]) -> None:
    """Merge every per-shard output into the full-dataset results, ablations, and plots."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)

    attack_budgets = get_attack_budgets(config, dataset)
    seed = int(config.get("seed", 0))
    total_images = int(pick_value(
        args.total_images, config.get("total_images"), config.get("n_images_to_eval", 100)
    ))
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    shards_dir = args.shards_dir

    # GeoShield is merged like any other attack as long as its shards exist on disk
    # (it is keyed by filename, which merge_shards stem-normalises to the photo id).
    geoshield_name = get_nested_config(config, "geoshield", "attack_name", default="geoshield")
    shards_root = shards_dir or os.path.join(results_dir, "shards")
    merge_attack_types = list(attack_types)
    if geoshield_name in attack_types and not glob.glob(os.path.join(shards_root, f"{geoshield_name}__*")):
        merge_attack_types = [at for at in attack_types if at != geoshield_name]
        print(f"Note: no '{geoshield_name}' shards found under {shards_root}; excluding it from the merge.")

    ctrls = _resolve_ablation_controls(args, config)
    from utils.ablations import model_type_label
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    plot_gps_true = bool(get_nested_config(config, "plot", "gps_true", default=True))

    print_run_banner(
        f"Merging shard results on {dataset.upper()} dataset",
        attack_types=merge_attack_types,
        attack_budgets=attack_budgets,
        total_images=total_images,
        results_directory=results_dir,
        plots_directory=plots_dir,
        shards_directory=shards_dir or os.path.join(results_dir, "shards"),
    )

    merge_shards(
        dataset_name=dataset,
        attack_types=merge_attack_types,
        attack_budgets=attack_budgets,
        total_images=total_images,
        seed=seed,
        results_dir=results_dir,
        plots_dir=plots_dir,
        stored_metrics=ctrls["stored_metrics"],
        dataset_roots=config.get("data_dirs", {}),
        plot_success_rate=plot_success_rate,
        success_rate_thresholds=ctrls["success_rate_thresholds"],
        plot_gps_true=plot_gps_true,
        run_sampling_steps_ablation=ctrls["run_sampling_steps_ablation"],
        eval_num_steps=ctrls["eval_num_steps"],
        run_robustness_ablation=ctrls["run_robustness_ablation"],
        robustness_attack_types=ctrls["robustness_attack_types"],
        robustness_jpeg_quality_factors=ctrls["robustness_jpeg_quality_factors"],
        robustness_gaussian_blur_sigmas=ctrls["robustness_gaussian_blur_sigmas"],
        robustness_num_steps=ctrls["robustness_num_steps"],
        run_model_transfer_ablation=bool(ctrls.get("run_model_transfer_ablation")),
        model_transfer_types=normalize_model_transfer_types(ctrls.get("model_transfer_types")),
        model_transfer_num_steps=ctrls.get("model_transfer_num_steps"),
        attacked_model_label=model_type_label(config.get("model_type", "") or ""),
        config_dump=config,
        shards_dir=shards_dir,
        run_cfg_ablation=ctrls["run_cfg_ablation"],
        eval_cfgs=ctrls["eval_cfgs"],
        cfg_num_steps=ctrls.get("cfg_num_steps"),
    )

    print(f"\nMerge complete! Results in {results_dir}, plots in {plots_dir}.")


def _load_run_attack_budgets(dataset: str, results_dir: str) -> Optional[List[float]]:
    """Return the attack budgets saved with the dataset's results, or None if absent.

    The per-attack result tensors are indexed ``[budget, image]``; using the budgets the
    run actually used keeps the localizability plot's budget axis aligned even if the
    current ``config.yaml`` lists different budgets.
    """
    from core import ResultsManager

    try:
        args = ResultsManager(results_dir, results_dir).load_attack_args(dataset)
    except FileNotFoundError:
        return None
    budgets = args.get("attack_budgets")
    return list(budgets) if budgets else None


def cmd_evaluate_localizability(args, config: Dict[str, Any]) -> None:
    """Evaluate how attack strength varies with image localizability, in two decoupled stages.

    Localizability is a property of the clean image, independent of any attack, so it is
    computed and stored once (``--stage compute``, optionally sharded across cluster jobs
    by image window) and then joined to the separately-produced attack results at plot
    time (``--stage plot``). ``--stage merge`` stitches the per-window compute shards into
    the full-dataset ``<dataset>_localizability.pt`` (the plot stage does this implicitly
    if the merged file is missing).
    """
    stage = getattr(args, "stage", "compute") or "compute"
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))

    if stage == "merge":
        print_run_banner(
            f"Merging localizability shards on {dataset.upper()} dataset",
            results_directory=results_dir,
        )
        merge_localizability_scores(dataset_name=dataset, results_dir=results_dir)
        print(f"\nMerge complete! Localizability scores in: {results_dir}")
        return

    if stage == "plot":
        attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
        # Prefer the budgets actually used by the attack run (saved alongside the results)
        # so the budget axis of the result tensors lines up; fall back to config otherwise.
        attack_budgets = _load_run_attack_budgets(dataset, results_dir) or get_attack_budgets(config, dataset)
        # Combine multiple datasets into one figure (rows) when asked (e.g. yfcc + osv).
        datasets = args.datasets if getattr(args, "datasets", None) else [dataset]
        plot_budgets = args.plot_budgets if getattr(args, "plot_budgets", None) else None

        print_run_banner(
            "Plotting localizability vs attack strength",
            datasets=datasets,
            attack_types=attack_types,
            attack_budgets=attack_budgets,
            plot_budgets=plot_budgets if plot_budgets else "(all budgets)",
            results_directory=results_dir,
            plots_directory=plots_dir,
            tikz=bool(getattr(args, "tikz", False)),
        )
        if getattr(args, "tikz", False):
            plot_localizability_vs_attacks_tikz(
                datasets=datasets,
                attack_types=attack_types,
                attack_budgets=attack_budgets,
                results_dir=results_dir,
                plot_dir=plots_dir,
                plot_budgets=plot_budgets,
            )
        else:
            plot_localizability_vs_attacks(
                datasets=datasets,
                attack_types=attack_types,
                attack_budgets=attack_budgets,
                results_dir=results_dir,
                plot_dir=plots_dir,
                plot_budgets=plot_budgets,
            )
        print(f"\nPlots saved to: {plots_dir}")
        return

    # stage == "compute"
    n_images = pick_value(args.n_images, config.get("total_images"), config.get("n_images_to_eval", 100))
    n_images = int(n_images)
    window_start, window_end = _resolve_localizability_window(args, n_images, config)
    window_end_label = window_end if window_end is not None else n_images
    if window_start >= window_end_label:
        print(f"Localizability window [{window_start}:{window_end_label}] is empty "
              f"(pool size {n_images}); nothing to do.")
        return
    num_mc_samples = int(pick_value(
        args.num_mc_samples,
        get_nested_config(config, "localizability", "num_monte_carlo_samples", default=None),
        DEFAULT_LOCALIZABILITY_MC_SAMPLES,
    ))

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)

    print_run_banner(
        f"Computing localizability on {dataset.upper()} dataset",
        image_window=f"[{window_start}:{window_end_label}] of {n_images}",
        monte_carlo_samples=num_mc_samples,
        results_directory=results_dir,
    )
    compute_localizability_scores(
        pipeline=pipeline,
        dataset_name=dataset,
        seed=seed,
        n_images_to_eval=n_images,
        results_dir=results_dir,
        dataset_roots=config.get("data_dirs", {}),
        window_start=window_start,
        window_end=window_end,
        num_monte_carlo_samples=num_mc_samples,
        config_dump=config,
    )
    print(f"\nLocalizability scores saved under: {results_dir}/localizability_shards")


def cmd_evaluate_restarts(args, config: Dict[str, Any]) -> None:
    """Standalone restart ablation."""
    ctx = prepare_training_run(args, config, default_n_images=1)
    max_restarts = pick_value(args.max_restarts, config.get("max_restarts"), DEFAULT_MAX_RESTARTS)

    print_run_banner(
        f"Evaluating restart sensitivity on {ctx.dataset.upper()} dataset",
        attack_types=ctx.attack_types,
        attack_budgets=ctx.attack_budgets,
        images_to_evaluate=ctx.n_images,
        max_restarts=max_restarts,
        results_directory=ctx.results_dir,
        plots_directory=ctx.plots_dir,
    )

    json_results = evaluate_restarts(
        attack_types=ctx.attack_types,
        pipeline=ctx.pipeline,
        dataset_name=ctx.dataset,
        seed=ctx.seed,
        n_images_to_eval=ctx.n_images,
        max_restarts=max_restarts,
        results_dir=ctx.results_dir,
        attack_budgets=ctx.attack_budgets,
        attack_kwargs=ctx.attack_kwargs,
        dataset_roots=config.get("data_dirs", {}),
        config_dump=config,
        attack_type_kwargs=ctx.attack_type_kwargs,
    )

    plot_restarts_success(json_results=json_results, plot_dir=ctx.plots_dir)

    print(f"\nEvaluation complete! Results saved to: {ctx.results_dir}")
    print(f"Plots saved to: {ctx.plots_dir}")


def cmd_evaluate_sampling_steps(args, config: Dict[str, Any]) -> None:
    """Standalone sampling-steps ablation."""
    ctx = prepare_training_run(args, config, default_n_images=20)
    eval_num_steps = pick_value(args.eval_num_steps, config.get("eval_num_steps"), DEFAULT_EVAL_NUM_STEPS)
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    print_run_banner(
        f"Evaluating sampling-step sensitivity on {ctx.dataset.upper()} dataset",
        attack_types=ctx.attack_types,
        attack_budgets=ctx.attack_budgets,
        images_to_evaluate=ctx.n_images,
        evaluation_step_counts=eval_num_steps,
        results_directory=ctx.results_dir,
        plots_directory=ctx.plots_dir,
    )

    json_results = evaluate_sampling_steps(
        attack_types=ctx.attack_types,
        pipeline=ctx.pipeline,
        dataset_name=ctx.dataset,
        seed=ctx.seed,
        n_images_to_eval=ctx.n_images,
        eval_num_steps=eval_num_steps,
        results_dir=ctx.results_dir,
        attack_budgets=ctx.attack_budgets,
        attack_kwargs=ctx.attack_kwargs,
        success_rate_thresholds=success_rate_thresholds,
        dataset_roots=config.get("data_dirs", {}),
        config_dump=config,
        attack_type_kwargs=ctx.attack_type_kwargs,
    )

    plot_sampling_steps_success_rate(json_results=json_results, plot_dir=ctx.plots_dir)

    print(f"\nEvaluation complete! Results saved to: {ctx.results_dir}")
    print(f"Plots saved to: {ctx.plots_dir}")


def cmd_evaluate_robustness(args, config: Dict[str, Any]) -> None:
    """Standalone robustness ablation (JPEG compression / Gaussian blur)."""
    ctx = prepare_training_run(args, config, default_n_images=20)
    jpeg_quality_factors = pick_value(
        args.jpeg_quality_factors,
        get_nested_config(config, "robustness", "jpeg_quality_factors", default=None),
        DEFAULT_ROBUSTNESS_JPEG_QUALITY_FACTORS,
    )
    gaussian_blur_sigmas = pick_value(
        args.gaussian_blur_sigmas,
        get_nested_config(config, "robustness", "gaussian_blur_sigmas", default=None),
        DEFAULT_ROBUSTNESS_GAUSSIAN_BLUR_SIGMAS,
    )
    robustness_num_steps = pick_value(
        args.robustness_num_steps,
        get_nested_config(config, "robustness", "num_steps", default=None),
        None,
    )
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    print_run_banner(
        f"Evaluating robustness to JPEG/blur on {ctx.dataset.upper()} dataset",
        attack_types=ctx.attack_types,
        attack_budgets=ctx.attack_budgets,
        images_to_evaluate=ctx.n_images,
        jpeg_quality_factors=jpeg_quality_factors,
        gaussian_blur_sigmas=gaussian_blur_sigmas,
        eval_num_steps=robustness_num_steps if robustness_num_steps is not None else "baseline",
        results_directory=ctx.results_dir,
        plots_directory=ctx.plots_dir,
    )

    json_results = evaluate_robustness(
        attack_types=ctx.attack_types,
        pipeline=ctx.pipeline,
        dataset_name=ctx.dataset,
        seed=ctx.seed,
        n_images_to_eval=ctx.n_images,
        jpeg_quality_factors=jpeg_quality_factors,
        gaussian_blur_sigmas=gaussian_blur_sigmas,
        robustness_num_steps=robustness_num_steps,
        results_dir=ctx.results_dir,
        attack_budgets=ctx.attack_budgets,
        attack_kwargs=ctx.attack_kwargs,
        success_rate_thresholds=success_rate_thresholds,
        dataset_roots=config.get("data_dirs", {}),
        config_dump=config,
        attack_type_kwargs=ctx.attack_type_kwargs,
    )

    plot_robustness_results(json_results=json_results, plot_dir=ctx.plots_dir)

    print(f"\nEvaluation complete! Results saved to: {ctx.results_dir}")
    print(f"Plots saved to: {ctx.plots_dir}")


def run_precomputed_attack_eval(
    pipeline,
    dataset: str,
    attack_name: str,
    attack_budgets: List[float],
    clean_image_dirs: List[str],
    attacked_image_dirs: List[str],
    n_images: Optional[int],
    seed: int,
    device: str,
    results_dir: str,
    plots_dir: str,
    stored_metrics: List[str],
    run_config: Optional[Dict[str, Any]] = None,
    gps_by_id: Optional[Dict[str, Any]] = None,
    run_sampling_steps_ablation: bool = False,
    eval_num_steps: Optional[Sequence[int]] = None,
    run_robustness_ablation: bool = False,
    robustness_jpeg_quality_factors: Optional[Sequence[int]] = None,
    robustness_gaussian_blur_sigmas: Optional[Sequence[float]] = None,
    robustness_num_steps: Optional[int] = None,
    run_cfg_ablation: bool = False,
    eval_cfgs: Optional[Sequence[float]] = None,
    cfg_num_steps: Optional[int] = None,
) -> "PrecomputedPairEvaluationRunner":
    """Evaluate precomputed clean/attacked pairs and persist results like any attack.

    Shared by ``evaluate-geoshield-vs-diffusion`` and the GeoShield step folded into
    ``evaluate-dataset``. Returns the runner (use ``.metrics_collector.get_results()`` for
    the per-attack results); plotting is left to the caller so it can overlay GeoShield
    with the other attacks. ``gps_by_id`` (filename -> (lat, lon)) enables the
    true-position displacement metric when labels are available. The sampling-steps /
    robustness ablation flags, when set, record those eval-time ablations for the pairs
    too (the restart ablation does not apply to a single precomputed image).
    """
    precomputed_config = PrecomputedPairEvaluationConfig(
        dataset=dataset,
        attack_name=attack_name,
        seed=seed,
        attack_budgets=list(attack_budgets),
        clean_image_dirs=list(clean_image_dirs),
        attacked_image_dirs=list(attacked_image_dirs),
        results_dir=results_dir,
        plots_dir=plots_dir,
        stored_metrics=stored_metrics,
        device=device,
        n_images=n_images,
        gps_by_id=gps_by_id,
        run_sampling_steps_ablation=run_sampling_steps_ablation,
        eval_num_steps=list(eval_num_steps) if eval_num_steps else None,
        run_robustness_ablation=run_robustness_ablation,
        robustness_jpeg_quality_factors=list(robustness_jpeg_quality_factors) if robustness_jpeg_quality_factors else None,
        robustness_gaussian_blur_sigmas=list(robustness_gaussian_blur_sigmas) if robustness_gaussian_blur_sigmas else None,
        robustness_num_steps=robustness_num_steps,
        run_cfg_ablation=run_cfg_ablation,
        eval_cfgs=[float(c) for c in eval_cfgs] if eval_cfgs else None,
        cfg_num_steps=cfg_num_steps,
    )
    runner = PrecomputedPairEvaluationRunner(precomputed_config, pipeline)
    if run_config is not None:
        runner.save_run_config(run_config, suffix=precomputed_config.state_suffix)
    run_precomputed_pair_evaluation(runner)
    runner.save_results()
    return runner


def _precomputed_stored_metrics(config: Dict[str, Any]) -> List[str]:
    """Stored metrics for precomputed pairs: drop the true-GPS metric (no labels)."""
    stored_metrics = get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS)
    return [m for m in stored_metrics if m != "final_step_displacement_true"] or [
        "final_step_displacement_predicted"
    ]


def _resolve_precomputed_folders(args, config, dataset, attack_budgets):
    """Resolve and validate the clean/attacked folder lists for precomputed evaluation."""
    clean_image_dirs = pick_value(args.clean_image_dirs, config.get("clean_image_dirs"), None)
    attacked_image_dirs = pick_value(args.attacked_image_dirs, config.get("attacked_image_dirs"), None)
    if clean_image_dirs is None or attacked_image_dirs is None:
        raise ValueError(
            "Both --clean-image-dirs and --attacked-image-dirs must be provided, either on the CLI or in the config"
        )
    clean_image_dirs = expand_to_budget_count(clean_image_dirs, len(attack_budgets), "clean_image_dirs")
    attacked_image_dirs = expand_to_budget_count(attacked_image_dirs, len(attack_budgets), "attacked_image_dirs")
    return clean_image_dirs, attacked_image_dirs


def cmd_evaluate_geoshield_vs_diffusion(args, config: Dict[str, Any]) -> None:
    """Evaluate precomputed clean/attacked image pairs and plot the results."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_name = pick_value(args.attack_name, config.get("attack_name"), DEFAULT_GEOSHIELD_ATTACK_NAME)

    attack_budgets = pick_value(args.attack_budgets, config.get("attack_budgets"), None)
    if attack_budgets is None:
        attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError("No attack budgets configured for the precomputed folder evaluation")

    clean_image_dirs, attacked_image_dirs = _resolve_precomputed_folders(args, config, dataset, attack_budgets)

    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), None)
    seed = int(config.get("seed", 0))
    device = get_device(config)

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)

    plot_gps_true = bool(get_nested_config(config, "plot", "gps_true", default=True))
    if plot_gps_true:
        print("Warning: plot.gps_true is enabled, but precomputed folder evaluation does not have ground-truth GPS labels. Using predicted displacement plots instead.")
        plot_gps_true = False
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    resolved_stored_metrics = _precomputed_stored_metrics(config)

    resolved_config = dict(config)
    resolved_config.update(
        {
            "dataset": dataset,
            "attack_name": attack_name,
            "attack_budgets": list(attack_budgets),
            "clean_image_dirs": list(clean_image_dirs),
            "attacked_image_dirs": list(attacked_image_dirs),
            "results_dir": results_dir,
            "plots_dir": plots_dir,
            "n_images_to_eval": n_images,
            "plot": {
                **dict(config.get("plot", {})),
                "gps_true": plot_gps_true,
                "plot_success_rate": plot_success_rate,
                "attack_success_rate_thresholds": list(success_rate_thresholds),
                "stored_metrics": resolved_stored_metrics,
            },
        }
    )

    print_run_banner(
        f"Evaluating precomputed {attack_name} images on {dataset.upper()}",
        attack_budgets=attack_budgets,
        clean_image_folders=clean_image_dirs,
        attacked_image_folders=attacked_image_dirs,
        results_directory=results_dir,
        plots_directory=plots_dir,
        images_to_evaluate=n_images,
    )

    precomputed_runner = run_precomputed_attack_eval(
        pipeline=pipeline,
        dataset=dataset,
        attack_name=attack_name,
        attack_budgets=list(attack_budgets),
        clean_image_dirs=list(clean_image_dirs),
        attacked_image_dirs=list(attacked_image_dirs),
        n_images=n_images,
        seed=seed,
        device=device,
        results_dir=results_dir,
        plots_dir=plots_dir,
        stored_metrics=resolved_stored_metrics,
        run_config=resolved_config,
    )
    all_results = precomputed_runner.metrics_collector.get_results()

    plot_results(
        results_dir=results_dir,
        attack_budgets=list(attack_budgets),
        plot_dir=plots_dir,
        dataset_name=dataset,
        attack_types=[attack_name],
        all_results=all_results,
        stored_metrics=resolved_stored_metrics,
        gps_true=False,
    )

    if plot_success_rate:
        plot_attack_success_rate(
            results_dir=results_dir,
            attack_budgets=list(attack_budgets),
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=[attack_name],
            all_results=all_results,
            threshold_km=list(success_rate_thresholds),
            gps_true=False,
        )

    print(f"\nEvaluation complete! Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")


def cmd_evaluate_sampling_steps_precomputed(args, config: Dict[str, Any]) -> None:
    """Sampling-steps ablation for precomputed clean/attacked pairs (e.g. GeoShield)."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_name = pick_value(args.attack_name, config.get("attack_name"), DEFAULT_GEOSHIELD_ATTACK_NAME)
    eval_num_steps = pick_value(args.eval_num_steps, config.get("eval_num_steps"), DEFAULT_EVAL_NUM_STEPS)

    attack_budgets = pick_value(args.attack_budgets, config.get("attack_budgets"), None)
    if attack_budgets is None:
        attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError("No attack budgets configured for the precomputed sampling-steps evaluation")

    clean_image_dirs, attacked_image_dirs = _resolve_precomputed_folders(args, config, dataset, attack_budgets)

    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), None)
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))
    device = get_device(config)
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)

    print_run_banner(
        f"Evaluating {attack_name} sampling-step sensitivity on {dataset.upper()}",
        attack_budgets=attack_budgets,
        clean_image_folders=clean_image_dirs,
        attacked_image_folders=attacked_image_dirs,
        evaluation_step_counts=eval_num_steps,
        results_directory=results_dir,
        plots_directory=plots_dir,
        images_per_budget=n_images,
    )

    json_results = evaluate_sampling_steps_precomputed(
        attack_name=attack_name,
        pipeline=pipeline,
        dataset_name=dataset,
        clean_image_dirs=list(clean_image_dirs),
        attacked_image_dirs=list(attacked_image_dirs),
        attack_budgets=list(attack_budgets),
        seed=seed,
        eval_num_steps=eval_num_steps,
        n_images=n_images,
        results_dir=results_dir,
        cfg=float(get_nested_config(config, "attack_train_args", dataset, "restart_eval_cfg") or 10.0),
        success_rate_thresholds=success_rate_thresholds,
        device=device,
        config_dump=config,
    )

    plot_sampling_steps_success_rate(json_results=json_results, plot_dir=plots_dir)

    print(f"\nEvaluation complete! Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")


def cmd_plot(args, config: Dict[str, Any]) -> None:
    """Plot saved results."""
    plot_type = pick_value(args.plot_type, config.get("plot_type"), "results")
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    os.makedirs(plots_dir, exist_ok=True)

    print_run_banner(
        f"Generating {plot_type} plots for {dataset.upper()}",
        results_directory=results_dir,
        plots_directory=plots_dir,
    )

    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    gps_true = bool(get_nested_config(config, "plot", "gps_true", default=True))
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )
    tikz = bool(getattr(args, "tikz", False))

    if plot_type == "results":
        attack_budgets = get_attack_budgets(config, dataset)
        print(f"Plotting results for attacks: {attack_types}")
        results_fn = plot_results_tikz if tikz else plot_results
        results_fn(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
            all_results=None,
            stored_metrics=get_nested_config(config, "plot", "stored_metrics", default=DEFAULT_STORED_METRICS),
        )
        if plot_success_rate:
            success_rate_fn = plot_attack_success_rate_tikz if tikz else plot_attack_success_rate
            success_rate_fn(
                results_dir=results_dir,
                attack_budgets=attack_budgets,
                plot_dir=plots_dir,
                dataset_name=dataset,
                attack_types=attack_types,
                gps_true=gps_true,
                threshold_km=success_rate_thresholds,
            )

    elif plot_type == "success-rate":
        attack_budgets = get_attack_budgets(config, dataset)
        print(f"Plotting attack success rates with distance thresholds: {success_rate_thresholds} km")
        success_rate_fn = plot_attack_success_rate_tikz if tikz else plot_attack_success_rate
        success_rate_fn(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
            gps_true=gps_true,
            threshold_km=success_rate_thresholds,
        )

    elif plot_type == "dtd-variance":
        attack_budgets = get_attack_budgets(config, dataset)
        print(f"Plotting attack displacement variance (\"DTD\") for attacks: {attack_types}")
        dtd_variance_fn = plot_attack_dtd_variance_tikz if tikz else plot_attack_dtd_variance
        dtd_variance_fn(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
            gps_true=gps_true,
        )

    elif plot_type == "loss-vs-fsd":
        attack_budgets = get_attack_budgets(config, dataset)
        loss_attack_types = [at for at in attack_types if at in ("dtd", "training_loss", "sampling")] or ["dtd", "training_loss", "sampling"]
        print(f"Plotting training loss vs. achieved FSD for attacks: {loss_attack_types}")
        loss_vs_fsd_fn = plot_loss_vs_fsd_tikz if tikz else plot_loss_vs_fsd
        loss_vs_fsd_fn(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=loss_attack_types,
            gps_true=gps_true,
        )

    elif plot_type == "clean-vs-attacked-displacement":
        attack_budgets = get_attack_budgets(config, dataset)
        print(f"Plotting clean-vs-attacked true-GPS displacement for attacks: {attack_types}")
        clean_vs_attacked_fn = plot_clean_vs_attacked_displacement_tikz if tikz else plot_clean_vs_attacked_displacement
        clean_vs_attacked_fn(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
        )

    elif plot_type == "sampling-steps":
        default_results_file = os.path.join(results_dir, f"{dataset}_sampling_steps_results.json")
        results_files = pick_value(args.results_files, config.get("results_files"), [default_results_file])
        print(f"Merging {len(results_files)} result file(s) for joint sampling-steps plot")
        merged = merge_sampling_steps_results(results_files)
        sampling_steps_fn = plot_sampling_steps_success_rate_tikz if tikz else plot_sampling_steps_success_rate
        sampling_steps_fn(json_results=merged, plot_dir=plots_dir)

    elif plot_type == "robustness":
        results_file = pick_value(
            args.results_files[0] if args.results_files else None,
            config.get("results_file"),
            os.path.join(results_dir, f"{dataset}_robustness_results.json"),
        )
        print(f"Plotting robustness (JPEG/blur) results from: {results_file}")
        with open(results_file) as f:
            json_results = json.load(f)
        robustness_fn = plot_robustness_results_tikz if tikz else plot_robustness_results
        robustness_fn(json_results=json_results, plot_dir=plots_dir)

    elif plot_type == "robustness-and-sampling-steps":
        if not tikz:
            raise ValueError("plot_type 'robustness-and-sampling-steps' is only implemented for --tikz.")
        robustness_file = pick_value(
            args.results_files[0] if args.results_files else None,
            config.get("results_file"),
            os.path.join(results_dir, f"{dataset}_robustness_results.json"),
        )
        sampling_steps_file = pick_value(
            args.results_files[1] if args.results_files and len(args.results_files) > 1 else None,
            None,
            os.path.join(results_dir, f"{dataset}_sampling_steps_results.json"),
        )
        print(f"Plotting combined robustness + sampling-steps results from: {robustness_file}, {sampling_steps_file}")
        with open(robustness_file) as f:
            robustness_json = json.load(f)
        sampling_steps_json = merge_sampling_steps_results([sampling_steps_file])
        plot_robustness_and_sampling_steps_tikz(
            robustness_json=robustness_json, sampling_steps_json=sampling_steps_json, plot_dir=plots_dir,
        )

    elif plot_type == "restarts":
        if tikz:
            raise ValueError("--tikz is not supported for plot_type 'restarts' (not part of the TikZ backend).")
        results_file = pick_value(
            args.results_files[0] if args.results_files else None,
            config.get("results_file"),
            os.path.join(results_dir, f"{dataset}_restarts_results.json"),
        )
        print(f"Plotting restart-success ablation from: {results_file}")
        with open(results_file) as f:
            json_results = json.load(f)
        # Full-dataset runs (e.g. the restart ablation folded into evaluate-dataset)
        # have one image per per-image figure -- skip those and keep only the
        # mean-+-std summary plot. Standalone single/few-image ablations still get
        # their per-image figures.
        per_image = json_results.get("n_images", 1) == 1
        plot_restarts_success(json_results=json_results, plot_dir=plots_dir, per_image=per_image)

    elif plot_type == "model-transfer":
        results_file = pick_value(
            args.results_files[0] if args.results_files else None,
            config.get("results_file"),
            os.path.join(results_dir, f"{dataset}_model_transfer_results.json"),
        )
        print(f"Plotting cross-model transfer results from: {results_file}")
        with open(results_file) as f:
            json_results = json.load(f)
        model_transfer_fn = plot_model_transfer_success_rate_tikz if tikz else plot_model_transfer_success_rate
        model_transfer_fn(json_results=json_results, plot_dir=plots_dir)

    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

    print(f"\nPlots saved to: {plots_dir}")


def cmd_list_configs(args, config: Dict[str, Any]) -> None:
    """List available configuration parameters."""
    print("\n" + "=" * 60)
    print("BASELINE CONFIGURATION")
    print("=" * 60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("\n" + "=" * 60)
    print("CONFIGURATION OVERRIDE EXAMPLES")
    print("=" * 60)
    print("""
Examples of using --override flag:

  # Change device
  --override device=cpu

  # Change number of images to evaluate
  --override n_images_to_eval=50

  # Change attack budgets for YFCC (using list syntax)
  --override 'attack_budgets.yfcc=[0.01, 0.05, 0.1]'

  # Change parallel workers
  --override parallel_workers=4

  # Enable the sampling-steps ablation inside evaluate-dataset
  --override plot.run_sampling_steps_ablation=true
  --override 'eval_num_steps=[16, 64, 250]'
""")


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #


def add_common_eval_args(parser: argparse.ArgumentParser, n_images_help: str = "Number of images to evaluate") -> None:
    """Add the arguments shared by every attack-training command."""
    parser.add_argument("--dataset", choices=["yfcc", "osv"], help="Dataset to evaluate on")
    parser.add_argument("--attack-types", nargs="+", help="Attack types to evaluate (default: encoder diffusion)")
    parser.add_argument("--n-images", type=int, help=n_images_help)
    parser.add_argument("--results-dir", help="Directory to save results")
    parser.add_argument("--plots-dir", help="Directory to save plots")


def add_ablation_args(parser: argparse.ArgumentParser) -> None:
    """Add the ablation controls shared by evaluate-dataset / evaluate-dataset-shard / merge-shards."""
    parser.add_argument("--run-sampling-steps-ablation", action="store_true", default=None,
                        help="Also re-evaluate each best perturbation across --eval-num-steps")
    parser.add_argument("--eval-num-steps", nargs="+", type=int,
                        help=f"Sampling step counts for the ablation (default: {DEFAULT_EVAL_NUM_STEPS})")
    parser.add_argument("--max-restarts", type=int,
                        help="Override num_restarts for this run (sets the restart-ablation depth)")
    parser.add_argument("--run-robustness-ablation", action="store_true", default=None,
                        help="Also degrade each best perturbation with JPEG/blur and re-evaluate "
                             "(only for --robustness-attack-types; baseline sampling steps)")
    parser.add_argument("--robustness-attack-types", nargs="+",
                        help=f"Attack types the robustness ablation runs for (default: {DEFAULT_ROBUSTNESS_ATTACK_TYPES})")
    parser.add_argument("--run-model-transfer-ablation", action="store_true", default=None,
                        help="Also re-evaluate each best perturbation's attacked image against every "
                             "--model-transfer-types PLONK variant (cross-model transfer)")
    parser.add_argument("--model-transfer-types", nargs="+",
                        help="model_type values for the transfer ablation; '' is Riemannian flow matching. "
                             f"Pass rfm for '' (default: {DEFAULT_MODEL_TRANSFER_TYPES})")
    parser.add_argument("--model-transfer-num-steps", type=int,
                        help="Sampling steps for the model-transfer eval (default: each attack's baseline)")
    parser.add_argument("--run-cfg-ablation", action="store_true", default=None,
                        help="Also re-evaluate each best perturbation across --eval-cfgs guidance scales")
    parser.add_argument("--eval-cfgs", nargs="+", type=float,
                        help=f"Guidance scales (cfg) for the cfg ablation (default: {DEFAULT_EVAL_CFGS})")
    parser.add_argument("--cfg-num-steps", type=int,
                        help="Sampling steps for the cfg eval (default: each attack's baseline)")


def add_precomputed_folder_args(parser: argparse.ArgumentParser) -> None:
    """Add the clean/attacked folder arguments shared by the precomputed commands."""
    parser.add_argument("--dataset", choices=["yfcc", "osv"], help="Dataset label used for saving results and plots")
    parser.add_argument("--attack-name", help=f"Label for the evaluated attack (default: {DEFAULT_GEOSHIELD_ATTACK_NAME})")
    parser.add_argument("--attack-budgets", nargs="+", type=float, help="Attack budgets aligned with the clean/attacked folder list")
    parser.add_argument("--clean-image-dirs", nargs="+", help="One clean image directory per budget")
    parser.add_argument("--attacked-image-dirs", nargs="+", help="One attacked image directory per budget")
    parser.add_argument("--n-images", type=int, help="Optional number of images to evaluate after matching pairs")
    parser.add_argument("--results-dir", help="Directory to save results")
    parser.add_argument("--plots-dir", help="Directory to save plots")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        description="Adversarial attack experiments CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate attacks on YFCC dataset (main results + restart ablation)
  python main.py evaluate-dataset --dataset yfcc

  # Also run the sampling-steps ablation in the same pass
  python main.py evaluate-dataset --dataset yfcc \\
    --run-sampling-steps-ablation --eval-num-steps 16 64 250

  # Also run the robustness (JPEG/blur) ablation, scoped to the dtd attack
  python main.py evaluate-dataset --dataset yfcc --attack-types dtd encoder \\
    --run-robustness-ablation --robustness-attack-types dtd

  # Standalone robustness ablation
  python main.py evaluate-robustness --dataset yfcc --attack-types dtd

  # Multi-node: run one shard (attack x image window), then merge everything.
  # Usually launched by the SLURM array in scripts/cluster/ rather than by hand.
  python main.py evaluate-dataset-shard --dataset yfcc --attack-types encoder \\
    --total-images 4000 --images-per-shard 100 --image-shard-index 7 \\
    --results-dir results/yfcc4k_full --run-sampling-steps-ablation
  python main.py merge-shards --dataset yfcc \\
    --attack-types encoder sampling diffusion_l2 dtd unidef unidef_nofdje ace \\
    --total-images 4000 --results-dir results/yfcc4k_full \\
    --plots-dir results/yfcc4k_full/plots --run-sampling-steps-ablation

  # Evaluate localizability
  python main.py evaluate-localizability --dataset yfcc

  # Plot saved results / success rates
  python main.py plot results --dataset yfcc
  python main.py plot success-rate --dataset osv

  # Override config parameters (dot notation + YAML typing)
  python main.py evaluate-dataset --dataset yfcc \\
    --override 'attack_budgets.yfcc=[0.01, 0.05]' \\
    --override parallel_workers=4
        """,
    )

    # Global arguments shared by every subcommand.
    global_parser = argparse.ArgumentParser(add_help=False)
    global_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Path to config file (default: {DEFAULT_CONFIG_PATH})",
    )
    global_parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config parameters (format: key=value). Can be used multiple times.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # evaluate-dataset
    eval_dataset = subparsers.add_parser("evaluate-dataset", help="Evaluate attacks on a dataset", parents=[global_parser])
    add_common_eval_args(eval_dataset)
    eval_dataset.add_argument("--use-real-gps", action="store_true", default=None,
                              help="Use real GPS coordinates from dataset instead of clean predictions")
    eval_dataset.add_argument("--parallel-workers", type=int, help="Number of parallel workers for evaluation")
    add_ablation_args(eval_dataset)

    # evaluate-dataset-shard (one attack x one image window; used by the SLURM array)
    eval_shard = subparsers.add_parser("evaluate-dataset-shard",
                                       help="Run one shard (attack x image window) of a multi-node evaluation",
                                       parents=[global_parser])
    add_common_eval_args(eval_shard, n_images_help="(unused; the window is set via --total-images + --image-shard-index)")
    eval_shard.add_argument("--use-real-gps", action="store_true", default=None,
                            help="Use real GPS coordinates from dataset instead of clean predictions")
    eval_shard.add_argument("--parallel-workers", type=int, help="Number of parallel workers for evaluation")
    eval_shard.add_argument("--total-images", type=int,
                            help="Size of the full seeded image pool the windows are taken from (e.g. 4000)")
    eval_shard.add_argument("--images-per-shard", type=int,
                            help=f"Images per shard window (default: {DEFAULT_IMAGES_PER_SHARD})")
    eval_shard.add_argument("--image-shard-index", type=int,
                            help="0-based window index; window = [k*images_per_shard, (k+1)*images_per_shard)")
    eval_shard.add_argument("--window-start", type=int,
                            help="Explicit window start (overrides --image-shard-index)")
    eval_shard.add_argument("--window-end", type=int,
                            help="Explicit window end (exclusive; overrides --image-shard-index)")
    add_ablation_args(eval_shard)

    # merge-shards (stitch every shard back into the full results + plots)
    merge_cmd = subparsers.add_parser("merge-shards",
                                      help="Merge per-shard outputs into the full-dataset results, ablations, and plots",
                                      parents=[global_parser])
    add_common_eval_args(merge_cmd, n_images_help="(unused; use --total-images)")
    merge_cmd.add_argument("--total-images", type=int,
                           help="Size of the full seeded image pool that was sharded (e.g. 4000)")
    merge_cmd.add_argument("--shards-dir",
                           help="Directory holding the per-shard subdirectories (default: <results-dir>/shards)")
    add_ablation_args(merge_cmd)

    # evaluate-localizability (two stages: compute scores, then plot vs attack results)
    eval_local = subparsers.add_parser("evaluate-localizability",
                                       help="Evaluate attack effectiveness by image localizability",
                                       parents=[global_parser])
    add_common_eval_args(eval_local,
                         n_images_help="Size of the seeded image pool to score (default: total_images / n_images_to_eval)")
    eval_local.add_argument("--stage", choices=["compute", "merge", "plot"], default="compute",
                            help="compute: score images (optionally one window); merge: stitch "
                                 "window shards; plot: join with attack results and plot (default: compute)")
    # Sharding controls for the compute stage (mirror evaluate-dataset-shard).
    eval_local.add_argument("--images-per-shard", type=int,
                            help=f"Images per compute window (default: {DEFAULT_IMAGES_PER_SHARD})")
    eval_local.add_argument("--image-shard-index", type=int,
                            help="0-based window index; window = [k*images_per_shard, (k+1)*images_per_shard)")
    eval_local.add_argument("--window-start", type=int, help="Explicit window start (overrides --image-shard-index)")
    eval_local.add_argument("--window-end", type=int, help="Explicit window end (exclusive)")
    eval_local.add_argument("--num-mc-samples", type=int,
                            help=f"Monte-Carlo samples for the localizability estimate "
                                 f"(default: {DEFAULT_LOCALIZABILITY_MC_SAMPLES})")
    # Plot stage controls.
    eval_local.add_argument("--datasets", nargs="+", choices=["yfcc", "osv"],
                            help="Datasets to stack as rows in the plot (default: just --dataset)")
    eval_local.add_argument("--plot-budgets", nargs="+", type=float,
                            help="Budgets to render one figure each for (default: every configured budget)")
    eval_local.add_argument("--tikz", action="store_true",
                            help="Emit standalone pgfplots .tex (+ best-effort compiled PDF/PNG preview) "
                                 "instead of a matplotlib figure, for --stage plot")

    # evaluate-restarts
    eval_restarts = subparsers.add_parser("evaluate-restarts",
                                          help="Evaluate how attack success varies with the number of restarts",
                                          parents=[global_parser])
    add_common_eval_args(eval_restarts, n_images_help="Number of images to evaluate (default: 1)")
    eval_restarts.add_argument("--max-restarts", type=int,
                               help=f"Maximum number of restarts to run (default: {DEFAULT_MAX_RESTARTS})")

    # evaluate-sampling-steps
    eval_steps = subparsers.add_parser("evaluate-sampling-steps",
                                       help="Evaluate how attack success varies with the number of sampling steps",
                                       parents=[global_parser])
    add_common_eval_args(eval_steps)
    eval_steps.add_argument("--eval-num-steps", nargs="+", type=int,
                            help=f"Sampling step counts to evaluate at (default: {DEFAULT_EVAL_NUM_STEPS})")

    # evaluate-robustness
    eval_robust = subparsers.add_parser("evaluate-robustness",
                                        help="Evaluate attack robustness to JPEG compression and Gaussian blur",
                                        parents=[global_parser])
    add_common_eval_args(eval_robust)
    eval_robust.add_argument("--jpeg-quality-factors", nargs="+", type=int,
                             help=f"JPEG quality factors to evaluate at (default: {DEFAULT_ROBUSTNESS_JPEG_QUALITY_FACTORS})")
    eval_robust.add_argument("--gaussian-blur-sigmas", nargs="+", type=float,
                             help=f"Gaussian blur sigmas to evaluate at (default: {DEFAULT_ROBUSTNESS_GAUSSIAN_BLUR_SIGMAS})")
    eval_robust.add_argument("--robustness-num-steps", type=int,
                             help="Baseline sampling steps for re-evaluation (default: each attack's restart_eval_num_steps)")

    # evaluate-geoshield-vs-diffusion
    eval_geo = subparsers.add_parser("evaluate-geoshield-vs-diffusion",
                                     help="Evaluate precomputed clean/attacked image pairs",
                                     parents=[global_parser])
    add_precomputed_folder_args(eval_geo)

    # evaluate-sampling-steps-precomputed
    eval_steps_pre = subparsers.add_parser("evaluate-sampling-steps-precomputed",
                                           help="Evaluate sampling-step sensitivity for a precomputed attack (e.g. GeoShield)",
                                           parents=[global_parser])
    add_precomputed_folder_args(eval_steps_pre)
    eval_steps_pre.add_argument("--eval-num-steps", nargs="+", type=int,
                                help=f"Sampling step counts to evaluate at (default: {DEFAULT_EVAL_NUM_STEPS})")

    # plot
    plot_cmd = subparsers.add_parser("plot", help="Plot saved results", parents=[global_parser])
    plot_cmd.add_argument("plot_type", nargs="?",
                          choices=["results", "success-rate", "sampling-steps", "dtd-variance",
                                   "loss-vs-fsd", "clean-vs-attacked-displacement", "restarts",
                                   "robustness", "model-transfer", "robustness-and-sampling-steps"],
                          help="Type of plot to generate")
    plot_cmd.add_argument("--dataset", choices=["yfcc", "osv"], help="Dataset to plot")
    plot_cmd.add_argument("--attack-types", nargs="+", help="Attack types to plot (for 'results' plot type)")
    plot_cmd.add_argument("--results-dir", help="Directory containing saved results")
    plot_cmd.add_argument("--plots-dir", help="Directory to save plots")
    plot_cmd.add_argument("--results-files", nargs="+", help="JSON result files to merge for 'sampling-steps' plot type")
    plot_cmd.add_argument("--tikz", action="store_true",
                          help="Emit standalone pgfplots .tex (+ best-effort compiled PDF/PNG preview) "
                               "under <plots-dir>/tikz/ instead of a matplotlib figure "
                               "(not supported for plot_type 'restarts')")

    # list-configs
    subparsers.add_parser("list-configs", help="List available configuration parameters", parents=[global_parser])

    return parser


COMMAND_HANDLERS = {
    "evaluate-dataset": cmd_evaluate_dataset,
    "evaluate-dataset-shard": cmd_evaluate_dataset_shard,
    "merge-shards": cmd_merge_shards,
    "evaluate-localizability": cmd_evaluate_localizability,
    "evaluate-geoshield-vs-diffusion": cmd_evaluate_geoshield_vs_diffusion,
    "evaluate-sampling-steps-precomputed": cmd_evaluate_sampling_steps_precomputed,
    "evaluate-restarts": cmd_evaluate_restarts,
    "evaluate-sampling-steps": cmd_evaluate_sampling_steps,
    "evaluate-robustness": cmd_evaluate_robustness,
    "plot": cmd_plot,
    "list-configs": cmd_list_configs,
}


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.override:
        overrides = {}
        for override_arg in args.override:
            try:
                key, value = parse_override_arg(override_arg)
                overrides[key] = value
            except ValueError as e:
                print(f"Error parsing override: {e}", file=sys.stderr)
                sys.exit(1)
        config = merge_overrides(config, overrides)

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(0)
    handler(args, config)


if __name__ == "__main__":
    main()
