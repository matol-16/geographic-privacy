#!/usr/bin/env python3
"""
Command-line interface for adversarial attack experiments.

Supports:
  - evaluate-dataset: Evaluate attacks on a dataset
  - evaluate-localizability: Evaluate attack effectiveness by image localizability
    - evaluate-geoshield-vs-diffusion: Evaluate precomputed clean/attacked pairs
  - plot: Plot saved results or attack success rates
  - list-configs: List available config parameters

Usage:
  python main.py evaluate-dataset --dataset yfcc --attack-types encoder diffusion
  python main.py evaluate-localizability --dataset osv
  python main.py plot success-rate --dataset yfcc
"""

import argparse
import json
import os
import sys
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch

from utils.pipe_trajectory import PlonkPipelineTrajectory
from utils.adversarial_eval import (
    evaluate_attack_on_dataset,
    evaluate_localizability,
    evaluate_sampling_steps,
)
from utils.adversarial_utils import seed_everything, expand_to_budget_count
from utils.plots_adversarial_attacks import (
    plot_results,
    plot_attack_success_rate,
    plot_sampling_steps_success_rate,
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


def _parse_override_value(value_str: str) -> Any:
    """Parse override values using YAML rules, matching OmegaConf-style typing."""
    if value_str == "":
        return ""

    try:
        return yaml.safe_load(value_str)
    except yaml.YAMLError:
        return value_str


def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set a dotted key path inside a nested config mapping.

    This intentionally mirrors OmegaConf dotlist behavior for dictionary-like
    config trees while keeping the runtime object a plain Python dict.
    """
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
    """
    Merge command-line overrides into config dictionary.

    Nested keys use dot notation, matching OmegaConf-style dotlist overrides.
    """
    for key, value in overrides.items():
        if "." in key:
            _set_nested_value(config, key, value)
            continue

        config[key] = value
    
    return config


def parse_override_arg(arg: str) -> tuple[str, Any]:
    """
    Parse a single override argument of the form 'key=value'.
    
    Attempts to parse value as YAML (numbers, booleans, lists, etc.).
    """
    if "=" not in arg:
        raise ValueError(f"Invalid override format: {arg}. Expected 'key=value'")
    
    key, value_str = arg.split("=", 1)
    key = key.strip()
    value_str = value_str.strip()

    return key, _parse_override_value(value_str)


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


def get_device(config: Dict[str, Any]) -> str:
    """Get device from config, defaulting to cuda if available."""
    device = config.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    return device


def get_pipeline(config: Dict[str, Any], dataset: str) -> PlonkPipelineTrajectory:
    """Load and initialize PLONK pipeline for the given dataset."""
    device = get_device(config)
    
    pipelines = config.get("pipelines", {})
    if dataset not in pipelines:
        raise ValueError(f"No pipeline configuration for dataset: {dataset}")
    
    model_name = pipelines[dataset]
    # If a model_type is configured, append it as a suffix to the pipeline name
    model_type = config.get("model_type", "")
    if model_type:
        suffix = f"_{model_type}" if model_type !="" else "" #riemannian FM is just ""
        if not model_name.endswith(suffix):
            model_name = model_name + suffix
    print(f"Loading pipeline: {model_name}")
    pipeline = PlonkPipelineTrajectory(model_name).to(device)
    
    return pipeline


def get_attack_kwargs(
    config: Dict[str, Any],
    dataset: str,
) -> List[Dict[str, Any]]:
    """Return the base attack kwargs for the selected dataset."""
    device = get_device(config)
    global_seed = int(config.get("seed", 0))
    
    base_kwargs = config.get("attack_train_args", {}).get(dataset, {})
    base_kwargs = dict(base_kwargs)  # Make a copy
    base_kwargs["device"] = device
    restart_eval_seed = base_kwargs.get("restart_eval_seed")
    if restart_eval_seed is None or restart_eval_seed == "seed":
        base_kwargs["restart_eval_seed"] = global_seed
    else:
        base_kwargs["restart_eval_seed"] = int(restart_eval_seed)
    
    return [base_kwargs]


def cmd_evaluate_dataset(args, config: Dict[str, Any]) -> None:
    """Execute evaluate-dataset command."""
    # Get parameters from args or config
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), 100)
    parallel_workers = pick_value(args.parallel_workers, config.get("parallel_workers"), 1)
    
    attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError(f"No attack budgets configured for dataset: {dataset}")
    
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))

    seed_everything(seed)
    
    # Get pipeline
    pipeline = get_pipeline(config, dataset)
    
    # Expand attack kwargs
    attack_kwargs = get_attack_kwargs(config, dataset)
    plot_gps_true = bool(get_nested_config(config, "plot", "gps_true", default=False))
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config,
        "plot",
        "attack_success_rate_thresholds",
        default=DEFAULT_SUCCESS_RATE_THRESHOLDS,
    )
    
    print(f"\n{'='*60}")
    print(f"Evaluating attacks on {dataset.upper()} dataset")
    print(f"{'='*60}")
    print(f"Attack types: {attack_types}")
    print(f"Attack budgets: {attack_budgets}")
    print(f"Images to evaluate: {n_images}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Parallel workers: {parallel_workers}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    evaluate_attack_on_dataset(
        attack_types=attack_types,
        pipeline=pipeline,
        dataset_name=dataset,
        source_image=None,
        seed=seed,
        use_real_gps=pick_value(args.use_real_gps, config.get("use_real_gps"), False),
        n_images_to_eval=n_images,
        plot_dir=plots_dir,
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        stored_metrics=get_nested_config(
            config,
            "plot",
            "stored_metrics",
            default=DEFAULT_STORED_METRICS,
        ),
        attack_kwargs=attack_kwargs,
        parallel_workers=parallel_workers,
        use_cuda_streams=bool(config.get("use_cuda_streams", True)),
        dataset_roots=config.get("data_dirs", {}),
        plot_success_rate=plot_success_rate,
        plot_success_rate_thresholds=success_rate_thresholds,
        plot_gps_true=plot_gps_true,
        config_dump=config,
    )
    
    print(f"\nEvaluation complete! Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")


def cmd_evaluate_localizability(args, config: Dict[str, Any]) -> None:
    """Execute evaluate-localizability command."""
    # Get parameters from args or config
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), 100)
    
    attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError(f"No attack budgets configured for dataset: {dataset}")
    
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))

    seed_everything(seed)
    
    # Get pipeline
    pipeline = get_pipeline(config, dataset)
    
    # Expand attack kwargs
    attack_kwargs = get_attack_kwargs(config, dataset)
    
    print(f"\n{'='*60}")
    print(f"Evaluating localizability on {dataset.upper()} dataset")
    print(f"{'='*60}")
    print(f"Attack types: {attack_types}")
    print(f"Attack budgets: {attack_budgets}")
    print(f"Images to evaluate: {n_images}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"{'='*60}\n")
    
    # Run evaluation
    evaluate_localizability(
        attack_types=attack_types,
        pipeline=pipeline,
        dataset_name=dataset,
        seed=seed,
        n_images_to_eval=n_images,
        plot_dir=plots_dir,
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        dataset_roots=config.get("data_dirs", {}),
        config_dump=config,
    )
    
    print(f"\nEvaluation complete! Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    
    
    
def cmd_evaluate_geoshield_vs_diffusion(args, config: Dict[str, Any]) -> None:
    """Evaluate precomputed clean/attacked image pairs and plot the results."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_name = pick_value(args.attack_name, config.get("attack_name"), DEFAULT_GEOSHIELD_ATTACK_NAME)

    attack_budgets = pick_value(args.attack_budgets, config.get("attack_budgets"), None)
    if attack_budgets is None:
        attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError("No attack budgets configured for the precomputed folder evaluation")

    clean_image_dirs = pick_value(args.clean_image_dirs, config.get("clean_image_dirs"), None)
    attacked_image_dirs = pick_value(args.attacked_image_dirs, config.get("attacked_image_dirs"), None)
    if clean_image_dirs is None or attacked_image_dirs is None:
        raise ValueError(
            "Both --clean-image-dirs and --attacked-image-dirs must be provided, either on the CLI or in the config"
        )

    clean_image_dirs = expand_to_budget_count(clean_image_dirs, len(attack_budgets), "clean_image_dirs")
    attacked_image_dirs = expand_to_budget_count(attacked_image_dirs, len(attack_budgets), "attacked_image_dirs")

    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), None)
    seed = int(config.get("seed", 0))
    device = get_device(config)

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)

    plot_gps_true = bool(get_nested_config(config, "plot", "gps_true", default=False))
    if plot_gps_true:
        print("Warning: plot.gps_true is enabled, but precomputed folder evaluation does not have ground-truth GPS labels. Using predicted displacement plots instead.")
        plot_gps_true = False
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config,
        "plot",
        "attack_success_rate_thresholds",
        default=DEFAULT_SUCCESS_RATE_THRESHOLDS,
    )

    stored_metrics = get_nested_config(
        config,
        "plot",
        "stored_metrics",
        default=DEFAULT_STORED_METRICS,
    )
    resolved_stored_metrics = [
        metric for metric in stored_metrics
        if metric != "final_step_displacement_true"
    ] or ["final_step_displacement_predicted"]

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

    print(f"\n{'='*60}")
    print(f"Evaluating precomputed {attack_name} images on {dataset.upper()}")
    print(f"{'='*60}")
    print(f"Attack budgets: {attack_budgets}")
    print(f"Clean image folders: {clean_image_dirs}")
    print(f"Attacked image folders: {attacked_image_dirs}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    if n_images is not None:
        print(f"Images to evaluate: {n_images}")
    print(f"{'='*60}\n")

    precomputed_config = PrecomputedPairEvaluationConfig(
        dataset=dataset,
        attack_name=attack_name,
        seed=seed,
        attack_budgets=list(attack_budgets),
        clean_image_dirs=list(clean_image_dirs),
        attacked_image_dirs=list(attacked_image_dirs),
        results_dir=results_dir,
        plots_dir=plots_dir,
        stored_metrics=resolved_stored_metrics,
        device=device,
        n_images=n_images,
    )
    runner = PrecomputedPairEvaluationRunner(precomputed_config, pipeline)
    runner.save_run_config(resolved_config, suffix=precomputed_config.state_suffix)

    run_precomputed_pair_evaluation(runner)
    runner.save_results()

    all_results = runner.metrics_collector.get_results()

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


def cmd_evaluate_sampling_steps(args, config: Dict[str, Any]) -> None:
    """Execute evaluate-sampling-steps command."""
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    n_images = pick_value(args.n_images, config.get("n_images_to_eval"), 20)
    eval_num_steps = pick_value(args.eval_num_steps, config.get("eval_num_steps"), DEFAULT_EVAL_NUM_STEPS)

    attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError(f"No attack budgets configured for dataset: {dataset}")

    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    seed = int(config.get("seed", 0))
    success_rate_thresholds = get_nested_config(
        config, "plot", "attack_success_rate_thresholds", default=DEFAULT_SUCCESS_RATE_THRESHOLDS
    )

    seed_everything(seed)
    pipeline = get_pipeline(config, dataset)
    attack_kwargs = get_attack_kwargs(config, dataset)

    print(f"\n{'='*60}")
    print(f"Evaluating sampling-step sensitivity on {dataset.upper()} dataset")
    print(f"{'='*60}")
    print(f"Attack types: {attack_types}")
    print(f"Attack budgets: {attack_budgets}")
    print(f"Images to evaluate: {n_images}")
    print(f"Evaluation step counts: {eval_num_steps}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"{'='*60}\n")

    json_results = evaluate_sampling_steps(
        attack_types=attack_types,
        pipeline=pipeline,
        dataset_name=dataset,
        seed=seed,
        n_images_to_eval=n_images,
        eval_num_steps=eval_num_steps,
        results_dir=results_dir,
        attack_budgets=attack_budgets,
        attack_kwargs=attack_kwargs,
        success_rate_thresholds=success_rate_thresholds,
        dataset_roots=config.get("data_dirs", {}),
        config_dump=config,
    )

    plot_sampling_steps_success_rate(json_results=json_results, plot_dir=plots_dir)

    print(f"\nEvaluation complete! Results saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")


def cmd_plot(args, config: Dict[str, Any]) -> None:
    """Execute plot command."""
    plot_type = pick_value(args.plot_type, config.get("plot_type"), "results")
    dataset = pick_value(args.dataset, config.get("dataset"), "yfcc")
    results_dir = pick_value(args.results_dir, config.get("results_dir"), str(DEFAULT_RESULTS_DIR))
    plots_dir = pick_value(args.plots_dir, config.get("plots_dir"), str(DEFAULT_PLOTS_DIR))
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating {plot_type} plots for {dataset.upper()}")
    print(f"{'='*60}")
    print(f"Results directory: {results_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"{'='*60}\n")
    
    attack_budgets = config.get("attack_budgets", {}).get(dataset)
    if not attack_budgets:
        raise ValueError(f"No attack budgets configured for dataset: {dataset}")
    
    attack_types = pick_value(args.attack_types, config.get("attack_types"), DEFAULT_ATTACK_TYPES)
    gps_true = bool(get_nested_config(config, "plot", "gps_true", default=False))
    plot_success_rate = bool(get_nested_config(config, "plot", "plot_success_rate", default=False))
    success_rate_thresholds = get_nested_config(
        config,
        "plot",
        "attack_success_rate_thresholds",
        default=DEFAULT_SUCCESS_RATE_THRESHOLDS,
    )
        
    
    if plot_type == "results":
        
        print(f"Plotting results for attacks: {attack_types}")
        
        plot_results(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
            all_results=None,
            stored_metrics=get_nested_config(
                config,
                "plot",
                "stored_metrics",
                default=DEFAULT_STORED_METRICS,
            ),
        )

        if plot_success_rate:
            plot_attack_success_rate(
                results_dir=results_dir,
                attack_budgets=attack_budgets,
                plot_dir=plots_dir,
                dataset_name=dataset,
                attack_types=attack_types,
                gps_true=gps_true,
                threshold_km=success_rate_thresholds,
            )
    
    elif plot_type == "success-rate":
        attack_budgets = config.get("attack_budgets", {}).get(dataset)
        if not attack_budgets:
            raise ValueError(f"No attack budgets configured for dataset: {dataset}")
        
        thresholds = get_nested_config(
            config,
            "plot",
            "attack_success_rate_thresholds",
            default=DEFAULT_SUCCESS_RATE_THRESHOLDS,
        )
        
        print(f"Plotting attack success rates with distance thresholds: {thresholds} km")
        gps_true = bool(get_nested_config(config, "plot", "gps_true", default=False))
        
        plot_attack_success_rate(
            results_dir=results_dir,
            attack_budgets=attack_budgets,
            plot_dir=plots_dir,
            dataset_name=dataset,
            attack_types=attack_types,
            gps_true=gps_true,
            threshold_km=success_rate_thresholds,
        )
    
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    print(f"\nPlots saved to: {plots_dir}")


def cmd_list_configs(args, config: Dict[str, Any]) -> None:
    """List available configuration parameters."""
    print("\n" + "="*60)
    print("BASELINE CONFIGURATION")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    print("\n" + "="*60)
    print("CONFIGURATION OVERRIDE EXAMPLES")
    print("="*60)
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
  
  # Change directories
  --override results_dir=./custom_results plots_dir=./custom_plots
  
  # Disable CUDA streams
  --override use_cuda_streams=false
""")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Adversarial attack experiments CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate attacks on YFCC dataset
  python main.py evaluate-dataset --dataset yfcc
  
  # Evaluate on OSV with custom parameters
  python main.py evaluate-dataset --dataset osv --n-images 50
  
  # Evaluate localizability
  python main.py evaluate-localizability --dataset yfcc
  
  # Plot results
  python main.py plot results --dataset yfcc
  
  # Plot attack success rates
  python main.py plot success-rate --dataset osv
  
  # Override config parameters
  python main.py evaluate-dataset --dataset yfcc \\
    --override 'attack_budgets.yfcc=[0.01, 0.05]' \\
    --override parallel_workers=4

    # Dot notation and YAML typing work together
    python main.py evaluate-dataset --dataset yfcc \
        --override plot.gps_true=true \
        --override 'plot.stored_metrics=["final_step_displacement_predicted"]'
  
  # Use custom config file
  python main.py --config custom_config.yaml evaluate-dataset --dataset yfcc
        """,
    )
    
    # Global arguments
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
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # evaluate-dataset command
    eval_dataset = subparsers.add_parser(
        "evaluate-dataset",
        help="Evaluate attacks on a dataset",
        parents=[global_parser],
    )
    eval_dataset.add_argument(
        "--dataset",
        choices=["yfcc", "osv"],
        help="Dataset to evaluate on",
    )
    eval_dataset.add_argument(
        "--attack-types",
        nargs="+",
        help="Attack types to evaluate (default: encoder diffusion)",
    )
    eval_dataset.add_argument(
        "--n-images",
        type=int,
        help="Number of images to evaluate",
    )
    eval_dataset.add_argument(
        "--use-real-gps",
        action="store_true",
        default=None,
        help="Use real GPS coordinates from dataset instead of clean predictions",
    )
    eval_dataset.add_argument(
        "--results-dir",
        help="Directory to save results",
    )
    eval_dataset.add_argument(
        "--plots-dir",
        help="Directory to save plots",
    )
    eval_dataset.add_argument(
        "--parallel-workers",
        type=int,
        help="Number of parallel workers for evaluation",
    )
    
    # evaluate-localizability command
    eval_local = subparsers.add_parser(
        "evaluate-localizability",
        help="Evaluate attack effectiveness by image localizability",
        parents=[global_parser],
    )
    eval_local.add_argument(
        "--dataset",
        choices=["yfcc", "osv"],
        help="Dataset to evaluate on",
    )
    eval_local.add_argument(
        "--attack-types",
        nargs="+",
        help="Attack types to evaluate (default: encoder diffusion)",
    )
    eval_local.add_argument(
        "--n-images",
        type=int,
        help="Number of images to evaluate",
    )
    eval_local.add_argument(
        "--results-dir",
        help="Directory to save results",
    )
    eval_local.add_argument(
        "--plots-dir",
        help="Directory to save plots",
    )

    # evaluate-geoshield-vs-diffusion command
    eval_geo = subparsers.add_parser(
        "evaluate-geoshield-vs-diffusion",
        help="Evaluate precomputed clean/attacked image pairs",
        parents=[global_parser],
    )
    eval_geo.add_argument(
        "--dataset",
        choices=["yfcc", "osv"],
        help="Dataset label used for saving results and plots",
    )
    eval_geo.add_argument(
        "--attack-name",
        help=f"Label for the evaluated attack (default: {DEFAULT_GEOSHIELD_ATTACK_NAME})",
    )
    eval_geo.add_argument(
        "--attack-budgets",
        nargs="+",
        type=float,
        help="Attack budgets aligned with the clean/attacked folder table",
    )
    eval_geo.add_argument(
        "--clean-image-dirs",
        nargs="+",
        help="One clean image directory per budget",
    )
    eval_geo.add_argument(
        "--attacked-image-dirs",
        nargs="+",
        help="One attacked image directory per budget",
    )
    eval_geo.add_argument(
        "--n-images",
        type=int,
        help="Optional number of images to evaluate after matching pairs",
    )
    eval_geo.add_argument(
        "--results-dir",
        help="Directory to save results",
    )
    eval_geo.add_argument(
        "--plots-dir",
        help="Directory to save plots",
    )
    
    # evaluate-sampling-steps command
    eval_steps = subparsers.add_parser(
        "evaluate-sampling-steps",
        help="Evaluate how attack success varies with the number of sampling steps",
        parents=[global_parser],
    )
    eval_steps.add_argument(
        "--dataset",
        choices=["yfcc", "osv"],
        help="Dataset to evaluate on",
    )
    eval_steps.add_argument(
        "--attack-types",
        nargs="+",
        help="Attack types to evaluate (default: encoder diffusion)",
    )
    eval_steps.add_argument(
        "--n-images",
        type=int,
        help="Number of images to evaluate",
    )
    eval_steps.add_argument(
        "--eval-num-steps",
        nargs="+",
        type=int,
        help=f"Sampling step counts to evaluate at (default: {DEFAULT_EVAL_NUM_STEPS})",
    )
    eval_steps.add_argument(
        "--results-dir",
        help="Directory to save results",
    )
    eval_steps.add_argument(
        "--plots-dir",
        help="Directory to save plots",
    )

    # plot command
    plot_cmd = subparsers.add_parser(
        "plot",
        help="Plot saved results",
        parents=[global_parser],
    )
    plot_cmd.add_argument(
        "plot_type",
        nargs="?",
        choices=["results", "success-rate"],
        help="Type of plot to generate",
    )
    plot_cmd.add_argument(
        "--dataset",
        choices=["yfcc", "osv"],
        help="Dataset to plot",
    )
    plot_cmd.add_argument(
        "--attack-types",
        nargs="+",
        help="Attack types to plot (for 'results' plot type)",
    )
    plot_cmd.add_argument(
        "--results-dir",
        help="Directory containing saved results",
    )
    plot_cmd.add_argument(
        "--plots-dir",
        help="Directory to save plots",
    )
    
    # list-configs command
    subparsers.add_parser(
        "list-configs",
        help="List available configuration parameters",
        parents=[global_parser],
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Apply overrides
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
    
    # Execute command
    if args.command == "evaluate-dataset":
        cmd_evaluate_dataset(args, config)
    elif args.command == "evaluate-localizability":
        cmd_evaluate_localizability(args, config)
    elif args.command == "evaluate-geoshield-vs-diffusion":
        cmd_evaluate_geoshield_vs_diffusion(args, config)
    elif args.command == "evaluate-sampling-steps":
        cmd_evaluate_sampling_steps(args, config)
    elif args.command == "plot":
        cmd_plot(args, config)
    elif args.command == "list-configs":
        cmd_list_configs(args, config)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
