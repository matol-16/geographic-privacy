#!/usr/bin/env python3
"""
Command-line interface for adversarial attack experiments.

Supports:
  - evaluate-dataset: Evaluate attacks on a dataset
  - evaluate-localizability: Evaluate attack effectiveness by image localizability
  - plot: Plot saved results or attack success rates
  - list-configs: List available config parameters

Usage:
  python main.py evaluate-dataset --dataset yfcc --attack-types encoder diffusion
  python main.py evaluate-localizability --dataset osv
  python main.py plot success-rate --dataset yfcc
"""

import argparse
import os
import sys
import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
from PIL import Image

from utils.pipe_trajectory import PlonkPipelineTrajectory
from utils.adversarial_eval import (
    evaluate_attack_on_dataset,
    evaluate_localizability,
)
from utils.adversarial_utils import seed_everything
from utils.plots_adversarial_attacks import (
    plot_results,
    plot_attack_success_rate,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")
DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
DEFAULT_PLOTS_DIR = Path(__file__).with_name("plots")
DEFAULT_ATTACK_TYPES = ["encoder", "diffusion"]
DEFAULT_STORED_METRICS = ["final_step_displacement_predicted", "final_step_displacement_true"]
DEFAULT_SUCCESS_RATE_THRESHOLDS = [200, 750, 2500]


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
    
    Handles nested keys with dot notation, e.g., 'attack_budgets.yfcc' or 'plot.stored_metrics'.
    """
    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys
            parts = key.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
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
    
    if value_str == "":
        value = ""
    else:
        # Try to parse as YAML to handle numbers, booleans, lists, etc.
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError:
            # Fallback to string
            value = value_str
    
    return key, value


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
    
    base_kwargs = config.get("attack_train_args", {}).get(dataset, {})
    base_kwargs = dict(base_kwargs)  # Make a copy
    base_kwargs["device"] = device
    
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
    elif args.command == "plot":
        cmd_plot(args, config)
    elif args.command == "list-configs":
        cmd_list_configs(args, config)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
