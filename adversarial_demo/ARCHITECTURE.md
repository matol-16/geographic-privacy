# Adversarial Demo - Refactored Architecture

## Overview

The adversarial demo code has been refactored to reduce duplication and provide a cleaner, more maintainable codebase. The new architecture separates concerns into distinct modules and provides reusable components.

## File Structure

### Core Modules

- **`main.py`** - Command-line interface for running all experiments
  - Provides commands: `evaluate-dataset`, `evaluate-localizability`, `plot`
  - Supports configuration file overrides from terminal
- **`config.yaml`** - Baseline configuration file
  - Stores default parameters for all experiments
  - Organized by dataset (yfcc, osv)
  - Easily overridable from command line

- **`core.py`** - Refactored evaluation engine (NEW)
  - `EvaluationConfig`: Configuration dataclass for evaluations
  - `ImageLoader`: Unified image loading interface
  - `ResultsManager`: Centralized results storage and loading
  - `MetricsCollector`: Unified metric collection
  - `EvaluationRunner`: Base class with common evaluation logic
  - `run_evaluation()`: Executes evaluations (parallel or sequential)

### Existing Modules (Refactored)

- **`adversarial_eval.py`** - Evaluation entry points
  - `evaluate_attack_on_dataset()` - Now uses core module
  - `evaluate_localizability()` - Now uses core module
  - `evaluate_attack_transferability()` - Kept for direct API (not in CLI)

- **`attacks.py`** - Attack dispatch and execution
- **`encoder_attacks.py`** - Encoder attack implementation
- **`trajectory_deviation.py`** - Diffusion attack implementation
- **`adversarial_metrics.py`** - Metric computation
- **`adversarial_utils.py`** - Utility functions
- **`plots_adversarial_attacks.py`** - Plotting functions

## Usage Examples

### Evaluate attacks on YFCC dataset

```bash
python main.py evaluate-dataset --dataset yfcc
```

### Evaluate attacks on OSV with custom parameters

```bash
python main.py evaluate-dataset --dataset osv \
  --override 'attack_budgets.osv=[0.01, 0.05, 0.1]' \
  --override parallel_workers=4
```

### Evaluate localizability

```bash
python main.py evaluate-localizability --dataset yfcc
```

### Plot results

```bash
python main.py plot results --dataset yfcc
python main.py plot success-rate --dataset osv
```

### List all configuration options

```bash
python main.py list-configs
```

## Architecture Benefits

### 1. **Code Reuse**

- `EvaluationRunner` consolidates common evaluation patterns
- `ResultsManager` unifies all results I/O
- `MetricsCollector` standardizes metric handling

### 2. **Separation of Concerns**

- Configuration management (config.yaml, main.py)
- Evaluation execution (core.py)
- Results persistence (core.py)
- Attack implementation (attacks.py, trajectory_deviation.py, encoder_attacks.py)
- Metric computation (adversarial_metrics.py)
- Visualization (plots_adversarial_attacks.py)

### 3. **Easy Extensibility**

- Add new evaluation types by extending `EvaluationRunner`
- Add new metrics by updating `MetricsCollector`
- Add new datasets by extending `ImageLoader`
- Add new plots by creating functions in `plots_adversarial_attacks.py`

### 4. **Cleaner API**

- All evaluations follow consistent pattern
- Configuration centralized in single file
- Terminal overrides work consistently across all commands

## Key Refactoring Changes

### Before

- Each evaluation function (`evaluate_attack_on_dataset`, `evaluate_localizability`, `evaluate_attack_transferability`) had duplicated code for:
  - Image loading
  - Progress bar management
  - Result storage
  - Device handling
  - Parallel execution logic

### After

- Common logic extracted to `core.py`
- Evaluation functions now 1/3 their original size
- Configuration centralized in `config.yaml`
- CLI provides easy parameter overrides

## Evaluation Modes

### Sequential (default)

```bash
python main.py evaluate-dataset --dataset yfcc
```

### Parallel

```bash
python main.py evaluate-dataset --dataset yfcc --parallel-workers 4
```

## Results Structure

All results are automatically saved to `./results/` by default:

- `{dataset}_{attack_type}_results.pt` - Metric tensors [budgets, images]
- `{dataset}_attack_args.pt` - Attack parameters used
- `{dataset}_metrics_localizability.pt` - Localizability evaluation results

Plots are saved to `./plots/` by default.

## Configuration Management

### Base Configuration (config.yaml)

Stores defaults for:

- Device, dataset, image count
- Attack budgets per dataset
- Training hyperparameters per dataset
- Pipeline models per dataset
- Metric options
- Plotting options

### Runtime Overrides

Override any parameter from CLI:

```bash
python main.py evaluate-dataset --dataset yfcc \
  --override device=cpu \
  --override n_images_to_eval=50 \
  --override 'attack_budgets.yfcc=[0.01, 0.05]' \
  --override parallel_workers=2
```

## API Reference

### Using core.py in Python

```python
from core import EvaluationConfig, EvaluationRunner, run_evaluation
from pipe_trajectory import PlonkPipelineTrajectory

# Create configuration
config = EvaluationConfig(
    dataset="yfcc",
    attack_types=["encoder", "diffusion"],
    attack_budgets=[1/255, 5/255, 20/255],
    attack_kwargs=[{...}, {...}, {...}],
    n_images=100,
    results_dir="./results",
    plots_dir="./plots",
    stored_metrics=["final_step_displacement_predicted"],
)

# Load pipeline
pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_YFCC_diffusion").to("cuda")

# Run evaluation
runner = EvaluationRunner(config, pipeline)
run_evaluation(runner)
runner.save_results()
```

## Next Steps for Further Refactoring

1. Create `plotting_core.py` for unified plot management
2. Create `attacks_core.py` to abstract attack dispatch
3. Add experiment tracking (MLflow, Wandb integration)
4. Add result caching to avoid re-evaluation
5. Create config validation and schema checking
