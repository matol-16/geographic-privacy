# Adversarial Demo - Refactored Architecture

## Overview

The adversarial demo code has been refactored to reduce duplication and provide a cleaner, more maintainable codebase. The new architecture separates concerns into distinct modules and provides reusable components.

## File Structure

### CLI + engine

- **`main.py`** - Command-line interface for all experiments
  - Commands: `evaluate-dataset`, `evaluate-localizability`, `evaluate-restarts`,
    `evaluate-sampling-steps`, `evaluate-robustness`, `evaluate-geoshield-vs-diffusion`,
    `evaluate-sampling-steps-precomputed`, `plot`, `list-configs`
  - Per-command boilerplate is collapsed into `prepare_training_run()` +
    `add_common_eval_args()`; commands dispatch through `COMMAND_HANDLERS`
  - Supports config-file overrides from the terminal (`--override key=value`)
- **`config.yaml`** - Baseline configuration (per-dataset budgets, hyperparameters,
  pipelines, plotting options), overridable from the command line

- **`core.py`** - Evaluation engine
  - `EvaluationConfig` / `PrecomputedPairEvaluationConfig`: run configuration
  - `ImageLoader`: dataset-agnostic image loading
  - `ResultsManager`: results / attack-args / run-config / resumable-state I/O
  - `MetricsCollector`: per-attack metrics, per-restart evaluations, and the
    optional per-image sampling-steps samples
  - `BaseEvaluationRunner`: shared resume/state/merge machinery (both runners
    subclass it; previously this logic was duplicated)
  - `EvaluationRunner` / `PrecomputedPairEvaluationRunner`
  - `run_evaluation()` / `run_precomputed_pair_evaluation()`: sequential or parallel

### Modules

- **`utils/adversarial_eval.py`** - Evaluation entry points (thin orchestration)
  - `evaluate_attack_on_dataset()` - main results **plus** the restart ablation
    (always) and the sampling-steps ablation (opt-in), from one training pass
  - `evaluate_localizability()`, `evaluate_restarts()`, `evaluate_sampling_steps()`,
    `evaluate_sampling_steps_precomputed()`, `merge_sampling_steps_results()`
- **`utils/datasets.py`** - YFCC4k / OSV-5M retrieval (split out so `core.py` no
  longer needs `adversarial_eval.py`, removing a circular import)
- **`utils/geoshield.py`** - folds GeoShield (an out-of-process attack from the
  separate Geoshield repo) into `evaluate-dataset`: seeded clean-image selection,
  running `Geoshield/geoshield.py` per budget, and returning the clean/attacked dir
  pairs for the precomputed-pair evaluator. Replaces the old `scripts/geoshield_common.sh`
- **`utils/ablations.py`** - shared ablation building blocks: `best_after_k`,
  `evaluate_delta_at_steps`, `evaluate_delta_under_transforms` (JPEG/blur), the
  JPEG/blur transforms, and the restart / sampling-steps / robustness JSON builders.
  Used by both `evaluate-dataset` and the standalone ablation commands
- **`utils/plots/`** - plotting package: `common` (data helpers + JSON dumping),
  `maps`, `results` (displacement + success-rate, also write JSON), `ablations`.
  `utils/plots_adversarial_attacks.py` re-exports everything for back-compat
- **`attacks/attacks.py`** - attack dispatch (`run_attack`) + per-attack wrappers
  sharing `_build_shared_x0_bank()`
- **`attacks/encoder_attacks.py`**, **`attacks/trajectory_deviation.py`**,
  **`attacks/diffusion_attack_salman.py`**, **`attacks/attacks_core.py`** - attack
  implementations (numerical mechanisms, left unchanged)
- **`utils/adversarial_metrics.py`** - metric computation
- **`utils/adversarial_utils.py`** - device/image/embedding/config utilities

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

## Integrated ablations (evaluate-dataset)

A single `evaluate-dataset` training pass produces the main results and both ablations:

- **Main results**: best-restart displacement per image → displacement and
  success-rate plots.
- **Restart ablation (always on)**: best-displacement-after-k-restarts, derived for
  free from the per-restart evaluations already collected during training. It reaches
  depth `num_restarts`; raise it for a run with `--max-restarts N`.
- **Sampling-steps ablation (opt-in)**: `--run-sampling-steps-ablation` re-evaluates
  each image's best perturbation at every count in `--eval-num-steps` (extra pipeline
  runs, no retraining).
- **Robustness ablation (opt-in, per attack type)**: `--run-robustness-ablation`
  degrades each best perturbation's protected image with JPEG compression / Gaussian
  blur (GeoShield Fig. 6 levels) and re-evaluates at the *baseline* sampling-step
  count (no sampling-steps sweep). Unlike the other ablations it is scoped to
  `robustness.attack_types` (config) / `--robustness-attack-types` (default `dtd`),
  so it only runs for the listed attacks and is a no-op otherwise. Transform levels
  and the baseline step count come from the `robustness:` config block. Each level
  records two displacements, matching the main results: `predicted` (vs the clean
  prediction) and `true` (vs the ground-truth GPS, the GeoShield metric).

```bash
python main.py evaluate-dataset --dataset yfcc \
  --max-restarts 10 \
  --run-sampling-steps-ablation --eval-num-steps 16 64 250 \
  --attack-types dtd --run-robustness-ablation --robustness-attack-types dtd
```

The standalone `evaluate-restarts` / `evaluate-sampling-steps` / `evaluate-robustness`
commands remain for focused runs and share the same `utils/ablations.py` code path
(identical JSON).

## GeoShield as just another attack (evaluate-dataset)

GeoShield is not a trainable attack type — its perturbations are produced by the
separate **Geoshield** repo (`Geoshield/geoshield.py`). It is folded into
`evaluate-dataset` so a single run covers it alongside the in-process attacks:

```bash
python main.py evaluate-dataset --dataset yfcc \
  --attack-types encoder dtd geoshield
```

When `geoshield` is in `--attack-types` (or `geoshield.enabled: true`), the run:

1. trains the in-process attacks as usual;
2. selects the **same seeded** clean images (`select_yfcc_image_paths`), runs
   `Geoshield/geoshield.py` once per budget (epsilon = `round(budget*255)` unless
   `geoshield.epsilons` is set), via `utils/geoshield.generate_geoshield_pairs`;
3. evaluates the clean/attacked pairs through the shared `PrecomputedPairEvaluationRunner`
   (`run_precomputed_attack_eval`), saving `{dataset}_geoshield_results.pt`;
4. re-plots the **combined** results so GeoShield overlays the other attacks.

GeoShield has no ground-truth GPS in the precomputed-pair evaluator, so it is
predicted-only (its line is absent from the `final_step_displacement_true` plot).
Generation settings live in the `geoshield:` config block (`steps`, `clean_dir`,
`output_base`, `epsilons`, `repos_root`, `script`, `python`); `repos_root` auto-detects
the directory holding both `plonk/` and `Geoshield/`. This replaces the former
three-step `scripts/test_all_attacks.sh` + `scripts/geoshield_common.sh` orchestration
with one command. The standalone `evaluate-geoshield-vs-diffusion` command remains for
evaluating already-generated pairs.

## Results Structure

Results are saved to the configured `results_dir` (default `./results/`):

- `{dataset}_{attack_type}_results.pt` - metric tensors [budgets, images] (+ image
  ids/indices, per-restart and per-image location results)
- `{dataset}_attack_args.pt` - attack parameters used
- `{dataset}_run_config{suffix}.yaml` - resolved config for the run (reproducibility)
- `{dataset}_seed{seed}_eval_state{suffix}.pt` - resumable state; rerunning the same
  config resumes from the next unfinished image/budget pair
- `{dataset}_restarts_results.json` - restart ablation data
- `{dataset}_sampling_steps_results.json` - sampling-steps ablation data
- `{dataset}_robustness_results.json` - robustness (JPEG/blur) ablation data
- `{dataset}_metrics_localizability.pt` - localizability evaluation results

Plots are saved to `plots_dir` (default `./plots/`). Every result/success-rate plot
now writes a JSON sidecar next to the PNG with the plotted numbers:

- `{dataset}_{attacks}_{metric}.json` - per-budget mean/median/q25/q75 (+ sample counts)
- `{dataset}_{attacks}_attack_success_rate.json` - per-threshold, per-budget success rates

## Reproducibility

- Every command calls `seed_everything(seed)`; image selection is seeded and
  prefix-stable (the first N of a large run match a smaller run).
- The global `seed` flows into `restart_eval_seed`, and the sampling-steps re-eval
  uses a fixed evaluation seed so clean vs perturbed share identical noise.
- For strict bitwise reproducibility use `parallel_workers: 1`. With
  `parallel_workers > 1`, the thread pool + per-worker CUDA streams can reorder GPU
  ops, which may perturb results slightly; parallelism is a throughput option.

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
