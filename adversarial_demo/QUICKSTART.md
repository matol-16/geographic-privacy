# Quick Start Guide

## Installation & Setup

1. Ensure all dependencies are installed.
2. Update [config.yaml](config.yaml) if needed for your setup, especially `data_root`, `data_dirs`, and `build_yfcc4k`.

## Running Experiments from Terminal

### Simplest Usage

```bash
# Evaluate encoder and diffusion attacks on YFCC
python main.py evaluate-dataset --dataset yfcc

# Evaluate on OSV
python main.py evaluate-dataset --dataset osv
```

The default dataset locations are read from [config.yaml](config.yaml). If your data lives elsewhere, update `data_root`, `data_dirs`, and `build_yfcc4k` there rather than passing ad hoc paths on the command line.

### Shell Launchers

For repeatable runs, use the scripts in [scripts/](scripts) and edit the variables at the top of each file:

```bash
bash scripts/evaluate_dataset.sh
bash scripts/evaluate_localizability.sh
bash scripts/plot_results.sh
bash scripts/plot_success_rate.sh
```

These scripts keep [config.yaml](config.yaml) as the baseline and only add command-line overrides when you edit the in-file variables.

### With Custom Parameters

```bash
# Use fewer images, custom results directory
python main.py evaluate-dataset --dataset yfcc \
  --n-images 50 \
  --results-dir ./my_results \
  --plots-dir ./my_plots
```

### Parallel Evaluation

```bash
# Use 4 parallel workers for faster evaluation
python main.py evaluate-dataset --dataset yfcc --parallel-workers 4
```

### Evaluate Localizability

```bash
# Evaluate how attack effectiveness varies by image localizability
python main.py evaluate-localizability --dataset yfcc

python main.py evaluate-localizability --dataset osv \
  --n-images 100
```

### Plotting Results

```bash
# Plot attack results (displacement metrics)
python main.py plot results --dataset yfcc

# Plot attack success rates
python main.py plot success-rate --dataset osv
```

### Advanced Configuration Overrides

```bash
# Override multiple parameters at once
python main.py evaluate-dataset --dataset yfcc \
  --override device=cpu \
  --override 'attack_budgets.yfcc=[0.01, 0.05, 0.1]' \
  --override n_images_to_eval=50 \
  --override parallel_workers=2 \
  --override use_cuda_streams=false
```

Overrides use dot notation for nested config values and YAML parsing for values, so booleans, numbers, lists, and dictionaries are handled naturally.

### Building YFCC4k

```bash
# Build the YFCC4k dataset using config defaults
python build_yfcc4k_from_revisiting_im2gps.py

# Use a custom config if needed
python build_yfcc4k_from_revisiting_im2gps.py --config ./config.yaml
```

The builder reads its defaults from [config.yaml](config.yaml). The intended layout is under `data_root`, with the YFCC source archive, metadata file, and built `yfcc4k` directory living inside the HF cache tree.

### List All Configuration Options

```bash
python main.py list-configs
```

## Gathering main results + ablations in one pass

`evaluate-dataset` produces the main results, the **restart ablation** (always — it
is free, derived from the per-restart evaluations done during training), and an
opt-in **sampling-steps ablation** from a single training pass:

```bash
# Main results + restart ablation up to 10 restarts + sampling-steps ablation
python main.py evaluate-dataset --dataset yfcc \
  --max-restarts 10 \
  --run-sampling-steps-ablation --eval-num-steps 16 64 250
```

- `--max-restarts N` runs each attack with N restarts (sets the restart-ablation depth).
- `--run-sampling-steps-ablation` re-evaluates each best perturbation at every
  `--eval-num-steps` count (extra pipeline runs, no retraining).

The standalone `evaluate-restarts` and `evaluate-sampling-steps` commands still exist
for focused runs and share the same code/JSON format.

## Output Structure

After running an evaluation, check:

- `results_dir` - all result files are saved there
  - `{dataset}_{attack_type}_results.pt` - metric tensors; also includes `image_ids`
    and `image_indices` so results map back to the source dataset order
  - `{dataset}_attack_args.pt` - hyperparameters used
  - `{dataset}_run_config.yaml` - the resolved config for the run
  - `{dataset}_restarts_results.json` - restart-ablation data
  - `{dataset}_sampling_steps_results.json` - sampling-steps-ablation data (when enabled)
- Resumable evaluations save `{dataset}_seed{seed}_eval_state.pt`; rerunning the same
  config resumes from the next unfinished image/budget pair
- `plots_dir` - all generated plots, **plus a JSON sidecar next to each result plot**:
  - `{dataset}_{attacks}_{metric}.json` - per-budget mean/median/q25/q75
  - `{dataset}_{attacks}_attack_success_rate.json` - per-threshold success rates

For strict reproducibility use `--parallel-workers 1` (or `parallel_workers: 1`);
parallel workers + CUDA streams trade exact determinism for throughput.

## Codebase Structure

The refactored codebase is organized as:

```
adversarial_demo/
├── main.py                  # CLI entry point (shared arg/context helpers)
├── config.yaml              # Configuration file
├── core.py                  # Evaluation engine (BaseEvaluationRunner + runners)
├── utils/
│   ├── adversarial_eval.py  # Evaluation entry points (thin orchestration)
│   ├── datasets.py          # YFCC4k / OSV-5M retrieval (breaks a circular import)
│   ├── ablations.py         # Shared restart + sampling-steps helpers
│   ├── adversarial_metrics.py
│   ├── adversarial_utils.py
│   ├── pipe_trajectory.py
│   ├── plots_adversarial_attacks.py  # Back-compat facade -> utils/plots/
│   └── plots/               # common, maps, results (+JSON), ablations
├── attacks/
│   ├── attacks.py           # Attack dispatch (run_attack) + wrappers
│   ├── attacks_core.py
│   ├── encoder_attacks.py
│   ├── trajectory_deviation.py
│   └── diffusion_attack_salman.py
├── ARCHITECTURE.md          # Detailed architecture documentation
└── QUICKSTART.md            # This file
```

## Key Design Improvements

1. **Reduced Code Duplication**: Common evaluation patterns consolidated in `core.py`
2. **Unified Configuration**: Single `config.yaml` file for all parameters
3. **Direct Attack Dispatch**: The evaluation loop calls the attack dispatcher directly, without extra wrapper layers
4. **Shared Evaluation Setup**: Restart evaluation and shared-noise inference are centralized in utility helpers
5. **Easy Parameter Overrides**: Change any parameter from terminal using `--override`
6. **Consistent Results Management**: All results saved/loaded through `ResultsManager`
7. **Standardized Metrics**: All metrics collected through `MetricsCollector`
8. **Flexible Execution**: Support for both sequential and parallel evaluations

## Using the Python API Directly

You can still use the evaluation functions directly in Python:

```python
from adversarial_eval import evaluate_attack_on_dataset
from pipe_trajectory import PlonkPipelineTrajectory

device = "cuda"
pipeline = PlonkPipelineTrajectory("nicolas-dufour/PLONK_YFCC_diffusion").to(device)

evaluate_attack_on_dataset(
    attack_types=["encoder", "diffusion"],
    pipeline=pipeline,
    dataset_name="yfcc",
    n_images_to_eval=100,
    attack_budgets=[1/255, 2/255, 5/255],
    attack_kwargs=[{...}, {...}, {...}],
    results_dir="./results",
    plot_dir="./plots",
    parallel_workers=4,
)
```

  For large-scale evaluation, the shared-noise source/perturbed comparison now runs under `torch.inference_mode()` to reduce overhead during metric computation.

## Troubleshooting

### "Config file not found"

```bash
# Make sure config.yaml exists in the current directory
# Or specify path: python main.py --config ./custom_config.yaml evaluate-dataset --dataset yfcc
```

### Dataset paths look wrong

If the builder or evaluator is pointing at the wrong location, update `data_root`, `data_dirs`, and `build_yfcc4k` in [config.yaml](config.yaml). The defaults are meant to keep YFCC and OSV under `Data/mathias.ollu/hf_cache/datasets`.

### "No attacks specified"

```bash
# Specify attack types
python main.py evaluate-dataset --dataset yfcc --attack-types encoder diffusion
```

### Out of memory with parallel workers

```bash
# Reduce parallel workers
python main.py evaluate-dataset --dataset yfcc --parallel-workers 1
```

### CUDA not available

```bash
# Fall back to CPU
python main.py evaluate-dataset --dataset yfcc --override device=cpu
```

## Performance Tips

1. **Parallel Workers**: Use 4-8 workers if you have enough GPU memory
2. **Batch Size**: Adjust in config.yaml's `attack_train_args`
3. **Image Count**: Start with 10-20 images for testing, scale up later
4. **Attack Budgets**: Start with fewer budgets ([1/255, 20/255, 50/255]) for faster iteration
5. **Dataset Root**: Keep the dataset tree under `data_root` so the builder and evaluation code stay aligned

## Further Reading

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation and examples of extending the system.
