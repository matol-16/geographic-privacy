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

## Output Structure

After running an evaluation, check:

- `results_dir` in [config.yaml](config.yaml) - All result files are saved there
  - `{dataset}_{attack_type}_results.pt` - Metric tensors
  - `{dataset}_{attack_type}_results.pt` also includes `image_ids` and `image_indices` so results can be mapped back to the source dataset order without storing extra per-image metadata
  - `{dataset}_attack_args.pt` - Hyperparameters used
- Resumable evaluations also save a state file named `{dataset}_seed{seed}_eval_state.pt` in `results_dir`, and rerunning the same evaluation resumes from the next unfinished image/budget pair when the state matches the current config
- `plots_dir` in [config.yaml](config.yaml) - All generated plots are saved there

The evaluation path is intentionally thin now: `core.py` orchestrates the loop, `attacks.py` dispatches to the attack implementation, and the attack classes handle the optimization details. That keeps the hot path focused on the per-image work, which matters most for large batches.

## Codebase Structure

The refactored codebase is organized as:

```
adversarial_demo/
├── main.py              # CLI entry point
├── config.yaml          # Configuration file
├── core.py              # Refactored evaluation engine (NEW)
├── adversarial_eval.py  # Evaluation functions (simplified)
├── attacks.py           # Attack dispatch and execution
├── encoder_attacks.py   # Encoder attack implementation
├── trajectory_deviation.py  # Diffusion attack implementation
├── plots_adversarial_attacks.py  # Plotting functions
├── adversarial_metrics.py  # Metric computation
├── adversarial_utils.py    # Utility functions
├── ARCHITECTURE.md      # Detailed architecture documentation
└── QUICKSTART.md        # This file
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
