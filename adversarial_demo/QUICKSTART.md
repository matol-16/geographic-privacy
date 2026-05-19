# Quick Start Guide

## Installation & Setup

1. Ensure all dependencies are installed
2. Update `config.yaml` if needed for your setup

## Running Experiments from Terminal

### Simplest Usage

```bash
# Evaluate encoder and diffusion attacks on YFCC
python main.py evaluate-dataset --dataset yfcc

# Evaluate on OSV
python main.py evaluate-dataset --dataset osv
```

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

### List All Configuration Options

```bash
python main.py list-configs
```

## Output Structure

After running an evaluation, check:

- `./results/` - All result files saved here
  - `{dataset}_{attack_type}_results.pt` - Metric tensors
  - `{dataset}_attack_args.pt` - Hyperparameters used
- `./plots/` - All generated plots saved here

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
3. **Easy Parameter Overrides**: Change any parameter from terminal using `--override`
4. **Consistent Results Management**: All results saved/loaded through `ResultsManager`
5. **Standardized Metrics**: All metrics collected through `MetricsCollector`
6. **Flexible Execution**: Support for both sequential and parallel evaluations

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

## Troubleshooting

### "Config file not found"

```bash
# Make sure config.yaml exists in the current directory
# Or specify path: python main.py --config ./custom_config.yaml evaluate-dataset --dataset yfcc
```

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

## Further Reading

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation and examples of extending the system.
