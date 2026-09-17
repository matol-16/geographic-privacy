# Adversarial Demo - Refactored Architecture

## Overview

The adversarial demo code has been refactored to reduce duplication and provide a cleaner, more maintainable codebase. The new architecture separates concerns into distinct modules and provides reusable components.

## Attack naming (paper vs. code)

The paper frames every attack evaluated here except GeoShield as an instance of the
**GeoFA** framework (Geographic Flow Attacks), so all of them carry the `GeoFA-` prefix;
the suffix names the loss/target the instance optimizes. The code keeps its historical
keys; the paper-facing labels come from one place,
`utils/plots/common.py::_display_attack_name`, which every figure and table goes through:

| code key | paper label | | code key | paper label |
|---|---|---|---|---|
| `dtd` | **GeoFA-TD** (headline; formerly "DTD") | | `training_loss` | GeoFA-AdvDM |
| `encoder` | GeoFA-enc. | | `ace` | GeoFA-ACE |
| `sampling` | GeoFA-sample | | `unidef` | GeoFA-UniDef |
| `diffusion_l2` | GeoFA-L² | | `geoshield` | GeoShield (external pipeline) |
| `targeted_l2` | GeoFA-DPA | | `dtd_blur` | GeoFA-TD-blur |

Config keys, CLI flags (`--attack-types dtd`, `plot dtd-variance`) and result file names
(`<dataset>_dtd_results.pt`) are unchanged.

**Versions and pins.** Several attacks exist in successive implementations under their own
keys (`dtd` → `dtd_v2` / `dtd_v3` / `dtd_v3_old_forward`, `diffusion_l2` → `_v2` / `_v3`,
`encoder_v3`, `training_loss_v3`, `sampling_v2`, `dtd_blur_v3_old_forward`; see the alias
table in `attacks/attacks.py::run_attack`). A run can keep the canonical key and pin the
implementation from the config: `attack_train_args.<ds>.per_attack.<key>.attack_variant:
<variant key>` makes `run_attack` dispatch to the variant while results, plots and tables
keep the canonical name; the pin is recorded in the saved run config and in each result's
`attack_type` label / attack `config`. `configs/cluster_full/config.yaml` pins every attack
of the final run this way (GeoFA-TD = `dtd_v3_old_forward`).

## Paper figures and tables

`scripts/make_paper_figures.sh [yfcc|osv|all]` regenerates every figure from the saved
full-run results (no GPU) into a fixed layout per dataset:

- `results/cluster_eval/yfcc4k_full/` — YFCC4k: `plots/` holds the main results **and**
  every ablation (sampling steps, model transfer, robustness at 2500 km, restarts,
  localizability); `plots_appendix/` holds the appendix variants.
- `results/cluster_eval_2/osv_full/` — OSV-5M (5000 images; the only OSV run kept):
  `plots/` holds the main results; **all** OSV ablations go to `plots_appendix/`.
- Appendix-only variants on both datasets: robustness figures with all three thresholds
  (`--thresholds all`) and the clean-vs-attacked heatmap over every attack; the main
  heatmap shows GeoShield, GeoFA-enc., GeoFA-AdvDM, GeoFA-DPA, GeoFA-TD on one line
  (`--n-cols 5`). Every appendix stem ends in `_appendix` (`plot --filename-suffix`).
- Each `plots*/` has a `tikz/` twin (pgfplots `.tex` + compiled PDF/PNG).

`scripts/make_success_rate_table.py` turns the `*_attack_success_rate.json` /
`*_final_step_displacement_true.json` sidecars of `plots/` into the main results table
(`results/tables/success_rate_{yfcc,osv,all,2500km}.{md,tex}`).

## Perturbation convention (pixel space)

Every attack optimises a perturbation `delta` **on the [0, 1] pixel scale** of the encoder's
336x336 input, and the `l_inf` budget (`attack_budgets`, e.g. `0.0314` = 8/255) is a budget in
grey levels -- the same convention as GeoShield's `epsilon`. `AttackBase.perturbed_input(delta)`
(`attacks/attacks_core.py`) builds the encoder input as
`normalize(clamp(pixels + delta, 0, 1))` with the encoder's own mean/std (ImageNet for
DINOv2 / YFCC, CLIP for StreetCLIP / OSV, read off the pipeline by
`utils.adversarial_utils.encoder_normalization`), and `add_perturbation_to_image` builds the
protected 8-bit image the evaluator and ablations feed back to the pipeline with the same
formula. Training and evaluation therefore see the same image up to quantisation.

History: until 2026-09-13 `delta` was added in *normalised* space (so a "4/255" budget was
~0.9/255 in pixels) and `tensor_to_pil` un-normalised with mean = std = 0.5, i.e. the evaluated
"protected" image was a ~2x contrast-stretched copy of the original (40-60 % clipped pixels)
even for `delta = 0`, while GeoShield images (read from disk) were not. Results produced
before that date (`results/cluster_eval*`) carry this confound and must be regenerated.

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
  - `compute_localizability_scores()` / `merge_localizability_scores()` /
    `plot_localizability_vs_attacks()` - the two-stage localizability pipeline
    (precompute + store scores by image id, then join to attack results and plot)
  - `evaluate_restarts()`, `evaluate_sampling_steps()`,
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

### Evaluate localizability (two decoupled stages)

Localizability is a property of the clean image, independent of any attack, so it is
precomputed + stored once and later joined to the (separately produced) attack results
by image id. This lets a full-dataset cluster run reuse one set of scores.

```bash
# Stage 1 — precompute + store per-image localizability scores (id -> score).
# Optionally shard by image window across cluster jobs (see scripts/cluster/localizability/).
python main.py evaluate-localizability --stage compute --dataset yfcc --n-images 4000
python main.py evaluate-localizability --stage merge   --dataset yfcc   # stitch window shards

# Stage 2 — join the stored scores with the attack results and plot (per budget).
python main.py evaluate-localizability --stage plot --dataset yfcc \
  --attack-types dtd encoder unidef ace --results-dir <dir> --plots-dir <dir>/plots
# Stack both datasets as rows in one figure (paper layout):
python main.py evaluate-localizability --stage plot --datasets yfcc osv \
  --attack-types dtd encoder --results-dir <dir> --plots-dir <dir>/plots
```

Run the precompute on a single local GPU with `scripts/run_localizability.sh` (defaults to
all of YFCC4k; `./run_localizability.sh osv` for a seeded 4000-image OSV-5M subset).

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

## Multi-node sharded evaluation (cluster)

A full-dataset run (e.g. all 4000 YFCC4k images × every attack × 2 budgets × 8 restarts)
is far too slow on one GPU, so `evaluate-dataset` can be split across many SLURM jobs and
merged. The single-GPU command is unchanged; this is an additive path.

- **`evaluate-dataset-shard`** runs ONE shard: one attack on one *image window*. Image
  selection is prefix-stable and deterministic, so window *k* covers a fixed slice
  `[k·images_per_shard, (k+1)·images_per_shard)` of the seeded ordering. The window is
  loaded via `window_start`/`window_end` (threaded through `ImageLoader.load_images` →
  `retrieve_{yfcc,osv}_images` → `_select_window` in `utils/datasets.py`). Each shard
  writes its results + resumable state into its own subdir
  `<results_dir>/shards/<attack>__w<start>_<end>/`, so shards are independent and
  individually restartable. Implemented by `evaluate_attack_shard()`
  (`utils/adversarial_eval.py`); it collects the same main + restart + sampling-steps +
  robustness buffers as `evaluate-dataset` but does **not** plot.
- **`merge-shards`** reads every shard's state file (which already holds all buffers,
  keyed by global image id), maps each window back onto the full seeded ordering, and
  reconstructs the exact artifacts a single-process run would produce: the combined
  `<dataset>_<attack>_results.pt`, the displacement / success-rate plots, and the
  restart / sampling-steps / robustness ablation JSONs + plots. The ablation JSONs are
  built with the same `build_and_save_*` helpers via a thin `metrics_collector` shim, so
  the output is byte-compatible with the single-GPU path. Implemented by `merge_shards()`.
- **`scripts/cluster/`** holds the SLURM scaffolding: `config.sh` (one place to define
  the run), `eval_shard.slurm` (the job array, task *t* → attack `t / num_windows`,
  window `t % num_windows`), `merge.slurm` (CPU job), and `submit.sh` (computes the array
  size, submits the array + a `afterok` merge job; `--dry-run` prints the mapping). See
  `scripts/cluster/README.md`.
- **GeoShield** is sharded too. A `geoshield` shard selects the same seeded image window,
  generates its perturbations out-of-process into shard-isolated dirs (a `run_tag` per
  window prevents concurrent shards from clobbering the shared clean/output dirs), and
  evaluates the pairs with `PrecomputedPairEvaluationRunner` — saving a state file in the
  same format. `merge_shards` stitches it in like any other attack; because GeoShield is
  keyed by filename (`123.jpg`) while trainable attacks are keyed by photo-id stem
  (`123`), the merge normalises ids by stem so both land at the same global index.
  GeoShield reports the true-GPS metric like the trainable attacks (the generator returns
  the labels of the images it copied) and takes part in the same eval-time ablations
  (sampling steps, cfg, robustness, and cross-model transfer via
  `PrecomputedPairEvaluationRunner(transfer_pipelines=...)`), scored with the dataset's
  `restart_eval_*` settings. With `clean_preprocess: model` its clean copies are produced by
  the attacked pipeline's own conditional preprocessing (DINOv2 chain on YFCC, StreetCLIP
  chain on OSV), so the perturbation is measured against exactly the evaluator's clean
  tensor. Implemented by `_run_geoshield_shard()` in `main.py` reusing
  `utils/geoshield.generate_geoshield_pairs` (window/run_tag aware, `pipeline=` for the
  model-exact copies).
- **The full run** (`scripts/cluster/preset_cluster_full.sh`, config
  `configs/cluster_full/config.yaml`): 9 attacks × 3 budgets × 4000 images per dataset with
  every ablation; `scripts/cluster/smoke_test_cluster_full.sh` reproduces the shard command
  locally on a 2-image window, `scripts/tests/check_preprocessing_roundtrip.py` checks the
  pixel-space round trip of both encoders. See `scripts/cluster/README.md`.

```bash
# one shard (normally launched by the SLURM array, not by hand)
python main.py evaluate-dataset-shard --dataset yfcc --attack-types encoder \
  --total-images 4000 --images-per-shard 100 --image-shard-index 7 \
  --results-dir results/yfcc4k_full --run-sampling-steps-ablation
# after all shards finish
python main.py merge-shards --dataset yfcc --total-images 4000 \
  --attack-types encoder sampling diffusion_l2 dtd unidef unidef_nofdje ace \
  --results-dir results/yfcc4k_full --plots-dir results/yfcc4k_full/plots \
  --run-sampling-steps-ablation
```

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
- `localizability_shards/{dataset}_loc__w<start>_<end>.pt` - per-window localizability
  scores (compute stage; resumable)
- `{dataset}_localizability.pt` - merged full-dataset localizability scores (id -> score),
  the input to the localizability plot stage

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
