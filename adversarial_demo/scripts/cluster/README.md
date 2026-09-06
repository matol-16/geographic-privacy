# Multi-node dataset evaluation (Jean Zay / SLURM)

Run the full-dataset attack evaluation (e.g. all of YFCC4k = 4000 images, the attacks
from `scripts/test_all_attacks.sh`) by **sharding it across many GPU jobs**, then merging.
A single GPU is far too slow: ~30 s/restart × 8 restarts × 4000 images × 7 attacks × 2
budgets is thousands of GPU-hours, so we split it into independent jobs.

## Running the new experiments

Three experiments are wrapped as one-command presets. Run them from this directory, and
**always `--dry-run` first** — it prints the job plan without submitting anything:

```bash
./preset_targeted_l2.sh yfcc    # TargetedL2  -> folds into yfcc4k_full   (40 jobs)
./preset_targeted_l2.sh osv     # TargetedL2  -> folds into osv_5k_full   (50 jobs, + replot step)
./preset_dtd_blur.sh            # DTD-blur robustness ablation            (40 jobs)
./preset_cfg_ablation.sh        # guidance-scale (cfg) sweep              (80 jobs)
```

Each submits a shard array **plus a merge job chained with `--dependency=afterok`**, so the
merge runs by itself once the array succeeds. Watch with `squeue -u $USER`; logs go to
`slurm_logs/`. Cap concurrency with `MAX_CONCURRENT=32 ./preset_...`.

- First time on a cluster, do the [one-time setup](#usage) below (SBATCH placeholders,
  data paths, conda env) — nothing schedules until that is done.
- The two `yfcc` runs rewrite `yfcc4k_full` **in place** (existing attacks are rebuilt
  from their own unchanged shards, so no data is lost); back up `plots/` if you want the
  current figures kept byte-for-byte.
- `osv` needs one extra GPU-free step afterwards; the command is printed at the end.
- What each experiment does and where its numbers land:
  [Presets](#presets-targeted_l2-dtd_blur-cfg-ablation).

## How it works

The work is a 2D grid: **one job per `(attack × image window)`**.

- Image selection is *prefix-stable and deterministic* (seeded shuffle of the dataset,
  see `utils/datasets.py`). Window *k* of an attack covers images
  `[k·IMAGES_PER_SHARD, (k+1)·IMAGES_PER_SHARD)` of that ordering.
- Each shard job (`main.py evaluate-dataset-shard`) loads only its window, trains +
  evaluates its attack, and writes its results + resumable state into its own
  subdirectory `RESULTS_BASE/shards/<attack>__w<start>_<end>/`. Shards never collide
  and never share state, so they are fully parallel and independently restartable.
- The merge job (`main.py merge-shards`) reads every shard's state file, maps each
  shard's images back onto the global ordering by image id, and reconstructs exactly
  what a single-process `evaluate-dataset` would have produced: the combined
  `<dataset>_<attack>_results.pt`, the displacement / success-rate plots, and the
  restart / sampling-steps / robustness ablation JSONs and plots.

The ordinary single-GPU command is unchanged: `python main.py evaluate-dataset ...`.

## Usage

0. **Point `adversarial_demo/config.yaml` at this machine** — `data_dirs.{yfcc,osv}` must
   reference this cluster's dataset copies, and
   `attack_train_args.*.per_attack.ace.target_image` this checkout's `.media/MIST.png`
   (the committed path is machine-specific). A `plonk` conda env must exist (name
   overridable via `CONDA_ENV` in `config.sh`). The presets in
   [Running the new experiments](#running-the-new-experiments) set everything else
   themselves, so for those you only need this step and step 2.
1. **Edit `config.sh`** — dataset, `TOTAL_IMAGES`, `IMAGES_PER_SHARD`, `ATTACK_TYPES`,
   `ATTACK_BUDGETS`, `EVAL_NUM_STEPS`, and `RESULTS_BASE`. *(Not needed for the presets.)*
2. **Fill the `#SBATCH` placeholders** in `eval_shard.slurm` (account/constraint/qos for
   your GPU partition) and `merge.slurm` (CPU `prepost` partition is ideal — merge needs
   no GPU). Lines start with `##SBATCH`; uncomment to one `#SBATCH` after editing.
3. **Preview** the plan and the task→(attack,window) mapping without submitting:
   ```bash
   ./submit.sh --dry-run
   ```
4. **Submit** the array + a merge job that runs after the array succeeds:
   ```bash
   ./submit.sh
   # cap concurrent array tasks (e.g. fair-share / quota):
   MAX_CONCURRENT=32 ./submit.sh
   ```

With the defaults (4000 images, 100/shard, 7 attacks) that is `7 × 40 = 280` GPU jobs,
each doing 100 images × 2 budgets × 8 restarts.

## Presets (targeted_l2, dtd_blur, cfg ablation)

Three ready-made launchers wrap `submit.sh` for the current experiments. They only set
environment variables (dataset, attacks, budgets, ablations, output base) and call
`submit.sh`, so the #SBATCH placeholders in `eval_shard.slurm` / `merge.slurm` still
have to be filled in first. Add `--dry-run` to any of them to preview.

### `preset_targeted_l2.sh <yfcc|osv>` — fold TargetedL2 into an existing full run

Evaluates `targeted_l2` (2 budgets, sampling-steps + robustness + model-transfer
ablations, matching the existing full runs) and integrates it into the established
results **in place**:

```bash
./preset_targeted_l2.sh yfcc          # -> results/cluster_eval/yfcc4k_full
./preset_targeted_l2.sh osv           # -> results/cluster_eval/osv_5k_full
```

- **YFCC**: `yfcc4k_full` still has every attack's shards, so this computes only the
  `targeted_l2` shards and the merge stitches the **full** attack set (existing 8 +
  targeted_l2) back into the combined results, ablation JSONs and plots. One command.
- **OSV**: `osv_5k_full` keeps the merged per-attack `.pt` files but **not** the shards,
  so the merge stitches only `targeted_l2` (→ `osv_targeted_l2_results.pt`, its own
  figures in `plots_targeted_l2/`). To rebuild the *combined* OSV figures across all
  attacks, run the printed follow-up:
  ```bash
  ./replot_combined.sh osv \
    "encoder sampling dtd ace training_loss unidef targeted_l2" \
    results/cluster_eval/osv_5k_full
  ```
  `replot_combined.sh` reads only the saved `.pt` files (no GPU, seconds).

The robustness transform grid is pinned (`EXTRA_OVERRIDES_STR`) to the JPEG/blur levels
the original runs used, so targeted_l2 lines up point-for-point with the other attacks.

### `preset_dtd_blur.sh` — blur-robustness ablation on YFCC4k

Tests whether training the perturbation with **additional noise** buys robustness to
Gaussian blur. `dtd_blur` is an expectation-over-transformation variant of DTD: each step
the DTD loss is evaluated on the clean protected image *and* a blurred copy at a random
sigma (config `attack_train_args.yfcc.per_attack.dtd_blur`: `blur_sigma_min/max`,
`blur_loss_weight`), whereas plain `dtd` trains on the clean image only.

```bash
./preset_dtd_blur.sh            # -> folds dtd_blur into results/cluster_eval/yfcc4k_full
```

`dtd` already has robustness samples in `yfcc4k_full`, so only the `dtd_blur` shards are
computed and the merge rebuilds the combined robustness JSON over **both**. The payoff:

```
yfcc4k_full/yfcc_robustness_results.json     # numbers: dtd vs dtd_blur
yfcc4k_full/plots/yfcc_robustness_results.*  # overlay across sigma 0..10
```

The blur sweep is sigma `[0 2 4 6 8 10]` while training draws sigma from `[0, 4]`, so the
plot covers both in-distribution and out-of-distribution (6–10) blur.

### `preset_cfg_ablation.sh` — guidance-scale sweep on YFCC4k

The cfg ablation is computed at shard time (each best perturbation re-evaluated at every
`--eval-cfgs` scale), so it needs a **fresh full run**, not a re-merge. This trains the
chosen attacks over the full 4000-image pool with `--run-cfg-ablation` on, into a
separate base (`yfcc4k_cfg`) so the established `yfcc4k_full` run is left untouched:

```bash
./preset_cfg_ablation.sh                       # attacks = dtd sampling (default)
ATTACKS="dtd sampling encoder" ./preset_cfg_ablation.sh
```

Set `ATTACKS` to the attacks you want swept. Only the cfg ablation is enabled by default
(flip `RUN_SAMPLING_STEPS_ABLATION` / `RUN_ROBUSTNESS_ABLATION` /
`RUN_MODEL_TRANSFER_ABLATION` to `1` to add the others).

### New knobs in `config.sh`

The integration presets auto-detect which attacks already have shards under
`RESULTS_BASE/shards` (`integrate_lib.sh`) and merge that set plus the new attack, so they
compose in any order. Robustness scope is the union, because `merge_shards` takes
`robustness_attack_types` explicitly and would otherwise drop the existing curves.

All are environment-overridable (the presets use them): `MERGE_ATTACK_TYPES_STR`
(merge a superset of what the array computed), `RESULTS_BASE` (integrate into an
existing base), `ROBUSTNESS_ATTACK_TYPES_STR` (scope robustness), the `RUN_*_ABLATION`
toggles including `RUN_CFG_ABLATION` + `EVAL_CFGS_STR`, and `EXTRA_OVERRIDES_STR`
(extra `--override key=value` pairs forwarded to both shard and merge). With no
overrides `config.sh` reproduces the original generic run exactly.

## Operational notes

- **Restartable.** Shards resume from their saved state, so a requeued/preempted job
  continues where it stopped. Re-submitting a single failed window is cheap.
- **Merge after partial failure.** `submit.sh` chains the merge with
  `--dependency=afterok`, so it is skipped if *any* shard fails. Re-run the failed
  shards, then submit the merge alone:
  ```bash
  sbatch --export=ALL,CLUSTER_DIR="$PWD" merge.slurm
  ```
  `merge-shards` plots whatever cells are present and prints a coverage summary
  (`Merged N/EXPECTED cells`), so an incomplete run still yields partial plots.
- **OSV-5M subset.** Set `DATASET=osv` and `TOTAL_IMAGES=<subset size>` in `config.sh`.
- **GeoShield** is sharded too: each `geoshield` task generates + evaluates its image
  window out-of-process (`Geoshield/geoshield.py`) into shard-isolated clean/attacked
  dirs (a `run_tag` per window keeps concurrent shards from clobbering each other), and
  `merge-shards` stitches it in like any other attack. It needs the **Geoshield repo**
  next to `plonk/` and the `geoshield.{clean_dir,output_base}` paths set in
  `config.yaml`; remove `geoshield` from `ATTACK_TYPES` in `config.sh` to skip it. Note
  it produces predicted-only results (no ground-truth-GPS line), matching the single-GPU
  GeoShield path, and its per-window clean/attacked image dirs use extra disk.
- **Logs** land in `scripts/cluster/slurm_logs/`.
