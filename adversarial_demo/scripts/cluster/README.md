# Multi-node dataset evaluation (Jean Zay / SLURM)

Run the full-dataset attack evaluation (e.g. all of YFCC4k = 4000 images, the attacks
from `scripts/test_all_attacks.sh`) by **sharding it across many GPU jobs**, then merging.
A single GPU is far too slow: ~30 s/restart × 8 restarts × 4000 images × 7 attacks × 2
budgets is thousands of GPU-hours, so we split it into independent jobs.

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

1. **Edit `config.sh`** — dataset, `TOTAL_IMAGES`, `IMAGES_PER_SHARD`, `ATTACK_TYPES`,
   `ATTACK_BUDGETS`, `EVAL_NUM_STEPS`, and `RESULTS_BASE`.
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
