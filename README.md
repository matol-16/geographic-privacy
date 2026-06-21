# Protecting images from Diffusion based geolocalization

Project for the class *Multimodal Generative AI* taught by Professor Vicky Kalogetion at Ecole Polytechnique (2026).


## Repository Organisation

This repository is a fork from the repository of the paper [Around the World in 80 Timesteps: A Generative Approach to Global Visual Geolocation](https://github.com/nicolas-dufour/plonk) (Dufour et al., 2025).

It is aimed at researching adversarial methods to prevent images from being localizable by the diffusion and flow matching models presented by Dufour et al. (2025).

All the new contributions are in the the *adversarial_demo* folder, which is added at the root of the *plonk* project.

The project paper and poster are also in the *adversarial_demo* folder.

Occasional additions are made to the requirements.txt file.

In order to install librairies and get the code running, refer to the README_ORIGINAL_PAPER.md file. Below, we propose a simple method to download dependencies, and present the code's structure

## Installation

Tested with **Python 3.10** and a CUDA-capable GPU.

### 1. Clone the repository

```bash
# Pick a parent folder (referred to below as <repos>). GeoShield, if used, is cloned
# next to plonk in this same folder (see "Installing GeoShield").
mkdir -p <repos> && cd <repos>
git clone https://github.com/<your-fork>/plonk.git
cd plonk
```

### 2. Create the conda env and install the package

```bash
conda create -n plonk python=3.10
conda activate plonk

# Installs the core PLONK package + its base dependencies (from setup.py).
pip install -e .
```

If you need a specific CUDA build of PyTorch, install it first following the
[PyTorch guide](https://pytorch.org/get-started/locally/), then run `pip install -e .`.

### 3. Install the extra dependencies for adversarial_demo

The base package does **not** cover everything the `adversarial_demo` code needs (plotting,
maps, metrics, dataset/model download). Install the full set from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Notes (the file marks every package added on top of the original PLONK env with `[added]`):

- **`cartopy`** (used for the map plots) needs system GEOS/PROJ libraries and can fail to
  build via pip. If so, install it from conda-forge instead:
  ```bash
  conda install -c conda-forge cartopy
  ```
### 4. Models and cache

The pretrained PLONK models (`nicolas-dufour/PLONK_YFCC`, `nicolas-dufour/PLONK_OSV_5M`,
configured in `adversarial_demo/config.yaml` under `pipelines:`) download automatically from
the HuggingFace Hub on first use. To keep large downloads off your home quota, point the HF
cache at a data disk (matching `data_root` in `config.yaml`):

```bash
export HF_HOME=/Data/<you>/hf_cache        # adjust to your scratch/data path
```

## Installing GeoShield (optional — needed for the `geoshield` attack)

GeoShield ([thinwayliu/Geoshield](https://github.com/thinwayliu/Geoshield.git)) is an
out-of-process attack we fold into the evaluation pipeline as just another attack. It is run
as a subprocess (`Geoshield/geoshield.py`), so the repo must sit **next to `plonk/`** in the
same parent folder — the integration auto-detects the parent that holds both:

```
<repos>/
├── plonk/        # this repository
└── Geoshield/    # GeoShield, cloned below (note the capital G)
```

```bash
cd <repos>
git clone https://github.com/thinwayliu/Geoshield.git
```

**Dependencies:** GeoShield uses CLIP surrogates via HuggingFace `transformers` plus
`hydra-core`/`omegaconf`, `torch`, `torchvision`, `numpy`, `pillow`, `tqdm` and `wandb` — all
already provided by the `plonk` env above, so **no extra install is needed**. Just run it with
`conda activate plonk`.

**Surrogate weights:** on first run GeoShield downloads its CLIP backbones from the HuggingFace
Hub (`openai/clip-vit-base-patch16`, `-patch32`, `-large-patch14-336`, and
`laion/CLIP-ViT-G-14-laion2B-s12B-b42K`). The LAION ViT-G/14 model is several GB, so allow disk
space and use the same `HF_HOME` as above. (The optional GroundingDINO region-aware mode from
GeoShield's own README is **not** required for our default `ensemble_3models` config.)

**Configuration:** GeoShield's settings live in the `geoshield:` block of
`adversarial_demo/config.yaml`:

```yaml
geoshield:
  attack_name: "geoshield"
  steps: 100
  clean_dir:   "/Data/<you>/hf_cache/clean_yfcc_images"          # where seeded clean images are copied
  output_base: "/Data/<you>/hf_cache/attacked_yfcc_images_geoshield"  # where attacked images are written
  epsilons: null          # null => derived per budget as round(budget * 255) (0.0314 -> 8/255)
  repos_root: null        # null => auto-detect the folder holding plonk/ and Geoshield/
  script: "Geoshield/geoshield.py"
  python: null            # null => the current interpreter (run inside the plonk env)
```

GeoShield generates adversarial images then evaluates them through the same backbone as the
trainable attacks, reporting both the predicted-displacement metric **and** the true-GPS
displacement metric (it selects labelled dataset images, so ground truth is available).
WandB logging is enabled inside GeoShield; run `wandb offline` (or set
`WANDB_MODE=disabled`) beforehand to skip uploads.

## Datasets

You can download the **YFCC4k** dataset by running the dedicated
`adversarial_demo/utils/build_yfcc4k_from_revisiting_im2gps.py` script. Dataset folders are
configured under `data_root` / `data_dirs` / `build_yfcc4k` in `adversarial_demo/config.yaml`,
and can also be overridden through the argument parser.

The **OSV-5M** test split downloads automatically (from the HuggingFace Hub) the first time an
evaluation uses it.

## Running evaluations

All commands below run from `adversarial_demo/` with `conda activate plonk`.

**Single GPU** (quick test: every attack on a small image count, with the ablations):

```bash
bash scripts/test_all_attacks.sh
```

**Full dataset across many GPUs** (e.g. all 4000 YFCC4k images on Jean Zay / SLURM). The
run is sharded into one job per `(attack × image window)`, then merged into the same
results + plots a single-GPU run would produce. Three steps, from `scripts/cluster/`:

```bash
cd scripts/cluster
# 1. Edit config.sh        -> dataset, TOTAL_IMAGES, IMAGES_PER_SHARD, ATTACK_TYPES, budgets, paths
# 2. Fill the #SBATCH placeholders in eval_shard.slurm and merge.slurm (account / GPU / qos)
# 3. Submit the shard array + a merge job that runs once it succeeds:
./submit.sh                 # add --dry-run first to preview the task -> (attack, window) grid
```

See [`adversarial_demo/scripts/cluster/README.md`](adversarial_demo/scripts/cluster/README.md)
for the full workflow (restartable shards, partial-failure recovery, OSV-5M subsets, GeoShield
sharding). The underlying commands — `python main.py evaluate-dataset-shard ...` then
`python main.py merge-shards ...` — can also be run by hand.

## Code structure

As mentioned above, all the new code in is *adversarial_demo*. 

It is organized as follows ($\dagger$ specifies if the given code was coded with the help of an AI coding assistant):

**Evaluation scripts**:
- 2 notebooks to test our framework in the folder *demo_notebooks*:
    - *eval_notebook_attacks.ipynb* is a simple notebook that allows testing our adversarial framework on an image saved in plonk/.media
    - *notebook_binary_test.ipynb $*\dagger$ allows testing (and reproducing our figures) our framework for image GPS localization manipulation.
- *scripts_eval.py* allows running several evaluations (attack final step displacement, link between attack success and localizability...). Evaluations can be very long (several tens of hours on an A5000 GPU whithout parallelization) because one needs to train every attack for multiple attack budgets on multiple images.



**Code used**:
- *adversarial_eval.py *$\dagger$ implements the code to evaluate our attacks on YFCC4K and OSV-5M.
- *adversarial_metrics.py* and *adversarial_utils.py* provide basic functions sued throughout our code. The metrics implemented are the ones described in section 4 of our report
- *attacks.py* coordinates the attack scripts, which are contained in *encoder_attacks.py* and *trajectory_attacks.py* for Encoder and Diffusion Trajectory Deviation respectively. It provides a single method to evaluate both attacks on an image.
- *pipe_trajectory* implements a hereditary class of *Plonk*'s *PlonkPipeline* class, which returns diffusion trajectories when calling the pipeline. This allowed testing our attack's effect on whole trajectories, not just predicted locations.
- *plots_adversarial_attacks.py *$\dagger$ provides the functions to plot the figures that present the results of attack evaluationsin the paper.
- *build_yfcc4k_from_revisiting_im2gps.py *$\dagger$ allows downloading the YFCC4K dataset (only needs to run once)

**Others**
- Evaluation results are stored in the *results* folder. One can run directly our plotting methods that retrieve stored results.
- *archive_code* consists of code used during the exploratory phase.

The adversarial demo evaluation pipeline in `adversarial_demo` now supports resuming interrupted runs from a saved state file, and the saved results include compact `image_ids` and `image_indices` so each output can be mapped back to the source dataset order.
