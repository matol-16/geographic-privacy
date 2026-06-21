#!/usr/bin/env bash
#
# Shared helper for running GeoShield as "just another attack":
#   1. Select clean images with the exact seeded, prefix-stable selection used by
#      the attack backbone (utils/datasets.select_yfcc_image_paths), so GeoShield
#      is attacked/evaluated on the same images as encoder/diffusion/etc.
#   2. Run GeoShield once per budget (epsilon on the 0-255 scale).
#   3. Evaluate the clean/attacked pairs via `evaluate-geoshield-vs-diffusion`, so
#      results are stored as `<dataset>_geoshield_results.pt` in <results_dir> --
#      the same naming/location as every other attack (ready to overlay in plots).
#
# Source this file, then call:
#
#   geoshield_generate_and_eval \
#       <project_dir> <repos_root> <config_path> <dataset> <dataset_root> \
#       <clean_images_path> <output_base> <n_images> <steps> <attack_name> \
#       <results_dir> <plots_dir> <budgets_array_name> <epsilons_array_name>
#
# The last two arguments are the NAMES of two parallel bash arrays, e.g.:
#   ATTACK_BUDGETS=( 0.0157 0.0314 )   # L-inf budgets (0-1 scale, for the evaluator)
#   EPSILONS=( 4 8 )                   # matching GeoShield epsilon (0-255 scale)
#   geoshield_generate_and_eval ... ATTACK_BUDGETS EPSILONS
#
# Note: YFCC-only image selection (uses select_yfcc_image_paths).

geoshield_generate_and_eval() {
  local project_dir="$1" repos_root="$2" config_path="$3" dataset="$4" dataset_root="$5"
  local clean_path="$6" output_base="$7" n_images="$8" steps="$9" attack_name="${10}"
  local results_dir="${11}" plots_dir="${12}"
  local -n _budgets="${13}" _epsilons="${14}"

  if [[ ${#_budgets[@]} -ne ${#_epsilons[@]} ]]; then
    echo "GeoShield: budgets (${#_budgets[@]}) and epsilons (${#_epsilons[@]}) must have equal length." >&2
    return 1
  fi

  # ---- 1. Seeded image selection (identical to the attack backbone). -------- #
  echo "GeoShield: selecting ${n_images} images with the seeded backbone selection..."
  local selected
  mapfile -t selected < <(
    PYTHONPATH="$project_dir" python - "$config_path" "$dataset" "$dataset_root" "$n_images" <<'PY'
import sys
import yaml
from utils.datasets import select_yfcc_image_paths

config_path, dataset, dataset_root, n_images = sys.argv[1:5]
with open(config_path) as f:
    config = yaml.safe_load(f)

seed = int(config.get("seed", 0))
local_dir = config.get("data_dirs", {}).get(dataset) or dataset_root

for path in select_yfcc_image_paths(int(n_images), seed=seed, local_dir=local_dir):
    print(path)
PY
  )
  if [[ ${#selected[@]} -eq 0 ]]; then
    echo "GeoShield: no images selected; check dataset_root / config." >&2
    return 1
  fi

  # Repopulate the clean dir with exactly the seeded selection (drop stale images
  # so the evaluator matches the same set the other attacks use).
  rm -rf "$clean_path"
  mkdir -p "$clean_path"
  cp "${selected[@]}" "$clean_path"
  echo "GeoShield: copied ${#selected[@]} clean images to ${clean_path}."

  local clean_basename
  clean_basename="$(basename "$clean_path")"

  # ---- 2. Run GeoShield once per budget, collecting the attacked dirs. ------ #
  local attacked_dirs=()
  local i epsilon output_path attacked_dir
  for i in "${!_budgets[@]}"; do
    epsilon="${_epsilons[$i]}"
    output_path="${output_base}_e_${epsilon}"
    mkdir -p "$output_path"

    echo "GeoShield: running attack (epsilon=${epsilon}/255, steps=${steps})..."
    ( cd "$repos_root" && python Geoshield/geoshield.py \
        data.cle_data_path="$clean_path" \
        data.tgt_data_path="$clean_path" \
        data.output="$output_path" \
        data.num_samples="$n_images" \
        optim.epsilon="$epsilon" \
        optim.steps="$steps" )

    # GeoShield writes to <output>/img/<config_hash>+geoshield/<clean_basename>/.
    # Pick the most recently written matching dir for this run.
    attacked_dir="$(find "${output_path}/img" -mindepth 2 -maxdepth 2 -type d \
        -name "$clean_basename" -path "*+geoshield/*" -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-)"
    if [[ -z "$attacked_dir" ]]; then
      echo "GeoShield: could not locate output under ${output_path}/img." >&2
      return 1
    fi
    echo "GeoShield: attacked images (eps=${epsilon}): ${attacked_dir}"
    attacked_dirs+=("$attacked_dir")
  done

  # One clean dir per budget (same seeded set for every budget).
  local clean_dirs=()
  for _ in "${_budgets[@]}"; do clean_dirs+=("$clean_path"); done

  # ---- 3. Evaluate the pairs -> results stored like other attacks. ---------- #
  echo "GeoShield: evaluating clean/attacked pairs with the shared backbone..."
  ( cd "$project_dir" && python main.py evaluate-geoshield-vs-diffusion \
      --config "$config_path" \
      --dataset "$dataset" \
      --attack-name "$attack_name" \
      --attack-budgets "${_budgets[@]}" \
      --clean-image-dirs "${clean_dirs[@]}" \
      --attacked-image-dirs "${attacked_dirs[@]}" \
      --n-images "$n_images" \
      --results-dir "$results_dir" \
      --plots-dir "$plots_dir" )
}
