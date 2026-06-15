
#!/usr/bin/env bash

DATA_IMAGES_PATH=/Data/mathias.ollu/hf_cache/datasets/YFCC100M/yfcc4k/images
OUTPUT_PATH=/Data/mathias.ollu/hf_cache/attacked_yfcc_images_geoshield
CLEAN_IMAGES_PATH=/Data/mathias.ollu/hf_cache/clean_yfcc_images

N_IMAGES_TO_EVAL=100

EPSILON=4


mkdir -p "$CLEAN_IMAGES_PATH"

mkdir -p "$OUTPUT_PATH"

#just uncomment once !
mapfile -t selected_images < <(
    find "$DATA_IMAGES_PATH" -maxdepth 1 -type f | shuf -n "$N_IMAGES_TO_EVAL"
)

cp "${selected_images[@]}" "$CLEAN_IMAGES_PATH"

python Geoshield/geoshield.py \
    data.cle_data_path=$CLEAN_IMAGES_PATH \
    data.tgt_data_path=$CLEAN_IMAGES_PATH \
    data.output="${OUTPUT_PATH}"_e_$EPSILON \
    data.num_samples=100 \
    optim.epsilon=$EPSILON \
    optim.steps=100