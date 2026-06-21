
#!/usr/bin/env bash

DATA_IMAGES_PATH=/Data/mathias.ollu/hf_cache/datasets/osv5m/images/test/00
OUTPUT_PATH=/Data/mathias.ollu/hf_cache/attacked_osv_images_mattack
CLEAN_IMAGES_PATH=/Data/mathias.ollu/hf_cache/clean_osv_images
TARGET_IMAGES_PATH=/Data/mathias.ollu/hf_cache/target_osv_images

N_IMAGES_TO_EVAL=100

EPSILON=8


mkdir -p "$CLEAN_IMAGES_PATH"

mkdir -p "$OUTPUT_PATH"

mkdir -p "$TARGET_IMAGES_PATH"

#uncomment just once. Usually already done.

# mapfile -t selected_images < <(
#     find "$DATA_IMAGES_PATH" -maxdepth 1 -type f | shuf -n "$N_IMAGES_TO_EVAL"
# )

# cp "${selected_images[@]}" "$CLEAN_IMAGES_PATH"

#Add random target images from OSV dataset. These should be different from clean images (random choice)

# mapfile -t selected_images < <(
#     find "$DATA_IMAGES_PATH" -maxdepth 1 -type f | shuf -n "$N_IMAGES_TO_EVAL"
# )

# cp "${selected_images[@]}" "$TARGET_IMAGES_PATH"


python Geoshield/m-attack.py \
    data.cle_data_path=$CLEAN_IMAGES_PATH \
    data.tgt_data_path=$CLEAN_IMAGES_PATH \
    data.output="${OUTPUT_PATH}"_e_$EPSILON \
    data.num_samples=100 \
    optim.epsilon=$EPSILON \
    optim.steps=100