#!/usr/bin/env bash
set -euo pipefail

SCENE_DIR="data/refnerf"
RESULT_DIR="results/benchmark/lora"
SCENE_LIST="sedan"
RENDER_TRAJ_PATH="ellipse"

CONFIG_NAME="default"
LORA_RANK=32

# Define LoRA target combinations
LORA_TARGET_SETS=(
    "quats scales"
    "quats scales opacities"
)

# Optional: fix extension once
for f in data/refnerf/sedan/images/*.jpg; do
    mv "$f" "${f%.jpg}.JPG" || true
done

for SCENE in $SCENE_LIST; do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ] || [ "$SCENE" = "sedan" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    for TARGETS in "${LORA_TARGET_SETS[@]}"; do
        echo "Running $SCENE with targets: $TARGETS"

        # Convert "colors scales" → "colors_scales" for folder name
        TARGET_TAG=$(echo "$TARGETS" | tr ' ' '_')

        SCENE_RESULT_DIR="${RESULT_DIR}/${SCENE}_${TARGET_TAG}"

        # Convert string → array
        read -a TARGET_ARRAY <<< "$TARGETS"

        CUDA_VISIBLE_DEVICES=0 python partial_lora_trainer.py "$CONFIG_NAME" \
            --max-steps 8000 \
            --eval-steps 2000 4000 8000 \
            --save-steps 2000 \
            --disable-viewer \
            --data-factor "$DATA_FACTOR" \
            --render-traj-path "$RENDER_TRAJ_PATH" \
            --data-dir "${SCENE_DIR}/${SCENE}/" \
            --result-dir "$SCENE_RESULT_DIR" \
            --lora-rank "$LORA_RANK" \
            --lora-target "${TARGET_ARRAY[@]}"

        OUT="${SCENE_RESULT_DIR}"

        echo "=== Cleaning up ckpts/ and renders/ for $OUT ==="
        rm -rf "$OUT/ckpts/" "$OUT/renders/"
    done
done

# Print stats
for SCENE in $SCENE_LIST; do
    for TARGETS in "${LORA_TARGET_SETS[@]}"; do
        TARGET_TAG=$(echo "$TARGETS" | tr ' ' '_')
        SCENE_RESULT_DIR="${RESULT_DIR}/${SCENE}_${TARGET_TAG}"

        echo "=== $SCENE ($TARGET_TAG): Eval Stats ==="
        for STATS in "${SCENE_RESULT_DIR}"/stats/val*.json; do
            [ -e "$STATS" ] || continue
            echo "$STATS"
            cat "$STATS"
            echo
        done

        echo "=== $SCENE ($TARGET_TAG): Train Stats ==="
        for STATS in "${SCENE_RESULT_DIR}"/stats/train*_rank0.json; do
            [ -e "$STATS" ] || continue
            echo "$STATS"
            cat "$STATS"
            echo
        done
    done
done