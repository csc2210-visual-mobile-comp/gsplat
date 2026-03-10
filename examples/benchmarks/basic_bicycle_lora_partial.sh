#!/usr/bin/env bash
set -euo pipefail

SCENE_DIR="data/360_v2"
RESULT_DIR="results/benchmark/lora"
SCENE_LIST="bicycle"   # "garden bicycle stump bonsai counter kitchen room"
RENDER_TRAJ_PATH="ellipse"

# Use the config preset that points to LoRATargetStrategyAB.
# If you changed your preset name, update this.
CONFIG_NAME="default"

# LoRA options
LORA_TARGETS=(colors scales quats)
LORA_RANK=16

for SCENE in $SCENE_LIST; do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    CUDA_VISIBLE_DEVICES=0 python partial_lora_trainer.py "$CONFIG_NAME" \
        --max_steps 8000 \
        --eval_steps 2000 4000 8000  \
        --save_steps 2000 \
        --disable_viewer \
        --data_factor "$DATA_FACTOR" \
        --render_traj_path "$RENDER_TRAJ_PATH" \
        --data_dir "${SCENE_DIR}/${SCENE}/" \
        --result_dir "${RESULT_DIR}/${SCENE}/" \
        --lora_rank "$LORA_RANK" \
        --lora_target "${LORA_TARGETS[@]}"

    for CKPT in "${RESULT_DIR}/${SCENE}"/ckpts/*.pt; do
        CUDA_VISIBLE_DEVICES=0 python partial_lora_trainer.py "$CONFIG_NAME" \
            --disable_viewer \
            --data_factor "$DATA_FACTOR" \
            --render_traj_path "$RENDER_TRAJ_PATH" \
            --data_dir "${SCENE_DIR}/${SCENE}/" \
            --result_dir "${RESULT_DIR}/${SCENE}/" \
            --lora_rank "$LORA_RANK" \
            --lora_target "${LORA_TARGETS[@]}" \
            --ckpt "$CKPT"
    done
done

for SCENE in $SCENE_LIST; do
    echo "=== $SCENE: Eval Stats ==="

    for STATS in "${RESULT_DIR}/${SCENE}"/stats/val*.json; do
        [ -e "$STATS" ] || continue
        echo "$STATS"
        cat "$STATS"
        echo
    done

    echo "=== $SCENE: Train Stats ==="

    for STATS in "${RESULT_DIR}/${SCENE}"/stats/train*_rank0.json; do
        [ -e "$STATS" ] || continue
        echo "$STATS"
        cat "$STATS"
        echo
    done
done