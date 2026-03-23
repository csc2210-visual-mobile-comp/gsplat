#!/usr/bin/env bash
set -euo pipefail
# LoRA dynamic-rank benchmark — refnerf dataset

SCENE_DIR="data/refnerf"
RESULT_DIR="results/benchmark/lora"
SCENE_LIST="gardenspheres sedan toycar"
RENDER_TRAJ_PATH="ellipse"
GPU=0

MAX_STEPS=7000
EVAL_STEPS="2000 4000 7000"
SAVE_STEPS="2000 7000"

# ---------------------------------------------------------------------------
# Fix image extensions once per scene (JPG → needed by the loader)
# ---------------------------------------------------------------------------
for f in "$SCENE_DIR/sedan/images/"*.jpg; do
    [ -e "$f" ] || continue
    mv "$f" "${f%.jpg}.JPG" || true
done

# ---------------------------------------------------------------------------
# Helper: train one variant, then evaluate every saved checkpoint
# ---------------------------------------------------------------------------
run_variant() {
    local SCENE=$1
    local DATA_FACTOR=$2
    local VARIANT=$3
    shift 3
    local EXTRA_ARGS=("$@")
    local OUT="$RESULT_DIR/$SCENE/$VARIANT"

    echo ""
    echo "=== Training  scene=$SCENE  variant=$VARIANT ==="

    CUDA_VISIBLE_DEVICES=$GPU python lora_trainer.py default \
        --disable-viewer \
        --data-factor "$DATA_FACTOR" \
        --data-dir "$SCENE_DIR/$SCENE/" \
        --result-dir "$OUT" \
        --max-steps "$MAX_STEPS" \
        --eval-steps $EVAL_STEPS \
        --save-steps $SAVE_STEPS \
        "${EXTRA_ARGS[@]}"

    echo "=== Evaluating  scene=$SCENE  variant=$VARIANT ==="

    for CKPT in "$OUT/ckpts/"*.pt; do
        [ -e "$CKPT" ] || continue
        CUDA_VISIBLE_DEVICES=$GPU python lora_trainer.py default \
            --disable-viewer \
            --data-factor "$DATA_FACTOR" \
            --data-dir "$SCENE_DIR/$SCENE/" \
            --result-dir "$OUT" \
            --ckpt "$CKPT" \
            "${EXTRA_ARGS[@]}"
    done

    # -----------------------------------------------------------------------
    # Cleanup to save disk
    # -----------------------------------------------------------------------
    echo "=== Cleaning up ckpts/ and renders/ for $OUT ==="
    rm -rf "$OUT/ckpts/" "$OUT/renders/"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SCENE in $SCENE_LIST; do
    if [ "$SCENE" = "bonsai" ] || \
       [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || \
       [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    # lora_d_high
    run_variant "$SCENE" "$DATA_FACTOR" lora_d_high \
        --lora-quota 0.40 0.40 0.20

    # lora_d_mid
    run_variant "$SCENE" "$DATA_FACTOR" lora_d_mid \
        --lora-quota 0.20 0.60 0.20

    # lora_d_low
    run_variant "$SCENE" "$DATA_FACTOR" lora_d_low \
        --lora-quota 0.10 0.20 0.70
done

# ---------------------------------------------------------------------------
# Print collected stats
# ---------------------------------------------------------------------------
echo ""
echo "================================================================"
echo " RESULTS SUMMARY"
echo "================================================================"

VARIANTS="lora_d_high lora_d_mid lora_d_low"

for SCENE in $SCENE_LIST; do
    echo ""
    echo "--- $SCENE ---"
    for VARIANT in $VARIANTS; do
        OUT="$RESULT_DIR/$SCENE/$VARIANT"
        echo ""
        echo "  [$VARIANT]"

        echo "  -- Eval stats --"
        for STATS in "$OUT/stats/"val*.json; do
            [ -f "$STATS" ] || continue
            echo "  $STATS"
            cat "$STATS"
            echo ""
        done

        echo "  -- Train stats --"
        for STATS in "$OUT/stats/"train*_rank0.json; do
            [ -f "$STATS" ] || continue
            echo "  $STATS"
            cat "$STATS"
            echo ""
        done
    done
done