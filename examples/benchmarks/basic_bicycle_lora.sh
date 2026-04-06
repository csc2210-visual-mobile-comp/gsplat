SCENE_DIR="data/refnerf"
SCENE_LIST="sedan"
LORA_RANK=32

for MODE in "add" "append"; do
    if [ "$MODE" = "append" ]; then
        LORA_APPEND_FLAG="--lora_append"
        RESULT_DIR="results/benchmark/lora_append"
    else
        LORA_APPEND_FLAG=""
        RESULT_DIR="results/benchmark/lora_add"
    fi

    for SCENE in $SCENE_LIST;
    do
        if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
            DATA_FACTOR=2
        else
            DATA_FACTOR=4
        fi

        echo "Running $SCENE (mode=$MODE)"

        # train without eval
        CUDA_VISIBLE_DEVICES=0 python lora_trainer.py default \
            --max_steps 7000 \
            --eval_steps 7000 \
            --save_steps 7000 \
            --disable_viewer \
            --data_factor $DATA_FACTOR \
            --data_dir $SCENE_DIR/$SCENE/ \
            --result_dir $RESULT_DIR/$SCENE/ \
            --lora_rank $LORA_RANK \
            $LORA_APPEND_FLAG

    done
done

