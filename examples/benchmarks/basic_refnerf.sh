SCENE_DIR="data/refnerf"
RESULT_DIR="results/benchmark/sh3"
SCENE_LIST="gardenspheres sedan toycar" # treehill flowers
RENDER_TRAJ_PATH="ellipse"

for SCENE in $SCENE_LIST;
do
    if [ "$SCENE" = "bonsai" ] || [ "$SCENE" = "counter" ] || [ "$SCENE" = "kitchen" ] || [ "$SCENE" = "room" ]; then
        DATA_FACTOR=2
    else
        DATA_FACTOR=4
    fi

    echo "Running $SCENE"

    # train without eval
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default --disable-viewer --data-factor $DATA_FACTOR \
        --max-steps 7000 \
        --eval-steps 2000 7000 \
        --save-steps 2000 \
        --render-traj-path $RENDER_TRAJ_PATH \
        --data-dir "${SCENE_DIR}/${SCENE}/" \
        --result-dir $RESULT_DIR/$SCENE/

    OUT="$RESULT_DIR/$SCENE"

    echo "=== Cleaning up ckpts/ and renders/ for $OUT ==="
    rm -rf "$OUT/ckpts/" "$OUT/renders/"
done


for SCENE in $SCENE_LIST;
do
    echo "=== Eval Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/val*.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done

    echo "=== Train Stats ==="

    for STATS in $RESULT_DIR/$SCENE/stats/train*_rank0.json;
    do  
        echo $STATS
        cat $STATS; 
        echo
    done
done