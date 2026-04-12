#!/bin/bash

DATASETS=(
    "data/refnerf/gardenspheres"
    "data/refnerf/sedan"
    "data/refnerf/toycar"
)

NAMES=(
    "ref_nerf_garden"
    "ref_nerf_sedan"
    "ref_nerf_toycar"
)

echo "Starting variance measurement batch training (3 runs per config)..."
echo "Total Steps: 15,000 | Densification Stop: 10,000 | Data Factor: 4"

run_and_time () {
    CMD="$1"
    RESULT_DIR="$2"

    mkdir -p "$RESULT_DIR"

    START=$(date +%s)
    eval "$CMD"
    END=$(date +%s)

    DURATION=$((END - START))

    # Save timing info
    {
        echo "=============================="
        echo "Run Timestamp: $(date)"
        echo "Duration (seconds): ${DURATION}"
        printf "Duration (hh:mm:ss): %02d:%02d:%02d\n" \
            $((DURATION/3600)) $((DURATION%3600/60)) $((DURATION%60))
    } > "${RESULT_DIR}/time.txt"

    # Cleanup
    rm -rf "${RESULT_DIR}/ckpts"
    rm -rf "${RESULT_DIR}/renders"
}


for i in "${!DATASETS[@]}"; do
    DATA_DIR="${DATASETS[$i]}"
    NAME="${NAMES[$i]}"

    echo "=================================================================="
    echo "Starting pipeline for: $NAME"
    echo "Path: $DATA_DIR"
    echo "=================================================================="

    for RUN in 1 2 3; do
        echo "--------------------------------------------------"
        echo " Executing RUN $RUN of 3 for $NAME"
        echo "--------------------------------------------------"

        # 1. LoRA Complete
        RESULT="results/${NAME}_4_lora_complete_run${RUN}"
        # run_and_time "
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --warmup-step 4000 \
            --data-dir \"$DATA_DIR\" \
            --result-dir \"$RESULT\"
        " "$RESULT"
        sleep 3

        # 2. LoRA Fixed
        RESULT="results/${NAME}_4_lora_fixed_run${RUN}"
        run_and_time "
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --warmup-step 0 \
            --disable-quota-analysis \
            --lora-quota 0.4 0.4 0.2 \
            --data-dir \"$DATA_DIR\" \
            --result-dir \"$RESULT\"
        " "$RESULT"
        sleep 3

        # 3. LoRA Static
        RESULT="results/${NAME}_4_lora_static_run${RUN}"
        run_and_time "
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --disable-dynamic-rank \
            --lora-rank 16 \
            --data-dir \"$DATA_DIR\" \
            --result-dir \"$RESULT\"
        " "$RESULT"
        sleep 3

        # 4. SH Degree 3
        RESULT="results/${NAME}_4_sh3_run${RUN}"
        run_and_time "
        python simple_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --sh-degree 3 \
            --eval-steps 7000 15000 \
            --data-dir \"$DATA_DIR\" \
            --result-dir \"$RESULT\"
        " "$RESULT"
        sleep 3

        # 5. SH Degree 2
        RESULT="results/${NAME}_4_sh2_run${RUN}"
        run_and_time "
        python simple_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --sh-degree 2 \
            --eval-steps 7000 15000 \
            --data-dir \"$DATA_DIR\" \
            --result-dir \"$RESULT\"
        " "$RESULT"
        sleep 3

    done
done

echo "All variance measurement runs completed successfully!"