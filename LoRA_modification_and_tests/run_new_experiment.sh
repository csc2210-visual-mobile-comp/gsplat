#!/bin/bash

# 1. Define ONLY the specific datasets you want to run tonight
DATASETS=(
    "data/ref_real/gardenspheres"
    "data/ref_real/sedan"
    "data/ref_real/toycar"
)

# 2. Define clean names for your output folders
NAMES=(
    "ref_nerf_garden"
    "ref_nerf_sedan"
    "ref_nerf_toycar"
)

echo "Starting variance measurement batch training (3 runs per config)..."
echo "Total Steps: 15,000 | Densification Stop: 10,000 | Data Factor: 4"

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

        # 1. LoRA Complete (Dynamic GMM Warmup)
        echo ">>> Running LoRA Complete..."
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --warmup-step 4000 \
            --data-dir "$DATA_DIR" \
            --result-dir "results/${NAME}_4_lora_complete_run${RUN}"
        sleep 3
        pkill -f "lora_trainer.py"
        sleep 3

        # 2. LoRA Fixed (Static Quotas)
        echo ">>> Running LoRA Fixed..."
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --warmup-step 0 \
            --disable-quota-analysis \
            --lora-quota 0.4 0.4 0.2 \
            --data-dir "$DATA_DIR" \
            --result-dir "results/${NAME}_4_lora_fixed_run${RUN}"
        sleep 3
        pkill -f "lora_trainer.py"
        sleep 3

        # 3. LoRA Static (Fixed Rank 16)
        echo ">>> Running LoRA Static..."
        python lora_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --disable-dynamic-rank \
            --lora-rank 16 \
            --data-dir "$DATA_DIR" \
            --result-dir "results/${NAME}_4_lora_static_run${RUN}"
        sleep 3
        pkill -f "lora_trainer.py"
        sleep 3

        # 4. Standard SH Degree 3
        echo ">>> Running Standard SH3..."
        python simple_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --sh-degree 3 \
            --data-dir "$DATA_DIR" \
            --result-dir "results/${NAME}_4_sh3_run${RUN}"
        sleep 3
        pkill -f "simple_trainer.py"
        sleep 3

        # 5. Standard SH Degree 2
        echo ">>> Running Standard SH2..."
        python simple_trainer.py default \
            --max-steps 15000 \
            --strategy.refine-stop-iter 10000 \
            --disable-viewer \
            --data-factor 4 \
            --sh-degree 2 \
            --data-dir "$DATA_DIR" \
            --result-dir "results/${NAME}_4_sh2_run${RUN}"
        sleep 3
        pkill -f "simple_trainer.py"
        sleep 3

    done
done

echo "All variance measurement runs completed successfully!"