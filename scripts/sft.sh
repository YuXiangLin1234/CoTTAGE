#!/bin/bash
# Stage 1: LoRA SFT on the distilled reasoning traces
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

mkdir -p "$SFT_OUTPUT_DIR"

ENABLE_AUDIO_OUTPUT=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift sft \
    --model "$BASE_MODEL" \
    --train_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --custom_register_path "$REPO_ROOT/cottage/dataset_plugin.py" \
    --dataset cottage_sft \
    --check_model false \
    --torch_dtype bfloat16 \
    --num_train_epochs "$SFT_EPOCHS" \
    --learning_rate "$SFT_LR" \
    --per_device_train_batch_size "$SFT_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --max_length "$SFT_MAX_LENGTH" \
    --split_dataset_ratio 0.01 \
    --dataloader_num_workers 8 \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --ddp_timeout 7200 \
    --eval_on_start false \
    --save_steps 1000 \
    --save_total_limit 3 \
    --save_only_model true \
    --add_version false \
    --report_to "$REPORT_TO" \
    --output_dir "$SFT_OUTPUT_DIR"
