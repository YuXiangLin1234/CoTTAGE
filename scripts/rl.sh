#!/bin/bash
# Stage 2: GRPO with format and global cross-instruction preference rewards.
# The base model is the merged SFT checkpoint produced by scripts/merge_lora.sh.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

if (( RL_NUM_GENERATIONS <= 0 || RL_BATCH_SIZE <= 0 || NPROC_PER_NODE <= 0 )); then
    echo "RL batch size, generation count, and GPU count must be positive" >&2
    exit 1
fi
if (( NUM_GENERATIONS != RL_NUM_GENERATIONS )); then
    echo "NUM_GENERATIONS must equal RL_NUM_GENERATIONS for reward grouping" >&2
    exit 1
fi
if (( RL_BATCH_SIZE % (2 * RL_NUM_GENERATIONS) != 0 )); then
    echo "RL_BATCH_SIZE must be a multiple of 2 * RL_NUM_GENERATIONS to keep partners on one GPU" >&2
    exit 1
fi
if (( PAIR_CHUNK_ROWS != RL_BATCH_SIZE * NPROC_PER_NODE / RL_NUM_GENERATIONS )); then
    echo "PAIR_CHUNK_ROWS must match the global number of clips per step" >&2
    exit 1
fi

RL_BASE_MODEL="${RL_BASE_MODEL:-$SFT_MERGED_DIR}"
mkdir -p "$RL_OUTPUT_DIR"

ENABLE_AUDIO_OUTPUT=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
swift rlhf \
    --rlhf_type grpo \
    --model "$RL_BASE_MODEL" \
    --train_type lora \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --custom_register_path "$REPO_ROOT/cottage/dataset_plugin.py" \
                           "$REPO_ROOT/cottage/reward_plugin.py" \
    --dataset cottage_rl \
    --check_model false \
    --torch_dtype bfloat16 \
    --num_train_epochs "$RL_EPOCHS" \
    --learning_rate "$RL_LR" \
    --warmup_ratio 0.05 \
    --beta "$RL_BETA" \
    --per_device_train_batch_size "$RL_BATCH_SIZE" \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --num_generations "$RL_NUM_GENERATIONS" \
    --temperature "$RL_TEMPERATURE" \
    --top_p 0.95 \
    --top_k 50 \
    --max_length "$RL_MAX_LENGTH" \
    --max_completion_length "$RL_MAX_COMPLETION_LENGTH" \
    --reward_funcs cottage_format cottage_preference \
    --reward_weights 1 1 \
    --dataset_shuffle false \
    --train_dataloader_shuffle false \
    --dataloader_num_workers 4 \
    --dataset_num_proc "$DATASET_NUM_PROC" \
    --ddp_timeout 7200 \
    --log_completions true \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len "$RL_MAX_LENGTH" \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora true \
    --vllm_max_lora_rank "$LORA_RANK" \
    --save_steps 200 \
    --save_total_limit 10 \
    --save_only_model true \
    --add_version false \
    --report_to "$REPORT_TO" \
    --output_dir "$RL_OUTPUT_DIR"
