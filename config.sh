#!/bin/bash
# Shared configuration. Every value can be overridden from the environment,
# e.g. `SFT_LR=5e-5 bash scripts/sft.sh`.

set -a

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PYTHON:=python3}"

: "${DATA_ROOT:=$REPO_ROOT/data}"
: "${CKPT_ROOT:=$REPO_ROOT/ckpts}"
: "${RESULTS_ROOT:=$REPO_ROOT/results}"

# Training audio
: "${RELATE_AUDIO_DIR:=$DATA_ROOT/relate/wav}"
: "${XACLE_AUDIO_DIR:=$DATA_ROOT/xacle/wav/train}"
: "${AUDIO_CACHE_DIR:=$DATA_ROOT/cache}"

# Evaluation audio and score sheets
: "${XACLE_TEST_DIR:=$DATA_ROOT/xacle/wav/test}"
: "${XACLE_TEST_CSV:=$DATA_ROOT/xacle/test.csv}"
: "${RELATE_CSV:=$DATA_ROOT/relate/scores/REL.csv}"
: "${PAM_CSV:=$DATA_ROOT/pam/human_eval/audio/scores.csv}"
: "${PAM_AUDIO_ROOT:=$DATA_ROOT/pam/human_eval/audio}"
: "${CLOTHO_CSV:=$DATA_ROOT/clotho/scores.csv}"
: "${CLOTHO_AUDIO_ROOT:=$DATA_ROOT/clotho/wav}"

# Models
: "${BASE_MODEL:=Qwen/Qwen2.5-Omni-7B}"
: "${SFT_OUTPUT_DIR:=$CKPT_ROOT/sft}"
: "${SFT_MERGED_DIR:=$CKPT_ROOT/sft-merged}"
: "${RL_OUTPUT_DIR:=$CKPT_ROOT/rl}"

# Hyper-parameters
: "${NPROC_PER_NODE:=8}"
: "${LORA_RANK:=64}"
: "${LORA_ALPHA:=128}"

: "${SFT_LR:=1e-4}"
: "${SFT_EPOCHS:=5}"
: "${SFT_BATCH_SIZE:=4}"
: "${SFT_MAX_LENGTH:=8192}"

: "${RL_LR:=5e-6}"
: "${RL_EPOCHS:=3}"
: "${RL_BATCH_SIZE:=16}"
: "${RL_NUM_GENERATIONS:=8}"
: "${RL_TEMPERATURE:=1.2}"
: "${RL_BETA:=0.04}"
# MOS gap on the already-normalized 1-5 scale for preference labels.
: "${FIDELITY_TIE_THRESHOLD:=0.0}"
: "${RL_MAX_LENGTH:=2048}"
: "${RL_MAX_COMPLETION_LENGTH:=512}"

: "${DATASET_NUM_PROC:=4}"
: "${DATASET_SEED:=531}"
# Dataset rows one optimizer step consumes globally; keeps a batch single-dimension.
: "${PAIR_CHUNK_ROWS:=$((RL_BATCH_SIZE * NPROC_PER_NODE / RL_NUM_GENERATIONS))}"
# Read by the preference reward to split completions into GRPO groups.
: "${NUM_GENERATIONS:=$RL_NUM_GENERATIONS}"

if [ -n "${WANDB_API_KEY:-}" ]; then : "${REPORT_TO:=wandb}"; else : "${REPORT_TO:=none}"; fi

set +a
