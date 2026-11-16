#!/bin/bash
# Merge a LoRA adapter into its base model.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

ADAPTER="${1:?usage: merge_lora.sh <adapter_dir> [output_dir] [base_model]}"
OUTPUT_DIR="${2:-$SFT_MERGED_DIR}"
BASE="${3:-$BASE_MODEL}"

swift export \
    --model "$BASE" \
    --adapters "$ADAPTER" \
    --merge_lora true \
    --output_dir "$OUTPUT_DIR"
