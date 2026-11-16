#!/bin/bash
# Run every benchmark and print the LCC / SRCC table
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/config.sh"

MODEL="${MODEL:-AutoAudioReasoningMOS/CoTTAGE}"
RUN_NAME="${RUN_NAME:-cottage}"
BENCHMARKS="${BENCHMARKS:-xacle relate pam clotho}"

for benchmark in $BENCHMARKS; do
    "$PYTHON" "$REPO_ROOT/cottage/run_inference.py" \
        --benchmark "$benchmark" \
        --model "$MODEL" \
        ${ADAPTER:+--adapter "$ADAPTER"} \
        --run_name "$RUN_NAME" \
        --output_dir "$RESULTS_ROOT" \
        --max_lora_rank "$LORA_RANK"
done

"$PYTHON" "$REPO_ROOT/cottage/compute_correlation.py" "$RESULTS_ROOT"/*/"${RUN_NAME}"_*.json
