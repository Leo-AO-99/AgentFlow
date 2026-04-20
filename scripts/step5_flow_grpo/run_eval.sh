#!/bin/bash
# Step 5: Evaluate the Flow-GRPO trained Qwen3.5-0.8B checkpoint on paper benchmarks.
#
# After training completes, download and merge the LoRA checkpoint, then run
# all 10 paper benchmarks (same as step 2) to compare with the baseline.
#
# Usage:
#   export CHECKPOINT_PATH="/path/to/merged/checkpoint"   # local path after download
#   export VLLM_BASE_URL="http://localhost:8000/v1"       # start vLLM manually first
#   (OPENAI_API_KEY and PORTKEY_API_KEY must be set in environment)
#   bash scripts/step5_flow_grpo/run_eval.sh

set -euo pipefail

export VLLM_API_KEY="${PORTKEY_API_KEY:?PORTKEY_API_KEY is not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_DIR/test"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
LABEL="Qwen3.5-0.8B-FlowGRPO"

BENCHMARKS=(bamboogle 2wiki hotpotqa musique gaia)
MODEL_ENGINE="trainable,gpt-4o-mini,gpt-4o-mini,gpt-4o-mini"
ENABLED_TOOLS="Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="Default,Default,Default,Default"
THREADS=6
SAMPLE_PCT=20      # 每个 benchmark 随机采样百分比（1-100，100=全量）
MAX_STEPS=10
MAX_TIME=300
TEMPERATURE=0.0

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Set CHECKPOINT_PATH to the merged LoRA checkpoint directory."
    echo "  1. Download from Modal: modal volume get ao-runs /runs/step5/checkpoints/ ./checkpoints/"
    echo "  2. Merge LoRA: python util/model_merger.py --checkpoint_dir ./checkpoints/latest"
    echo "  3. Serve: vllm serve \$CHECKPOINT_PATH --port 8000"
    echo "  4. Re-run this script."
    exit 1
fi

# Derive vllm engine name from the checkpoint path
LLM_ENGINE="vllm-${CHECKPOINT_PATH}"

cd "$TEST_DIR"

run_benchmark() {
    local TASK="$1"
    local DATA_FILE="$TASK/data/data.json"
    local LOG_DIR="$TASK/logs/$LABEL"
    local OUT_DIR="$TASK/results/$LABEL"
    local CACHE_DIR="$TASK/cache"

    mkdir -p "$LOG_DIR" "$OUT_DIR"

    local N_PROBLEMS
    N_PROBLEMS=$(python3 -c "import json; d=json.load(open('$DATA_FILE')); print(len(d))")
    local N_SAMPLE=$(( (N_PROBLEMS * SAMPLE_PCT + 99) / 100 ))
    local INDICES=($(python3 -c "
import random
random.seed(42)
n = $N_PROBLEMS
k = $N_SAMPLE
idx = random.sample(range(n), min(k, n))
idx.sort()
print(' '.join(map(str, idx)))
"))

    local NEW_INDICES=()
    for i in "${INDICES[@]}"; do
        [ ! -f "$OUT_DIR/output_$i.json" ] && NEW_INDICES+=("$i")
    done

    if [ "${#NEW_INDICES[@]}" -eq 0 ]; then
        echo "[$TASK/$LABEL] All done. Skipping."
        return
    fi

    echo "[$TASK/$LABEL] Evaluating ${#NEW_INDICES[@]}/$N_SAMPLE sampled (${SAMPLE_PCT}% of $N_PROBLEMS) problems..."

    _solve_one() {
        local idx="$1"
        python solve.py \
            --index "$idx" \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "$LLM_ENGINE" \
            --base_url "$VLLM_BASE_URL" \
            --model_engine "$MODEL_ENGINE" \
            --enabled_tools "$ENABLED_TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --output_types "direct" \
            --output_json_dir "$OUT_DIR" \
            --root_cache_dir "$CACHE_DIR" \
            --max_steps "$MAX_STEPS" \
            --max_time "$MAX_TIME" \
            --temperature "$TEMPERATURE" \
            2>&1 | tee "$LOG_DIR/$idx.log"
    }
    export -f _solve_one
    export LLM_ENGINE VLLM_BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE
    export OUT_DIR CACHE_DIR TASK DATA_FILE LOG_DIR MAX_STEPS MAX_TIME TEMPERATURE

    parallel -j "$THREADS" _solve_one ::: "${NEW_INDICES[@]}"

    python calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --response_type "direct_output" \
        --output_file "finalresults_direct_output.json" \
        | tee "$OUT_DIR/finalscore_direct_output.log"
}

export -f run_benchmark
export LABEL LLM_ENGINE VLLM_BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE
export THREADS SAMPLE_PCT MAX_STEPS MAX_TIME TEMPERATURE TEST_DIR

PIDS=()
for TASK in "${BENCHMARKS[@]}"; do
    run_benchmark "$TASK" &
    PIDS+=($!)
done
wait "${PIDS[@]}"

echo ""
echo "====== Step 5 evaluation complete ======"
echo "Results in: $TEST_DIR/<benchmark>/results/$LABEL/"
echo ""
echo "Compare with step 2 baseline (Qwen2.5-7B-no-grpo) and step 3 (Qwen3.5-0.8B baseline)."
