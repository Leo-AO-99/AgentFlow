#!/bin/bash
# Step 3: Run paper benchmarks with all 5 Qwen3.5 models via Portkey (DeepInfra).
#
# Requires: PORTKEY_API_KEY and OPENAI_API_KEY set in environment.
#
# Usage:
#   bash scripts/step3_qwen35/run_benchmarks.sh

set -euo pipefail

# ── API keys ───────────────────────────────────────────────────────────────────
export VLLM_API_KEY="${PORTKEY_API_KEY:?PORTKEY_API_KEY is not set}"
PORTKEY_URL="https://api.portkey.ai/v1"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_DIR/test"

# ── Models: all served via Portkey → DeepInfra ────────────────────────────────
# All 5 models share the same Portkey base URL; model name selects the backend.
declare -A MODEL_URLS=(
    ["@deepinfra/Qwen/Qwen3.5-0.8B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-2B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-4B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-9B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-27B-Instruct"]="$PORTKEY_URL"
)

# ── Benchmark config ───────────────────────────────────────────────────────────
BENCHMARKS=(bamboogle 2wiki hotpotqa musique gaia)
# fixed components 用 Qwen2.5-7B（贴近原文），planner_main 仍是被评测的 Qwen3.5 模型
FIXED_ENGINE="vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct"
MODEL_ENGINE="trainable,${FIXED_ENGINE},${FIXED_ENGINE},${FIXED_ENGINE}"
ENABLED_TOOLS="Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="Default,Default,Default,Default"
THREADS=6
SAMPLE_PCT=10      # 每个 benchmark 随机采样百分比（1-100，100=全量）
MAX_STEPS=10
MAX_TIME=300
TEMPERATURE=0.0
OUTPUT_TYPES="direct"

cd "$TEST_DIR"

run_benchmark() {
    local TASK="$1"
    local LLM_ENGINE="$2"
    local BASE_URL="$3"
    local LABEL="$4"

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

    echo "[$TASK/$LABEL] Running ${#NEW_INDICES[@]}/$N_SAMPLE sampled (${SAMPLE_PCT}% of $N_PROBLEMS) problems..."

    _solve_one() {
        local idx="$1"
        python solve.py \
            --index "$idx" \
            --task "$TASK" \
            --data_file "$DATA_FILE" \
            --llm_engine_name "$LLM_ENGINE" \
            --base_url "$BASE_URL" \
            --model_engine "$MODEL_ENGINE" \
            --enabled_tools "$ENABLED_TOOLS" \
            --tool_engine "$TOOL_ENGINE" \
            --output_types "$OUTPUT_TYPES" \
            --output_json_dir "$OUT_DIR" \
            --root_cache_dir "$CACHE_DIR" \
            --max_steps "$MAX_STEPS" \
            --max_time "$MAX_TIME" \
            --temperature "$TEMPERATURE" \
            2>&1 | tee "$LOG_DIR/$idx.log"
    }
    export -f _solve_one
    export LLM_ENGINE BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE OUTPUT_TYPES
    export OUT_DIR CACHE_DIR TASK DATA_FILE LOG_DIR MAX_STEPS MAX_TIME TEMPERATURE

    parallel -j "$THREADS" _solve_one ::: "${NEW_INDICES[@]}"

    echo "[$TASK/$LABEL] Scoring..."
    python calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --response_type "direct_output" \
        --output_file "finalresults_direct_output.json" \
        | tee "$OUT_DIR/finalscore_direct_output.log"
}

export -f run_benchmark
export MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE THREADS SAMPLE_PCT MAX_STEPS MAX_TIME TEMPERATURE OUTPUT_TYPES TEST_DIR

# Run each model sequentially (to avoid overlapping GPU/API costs),
# but run all benchmarks for that model in parallel.
for MODEL_NAME in "${!MODEL_URLS[@]}"; do
    BASE_URL="${MODEL_URLS[$MODEL_NAME]}"
    if [ -z "$BASE_URL" ]; then
        echo "WARN: No base URL for $MODEL_NAME — skipping. Set QWEN_*_URL or PROF_BASE_URL."
        continue
    fi

    # Derive a short label, e.g. "Qwen3.5-0.8B"
    LABEL=$(echo "$MODEL_NAME" | sed 's|@deepinfra/Qwen/||; s|-Instruct||')
    LLM_ENGINE="vllm-${MODEL_NAME}"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_NAME  →  label: $LABEL"
    echo "Endpoint: $BASE_URL"
    echo "========================================"

    PIDS=()
    for TASK in "${BENCHMARKS[@]}"; do
        run_benchmark "$TASK" "$LLM_ENGINE" "$BASE_URL" "$LABEL" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"

    echo "[$LABEL] All benchmarks complete."
done

echo ""
echo "====== Step 3 complete ======"
echo "Results in: $TEST_DIR/<benchmark>/results/Qwen3.5-*/"
