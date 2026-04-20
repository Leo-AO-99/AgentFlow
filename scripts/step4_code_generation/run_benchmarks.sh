#!/bin/bash
# Step 4: Run code generation benchmarks (HumanEval + MBPP) with all 5 Qwen3.5 models.
#
# Usage:
#   export QWEN_0_8B_URL="http://<endpoint>/v1"   # or set PROF_BASE_URL for all
#   export QWEN_2B_URL="..."
#   export QWEN_4B_URL="..."
#   export QWEN_9B_URL="..."
#   export QWEN_27B_URL="..."
#   export OPENAI_API_KEY="sk-..."
#   bash scripts/step4_code_generation/run_benchmarks.sh
#
# First run:
#   python scripts/step4_code_generation/data/prepare_data.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_DIR/test"
DATA_DIR="$SCRIPT_DIR/data"

# ── Check data files exist ─────────────────────────────────────────────────────
for f in humaneval_data.json mbpp_data.json; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "Data file $DATA_DIR/$f not found. Run prepare_data.py first:"
        echo "  python $SCRIPT_DIR/data/prepare_data.py"
        exit 1
    fi
done

# ── API keys ───────────────────────────────────────────────────────────────────
export VLLM_API_KEY="${PORTKEY_API_KEY:?PORTKEY_API_KEY is not set}"
PORTKEY_URL="https://api.portkey.ai/v1"

# ── Model → base URL map ───────────────────────────────────────────────────────
declare -A MODEL_URLS=(
    ["@deepinfra/Qwen/Qwen3.5-0.8B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-2B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-4B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-9B-Instruct"]="$PORTKEY_URL"
    ["@deepinfra/Qwen/Qwen3.5-27B-Instruct"]="$PORTKEY_URL"
)

BENCHMARKS=(
    "humaneval:$DATA_DIR/humaneval_data.json"
    "mbpp:$DATA_DIR/mbpp_data.json"
)

FIXED_ENGINE="vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct"
MODEL_ENGINE="trainable,${FIXED_ENGINE},${FIXED_ENGINE},${FIXED_ENGINE}"
ENABLED_TOOLS="Python_Coder_Tool,Base_Generator_Tool"
TOOL_ENGINE="Default,Default"
THREADS=6
SAMPLE_PCT=20      # 每个 benchmark 随机采样百分比（1-100，100=全量）
MAX_STEPS=5
MAX_TIME=120
TEMPERATURE=0.0
OUTPUT_TYPES="direct"
TASK_DESCRIPTION="Write a Python function to solve the following programming problem. Your answer must be valid, self-contained Python code."

# ── Inner solver function ──────────────────────────────────────────────────────
run_codegen_benchmark() {
    local BENCH_NAME="$1"
    local DATA_FILE="$2"
    local LLM_ENGINE="$3"
    local BASE_URL="$4"
    local LABEL="$5"

    local OUT_DIR="$SCRIPT_DIR/results/$BENCH_NAME/$LABEL"
    local LOG_DIR="$SCRIPT_DIR/logs/$BENCH_NAME/$LABEL"
    local CACHE_DIR="$SCRIPT_DIR/cache/$BENCH_NAME"
    local SCORE_FILE="$OUT_DIR/scores.json"

    mkdir -p "$OUT_DIR" "$LOG_DIR" "$CACHE_DIR"

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
        echo "[$BENCH_NAME/$LABEL] All done. Running scorer only."
    else
        echo "[$BENCH_NAME/$LABEL] Running ${#NEW_INDICES[@]}/$N_SAMPLE sampled (${SAMPLE_PCT}% of $N_PROBLEMS) problems..."

        _solve_one() {
            local idx="$1"
            # solve.py is in this step's directory; test dir for imports
            cd "$TEST_DIR"
            python "$SCRIPT_DIR/solve.py" \
                --index "$idx" \
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
                --task_description "$TASK_DESCRIPTION" \
                2>&1 | tee "$LOG_DIR/$idx.log"
        }
        export -f _solve_one
        export LLM_ENGINE BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE OUTPUT_TYPES
        export OUT_DIR CACHE_DIR DATA_FILE LOG_DIR MAX_STEPS MAX_TIME TEMPERATURE
        export TASK_DESCRIPTION SCRIPT_DIR TEST_DIR

        parallel -j "$THREADS" _solve_one ::: "${NEW_INDICES[@]}"
    fi

    echo "[$BENCH_NAME/$LABEL] Evaluating..."
    python3 "$SCRIPT_DIR/evaluate.py" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --output_file "$SCORE_FILE" \
        --workers 8

    echo "[$BENCH_NAME/$LABEL] Done. Score: $(python3 -c "import json; d=json.load(open('$SCORE_FILE')); print(f\"pass@1={d['pass@1']}%\")")"
}

export -f run_codegen_benchmark
export MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE THREADS SAMPLE_PCT MAX_STEPS MAX_TIME TEMPERATURE OUTPUT_TYPES
export TASK_DESCRIPTION SCRIPT_DIR TEST_DIR DATA_DIR PROJECT_DIR

# ── Main loop: models sequentially, benchmarks in parallel ────────────────────
for MODEL_NAME in "${!MODEL_URLS[@]}"; do
    BASE_URL="${MODEL_URLS[$MODEL_NAME]}"
    if [ -z "$BASE_URL" ]; then
        echo "WARN: No base URL for $MODEL_NAME. Skipping."
        continue
    fi

    LABEL=$(echo "$MODEL_NAME" | sed 's|@deepinfra/Qwen/||; s|-Instruct||')
    LLM_ENGINE="vllm-${MODEL_NAME}"

    echo ""
    echo "========================================"
    echo "Model: $MODEL_NAME  →  $LABEL"
    echo "========================================"

    PIDS=()
    for BENCH_SPEC in "${BENCHMARKS[@]}"; do
        BENCH_NAME="${BENCH_SPEC%%:*}"
        DATA_FILE="${BENCH_SPEC##*:}"
        run_codegen_benchmark "$BENCH_NAME" "$DATA_FILE" "$LLM_ENGINE" "$BASE_URL" "$LABEL" &
        PIDS+=($!)
    done
    wait "${PIDS[@]}"

    echo "[$LABEL] Both benchmarks done."
done

echo ""
echo "====== Step 4 complete ======"
echo "Results in: $SCRIPT_DIR/results/"
