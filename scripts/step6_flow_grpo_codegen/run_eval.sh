#!/bin/bash
# Step 6: Evaluate the Flow-GRPO trained Qwen3.5-0.8B checkpoint on code generation.
#
# Compares pass@1 vs. the Step 4 baseline (Qwen3.5-0.8B without Flow-GRPO).
#
# Usage:
#   export CHECKPOINT_PATH="/path/to/merged/checkpoint"
#   export VLLM_BASE_URL="http://localhost:8000/v1"
#   (OPENAI_API_KEY and PORTKEY_API_KEY must be set in environment)
#   bash scripts/step6_flow_grpo_codegen/run_eval.sh

set -euo pipefail

export VLLM_API_KEY="${PORTKEY_API_KEY:?PORTKEY_API_KEY is not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
STEP4_DIR="$PROJECT_DIR/scripts/step4_code_generation"
TEST_DIR="$PROJECT_DIR/test"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
LABEL="Qwen3.5-0.8B-FlowGRPO-CodeGen"

BENCHMARKS=(
    "humaneval:$STEP4_DIR/data/humaneval_data.json"
    "mbpp:$STEP4_DIR/data/mbpp_data.json"
)

MODEL_ENGINE="trainable,gpt-4o-mini,gpt-4o-mini,gpt-4o-mini"
ENABLED_TOOLS="Python_Coder_Tool,Base_Generator_Tool"
TOOL_ENGINE="Default,Default"
THREADS=6
SAMPLE_PCT=20      # 每个 benchmark 随机采样百分比（1-100，100=全量）
MAX_STEPS=5
MAX_TIME=120
TEMPERATURE=0.0
TASK_DESCRIPTION="Write a Python function to solve the following programming problem. Your answer must be valid, self-contained Python code."

if [ -z "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Set CHECKPOINT_PATH to the merged LoRA checkpoint directory."
    echo "  1. Download: modal volume get ao-runs /runs/step6/checkpoints/ ./step6_checkpoints/"
    echo "  2. Merge:    python util/model_merger.py --checkpoint_dir ./step6_checkpoints/latest"
    echo "  3. Serve:    vllm serve \$CHECKPOINT_PATH --port 8000"
    exit 1
fi

LLM_ENGINE="vllm-${CHECKPOINT_PATH}"

run_codegen_eval() {
    local BENCH_NAME="$1"
    local DATA_FILE="$2"

    local OUT_DIR="$STEP4_DIR/results/$BENCH_NAME/$LABEL"
    local LOG_DIR="$STEP4_DIR/logs/$BENCH_NAME/$LABEL"
    local CACHE_DIR="$STEP4_DIR/cache/$BENCH_NAME"
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

    if [ "${#NEW_INDICES[@]}" -gt 0 ]; then
        echo "[$BENCH_NAME/$LABEL] Running ${#NEW_INDICES[@]}/$N_SAMPLE sampled (${SAMPLE_PCT}% of $N_PROBLEMS) problems..."

        _solve_one() {
            local idx="$1"
            cd "$TEST_DIR"
            python "$STEP4_DIR/solve.py" \
                --index "$idx" \
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
                --task_description "$TASK_DESCRIPTION" \
                2>&1 | tee "$LOG_DIR/$idx.log"
        }
        export -f _solve_one
        export LLM_ENGINE VLLM_BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE
        export OUT_DIR CACHE_DIR DATA_FILE LOG_DIR MAX_STEPS MAX_TIME TEMPERATURE
        export TASK_DESCRIPTION STEP4_DIR TEST_DIR

        parallel -j "$THREADS" _solve_one ::: "${NEW_INDICES[@]}"
    fi

    echo "[$BENCH_NAME/$LABEL] Evaluating..."
    python3 "$STEP4_DIR/evaluate.py" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --output_file "$SCORE_FILE"

    local SCORE
    SCORE=$(python3 -c "import json; d=json.load(open('$SCORE_FILE')); print(f\"pass@1={d['pass@1']}%\")")
    echo "[$BENCH_NAME/$LABEL] $SCORE"

    # Print comparison with step 4 baseline
    BASELINE_SCORE_FILE="$STEP4_DIR/results/$BENCH_NAME/Qwen3.5-0.8B/scores.json"
    if [ -f "$BASELINE_SCORE_FILE" ]; then
        BASELINE=$(python3 -c "import json; d=json.load(open('$BASELINE_SCORE_FILE')); print(f\"pass@1={d['pass@1']}%\")")
        echo "[$BENCH_NAME] Baseline (no Flow-GRPO): $BASELINE"
        echo "[$BENCH_NAME] After Flow-GRPO:         $SCORE"
    fi
}

export -f run_codegen_eval
export LABEL LLM_ENGINE VLLM_BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE
export THREADS SAMPLE_PCT MAX_STEPS MAX_TIME TEMPERATURE TASK_DESCRIPTION
export STEP4_DIR TEST_DIR

PIDS=()
for BENCH_SPEC in "${BENCHMARKS[@]}"; do
    BENCH_NAME="${BENCH_SPEC%%:*}"
    DATA_FILE="${BENCH_SPEC##*:}"
    run_codegen_eval "$BENCH_NAME" "$DATA_FILE" &
    PIDS+=($!)
done
wait "${PIDS[@]}"

echo ""
echo "====== Step 6 evaluation complete ======"
echo "Results in: $STEP4_DIR/results/*/$LABEL/"
