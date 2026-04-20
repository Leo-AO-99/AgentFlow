#!/bin/bash
# Step 2: Reproduce AgentFlow (w/o Flow-GRPO) results on Qwen-2.5-7B-Instruct
#
# Uses Portkey (DeepInfra backend) for the Qwen planner model.
# Requires: PORTKEY_API_KEY and OPENAI_API_KEY set in environment.
#
# Usage:
#   bash scripts/step2_reproduce/run_benchmarks.sh
#
# Benchmarks run in parallel (background jobs).
# Within each benchmark, problems run with GNU parallel -j THREADS.

set -euo pipefail

# ── API keys ───────────────────────────────────────────────────────────────────
export VLLM_API_KEY="${PORTKEY_API_KEY:?PORTKEY_API_KEY is not set}"
BASE_URL="https://api.portkey.ai/v1"
# @deepinfra/ prefix routes through DeepInfra on Portkey
LLM_ENGINE="vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct"

# ── Configuration ──────────────────────────────────────────────────────────────
# planner_fixed / verifier / executor 也走 Qwen2.5-7B（贴近原文的 dashscope 配置）
# 因为引擎名与 LLM_ENGINE 相同，base_url 会自动传入，走 Portkey → DeepInfra
MODEL_ENGINE="trainable,${LLM_ENGINE},${LLM_ENGINE},${LLM_ENGINE}"
ENABLED_TOOLS="Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
TOOL_ENGINE="Default,Default,Default,Default"
THREADS=6          # 并行题数，降低以减少 CPU/API 压力
SAMPLE_PCT=100      # 每个 benchmark 随机采样百分比（1-100，100=全量）
MAX_STEPS=10
MAX_TIME=300
TEMPERATURE=0.0
OUTPUT_TYPES="direct"
LABEL="Qwen2.5-7B-no-grpo"

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_DIR="$PROJECT_DIR/test"

# ── Benchmark list ─────────────────────────────────────────────────────────────
BENCHMARKS=(bamboogle 2wiki hotpotqa musique gaia)

cd "$TEST_DIR"

run_benchmark() {
    local TASK="$1"
    local DATA_FILE="$TASK/data/data.json"
    local LOG_DIR="$TASK/logs/$LABEL"
    local OUT_DIR="$TASK/results/$LABEL"
    local CACHE_DIR="$TASK/cache"

    mkdir -p "$LOG_DIR" "$OUT_DIR"

    # Determine number of problems, then sample
    local N_PROBLEMS
    N_PROBLEMS=$(python3 -c "import json; d=json.load(open('$DATA_FILE')); print(len(d))")
    local N_SAMPLE=$(( (N_PROBLEMS * SAMPLE_PCT + 99) / 100 ))
    local INDICES=($(python3 -c "
import random, json
random.seed(42)
n = $N_PROBLEMS
k = $N_SAMPLE
idx = random.sample(range(n), min(k, n))
idx.sort()
print(' '.join(map(str, idx)))
"))

    # Skip already completed problems
    local NEW_INDICES=()
    for i in "${INDICES[@]}"; do
        [ ! -f "$OUT_DIR/output_$i.json" ] && NEW_INDICES+=("$i")
    done

    if [ "${#NEW_INDICES[@]}" -eq 0 ]; then
        echo "[$TASK] All $N_PROBLEMS problems already completed. Skipping."
    else
        echo "[$TASK] Running ${#NEW_INDICES[@]} / $N_SAMPLE sampled (${SAMPLE_PCT}% of $N_PROBLEMS) problems (label=$LABEL)..."

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
    fi

    echo "[$TASK] Scoring..."
    python calculate_score_unified.py \
        --task_name "$TASK" \
        --data_file "$DATA_FILE" \
        --result_dir "$OUT_DIR" \
        --response_type "direct_output" \
        --output_file "finalresults_direct_output.json" \
        | tee "$OUT_DIR/finalscore_direct_output.log"

    echo "[$TASK] Done."
}

export -f run_benchmark
export LABEL LLM_ENGINE BASE_URL MODEL_ENGINE ENABLED_TOOLS TOOL_ENGINE
export THREADS SAMPLE_PCT MAX_STEPS MAX_TIME TEMPERATURE OUTPUT_TYPES TEST_DIR

# Run all benchmarks in parallel (background)
PIDS=()
for TASK in "${BENCHMARKS[@]}"; do
    run_benchmark "$TASK" &
    PIDS+=($!)
done

echo "Running all ${#BENCHMARKS[@]} benchmarks in parallel (PIDs: ${PIDS[*]})..."
wait

echo ""
echo "====== All benchmarks complete ======"
echo "Results in: $TEST_DIR/<benchmark>/results/$LABEL/"
