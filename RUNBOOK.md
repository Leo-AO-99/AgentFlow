# AgentFlow Assignment — Runbook

## Environment setup

Confirm these environment variables are set:

```bash
echo $OPENAI_API_KEY     # OpenAI key (for scorer)
echo $PORTKEY_API_KEY    # Portkey key (for Qwen inference)
```

Confirm dependencies are installed:

```bash
uv sync                  # Install Python dependencies
which parallel           # GNU parallel (brew install parallel)
```

---

## Common parameters

All step inference scripts ultimately call `test/solve.py`. Core parameters:

| Parameter | Purpose | Typical value |
|-----------|---------|---------------|
| `--llm_engine_name` | Model for planner_main: `vllm-<model>` uses vLLM/Portkey; `gpt-4o-mini` uses OpenAI | `vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct` |
| `--base_url` | Inference endpoint URL, used with the `vllm-` prefix | `https://api.portkey.ai/v1` |
| `--model_engine` | Models for the four components: `planner_main,planner_fixed,verifier,executor`. `trainable` means same as `llm_engine_name` | `trainable,vllm-@deepinfra/...,vllm-@deepinfra/...,vllm-@deepinfra/...` |
| `--enabled_tools` | Enabled tools, comma-separated | `Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool` |
| `--tool_engine` | Inference engine per tool, aligned with `enabled_tools`; `Default` uses each tool’s default model | `Default,Default,Default,Default` |
| `--max_steps` | Max tool-calling steps for the agent | `10` |
| `--max_time` | Max wall time per problem (seconds); forces stop on timeout | `300` |
| `--temperature` | Sampling temperature; use `0.0` for eval | `0.0` |
| `--output_types` | Answer type: `base` (direct), `direct` (after tool chain), `final` (summarized after tool chain) | `direct` |
| `--index` | Run only problem index (0-based); omit for full set | `0` (single-problem debug) |
| `--output_json_dir` | Directory for result JSON files | `hotpotqa/results/Qwen2.5-7B-no-grpo` |

---

## Step 2 — Reproduce paper results (Qwen2.5-7B, no Flow-GRPO)

**Goal:** Reproduce the 10 benchmark scores for AGENTFLOW (w/o Flow-GRPO) (second-to-last row in the paper table).

**Model setup:**
- planner_main / planner_fixed / verifier / executor: all `Qwen2.5-7B-Instruct` (via Portkey → DeepInfra)
- scorer / judge: `gpt-4o-mini` (via OpenAI API)

**Run:**

```bash
bash scripts/step2_reproduce/run_benchmarks.sh
```

**Where results go:**

```
test/<benchmark>/results/Qwen2.5-7B-no-grpo/
├── output_0.json ~ output_N.json   # Full trace per problem
├── finalresults_direct_output.json  # Aggregated scores
└── finalscore_direct_output.log
```

**Single-problem debug (recommended to verify the pipeline first):**

```bash
cd test
python solve.py \
  --index 0 \
  --task hotpotqa \
  --data_file hotpotqa/data/data.json \
  --llm_engine_name "vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct" \
  --base_url "https://api.portkey.ai/v1" \
  --model_engine "trainable,vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct,vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct,vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct" \
  --enabled_tools "Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool" \
  --tool_engine "Default,Default,Default,Default" \
  --output_types direct \
  --max_steps 10 --max_time 300 --temperature 0.0 \
  --verbose True
```

**Parallelism:**
- All 10 benchmarks run in the background (`&` + `wait`)
- Within each benchmark, `parallel -j 20` runs 20 problems concurrently

---

## Step 3 — Paper benchmarks with Qwen3.5 family

**Goal:** Run the same 10 benchmarks on five Qwen3.5 sizes (0.8B / 2B / 4B / 9B / 27B) and compare with Step 2.

**Model setup:**
- planner_main: the Qwen3.5 model under test (via Portkey → DeepInfra)
- planner_fixed / verifier / executor: fixed to `Qwen2.5-7B-Instruct`

**Run:**

```bash
bash scripts/step3_qwen35/run_benchmarks.sh
```

**Where results go:**

```
test/<benchmark>/results/Qwen3.5-0.8B/
test/<benchmark>/results/Qwen3.5-2B/
test/<benchmark>/results/Qwen3.5-4B/
...
```

**Parallelism:**
- The five models run **sequentially** (to avoid excessive API concurrency)
- For each model, all 10 benchmarks run **in parallel**

> **Note:** Before a full run, confirm the exact Qwen3.5 model names on DeepInfra. You can test names by editing `pk.py` and running:
> ```bash
> # Set model in pk.py to @deepinfra/Qwen/Qwen3.5-0.8B-Instruct, then run
> python pk.py
> ```

---

## Step 4 — New benchmark: code generation (HumanEval + MBPP)

### 4.1 Prepare data (once)

```bash
python scripts/step4_code_generation/data/prepare_data.py
```

This produces:
- `scripts/step4_code_generation/data/humaneval_data.json` (164 problems)
- `scripts/step4_code_generation/data/mbpp_data.json` (374 problems)

### 4.2 Run eval

```bash
bash scripts/step4_code_generation/run_benchmarks.sh
```

**Where results go:**

```
scripts/step4_code_generation/results/humaneval/Qwen3.5-0.8B/scores.json
scripts/step4_code_generation/results/mbpp/Qwen3.5-0.8B/scores.json
...
```

**Scoring:** Execute generated code against each problem’s assert tests; all tests passing counts as pass@1 = 1. **No LLM judge** — fully deterministic.

**Single-problem debug:**

```bash
cd test
python ../scripts/step4_code_generation/solve.py \
  --index 0 \
  --data_file ../scripts/step4_code_generation/data/humaneval_data.json \
  --llm_engine_name "vllm-@deepinfra/Qwen/Qwen3.5-0.8B-Instruct" \
  --base_url "https://api.portkey.ai/v1" \
  --max_steps 5 --max_time 120 --temperature 0.0
```

---

## Step 5 — Flow-GRPO training (Qwen3.5-0.8B, paper benchmarks)

### 5.1 Upload code to Modal volume

Training data is downloaded and built inside the container — no local data prep required.

Upload code only (from repo root; path has no `/runs/` prefix):

```bash
modal volume put ao-runs . /step5/code
```

Paths inside the container:
- Code: `/runs/step5/code/`
- Data: auto-generated and cached under `/runs/step5/data/` (reused after first run)

### 5.2 Dev run (10 minutes to confirm no crash/OOM)

In `scripts/step5_flow_grpo/modal_train.py`, `timeout=600` (10 minutes).

```bash
modal run scripts/step5_flow_grpo/modal_train.py --dev
```

`--dev`: runs only 1 epoch and reduces batch size to 4 for a quick end-to-end smoke test.

### 5.3 Full training

After dev succeeds, **change `timeout=600` to `timeout=7200`** in `modal_train.py`, then:

```bash
modal run scripts/step5_flow_grpo/modal_train.py
```

Training metrics: W&B (`PROJECT_NAME: AgentFlow_Step5`).

**Key training settings (`config_0.8b.yaml`):**

| Parameter | Value | Notes |
|-----------|-------|--------|
| `BASE_MODEL` | `Qwen/Qwen3.5-0.8B-Instruct` | Base model |
| `lora_rank` | 64 | LoRA rank (larger → more parameters) |
| `lora_alpha` | 128 | LoRA scale (often 2× rank) |
| `train_batch_size` | 8 | Problems per training step |
| `rollout.n` | 4 | Samples per problem for GRPO comparison |
| `total_epochs` | 5 | Total epochs |
| `adv_estimator` | grpo | GRPO (Flow-GRPO) |
| `gpu_memory_utilization` | 0.5 | GPU memory fraction for vLLM rollout |
| `kl_loss_coef` | 0.001 | KL regularization to limit policy drift |

### 5.4 Evaluate the trained model

```bash
# 1. Download checkpoint
modal volume get ao-runs /runs/step5/checkpoints/ ./step5_checkpoints/

# 2. Merge LoRA weights
python util/model_merger.py --checkpoint_dir ./step5_checkpoints/latest

# 3. Start vLLM locally
vllm serve ./merged_model --port 8000

# 4. Run eval (compare with step 2)
export CHECKPOINT_PATH="./merged_model"
export VLLM_BASE_URL="http://localhost:8000/v1"
bash scripts/step5_flow_grpo/run_eval.sh
```

---

## Step 6 — Flow-GRPO training (Qwen3.5-0.8B, code generation)

### 6.1 Prepare training data

```bash
# Build MBPP parquet training data
python scripts/step6_flow_grpo_codegen/prepare_training_data.py --output_dir ./step6_data

# Upload to Modal
modal volume put ao-runs ./step6_data/ /runs/step6/data/
modal volume put ao-runs . /runs/step6/code/
```

### 6.2 Dev run

```bash
modal run scripts/step6_flow_grpo_codegen/modal_train.py --dev
```

### 6.3 Full training

Same as Step 5: set `timeout=7200`, then:

```bash
modal run scripts/step6_flow_grpo_codegen/modal_train.py
```

**Differences from Step 5:**
- Training data: MBPP (374 problems) instead of search/math data
- Reward: code execution → pass/fail (no LLM judge; cleaner signal)
- Tools: mainly `Python_Coder_Tool`

### 6.4 Evaluate and compare with Step 4

```bash
export CHECKPOINT_PATH="./step6_merged_model"
export VLLM_BASE_URL="http://localhost:8000/v1"
bash scripts/step6_flow_grpo_codegen/run_eval.sh
```

The script prints pass@1 before/after Flow-GRPO.

---

## Quick score check

All benchmark scores are in `finalresults_direct_output.json`:

```bash
# One benchmark
cat test/hotpotqa/results/Qwen2.5-7B-no-grpo/finalresults_direct_output.json

# Summarize all step2 benchmarks
for b in bamboogle 2wiki hotpotqa musique gaia; do
  score=$(python3 -c "
import json, os
f = 'test/$b/results/Qwen2.5-7B-no-grpo/finalresults_direct_output.json'
if os.path.exists(f):
    d = json.load(open(f))
    print(d.get('accuracy', d.get('score', '?')))
else:
    print('not done')
")
  echo "$b: $score"
done
```

---

## FAQ

**Q: A problem seems stuck?**  
Scripts support resume: problems that already have `output_N.json` are skipped; re-run the same command.

**Q: Portkey returns 404 / model not found?**  
Use `pk.py` to verify the model exists on DeepInfra; update `@deepinfra/...` names in scripts.

**Q: Modal training OOM?**  
Lower `data.train_batch_size` from 8 to 4 and `rollout.n` from 4 to 2 in `config_0.8b.yaml`.

**Q: No data in W&B?**  
Ensure Modal secret `wandb-secret` includes `WANDB_API_KEY`.
