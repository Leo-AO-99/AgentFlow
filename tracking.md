# AgentFlow Assignment — Model & API Tracking

This file records which model and which API/endpoint each step uses.

## Step 2 — Reproduce AGENTFLOW (w/o Flow-GRPO) on Qwen-2.5-7B-Instruct

| Part | Model | API / Endpoint | Notes |
|------|-------|----------------|-------|
| Planner (main) | `Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`https://api.portkey.ai/v1`, `PORTKEY_API_KEY`) | trainable component |
| Planner (fixed) | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | 贴近原文 dashscope 配置 |
| Verifier | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Executor | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Scorer / Judge | `gpt-4o-mini` | OpenAI API (`OPENAI_SECRET`) | answer extraction + comparison |

**Benchmarks:** Bamboogle, 2Wiki, HotpotQA, MuSiQue, GAIA, AIME24, AMC23, Game of 24, GPQA, MedQA
**Script:** `scripts/step2_reproduce/run_benchmarks.sh`

---

## Step 3 — Qwen3.5 Models on Paper Benchmarks

| Part | Model | API / Endpoint | Notes |
|------|-------|----------------|-------|
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-0.8B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-2B-Instruct` | same | |
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-4B-Instruct` | same | |
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-9B-Instruct` | same | |
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-27B-Instruct` | same | |
| Planner (fixed) | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Verifier | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Executor | `@deepinfra/Qwen/Qwen2.5-7B-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | |
| Scorer / Judge | `gpt-4o` | OpenAI API | |

**Benchmarks:** Same 10 as Step 2
**Scripts:** `scripts/step3_qwen35/serve_modal_vllm.py`, `scripts/step3_qwen35/run_benchmarks.sh`

---

## Step 4 — Code Generation New Benchmark (HumanEval + MBPP)

| Part | Model | API / Endpoint | Notes |
|------|-------|----------------|-------|
| Planner (main) | `@deepinfra/Qwen/Qwen3.5-{0.8B,2B,4B,9B,27B}-Instruct` | Portkey → DeepInfra (`PORTKEY_API_KEY`) | same as step 3 |
| Planner (fixed) | `gpt-4o-mini` | OpenAI API | |
| Verifier | `gpt-4o-mini` | OpenAI API | |
| Executor | `gpt-4o-mini` | OpenAI API | Python Coder tool |
| Scorer | — | Local execution | pass@1, deterministic — no LLM judge |

**Benchmarks:** HumanEval (164 problems), MBPP (374 problems)
**Scripts:** `scripts/step4_code_generation/`

---

## Step 5 — Flow-GRPO Training on Qwen3.5-0.8B (Paper Benchmarks)

| Part | Model | API / Endpoint | Notes |
|------|-------|----------------|-------|
| Training (actor) | `Qwen/Qwen3.5-0.8B-Instruct` + LoRA | Modal L40S | Flow-GRPO via verl |
| Rollout verifier | `gpt-4o-mini` | OpenAI API | during training rollout |
| Rollout executor | `gpt-4o-mini` | OpenAI API | during training rollout |
| Scorer / Judge | `gpt-4o` | OpenAI API | reward signal during training |
| Post-train eval | Fine-tuned Qwen3.5-0.8B | Modal vLLM or local | merged LoRA checkpoint |

**Modal GPU:** L40S (48 GB), timeout=600 (dev) → 7200 (final run)
**Scripts:** `scripts/step5_flow_grpo/`

---

## Step 6 — Flow-GRPO on Code Generation (Qwen3.5-0.8B)

| Part | Model | API / Endpoint | Notes |
|------|-------|----------------|-------|
| Training (actor) | `Qwen/Qwen3.5-0.8B-Instruct` + LoRA | Modal L40S | Flow-GRPO via verl |
| Rollout executor | `gpt-4o-mini` | OpenAI API | Python Coder tool during rollout |
| Scorer | — | Local execution | pass/fail on test cases — no LLM judge |
| Post-train eval | Fine-tuned Qwen3.5-0.8B | Modal vLLM or local | compared vs Step 4 baseline |

**Modal GPU:** L40S (48 GB), timeout=600 (dev) → 7200 (final run)
**Scripts:** `scripts/step6_flow_grpo_codegen/`

---

## API Keys Required

| Key | Used By | Where to Set |
|-----|---------|--------------|
| `OPENAI_API_KEY` | gpt-4o-mini scorer (steps 2–6) | already in env |
| `PORTKEY_API_KEY` | Portkey → DeepInfra inference (steps 2–4) | already in env; also exported as `VLLM_API_KEY` in each script |
| `HF_TOKEN` | HuggingFace model downloads | Modal secret `huggingface-secret` |
| `WANDB_API_KEY` | Training monitoring (steps 5–6) | Modal secret `wandb-secret` |
[ec2-user@ip-172-31-34-124 hedge_bot]$ python3.11 close_script.py --ccys SKL SKY SOMI SPK STEEM STX SYRUP T TAO THETA TIA TLM TNSE TON TRUMP TRX TUT UMA
SKY spot balance: 266.89259, future balance: 267
SKY close position: spot price(0.07357 * (1 - 0.0005))  > mark price(0.07345)
TLM spot balance: 12643.0, future balance: 12643
TLM close position: spot price(0.001625 * (1 - 0.0005))  > mark price(0.001624)
TRUMP spot balance: 5.99, future balance: 5.99
TRUMP close position: spot price(2.819 * (1 - 0.0005))  > mark price(2.809)
TRX spot balance: 26.6, future balance: 64
SKY close position success
TLM close position success
TRUMP close position success
STX spot balance: 81.971956, future balance: 82
STX close position: spot price(0.2139 * (1 - 0.0005))  > mark price(0.2137)
TAO spot balance: 8.689e-05, future balance: None
TON spot balance: 0.0, future balance: None
STX close position success
SYRUP spot balance: 95.986464, future balance: 96
SYRUP close position: spot price(0.2274 * (1 - 0.0005))  > mark price(0.22716)
SYRUP close position success
TIA spot balance: 67.0, future balance: 67
TIA close position: spot price(0.2959 * (1 - 0.0005))  > mark price(0.2956)
TIA close position success
TRX close position: spot price(0.3205 * (1 - 0.0005))  > mark price(0.32031)
TUT spot balance: 2321.0, future balance: 2321
TRX close position success
SOMI spot balance: 128.0, future balance: 128
SOMI close position: spot price(0.1587 * (1 - 0.0005))  > mark price(0.1585)
SOMI close position success
SPK spot balance: 963.67024, future balance: 964
SPK close position: spot price(0.020695 * (1 - 0.0005))  > mark price(0.02065)
SPK close position success
UMA spot balance: 49.0, future balance: 49
T spot balance: 3248.0, future balance: 3248
T close position: spot price(0.00609 * (1 - 0.0005))  > mark price(0.006072)
T close position success
THETA spot balance: 131.420023, future balance: 131.5
SKL spot balance: 3125.0, future balance: 3125
SKL close position: spot price(0.00627 * (1 - 0.0005))  > mark price(0.00625)
SKL close position success
TUT close position: spot price(0.00942 * (1 - 0.000495))  > mark price(0.00941)
TUT close position success
UMA close position: spot price(0.408 * (1 - 0.00048029800499999997))  > mark price(0.40778914)
UMA close position success
STEEM spot balance: 323.833806, future balance: 324
STEEM close position: spot price(0.05718 * (1 - 0.0005))  > mark price(0.05709)
STEEM close position success
THETA close position: spot price(0.168 * (1 - 0.0003850215729025775))  > mark price(0.16792673)
THETA close position success