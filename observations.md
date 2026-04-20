# AgentFlow — Experiment notes

## Step 2 — Qwen2.5-7B (w/o Flow-GRPO)

### High tool-selection failure rate (~50%)

**What we saw:**
```
No matched tool given: `Wikipedia_RAG_Search_Tool`   43.3%
No matched tool given: *Wikipedia_RAG_Search_Tool*    0.5%
No matched tool given: Wikipedia_RAG_Search_Tools     0.5%
No matched tool given: WebRagSearchTool               0.5%
... (roughly 50% total)
```

**Cause:** Qwen2.5-7B is weak at following the output format. It emits tool names with backticks, asterisks, camelCase, plurals, etc. `normalize_tool_name()` cannot match these, so the step is skipped.

**Expectation:** After Flow-GRPO (Step 5), tool-selection failures should drop sharply — a core claim of the paper.

---

## 2WikiMultiHopQA score much lower than the paper (8.0 vs 23.0)

**What we saw:** The other four benchmarks align with the paper; only 2wiki is ~15 points lower (full run).

**Root cause:** 2wiki items are almost all **comparison / multi-hop** (“who was born first between X and Y”, “are X and Y from the same country”), requiring precise retrieval of two entities’ attributes. Wikipedia RAG does poorly on fine-grained queries (specific birth dates, film origins). When the model cannot find information, it tends to guess and answers wrong.

**Typical failure:**
- Question: “Was Guillermo Matías Fernández or Chris Grundy born first?”
- Model finds Guillermo born 1991, cannot find Chris Grundy’s birth date, incorrectly infers Guillermo is older.
- Correct answer: Chris Grundy (born earlier).

**vs HotpotQA:** Also multi-hop, but HotpotQA answers are often reachable in one hop and are more forgiving.

**Expectation:** Flow-GRPO may improve search strategy, but the bottleneck is Wikipedia RAG quality; gains may be limited.

---

## Original bug — Executor judge engine wrongly routed to local vLLM

**Where:** `agentflow/agentflow/models/executor.py`

**What we saw:** Local vLLM logs during eval:
```
ERROR The model `@deepinfra/Qwen/Qwen2.5-7B-Instruct` does not exist → 404
```

**Root cause:** In `Executor.__init__` there was a branch:
```python
if base_url is not None:
    create_llm_engine(..., base_url=self.base_url, ...)
else:
    create_llm_engine(..., ...)  # base_url never passed
```
When `base_url=None` (correct for the judge engine), the `else` path runs; in `factory.py`, `kwargs.get("base_url", "http://localhost:8000/v1")` returns the default local URL because the key is missing. `ChatVLLM` then gets a non-`None` `base_url`, bypassing the `VLLM_BASE_URL` env fallback, so judge requests hit local vLLM.

Planner and Verifier do not have this bug (they always pass `base_url=None` explicitly), so Portkey traces for those are correct.

**Scope:** Only `llm_generate_tool_command` in Executor (tool-command generation). Planner and Verifier judge calls are fine. Most training data remains valid.

**Fix:** Merge branches and always pass `base_url` explicitly:
```python
self.llm_generate_tool_command = create_llm_engine(
    model_string=self.llm_engine_name, is_multimodal=False,
    base_url=self.base_url, temperature=self.temperature
)
```

**Related refactor:** `factory.py` adds a `portkey-` prefix path, hard-codes `https://api.portkey.ai/v1` and reads `PORTKEY_API_KEY`, instead of overloading the `vllm-` prefix for Portkey routing.
