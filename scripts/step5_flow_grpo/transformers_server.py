"""
Minimal OpenAI-compatible chat-completions server backed by HuggingFace transformers.

Endpoints:
  GET  /health
  POST /v1/chat/completions   (OpenAI format, text-only)

Run:
  python transformers_server.py --model_dir /path/to/model --port 8000
"""

import argparse
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# ── global state loaded before uvicorn starts ────────────────────────────────
_model     = None
_tokenizer = None
_model_dir = None


def _load(model_dir: str) -> None:
    global _model, _tokenizer, _model_dir
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[Server] Loading tokenizer from {model_dir} …", flush=True)
    _tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    print(f"[Server] Loading model from {model_dir} …", flush=True)
    _model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()
    _model_dir = model_dir
    print(f"[Server] Model ready on {next(_model.parameters()).device}", flush=True)


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


# ── request / response schemas ────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    stream: bool = False
    # accepted but ignored (transformers doesn't support these natively)
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    text = _tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    device = next(_model.parameters()).device
    inputs = _tokenizer(text, return_tensors="pt").to(device)

    gen_kw: dict = dict(
        max_new_tokens=min(req.max_tokens, 2048),
        top_p=req.top_p,
        pad_token_id=_tokenizer.eos_token_id,
    )
    if req.temperature > 0:
        gen_kw.update(temperature=req.temperature, do_sample=True)
    else:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kw)

    new_ids = out[0][inputs["input_ids"].shape[1]:]
    reply   = _tokenizer.decode(new_ids, skip_special_tokens=True)

    return {
        "id":      f"chatcmpl-{uuid.uuid4().hex}",
        "object":  "chat.completion",
        "created": int(time.time()),
        "model":   req.model,
        "choices": [{
            "index":         0,
            "message":       {"role": "assistant", "content": reply},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens":     inputs["input_ids"].shape[1],
            "completion_tokens": len(new_ids),
            "total_tokens":      inputs["input_ids"].shape[1] + len(new_ids),
        },
    }


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--port",      type=int, default=8000)
    args = parser.parse_args()

    _load(args.model_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
