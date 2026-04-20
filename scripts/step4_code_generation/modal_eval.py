"""
Step 4: Evaluate any HF model on MBPP (sanitized test split) via vLLM on Modal.

Upload code once:
    modal volume put ao-runs . /step4/code/

Run:
    modal run scripts/step4_code_generation/modal_eval.py --model Qwen/Qwen3.5-0.8B-Instruct
    modal run scripts/step4_code_generation/modal_eval.py --model Qwen/Qwen3.5-2B-Instruct
    modal run scripts/step4_code_generation/modal_eval.py --model Qwen/Qwen3.5-0.8B-Instruct --n-samples 20
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ───────────────────────────────────────────────────────────
app       = modal.App("Ao-agentflow-step4-eval")
hf        = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)
runs      = modal.Volume.from_name("ao-runs", create_if_missing=True)
ao_secret = modal.Secret.from_name("Ao-secret")

FLASH_ATTN_WHL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels"
    "/releases/download/v0.7.16"
    "/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(["git", "wget", "curl"])
    .pip_install([
        "torch==2.10.0",
        "peft==0.18.1",
        "trl==0.26.0",
        "vllm==0.19.0",
        "openai",
        "wandb",
        "datasets",
        "accelerate",
        "ray[default]",
        "math_verify",
        "aiohttp",
        "fastapi",
        "uvicorn",
        "fire",
        "agentops<=0.4.18",
        "flask",
        "setproctitle",
        "psutil",
        "httpdbg",
        "graphviz",
        "omegaconf",
        "codetiming",
        "tenacity",
        "filelock",
        "wikipedia",
    ])
    .run_commands("pip install git+https://github.com/verl-project/verl.git")
    .pip_install(["transformers==5.4.0"])
    .pip_install(FLASH_ATTN_WHL)
)


# ── vLLM helpers ──────────────────────────────────────────────────────────────

def _start_vllm(model_name: str, port: int) -> subprocess.Popen:
    print(f"[vLLM] Starting: {model_name}  port={port}")
    return subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", "32768",
            "--enforce-eager",
            "--gpu-memory-utilization", "0.85",
            "--trust-remote-code",
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"},
    )


def _wait_for_vllm(base_url: str, timeout: int = 300) -> None:
    import urllib.request, time
    health = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    print(f"[vLLM] Waiting for {health} …")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health, timeout=5) as r:
                if r.status == 200:
                    print("[vLLM] Ready.")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError(f"vLLM did not start within {timeout}s")


# ── MBPP eval ─────────────────────────────────────────────────────────────────

def _run_mbpp(vllm_base_url: str, n_samples: int | None, out_dir: str) -> dict:
    import json, random, re, subprocess as _sp, sys as _sys, tempfile
    from datasets import load_dataset
    from openai import OpenAI

    SYSTEM = (
        "You are an expert Python programmer. "
        "Write clean, correct Python code to solve the given programming problem. "
        "Your answer must be valid, self-contained Python code enclosed in a ```python``` block."
    )

    ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=False)

    if n_samples is not None and n_samples < len(ds):
        random.seed(42)
        indices = sorted(random.sample(range(len(ds)), n_samples))
        ds = ds.select(indices)

    client   = OpenAI(base_url=vllm_base_url, api_key="dummy")
    model_id = client.models.list().data[0].id
    print(f"[MBPP] Model: {model_id}  problems: {len(ds)}")

    def _extract_code(text: str) -> str:
        m = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    def _run_tests(code: str, test_list: list) -> bool:
        test_code = "\n".join(
            f"assert {tc}" if not tc.strip().startswith("assert") else tc
            for tc in test_list
        )
        full = code + "\n\n" + test_code + "\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(full)
            path = f.name
        try:
            r = _sp.run([_sys.executable, path], capture_output=True, text=True, timeout=10)
            return r.returncode == 0
        except Exception:
            return False
        finally:
            os.unlink(path)

    os.makedirs(out_dir, exist_ok=True)

    def solve_one(item: dict) -> dict:
        task_id  = item["task_id"]
        out_file = os.path.join(out_dir, f"task_{task_id}.json")

        if os.path.exists(out_file):
            with open(out_file) as f:
                prev = json.load(f)
            print(f"[MBPP] task_id={task_id}  pass={prev.get('passed')}  (cached)")
            return prev

        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user",   "content": item["prompt"]},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[MBPP] task_id={task_id}  vLLM error: {e}")
            raw = ""

        code = _extract_code(raw)
        ok   = _run_tests(code, item["test_list"])

        record = {"task_id": task_id, "prompt": item["prompt"], "code": code, "passed": ok, "raw": raw}
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[MBPP] task_id={task_id}  pass={ok}")
        return record

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(ds)) as pool:
        results = list(pool.map(solve_one, list(ds)))

    passed = sum(1 for r in results if r.get("passed"))

    total     = len(results)
    pass_rate = passed / total if total > 0 else 0.0
    summary   = {"pass@1": pass_rate, "passed": passed, "total": total}

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[MBPP] pass@1 = {passed}/{total} = {pass_rate:.3f}")
    return summary


# ── Shared eval body ─────────────────────────────────────────────────────────

def _eval_body(model: str, n_samples: int | None):
    runs.reload()

    label   = model.split("/")[-1]
    out_dir = f"/runs/step4/eval_results/{label}/mbpp"

    vllm_port     = 8000
    vllm_base_url = f"http://localhost:{vllm_port}/v1"
    vllm_proc     = _start_vllm(model, vllm_port)

    try:
        _wait_for_vllm(vllm_base_url, timeout=600)
        summary = _run_mbpp(vllm_base_url=vllm_base_url, n_samples=n_samples, out_dir=out_dir)
    finally:
        print("[vLLM] Shutting down …")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    runs.commit()

    print(f"\n====== Step 4 MBPP eval: {label} ======")
    print(f"pass@1 = {summary['passed']}/{summary['total']} = {summary['pass@1']:.3f}")
    print(f"Results → {out_dir}/")
    print(f"Download: modal volume get ao-runs /step4/eval_results/{label}/ ./{label}_results/")


# ── Modal functions (one per GPU tier) ───────────────────────────────────────

_common = dict(
    image=image,
    timeout=60 * 60 * 3,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=[ao_secret],
    memory=65536,
)

@app.function(gpu="L40S", **_common)
def run_eval_l40s(model: str, n_samples: int | None):
    _eval_body(model, n_samples)

@app.function(gpu="A100-80GB", **_common)
def run_eval_a100(model: str, n_samples: int | None):
    _eval_body(model, n_samples)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-0.8B-Instruct", n_samples: int = None, gpu: str = "L40S"):
    """
    Args:
        --model STR       HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-Instruct)
        --n-samples INT   Subsample N problems from test set (default: all ~257)
        --gpu STR         L40S (≤24B, default) | A100-80GB (27B+)
    """
    label = model.split("/")[-1]
    print(f"Step 4 MBPP eval: {label}  n_samples={n_samples or 'all'}  gpu={gpu}")

    fn = {"L40S": run_eval_l40s, "A100-80GB": run_eval_a100}.get(gpu)
    if fn is None:
        raise ValueError(f"Unknown gpu '{gpu}'. Choose: L40S, A100-80GB")
    fn.remote(model=model, n_samples=n_samples)
