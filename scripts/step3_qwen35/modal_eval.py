"""
Step 3: Evaluate any HF model on QA benchmarks (bamboogle/2wiki/hotpotqa/musique/gaia)
via vLLM on Modal.

Upload code once:
    modal volume put ao-runs . /step3/code/

Run:
    modal run scripts/step3_qwen35/modal_eval.py --model Qwen/Qwen3.5-0.8B-Instruct
    modal run scripts/step3_qwen35/modal_eval.py --model Qwen/Qwen3.5-2B-Instruct
    modal run scripts/step3_qwen35/modal_eval.py --model Qwen/Qwen3.5-27B-Instruct --gpu A100-80GB
    modal run scripts/step3_qwen35/modal_eval.py --model Qwen/Qwen3.5-0.8B-Instruct --n-samples 5
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ───────────────────────────────────────────────────────────
app       = modal.App("Ao-agentflow-step3-eval")
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


def _wait_for_vllm(base_url: str, timeout: int = 600) -> None:
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


# ── Benchmark runner (identical to step5) ────────────────────────────────────

def _run_benchmark(
    task: str, test_dir: str, out_dir: str, log_dir: str,
    llm_engine: str, vllm_base_url: str, model_engine: str,
    enabled_tools: str, tool_engine: str, n_samples: int,
    threads: int, env: dict,
) -> None:
    import json, random, concurrent.futures

    data_file = f"{test_dir}/{task}/data/data.json"
    cache_dir = f"{test_dir}/{task}/cache"
    os.makedirs(out_dir,   exist_ok=True)
    os.makedirs(log_dir,   exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    with open(data_file) as f:
        data = json.load(f)

    random.seed(42)
    k       = min(n_samples, len(data))
    indices = sorted(random.sample(range(len(data)), k))
    todo    = [i for i in indices if not os.path.exists(f"{out_dir}/output_{i}.json")]

    if not todo:
        print(f"[{task}] Already complete ({k} problems), skipping.")
        return

    print(f"[{task}] Running {len(todo)}/{k} problems …")

    def solve_one(idx: int):
        with open(f"{log_dir}/{idx}.log", "w") as lf:
            subprocess.run(
                [
                    sys.executable, f"{test_dir}/solve.py",
                    "--index",           str(idx),
                    "--task",            task,
                    "--data_file",       data_file,
                    "--llm_engine_name", llm_engine,
                    "--base_url",        vllm_base_url,
                    "--model_engine",    model_engine,
                    "--enabled_tools",   enabled_tools,
                    "--tool_engine",     tool_engine,
                    "--output_types",    "direct",
                    "--output_json_dir", out_dir,
                    "--root_cache_dir",  cache_dir,
                    "--max_steps",       "10",
                    "--max_time",        "300",
                    "--temperature",     "0.0",
                ],
                cwd=test_dir, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(solve_one, todo))

    subprocess.run(
        [
            sys.executable, f"{test_dir}/calculate_score_unified.py",
            "--task_name",     task,
            "--data_file",     data_file,
            "--result_dir",    out_dir,
            "--response_type", "direct_output",
            "--output_file",   "finalresults_direct_output.json",
        ],
        cwd=test_dir, env=env, check=False,
    )
    print(f"[{task}] Done.")


# ── Shared eval body ──────────────────────────────────────────────────────────

def _eval_body(model: str, n_samples: int):
    import shutil, concurrent.futures

    runs.reload()

    label        = model.split("/")[-1]          # e.g. "Qwen3.5-0.8B-Instruct"
    code_dir     = "/runs/step3/code"
    eval_results = f"/runs/step3/eval_results/{label}"
    os.makedirs(eval_results, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         f"{code_dir}/agentflow/", "-q"],
        cwd=code_dir, check=True,
    )

    vllm_port     = 8000
    vllm_base_url = f"http://localhost:{vllm_port}/v1"
    vllm_proc     = _start_vllm(model, vllm_port)

    try:
        _wait_for_vllm(vllm_base_url, timeout=600)

        benchmarks    = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
        test_dir      = f"{code_dir}/test"

        judge_engine  = "vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct"
        llm_engine    = f"vllm-{model}"
        model_engine  = f"trainable,{judge_engine},{judge_engine},{judge_engine}"
        enabled_tools = "Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
        tool_engine   = "self,self,Default,Default"
        threads       = 4

        env = {
            **os.environ,
            "PYTHONPATH":       code_dir,
            "PYTHONUNBUFFERED": "1",
            "VLLM_BASE_URL":    "https://api.portkey.ai/v1",
            "VLLM_API_KEY":     os.environ.get("PORTKEY_API_KEY", ""),
        }

        def run_task(task: str):
            _run_benchmark(
                task=task,
                test_dir=test_dir,
                out_dir=f"{test_dir}/{task}/results/{label}",
                log_dir=f"{test_dir}/{task}/logs/{label}",
                llm_engine=llm_engine,
                vllm_base_url=vllm_base_url,
                model_engine=model_engine,
                enabled_tools=enabled_tools,
                tool_engine=tool_engine,
                n_samples=n_samples,
                threads=threads,
                env=env,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(benchmarks)) as pool:
            list(pool.map(run_task, benchmarks))

        for task in benchmarks:
            src = f"{test_dir}/{task}/results/{label}"
            dst = f"{eval_results}/{task}"
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copytree(src, dst)

    finally:
        print("[vLLM] Shutting down …")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    runs.commit()

    print(f"\n====== Step 3 eval: {label} ======")
    print(f"Results → {eval_results}/")
    print(f"Download: modal volume get ao-runs /step3/eval_results/{label}/ ./{label}_results/")


# ── Modal functions (one per GPU tier) ───────────────────────────────────────

_common = dict(
    image=image,
    timeout=60 * 60 * 3,
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=[ao_secret],
    memory=65536,
)

@app.function(gpu="L40S", **_common)
def run_eval_l40s(model: str, n_samples: int):
    _eval_body(model, n_samples)

@app.function(gpu="A100-80GB", **_common)
def run_eval_a100(model: str, n_samples: int):
    _eval_body(model, n_samples)


# ── Local entrypoint ──────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3.5-0.8B-Instruct", n_samples: int = 5, gpu: str = "L40S"):
    """
    Args:
        --model STR       HuggingFace model ID (default: Qwen/Qwen3.5-0.8B-Instruct)
        --n-samples INT   Problems per benchmark (default: 5)
        --gpu STR         L40S (≤24B, default) | A100-80GB (27B+)
    """
    label = model.split("/")[-1]
    print(f"Step 3 eval: {label}  n_samples={n_samples}  gpu={gpu}")

    fn = {"L40S": run_eval_l40s, "A100-80GB": run_eval_a100}.get(gpu)
    if fn is None:
        raise ValueError(f"Unknown gpu '{gpu}'. Choose: L40S, A100-80GB")
    fn.remote(model=model, n_samples=n_samples)
