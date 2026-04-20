"""
Step 5: Evaluate the Flow-GRPO trained model on Modal.

verl saves a standard PEFT adapter at actor/lora_adapter/:
  - adapter_model.safetensors  (LoRA A/B weights)
  - adapter_config.json        (r, alpha, target_modules, …)

We load that directly — no FSDP shard merging needed.

Usage
-----
    modal run scripts/step5_flow_grpo/modal_eval.py            # latest checkpoint
    modal run scripts/step5_flow_grpo/modal_eval.py --step 10  # specific step
    modal run scripts/step5_flow_grpo/modal_eval.py --force    # re-merge
    modal run scripts/step5_flow_grpo/modal_eval.py --n-samples 5  # problems per benchmark
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ───────────────────────────────────────────────────────────
app       = modal.App("Ao-agentflow-step5-eval")
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: find checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def _find_actor_dir(ckpt_root: str, step: int | None) -> str:
    """Return the actor/ dir for the requested (or latest) global step."""
    import glob as _glob, re

    pattern = f"{ckpt_root}/**/global_step_*/actor"
    all_dirs = _glob.glob(pattern, recursive=True)
    if not all_dirs:
        raise FileNotFoundError(f"No checkpoints found under {ckpt_root}")

    def _step(path: str) -> int:
        m = re.search(r"global_step_(\d+)", path)
        return int(m.group(1)) if m else -1

    if step is not None:
        matched = [d for d in all_dirs if _step(d) == step]
        if not matched:
            available = sorted({_step(d) for d in all_dirs})
            raise FileNotFoundError(
                f"Checkpoint step {step} not found. Available: {available}"
            )
        return matched[0]

    best = max(all_dirs, key=_step)
    print(f"[Eval] Latest checkpoint: {best}  (step {_step(best)})")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build merged model directory from lora_adapter/
# ─────────────────────────────────────────────────────────────────────────────

def _prepare_model_dir(
    actor_dir: str,
    base_model_name: str,
    output_dir: str,
    force: bool = False,
) -> None:
    """
    Build a complete, vLLM-ready model directory.

    verl saves a standard PEFT adapter at actor/lora_adapter/:
      adapter_model.safetensors + adapter_config.json

    Steps
    -----
    1. Load base model + PeftModel.from_pretrained(lora_adapter/)
    2. merge_and_unload() → save_pretrained(output_dir)
       (saves weights + all config files in one shot)
    3. Back-fill any HF files missing from save_pretrained
       (preprocessor_config.json, chat_template.jinja, etc.)
    """
    import shutil, torch, transformers
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    lora_adapter_path = os.path.join(actor_dir, "lora_adapter")
    if not os.path.isdir(lora_adapter_path):
        raise FileNotFoundError(
            f"lora_adapter/ not found at {lora_adapter_path}.\n"
            "Make sure training reached at least one save_freq step and\n"
            "that lora_rank > 0 is set in the config."
        )

    if force and os.path.isdir(output_dir):
        print(f"[Model] --force: removing {output_dir}")
        shutil.rmtree(output_dir)

    # Check for existing weight files (model.safetensors or shards)
    weight_files = (
        [f for f in os.listdir(output_dir)
         if f.endswith(".safetensors") or f.endswith(".bin")]
        if os.path.isdir(output_dir)
        else []
    )
    if weight_files:
        print(f"[Model] Merged model already at {output_dir} ({weight_files}), skipping.")
        return

    # Clean up any partial state from previous failed runs
    if os.path.isdir(output_dir):
        print(f"[Model] Removing incomplete dir {output_dir} …")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # ── Step 1: load base model ───────────────────────────────────────────────
    print(f"[Model] Loading base model {base_model_name} …")
    base = None
    for cls_name in ["AutoModelForVision2Seq",
                     "AutoModelForConditionalGeneration",
                     "AutoModelForCausalLM"]:
        try:
            cls  = getattr(transformers, cls_name)
            base = cls.from_pretrained(
                base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
            )
            print(f"[Model] Loaded base as {cls_name} ({type(base).__name__})")
            break
        except Exception as e:
            print(f"[Model] {cls_name} failed: {e}")
    if base is None:
        raise RuntimeError(f"Could not load base model {base_model_name}")

    # ── Step 2: apply LoRA + merge ────────────────────────────────────────────
    print(f"[Model] Applying LoRA from {lora_adapter_path} …")
    peft_model = PeftModel.from_pretrained(base, lora_adapter_path)

    print("[Model] Merging LoRA weights …")
    merged = peft_model.merge_and_unload()

    # save_pretrained writes weights + config.json + tokenizer in one shot
    print(f"[Model] Saving merged model to {output_dir} …")
    merged.save_pretrained(output_dir, safe_serialization=True)

    # Verify weight files were actually written
    saved = [f for f in os.listdir(output_dir)
             if f.endswith(".safetensors") or f.endswith(".bin")]
    if not saved:
        raise RuntimeError(
            f"save_pretrained completed but no weight files found in {output_dir}!\n"
            f"Contents: {os.listdir(output_dir)}"
        )
    print(f"[Model] Weights saved: {saved}")

    # ── Step 3: back-fill missing HF files ───────────────────────────────────
    # preprocessor_config.json, chat_template.jinja, etc. are not saved by
    # save_pretrained for the base model class but vLLM needs them.
    print(f"[Model] Back-filling missing config files from HF snapshot …")
    hf_snapshot = snapshot_download(
        base_model_name,
        ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"],
    )
    added = []
    for fname in os.listdir(hf_snapshot):
        src = os.path.join(hf_snapshot, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            added.append(fname)
    if added:
        print(f"[Model] Added from HF: {added}")

    print(f"[Model] Model directory ready: {output_dir}")
    print(f"[Model] Contents: {sorted(os.listdir(output_dir))}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: patch config.json for vLLM compatibility
# ─────────────────────────────────────────────────────────────────────────────

def _patch_config_for_vllm(actor_dir: str, model_dir: str) -> None:
    """
    vLLM 0.19.0 registers its own Qwen3_5Config (VL-style, nested text_config +
    vision_config) for model_type='qwen3_5'.  AutoModelForCausalLM.save_pretrained()
    writes a flat Qwen3_5TextConfig (model_type='qwen3_5_text') which fails vLLM's
    isinstance() check.

    Fix: replace the merged model's config.json with the original VL config from the
    checkpoint (model_type='qwen3_5'), but override architectures to
    ['Qwen3_5ForCausalLM'] so vLLM loads the text-only model class and doesn't expect
    visual encoder weights.
    """
    import json, shutil

    orig_cfg = os.path.join(actor_dir, "huggingface", "config.json")
    if not os.path.exists(orig_cfg):
        print("[Config] Original VL config not found, skipping patch.")
        return

    with open(orig_cfg) as f:
        cfg = json.load(f)

    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    # Remove attn_implementation if set — vLLM handles this itself
    cfg.pop("attn_implementation", None)

    dst = os.path.join(model_dir, "config.json")
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[Config] Patched config.json → architectures=Qwen3_5ForCausalLM (vLLM compat)")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: vLLM
# ─────────────────────────────────────────────────────────────────────────────

def _start_vllm(
    model_dir: str,
    port: int,
    gpu_memory_utilization: float = 0.85,
    max_model_len: int = 32768,
) -> subprocess.Popen:
    print(f"[vLLM] Starting server: {model_dir}  port={port}  gpu_mem={gpu_memory_utilization}")
    return subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_dir,
            "--port", str(port),
            "--dtype", "bfloat16",
            "--max-model-len", str(max_model_len),
            "--enforce-eager",
            "--gpu-memory-utilization", str(gpu_memory_utilization),
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


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _run_benchmark(
    task: str,
    test_dir: str,
    out_dir: str,
    log_dir: str,
    llm_engine: str,
    vllm_base_url: str,
    model_engine: str,
    enabled_tools: str,
    tool_engine: str,
    n_samples: int,
    threads: int,
    env: dict,
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
                    "--index",        str(idx),
                    "--task",         task,
                    "--data_file",    data_file,
                    "--llm_engine_name", llm_engine,
                    "--base_url",     vllm_base_url,
                    "--model_engine", model_engine,
                    "--enabled_tools", enabled_tools,
                    "--tool_engine",  tool_engine,
                    "--output_types", "direct",
                    "--output_json_dir", out_dir,
                    "--root_cache_dir",  cache_dir,
                    "--max_steps",    "10",
                    "--max_time",     "300",
                    "--temperature",  "0.0",
                ],
                cwd=test_dir, env=env,
                stdout=lf, stderr=subprocess.STDOUT,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(solve_one, todo))

    subprocess.run(
        [
            sys.executable, f"{test_dir}/calculate_score_unified.py",
            "--task_name",    task,
            "--data_file",    data_file,
            "--result_dir",   out_dir,
            "--response_type", "direct_output",
            "--output_file",  "finalresults_direct_output.json",
        ],
        cwd=test_dir, env=env, check=False,
    )
    print(f"[{task}] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# Main Modal function
# ─────────────────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
    },
    secrets=[ao_secret],
    memory=65536,
)
def eval_step5(step: int = None, force: bool = False, n_samples: int = 5):
    import shutil, concurrent.futures

    runs.reload()

    code_dir     = "/runs/step5/code"
    ckpt_root    = "/runs/step5/checkpoints"
    merged_dir   = "/runs/step5/merged_model"
    eval_results = "/runs/step5/eval_results"
    os.makedirs(eval_results, exist_ok=True)

    # ── install agentflow ─────────────────────────────────────────────────────
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         f"{code_dir}/agentflow/", "-q"],
        cwd=code_dir, check=True,
    )

    # ── find checkpoint ───────────────────────────────────────────────────────
    actor_dir = _find_actor_dir(ckpt_root, step)

    # ── build merged model dir (base + LoRA from lora_adapter/) ─────────────
    _prepare_model_dir(
        actor_dir=actor_dir,
        base_model_name="Qwen/Qwen3.5-0.8B",
        output_dir=merged_dir,
        force=force,
    )

    # ── patch config.json so vLLM accepts the Qwen3.5 model ─────────────────
    _patch_config_for_vllm(actor_dir=actor_dir, model_dir=merged_dir)

    # ── start vLLM (trained model only; judges use Portkey) ─────────────────
    vllm_port     = 8000
    vllm_base_url = f"http://localhost:{vllm_port}/v1"
    vllm_proc     = _start_vllm(merged_dir, vllm_port)

    try:
        _wait_for_vllm(vllm_base_url, timeout=300)

        # ── benchmarks (same 5 as step 2) ─────────────────────────────────────
        benchmarks    = ["bamboogle", "2wiki", "hotpotqa", "musique", "gaia"]
        label         = "Qwen3.5-0.8B-FlowGRPO"
        test_dir      = f"{code_dir}/test"

        # planner_main = trained model; fixed/verifier/executor = Portkey → Qwen2.5-7B
        judge_engine  = "vllm-@deepinfra/Qwen/Qwen2.5-7B-Instruct"
        llm_engine    = f"vllm-{merged_dir}"
        model_engine  = f"trainable,{judge_engine},{judge_engine},{judge_engine}"
        enabled_tools = "Base_Generator_Tool,Python_Coder_Tool,Google_Search_Tool,Wikipedia_Search_Tool"
        tool_engine   = "Default,Default,Default,Default"
        threads       = 4

        env = {
            **os.environ,
            "PYTHONPATH":       code_dir,
            "PYTHONUNBUFFERED": "1",
            "VLLM_BASE_URL":    "https://api.portkey.ai/v1",  # judge engines fallback
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

        # ── persist results to volume ──────────────────────────────────────────
        for task in benchmarks:
            src = f"{test_dir}/{task}/results/{label}"
            dst = f"{eval_results}/{task}/{label}"
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

    print("\n====== Step 5 eval complete ======")
    print(f"Results → {eval_results}/<benchmark>/{label}/")
    print("Download:")
    print(f"  modal volume get ao-runs /step5/eval_results/ ./step5_eval_results/")


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(step: int = None, force: bool = False, n_samples: int = 5):
    """
    Args:
        --step INT        Checkpoint step to evaluate (default: latest).
        --force           Delete merged_model dir and re-merge from scratch.
        --n-samples INT   Number of problems per benchmark (default: 5).
    """
    print(f"Step 5 eval  step={step or 'latest'}  force={force}  n_samples={n_samples}")
    eval_step5.remote(step=step, force=force, n_samples=n_samples)
