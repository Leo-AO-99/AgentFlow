"""
Step 6: Evaluate the Flow-GRPO (CodeGen) trained model on MBPP test set.

Loads the MBPP sanitized test split, queries the merged model via vLLM,
and reports pass@1 based on executing generated code against test cases.

Usage
-----
    modal run scripts/step6_flow_grpo_codegen/modal_eval.py            # all test problems
    modal run scripts/step6_flow_grpo_codegen/modal_eval.py --step 10  # specific checkpoint
    modal run scripts/step6_flow_grpo_codegen/modal_eval.py --force    # re-merge
    modal run scripts/step6_flow_grpo_codegen/modal_eval.py --n-samples 50
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ───────────────────────────────────────────────────────────
app       = modal.App("Ao-agentflow-step6-eval")
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
# Helpers (identical to step5)
# ─────────────────────────────────────────────────────────────────────────────

def _find_actor_dir(ckpt_root: str, step: int | None) -> str:
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


def _prepare_model_dir(
    actor_dir: str,
    base_model_name: str,
    output_dir: str,
    force: bool = False,
) -> None:
    import shutil, torch, transformers
    from huggingface_hub import snapshot_download
    from peft import PeftModel

    lora_adapter_path = os.path.join(actor_dir, "lora_adapter")
    if not os.path.isdir(lora_adapter_path):
        raise FileNotFoundError(f"lora_adapter/ not found at {lora_adapter_path}.")

    if force and os.path.isdir(output_dir):
        print(f"[Model] --force: removing {output_dir}")
        shutil.rmtree(output_dir)

    weight_files = (
        [f for f in os.listdir(output_dir)
         if f.endswith(".safetensors") or f.endswith(".bin")]
        if os.path.isdir(output_dir) else []
    )
    if weight_files:
        print(f"[Model] Merged model already at {output_dir} ({weight_files}), skipping.")
        return

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"[Model] Loading base model {base_model_name} …")
    base = None
    for cls_name in ["AutoModelForVision2Seq", "AutoModelForConditionalGeneration", "AutoModelForCausalLM"]:
        try:
            cls  = getattr(transformers, cls_name)
            base = cls.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, trust_remote_code=True)
            print(f"[Model] Loaded base as {cls_name} ({type(base).__name__})")
            break
        except Exception as e:
            print(f"[Model] {cls_name} failed: {e}")
    if base is None:
        raise RuntimeError(f"Could not load base model {base_model_name}")

    print(f"[Model] Applying LoRA from {lora_adapter_path} …")
    merged = PeftModel.from_pretrained(base, lora_adapter_path).merge_and_unload()

    print(f"[Model] Saving merged model to {output_dir} …")
    merged.save_pretrained(output_dir, safe_serialization=True)

    saved = [f for f in os.listdir(output_dir) if f.endswith(".safetensors") or f.endswith(".bin")]
    if not saved:
        raise RuntimeError(f"save_pretrained wrote no weight files! Contents: {os.listdir(output_dir)}")
    print(f"[Model] Weights saved: {saved}")

    print(f"[Model] Back-filling missing config files from HF snapshot …")
    hf_snapshot = snapshot_download(base_model_name, ignore_patterns=["*.safetensors", "*.bin", "*.pt", "*.gguf"])
    added = []
    for fname in os.listdir(hf_snapshot):
        src = os.path.join(hf_snapshot, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            import shutil as _sh
            _sh.copy2(src, dst)
            added.append(fname)
    if added:
        print(f"[Model] Added from HF: {added}")
    print(f"[Model] Contents: {sorted(os.listdir(output_dir))}")


def _patch_config_for_vllm(actor_dir: str, model_dir: str) -> None:
    import json
    orig_cfg = os.path.join(actor_dir, "huggingface", "config.json")
    if not os.path.exists(orig_cfg):
        print("[Config] Original VL config not found, skipping patch.")
        return
    with open(orig_cfg) as f:
        cfg = json.load(f)
    cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    cfg.pop("attn_implementation", None)
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("[Config] Patched config.json → architectures=Qwen3_5ForCausalLM (vLLM compat)")


def _start_vllm(model_dir: str, port: int, gpu_memory_utilization: float = 0.85, max_model_len: int = 32768) -> subprocess.Popen:
    print(f"[vLLM] Starting server: {model_dir}  port={port}")
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


def _run_mbpp_eval(vllm_base_url: str, n_samples: int | None, out_dir: str) -> dict:
    """Evaluate on MBPP sanitized test split. Returns pass@1 stats."""
    import json, random, re, subprocess as _sp, sys as _sys, tempfile
    from datasets import load_dataset
    from openai import OpenAI

    SYSTEM_PROMPT = (
        "You are an expert Python programmer. "
        "Write clean, correct Python code to solve the given programming problem."
    )
    TASK_INSTRUCTION = (
        "Write a Python function to solve the following programming problem. "
        "Your answer must be valid, self-contained Python code enclosed in a ```python``` block."
    )

    ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=False)

    if n_samples is not None and n_samples < len(ds):
        random.seed(42)
        indices = sorted(random.sample(range(len(ds)), n_samples))
        ds = ds.select(indices)

    client = OpenAI(base_url=vllm_base_url, api_key="dummy")
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
    results = []
    passed = 0

    for item in ds:
        task_id = item["task_id"]
        out_file = os.path.join(out_dir, f"task_{task_id}.json")

        # Resume: skip already-evaluated problems
        if os.path.exists(out_file):
            with open(out_file) as f:
                prev = json.load(f)
            if prev.get("passed"):
                passed += 1
            results.append(prev)
            print(f"[MBPP] task_id={task_id}  pass={prev.get('passed')}  (cached)")
            continue

        prompt_text = f"{TASK_INSTRUCTION}\n\n{item['prompt']}"
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt_text},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            raw_answer = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[MBPP] task_id={task_id}  vLLM error: {e}")
            raw_answer = ""

        code = _extract_code(raw_answer)
        ok   = _run_tests(code, item["test_list"])
        if ok:
            passed += 1

        record = {
            "task_id":  task_id,
            "prompt":   item["prompt"],
            "code":     code,
            "passed":   ok,
            "raw":      raw_answer,
        }
        with open(out_file, "w") as f:
            json.dump(record, f, indent=2)
        results.append(record)
        print(f"[MBPP] task_id={task_id}  pass={ok}")

    total     = len(results)
    pass_rate = passed / total if total > 0 else 0.0
    summary   = {"pass@1": pass_rate, "passed": passed, "total": total}

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[MBPP] pass@1 = {passed}/{total} = {pass_rate:.3f}")
    return summary


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
def eval_step6(step: int = None, force: bool = False, n_samples: int = None):
    runs.reload()

    code_dir     = "/runs/step6/code"
    ckpt_root    = "/runs/step6/checkpoints"
    merged_dir   = "/runs/step6/merged_model"
    eval_results = "/runs/step6/eval_results/mbpp"
    os.makedirs(eval_results, exist_ok=True)

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         f"{code_dir}/agentflow/", "-q"],
        cwd=code_dir, check=True,
    )

    actor_dir = _find_actor_dir(ckpt_root, step)

    _prepare_model_dir(
        actor_dir=actor_dir,
        base_model_name="Qwen/Qwen3.5-0.8B",
        output_dir=merged_dir,
        force=force,
    )

    _patch_config_for_vllm(actor_dir=actor_dir, model_dir=merged_dir)

    vllm_port     = 8000
    vllm_base_url = f"http://localhost:{vllm_port}/v1"
    vllm_proc     = _start_vllm(merged_dir, vllm_port)

    try:
        _wait_for_vllm(vllm_base_url, timeout=300)
        summary = _run_mbpp_eval(
            vllm_base_url=vllm_base_url,
            n_samples=n_samples,
            out_dir=eval_results,
        )
    finally:
        print("[vLLM] Shutting down …")
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()

    runs.commit()

    print("\n====== Step 6 MBPP eval complete ======")
    print(f"pass@1 = {summary['passed']}/{summary['total']} = {summary['pass@1']:.3f}")
    print(f"Results → {eval_results}/")
    print("Download:")
    print(f"  modal volume get ao-runs /step6/eval_results/ ./step6_eval_results/")


# ─────────────────────────────────────────────────────────────────────────────
# Local entrypoint
# ─────────────────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(step: int = None, force: bool = False, n_samples: int = None):
    """
    Args:
        --step INT        Checkpoint step to evaluate (default: latest).
        --force           Delete merged_model dir and re-merge from scratch.
        --n-samples INT   Subsample N problems from the test set (default: all ~257).
    """
    print(f"Step 6 MBPP eval  step={step or 'latest'}  force={force}  n_samples={n_samples or 'all'}")
    eval_step6.remote(step=step, force=force, n_samples=n_samples)
