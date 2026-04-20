"""
Step 6: Flow-GRPO training on Qwen3.5-0.8B for Code Generation (MBPP).

Infrastructure is identical to step5 (same image, same launch pattern).
Only the training data and reward function differ.

Dev workflow:
    1. Prepare and upload data (run locally once):
           python scripts/step6_flow_grpo_codegen/prepare_training_data.py --output_dir ./step6_data
           modal volume put ao-runs ./step6_data/ /step6/data/
           modal volume put ao-runs . /step6/code/
    2. Dev run (10 min timeout to check for OOM/errors):
           modal run scripts/step6_flow_grpo_codegen/modal_train.py --dev
    3. Full run (change timeout=600 → 18000 first):
           modal run scripts/step6_flow_grpo_codegen/modal_train.py
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ─────────────────────────────────────────────────────────────
app = modal.App("Ao-agentflow-step6-train")
hf = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)
runs = modal.Volume.from_name("ao-runs", create_if_missing=True)
ao_secret = modal.Secret.from_name("Ao-secret")

FLASH_ATTN_WHL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels"
    "/releases/download/v0.7.16"
    "/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

# ── Image (keep identical to step5) ─────────────────────────────────────────────
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

# ── Code execution reward function ───────────────────────────────────────────────
# Appended to train/utils.py before training starts so the training pipeline
# can call it via REWARD_FUNCTION=code_execution.
_CODEGEN_REWARD = '''

# === STEP 6: CODE EXECUTION REWARD ===
import re as _re, subprocess as _sp, tempfile as _tmp, sys as _sys, os as _os

def compute_score_codegen(question: str, groundtruth: dict, answer_extracted: str) -> float:
    """Execute generated code against test cases. Returns 1.0 if all pass, 0.0 otherwise."""
    def _extract(text):
        m = _re.search(r"```(?:python)?\\n(.*?)```", text, _re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    code = _extract(answer_extracted)
    full_code = code + "\\n\\n" + groundtruth.get("test_code", "") + "\\n"
    with _tmp.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_code)
        path = f.name
    try:
        r = _sp.run([_sys.executable, path], capture_output=True, text=True, timeout=10)
        return 1.0 if r.returncode == 0 else 0.0
    except Exception:
        return 0.0
    finally:
        _os.unlink(path)
'''


@app.function(
    image=image,
    gpu="L40S",
    timeout=60*60*5,          # 10 min for dev; change to 18000 for full run
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
    },
    secrets=[ao_secret],
    memory=65536,
)
def train(dev: bool = False):
    """Run Flow-GRPO training for Step 6 (Code Generation / MBPP)."""
    runs.reload()

    code_dir    = "/runs/step6/code"
    data_dir    = "/runs/step6/data"
    config_path = f"{code_dir}/scripts/step6_flow_grpo_codegen/config_0.8b_codegen.yaml"

    # ── 1. Working directory + env ───────────────────────────────────────────
    os.chdir(code_dir)

    env = {
        **os.environ,
        "PYTHONPATH":       code_dir,
        "PYTHONUNBUFFERED": "1",
        "RAY_DEBUG":        "legacy",
        "HYDRA_FULL_ERROR": "1",
        "VLLM_USE_V1":      "0",
        "CUDA_VISIBLE_DEVICES": "0",
        "VLLM_BASE_URL":    "https://api.portkey.ai/v1",
        "VLLM_API_KEY":     os.environ.get("PORTKEY_API_KEY", ""),
        "REWARD_FUNCTION":  "code_execution",
    }

    # ── 2. Install agentflow (no-deps, from volume) ──────────────────────────
    print("[Setup] Installing agentflow package (no-deps)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         f"{code_dir}/agentflow/", "-q"],
        cwd=code_dir, check=True,
    )

    # ── 3. Inject code execution reward into train/utils.py ──────────────────
    utils_path = f"{code_dir}/train/utils.py"
    with open(utils_path) as f:
        utils_src = f.read()
    if "compute_score_codegen" not in utils_src:
        with open(utils_path, "a") as f:
            f.write(_CODEGEN_REWARD)
        print("[Setup] Injected compute_score_codegen into train/utils.py")
    else:
        print("[Setup] compute_score_codegen already present in train/utils.py")

    # ── 4. Verify training data ───────────────────────────────────────────────
    train_parquet = f"{data_dir}/train/mbpp_train.parquet"
    val_parquet   = f"{data_dir}/val/mbpp_val.parquet"

    for path in [train_parquet, val_parquet]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training data not found: {path}\n"
                "Run locally first:\n"
                "  python scripts/step6_flow_grpo_codegen/prepare_training_data.py --output_dir ./step6_data\n"
                "  modal volume put ao-runs ./step6_data/ /step6/data/"
            )
    print(f"[Data] Found: {train_parquet}")
    print(f"[Data] Found: {val_parquet}")

    # ── 5. Dev-mode overrides ─────────────────────────────────────────────────
    dev_overrides: list[str] = []
    if dev:
        dev_overrides = [
            "data.train_batch_size=4",
            "actor_rollout_ref.rollout.n=2",
            "trainer.total_epochs=1",
            "trainer.val_before_train=False",
        ]
        print(f"[Dev] Active overrides: {dev_overrides}")

    # ── 6. Launch rollout workers (background) ────────────────────────────────
    rollout_cmd = [sys.executable, f"{code_dir}/train/rollout.py", "--config", config_path]
    print(f"[Rollout] Starting workers: {' '.join(rollout_cmd)}")
    rollout_proc = subprocess.Popen(rollout_cmd, cwd=code_dir, env=env)

    # ── 7. Launch verl training (foreground) ─────────────────────────────────
    train_cmd = [
        sys.executable, f"{code_dir}/train/train_agent.py",
        "--config", config_path,
    ] + dev_overrides
    print(f"[Train] Starting verl: {' '.join(train_cmd)}")

    try:
        subprocess.run(train_cmd, cwd=code_dir, env=env, check=True)
    finally:
        print("[Cleanup] Terminating rollout workers...")
        rollout_proc.terminate()
        try:
            rollout_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            rollout_proc.kill()

        print("[Volume] Committing ao-runs volume...")
        runs.commit()

    print("[Done] Step 6 training complete.")
    print("       Checkpoints → /runs/step6/checkpoints/")
    print("       Download:    modal volume get ao-runs /step6/checkpoints/ ./step6_checkpoints/")


@app.local_entrypoint()
def main(dev: bool = False):
    """
    Args:
        --dev: Short 1-epoch dev check. Change timeout to 18000 for full run.
    """
    print(f"Starting Step 6 training (dev={dev})...")
    train.remote(dev=dev)
