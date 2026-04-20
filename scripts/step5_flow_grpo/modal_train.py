"""
Step 5: Flow-GRPO training on Qwen3.5-0.8B-Instruct with LoRA on Modal.
Uses paper benchmarks (search + math + science).

Dev workflow:
    1. Upload code (from project root):
           modal volume put ao-runs . /step5/code
    2. Dev run (10 min timeout to check for OOM/errors):
           modal run scripts/step5_flow_grpo/modal_train.py --dev
    3. Full run (change timeout=600 → timeout=7200 first):
           modal run scripts/step5_flow_grpo/modal_train.py

GPU: L40S (48 GB VRAM) — sufficient for Qwen3.5-0.8B with LoRA + vLLM rollout.
"""

import modal
import subprocess
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Modal resources ─────────────────────────────────────────────────────────────
app = modal.App("Ao-agentflow-step5-train")
hf = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)
runs = modal.Volume.from_name("ao-runs", create_if_missing=True)
ao_secret = modal.Secret.from_name("Ao-secret")  # for gpt-4o-mini fixed components

FLASH_ATTN_WHL = (
    "https://github.com/mjun0812/flash-attention-prebuild-wheels"
    "/releases/download/v0.7.16"
    "/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"
)

# ── Image ────────────────────────────────────────────────────────────────────────
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTHONUNBUFFERED": "1", "DEBIAN_FRONTEND": "noninteractive"})
    .apt_install(["git", "wget", "curl"])
    .pip_install([
        "torch==2.10.0",
        "peft==0.18.1",
        "trl==0.26.0",
        "vllm==0.19.0",
        "openai",        # version resolved by pip to satisfy vllm 0.19.0 (needs >=1.52.0)
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


@app.function(
    image=image,
    gpu="L40S",
    timeout=60*60*5,         # 24h max; resume_mode=auto continues from checkpoint if re-run
    volumes={
        "/root/.cache/huggingface": hf,
        "/runs": runs,
    },
    secrets=[ao_secret],
    memory=65536,          # 64 GB RAM
)
def train(dev: bool = False):
    """Run Flow-GRPO training for Step 5.

    Code layout inside the container (uploaded via `modal volume put ao-runs . /step5/code`):
        /runs/step5/code/   ← project root
        /runs/step5/data/   ← training data (auto-generated + cached)
        /runs/step5/checkpoints/ ← verl checkpoints (persisted on volume)

    Two processes run in parallel:
        1. train_agent.py  → starts verl + AgentFlow server (provides tasks to rollout workers)
        2. rollout.py      → connects to the server, runs agent rollouts, sends results back
    """
    import os
    import sys
    import subprocess

    runs.reload()   # pull latest volume state so new container sees prior checkpoints

    code_dir = "/runs/step5/code"
    data_dir = "/runs/step5/data"
    config_path = f"{code_dir}/scripts/step5_flow_grpo/config_0.8b.yaml"

    # ── 1. Working directory + PYTHONPATH ────────────────────────────────────
    os.chdir(code_dir)

    env = {
        **os.environ,
        # Project root on path → 'util', 'data', etc. importable
        # Script dir is added by Python automatically (train/ for rollout.py)
        "PYTHONPATH": code_dir,
        "PYTHONUNBUFFERED": "1",
        "RAY_DEBUG": "legacy",
        "HYDRA_FULL_ERROR": "1",
        "VLLM_USE_V1": "0",
        "CUDA_VISIBLE_DEVICES": "0",
        # Route non-training vllm- model calls through Portkey → DeepInfra.
        # ChatVLLM uses these env vars when base_url is not explicitly passed.
        # The trainable model always gets its URL from resources, so it is unaffected.
        "VLLM_BASE_URL": "https://api.portkey.ai/v1",
        "VLLM_API_KEY": os.environ.get("PORTKEY_API_KEY", ""),
    }

    # ── 2. Install agentflow package from volume code ─────────────────────────
    # Install only the agentflow package itself — deps are already baked into the image.
    # --no-deps avoids re-resolving requirements.txt which could downgrade vllm/transformers.
    print("[Setup] Installing agentflow package (no-deps)...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "-e",
         f"{code_dir}/agentflow/", "-q"],
        cwd=code_dir, check=True,
    )

    # ── 3. Generate training/validation data (cached after first run) ─────────
    train_parquet = f"{data_dir}/train/combined_train.parquet"
    val_parquet   = f"{data_dir}/val/aime24.parquet"

    if not os.path.exists(train_parquet):
        print("[Data] Generating train data (NQ + MathHard) — this may take ~10 min...")
        os.makedirs(f"{data_dir}/train", exist_ok=True)
        subprocess.run(
            [sys.executable, "data/get_train_data.py",
             f"--output_dir={data_dir}/train"],
            cwd=code_dir, env=env, check=True,
        )
        print(f"[Data] Saved: {train_parquet}")
    else:
        print(f"[Data] Cached train data found: {train_parquet}")

    if not os.path.exists(val_parquet):
        print("[Data] Generating val data (AIME24)...")
        os.makedirs(f"{data_dir}/val", exist_ok=True)
        subprocess.run(
            [sys.executable, "data/aime24_data.py",
             f"--output_dir={data_dir}/val"],
            cwd=code_dir, env=env, check=True,
        )
        print(f"[Data] Saved: {val_parquet}")
    else:
        print(f"[Data] Cached val data found: {val_parquet}")

    # ── 4. Dev-mode overrides ─────────────────────────────────────────────────
    # Note: do NOT run `ray start --head` here.
    # verl's entrypoint does ray.init(num_cpus=N) which creates the local cluster.
    # Pre-starting Ray causes "When connecting to an existing cluster, num_cpus
    # must not be provided" — the error we hit before.
    dev_overrides: list[str] = []
    if dev:
        dev_overrides = [
            "data.train_batch_size=4",
            "actor_rollout_ref.rollout.n=2",
            "trainer.total_epochs=1",
            "trainer.val_before_train=False",  # skip 9-min validation with pretrained weights
        ]
        print(f"[Dev] Active overrides: {dev_overrides}")

    # ── 6. Launch rollout workers (background) ────────────────────────────────
    rollout_cmd = [
        sys.executable, f"{code_dir}/train/rollout.py",
        "--config", config_path,
    ]
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

        # ── 8. Commit volume regardless of success/failure ───────────────────
        # Ensures checkpoints saved mid-run are persisted even if training crashes.
        print("[Volume] Committing ao-runs volume...")
        runs.commit()

    print("[Done] Step 5 training complete.")
    print("       Checkpoints → /runs/step5/checkpoints/")
    print("       Download:    modal volume get ao-runs /runs/step5/checkpoints/ ./step5_checkpoints/")


@app.local_entrypoint()
def main(dev: bool = False):
    """
    Args:
        --dev: Run a short 1-epoch dev check (10-min timeout).
               Remove --dev for the full training run (change timeout to 7200).
    """
    print(f"Starting Step 5 training (dev={dev})...")
    train.remote(dev=dev)
