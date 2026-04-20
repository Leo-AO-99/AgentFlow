"""
Inspect the verl checkpoint to understand exact file structure and weight keys.
Run:
    modal run scripts/step5_flow_grpo/modal_inspect_ckpt.py
"""
import modal

app  = modal.App("Ao-inspect-ckpt")
runs = modal.Volume.from_name("ao-runs", create_if_missing=True)
hf   = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)

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

@app.function(
    image=image,
    volumes={
        "/runs": runs,
        "/root/.cache/huggingface": hf,
    },
    timeout=300,
)
def inspect():
    import os, json
    import torch
    from safetensors import safe_open

    runs.reload()

    ckpt_root = "/runs/step5/checkpoints"

    # ── 1. show overall checkpoint tree ──────────────────────────────────────
    print("\n=== Checkpoint tree ===")
    for root, dirs, files in os.walk(ckpt_root):
        depth = root.replace(ckpt_root, "").count(os.sep)
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/")
        for f in sorted(files):
            size = os.path.getsize(os.path.join(root, f))
            print(f"{indent}  {f}  ({size/1024/1024:.1f} MB)")

    # ── 2. find actor dir ─────────────────────────────────────────────────────
    import glob, re
    actor_dirs = glob.glob(f"{ckpt_root}/**/global_step_*/actor", recursive=True)
    if not actor_dirs:
        print("No actor dirs found!")
        return
    actor_dir = max(actor_dirs, key=lambda p: int(re.search(r"global_step_(\d+)", p).group(1)))
    print(f"\n=== Using actor dir: {actor_dir} ===")

    # ── 3. inspect lora_adapter ───────────────────────────────────────────────
    lora_path = os.path.join(actor_dir, "lora_adapter")
    print(f"\n=== lora_adapter/ contents ===")
    for f in os.listdir(lora_path):
        print(f"  {f}")

    adapter_cfg = os.path.join(lora_path, "adapter_config.json")
    with open(adapter_cfg) as fh:
        cfg = json.load(fh)
    print(f"\nadapter_config.json:")
    print(json.dumps(cfg, indent=2))

    safetensor_file = os.path.join(lora_path, "adapter_model.safetensors")
    print(f"\nadapter_model.safetensors keys (first 20):")
    with safe_open(safetensor_file, framework="pt") as f:
        keys = list(f.keys())
        for k in keys[:20]:
            t = f.get_tensor(k)
            print(f"  {k:80s}  {tuple(t.shape)}  {t.dtype}")
    print(f"  ... total {len(keys)} keys")

    # ── 4. inspect FSDP shard ─────────────────────────────────────────────────
    shard_file = os.path.join(actor_dir, "model_world_size_1_rank_0.pt")
    print(f"\n=== FSDP shard: model_world_size_1_rank_0.pt ===")
    sd = torch.load(shard_file, map_location="cpu", weights_only=False)
    shard_keys = list(sd.keys())
    print(f"Total keys: {len(shard_keys)}")
    for k in shard_keys[:10]:
        v = sd[k]
        shape = tuple(v.shape) if hasattr(v, "shape") else type(v).__name__
        dtype = v.dtype if hasattr(v, "dtype") else "N/A"
        print(f"  {k:80s}  {shape}  {dtype}")
    print("  ...")

    # ── 5. inspect huggingface/ dir ───────────────────────────────────────────
    hf_dir = os.path.join(actor_dir, "huggingface")
    print(f"\n=== huggingface/ contents ===")
    for f in sorted(os.listdir(hf_dir)):
        print(f"  {f}")


@app.local_entrypoint()
def main():
    inspect.remote()
