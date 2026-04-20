"""
Step 3: Serve a Qwen3.5 model via vLLM on Modal as an OpenAI-compatible endpoint.

Usage:
    # Deploy and get the endpoint URL
    modal run scripts/step3_qwen35/serve_modal_vllm.py --model-name Qwen/Qwen3.5-0.8B-Instruct

    # Or serve persistently as a Modal deployment
    modal deploy scripts/step3_qwen35/serve_modal_vllm.py

Note: Upload this file via:
    modal volume put ao-runs scripts/step3_qwen35/serve_modal_vllm.py /runs/step3/serve_modal_vllm.py
"""

import modal
import subprocess
import time

# ── Modal resources ─────────────────────────────────────────────────────────────
app = modal.App("agentflow-step3-vllm")
hf = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface-secret")

# Use A100-40GB for 9B/27B models; L40S for smaller ones
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "vllm==0.6.4",
        "transformers>=4.46.0",
        "torch==2.4.0",
        "huggingface_hub",
    ])
)

VLLM_PORT = 8000


@app.function(
    image=image,
    gpu="L40S",           # override to "A100-40GB" for 9B/27B models
    timeout=600,          # 10 min; increase for long-running serving
    volumes={"/root/.cache/huggingface": hf},
    secrets=[hf_secret],
)
def serve_vllm(model_name: str, gpu_memory_utilization: float = 0.85) -> str:
    """
    Start a vLLM server for the given model and return the base URL.
    Blocks until server is ready.
    """
    import socket
    import os

    # Get the container's public URL via Modal tunnel (uses the MODAL_TASK_ID env trick)
    # For simplicity, we serve locally and the caller uses modal's tunneling
    proc = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--port", str(VLLM_PORT),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--max-model-len", "8192",
            "--trust-remote-code",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for server to be ready
    print(f"Starting vLLM server for {model_name}...")
    for _ in range(120):  # wait up to 2 min
        time.sleep(2)
        try:
            import urllib.request
            urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/health")
            print(f"vLLM server ready at localhost:{VLLM_PORT}")
            break
        except Exception:
            pass
    else:
        raise RuntimeError("vLLM server did not start in time")

    # Keep alive until timeout
    proc.wait()
    return f"http://localhost:{VLLM_PORT}/v1"


@app.local_entrypoint()
def main(model_name: str = "Qwen/Qwen3.5-0.8B-Instruct"):
    print(f"Serving {model_name} via Modal vLLM...")
    print("Use the Modal tunnel URL as --base_url in run_benchmarks.sh")
    serve_vllm.remote(model_name)
