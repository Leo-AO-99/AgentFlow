import modal

image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-cudnn-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTHONUNBUFFERED": "1"})
    .pip_install("torch==2.11.0")
    .pip_install(["peft==0.18.1", "transformers==5.4.0", "trl==0.29.1", "math_verify==0.9.0", "wandb"])
    # .add_local_file("/Users/liao/LeoGit/AgentFlow/files/test_qwen.py", "/root/test_qwen.py", copy=True)
)

app = modal.App("ao-test-qwen")
hf = modal.Volume.from_name("ao-hf", version=2, create_if_missing=True)
runs = modal.Volume.from_name("ao-runs", create_if_missing=True)
hf_secret = modal.Secret.from_name("Ao-secret")

@app.function(
    image=image, gpu="L40S",
    volumes={"/root/.cache/huggingface": hf, "/runs": runs},
    secrets=[hf_secret], timeout=7200)
def run(code_path: str):
    import subprocess
    import os
    
    # reload volume to see uploaded files
    runs.reload()
    
    # debug: list files to confirm
    for root, dirs, files in os.walk("/runs"):
        for f in files:
            print(os.path.join(root, f))
    
    subprocess.run(["python", code_path])