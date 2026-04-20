"""
Step 6: Prepare MBPP training/validation data in verl parquet format for Flow-GRPO.

The training data format follows the existing data/get_train_data.py pattern:
    - 'data_source': dataset name
    - 'prompt': list of chat messages [{'role': 'user', 'content': ...}]
    - 'ability': task type
    - 'reward_model': {'style': 'rule', 'ground_truth': ...}
    - 'extra_info': {'split': 'train', 'index': ...}

Usage:
    python scripts/step6_flow_grpo_codegen/prepare_training_data.py \
        --output_dir /path/to/data/

Then upload:
    modal volume put ao-runs /path/to/data/ /runs/step6/data/
"""

import argparse
import json
import os
import random

import pandas as pd


SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Write clean, correct Python code to solve the given programming problem."
)

TASK_INSTRUCTION = (
    "Write a Python function to solve the following programming problem. "
    "Your answer must be valid, self-contained Python code enclosed in a ```python``` block."
)


def mbpp_to_verl_record(item: dict, split: str, index: int) -> dict:
    """Convert an MBPP item to verl training format."""
    prompt_text = f"{TASK_INSTRUCTION}\n\n{item.get('text') or item.get('prompt', '')}"
    test_code = "\n".join(
        f"assert {tc}" if not tc.strip().startswith("assert") else tc
        for tc in item["test_list"]
    )

    return {
        "data_source": "mbpp",
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        "ability": "code_generation",
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "answer": item["code"],
                "test_code": test_code,
                "task_id": item["task_id"],
            },
        },
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": item["task_id"],
        },
    }


def prepare(output_dir: str):
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)

    # Load MBPP sanitized split
    ds_train = load_dataset("mbpp", "sanitized", split="train", trust_remote_code=False)
    ds_val = load_dataset("mbpp", "sanitized", split="validation", trust_remote_code=False)
    ds_test = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=False)

    train_records = [mbpp_to_verl_record(item, "train", i) for i, item in enumerate(ds_train)]
    val_records = [mbpp_to_verl_record(item, "val", i) for i, item in enumerate(ds_val)]

    # Shuffle training data
    random.seed(42)
    random.shuffle(train_records)

    train_df = pd.DataFrame(train_records)
    val_df = pd.DataFrame(val_records)

    train_path = os.path.join(output_dir, "train", "mbpp_train.parquet")
    val_path = os.path.join(output_dir, "val", "mbpp_val.parquet")
    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    os.makedirs(os.path.dirname(val_path), exist_ok=True)

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train: {len(train_records)} records → {train_path}")
    print(f"Val:   {len(val_records)} records → {val_path}")
    print(f"\nUpload with:")
    print(f"  modal volume put ao-runs {output_dir}/ /runs/step6/data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="./step6_data")
    args = parser.parse_args()
    prepare(args.output_dir)
