"""
Step 4: Prepare HumanEval and MBPP datasets in AgentFlow JSON format.

Output format (compatible with test/solve.py):
    [
      {
        "pid": "HumanEval/0",
        "query": "<prompt>",
        "answer": "<canonical solution or function signature>",
        "test_code": "<test assertions>",
        "entry_point": "<function name>"
      },
      ...
    ]

Usage:
    pip install datasets human-eval
    python scripts/step4_code_generation/data/prepare_data.py
"""

import json
import os

OUT_DIR = os.path.join(os.path.dirname(__file__))


def prepare_humaneval():
    from datasets import load_dataset

    ds = load_dataset("openai_humaneval", split="test", trust_remote_code=True)
    records = []
    for item in ds:
        records.append({
            "pid": item["task_id"],
            "query": (
                "Write a Python function to solve the following problem.\n\n"
                + item["prompt"]
            ),
            "answer": item["canonical_solution"],
            "test_code": item["test"],
            "entry_point": item["entry_point"],
        })
    out_path = os.path.join(OUT_DIR, "humaneval_data.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"HumanEval: {len(records)} problems → {out_path}")
    return out_path


def prepare_mbpp():
    from datasets import load_dataset

    # Use the sanitized split (cleaner test cases)
    ds = load_dataset("mbpp", "sanitized", split="test", trust_remote_code=True)
    records = []
    for item in ds:
        test_code = "\n".join(
            f"assert {tc}" if not tc.strip().startswith("assert") else tc
            for tc in item["test_list"]
        )
        records.append({
            "pid": f"MBPP/{item['task_id']}",
            "query": (
                "Write a Python function to solve the following problem.\n\n"
                + item["prompt"]
            ),
            "answer": item["code"],
            "test_code": test_code,
            "entry_point": None,  # MBPP doesn't always have a single entry point
        })
    out_path = os.path.join(OUT_DIR, "mbpp_data.json")
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"MBPP sanitized: {len(records)} problems → {out_path}")
    return out_path


if __name__ == "__main__":
    prepare_humaneval()
    prepare_mbpp()
    print("Done. Data files written to:", OUT_DIR)
