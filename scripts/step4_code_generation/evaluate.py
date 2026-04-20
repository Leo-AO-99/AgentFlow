"""
Step 4: Evaluate AgentFlow code generation results.

Computes pass@1 by executing the model's output against each problem's test cases
in an isolated subprocess sandbox. No LLM judge needed.

Usage:
    python scripts/step4_code_generation/evaluate.py \
        --data_file scripts/step4_code_generation/data/humaneval_data.json \
        --result_dir results/humaneval/Qwen3.5-0.8B \
        --output_file results/humaneval/Qwen3.5-0.8B/scores.json
"""

import json
import os
import re
import subprocess
import sys
import argparse
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed


# ── Code extraction ────────────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract Python code block from model output."""
    # Try ```python ... ``` block first
    match = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fall back to the whole text
    return text.strip()


# ── Sandboxed execution ────────────────────────────────────────────────────────

def run_code_with_tests(code: str, test_code: str, entry_point: str | None, timeout: int = 10) -> bool:
    """
    Run `code` + `test_code` in a subprocess sandbox.
    Returns True if all assertions pass without exception.
    """
    full_code = code + "\n\n" + test_code + "\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(tmp_path)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_single(problem: dict, result_dir: str) -> dict:
    pid = problem["pid"]
    # result files are named output_<index>.json
    # We match by pid stored inside
    idx = None
    for fname in os.listdir(result_dir):
        if not fname.startswith("output_") or not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(result_dir, fname)) as f:
                data = json.load(f)
            if data.get("pid") == pid:
                idx = fname
                model_output = data.get("direct_output", "")
                break
        except Exception:
            continue

    if idx is None:
        return {"pid": pid, "pass": False, "reason": "no_result_file"}

    code = extract_code(model_output)
    test_code = problem.get("test_code", "")
    entry_point = problem.get("entry_point")
    passed = run_code_with_tests(code, test_code, entry_point)

    return {
        "pid": pid,
        "pass": passed,
        "result_file": idx,
    }


def evaluate(data_file: str, result_dir: str, output_file: str, workers: int = 8):
    with open(data_file) as f:
        problems = json.load(f)

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(evaluate_single, p, result_dir): p for p in problems}
        for fut in as_completed(futures):
            results.append(fut.result())

    total = len(results)
    passed = sum(1 for r in results if r["pass"])
    pass_at_1 = passed / total if total > 0 else 0.0

    summary = {
        "pass@1": round(pass_at_1 * 100, 2),
        "passed": passed,
        "total": total,
        "results": sorted(results, key=lambda x: x["pid"]),
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"pass@1 = {summary['pass@1']}%  ({passed}/{total})")
    print(f"Results saved to: {output_file}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.data_file, args.result_dir, args.output_file, args.workers)
