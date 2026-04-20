"""
Step 4: AgentFlow solver for code generation benchmarks (HumanEval / MBPP).

Wraps test/solve.py with code-generation-specific defaults.

Usage:
    python scripts/step4_code_generation/solve.py \
        --index 0 \
        --data_file scripts/step4_code_generation/data/humaneval_data.json \
        --llm_engine_name "vllm-Qwen/Qwen3.5-0.8B-Instruct" \
        --base_url "http://localhost:8000/v1" \
        --output_json_dir results/humaneval/Qwen3.5-0.8B
"""

import os
import sys
import argparse

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import test/solve.py's main logic
TEST_DIR = os.path.join(PROJECT_ROOT, "test")
sys.path.insert(0, TEST_DIR)

from solve import parse_arguments as _parse_base_args, main as _base_main  # noqa: E402


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="AgentFlow code generation solver (HumanEval / MBPP)",
        parents=[],
        add_help=True,
    )
    # Mirror all args from test/solve.py but with code-gen defaults
    parser.add_argument("--llm_engine_name", default="gpt-4o-mini")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--run_baseline_only", type=bool, default=False)
    parser.add_argument("--task", default="code_generation")
    parser.add_argument(
        "--data_file",
        default=os.path.join(os.path.dirname(__file__), "data", "humaneval_data.json"),
    )
    parser.add_argument("--task_description", default=(
        "Write a Python function to solve the following programming problem. "
        "Your answer must be valid, self-contained Python code."
    ))
    parser.add_argument("--output_types", default="direct")
    # Code generation: Python Coder is the primary tool
    parser.add_argument(
        "--enabled_tools",
        default="Python_Coder_Tool,Base_Generator_Tool",
    )
    parser.add_argument("--tool_engine", default="Default,Default")
    parser.add_argument(
        "--model_engine",
        default="trainable,gpt-4o-mini,gpt-4o-mini,gpt-4o-mini",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--root_cache_dir", default="solver_cache")
    parser.add_argument("--output_json_dir", default="results/code_generation")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--max_time", type=int, default=120)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vllm_config_path", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--check_model", type=bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    # Change to test dir so test/solve.py's relative imports work
    os.chdir(TEST_DIR)
    args = parse_arguments()
    # Reuse base main directly
    _base_main(args)
