"""Test exact-output stability of the configured workflow-memory builder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT = Path("memory_builder_stability.json")
TEST_QUERY = "heat an apple and put it in the fridge"
TEST_TRAJECTORY = [
    {
        "from": "human",
        "value": "Your task is to: heat an apple and put it in the fridge.",
    },
    {
        "from": "gpt",
        "value": "Thought: I need to find the apple.\nAction: go to countertop 1",
    },
    {
        "from": "human",
        "value": "Observation: On countertop 1, you see an apple 1.",
    },
    {
        "from": "gpt",
        "value": "Thought: I found it.\nAction: take apple 1 from countertop 1",
    },
    {
        "from": "human",
        "value": "Observation: You pick up the apple 1.",
    },
    {
        "from": "gpt",
        "value": "Thought: Heat it next.\nAction: heat apple 1 with microwave 1",
    },
    {
        "from": "human",
        "value": "Observation: You heat the apple 1.",
    },
    {
        "from": "gpt",
        "value": "Thought: Put it in the fridge.\nAction: move apple 1 to fridge 1",
    },
    {"from": "human", "value": "Observation: Task accomplished."},
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calls",
        type=int,
        default=10,
        help="Number of requests in each of the sequential and concurrent phases.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Maximum concurrent builder requests.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSON artifact containing hashes and complete outputs.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        help="Optional dotenv file; defaults to the repository .env.",
    )
    parser.add_argument(
        "--fail-on-variation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return exit code 1 if any exact-output variation is detected.",
    )
    return parser


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _summarize(outputs: Sequence[str]) -> dict[str, object]:
    hashes = [_sha256(output) for output in outputs]
    return {
        "request_count": len(outputs),
        "unique_output_count": len(set(hashes)),
        "hash_counts": dict(sorted(Counter(hashes).items())),
        "hashes": hashes,
        "outputs": list(outputs),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calls < 1:
        raise SystemExit("--calls must be at least 1")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")

    from ProcedureMem.Alfworld.memory_prompts import (
        generate_workflow_from_trajectory_prompt,
    )
    from ProcedureMem.llm_api import (
        get_llm_response,
        resolve_memory_build_seed,
        resolve_memory_build_temperature,
    )
    from ProcedureMem.runtime_config import load_environment

    env_path = load_environment(args.env_file)
    model = _required_env("MEMORY_BUILD_MODEL_NAME")
    api_key = _required_env("MEMORY_BUILD_API_KEY")
    api_base_url = os.getenv("MEMORY_BUILD_API_BASE_URL") or None
    temperature = resolve_memory_build_temperature()
    seed = resolve_memory_build_seed()
    messages = generate_workflow_from_trajectory_prompt(
        TEST_QUERY, TEST_TRAJECTORY
    )

    def build_once() -> str:
        return get_llm_response(
            messages,
            is_string=True,
            model=model,
            api_key=api_key,
            api_base_url=api_base_url,
            temperature=temperature,
            seed=seed,
        )

    print(f"Environment: {env_path}")
    print(f"Builder model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Seed: {seed}")
    print(f"Sequential requests: {args.calls}")
    sequential_outputs = [build_once() for _ in range(args.calls)]

    worker_count = min(args.workers, args.calls)
    print(f"Concurrent requests: {args.calls} (workers={worker_count})")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        concurrent_outputs = list(executor.map(lambda _: build_once(), range(args.calls)))

    sequential = _summarize(sequential_outputs)
    concurrent = _summarize(concurrent_outputs)
    overall = _summarize([*sequential_outputs, *concurrent_outputs])
    stable = overall["unique_output_count"] == 1

    artifact = {
        "schema_version": 1,
        "test_type": "memory_builder_exact_output_stability",
        "configuration": {
            "model": model,
            "api_base_url": api_base_url,
            "temperature": temperature,
            "seed": seed,
            "enable_thinking": os.getenv("MEMORY_BUILD_ENABLE_THINKING"),
            "calls_per_phase": args.calls,
            "concurrent_workers": worker_count,
        },
        "prompt_sha256": _sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True)
        ),
        "sequential": sequential,
        "concurrent": concurrent,
        "overall_unique_output_count": overall["unique_output_count"],
        "overall_hash_counts": overall["hash_counts"],
        "stable": stable,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Sequential unique outputs: {sequential['unique_output_count']}")
    print(f"Concurrent unique outputs: {concurrent['unique_output_count']}")
    print(f"Overall unique outputs: {overall['unique_output_count']}")
    print(f"Stable: {stable}")
    print(f"Artifact: {output_path}")
    return 1 if args.fail_on_variation and not stable else 0


if __name__ == "__main__":
    raise SystemExit(main())
