"""Test exact-output stability of the configured ALFWorld agent model."""

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

DEFAULT_OUTPUT = Path("agent_inference_stability.json")
TEST_OBSERVATION = """You are in the middle of a room. Looking quickly around you, you see a cabinet 1, a countertop 1, a drawer 1, a fridge 1, and a microwave 1.

Your task is to: put an apple in fridge."""
TEST_TASK_NAME = "pick_and_place_simple"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model")
    parser.add_argument(
        "--few-shot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument(
        "--fail-on-variation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


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
    if args.temperature < 0:
        raise SystemExit("--temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")

    from litellm import completion

    from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
    from ProcedureMem.alfworld_agent import build_messages, resolve_litellm_model
    from ProcedureMem.runtime_config import (
        DEFAULT_EXAMPLES_PATH,
        configure_runtime,
        load_environment,
    )

    env_path = load_environment(args.env_file)
    settings = configure_runtime(model_name=args.model, require_llm=True)
    api_base = settings.api_base_url
    routed_model = resolve_litellm_model(settings.model_name, api_base)
    examples = json.loads(DEFAULT_EXAMPLES_PATH.read_text(encoding="utf-8"))
    messages = build_messages(
        TEST_OBSERVATION,
        TEST_TASK_NAME,
        system_prompt=alfworld_system_prompt,
        few_shot=args.few_shot,
        examples=examples,
    )

    def infer_once() -> str:
        request = {
            "model": routed_model,
            "messages": messages,
            "api_key": settings.api_key,
            "num_retries": 10,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
        }
        if api_base:
            request["api_base"] = api_base
        response = completion(**request)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Agent model returned an empty response")
        return content

    print(f"Environment: {env_path}")
    print(f"Agent model: {settings.model_name}")
    print(f"Routed model: {routed_model}")
    print(f"API base: {api_base}")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Seed: {args.seed}")
    print(f"Few-shot: {args.few_shot}")
    print(f"Sequential requests: {args.calls}")
    sequential_outputs = [infer_once() for _ in range(args.calls)]

    worker_count = min(args.workers, args.calls)
    print(f"Concurrent requests: {args.calls} (workers={worker_count})")
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        concurrent_outputs = list(
            executor.map(lambda _: infer_once(), range(args.calls))
        )

    sequential = _summarize(sequential_outputs)
    concurrent = _summarize(concurrent_outputs)
    overall = _summarize([*sequential_outputs, *concurrent_outputs])
    stable = overall["unique_output_count"] == 1

    artifact = {
        "schema_version": 1,
        "test_type": "alfworld_agent_exact_output_stability",
        "configuration": {
            "model": settings.model_name,
            "routed_model": routed_model,
            "api_base_url": api_base,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "few_shot": args.few_shot,
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
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Sequential unique outputs: {sequential['unique_output_count']}")
    print(f"Concurrent unique outputs: {concurrent['unique_output_count']}")
    print(f"Overall unique outputs: {overall['unique_output_count']}")
    print(f"Stable: {stable}")
    print(f"Artifact: {output_path}")
    return 1 if args.fail_on_variation and not stable else 0


if __name__ == "__main__":
    raise SystemExit(main())
