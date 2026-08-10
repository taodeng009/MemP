"""Run paired no-memory or workflow-memory ALFWorld evaluations."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Sequence

from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.alfworld_agent import resolve_litellm_model, run_alfworld_batch
from ProcedureMem.alfworld_experiment import (
    CONDITIONS,
    SPLIT_NAMES,
    build_task_manifest,
    inject_memory,
    load_json,
    manifest_sha256,
    maybe_write_paired_comparison,
    retrieval_records,
    task_id_from_gamefile,
    task_query,
    validate_task_manifest,
    write_json,
    write_results,
)
from ProcedureMem.runtime_config import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MEMORY_CONFIG,
    DEFAULT_RESULTS_DIR,
    configure_runtime,
    load_alfworld_config,
    load_memory_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--split", choices=tuple(SPLIT_NAMES), default="valid_unseen")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-manifest", type=Path)
    parser.add_argument("--limit-tasks", type=int)
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--create-manifest-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--few-shot", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model")
    parser.add_argument("--memory-build-model")
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    parser.add_argument("--alfworld-data")
    parser.add_argument("--config", default=str(DEFAULT_ALFWORLD_CONFIG))
    parser.add_argument("--experiment-name", default="alfworld_paired")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("batch_size", "max_steps", "top_k"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.limit_tasks is not None and args.limit_tasks < 1:
        parser.error("--limit-tasks must be at least 1")
    if not args.experiment_name.strip():
        parser.error("--experiment-name cannot be empty")
    if Path(args.experiment_name).name != args.experiment_name:
        parser.error("--experiment-name must be a name, not a path")


def _default_manifest_path(split: str, seed: int, limit_tasks: int | None) -> Path:
    count = str(limit_tasks) if limit_tasks is not None else "all"
    return DEFAULT_RESULTS_DIR / "manifests" / f"{split}_seed{seed}_n{count}.json"


def _make_llm(model: str, temperature: float, seed: int):
    from litellm import completion

    api_base = os.getenv("OPENAI_API_BASE")
    routed_model = resolve_litellm_model(model, api_base)

    def call(messages: list[dict[str, str]]) -> str:
        kwargs: dict[str, Any] = {
            "model": routed_model,
            "messages": messages,
            "api_key": os.environ["OPENAI_API_KEY"],
            "num_retries": 10,
            "temperature": temperature,
            "seed": seed,
        }
        if api_base:
            kwargs["api_base"] = api_base
        response = completion(**kwargs)
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM returned an empty response")
        return content

    return call, routed_model


def _initial_observation(observation: str) -> str:
    blocks = observation.split("\n\n")
    return "\n".join(blocks[1:]) if len(blocks) > 1 else observation


def _load_memory(args: argparse.Namespace):
    from ProcedureMem.memory import Memory

    config = load_memory_config(args.memory_config)
    config["retrieve_num"] = args.top_k
    config["build_model"] = args.memory_build_model
    config["is_cold_start"] = True
    return Memory(**config)


def _task_name(task_id: str) -> str:
    parts = Path(task_id).parts
    return "/".join(parts[-3:-1])


def _prepare_output(directory: Path, overwrite: bool) -> None:
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {directory}. Use a new "
                "--experiment-name or pass --overwrite."
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    settings = configure_runtime(alfworld_data=args.alfworld_data)
    config = load_alfworld_config(args.config)
    config["general"]["random_seed"] = args.seed
    random.seed(args.seed)
    import numpy as np

    np.random.seed(args.seed)

    from alfworld.agents.environment import get_environment

    split_name = SPLIT_NAMES[args.split]
    source_env = get_environment(config["env"]["type"])(config, train_eval=split_name)
    available_gamefiles = list(source_env.game_files)
    manifest_path = args.task_manifest or _default_manifest_path(
        args.split, args.seed, args.limit_tasks
    )

    if manifest_path.is_file():
        manifest = load_json(manifest_path)
        if manifest.get("seed") != args.seed:
            raise ValueError("Existing manifest seed differs from --seed")
        if args.limit_tasks is not None and manifest.get("task_count") != args.limit_tasks:
            raise ValueError("Existing manifest task_count differs from --limit-tasks")
        if args.task_ids and [task["task_id"] for task in manifest["tasks"]] != args.task_ids:
            raise ValueError("Existing manifest tasks differ from --task-ids")
    else:
        manifest = build_task_manifest(
            available_gamefiles,
            data_root=settings.alfworld_data,
            split=args.split,
            seed=args.seed,
            limit_tasks=args.limit_tasks,
            task_ids=args.task_ids,
        )
        write_json(manifest_path, manifest)

    selected_gamefiles = validate_task_manifest(
        manifest,
        split=args.split,
        available_gamefiles=available_gamefiles,
        data_root=settings.alfworld_data,
    )
    print(f"Task manifest: {manifest_path.resolve()} ({len(selected_gamefiles)} tasks)")
    if args.create_manifest_only:
        return 0

    settings = configure_runtime(
        model_name=args.model,
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=args.condition == "memory",
    )
    llm, routed_model = _make_llm(settings.model_name, args.temperature, manifest["seed"])
    memory = _load_memory(args) if args.condition == "memory" else None

    experiment_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else DEFAULT_RESULTS_DIR / "paired" / args.experiment_name
    )
    condition_dir = experiment_dir / args.condition
    _prepare_output(condition_dir, args.overwrite)

    parameters = {
        "model": settings.model_name,
        "routed_model": routed_model,
        "agent_api_base_url": settings.api_base_url,
        "embedding_model": settings.embedding_model,
        "split": args.split,
        "condition": args.condition,
        "seed": manifest["seed"],
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "top_p": 1.0,
        "few_shot": args.few_shot,
        "top_k": args.top_k,
        "memory_config": str(Path(args.memory_config).resolve()),
        "memory_build_model": args.memory_build_model,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256(manifest),
    }
    write_json(condition_dir / "experiment.json", parameters)
    examples = json.loads(DEFAULT_EXAMPLES_PATH.read_text(encoding="utf-8"))

    results: list[dict[str, Any]] = []
    for offset in range(0, len(selected_gamefiles), args.batch_size):
        chunk = selected_gamefiles[offset : offset + args.batch_size]
        expected_ids = [
            task["task_id"] for task in manifest["tasks"][offset : offset + len(chunk)]
        ]
        source_env.game_files = list(chunk)
        source_env.num_games = len(chunk)
        env = source_env.init_env(batch_size=len(chunk))
        try:
            observations, info = env.reset()
            actual_ids = [
                task_id_from_gamefile(gamefile, settings.alfworld_data)
                for gamefile in info["extra.gamefile"]
            ]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "ALFWorld reset order differs from task manifest: "
                    f"expected {expected_ids}, got {actual_ids}"
                )

            observations = [
                _initial_observation(observation) for observation in observations
            ]
            retrieved_by_task: list[list[dict[str, Any]]] = [
                [] for _ in observations
            ]
            if memory is not None:
                for index, observation in enumerate(observations):
                    retrieved = retrieval_records(
                        memory.retrieve(task_query(observation))
                    )
                    retrieved_by_task[index] = retrieved
                    observations[index] = inject_memory(observation, retrieved)

            batch_results = run_alfworld_batch(
                env=env,
                observations=observations,
                names=[_task_name(task_id) for task_id in actual_ids],
                llm_fn=llm,
                system_prompt=alfworld_system_prompt,
                few_shot=args.few_shot,
                max_steps=args.max_steps,
                examples=examples,
            )
        finally:
            if hasattr(env, "close"):
                env.close()

        for local_index, (task_id, result) in enumerate(zip(actual_ids, batch_results)):
            result.update(
                {
                    "schema_version": 1,
                    "experiment_name": args.experiment_name,
                    "task_id": task_id,
                    "task_index": offset + local_index,
                    "task_type": _task_name(task_id).split("/", 1)[0],
                    "split": args.split,
                    "condition": args.condition,
                    "retrieved_memories": retrieved_by_task[local_index],
                    "model": settings.model_name,
                    "parameters": parameters,
                }
            )
            results.append(result)
        print(f"Completed {len(results)}/{len(selected_gamefiles)} tasks")

    summary = write_results(
        condition_dir,
        results,
        summary_metadata={
            "experiment_name": args.experiment_name,
            "model": settings.model_name,
            "parameters": parameters,
        },
    )
    print(
        f"{args.condition}: SR={summary['success_rate']:.4f} "
        f"({summary['success_count']}/{summary['task_count']})"
    )
    comparison = maybe_write_paired_comparison(experiment_dir)
    if comparison:
        print(
            "Paired comparison: "
            f"{comparison['absolute_improvement_percentage_points']:+.2f} percentage points"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
