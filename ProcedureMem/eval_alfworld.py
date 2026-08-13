"""Run paired no-memory or workflow-memory ALFWorld evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.alfworld_agent import resolve_litellm_model, run_alfworld_batch
from ProcedureMem.alfworld_experiment import (
    EVAL_CONDITIONS,
    SPLIT_NAMES,
    build_task_manifest,
    inject_memory,
    inject_trajectories,
    load_json,
    manifest_sha256,
    maybe_write_paired_comparison,
    maybe_write_memory_rerank_comparison,
    reranked_retrieval_records,
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
    DEFAULT_TRAJECTORY_PATH,
    configure_runtime,
    load_alfworld_config,
    load_memory_config,
)
from ProcedureMem.build_edge_subsets import DEFAULT_OUTPUT as DEFAULT_EDGE_SUBSET_MANIFEST
from ProcedureMem.benchmark_config import candidate_score_threshold
from ProcedureMem.reranker import DEFAULT_MODEL, OpenMemReranker


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=EVAL_CONDITIONS, required=True)
    parser.add_argument("--condition-name")
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
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--rerank-candidate-k", type=int, default=20)
    parser.add_argument("--rerank-top-n", type=int, default=10)
    parser.add_argument("--rerank-model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank-timeout", type=float)
    parser.add_argument("--candidate-score-threshold", type=float)
    parser.add_argument("--model")
    parser.add_argument("--memory-build-model")
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    parser.add_argument("--edge-capacity", type=int)
    parser.add_argument(
        "--edge-subset-manifest", type=Path, default=DEFAULT_EDGE_SUBSET_MANIFEST
    )
    parser.add_argument("--edge-memory-dir", type=Path)
    parser.add_argument("--trajectory-file", type=Path, default=DEFAULT_TRAJECTORY_PATH)
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
    if args.condition_name and Path(args.condition_name).name != args.condition_name:
        parser.error("--condition-name must be a name, not a path")
    if args.score_threshold is not None and args.score_threshold < 0:
        parser.error("--score-threshold must be non-negative")
    if args.condition == "edge_raw" and args.edge_capacity is None:
        parser.error("--edge-capacity is required for --condition edge_raw")
    if args.condition != "edge_raw" and args.edge_capacity is not None:
        parser.error("--edge-capacity is only valid for --condition edge_raw")
    if args.condition != "edge_raw" and args.score_threshold is not None:
        parser.error("--score-threshold is only valid for --condition edge_raw")
    if args.condition == "memory_rerank":
        if args.rerank_candidate_k < 1 or args.rerank_top_n < 1:
            parser.error("rerank candidate-k and top-n must be at least 1")
        if args.rerank_top_n > args.rerank_candidate_k:
            parser.error("--rerank-top-n cannot exceed --rerank-candidate-k")
        if args.rerank_model != DEFAULT_MODEL:
            parser.error(f"--rerank-model must be {DEFAULT_MODEL}")
        if args.rerank_timeout is not None and args.rerank_timeout <= 0:
            parser.error("--rerank-timeout must be positive")
        try:
            args.candidate_score_threshold = candidate_score_threshold(
                args.candidate_score_threshold
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.candidate_score_threshold is not None:
        parser.error(
            "--candidate-score-threshold is only valid for --condition memory_rerank"
        )


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
    config["retrieve_num"] = (
        args.rerank_top_n if args.condition == "memory_rerank" else args.top_k
    )
    config["build_model"] = args.memory_build_model
    config["is_cold_start"] = True
    return Memory(**config)


def _memory_identity(record: dict[str, Any]) -> str:
    payload = f"{record.get('task_name')}\0{record.get('workflow')}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_edge_memory(args: argparse.Namespace):
    from ProcedureMem.edge_memory import RawTrajectoryMemory

    memory_dir = args.edge_memory_dir or (
        Path(__file__).resolve().parent
        / "memory"
        / "alfworld"
        / f"edge_raw_{args.edge_capacity}"
    )
    return RawTrajectoryMemory(
        trajectory_file=args.trajectory_file,
        subset_manifest=args.edge_subset_manifest,
        capacity=args.edge_capacity,
        memory_dir=memory_dir,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
    )


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
        require_embedding=args.condition in {"memory", "memory_rerank", "edge_raw"},
    )
    llm, routed_model = _make_llm(settings.model_name, args.temperature, manifest["seed"])
    if args.condition in {"memory", "memory_rerank"}:
        memory = _load_memory(args)
    elif args.condition == "edge_raw":
        memory = _load_edge_memory(args)
    else:
        memory = None
    reranker = (
        OpenMemReranker(model=args.rerank_model, timeout=args.rerank_timeout)
        if args.condition == "memory_rerank"
        else None
    )

    condition_name = args.condition_name or (
        f"edge_raw_{args.edge_capacity}"
        if args.condition == "edge_raw"
        else args.condition
    )

    experiment_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else DEFAULT_RESULTS_DIR / "paired" / args.experiment_name
    )
    condition_dir = experiment_dir / condition_name
    _prepare_output(condition_dir, args.overwrite)

    parameters = {
        "model": settings.model_name,
        "routed_model": routed_model,
        "agent_api_base_url": settings.api_base_url,
        "embedding_model": settings.embedding_model,
        "split": args.split,
        "condition": condition_name,
        "condition_mode": args.condition,
        "seed": manifest["seed"],
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "temperature": args.temperature,
        "top_p": 1.0,
        "few_shot": args.few_shot,
        "top_k": (
            args.rerank_top_n if args.condition == "memory_rerank" else args.top_k
        ),
        "score_threshold": args.score_threshold,
        "memory_type": (
            "raw_trajectory"
            if args.condition == "edge_raw"
            else "workflow"
            if args.condition in {"memory", "memory_rerank"}
            else None
        ),
        "retrieval_pipeline": (
            "faiss_then_openmem_rerank"
            if args.condition == "memory_rerank"
            else "faiss_similarity"
            if args.condition == "memory"
            else None
        ),
        "memory_config": (
            str(Path(args.memory_config).resolve())
            if args.condition in {"memory", "memory_rerank"}
            else None
        ),
        "memory_build_model": (
            args.memory_build_model
            if args.condition in {"memory", "memory_rerank"}
            else None
        ),
        "rerank_model": args.rerank_model if args.condition == "memory_rerank" else None,
        "rerank_candidate_k": (
            args.rerank_candidate_k if args.condition == "memory_rerank" else None
        ),
        "rerank_top_n": (
            args.rerank_top_n if args.condition == "memory_rerank" else None
        ),
        "rerank_timeout": (
            reranker.timeout if args.condition == "memory_rerank" else None
        ),
        "rerank_candidate_score_threshold": (
            args.candidate_score_threshold
            if args.condition == "memory_rerank"
            else None
        ),
        "rerank_base_url": (
            reranker.base_url if args.condition == "memory_rerank" else None
        ),
        "edge_capacity": args.edge_capacity,
        "edge_subset_manifest": (
            str(args.edge_subset_manifest.resolve())
            if args.condition == "edge_raw"
            else None
        ),
        "edge_memory_dir": (
            str(memory.memory_dir) if args.condition == "edge_raw" else None
        ),
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
            rerank_by_task: list[dict[str, Any] | None] = [
                None for _ in observations
            ]
            retrieval_by_task: list[dict[str, Any] | None] = [
                None for _ in observations
            ]
            if memory is not None:
                for index, observation in enumerate(observations):
                    query = task_query(observation)
                    if args.condition == "memory_rerank":
                        rerank_output = memory.retrieve_with_rerank(
                            query,
                            reranker=reranker,
                            candidate_k=args.rerank_candidate_k,
                            top_n=args.rerank_top_n,
                            score_threshold=args.candidate_score_threshold,
                        )
                        retrieved = reranked_retrieval_records(
                            rerank_output["items"]
                        )
                        vector_top_n = retrieval_records(
                            rerank_output["candidates"][: args.rerank_top_n]
                        )
                        vector_ids = [_memory_identity(item) for item in vector_top_n]
                        rerank_ids = [_memory_identity(item) for item in retrieved]
                        overlap = len(set(vector_ids) & set(rerank_ids))
                        rerank_by_task[index] = {
                            "actual_candidate_count": len(rerank_output["candidates"]),
                            "candidate_search_latency_ms": rerank_output[
                                "candidate_search_latency_ms"
                            ],
                            "rerank_api_latency_ms": rerank_output[
                                "rerank_api_latency_ms"
                            ],
                            "rerank_pipeline_latency_ms": rerank_output[
                                "rerank_pipeline_latency_ms"
                            ],
                            "rerank_changed_top1": bool(
                                vector_ids and rerank_ids and vector_ids[0] != rerank_ids[0]
                            ),
                            "rerank_top_n_overlap_count": overlap,
                            "rerank_top_n_overlap_rate": overlap
                            / max(1, len(vector_ids)),
                            "rerank_request_id": rerank_output["request_id"],
                            "rerank_prompt_tokens": rerank_output["prompt_tokens"],
                            "rerank_total_tokens": rerank_output["total_tokens"],
                            "vector_top_n": vector_top_n,
                        }
                    else:
                        retrieval_started = time.perf_counter()
                        retrieved = retrieval_records(memory.retrieve(query))
                        if args.condition == "memory":
                            retrieval_by_task[index] = {
                                "similarity_search_latency_ms": (
                                    time.perf_counter() - retrieval_started
                                )
                                * 1000.0
                            }
                    if (
                        args.condition == "edge_raw"
                        and args.score_threshold is None
                        and not retrieved
                    ):
                        raise RuntimeError(
                            "Edge retrieval returned no trajectory with threshold disabled"
                        )
                    retrieved_by_task[index] = retrieved
                    if args.condition == "edge_raw":
                        observations[index] = inject_trajectories(observation, retrieved)
                    else:
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
            retrieved_records = retrieved_by_task[local_index]
            top1_record = retrieved_records[0] if retrieved_records else {}
            rerank_record = rerank_by_task[local_index]
            retrieval_record = retrieval_by_task[local_index]
            result_fields = {
                    "schema_version": 1,
                    "experiment_name": args.experiment_name,
                    "task_id": task_id,
                    "task_index": offset + local_index,
                    "task_type": _task_name(task_id).split("/", 1)[0],
                    "split": args.split,
                    "condition": condition_name,
                    "condition_mode": args.condition,
                    "edge_capacity": args.edge_capacity,
                    "retrieved_count": len(retrieved_records),
                    "top1_raw_score": top1_record.get("raw_score"),
                    "top1_trajectory_index": top1_record.get("trajectory_index"),
                    "retrieved_memories": retrieved_records,
                    "model": settings.model_name,
                    "parameters": parameters,
                }
            if rerank_record is not None:
                result_fields["rerank"] = rerank_record
            if retrieval_record is not None:
                result_fields["retrieval"] = retrieval_record
            result.update(result_fields)
            results.append(result)
        print(f"Completed {len(results)}/{len(selected_gamefiles)} tasks")

    rerank_metadata = None
    if args.condition == "memory_rerank":
        rerank_rows = [result["rerank"] for result in results]
        rerank_metadata = {
            "candidate_count_mean": statistics.fmean(
                row["actual_candidate_count"] for row in rerank_rows
            ),
            "candidate_count_min": min(
                row["actual_candidate_count"] for row in rerank_rows
            ),
            "candidate_count_max": max(
                row["actual_candidate_count"] for row in rerank_rows
            ),
            "rerank_changed_top1_count": sum(
                row["rerank_changed_top1"] for row in rerank_rows
            ),
            "rerank_top_n_overlap_rate_mean": statistics.fmean(
                row["rerank_top_n_overlap_rate"] for row in rerank_rows
            ),
            "candidate_search_latency_ms_mean": statistics.fmean(
                row["candidate_search_latency_ms"] for row in rerank_rows
            ),
            "rerank_api_latency_ms_mean": statistics.fmean(
                row["rerank_api_latency_ms"] for row in rerank_rows
            ),
            "rerank_pipeline_latency_ms_mean": statistics.fmean(
                row["rerank_pipeline_latency_ms"] for row in rerank_rows
            ),
        }
    retrieval_metadata = None
    if args.condition == "memory":
        retrieval_metadata = {
            "similarity_search_latency_ms_mean": statistics.fmean(
                result["retrieval"]["similarity_search_latency_ms"]
                for result in results
            )
        }
    summary = write_results(
        condition_dir,
        results,
        summary_metadata={
            "experiment_name": args.experiment_name,
            "model": settings.model_name,
            "parameters": parameters,
            "rerank_summary": rerank_metadata,
            "retrieval_summary": retrieval_metadata,
        },
    )
    print(
        f"{condition_name}: SR={summary['success_rate']:.4f} "
        f"({summary['success_count']}/{summary['task_count']})"
    )
    comparison = maybe_write_paired_comparison(experiment_dir)
    if comparison:
        print(
            "Paired comparison: "
            f"{comparison['absolute_improvement_percentage_points']:+.2f} percentage points"
        )
    rerank_comparison = maybe_write_memory_rerank_comparison(experiment_dir)
    if rerank_comparison:
        print(
            "Memory vs rerank: "
            f"{rerank_comparison['absolute_improvement_percentage_points']:+.2f} "
            "percentage points; "
            f"flips +{rerank_comparison['failure_to_success']} "
            f"/-{rerank_comparison['success_to_failure']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
