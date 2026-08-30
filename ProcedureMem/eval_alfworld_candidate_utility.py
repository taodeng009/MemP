"""Evaluate single-memory downstream utility from a frozen online snapshot."""

from __future__ import annotations

import argparse
import copy
import json
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ProcedureMem.alfworld_agent import run_alfworld_batch
from ProcedureMem.alfworld_experiment import (
    SPLIT_NAMES,
    inject_memory,
    retrieval_records,
    task_id_from_gamefile,
    task_query,
    validate_task_manifest,
    write_json,
)
from ProcedureMem.candidate_utility import (
    condition_memory_ids,
    coverage_proxy_scores,
    explicit_selection,
    load_snapshot,
    load_workflow_cache,
    stratified_proxy_selection,
    summarize_candidate_utility,
    validate_workflow_cache,
    write_jsonl,
    write_task_manifest,
    write_utility_csv,
)
from ProcedureMem.cloud_scheduling import (
    CandidateMemory,
    ScheduledWorkflowMemory,
    load_cached_embedding,
)
from ProcedureMem.eval_alfworld import _initial_observation, _make_llm, _task_name
from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.runtime_config import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MEMORY_CONFIG,
    configure_runtime,
    load_alfworld_config,
    load_memory_config,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-interval", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-workflow-cache", type=Path)
    parser.add_argument("--build-candidate-workflows", action="store_true")
    parser.add_argument("--candidate-count", type=int, default=6)
    parser.add_argument("--candidate-queue-ids", nargs="+")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--score-threshold", type=float)
    parser.add_argument("--model")
    parser.add_argument("--memory-build-model")
    parser.add_argument("--memory-build-temperature", type=float)
    parser.add_argument("--memory-build-seed", type=int)
    parser.add_argument("--memory-build-top-k", type=int)
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    parser.add_argument("--alfworld-data")
    parser.add_argument("--config", default=str(DEFAULT_ALFWORLD_CONFIG))
    return parser


def _resolve_settings(args: argparse.Namespace, snapshot: Mapping[str, Any]) -> dict[str, Any]:
    source = snapshot["source_parameters"]
    def source_value(name: str, default: Any) -> Any:
        value = source.get(name)
        return default if value is None else value

    values = {
        "model": args.model or source.get("model"),
        "batch_size": (
            args.batch_size
            if args.batch_size is not None
            else int(source_value("batch_size", 1))
        ),
        "max_steps": (
            args.max_steps
            if args.max_steps is not None
            else int(source_value("max_steps", 30))
        ),
        "temperature": (
            args.temperature
            if args.temperature is not None
            else float(source_value("temperature", 0.0))
        ),
        "top_k": (
            args.top_k
            if args.top_k is not None
            else int(source_value("top_k", 3))
        ),
        "score_threshold": (
            args.score_threshold
            if args.score_threshold is not None
            else float(source_value("score_threshold", 0.5))
        ),
        "few_shot": bool(source.get("few_shot", True)),
        "seed": int(source.get("seed", 42)),
        "split": source.get("split"),
        "memory_build_model": (
            args.memory_build_model or source.get("memory_build_model")
        ),
        "memory_build_temperature": (
            args.memory_build_temperature
            if args.memory_build_temperature is not None
            else float(source_value("memory_build_temperature", 0.0))
        ),
        "memory_build_seed": (
            args.memory_build_seed
            if args.memory_build_seed is not None
            else int(source_value("memory_build_seed", 42))
        ),
        "memory_build_top_k": (
            args.memory_build_top_k
            if args.memory_build_top_k is not None
            else int(source_value("memory_build_top_k", 1))
        ),
    }
    for name in ("batch_size", "max_steps", "top_k", "memory_build_top_k"):
        if int(values[name]) < 1:
            raise ValueError(f"{name} must be at least 1")
    if values["score_threshold"] < 0:
        raise ValueError("score_threshold cannot be negative")
    if values["split"] not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {values['split']!r}")
    if not values["model"]:
        raise ValueError("No agent model configured")
    if not values["memory_build_model"]:
        raise ValueError("No memory build model configured")
    return values


def _selected_candidates(
    snapshot: Mapping[str, Any],
    proxy_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if args.candidate_queue_ids:
        selected = explicit_selection(proxy_rows, args.candidate_queue_ids)
        selected_ids = {row["queue_id"] for row in selected}
        annotated = [
            {
                **dict(row),
                "selected": row["queue_id"] in selected_ids,
                "proxy_stratum": (
                    "explicit" if row["queue_id"] in selected_ids else None
                ),
            }
            for row in proxy_rows
        ]
        return selected, annotated
    return stratified_proxy_selection(proxy_rows, candidate_count=args.candidate_count)


def _candidate_memory_id(queue_id: str) -> str:
    return "probe_" + queue_id.removeprefix("traj_")


def _experiment_payload(
    snapshot: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source_run_dir": snapshot["source_run_dir"],
        "snapshot_interval": snapshot["snapshot_interval"],
        "downstream_interval": snapshot["downstream_interval"],
        "baseline_memory_ids": [
            row["memory_id"] for row in snapshot["baseline_memories"]
        ],
        "pending_queue_ids": list(snapshot["pending_queue_ids"]),
        "selected_candidates": [
            {
                **dict(row),
                "candidate_memory_id": _candidate_memory_id(row["queue_id"]),
            }
            for row in selected
        ],
        "downstream_task_ids": list(snapshot["downstream_task_ids"]),
        "evaluation_settings": dict(settings),
        "online_update_enabled": False,
        "reflection_enabled": False,
        "construction_enabled_during_evaluation": False,
        "frozen_task_manifest": True,
        "memory_difference": "single_candidate_only",
    }


def _prepare_candidates(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
    settings: Mapping[str, Any],
    output_dir: Path,
    cache_path: Path,
) -> None:
    configure_runtime(
        model_name=settings["model"],
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=True,
    )
    embedding = load_cached_embedding(output_dir / "embedding_cache")
    proxy_rows = coverage_proxy_scores(snapshot, embedding)
    selected, annotated = _selected_candidates(snapshot, proxy_rows, args)
    write_json(output_dir / "candidate_proxy_scores.json", annotated)
    write_task_manifest(output_dir / "task_manifest.json", snapshot)
    experiment = _experiment_payload(snapshot, selected, settings)
    write_json(output_dir / "experiment.json", experiment)

    from ProcedureMem.memory import Memory

    memory_config = load_memory_config(args.memory_config)
    memory_config["policy"] = {
        "build": "direct",
        "retrieve": "query",
        "update": None,
    }
    memory_config["retrieve_num"] = settings["top_k"]
    memory_config["build_model"] = settings["memory_build_model"]
    memory_config["build_temperature"] = settings["memory_build_temperature"]
    memory_config["build_seed"] = settings["memory_build_seed"]
    memory_config["build_top_k"] = settings["memory_build_top_k"]
    memory_config["is_cold_start"] = False
    memory_config["memory_dir"] = str(output_dir / "builder_workspace")
    builder = Memory(**memory_config)

    pending_by_id = {
        row["queue_id"]: row for row in snapshot["pending_candidates"]
    }
    existing = load_workflow_cache(cache_path)
    cache_by_id = {row.get("queue_id"): dict(row) for row in existing}
    for selected_row in selected:
        queue_id = selected_row["queue_id"]
        cached = cache_by_id.get(queue_id)
        if (
            cached
            and cached.get("construction_result") == "success"
            and isinstance(cached.get("workflow"), str)
            and cached["workflow"].strip()
        ):
            print(f"[CACHE] {queue_id}")
            continue
        candidate = pending_by_id[queue_id]
        memory_id = _candidate_memory_id(queue_id)
        try:
            document = builder.build_document(
                {
                    "source": "candidate_utility_probe",
                    "query": candidate["query"],
                    "trajectory": candidate["trajectory"],
                    "memory_id": memory_id,
                    "metadata": {
                        "memory_type": "workflow",
                        "memory_origin": "candidate_utility",
                        "source_queue_id": queue_id,
                        "source_task_id": candidate["task_id"],
                    },
                }
            )
            if document is None:
                raise RuntimeError("Builder returned no workflow document")
            workflow = document.metadata.get("workflow")
            if not isinstance(workflow, str) or not workflow.strip():
                raise RuntimeError("Builder returned an empty workflow")
            cache_by_id[queue_id] = {
                "queue_id": queue_id,
                "candidate_memory_id": memory_id,
                "task_id": candidate["task_id"],
                "query": candidate["query"],
                "source_steps": candidate["source_steps"],
                "proxy_stratum": selected_row["proxy_stratum"],
                "coverage_proxy_gain": selected_row["coverage_proxy_gain"],
                "workflow": workflow.strip(),
                "construction_result": "success",
                "error": None,
            }
            print(f"[BUILT] {queue_id}")
        except Exception as exc:
            cache_by_id[queue_id] = {
                "queue_id": queue_id,
                "candidate_memory_id": memory_id,
                "task_id": candidate["task_id"],
                "query": candidate["query"],
                "source_steps": candidate["source_steps"],
                "proxy_stratum": selected_row["proxy_stratum"],
                "coverage_proxy_gain": selected_row["coverage_proxy_gain"],
                "workflow": None,
                "construction_result": "failure",
                "error": str(exc),
            }
            print(f"[FAILED] {queue_id}: {exc}")
        ordered_cache = [
            cache_by_id[row["queue_id"]]
            for row in selected
            if row["queue_id"] in cache_by_id
        ]
        write_jsonl(cache_path, ordered_cache)

    ordered_cache = [cache_by_id[row["queue_id"]] for row in selected]
    write_jsonl(cache_path, ordered_cache)
    selected_ids = [row["queue_id"] for row in selected]
    validate_workflow_cache(ordered_cache, selected_ids)
    write_json(
        output_dir / "candidate_workflow_summary.json",
        {
            "selected_count": len(selected_ids),
            "construction_success_count": len(selected_ids),
            "construction_failure_count": 0,
            "selected_queue_ids": selected_ids,
        },
    )
    print(f"Prepared {len(selected_ids)} candidate workflows: {cache_path}")


def _condition_memory(
    snapshot: Mapping[str, Any],
    embedding: Any,
    settings: Mapping[str, Any],
    candidate: Mapping[str, Any] | None = None,
) -> tuple[ScheduledWorkflowMemory, str | None]:
    items = [
        CandidateMemory(
            memory_id=row["memory_id"],
            query=row["query"],
            workflow=row["workflow"],
        )
        for row in snapshot["baseline_memories"]
    ]
    candidate_memory_id = None
    if candidate is not None:
        candidate_memory_id = str(candidate["candidate_memory_id"])
        items.append(
            CandidateMemory(
                memory_id=candidate_memory_id,
                query=str(candidate["query"]),
                workflow=str(candidate["workflow"]),
            )
        )
    expected_ids = condition_memory_ids(
        [row["memory_id"] for row in snapshot["baseline_memories"]],
        candidate_memory_id,
    )
    actual_ids = [item.memory_id for item in items]
    if actual_ids != expected_ids:
        raise RuntimeError("Condition memory differs by more than one candidate")
    memory = ScheduledWorkflowMemory(
        items,
        embedding=embedding,
        retrieve_num=int(settings["top_k"]),
        score_threshold=float(settings["score_threshold"]),
    )
    memory.activate(actual_ids, interval_id=0)
    memory.rebuild_available_index()
    return memory, candidate_memory_id


def _run_condition(
    *,
    condition_name: str,
    candidate_memory_id: str | None,
    memory: ScheduledWorkflowMemory,
    snapshot: Mapping[str, Any],
    settings: Mapping[str, Any],
    args: argparse.Namespace,
    llm: Any,
    examples: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    config = load_alfworld_config(args.config)
    config["general"]["random_seed"] = int(settings["seed"])
    random.seed(int(settings["seed"]))
    try:
        import numpy as np

        np.random.seed(int(settings["seed"]))
    except ImportError:
        pass
    from alfworld.agents.environment import get_environment

    source_env = get_environment(config["env"]["type"])(
        copy.deepcopy(config), train_eval=SPLIT_NAMES[str(settings["split"])]
    )
    runtime = configure_runtime(
        model_name=str(settings["model"]),
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=True,
    )
    gamefiles = validate_task_manifest(
        dict(snapshot["task_manifest"]),
        split=str(settings["split"]),
        available_gamefiles=list(source_env.game_files),
        data_root=runtime.alfworld_data,
    )
    task_records = list(snapshot["task_manifest"]["tasks"])
    results: list[dict[str, Any]] = []
    batch_size = int(settings["batch_size"])
    for offset in range(0, len(gamefiles), batch_size):
        chunk = gamefiles[offset : offset + batch_size]
        expected_ids = [
            row["task_id"] for row in task_records[offset : offset + len(chunk)]
        ]
        source_env.game_files = list(chunk)
        source_env.num_games = len(chunk)
        env = source_env.init_env(batch_size=len(chunk))
        try:
            observations, info = env.reset()
            actual_ids = [
                task_id_from_gamefile(gamefile, runtime.alfworld_data)
                for gamefile in info["extra.gamefile"]
            ]
            if actual_ids != expected_ids:
                raise RuntimeError(
                    "ALFWorld reset order differs from frozen manifest: "
                    f"expected={expected_ids}, actual={actual_ids}"
                )
            clean_observations = [
                _initial_observation(observation) for observation in observations
            ]
            injected_observations = []
            retrieved_by_task = []
            for observation in clean_observations:
                records = retrieval_records(memory.retrieve(task_query(observation)))
                retrieved_by_task.append(records)
                injected_observations.append(inject_memory(observation, records))
            batch_results = run_alfworld_batch(
                env=env,
                observations=injected_observations,
                names=[_task_name(task_id) for task_id in actual_ids],
                llm_fn=llm,
                system_prompt=alfworld_system_prompt,
                few_shot=bool(settings["few_shot"]),
                max_steps=int(settings["max_steps"]),
                examples=examples,
            )
        finally:
            if hasattr(env, "close"):
                env.close()
        for local_index, (task_id, result) in enumerate(zip(actual_ids, batch_results)):
            records = retrieved_by_task[local_index]
            retrieved_ids = [
                row["memory_id"] for row in records if row.get("memory_id") is not None
            ]
            result.update(
                {
                    "schema_version": 1,
                    "condition": condition_name,
                    "task_id": task_id,
                    "task_index": offset + local_index,
                    "source_task_index": task_records[offset + local_index].get(
                        "source_task_index"
                    ),
                    "task_type": _task_name(task_id).split("/", 1)[0],
                    "query": task_query(clean_observations[local_index]),
                    "retrieved_memories": records,
                    "retrieved_memory_ids": retrieved_ids,
                    "candidate_memory_id": candidate_memory_id,
                    "candidate_retrieved": (
                        candidate_memory_id in retrieved_ids
                        if candidate_memory_id is not None
                        else False
                    ),
                    "online_update_enabled": False,
                    "reflection_enabled": False,
                    "construction_enabled": False,
                }
            )
            results.append(result)
        print(f"[{condition_name}] completed {len(results)}/{len(gamefiles)} tasks")
    return results


def _write_condition_results(directory: Path, results: Sequence[Mapping[str, Any]]) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Condition output already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / "results.jsonl", results)
    success_count = sum(bool(row["reward"]) for row in results)
    write_json(
        directory / "summary.json",
        {
            "task_count": len(results),
            "success_count": success_count,
            "success_rate": success_count / len(results),
            "average_steps": statistics.fmean(int(row["steps"]) for row in results),
            "task_ids": [row["task_id"] for row in results],
        },
    )


def _evaluate(
    args: argparse.Namespace,
    snapshot: Mapping[str, Any],
    settings: Mapping[str, Any],
    output_dir: Path,
    cache_path: Path,
) -> None:
    experiment_path = output_dir / "experiment.json"
    if not experiment_path.is_file():
        raise FileNotFoundError(
            "Missing experiment.json; run --build-candidate-workflows first"
        )
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    expected_keys = (
        "source_run_dir",
        "snapshot_interval",
        "downstream_task_ids",
        "evaluation_settings",
    )
    current = _experiment_payload(
        snapshot,
        experiment.get("selected_candidates") or [],
        settings,
    )
    for key in expected_keys:
        if experiment.get(key) != current.get(key):
            raise ValueError(f"Prepared experiment differs from evaluation: {key}")
    selected = list(experiment.get("selected_candidates") or [])
    if not selected:
        raise ValueError("Prepared experiment has no selected candidates")
    selected_ids = [row["queue_id"] for row in selected]
    cache = validate_workflow_cache(load_workflow_cache(cache_path), selected_ids)

    runtime = configure_runtime(
        model_name=str(settings["model"]),
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=True,
    )
    llm, routed_model = _make_llm(
        str(runtime.model_name), float(settings["temperature"]), int(settings["seed"])
    )
    embedding = load_cached_embedding(output_dir / "embedding_cache")
    examples = json.loads(DEFAULT_EXAMPLES_PATH.read_text(encoding="utf-8"))

    baseline_memory, _ = _condition_memory(snapshot, embedding, settings)
    baseline_results = _run_condition(
        condition_name="baseline",
        candidate_memory_id=None,
        memory=baseline_memory,
        snapshot=snapshot,
        settings=settings,
        args=args,
        llm=llm,
        examples=examples,
    )
    _write_condition_results(output_dir / "baseline", baseline_results)

    utility_rows = []
    for selected_row in selected:
        queue_id = selected_row["queue_id"]
        cached = cache[queue_id]
        candidate_memory, candidate_memory_id = _condition_memory(
            snapshot, embedding, settings, cached
        )
        candidate_results = _run_condition(
            condition_name=queue_id,
            candidate_memory_id=candidate_memory_id,
            memory=candidate_memory,
            snapshot=snapshot,
            settings=settings,
            args=args,
            llm=llm,
            examples=examples,
        )
        _write_condition_results(
            output_dir / "candidates" / queue_id, candidate_results
        )
        utility_rows.append(
            summarize_candidate_utility(
                baseline_results,
                candidate_results,
                candidate_memory_id=str(candidate_memory_id),
                candidate_metadata={
                    "queue_id": queue_id,
                    "proxy_stratum": selected_row.get("proxy_stratum"),
                    "coverage_proxy_gain": selected_row.get("coverage_proxy_gain"),
                },
            )
        )
    write_json(output_dir / "candidate_utility.json", utility_rows)
    write_utility_csv(output_dir / "candidate_utility.csv", utility_rows)
    write_json(
        output_dir / "summary.json",
        {
            "routed_model": routed_model,
            "baseline_success_count": sum(
                bool(row["reward"]) for row in baseline_results
            ),
            "candidate_count": len(utility_rows),
            "utilities": [row["utility"] for row in utility_rows],
            "utility_min": min(row["utility"] for row in utility_rows),
            "utility_max": max(row["utility"] for row in utility_rows),
            "utility_range": max(row["utility"] for row in utility_rows)
            - min(row["utility"] for row in utility_rows),
        },
    )
    print(f"Candidate utility results: {output_dir / 'candidate_utility.json'}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    snapshot = load_snapshot(args.source_run_dir, args.snapshot_interval)
    settings = _resolve_settings(args, snapshot)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        args.candidate_workflow_cache.expanduser().resolve()
        if args.candidate_workflow_cache
        else output_dir / "candidate_workflows.jsonl"
    )
    if args.build_candidate_workflows:
        _prepare_candidates(args, snapshot, settings, output_dir, cache_path)
    else:
        _evaluate(args, snapshot, settings, output_dir, cache_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
