"""Task manifests and auditable result summaries for ALFWorld evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CONDITIONS = ("no_memory", "memory")
EVAL_CONDITIONS = CONDITIONS + ("memory_rerank", "edge_raw", "cloud_scheduled")
SPLIT_NAMES = {
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def task_id_from_gamefile(gamefile: str | Path, data_root: str | Path) -> str:
    path = Path(gamefile).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Gamefile {path} is outside ALFWorld data root {root}") from exc


def available_task_map(
    gamefiles: Iterable[str | Path], data_root: str | Path
) -> dict[str, str]:
    gamefile_list = list(gamefiles)
    tasks = {
        task_id_from_gamefile(gamefile, data_root): str(Path(gamefile).resolve())
        for gamefile in gamefile_list
    }
    if len(tasks) != len(gamefile_list):
        raise ValueError("ALFWorld returned duplicate task IDs")
    return tasks


def build_task_manifest(
    gamefiles: Sequence[str | Path],
    *,
    data_root: str | Path,
    split: str,
    seed: int,
    limit_tasks: int | None = None,
    task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {split}")
    available = available_task_map(gamefiles, data_root)
    ordered_ids = sorted(available)

    if task_ids:
        selected_ids = list(task_ids)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("task_ids contains duplicates")
        missing = [task_id for task_id in selected_ids if task_id not in available]
        if missing:
            raise ValueError("Unknown task IDs: " + ", ".join(missing[:5]))
        if limit_tasks is not None and limit_tasks != len(selected_ids):
            raise ValueError("limit_tasks must match the number of explicit task_ids")
    else:
        rng = random.Random(seed)
        rng.shuffle(ordered_ids)
        if limit_tasks is not None:
            if limit_tasks < 1:
                raise ValueError("limit_tasks must be at least 1")
            if limit_tasks > len(ordered_ids):
                raise ValueError(
                    f"Requested {limit_tasks} tasks, but split has {len(ordered_ids)}"
                )
            ordered_ids = ordered_ids[:limit_tasks]
        selected_ids = ordered_ids

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split": split,
        "alfworld_split": SPLIT_NAMES[split],
        "seed": seed,
        "task_count": len(selected_ids),
        "tasks": [{"index": index, "task_id": task_id} for index, task_id in enumerate(selected_ids)],
    }


def validate_task_manifest(
    manifest: dict[str, Any],
    *,
    split: str,
    available_gamefiles: Sequence[str | Path],
    data_root: str | Path,
) -> list[str]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported task manifest schema_version")
    if manifest.get("split") != split:
        raise ValueError(
            f"Manifest split is {manifest.get('split')!r}, requested {split!r}"
        )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest has no tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("Task manifest contains an invalid task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task manifest contains duplicate task IDs")
    if manifest.get("task_count") != len(task_ids):
        raise ValueError("Task manifest task_count does not match tasks")

    available = available_task_map(available_gamefiles, data_root)
    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        raise ValueError("Manifest tasks missing from ALFWorld data: " + ", ".join(missing[:5]))
    return [available[task_id] for task_id in task_ids]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_query(observation: str) -> str:
    marker = "Your task is to: "
    return observation.split(marker, 1)[1].strip() if marker in observation else observation.strip()


def retrieval_records(items: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for rank, item in enumerate(items, start=1):
        if isinstance(item, tuple):
            document, score = item
        else:
            document, score = item, None
        metadata = document.metadata
        record = {
            "rank": rank,
            "memory_id": metadata.get("memory_id"),
            "task_name": metadata.get("query"),
            "score": float(score) if score is not None else None,
            "source": metadata.get("source"),
            "activated_interval": metadata.get("activated_interval"),
        }
        if metadata.get("memory_type") == "raw_trajectory" or "trajectory" in metadata:
            record.update(
                {
                    "memory_type": "raw_trajectory",
                    "trajectory": metadata.get("trajectory"),
                    "trajectory_index": metadata.get("trajectory_index"),
                    "task_type": metadata.get("task_type"),
                    "raw_score": float(score) if score is not None else None,
                    "score_type": "faiss_l2_distance" if score is not None else None,
                    "higher_is_better": False if score is not None else None,
                }
            )
        else:
            record["workflow"] = metadata.get("workflow")
            if metadata.get("trajectory_index") is not None:
                record["trajectory_index"] = metadata.get("trajectory_index")
        records.append(record)
    return records


def reranked_retrieval_records(
    items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert structured reranker output without conflating score directions."""
    records = []
    for item in items:
        document = item["document"]
        metadata = document.metadata
        records.append(
            {
                "rank": int(item["rerank_rank"]),
                "task_name": metadata.get("query"),
                "workflow": metadata.get("workflow"),
                "source": metadata.get("source"),
                "vector_rank": int(item["vector_rank"]),
                "vector_score": float(item["vector_score"]),
                "vector_score_type": "faiss_l2_distance",
                "vector_higher_is_better": False,
                "rerank_rank": int(item["rerank_rank"]),
                "rerank_score": float(item["rerank_score"]),
                "rerank_score_type": "openmem_relevance_score",
                "rerank_higher_is_better": True,
            }
        )
    return records


def inject_memory(observation: str, records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return observation
    guidelines = [
        {"task_name": record["task_name"], "guidelines": record["workflow"]}
        for record in records
    ]
    return (
        observation
        + "\n\nHere are some guidelines for solving similar tasks:\n"
        + json.dumps(guidelines, indent=2, ensure_ascii=False)
        + "\n"
    )


def _multiline_json_string(value: str) -> str:
    """Escape JSON-sensitive characters while keeping line breaks readable."""
    return json.dumps(value, ensure_ascii=False)[1:-1].replace("\\n", "\n")


def inject_trajectories(
    observation: str, records: Sequence[dict[str, Any]]
) -> str:
    if not records:
        return observation
    rendered = []
    for record in records:
        task_name = json.dumps(record["task_name"], ensure_ascii=False)
        trajectory = _multiline_json_string(record["trajectory"])
        rendered.append(
            "  {\n"
            f'    "task_name": {task_name},\n'
            '    "trajectory": "\n'
            f"{trajectory}\n"
            '"\n'
            "  }"
        )
    return (
        observation
        + "\n\nHere are some trajectories for solving similar tasks:\n[\n"
        + ",\n".join(rendered)
        + "\n]\n"
    )


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty result set")
    task_ids = [result["task_id"] for result in results]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Results contain duplicate task IDs")

    successes = [result for result in results if bool(result["reward"])]
    steps = [int(result["steps"]) for result in results]
    successful_steps = [int(result["steps"]) for result in successes]
    termination_counts = Counter(result["termination_reason"] for result in results)
    error_count = sum(bool(result.get("error")) for result in results)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "condition": results[0]["condition"],
        "split": results[0]["split"],
        "task_count": len(results),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "success_rate": len(successes) / len(results),
        "average_steps": sum(steps) / len(steps),
        "average_success_steps": (
            sum(successful_steps) / len(successful_steps) if successful_steps else None
        ),
        "error_count": error_count,
        "termination_counts": dict(sorted(termination_counts.items())),
        "task_ids": task_ids,
    }


def write_results(
    output_dir: str | Path,
    results: Sequence[dict[str, Any]],
    *,
    summary_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(output_dir)
    task_dir = directory / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        digest = hashlib.sha256(result["task_id"].encode("utf-8")).hexdigest()[:12]
        write_json(task_dir / f"{index:05d}_{digest}.json", result)

    jsonl = "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results)
    (directory / "results.jsonl").write_text(jsonl, encoding="utf-8")
    summary = summarize_results(results)
    if summary_metadata:
        summary.update(summary_metadata)
    write_json(directory / "summary.json", summary)
    _write_csv(directory / "summary.csv", [summary])
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(scalar_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(scalar_rows)


def build_paired_comparison(
    no_memory: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    if no_memory.get("task_ids") != memory.get("task_ids"):
        raise ValueError("Cannot compare conditions with different task IDs or order")
    if no_memory.get("split") != memory.get("split"):
        raise ValueError("Cannot compare conditions from different splits")
    no_memory_parameters = no_memory.get("parameters", {})
    memory_parameters = memory.get("parameters", {})
    paired_keys = (
        "model",
        "agent_api_base_url",
        "embedding_model",
        "split",
        "seed",
        "batch_size",
        "max_steps",
        "temperature",
        "top_p",
        "few_shot",
        "top_k",
        "manifest_sha256",
    )
    mismatches = [
        key
        for key in paired_keys
        if no_memory_parameters.get(key) != memory_parameters.get(key)
    ]
    if mismatches:
        raise ValueError("Paired experiment parameter mismatch: " + ", ".join(mismatches))
    no_memory_sr = float(no_memory["success_rate"])
    memory_sr = float(memory["success_rate"])
    absolute = memory_sr - no_memory_sr
    relative = absolute / no_memory_sr if no_memory_sr else None
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "split": no_memory["split"],
        "task_count": no_memory["task_count"],
        "no_memory_success_rate": no_memory_sr,
        "memory_success_rate": memory_sr,
        "absolute_improvement": absolute,
        "absolute_improvement_percentage_points": absolute * 100,
        "relative_improvement": relative,
        "task_ids": no_memory["task_ids"],
    }


def maybe_write_paired_comparison(experiment_dir: str | Path) -> dict[str, Any] | None:
    root = Path(experiment_dir)
    paths = {condition: root / condition / "summary.json" for condition in CONDITIONS}
    if not all(path.is_file() for path in paths.values()):
        return None
    comparison = build_paired_comparison(
        load_json(paths["no_memory"]), load_json(paths["memory"])
    )
    write_json(root / "comparison.json", comparison)
    _write_csv(root / "comparison.csv", [comparison])
    return comparison


def build_condition_comparison(
    baseline_summary: dict[str, Any],
    rerank_summary: dict[str, Any],
    baseline_results: Sequence[dict[str, Any]],
    rerank_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare workflow-memory baseline and reranked workflow memory."""
    if baseline_summary.get("task_ids") != rerank_summary.get("task_ids"):
        raise ValueError("Cannot compare conditions with different task IDs or order")
    baseline_parameters = baseline_summary.get("parameters", {})
    rerank_parameters = rerank_summary.get("parameters", {})
    paired_keys = (
        "model",
        "agent_api_base_url",
        "embedding_model",
        "split",
        "seed",
        "batch_size",
        "max_steps",
        "temperature",
        "top_p",
        "few_shot",
        "memory_config",
        "memory_build_model",
        "manifest_sha256",
    )
    mismatches = [
        key
        for key in paired_keys
        if baseline_parameters.get(key) != rerank_parameters.get(key)
    ]
    if mismatches:
        raise ValueError("Paired experiment parameter mismatch: " + ", ".join(mismatches))
    baseline_by_id = {item["task_id"]: bool(item["reward"]) for item in baseline_results}
    rerank_by_id = {item["task_id"]: bool(item["reward"]) for item in rerank_results}
    if list(baseline_by_id) != list(rerank_by_id):
        raise ValueError("Cannot compare result files with different task IDs or order")

    failure_to_success = sum(
        not baseline_by_id[task_id] and rerank_by_id[task_id]
        for task_id in baseline_by_id
    )
    success_to_failure = sum(
        baseline_by_id[task_id] and not rerank_by_id[task_id]
        for task_id in baseline_by_id
    )
    both_success = sum(
        baseline_by_id[task_id] and rerank_by_id[task_id]
        for task_id in baseline_by_id
    )
    both_failure = len(baseline_by_id) - failure_to_success - success_to_failure - both_success
    baseline_sr = float(baseline_summary["success_rate"])
    rerank_sr = float(rerank_summary["success_rate"])
    baseline_retrieval = baseline_summary.get("retrieval_summary") or {}
    rerank_retrieval = rerank_summary.get("rerank_summary") or {}
    baseline_latency = baseline_retrieval.get("similarity_search_latency_ms_mean")
    if baseline_latency is None:
        baseline_latency = rerank_retrieval.get(
            "baseline_similarity_search_latency_ms_mean"
        )
    rerank_latency = rerank_retrieval.get("rerank_pipeline_latency_ms_mean")
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "baseline_condition": baseline_summary["condition"],
        "rerank_condition": rerank_summary["condition"],
        "split": baseline_summary["split"],
        "task_count": len(baseline_by_id),
        "baseline_success_rate": baseline_sr,
        "rerank_success_rate": rerank_sr,
        "absolute_improvement": rerank_sr - baseline_sr,
        "absolute_improvement_percentage_points": (rerank_sr - baseline_sr) * 100,
        "failure_to_success": failure_to_success,
        "success_to_failure": success_to_failure,
        "both_success": both_success,
        "both_failure": both_failure,
        "baseline_retrieval_latency_ms_mean": baseline_latency,
        "rerank_pipeline_latency_ms_mean": rerank_latency,
        "rerank_added_latency_ms_mean": (
            rerank_latency - baseline_latency
            if rerank_latency is not None and baseline_latency is not None
            else None
        ),
        "task_ids": list(baseline_by_id),
    }


def maybe_write_memory_rerank_comparison(
    experiment_dir: str | Path,
) -> dict[str, Any] | None:
    root = Path(experiment_dir)
    required = {
        "memory_summary": root / "memory" / "summary.json",
        "rerank_summary": root / "memory_rerank" / "summary.json",
        "memory_results": root / "memory" / "results.jsonl",
        "rerank_results": root / "memory_rerank" / "results.jsonl",
    }
    if not all(path.is_file() for path in required.values()):
        return None

    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    comparison = build_condition_comparison(
        load_json(required["memory_summary"]),
        load_json(required["rerank_summary"]),
        load_jsonl(required["memory_results"]),
        load_jsonl(required["rerank_results"]),
    )
    write_json(root / "memory_vs_memory_rerank_comparison.json", comparison)
    _write_csv(root / "memory_vs_memory_rerank_comparison.csv", [comparison])
    return comparison


def maybe_write_scheduling_comparison(
    experiment_dir: str | Path,
) -> dict[str, Any] | None:
    """Write Random versus available Oracle scheduling comparisons."""
    root = Path(experiment_dir)
    summaries = []
    if not root.is_dir():
        return None
    for path in sorted(root.glob("*/summary.json")):
        summary = load_json(path)
        parameters = summary.get("parameters") or {}
        if parameters.get("condition_mode") == "cloud_scheduled":
            summaries.append(summary)

    random_summaries = [
        item for item in summaries
        if item.get("parameters", {}).get("schedule_policy") == "random"
    ]
    oracle_summaries = [
        item
        for item in summaries
        if item.get("parameters", {}).get("schedule_policy")
        in {"oracle_high", "oracle_sum", "oracle_coverage"}
    ]
    if not random_summaries or not oracle_summaries:
        return None
    oracle_policies = [
        item.get("parameters", {}).get("schedule_policy")
        for item in oracle_summaries
    ]
    if len(oracle_policies) != len(set(oracle_policies)):
        raise ValueError("Scheduling comparison requires one summary per Oracle policy")

    reference = random_summaries[0]
    reference_parameters = reference["parameters"]
    controlled_keys = (
        "model",
        "agent_api_base_url",
        "embedding_model",
        "split",
        "seed",
        "batch_size",
        "max_steps",
        "temperature",
        "top_p",
        "few_shot",
        "top_k",
        "manifest_sha256",
        "candidate_pool_sha256",
        "interval_size",
        "construction_capacity",
        "scheduled_score_threshold",
    )

    def normalized_warm_start(
        parameters: dict[str, Any],
    ) -> tuple[int, int | None, tuple[str, ...], str | None]:
        count = int(parameters.get("warm_start_count") or 0)
        memory_ids = tuple(parameters.get("initial_available_memory_ids") or ())
        if len(memory_ids) != count or len(memory_ids) != len(set(memory_ids)):
            raise ValueError(
                "Scheduling summary has an invalid initial available memory pool"
            )
        if count == 0:
            return 0, None, (), None
        seed = parameters.get("warm_start_seed")
        pool_sha256 = parameters.get("initial_available_pool_sha256")
        if seed is None or not pool_sha256:
            raise ValueError("Warm-start scheduling summary is missing pool metadata")
        return count, int(seed), memory_ids, str(pool_sha256)

    reference_warm_start = normalized_warm_start(reference_parameters)
    for summary in random_summaries[1:] + oracle_summaries:
        if summary.get("task_ids") != reference.get("task_ids"):
            raise ValueError(
                "Scheduling conditions use different task IDs or task order"
            )
        parameters = summary.get("parameters") or {}
        mismatches = [
            key
            for key in controlled_keys
            if parameters.get(key) != reference_parameters.get(key)
        ]
        if mismatches:
            raise ValueError(
                "Scheduling experiment parameter mismatch: " + ", ".join(mismatches)
            )
        if normalized_warm_start(parameters) != reference_warm_start:
            raise ValueError(
                "Scheduling conditions use different warm-start memory pools"
            )

    random_success_rates = [float(item["success_rate"]) for item in random_summaries]
    random_steps = [float(item["average_steps"]) for item in random_summaries]
    random_sr_mean = statistics.fmean(random_success_rates)
    random_steps_mean = statistics.fmean(random_steps)
    oracle_runs = []
    for oracle in oracle_summaries:
        oracle_sr = float(oracle["success_rate"])
        oracle_steps = float(oracle["average_steps"])
        oracle_runs.append(
            {
                "policy": oracle["parameters"]["schedule_policy"],
                "condition": oracle["condition"],
                "success_rate": oracle_sr,
                "average_steps": oracle_steps,
                "minus_random_success_rate": oracle_sr - random_sr_mean,
                "minus_random_success_rate_percentage_points": (
                    oracle_sr - random_sr_mean
                )
                * 100,
                "minus_random_average_steps": oracle_steps - random_steps_mean,
            }
        )
    oracle_runs.sort(key=lambda item: item["policy"])
    primary_policy_order = ("oracle_high", "oracle_sum", "oracle_coverage")
    primary_oracle = next(
        item
        for policy in primary_policy_order
        for item in oracle_runs
        if item["policy"] == policy
    )
    comparison = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "experiment_type": "construction_scheduling_feasibility",
        "random_run_count": len(random_summaries),
        "random_runs": [
            {
                "condition": item["condition"],
                "scheduler_seed": item["parameters"].get("scheduler_seed"),
                "success_rate": item["success_rate"],
                "average_steps": item["average_steps"],
            }
            for item in random_summaries
        ],
        "random_success_rate_mean": random_sr_mean,
        "random_success_rate_std": statistics.pstdev(random_success_rates),
        "random_average_steps_mean": random_steps_mean,
        "random_average_steps_std": statistics.pstdev(random_steps),
        "warm_start_count": reference_warm_start[0],
        "warm_start_seed": reference_warm_start[1],
        "initial_available_memory_ids": list(reference_warm_start[2]),
        "initial_available_pool_sha256": reference_warm_start[3],
        "oracle_runs": oracle_runs,
        # Preserve the original scalar fields for existing result consumers.
        "oracle_condition": primary_oracle["condition"],
        "oracle_success_rate": primary_oracle["success_rate"],
        "oracle_average_steps": primary_oracle["average_steps"],
        "oracle_minus_random_success_rate": primary_oracle[
            "minus_random_success_rate"
        ],
        "oracle_minus_random_success_rate_percentage_points": (
            primary_oracle["minus_random_success_rate_percentage_points"]
        ),
        "oracle_minus_random_average_steps": primary_oracle[
            "minus_random_average_steps"
        ],
    }
    by_policy = {item["policy"]: item for item in oracle_runs}
    if "oracle_sum" in by_policy and "oracle_coverage" in by_policy:
        oracle_sum = by_policy["oracle_sum"]
        oracle_coverage = by_policy["oracle_coverage"]
        comparison.update(
            {
                "oracle_coverage_minus_oracle_sum_success_rate": (
                    oracle_coverage["success_rate"] - oracle_sum["success_rate"]
                ),
                "oracle_coverage_minus_oracle_sum_success_rate_percentage_points": (
                    oracle_coverage["success_rate"] - oracle_sum["success_rate"]
                )
                * 100,
                "oracle_coverage_minus_oracle_sum_average_steps": (
                    oracle_coverage["average_steps"] - oracle_sum["average_steps"]
                ),
            }
        )
    write_json(root / "scheduling_comparison.json", comparison)
    _write_csv(root / "scheduling_comparison.csv", [comparison])
    return comparison
