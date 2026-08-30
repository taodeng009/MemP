"""Frozen-snapshot candidate downstream-utility experiment helpers."""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ProcedureMem.alfworld_experiment import SPLIT_NAMES, write_json


REQUIRED_SOURCE_FILES = (
    "summary.json",
    "results.jsonl",
    "online_trajectories.jsonl",
    "queue_events.jsonl",
    "construction_events.jsonl",
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    rows = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source}:{line_number} is not a JSON object")
        rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(destination)


def load_snapshot(source_run_dir: str | Path, snapshot_interval: int) -> dict[str, Any]:
    source = Path(source_run_dir).expanduser().resolve()
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval cannot be negative")
    missing = [name for name in REQUIRED_SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Source run is missing required files: {', '.join(missing)}"
        )

    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    parameters = summary.get("parameters") or {}
    if parameters.get("condition_mode") != "online_construction":
        raise ValueError("Source run must be an online_construction condition")
    if int(parameters.get("warm_start_count") or 0) != 0:
        raise ValueError("V1 candidate utility requires a zero-warm-start source run")

    results = read_jsonl(source / "results.jsonl")
    trajectories = read_jsonl(source / "online_trajectories.jsonl")
    queue_events = read_jsonl(source / "queue_events.jsonl")
    construction_events = read_jsonl(source / "construction_events.jsonl")

    matching_events = [
        row for row in queue_events if int(row.get("interval_id", -1)) == snapshot_interval
    ]
    if len(matching_events) != 1:
        raise ValueError(
            f"Expected one queue event for interval {snapshot_interval}, "
            f"found {len(matching_events)}"
        )
    queue_event = matching_events[0]
    pending_ids = list(queue_event.get("pending_queue_ids_before_selection") or [])
    if not pending_ids:
        raise ValueError("Snapshot pending queue is empty")
    if len(pending_ids) != len(set(pending_ids)):
        raise ValueError("Snapshot pending queue contains duplicate IDs")

    downstream_results = [
        row
        for row in results
        if int(row.get("interval_id", -1)) == snapshot_interval + 1
    ]
    downstream_results.sort(key=lambda row: int(row["task_index"]))
    if not downstream_results:
        raise ValueError(
            f"Snapshot interval {snapshot_interval} has no downstream interval"
        )
    downstream_ids = [str(row["task_id"]) for row in downstream_results]
    if len(downstream_ids) != len(set(downstream_ids)):
        raise ValueError("Downstream tasks contain duplicate task IDs")

    trajectory_by_id: dict[str, dict[str, Any]] = {}
    for row in trajectories:
        queue_id = row.get("queue_id")
        if not isinstance(queue_id, str) or not queue_id:
            raise ValueError("Online trajectory has no queue_id")
        if queue_id in trajectory_by_id:
            raise ValueError(f"Duplicate online trajectory queue_id: {queue_id}")
        trajectory_by_id[queue_id] = row
    missing_trajectories = [
        queue_id for queue_id in pending_ids if queue_id not in trajectory_by_id
    ]
    if missing_trajectories:
        raise ValueError(
            "Pending queue IDs missing trajectories: "
            + ", ".join(missing_trajectories[:5])
        )
    pending_candidates = []
    for pending_order, queue_id in enumerate(pending_ids):
        row = trajectory_by_id[queue_id]
        query = row.get("query")
        trajectory = row.get("trajectory")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Pending candidate {queue_id} has no query")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"Pending candidate {queue_id} has no trajectory")
        pending_candidates.append(
            {
                "queue_id": queue_id,
                "pending_order": pending_order,
                "task_id": row.get("task_id"),
                "task_index": int(row.get("task_index")),
                "task_type": row.get("task_type"),
                "query": query.strip(),
                "trajectory": trajectory,
                "source_steps": int(row.get("steps")),
                "arrival_interval": int(row.get("arrival_interval")),
            }
        )

    baseline_events = [
        row
        for row in construction_events
        if row.get("construction_result") == "success"
        and row.get("available_from_interval") is not None
        and int(row["available_from_interval"]) <= snapshot_interval
    ]
    if not baseline_events:
        raise ValueError("V1 candidate utility requires non-empty baseline memory")
    baseline_memories = []
    for row in baseline_events:
        queue_id = row.get("queue_id")
        trajectory_row = trajectory_by_id.get(str(queue_id))
        workflow = row.get("workflow")
        memory_id = row.get("constructed_memory_id")
        if trajectory_row is None:
            raise ValueError(f"Baseline memory {memory_id} has no source trajectory")
        if not isinstance(workflow, str) or not workflow.strip():
            raise ValueError(f"Baseline memory {memory_id} has no workflow")
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("Baseline construction event has no memory ID")
        baseline_memories.append(
            {
                "memory_id": memory_id,
                "queue_id": queue_id,
                "query": str(trajectory_row["query"]).strip(),
                "workflow": workflow.strip(),
            }
        )
    baseline_ids = [row["memory_id"] for row in baseline_memories]
    if len(baseline_ids) != len(set(baseline_ids)):
        raise ValueError("Baseline memory contains duplicate IDs")

    split = parameters.get("split")
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported source split: {split!r}")
    seed = int(parameters.get("seed", 42))
    task_manifest = {
        "schema_version": 1,
        "split": split,
        "alfworld_split": SPLIT_NAMES[split],
        "seed": seed,
        "task_count": len(downstream_ids),
        "tasks": [
            {
                "index": index,
                "task_id": task_id,
                "source_task_index": int(downstream_results[index]["task_index"]),
            }
            for index, task_id in enumerate(downstream_ids)
        ],
    }
    return {
        "source_run_dir": str(source),
        "snapshot_interval": snapshot_interval,
        "downstream_interval": snapshot_interval + 1,
        "source_parameters": parameters,
        "baseline_memories": baseline_memories,
        "pending_candidates": pending_candidates,
        "pending_queue_ids": pending_ids,
        "downstream_task_ids": downstream_ids,
        "downstream_queries": [str(row["query"]) for row in downstream_results],
        "task_manifest": task_manifest,
    }


def _squared_l2(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ")
    return float(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def coverage_proxy_scores(
    snapshot: Mapping[str, Any], embedding: Any
) -> list[dict[str, Any]]:
    baseline = list(snapshot["baseline_memories"])
    pending = list(snapshot["pending_candidates"])
    downstream_queries = list(snapshot["downstream_queries"])
    if not baseline or not pending or not downstream_queries:
        raise ValueError("Coverage proxy requires baseline, pending, and downstream data")

    texts = (
        [row["query"] for row in baseline]
        + [row["query"] for row in pending]
        + downstream_queries
    )
    vectors = list(embedding.embed_documents(texts))
    if len(vectors) != len(texts):
        raise ValueError("Embedding client returned the wrong vector count")
    baseline_vectors = vectors[: len(baseline)]
    pending_start = len(baseline)
    query_start = pending_start + len(pending)
    pending_vectors = vectors[pending_start:query_start]
    query_vectors = vectors[query_start:]

    best_baseline = [
        min(_squared_l2(query_vector, memory_vector) for memory_vector in baseline_vectors)
        for query_vector in query_vectors
    ]
    rows = []
    for candidate, candidate_vector in zip(pending, pending_vectors):
        distances = [
            _squared_l2(query_vector, candidate_vector)
            for query_vector in query_vectors
        ]
        gain = float(
            sum(max(0.0, old - new) for old, new in zip(best_baseline, distances))
        )
        rows.append(
            {
                "queue_id": candidate["queue_id"],
                "pending_order": int(candidate["pending_order"]),
                "coverage_proxy_gain": gain,
                "proxy_score_type": "single_candidate_faiss_squared_l2_marginal_gain",
                "higher_is_better": True,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["coverage_proxy_gain"]),
            int(row["pending_order"]),
            row["queue_id"],
        ),
    )
    for rank, row in enumerate(ranked, start=1):
        row["proxy_rank"] = rank
    return ranked


def stratified_proxy_selection(
    proxy_rows: Sequence[Mapping[str, Any]], *, candidate_count: int = 6
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if candidate_count != 6:
        raise ValueError("V1 automatic stratification requires candidate_count=6")
    if len(proxy_rows) < candidate_count:
        raise ValueError("At least 6 pending candidates are required")
    ranked = [dict(row) for row in proxy_rows]
    ranked.sort(
        key=lambda row: (
            -float(row["coverage_proxy_gain"]),
            int(row["pending_order"]),
            row["queue_id"],
        )
    )
    high = ranked[:2]
    high_ids = {row["queue_id"] for row in high}

    zero_rows = [
        row
        for row in ranked
        if row["queue_id"] not in high_ids
        if math.isclose(
            float(row["coverage_proxy_gain"]), 0.0, abs_tol=1e-12
        )
    ]
    low = zero_rows[:2]
    low_ids = {row["queue_id"] for row in low}
    if len(low) < 2:
        for row in reversed(ranked):
            if row["queue_id"] in high_ids | low_ids:
                continue
            low.append(row)
            low_ids.add(row["queue_id"])
            if len(low) == 2:
                break

    values = sorted(float(row["coverage_proxy_gain"]) for row in ranked)
    middle = len(values) // 2
    median = (
        values[middle]
        if len(values) % 2
        else (values[middle - 1] + values[middle]) / 2.0
    )
    remaining = [
        row for row in ranked if row["queue_id"] not in high_ids | low_ids
    ]
    medium = sorted(
        remaining,
        key=lambda row: (
            abs(float(row["coverage_proxy_gain"]) - median),
            int(row["pending_order"]),
            row["queue_id"],
        ),
    )[:2]
    if len(medium) != 2:
        raise ValueError("Could not choose two non-overlapping medium candidates")

    selected = []
    for stratum, items in (("high", high), ("medium", medium), ("low", low)):
        for item in items:
            selected.append({**item, "proxy_stratum": stratum})
    selected_ids = {row["queue_id"] for row in selected}
    if len(selected_ids) != 6:
        raise ValueError("Proxy strata overlap")
    annotated = []
    selected_by_id = {row["queue_id"]: row for row in selected}
    for row in ranked:
        chosen = selected_by_id.get(row["queue_id"])
        annotated.append(
            {
                **row,
                "selected": chosen is not None,
                "proxy_stratum": chosen.get("proxy_stratum") if chosen else None,
            }
        )
    return selected, annotated


def explicit_selection(
    proxy_rows: Sequence[Mapping[str, Any]], queue_ids: Sequence[str]
) -> list[dict[str, Any]]:
    requested = list(queue_ids)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("Explicit candidate queue IDs must be non-empty and unique")
    by_id = {row["queue_id"]: row for row in proxy_rows}
    unknown = [queue_id for queue_id in requested if queue_id not in by_id]
    if unknown:
        raise ValueError("Candidate IDs are not pending: " + ", ".join(unknown))
    return [{**dict(by_id[queue_id]), "proxy_stratum": "explicit"} for queue_id in requested]


def load_workflow_cache(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return read_jsonl(source) if source.is_file() else []


def validate_workflow_cache(
    rows: Sequence[Mapping[str, Any]], selected_queue_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        queue_id = row.get("queue_id")
        if not isinstance(queue_id, str) or not queue_id:
            raise ValueError("Workflow cache row has no queue_id")
        if queue_id in by_id:
            raise ValueError(f"Duplicate workflow cache queue_id: {queue_id}")
        by_id[queue_id] = dict(row)
    required = list(selected_queue_ids)
    missing = [queue_id for queue_id in required if queue_id not in by_id]
    failed = [
        queue_id
        for queue_id in required
        if queue_id in by_id
        and (
            by_id[queue_id].get("construction_result") != "success"
            or not isinstance(by_id[queue_id].get("workflow"), str)
            or not by_id[queue_id]["workflow"].strip()
        )
    ]
    if missing or failed:
        raise ValueError(
            f"Workflow cache incomplete; missing={missing[:5]}, failed={failed[:5]}"
        )
    return {queue_id: by_id[queue_id] for queue_id in required}


def condition_memory_ids(
    baseline_memory_ids: Sequence[str], candidate_memory_id: str | None = None
) -> list[str]:
    ids = list(baseline_memory_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Baseline memory IDs contain duplicates")
    if candidate_memory_id is not None:
        if candidate_memory_id in ids:
            raise ValueError("Candidate memory already exists in baseline")
        ids.append(candidate_memory_id)
    return ids


def summarize_candidate_utility(
    baseline_results: Sequence[Mapping[str, Any]],
    candidate_results: Sequence[Mapping[str, Any]],
    *,
    candidate_memory_id: str,
    candidate_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_by_id = {row["task_id"]: row for row in baseline_results}
    candidate_by_id = {row["task_id"]: row for row in candidate_results}
    if list(baseline_by_id) != list(candidate_by_id):
        raise ValueError("Baseline and candidate task IDs/order differ")

    gained = []
    lost = []
    both_success = []
    both_fail = []
    retrieved = []
    for task_id in baseline_by_id:
        baseline_success = bool(baseline_by_id[task_id]["reward"])
        candidate_success = bool(candidate_by_id[task_id]["reward"])
        retrieved_ids = candidate_by_id[task_id].get("retrieved_memory_ids") or []
        if candidate_memory_id in retrieved_ids:
            retrieved.append(task_id)
        if candidate_success and not baseline_success:
            gained.append(task_id)
        elif baseline_success and not candidate_success:
            lost.append(task_id)
        elif baseline_success:
            both_success.append(task_id)
        else:
            both_fail.append(task_id)
    retrieved_set = set(retrieved)
    baseline_success_count = sum(
        bool(row["reward"]) for row in baseline_results
    )
    candidate_success_count = sum(
        bool(row["reward"]) for row in candidate_results
    )
    return {
        **dict(candidate_metadata),
        "candidate_memory_id": candidate_memory_id,
        "baseline_success_count": baseline_success_count,
        "candidate_success_count": candidate_success_count,
        "utility": candidate_success_count - baseline_success_count,
        "gained_task_ids": gained,
        "lost_task_ids": lost,
        "both_success_task_ids": both_success,
        "both_fail_task_ids": both_fail,
        "candidate_retrieved_task_ids": retrieved,
        "candidate_retrieved_task_count": len(retrieved),
        "gained_retrieved_task_ids": [task for task in gained if task in retrieved_set],
        "lost_retrieved_task_ids": [task for task in lost if task in retrieved_set],
        "gained_unretrieved_task_ids": [task for task in gained if task not in retrieved_set],
        "lost_unretrieved_task_ids": [task for task in lost if task not in retrieved_set],
    }


def summarize_baseline_stability(
    repeat_results: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    if len(repeat_results) < 2:
        raise ValueError("Baseline stability requires at least two repeats")
    task_ids = [str(row["task_id"]) for row in repeat_results[0]]
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Baseline repeat task IDs must be non-empty and unique")

    normalized_repeats: list[dict[str, Mapping[str, Any]]] = []
    repeat_summaries = []
    for repeat_index, results in enumerate(repeat_results, start=1):
        repeat_ids = [str(row["task_id"]) for row in results]
        if repeat_ids != task_ids:
            raise ValueError(
                f"Baseline repeat {repeat_index} task IDs/order differ"
            )
        by_id = {str(row["task_id"]): row for row in results}
        normalized_repeats.append(by_id)
        success_count = sum(bool(row["reward"]) for row in results)
        repeat_summaries.append(
            {
                "repeat_id": repeat_index,
                "success_count": success_count,
                "success_rate": success_count / len(task_ids),
                "average_steps": sum(int(row["steps"]) for row in results)
                / len(task_ids),
            }
        )

    pairwise = {
        "same_retrieval": {
            "comparisons": 0,
            "flips": 0,
            "gained": 0,
            "lost": 0,
        },
        "changed_retrieval": {
            "comparisons": 0,
            "flips": 0,
            "gained": 0,
            "lost": 0,
        },
    }
    task_rows = []
    for task_id in task_ids:
        rows = [repeat[task_id] for repeat in normalized_repeats]
        outcomes = [bool(row["reward"]) for row in rows]
        queries = [str(row.get("query") or "") for row in rows]
        retrievals = [
            tuple(str(memory_id) for memory_id in row.get("retrieved_memory_ids") or [])
            for row in rows
        ]
        success_count = sum(outcomes)
        retrieval_variants = []
        for retrieval in retrievals:
            rendered = list(retrieval)
            if rendered not in retrieval_variants:
                retrieval_variants.append(rendered)
        task_pair_flips = 0
        for left_index, right_index in itertools.combinations(range(len(rows)), 2):
            key = (
                "same_retrieval"
                if retrievals[left_index] == retrievals[right_index]
                else "changed_retrieval"
            )
            bucket = pairwise[key]
            bucket["comparisons"] += 1
            if outcomes[left_index] == outcomes[right_index]:
                continue
            bucket["flips"] += 1
            task_pair_flips += 1
            if outcomes[right_index]:
                bucket["gained"] += 1
            else:
                bucket["lost"] += 1
        task_rows.append(
            {
                "task_id": task_id,
                "success_count": success_count,
                "success_rate": success_count / len(rows),
                "outcomes": outcomes,
                "steps": [int(row["steps"]) for row in rows],
                "successful_steps": [
                    int(row["steps"])
                    for row, success in zip(rows, outcomes)
                    if success
                ],
                "stable_outcome": success_count in {0, len(rows)},
                "stability_class": (
                    "stable_success"
                    if success_count == len(rows)
                    else "stable_failure"
                    if success_count == 0
                    else "unstable"
                ),
                "query_stable": len(set(queries)) == 1,
                "retrieval_stable": len(set(retrievals)) == 1,
                "retrieval_variants": retrieval_variants,
                "pairwise_flips": task_pair_flips,
            }
        )

    for bucket in pairwise.values():
        comparisons = int(bucket["comparisons"])
        bucket["flip_rate"] = (
            int(bucket["flips"]) / comparisons if comparisons else 0.0
        )
    success_counts = [int(row["success_count"]) for row in repeat_summaries]
    success_mean = sum(success_counts) / len(success_counts)
    total_pairwise_comparisons = sum(
        int(bucket["comparisons"]) for bucket in pairwise.values()
    )
    total_pairwise_flips = sum(int(bucket["flips"]) for bucket in pairwise.values())
    return {
        "repeat_count": len(repeat_results),
        "task_count": len(task_ids),
        "repeat_summaries": repeat_summaries,
        "success_count_mean": success_mean,
        "success_count_population_std": math.sqrt(
            sum((value - success_mean) ** 2 for value in success_counts)
            / len(success_counts)
        ),
        "success_count_min": min(success_counts),
        "success_count_max": max(success_counts),
        "stable_success_task_count": sum(
            row["stability_class"] == "stable_success" for row in task_rows
        ),
        "stable_failure_task_count": sum(
            row["stability_class"] == "stable_failure" for row in task_rows
        ),
        "unstable_task_count": sum(
            row["stability_class"] == "unstable" for row in task_rows
        ),
        "unstable_task_ids": [
            row["task_id"] for row in task_rows if row["stability_class"] == "unstable"
        ],
        "query_unstable_task_ids": [
            row["task_id"] for row in task_rows if not row["query_stable"]
        ],
        "retrieval_unstable_task_ids": [
            row["task_id"] for row in task_rows if not row["retrieval_stable"]
        ],
        "pairwise": pairwise,
        "total_pairwise_comparisons": total_pairwise_comparisons,
        "total_pairwise_flips": total_pairwise_flips,
        "overall_pairwise_flip_rate": (
            total_pairwise_flips / total_pairwise_comparisons
            if total_pairwise_comparisons
            else 0.0
        ),
        "task_stability": task_rows,
    }


def write_baseline_stability_csv(
    path: str | Path, task_rows: Sequence[Mapping[str, Any]]
) -> None:
    fields = (
        "task_id",
        "stability_class",
        "success_count",
        "success_rate",
        "pairwise_flips",
        "query_stable",
        "retrieval_stable",
        "outcomes",
        "steps",
        "successful_steps",
        "retrieval_variants",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in task_rows:
            writer.writerow(
                {
                    field: (
                        json.dumps(row.get(field), ensure_ascii=False)
                        if isinstance(row.get(field), (list, dict))
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def write_utility_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "queue_id",
        "candidate_memory_id",
        "proxy_stratum",
        "coverage_proxy_gain",
        "baseline_success_count",
        "candidate_success_count",
        "utility",
        "candidate_retrieved_task_count",
        "gained_task_ids",
        "lost_task_ids",
        "gained_retrieved_task_ids",
        "lost_retrieved_task_ids",
        "gained_unretrieved_task_ids",
        "lost_unretrieved_task_ids",
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            rendered = {}
            for field in fields:
                value = row.get(field)
                rendered[field] = (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
            writer.writerow(rendered)


def write_task_manifest(path: str | Path, snapshot: Mapping[str, Any]) -> None:
    write_json(path, snapshot["task_manifest"])
