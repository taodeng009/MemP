"""Offline same-state diagnostic for historical cross-task coverage."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ProcedureMem.candidate_utility import read_jsonl
from ProcedureMem.cloud_scheduling import OracleCoverageScheduler, load_cached_embedding


REQUIRED_FILES = (
    "summary.json",
    "results.jsonl",
    "online_trajectories.jsonl",
    "queue_events.jsonl",
    "construction_events.jsonl",
)


def load_same_state_snapshot(
    source_run_dir: str | Path, snapshot_interval: int
) -> dict[str, Any]:
    """Reconstruct a fixed (H_t, M_t, Q_t) and the realized next interval."""
    source = Path(source_run_dir).expanduser().resolve()
    if snapshot_interval < 0:
        raise ValueError("snapshot_interval cannot be negative")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Source run is missing required files: " + ", ".join(missing)
        )
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    parameters = summary.get("parameters") or {}
    if parameters.get("condition_mode") != "online_construction":
        raise ValueError("Source run must be an online_construction condition")
    if int(parameters.get("warm_start_count") or 0) != 0:
        raise ValueError("V1 same-state diagnostic requires a zero-warm-start run")

    results = read_jsonl(source / "results.jsonl")
    trajectories = read_jsonl(source / "online_trajectories.jsonl")
    queue_events = read_jsonl(source / "queue_events.jsonl")
    construction_events = read_jsonl(source / "construction_events.jsonl")
    matching_events = [
        row
        for row in queue_events
        if int(row.get("interval_id", -1)) == snapshot_interval
    ]
    if len(matching_events) != 1:
        raise ValueError(
            f"Expected one queue event for interval {snapshot_interval}, "
            f"found {len(matching_events)}"
        )
    pending_ids = list(
        matching_events[0].get("pending_queue_ids_before_selection") or []
    )
    if not pending_ids or len(pending_ids) != len(set(pending_ids)):
        raise ValueError("Snapshot pending queue must be non-empty and unique")

    trajectory_by_id = {str(row["queue_id"]): row for row in trajectories}
    if len(trajectory_by_id) != len(trajectories):
        raise ValueError("Online trajectories contain duplicate queue IDs")
    missing_pending = [item for item in pending_ids if item not in trajectory_by_id]
    if missing_pending:
        raise ValueError(
            "Pending queue IDs missing trajectories: "
            + ", ".join(missing_pending[:5])
        )
    pending = [
        {
            "memory_id": queue_id,
            "queue_id": queue_id,
            "pending_order": order,
            "source_task_index": int(trajectory_by_id[queue_id]["task_index"]),
            "query": str(trajectory_by_id[queue_id]["query"]).strip(),
        }
        for order, queue_id in enumerate(pending_ids)
    ]

    available = []
    for event in construction_events:
        if (
            event.get("construction_result") != "success"
            or event.get("available_from_interval") is None
            or int(event["available_from_interval"]) > snapshot_interval
        ):
            continue
        queue_id = str(event["queue_id"])
        trajectory = trajectory_by_id.get(queue_id)
        if trajectory is None:
            raise ValueError(
                f"Available memory {event.get('constructed_memory_id')} has no trajectory"
            )
        available.append(
            {
                "memory_id": str(event["constructed_memory_id"]),
                "source_task_index": int(trajectory["task_index"]),
                "query": str(trajectory["query"]).strip(),
            }
        )
    history = [
        {
            "task_id": str(row["task_id"]),
            "task_index": int(row["task_index"]),
            "query": str(row["query"]).strip(),
            "interval_id": int(row["interval_id"]),
        }
        for row in results
        if int(row.get("interval_id", -1)) <= snapshot_interval
    ]
    history.sort(key=lambda row: int(row["task_index"]))
    future = [
        {
            "task_id": str(row["task_id"]),
            "task_index": int(row["task_index"]),
            "query": str(row["query"]).strip(),
        }
        for row in results
        if int(row.get("interval_id", -1)) == snapshot_interval + 1
    ]
    future.sort(key=lambda row: int(row["task_index"]))
    if not history or not future:
        raise ValueError("Snapshot requires non-empty history and next interval")
    history_indices = {int(row["task_index"]) for row in history}
    missing_sources = sorted(
        {
            int(row["source_task_index"])
            for row in (*available, *pending)
            if int(row["source_task_index"]) not in history_indices
        }
    )
    if missing_sources:
        raise ValueError(
            "Memory sources absent from historical tasks: "
            + ", ".join(str(value) for value in missing_sources[:5])
        )
    return {
        "source_run_dir": str(source),
        "snapshot_interval": snapshot_interval,
        "history": history,
        "available_memories": available,
        "pending_candidates": pending,
        "future_tasks": future,
        "source_parameters": parameters,
    }


def _distance_matrix(
    memories: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    embedding: Any,
) -> dict[str, tuple[float, ...]]:
    memory_vectors = embedding.embed_documents([str(row["query"]) for row in memories])
    task_vectors = embedding.embed_documents([str(row["query"]) for row in tasks])
    return {
        str(memory["memory_id"]): tuple(
            float(
                sum(
                    (float(left) - float(right)) ** 2
                    for left, right in zip(memory_vector, task_vector)
                )
            )
            for task_vector in task_vectors
        )
        for memory, memory_vector in zip(memories, memory_vectors)
    }


def _average_descending_ranks(scores: Mapping[str, tuple[float, ...]]) -> dict[str, float]:
    groups: dict[tuple[float, ...], list[str]] = {}
    for item_id, score in scores.items():
        groups.setdefault(tuple(score), []).append(item_id)
    ranks: dict[str, float] = {}
    position = 1
    for score in sorted(groups, reverse=True):
        ids = groups[score]
        average = (position + position + len(ids) - 1) / 2.0
        for item_id in ids:
            ranks[item_id] = average
        position += len(ids)
    return ranks


def _spearman(
    left_scores: Mapping[str, tuple[float, ...]],
    right_scores: Mapping[str, tuple[float, ...]],
) -> float | None:
    if set(left_scores) != set(right_scores) or len(left_scores) < 2:
        return None
    left = _average_descending_ranks(left_scores)
    right = _average_descending_ranks(right_scores)
    ids = list(left_scores)
    left_mean = sum(left[item_id] for item_id in ids) / len(ids)
    right_mean = sum(right[item_id] for item_id in ids) / len(ids)
    numerator = sum(
        (left[item_id] - left_mean) * (right[item_id] - right_mean)
        for item_id in ids
    )
    left_scale = math.sqrt(
        sum((left[item_id] - left_mean) ** 2 for item_id in ids)
    )
    right_scale = math.sqrt(
        sum((right[item_id] - right_mean) ** 2 for item_id in ids)
    )
    if left_scale == 0 or right_scale == 0:
        return None
    return float(numerator / (left_scale * right_scale))


def _future_coverage_gain(
    available_ids: Sequence[str],
    selected_ids: Sequence[str],
    distances: Mapping[str, Sequence[float]],
) -> float | None:
    if not available_ids:
        return None
    query_count = len(next(iter(distances.values())))
    baseline = [
        min(float(distances[memory_id][index]) for memory_id in available_ids)
        for index in range(query_count)
    ]
    return float(
        sum(
            max(
                0.0,
                baseline[index]
                - min(float(distances[item_id][index]) for item_id in selected_ids),
            )
            for index in range(query_count)
        )
    ) if selected_ids else 0.0


def _future_distance_sum(
    available_ids: Sequence[str],
    selected_ids: Sequence[str],
    distances: Mapping[str, Sequence[float]],
) -> float:
    pool = [*available_ids, *selected_ids]
    if not pool:
        raise ValueError("Future distance requires a non-empty virtual memory pool")
    query_count = len(next(iter(distances.values())))
    return float(
        sum(
            min(float(distances[memory_id][index]) for memory_id in pool)
            for index in range(query_count)
        )
    )


def analyze_same_state(
    snapshot: Mapping[str, Any], embedding: Any, *, capacity: int
) -> dict[str, Any]:
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    available = list(snapshot["available_memories"])
    pending = list(snapshot["pending_candidates"])
    history = list(snapshot["history"])
    future = list(snapshot["future_tasks"])
    if not pending or not history or not future:
        raise ValueError("Same-state diagnostic received an incomplete snapshot")
    memories = [*available, *pending]
    memory_ids = [str(row["memory_id"]) for row in memories]
    available_ids = [str(row["memory_id"]) for row in available]
    pending_ids = [str(row["memory_id"]) for row in pending]
    historical_distances = _distance_matrix(memories, history, embedding)
    future_distances = _distance_matrix(memories, future, embedding)
    sources = {
        str(row["memory_id"]): int(row["source_task_index"])
        for row in memories
    }
    eligibility = {
        memory_id: tuple(
            sources[memory_id] != int(task["task_index"]) for task in history
        )
        for memory_id in memory_ids
    }

    def scorer(matrix):
        return lambda _queries, requested: {
            item_id: matrix[item_id] for item_id in requested
        }

    scheduler = OracleCoverageScheduler()
    historical = scheduler.select(
        pending_ids,
        capacity,
        available_ids=available_ids,
        next_interval_queries=[str(row["query"]) for row in history],
        distance_scorer=scorer(historical_distances),
        eligibility_by_id=eligibility,
    )
    oracle = scheduler.select(
        pending_ids,
        capacity,
        available_ids=available_ids,
        next_interval_queries=[str(row["query"]) for row in future],
        distance_scorer=scorer(future_distances),
    )
    fifo_ids = pending_ids[:capacity]
    historical_ids = list(historical.memory_ids)
    oracle_ids = list(oracle.memory_ids)

    historical_first_scores: dict[str, tuple[float, ...]] = {}
    oracle_first_scores: dict[str, tuple[float, ...]] = {}
    historical_best = [
        min(
            (
                historical_distances[memory_id][index]
                for memory_id in available_ids
                if eligibility[memory_id][index]
            ),
            default=float("inf"),
        )
        for index in range(len(history))
    ]
    oracle_best = [
        min(
            (future_distances[memory_id][index] for memory_id in available_ids),
            default=float("inf"),
        )
        for index in range(len(future))
    ]
    for memory_id in pending_ids:
        historical_new = sum(
            eligibility[memory_id][index] and not math.isfinite(historical_best[index])
            for index in range(len(history))
        )
        historical_gain = sum(
            max(
                0.0,
                historical_best[index] - historical_distances[memory_id][index],
            )
            for index in range(len(history))
            if eligibility[memory_id][index]
            and math.isfinite(historical_best[index])
        )
        oracle_gain = sum(
            max(0.0, oracle_best[index] - future_distances[memory_id][index])
            for index in range(len(future))
        )
        if available_ids:
            historical_first_scores[memory_id] = (
                float(historical_new),
                float(historical_gain),
            )
            oracle_first_scores[memory_id] = (float(oracle_gain),)
        else:
            historical_distance_sum = sum(
                historical_distances[memory_id][index]
                for index in range(len(history))
                if eligibility[memory_id][index]
            )
            oracle_distance_sum = sum(future_distances[memory_id])
            historical_first_scores[memory_id] = (
                -float(historical_distance_sum),
            )
            oracle_first_scores[memory_id] = (-float(oracle_distance_sum),)

    gains = {
        "fifo": _future_coverage_gain(available_ids, fifo_ids, future_distances),
        "historical_cross_task": _future_coverage_gain(
            available_ids, historical_ids, future_distances
        ),
        "future_oracle": _future_coverage_gain(
            available_ids, oracle_ids, future_distances
        ),
    }
    distance_sums = {
        "fifo": _future_distance_sum(available_ids, fifo_ids, future_distances),
        "historical_cross_task": _future_distance_sum(
            available_ids, historical_ids, future_distances
        ),
        "future_oracle": _future_distance_sum(
            available_ids, oracle_ids, future_distances
        ),
    }
    oracle_gain = gains["future_oracle"]
    overlap = set(historical_ids) & set(oracle_ids)
    fifo_overlap = set(fifo_ids) & set(oracle_ids)
    return {
        "snapshot_interval": int(snapshot["snapshot_interval"]),
        "capacity": min(capacity, len(pending_ids)),
        "history_count": len(history),
        "available_count": len(available_ids),
        "pending_count": len(pending_ids),
        "future_task_count": len(future),
        "selections": {
            "fifo": fifo_ids,
            "historical_cross_task": historical_ids,
            "future_oracle": oracle_ids,
        },
        "top_cc_overlap": {
            "historical_vs_oracle_count": len(overlap),
            "historical_vs_oracle_fraction": len(overlap) / len(oracle_ids),
            "historical_vs_oracle_jaccard": len(overlap)
            / len(set(historical_ids) | set(oracle_ids)),
            "fifo_vs_oracle_count": len(fifo_overlap),
            "fifo_vs_oracle_fraction": len(fifo_overlap) / len(oracle_ids),
        },
        "historical_vs_oracle_first_step_spearman": _spearman(
            historical_first_scores, oracle_first_scores
        ),
        "first_step_scores": {
            item_id: (
                {
                    "historical_newly_covered": historical_first_scores[item_id][0],
                    "historical_finite_gain": historical_first_scores[item_id][1],
                    "future_oracle_gain": oracle_first_scores[item_id][0],
                }
                if available_ids
                else {
                    "historical_bootstrap_distance_sum": -historical_first_scores[
                        item_id
                    ][0],
                    "future_oracle_bootstrap_distance_sum": -oracle_first_scores[
                        item_id
                    ][0],
                }
            )
            for item_id in pending_ids
        },
        "realized_future_coverage_gain": gains,
        "realized_future_nearest_distance_sum": distance_sums,
        "oracle_gain_recovery": {
            "historical": (
                gains["historical_cross_task"] / oracle_gain
                if oracle_gain is not None
                and oracle_gain > 0
                and gains["historical_cross_task"] is not None
                else None
            ),
            "fifo": (
                gains["fifo"] / oracle_gain
                if oracle_gain is not None
                and oracle_gain > 0
                and gains["fifo"] is not None
                else None
            ),
            "historical_regret": (
                oracle_gain - gains["historical_cross_task"]
                if oracle_gain is not None
                and gains["historical_cross_task"] is not None
                else None
            ),
            "fifo_regret": (
                oracle_gain - gains["fifo"]
                if oracle_gain is not None and gains["fifo"] is not None
                else None
            ),
        },
        "oracle_distance_recovery_from_fifo": (
            (
                distance_sums["fifo"] - distance_sums["historical_cross_task"]
            )
            / (distance_sums["fifo"] - distance_sums["future_oracle"])
            if distance_sums["fifo"] > distance_sums["future_oracle"]
            else None
        ),
    }


def snapshot_intervals(source_run_dir: str | Path) -> list[int]:
    """Return every queue boundary that has pending candidates and a next interval."""
    source = Path(source_run_dir).expanduser().resolve()
    queue_events = read_jsonl(source / "queue_events.jsonl")
    result_intervals = {
        int(row["interval_id"])
        for row in read_jsonl(source / "results.jsonl")
    }
    return sorted(
        int(event["interval_id"])
        for event in queue_events
        if event.get("pending_queue_ids_before_selection")
        and int(event["interval_id"]) + 1 in result_intervals
    )


def source_construction_capacity(source_run_dir: str | Path) -> int:
    source = Path(source_run_dir).expanduser().resolve()
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    value = (summary.get("parameters") or {}).get("construction_capacity")
    if value is None or int(value) < 1:
        raise ValueError(
            "Source run has no positive construction_capacity; pass --capacity"
        )
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--snapshot-interval", type=int)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    capacity = (
        args.capacity
        if args.capacity is not None
        else source_construction_capacity(args.source_run_dir)
    )
    if capacity < 1:
        parser.error("--capacity must be at least 1")
    intervals = (
        [args.snapshot_interval]
        if args.snapshot_interval is not None
        else snapshot_intervals(args.source_run_dir)
    )
    if not intervals:
        raise ValueError("Source run has no eligible snapshot intervals")
    output = args.output or (
        args.source_run_dir
        / (
            f"historical_same_state_interval_{args.snapshot_interval}.json"
            if args.snapshot_interval is not None
            else "historical_same_state_all_intervals.json"
        )
    )
    embedding = load_cached_embedding(output.parent / "same_state_embedding_cache")
    reports = [
        analyze_same_state(
            load_same_state_snapshot(args.source_run_dir, interval),
            embedding,
            capacity=capacity,
        )
        for interval in intervals
    ]
    report = (
        reports[0]
        if args.snapshot_interval is not None
        else {
            "source_run_dir": str(args.source_run_dir.expanduser().resolve()),
            "capacity": capacity,
            "snapshot_count": len(reports),
            "snapshot_intervals": intervals,
            "snapshots": reports,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
