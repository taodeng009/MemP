"""Offline same-state diagnostic for historical cross-task scheduling."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from ProcedureMem.candidate_utility import read_jsonl
from ProcedureMem.cloud_scheduling import (
    OracleCoverageScheduler,
    OracleExactRetrievalScheduler,
    OracleHitQualityScheduler,
    load_cached_embedding,
)


REQUIRED_FILES = (
    "summary.json",
    "results.jsonl",
    "online_trajectories.jsonl",
    "queue_events.jsonl",
    "construction_events.jsonl",
)

HISTORICAL_POLICIES = (
    "historical_cross_task_coverage",
    "historical_cross_task_exact_retrieval",
    "historical_cross_task_hit_quality",
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


def _future_retrieval_metrics(
    available_ids: Sequence[str],
    selected_ids: Sequence[str],
    distances: Mapping[str, Sequence[float]],
    *,
    top_k: int,
    score_threshold: float,
) -> dict[str, Any]:
    """Evaluate virtual post-selection retrieval using the experiment settings."""
    if top_k < 1:
        raise ValueError("retrieval top_k must be at least 1")
    if score_threshold < 0:
        raise ValueError("retrieval score_threshold cannot be negative")
    pool = [*available_ids, *selected_ids]
    if not pool:
        raise ValueError("Future retrieval requires a non-empty virtual memory pool")
    query_count = len(next(iter(distances.values())))
    hit_count = 0
    best_distance_sum_hit = 0.0
    retrieval_utility_sum = 0.0
    for index in range(query_count):
        top_distances = sorted(
            float(distances[memory_id][index]) for memory_id in pool
        )[:top_k]
        retrieved = [
            distance
            for distance in top_distances
            if distance <= score_threshold
        ]
        if retrieved:
            hit_count += 1
            best_distance_sum_hit += retrieved[0]
        retrieval_utility_sum += sum(
            max(0.0, score_threshold - distance) for distance in retrieved
        )
    return {
        "task_count": query_count,
        "hit_count": hit_count,
        "hr": hit_count / query_count if query_count else None,
        "bd": (
            best_distance_sum_hit / hit_count if hit_count else None
        ),
        "ru": (
            retrieval_utility_sum / query_count if query_count else None
        ),
        "best_distance_sum_hit": best_distance_sum_hit,
        "retrieval_utility_sum": retrieval_utility_sum,
    }


def aggregate_retrieval_metrics(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pool interval sufficient statistics into overall HR, conditional BD, and RU."""
    strategy_names = (
        "fifo",
        "historical_cross_task",
        "future_oracle",
    )
    aggregated: dict[str, dict[str, Any]] = {}
    for strategy in strategy_names:
        rows = [
            report["realized_future_retrieval_metrics"][strategy]
            for report in reports
        ]
        task_count = sum(int(row["task_count"]) for row in rows)
        hit_count = sum(int(row["hit_count"]) for row in rows)
        best_distance_sum_hit = sum(
            float(row["best_distance_sum_hit"]) for row in rows
        )
        retrieval_utility_sum = sum(
            float(row["retrieval_utility_sum"]) for row in rows
        )
        aggregated[strategy] = {
            "task_count": task_count,
            "hit_count": hit_count,
            "hr": hit_count / task_count if task_count else None,
            "bd": (
                best_distance_sum_hit / hit_count if hit_count else None
            ),
            "ru": (
                retrieval_utility_sum / task_count if task_count else None
            ),
            "best_distance_sum_hit": best_distance_sum_hit,
            "retrieval_utility_sum": retrieval_utility_sum,
        }
    return aggregated


def _select_policy(
    policy: str,
    pending_ids: Sequence[str],
    capacity: int,
    *,
    available_ids: Sequence[str],
    queries: Sequence[str],
    distances: Mapping[str, Sequence[float]],
    top_k: int,
    score_threshold: float,
    hit_quality_alpha: float,
    hit_quality_metric: str,
    eligibility_by_id: Mapping[str, Sequence[bool]] | None = None,
):
    def scorer(_queries, requested):
        return {item_id: distances[item_id] for item_id in requested}

    if policy == "historical_cross_task_coverage":
        return OracleCoverageScheduler().select(
            pending_ids,
            capacity,
            available_ids=available_ids,
            next_interval_queries=queries,
            distance_scorer=scorer,
            eligibility_by_id=eligibility_by_id,
        )
    if policy == "historical_cross_task_exact_retrieval":
        return OracleExactRetrievalScheduler().select(
            pending_ids,
            capacity,
            available_ids=available_ids,
            future_queries=queries,
            distance_scorer=scorer,
            top_k=top_k,
            score_threshold=score_threshold,
            eligibility_by_id=eligibility_by_id,
        )
    if policy == "historical_cross_task_hit_quality":
        return OracleHitQualityScheduler().select(
            pending_ids,
            capacity,
            available_ids=available_ids,
            future_queries=queries,
            distance_scorer=scorer,
            top_k=top_k,
            score_threshold=score_threshold,
            alpha=hit_quality_alpha,
            metric=hit_quality_metric,
            eligibility_by_id=eligibility_by_id,
        )
    raise ValueError(f"Unsupported historical policy: {policy}")


def _first_step_policy_scores(
    policy: str,
    pending_ids: Sequence[str],
    available_ids: Sequence[str],
    distances: Mapping[str, Sequence[float]],
    *,
    top_k: int,
    score_threshold: float,
    hit_quality_alpha: float,
    hit_quality_metric: str,
    eligibility_by_id: Mapping[str, Sequence[bool]] | None = None,
) -> tuple[dict[str, tuple[float, ...]], dict[str, dict[str, Any]]]:
    """Return each candidate's score in the scheduler's first greedy step."""
    query_count = len(next(iter(distances.values())))
    eligibility = {
        memory_id: tuple(
            bool(value) for value in eligibility_by_id[memory_id]
        )
        if eligibility_by_id is not None
        else (True,) * query_count
        for memory_id in (*available_ids, *pending_ids)
    }
    best = [
        min(
            (
                float(distances[memory_id][index])
                for memory_id in available_ids
                if eligibility[memory_id][index]
            ),
            default=float("inf"),
        )
        for index in range(query_count)
    ]
    if policy == "historical_cross_task_coverage":
        scores: dict[str, tuple[float, ...]] = {}
        details: dict[str, dict[str, Any]] = {}
        for memory_id in pending_ids:
            if available_ids:
                newly_covered = sum(
                    eligibility[memory_id][index]
                    and not math.isfinite(best[index])
                    for index in range(query_count)
                )
                finite_gain = sum(
                    max(
                        0.0,
                        best[index] - float(distances[memory_id][index]),
                    )
                    for index in range(query_count)
                    if eligibility[memory_id][index]
                    and math.isfinite(best[index])
                )
                scores[memory_id] = (
                    float(newly_covered),
                    float(finite_gain),
                )
                details[memory_id] = {
                    "newly_covered": float(newly_covered),
                    "finite_gain": float(finite_gain),
                }
            else:
                distance_sum = sum(
                    float(distances[memory_id][index])
                    for index in range(query_count)
                    if eligibility[memory_id][index]
                )
                scores[memory_id] = (-float(distance_sum),)
                details[memory_id] = {
                    "bootstrap_distance_sum": float(distance_sum)
                }
        return scores, details

    current_top = [
        tuple(
            sorted(
                float(distances[memory_id][index])
                for memory_id in available_ids
                if eligibility[memory_id][index]
            )[:top_k]
        )
        for index in range(query_count)
    ]
    utility = lambda values: float(
        sum(max(0.0, score_threshold - value) for value in values)
    )
    before_utility = sum(utility(values) for values in current_top)
    updated_top: dict[str, list[tuple[float, ...]]] = {}
    utility_gains: dict[str, float] = {}
    hit_gains: dict[str, float] = {}
    bd_gains: dict[str, float] = {}
    for memory_id in pending_ids:
        updated_top[memory_id] = [
            tuple(
                sorted(
                    (
                        *current_top[index],
                        *(
                            (float(distances[memory_id][index]),)
                            if eligibility[memory_id][index]
                            else ()
                        ),
                    )
                )[:top_k]
            )
            for index in range(query_count)
        ]
        utility_gains[memory_id] = max(
            0.0,
            sum(utility(values) for values in updated_top[memory_id])
            - before_utility,
        )
        hit_gains[memory_id] = float(
            sum(
                eligibility[memory_id][index]
                and best[index] > score_threshold
                and float(distances[memory_id][index]) <= score_threshold
                for index in range(query_count)
            )
        )
        bd_gains[memory_id] = float(
            sum(
                max(
                    0.0,
                    best[index] - float(distances[memory_id][index]),
                )
                for index in range(query_count)
                if eligibility[memory_id][index]
                and math.isfinite(best[index])
            )
        )
    if policy == "historical_cross_task_exact_retrieval":
        return (
            {memory_id: (utility_gains[memory_id],) for memory_id in pending_ids},
            {
                memory_id: {"retrieval_utility_gain": utility_gains[memory_id]}
                for memory_id in pending_ids
            },
        )

    if hit_quality_metric not in {"bd", "ru"}:
        raise ValueError("Hit quality metric must be bd or ru")
    bootstrap = hit_quality_metric == "bd" and not available_ids
    if bootstrap:
        scores = {}
        details = {}
        for memory_id in pending_ids:
            distance_sum = sum(
                float(distances[memory_id][index])
                for index in range(query_count)
                if eligibility[memory_id][index]
            )
            scores[memory_id] = (-float(distance_sum),)
            details[memory_id] = {
                "coverage_bootstrap": True,
                "bootstrap_distance_sum": float(distance_sum),
                "hit_gain": hit_gains[memory_id],
            }
        return scores, details

    quality_gains = bd_gains if hit_quality_metric == "bd" else utility_gains
    hit_max = max(hit_gains.values(), default=0.0)
    quality_max = max(quality_gains.values(), default=0.0)
    normalized_hit = {
        memory_id: hit_gains[memory_id] / (hit_max + 1e-8)
        for memory_id in pending_ids
    }
    normalized_quality = {
        memory_id: quality_gains[memory_id] / (quality_max + 1e-8)
        for memory_id in pending_ids
    }
    priorities = {
        memory_id: hit_quality_alpha * normalized_hit[memory_id]
        + (1.0 - hit_quality_alpha) * normalized_quality[memory_id]
        for memory_id in pending_ids
    }
    ordering = (
        quality_gains
        if hit_quality_alpha == 0
        else hit_gains
        if hit_quality_alpha == 1
        else priorities
    )
    return (
        {memory_id: (float(ordering[memory_id]),) for memory_id in pending_ids},
        {
            memory_id: {
                "coverage_bootstrap": False,
                "hit_gain": hit_gains[memory_id],
                "quality_gain": quality_gains[memory_id],
                "normalized_hit_gain": normalized_hit[memory_id],
                "normalized_quality_gain": normalized_quality[memory_id],
                "priority": priorities[memory_id],
            }
            for memory_id in pending_ids
        },
    )


def analyze_same_state(
    snapshot: Mapping[str, Any],
    embedding: Any,
    *,
    capacity: int,
    historical_policy: str = "historical_cross_task_coverage",
    hit_quality_alpha: float = 0.5,
    hit_quality_metric: str = "ru",
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
    if historical_policy not in HISTORICAL_POLICIES:
        raise ValueError(f"Unsupported historical policy: {historical_policy}")
    if not math.isfinite(hit_quality_alpha) or not 0 <= hit_quality_alpha <= 1:
        raise ValueError("hit_quality_alpha must be finite and in [0, 1]")
    if hit_quality_metric not in {"bd", "ru"}:
        raise ValueError("hit_quality_metric must be bd or ru")
    source_parameters = snapshot.get("source_parameters") or {}
    top_k_value = source_parameters.get("top_k")
    retrieval_top_k = int(3 if top_k_value is None else top_k_value)
    threshold_value = source_parameters.get("score_threshold")
    retrieval_threshold = float(
        0.5 if threshold_value is None else threshold_value
    )
    historical_queries = [str(row["query"]) for row in history]
    future_queries = [str(row["query"]) for row in future]
    historical = _select_policy(
        historical_policy,
        pending_ids,
        capacity,
        available_ids=available_ids,
        queries=historical_queries,
        distances=historical_distances,
        top_k=retrieval_top_k,
        score_threshold=retrieval_threshold,
        hit_quality_alpha=hit_quality_alpha,
        hit_quality_metric=hit_quality_metric,
        eligibility_by_id=eligibility,
    )
    oracle = _select_policy(
        historical_policy,
        pending_ids,
        capacity,
        available_ids=available_ids,
        queries=future_queries,
        distances=future_distances,
        top_k=retrieval_top_k,
        score_threshold=retrieval_threshold,
        hit_quality_alpha=hit_quality_alpha,
        hit_quality_metric=hit_quality_metric,
    )
    fifo_ids = pending_ids[:capacity]
    historical_ids = list(historical.memory_ids)
    oracle_ids = list(oracle.memory_ids)
    historical_first_scores, historical_first_details = _first_step_policy_scores(
        historical_policy,
        pending_ids,
        available_ids,
        historical_distances,
        top_k=retrieval_top_k,
        score_threshold=retrieval_threshold,
        hit_quality_alpha=hit_quality_alpha,
        hit_quality_metric=hit_quality_metric,
        eligibility_by_id=eligibility,
    )
    oracle_first_scores, oracle_first_details = _first_step_policy_scores(
        historical_policy,
        pending_ids,
        available_ids,
        future_distances,
        top_k=retrieval_top_k,
        score_threshold=retrieval_threshold,
        hit_quality_alpha=hit_quality_alpha,
        hit_quality_metric=hit_quality_metric,
    )

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
    retrieval_metrics = {
        "fifo": _future_retrieval_metrics(
            available_ids,
            fifo_ids,
            future_distances,
            top_k=retrieval_top_k,
            score_threshold=retrieval_threshold,
        ),
        "historical_cross_task": _future_retrieval_metrics(
            available_ids,
            historical_ids,
            future_distances,
            top_k=retrieval_top_k,
            score_threshold=retrieval_threshold,
        ),
        "future_oracle": _future_retrieval_metrics(
            available_ids,
            oracle_ids,
            future_distances,
            top_k=retrieval_top_k,
            score_threshold=retrieval_threshold,
        ),
    }
    oracle_gain = gains["future_oracle"]
    overlap = set(historical_ids) & set(oracle_ids)
    fifo_overlap = set(fifo_ids) & set(oracle_ids)
    if historical_policy == "historical_cross_task_coverage":
        first_step_details = {
            item_id: (
                {
                    "historical_newly_covered": historical_first_details[item_id][
                        "newly_covered"
                    ],
                    "historical_finite_gain": historical_first_details[item_id][
                        "finite_gain"
                    ],
                    "future_oracle_gain": oracle_first_details[item_id][
                        "finite_gain"
                    ],
                }
                if available_ids
                else {
                    "historical_bootstrap_distance_sum": historical_first_details[
                        item_id
                    ]["bootstrap_distance_sum"],
                    "future_oracle_bootstrap_distance_sum": oracle_first_details[
                        item_id
                    ]["bootstrap_distance_sum"],
                }
            )
            for item_id in pending_ids
        }
    else:
        first_step_details = {
            item_id: {
                "historical": historical_first_details[item_id],
                "future_oracle": oracle_first_details[item_id],
            }
            for item_id in pending_ids
        }
    return {
        "snapshot_interval": int(snapshot["snapshot_interval"]),
        "historical_policy": historical_policy,
        "future_oracle_policy": historical_policy.replace(
            "historical_cross_task_", "oracle_"
        ),
        "hit_quality_alpha": (
            hit_quality_alpha
            if historical_policy == "historical_cross_task_hit_quality"
            else None
        ),
        "hit_quality_metric": (
            hit_quality_metric
            if historical_policy == "historical_cross_task_hit_quality"
            else None
        ),
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
        "first_step_scores": first_step_details,
        "realized_future_coverage_gain": gains,
        "realized_future_nearest_distance_sum": distance_sums,
        "future_retrieval_config": {
            "top_k": retrieval_top_k,
            "score_threshold": retrieval_threshold,
        },
        "realized_future_retrieval_metrics": retrieval_metrics,
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
    parser.add_argument(
        "--historical-policy",
        choices=HISTORICAL_POLICIES,
        default="historical_cross_task_coverage",
    )
    parser.add_argument("--hit-quality-alpha", type=float, default=0.5)
    parser.add_argument(
        "--hit-quality-metric", choices=("ru", "bd"), default="ru"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    capacity = (
        args.capacity
        if args.capacity is not None
        else source_construction_capacity(args.source_run_dir)
    )
    if capacity < 1:
        parser.error("--capacity must be at least 1")
    if not math.isfinite(args.hit_quality_alpha) or not 0 <= args.hit_quality_alpha <= 1:
        parser.error("--hit-quality-alpha must be finite and in [0, 1]")
    intervals = (
        [args.snapshot_interval]
        if args.snapshot_interval is not None
        else snapshot_intervals(args.source_run_dir)
    )
    if not intervals:
        raise ValueError("Source run has no eligible snapshot intervals")
    if args.historical_policy == "historical_cross_task_coverage":
        output_stem = "historical_same_state"
    elif args.historical_policy == "historical_cross_task_exact_retrieval":
        output_stem = "historical_exact_retrieval_same_state"
    else:
        alpha_label = format(args.hit_quality_alpha, "g").replace(".", "p")
        output_stem = (
            f"historical_hit_quality_{args.hit_quality_metric}_alpha{alpha_label}"
            "_same_state"
        )
    output = args.output or args.source_run_dir / (
        f"{output_stem}_interval_{args.snapshot_interval}.json"
        if args.snapshot_interval is not None
        else f"{output_stem}_all_intervals.json"
    )
    embedding = load_cached_embedding(output.parent / "same_state_embedding_cache")
    reports = [
        analyze_same_state(
            load_same_state_snapshot(args.source_run_dir, interval),
            embedding,
            capacity=capacity,
            historical_policy=args.historical_policy,
            hit_quality_alpha=args.hit_quality_alpha,
            hit_quality_metric=args.hit_quality_metric,
        )
        for interval in intervals
    ]
    report = (
        reports[0]
        if args.snapshot_interval is not None
        else {
            "source_run_dir": str(args.source_run_dir.expanduser().resolve()),
            "historical_policy": args.historical_policy,
            "future_oracle_policy": reports[0]["future_oracle_policy"],
            "hit_quality_alpha": reports[0]["hit_quality_alpha"],
            "hit_quality_metric": reports[0]["hit_quality_metric"],
            "capacity": capacity,
            "snapshot_count": len(reports),
            "snapshot_intervals": intervals,
            "future_retrieval_config": reports[0]["future_retrieval_config"],
            "aggregate_future_retrieval_metrics": aggregate_retrieval_metrics(
                reports
            ),
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
