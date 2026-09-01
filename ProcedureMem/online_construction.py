"""Queue-based online workflow-memory construction primitives for ALFWorld."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ProcedureMem.cloud_scheduling import (
    GreedyNoveltyScheduler,
    OracleCoverageScheduler,
    OracleExactRetrievalScheduler,
    ScheduleSelection,
    load_candidate_memories,
    select_warm_start_ids,
)


ONLINE_POLICIES = (
    "fifo",
    "fifo_shortest_first",
    "random",
    "greedy_novelty",
    "oracle_coverage",
    "oracle_exact_retrieval",
    "oracle_exact_retrieval_historical_utility",
)

EXACT_RETRIEVAL_POLICIES = {
    "oracle_exact_retrieval",
    "oracle_exact_retrieval_historical_utility",
}
HISTORICAL_UTILITY_MIN_COUNT = 5
HISTORICAL_UTILITY_LAMBDA = 1.0
HISTORICAL_UTILITY_EPSILON = 1e-8


def parse_oracle_lookahead_horizon(value: str) -> int | str:
    """Parse a positive interval horizon or the all-remaining sentinel."""
    normalized = str(value).strip().lower()
    if normalized == "all_remaining":
        return normalized
    try:
        horizon = int(normalized)
    except ValueError as exc:
        raise ValueError(
            "Oracle lookahead horizon must be a positive integer or all_remaining"
        ) from exc
    if horizon < 1 or str(horizon) != normalized:
        raise ValueError(
            "Oracle lookahead horizon must be a positive integer or all_remaining"
        )
    return horizon


def oracle_future_query_window(
    task_queries: Sequence[str],
    *,
    current_end: int,
    interval_size: int,
    requested_horizon: int | str,
) -> tuple[tuple[str, ...], int]:
    """Return future queries and the clamped effective interval horizon."""
    if interval_size < 1:
        raise ValueError("Interval size must be at least 1")
    if current_end < 0 or current_end > len(task_queries):
        raise ValueError("Current interval end is outside the task query sequence")
    remaining_queries = len(task_queries) - current_end
    remaining_intervals = (
        (remaining_queries + interval_size - 1) // interval_size
        if remaining_queries
        else 0
    )
    if requested_horizon == "all_remaining":
        effective_horizon = remaining_intervals
    elif isinstance(requested_horizon, int) and requested_horizon >= 1:
        effective_horizon = min(requested_horizon, remaining_intervals)
    else:
        raise ValueError(
            "Oracle lookahead horizon must be a positive integer or all_remaining"
        )
    future_end = min(
        current_end + effective_horizon * interval_size,
        len(task_queries),
    )
    return tuple(task_queries[current_end:future_end]), effective_horizon


@dataclass
class OnlineTrajectoryCandidate:
    queue_id: str
    task_id: str
    task_index: int
    task_type: str
    query: str
    trajectory: list[dict[str, str]]
    steps: int
    arrival_interval: int
    arrival_order: int
    selected_count: int = 0
    last_selected_interval: int | None = None
    last_construction_result: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def trajectory_queue_id(
    *, task_id: str, task_index: int, trajectory: Sequence[Mapping[str, str]]
) -> str:
    payload = {
        "task_id": task_id,
        "task_index": int(task_index),
        "trajectory": list(trajectory),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "traj_" + hashlib.sha256(encoded).hexdigest()[:16]


class OnlineConstructionQueue:
    """Persistent arrival-ordered queue of successful task trajectories."""

    def __init__(self) -> None:
        self._pending: dict[str, OnlineTrajectoryCandidate] = {}

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def pending_ids(self) -> tuple[str, ...]:
        return tuple(self._pending)

    def add(self, candidate: OnlineTrajectoryCandidate) -> None:
        if candidate.queue_id in self._pending:
            raise ValueError(f"Queue item already exists: {candidate.queue_id}")
        self._pending[candidate.queue_id] = candidate

    def get(self, queue_id: str) -> OnlineTrajectoryCandidate:
        try:
            return self._pending[queue_id]
        except KeyError as exc:
            raise KeyError(f"Unknown pending queue ID: {queue_id}") from exc

    def remove_constructed(self, queue_id: str) -> OnlineTrajectoryCandidate:
        return self._pending.pop(queue_id)

    def snapshot(self) -> list[dict[str, Any]]:
        return [candidate.as_dict() for candidate in self._pending.values()]


class FIFOScheduler:
    def select(
        self, pending_ids: Iterable[str], capacity: int, **_: Any
    ) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        return ScheduleSelection(memory_ids=tuple(pending_ids)[:capacity])


class FIFOShortestFirstScheduler:
    """Preserve interval FIFO and prefer shorter successes within an interval."""

    def select(
        self,
        pending_ids: Iterable[str],
        capacity: int,
        *,
        candidates: Mapping[str, OnlineTrajectoryCandidate],
        **_: Any,
    ) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        ids = list(pending_ids)
        unknown = set(ids) - set(candidates)
        if unknown:
            raise ValueError(
                "Unknown shortest-first candidates: "
                + ", ".join(sorted(unknown)[:5])
            )
        selected = sorted(
            ids,
            key=lambda queue_id: (
                candidates[queue_id].arrival_interval,
                candidates[queue_id].steps,
                candidates[queue_id].task_index,
                queue_id,
            ),
        )[:capacity]
        return ScheduleSelection(memory_ids=tuple(selected))


class OnlineRandomScheduler:
    """Optional deterministic diagnostic baseline for a dynamic queue."""

    def __init__(self, *, seed: int) -> None:
        self._random = random.Random(seed)

    def select(
        self, pending_ids: Iterable[str], capacity: int, **_: Any
    ) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        ids = list(pending_ids)
        self._random.shuffle(ids)
        return ScheduleSelection(memory_ids=tuple(ids[:capacity]))


def _distance_matrix(
    queries_by_id: Mapping[str, str], embedding: Any
) -> dict[str, dict[str, float]]:
    ordered_ids = list(queries_by_id)
    if not ordered_ids:
        return {}
    vectors = np.asarray(
        embedding.embed_documents([queries_by_id[item_id] for item_id in ordered_ids]),
        dtype=np.float32,
    )
    squared_norms = np.sum(vectors * vectors, axis=1, keepdims=True)
    distances = np.maximum(
        squared_norms + squared_norms.T - 2.0 * (vectors @ vectors.T), 0.0
    )
    return {
        item_id: {
            reference_id: float(distances[row, column])
            for column, reference_id in enumerate(ordered_ids)
        }
        for row, item_id in enumerate(ordered_ids)
    }


def _query_distance_matrix(
    queries_by_id: Mapping[str, str],
    task_queries: Sequence[str],
    embedding: Any,
) -> dict[str, tuple[float, ...]]:
    """Return candidate-to-task squared-L2 distances in task-query order."""
    ordered_ids = list(queries_by_id)
    queries = [query for query in task_queries if query.strip()]
    if not queries:
        raise ValueError("Oracle scheduling requires next-interval task queries")
    if not ordered_ids:
        return {}
    candidate_vectors = np.asarray(
        embedding.embed_documents(
            [queries_by_id[item_id] for item_id in ordered_ids]
        ),
        dtype=np.float32,
    )
    task_vectors = np.asarray(
        [embedding.embed_query(query) for query in queries],
        dtype=np.float32,
    )
    distances = np.sum(
        (candidate_vectors[:, np.newaxis, :] - task_vectors[np.newaxis, :, :]) ** 2,
        axis=2,
    )
    return {
        item_id: tuple(float(value) for value in distances[index])
        for index, item_id in enumerate(ordered_ids)
    }


def estimate_historical_utilities(
    pending_queries: Mapping[str, str],
    reference_queries: Mapping[str, str],
    reference_utilities: Mapping[str, float],
    embedding: Any,
    *,
    epsilon: float = HISTORICAL_UTILITY_EPSILON,
) -> dict[str, float]:
    """Transfer memory utility to pending candidates by task-query similarity."""
    if epsilon <= 0:
        raise ValueError("Historical utility epsilon must be positive")
    if set(reference_queries) != set(reference_utilities):
        raise ValueError("Historical reference queries and utilities must align")
    if not pending_queries:
        return {}
    if not reference_queries:
        return {pending_id: 0.0 for pending_id in pending_queries}
    for reference_id, utility in reference_utilities.items():
        if not 0.0 <= float(utility) <= 1.0:
            raise ValueError(
                f"Historical utility for {reference_id} must be in [0, 1]"
            )

    pending_ids = list(pending_queries)
    reference_ids = list(reference_queries)
    texts = [pending_queries[item_id].strip() for item_id in pending_ids]
    texts.extend(reference_queries[item_id].strip() for item_id in reference_ids)
    vectors = np.asarray(embedding.embed_documents(texts), dtype=np.float32)
    pending_vectors = vectors[: len(pending_ids)]
    reference_vectors = vectors[len(pending_ids) :]
    squared_distances = np.sum(
        (pending_vectors[:, np.newaxis, :] - reference_vectors[np.newaxis, :, :])
        ** 2,
        axis=2,
    )
    utilities = np.asarray(
        [float(reference_utilities[item_id]) for item_id in reference_ids],
        dtype=np.float64,
    )
    weights = 1.0 / (squared_distances.astype(np.float64) + epsilon)
    estimates = (weights @ utilities) / np.sum(weights, axis=1)
    return {
        pending_id: float(estimates[index])
        for index, pending_id in enumerate(pending_ids)
    }


def load_warm_start_documents(
    path: str | Path, *, count: int, seed: int
) -> tuple[list[Any], tuple[str, ...]]:
    """Load a deterministic shared initial pool from workflow documents."""
    candidates = load_candidate_memories(path, limit=None)
    candidate_ids = tuple(candidate.memory_id for candidate in candidates)
    selected_ids = select_warm_start_ids(candidate_ids, count=count, seed=seed)
    selected = set(selected_ids)
    try:
        from langchain_core.documents import Document
    except ImportError:  # Lightweight unit-test runtime.
        Document = SimpleNamespace

    documents = [
        Document(
            page_content=candidate.query,
            metadata={
                "memory_id": candidate.memory_id,
                "query": candidate.query,
                "workflow": candidate.workflow,
                "memory_type": "workflow",
                "source": "warm_start",
                "memory_origin": "warm_start",
                "activated_interval": 0,
                "available_from_interval": 0,
            },
        )
        for candidate in candidates
        if candidate.memory_id in selected
    ]
    return documents, selected_ids


class OnlineConstructionController:
    """Own queue state and causal construction/activation timing."""

    def __init__(
        self,
        *,
        memory: Any,
        policy: str,
        capacity: int,
        scheduler_seed: int = 42,
        retrieval_top_k: int | None = None,
        retrieval_score_threshold: float = 0.5,
        historical_utility_min_count: int = HISTORICAL_UTILITY_MIN_COUNT,
        historical_utility_lambda: float = HISTORICAL_UTILITY_LAMBDA,
        historical_utility_epsilon: float = HISTORICAL_UTILITY_EPSILON,
    ) -> None:
        if policy not in ONLINE_POLICIES:
            raise ValueError(f"Unsupported online scheduling policy: {policy}")
        if capacity < 0:
            raise ValueError("Construction capacity cannot be negative")
        if historical_utility_min_count < 1:
            raise ValueError("Historical utility min count must be at least 1")
        if historical_utility_lambda < 0:
            raise ValueError("Historical utility lambda must be non-negative")
        if historical_utility_epsilon <= 0:
            raise ValueError("Historical utility epsilon must be positive")
        self.memory = memory
        self.policy = policy
        self.capacity = capacity
        self.retrieval_top_k = (
            int(retrieval_top_k)
            if retrieval_top_k is not None
            else int(getattr(memory, "retrieve_num", 0))
        )
        self.retrieval_score_threshold = float(retrieval_score_threshold)
        self.historical_utility_min_count = int(historical_utility_min_count)
        self.historical_utility_lambda = float(historical_utility_lambda)
        self.historical_utility_epsilon = float(historical_utility_epsilon)
        self.queue = OnlineConstructionQueue()
        self.staged_documents: list[Any] = []
        self.queue_events: list[dict[str, Any]] = []
        self.construction_events: list[dict[str, Any]] = []
        self.trajectory_events: list[dict[str, Any]] = []
        self.historical_memory_stats: dict[str, dict[str, int]] = {}
        self._arrival_order = 0
        self._register_available_memories()
        if policy == "fifo":
            self.scheduler = FIFOScheduler()
        elif policy == "fifo_shortest_first":
            self.scheduler = FIFOShortestFirstScheduler()
        elif policy == "random":
            self.scheduler = OnlineRandomScheduler(seed=scheduler_seed)
        elif policy == "greedy_novelty":
            self.scheduler = GreedyNoveltyScheduler()
        elif policy == "oracle_coverage":
            self.scheduler = OracleCoverageScheduler()
        else:
            self.scheduler = OracleExactRetrievalScheduler()

    @property
    def available_memory_count(self) -> int:
        return len(self.memory.documents)

    def _register_available_memories(self) -> None:
        for memory_id in self._available_queries():
            self.historical_memory_stats.setdefault(
                memory_id,
                {"retrieval_count": 0, "success_count": 0},
            )

    def record_retrieval_outcomes(
        self, results: Sequence[Mapping[str, Any]]
    ) -> None:
        """Update run-local memory outcomes from completed tasks exactly once."""
        self._register_available_memories()
        for result in results:
            retrieved_ids = {
                str(memory_id)
                for memory_id in result.get("retrieved_memory_ids", [])
                if memory_id is not None
            }
            for memory_id in retrieved_ids:
                counters = self.historical_memory_stats.setdefault(
                    memory_id,
                    {"retrieval_count": 0, "success_count": 0},
                )
                counters["retrieval_count"] += 1
                if bool(result.get("reward")):
                    counters["success_count"] += 1

    def activate_staged(self, *, interval_id: int) -> list[str]:
        if not self.staged_documents:
            return []
        documents = list(self.staged_documents)
        for document in documents:
            if document.metadata.get("available_from_interval") != interval_id:
                raise ValueError("Staged workflow activated in the wrong interval")
            document.metadata["activated_interval"] = interval_id
        self.memory.append_documents(documents)
        self.memory.save_documents()
        self.memory.rebuild_index()
        self.staged_documents.clear()
        self._register_available_memories()
        return [document.metadata["memory_id"] for document in documents]

    def admit_results(
        self, results: Sequence[Mapping[str, Any]], *, interval_id: int
    ) -> list[str]:
        arrived: list[str] = []
        for result in results:
            if not bool(result.get("reward")):
                continue
            trajectory = result.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                raise ValueError("Successful online result has no clean trajectory")
            queue_id = trajectory_queue_id(
                task_id=str(result["task_id"]),
                task_index=int(result["task_index"]),
                trajectory=trajectory,
            )
            candidate = OnlineTrajectoryCandidate(
                queue_id=queue_id,
                task_id=str(result["task_id"]),
                task_index=int(result["task_index"]),
                task_type=str(result["task_type"]),
                query=str(result["query"]),
                trajectory=list(trajectory),
                steps=int(result["steps"]),
                arrival_interval=interval_id,
                arrival_order=self._arrival_order,
            )
            self._arrival_order += 1
            self.queue.add(candidate)
            self.trajectory_events.append(candidate.as_dict())
            arrived.append(queue_id)
        return arrived

    def _available_queries(self) -> dict[str, str]:
        queries: dict[str, str] = {}
        for index, document in enumerate(self.memory.documents):
            memory_id = document.metadata.get("memory_id") or f"available_{index:04d}"
            queries[str(memory_id)] = str(
                document.metadata.get("query") or document.page_content
            )
        return queries

    def _exact_retrieval_config(self) -> tuple[int, float]:
        top_k = self.retrieval_top_k
        threshold = self.retrieval_score_threshold
        if top_k < 1:
            raise ValueError("Exact-retrieval Oracle requires memory retrieve_num >= 1")
        if threshold < 0:
            raise ValueError(
                "Exact-retrieval Oracle requires a non-negative memory score_threshold"
            )
        return top_k, threshold

    def _historical_references(
        self,
    ) -> tuple[dict[str, str], dict[str, float]]:
        reference_queries = {
            memory_id: query
            for memory_id, query in self._available_queries().items()
            if self.historical_memory_stats.get(memory_id, {}).get(
                "retrieval_count", 0
            )
            >= self.historical_utility_min_count
        }
        reference_utilities = {
            memory_id: (
                self.historical_memory_stats[memory_id]["success_count"]
                / self.historical_memory_stats[memory_id]["retrieval_count"]
            )
            for memory_id in reference_queries
        }
        return reference_queries, reference_utilities

    def _selection(
        self,
        *,
        next_interval_queries: Sequence[str] | None = None,
        future_queries: Sequence[str] | None = None,
    ) -> ScheduleSelection:
        pending_ids = self.queue.pending_ids
        # Capacity zero is the warm-start-only control: successful trajectories
        # are still admitted and logged, but no online memory is constructed.
        if self.capacity == 0:
            return ScheduleSelection(memory_ids=())
        if self.policy == "fifo_shortest_first":
            return self.scheduler.select(
                pending_ids,
                self.capacity,
                candidates={
                    queue_id: self.queue.get(queue_id)
                    for queue_id in pending_ids
                },
            )
        if self.policy not in {
            "greedy_novelty",
            "oracle_coverage",
            *EXACT_RETRIEVAL_POLICIES,
        }:
            return self.scheduler.select(pending_ids, self.capacity)
        available_queries = self._available_queries()
        pending_queries = {
            queue_id: self.queue.get(queue_id).query for queue_id in pending_ids
        }
        overlap = set(available_queries) & set(pending_queries)
        if overlap:
            raise ValueError(
                "Available and pending IDs overlap: " + ", ".join(sorted(overlap)[:5])
            )
        queries = {**available_queries, **pending_queries}
        embedder = getattr(self.memory, "cached_embedder", self.memory.embedding)
        if self.policy == "oracle_coverage":
            if next_interval_queries is None:
                raise ValueError(
                    "Oracle coverage requires next-interval task queries"
                )
            distances = _query_distance_matrix(
                queries, next_interval_queries, embedder
            )

            def distance_scorer(
                _: Sequence[str], requested_ids: Iterable[str]
            ) -> dict[str, tuple[float, ...]]:
                requested = set(requested_ids)
                unknown = requested - set(distances)
                if unknown:
                    raise ValueError(
                        "Unknown online Oracle IDs: "
                        + ", ".join(sorted(unknown)[:5])
                    )
                return {
                    item_id: distances[item_id]
                    for item_id in queries
                    if item_id in requested
                }

            return self.scheduler.select(
                pending_ids,
                self.capacity,
                available_ids=available_queries,
                next_interval_queries=next_interval_queries,
                distance_scorer=distance_scorer,
            )
        if self.policy in EXACT_RETRIEVAL_POLICIES:
            if future_queries is None:
                raise ValueError(
                    "Exact-retrieval Oracle requires future task queries"
                )
            distances = _query_distance_matrix(queries, future_queries, embedder)

            def exact_distance_scorer(
                _: Sequence[str], requested_ids: Iterable[str]
            ) -> dict[str, tuple[float, ...]]:
                requested = set(requested_ids)
                unknown = requested - set(distances)
                if unknown:
                    raise ValueError(
                        "Unknown online exact-retrieval Oracle IDs: "
                        + ", ".join(sorted(unknown)[:5])
                    )
                return {
                    item_id: distances[item_id]
                    for item_id in queries
                    if item_id in requested
                }

            top_k, threshold = self._exact_retrieval_config()
            historical_estimates = None
            historical_reference_count = 0
            historical_lambda = 0.0
            if self.policy == "oracle_exact_retrieval_historical_utility":
                reference_queries, reference_utilities = (
                    self._historical_references()
                )
                historical_estimates = estimate_historical_utilities(
                    pending_queries,
                    reference_queries,
                    reference_utilities,
                    embedder,
                    epsilon=self.historical_utility_epsilon,
                )
                historical_reference_count = len(reference_queries)
                historical_lambda = self.historical_utility_lambda
            return self.scheduler.select(
                pending_ids,
                self.capacity,
                available_ids=available_queries,
                future_queries=future_queries,
                distance_scorer=exact_distance_scorer,
                top_k=top_k,
                score_threshold=threshold,
                historical_utility_estimates=historical_estimates,
                historical_reference_count=historical_reference_count,
                historical_utility_lambda=historical_lambda,
            )
        return self.scheduler.select(
            pending_ids,
            self.capacity,
            available_ids=available_queries,
            distance_matrix=_distance_matrix(queries, embedder),
        )

    def construct(
        self,
        *,
        interval_id: int,
        next_interval_queries: Sequence[str] | None = None,
        future_queries: Sequence[str] | None = None,
        requested_lookahead_horizon: int | str | None = None,
        effective_lookahead_horizon: int | None = None,
        future_interval_count: int | None = None,
    ) -> dict[str, Any]:
        before_ids = list(self.queue.pending_ids)
        selection = self._selection(
            next_interval_queries=next_interval_queries,
            future_queries=future_queries,
        )
        selected_ids = list(selection.memory_ids)
        construction_results: list[dict[str, Any]] = []
        for rank, queue_id in enumerate(selected_ids, start=1):
            candidate = self.queue.get(queue_id)
            candidate.selected_count += 1
            candidate.last_selected_interval = interval_id
            waiting_intervals = interval_id - candidate.arrival_interval
            score = None
            if selection.scheduler_scores:
                score = selection.scheduler_scores.get(queue_id)
            oracle_score = None
            if selection.oracle_scores:
                oracle_score = selection.oracle_scores.get(queue_id)
            memory_id = "online_" + queue_id.removeprefix("traj_")
            try:
                document = self.memory.build_document(
                    {
                        "source": "online_success",
                        "query": candidate.query,
                        "trajectory": candidate.trajectory,
                        "memory_id": memory_id,
                        "metadata": {
                            "memory_type": "workflow",
                            "memory_origin": "online",
                            "source_queue_id": queue_id,
                            "source_task_id": candidate.task_id,
                            "source_task_index": candidate.task_index,
                            "source_task_type": candidate.task_type,
                            "arrival_interval": candidate.arrival_interval,
                            "constructed_after_interval": interval_id,
                            "available_from_interval": interval_id + 1,
                        },
                    }
                )
                if document is None:
                    raise RuntimeError("Builder returned no workflow document")
                self.staged_documents.append(document)
                self.queue.remove_constructed(queue_id)
                candidate.last_construction_result = "success"
                event = {
                    "interval_id": interval_id,
                    "queue_id": queue_id,
                    "source_task_id": candidate.task_id,
                    "source_steps": candidate.steps,
                    "selection_rank": rank,
                    "scheduler_score": score,
                    "oracle_score": oracle_score,
                    "waiting_intervals": waiting_intervals,
                    "construction_result": "success",
                    "workflow": document.metadata.get("workflow"),
                    "error": None,
                    "constructed_memory_id": memory_id,
                    "available_from_interval": interval_id + 1,
                }
            except Exception as exc:
                candidate.last_construction_result = "failure"
                event = {
                    "interval_id": interval_id,
                    "queue_id": queue_id,
                    "source_task_id": candidate.task_id,
                    "source_steps": candidate.steps,
                    "selection_rank": rank,
                    "scheduler_score": score,
                    "oracle_score": oracle_score,
                    "waiting_intervals": waiting_intervals,
                    "construction_result": "failure",
                    "workflow": None,
                    "error": str(exc),
                    "constructed_memory_id": None,
                    "available_from_interval": None,
                }
            if self.policy == "oracle_exact_retrieval_historical_utility":
                event.update(
                    {
                        "adjusted_score": oracle_score.get("adjusted_score")
                        if oracle_score
                        else None,
                        "base_retrieval_value": oracle_score.get(
                            "base_retrieval_value"
                        )
                        if oracle_score
                        else None,
                        "historical_utility_estimate": oracle_score.get(
                            "historical_utility_estimate"
                        )
                        if oracle_score
                        else None,
                        "historical_reference_count": oracle_score.get(
                            "historical_reference_count"
                        )
                        if oracle_score
                        else 0,
                    }
                )
            self.construction_events.append(event)
            construction_results.append(event)

        queue_event = {
            "interval_id": interval_id,
            "queue_length_before_selection": len(before_ids),
            "pending_queue_ids_before_selection": before_ids,
            "selected_queue_ids": selected_ids,
            "scheduler_scores": selection.scheduler_scores,
            "oracle_scores": selection.oracle_scores,
            "oracle_next_interval_query_count": (
                len([query for query in next_interval_queries if query.strip()])
                if self.policy == "oracle_coverage" and next_interval_queries
                else None
            ),
            "oracle_requested_lookahead_horizon": (
                requested_lookahead_horizon
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_effective_lookahead_horizon": (
                effective_lookahead_horizon
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_future_interval_count": (
                future_interval_count
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_future_query_count": (
                len([query for query in future_queries if query.strip()])
                if self.policy in EXACT_RETRIEVAL_POLICIES and future_queries
                else None
            ),
            "oracle_retrieval_top_k": (
                self._exact_retrieval_config()[0]
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_retrieval_threshold": (
                self._exact_retrieval_config()[1]
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "queue_length_after_construction": len(self.queue),
            "pending_queue_ids_after_construction": list(self.queue.pending_ids),
            "construction_results": construction_results,
        }
        self.queue_events.append(queue_event)
        return queue_event

    def record_final_queue(
        self,
        *,
        interval_id: int,
        requested_lookahead_horizon: int | str | None = None,
    ) -> dict[str, Any]:
        ids = list(self.queue.pending_ids)
        event = {
            "interval_id": interval_id,
            "queue_length_before_selection": len(ids),
            "pending_queue_ids_before_selection": ids,
            "selected_queue_ids": [],
            "queue_length_after_construction": len(ids),
            "pending_queue_ids_after_construction": ids,
            "construction_results": [],
            "final_interval_no_construction": True,
            "oracle_requested_lookahead_horizon": (
                requested_lookahead_horizon
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_effective_lookahead_horizon": (
                0 if self.policy in EXACT_RETRIEVAL_POLICIES else None
            ),
            "oracle_future_interval_count": (
                0 if self.policy in EXACT_RETRIEVAL_POLICIES else None
            ),
            "oracle_future_query_count": (
                0 if self.policy in EXACT_RETRIEVAL_POLICIES else None
            ),
            "oracle_retrieval_top_k": (
                self._exact_retrieval_config()[0]
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
            "oracle_retrieval_threshold": (
                self._exact_retrieval_config()[1]
                if self.policy in EXACT_RETRIEVAL_POLICIES
                else None
            ),
        }
        self.queue_events.append(event)
        return event
