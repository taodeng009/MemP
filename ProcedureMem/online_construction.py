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
    ScheduleSelection,
    load_candidate_memories,
    select_warm_start_ids,
)


ONLINE_POLICIES = ("fifo", "random", "greedy_novelty", "oracle_coverage")


@dataclass
class OnlineTrajectoryCandidate:
    queue_id: str
    task_id: str
    task_index: int
    task_type: str
    query: str
    trajectory: list[dict[str, str]]
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
    ) -> None:
        if policy not in ONLINE_POLICIES:
            raise ValueError(f"Unsupported online scheduling policy: {policy}")
        if capacity < 0:
            raise ValueError("Construction capacity cannot be negative")
        self.memory = memory
        self.policy = policy
        self.capacity = capacity
        self.queue = OnlineConstructionQueue()
        self.staged_documents: list[Any] = []
        self.queue_events: list[dict[str, Any]] = []
        self.construction_events: list[dict[str, Any]] = []
        self.trajectory_events: list[dict[str, Any]] = []
        self._arrival_order = 0
        if policy == "fifo":
            self.scheduler = FIFOScheduler()
        elif policy == "random":
            self.scheduler = OnlineRandomScheduler(seed=scheduler_seed)
        elif policy == "greedy_novelty":
            self.scheduler = GreedyNoveltyScheduler()
        else:
            self.scheduler = OracleCoverageScheduler()

    @property
    def available_memory_count(self) -> int:
        return len(self.memory.documents)

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

    def _selection(
        self, *, next_interval_queries: Sequence[str] | None = None
    ) -> ScheduleSelection:
        pending_ids = self.queue.pending_ids
        # Capacity zero is the warm-start-only control: successful trajectories
        # are still admitted and logged, but no online memory is constructed.
        if self.capacity == 0:
            return ScheduleSelection(memory_ids=())
        if self.policy not in {"greedy_novelty", "oracle_coverage"}:
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
    ) -> dict[str, Any]:
        before_ids = list(self.queue.pending_ids)
        selection = self._selection(next_interval_queries=next_interval_queries)
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
            "queue_length_after_construction": len(self.queue),
            "pending_queue_ids_after_construction": list(self.queue.pending_ids),
            "construction_results": construction_results,
        }
        self.queue_events.append(queue_event)
        return queue_event

    def record_final_queue(self, *, interval_id: int) -> dict[str, Any]:
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
        }
        self.queue_events.append(event)
        return event
