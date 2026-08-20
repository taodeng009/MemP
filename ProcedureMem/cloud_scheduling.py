"""Minimal Cloud workflow-memory construction scheduling primitives."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CandidateMemory:
    memory_id: str
    query: str
    workflow: str
    trajectory_index: int | None = None


@dataclass(frozen=True)
class ScheduleSelection:
    memory_ids: tuple[str, ...]
    oracle_distances: dict[str, float] | None = None
    oracle_scores: dict[str, dict[str, Any]] | None = None


def load_candidate_memories(
    path: str | Path,
    *,
    limit: int | None = 300,
) -> list[CandidateMemory]:
    """Load existing workflow documents and assign stable positional IDs."""
    source = Path(path).expanduser().resolve()
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Candidate memory file is empty or invalid: {source}")
    if limit is not None:
        if limit < 1:
            raise ValueError("Candidate memory limit must be at least 1")
        if len(raw) < limit:
            raise ValueError(
                f"Candidate memory file has {len(raw)} documents, expected at least {limit}"
            )
        raw = raw[:limit]

    candidates: list[CandidateMemory] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Candidate document {index} is not an object")
        metadata = item.get("metadata") or {}
        query = metadata.get("query") or item.get("page_content")
        workflow = metadata.get("workflow")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Candidate document {index} has no query")
        if not isinstance(workflow, str) or not workflow.strip():
            raise ValueError(f"Candidate document {index} has no workflow")
        trajectory_index = metadata.get("trajectory_index")
        if trajectory_index is not None:
            trajectory_index = int(trajectory_index)
        candidates.append(
            CandidateMemory(
                memory_id=f"mem_{index:04d}",
                query=query.strip(),
                workflow=workflow.strip(),
                trajectory_index=trajectory_index,
            )
        )
    return candidates


def candidate_pool_sha256(candidates: Sequence[CandidateMemory]) -> str:
    payload = [
        {
            "memory_id": item.memory_id,
            "query": item.query,
            "workflow": item.workflow,
        }
        for item in candidates
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_warm_start_ids(
    candidate_ids: Sequence[str],
    *,
    count: int,
    seed: int,
) -> tuple[str, ...]:
    """Select a deterministic initial pool and return it in candidate order."""
    ids = list(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate memory IDs must be unique")
    if count < 0 or count > len(ids):
        raise ValueError(
            f"Warm-start count must be between 0 and {len(ids)}, got {count}"
        )
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    selected = set(shuffled[:count])
    return tuple(memory_id for memory_id in ids if memory_id in selected)


def memory_id_pool_sha256(memory_ids: Sequence[str]) -> str:
    """Hash an already-stabilized sequence of memory IDs."""
    encoded = json.dumps(
        list(memory_ids),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_interval_batches(
    task_count: int,
    *,
    batch_size: int,
    interval_size: int,
) -> list[tuple[int, int, int, bool, bool]]:
    """Return batches that never cross a logical interval boundary."""
    if task_count < 0:
        raise ValueError("task_count cannot be negative")
    if batch_size < 1 or interval_size < 1:
        raise ValueError("batch_size and interval_size must be at least 1")
    batches = []
    for interval_start in range(0, task_count, interval_size):
        logical_end = min(interval_start + interval_size, task_count)
        interval_id = interval_start // interval_size
        for offset in range(interval_start, logical_end, batch_size):
            batch_end = min(offset + batch_size, logical_end)
            batches.append(
                (
                    offset,
                    batch_end,
                    interval_id,
                    offset == interval_start,
                    batch_end == logical_end,
                )
            )
    return batches


class ScheduledWorkflowMemory:
    """Expose only activated candidate documents to Agent retrieval."""

    def __init__(
        self,
        candidates: Sequence[CandidateMemory],
        *,
        embedding: Any,
        retrieve_num: int,
        score_threshold: float | None = 0.5,
        vector_store_factory: Callable[[Sequence[Any], Any], Any] | None = None,
    ) -> None:
        if not candidates:
            raise ValueError("Scheduled workflow memory requires candidates")
        if retrieve_num < 1:
            raise ValueError("retrieve_num must be at least 1")
        ids = [item.memory_id for item in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate memory IDs must be unique")
        self.candidates = {item.memory_id: item for item in candidates}
        self.candidate_order = tuple(ids)
        self.embedding = embedding
        self.retrieve_num = retrieve_num
        self.score_threshold = score_threshold
        self.vector_store_factory = vector_store_factory
        self.available_ids: set[str] = set()
        self.activation_intervals: dict[str, int] = {}
        self.vector_store = None
        self._candidate_embeddings: dict[str, np.ndarray] = {}

    @property
    def pending_ids(self) -> set[str]:
        return set(self.candidates) - self.available_ids

    def activate(self, memory_ids: Sequence[str], *, interval_id: int) -> None:
        selected = list(memory_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("Scheduler selected duplicate memory IDs")
        unknown = [memory_id for memory_id in selected if memory_id not in self.candidates]
        if unknown:
            raise ValueError("Unknown candidate memory IDs: " + ", ".join(unknown))
        repeated = [memory_id for memory_id in selected if memory_id in self.available_ids]
        if repeated:
            raise ValueError("Memory is already available: " + ", ".join(repeated))
        self.available_ids.update(selected)
        for memory_id in selected:
            self.activation_intervals[memory_id] = interval_id

    def _documents_for_available_ids(self) -> list[Any]:
        if self.vector_store_factory is None:
            from langchain_core.documents import Document
        else:
            Document = SimpleNamespace

        documents = []
        for memory_id in self.candidate_order:
            if memory_id not in self.available_ids:
                continue
            candidate = self.candidates[memory_id]
            documents.append(
                Document(
                    page_content=candidate.query,
                    metadata={
                        "memory_id": memory_id,
                        "query": candidate.query,
                        "workflow": candidate.workflow,
                        "trajectory_index": candidate.trajectory_index,
                        "memory_type": "workflow",
                        "source": "alfworld_train",
                        "activated_interval": self.activation_intervals[memory_id],
                    },
                )
            )
        return documents

    def rebuild_available_index(self) -> None:
        documents = self._documents_for_available_ids()
        if not documents:
            self.vector_store = None
            return
        if self.vector_store_factory is not None:
            self.vector_store = self.vector_store_factory(documents, self.embedding)
            return
        from langchain_community.vectorstores import FAISS

        self.vector_store = FAISS.from_documents(documents, self.embedding)

    def retrieve(self, query: str) -> list[Any]:
        if self.vector_store is None or not self.available_ids:
            return []
        kwargs: dict[str, Any] = {
            "k": min(self.retrieve_num, len(self.available_ids))
        }
        if self.score_threshold is not None:
            kwargs["score_threshold"] = self.score_threshold
        return self.vector_store.similarity_search_with_score(query, **kwargs)

    def oracle_distance_sums(
        self,
        next_interval_queries: Sequence[str],
        pending_ids: Iterable[str],
    ) -> dict[str, float]:
        """Return summed squared-L2 distances; lower values are better."""
        distance_matrix = self.oracle_distance_matrix(
            next_interval_queries,
            pending_ids,
        )
        return {
            memory_id: float(np.sum(np.asarray(distances, dtype=np.float32)))
            for memory_id, distances in distance_matrix.items()
        }

    def oracle_distance_matrix(
        self,
        next_interval_queries: Sequence[str],
        memory_ids: Iterable[str],
    ) -> dict[str, tuple[float, ...]]:
        """Return per-query squared-L2 distances in query input order."""
        queries = [query for query in next_interval_queries if query.strip()]
        if not queries:
            raise ValueError("Oracle scheduling requires next-interval task queries")
        requested = set(memory_ids)
        ordered_ids = [
            memory_id for memory_id in self.candidate_order if memory_id in requested
        ]
        if not ordered_ids:
            return {}

        missing_ids = [
            memory_id for memory_id in ordered_ids
            if memory_id not in self._candidate_embeddings
        ]
        if missing_ids:
            vectors = self.embedding.embed_documents(
                [self.candidates[memory_id].query for memory_id in missing_ids]
            )
            self._candidate_embeddings.update(
                {
                    memory_id: np.asarray(vector, dtype=np.float32)
                    for memory_id, vector in zip(missing_ids, vectors)
                }
            )

        query_vectors = np.asarray(
            [self.embedding.embed_query(query) for query in queries],
            dtype=np.float32,
        )
        distances: dict[str, tuple[float, ...]] = {}
        for memory_id in ordered_ids:
            candidate_vector = self._candidate_embeddings[memory_id]
            delta = query_vectors - candidate_vector
            distances[memory_id] = tuple(
                float(value) for value in np.sum(delta * delta, axis=1)
            )
        return distances


class RandomScheduler:
    def __init__(self, candidate_ids: Sequence[str], *, seed: int) -> None:
        self.order = list(candidate_ids)
        random.Random(seed).shuffle(self.order)

    def select(self, pending_ids: Iterable[str], capacity: int) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        pending = set(pending_ids)
        selected = tuple(
            memory_id for memory_id in self.order if memory_id in pending
        )[:capacity]
        return ScheduleSelection(memory_ids=selected)


class OracleHighScheduler:
    """Legacy sum-distance Oracle, retained for backward compatibility."""

    def select(
        self,
        pending_ids: Iterable[str],
        capacity: int,
        *,
        next_interval_queries: Sequence[str],
        distance_scorer: Callable[[Sequence[str], Iterable[str]], Mapping[str, float]],
    ) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        pending = set(pending_ids)
        if not pending:
            return ScheduleSelection(memory_ids=(), oracle_distances={})
        distances = {
            memory_id: float(value)
            for memory_id, value in distance_scorer(
                next_interval_queries, pending
            ).items()
        }
        if set(distances) != pending:
            missing = sorted(pending - set(distances))
            unknown = sorted(set(distances) - pending)
            raise ValueError(
                "Oracle distance scorer returned the wrong candidates: "
                f"missing={missing[:5]}, unknown={unknown[:5]}"
            )
        selected = tuple(
            sorted(pending, key=lambda memory_id: (distances[memory_id], memory_id))[
                :capacity
            ]
        )
        return ScheduleSelection(
            memory_ids=selected,
            oracle_distances={memory_id: distances[memory_id] for memory_id in selected},
        )


OracleSumScheduler = OracleHighScheduler


class OracleCoverageScheduler:
    """Greedily maximize marginal distance improvement over available memory."""

    def select(
        self,
        pending_ids: Iterable[str],
        capacity: int,
        *,
        available_ids: Iterable[str],
        next_interval_queries: Sequence[str],
        distance_scorer: Callable[
            [Sequence[str], Iterable[str]],
            Mapping[str, Sequence[float]],
        ],
    ) -> ScheduleSelection:
        if capacity < 1:
            raise ValueError("Construction capacity must be at least 1")
        pending = set(pending_ids)
        if not pending:
            return ScheduleSelection(memory_ids=(), oracle_scores={})
        available = set(available_ids)
        overlap = pending & available
        if overlap:
            raise ValueError(
                "Available and pending memory IDs overlap: "
                + ", ".join(sorted(overlap)[:5])
            )

        scored_ids = pending | available
        distance_matrix = {
            memory_id: tuple(float(value) for value in values)
            for memory_id, values in distance_scorer(
                next_interval_queries,
                scored_ids,
            ).items()
        }
        if set(distance_matrix) != scored_ids:
            missing = sorted(scored_ids - set(distance_matrix))
            unknown = sorted(set(distance_matrix) - scored_ids)
            raise ValueError(
                "Oracle distance scorer returned the wrong memories: "
                f"missing={missing[:5]}, unknown={unknown[:5]}"
            )
        query_count = len([query for query in next_interval_queries if query.strip()])
        wrong_lengths = sorted(
            memory_id
            for memory_id, distances in distance_matrix.items()
            if len(distances) != query_count
        )
        if wrong_lengths:
            raise ValueError(
                "Oracle distance scorer returned the wrong query count for: "
                + ", ".join(wrong_lengths[:5])
            )

        selected: list[str] = []
        scores: dict[str, dict[str, Any]] = {}
        if available:
            best_distances = [
                min(distance_matrix[memory_id][query_index] for memory_id in available)
                for query_index in range(query_count)
            ]
        else:
            first_id = min(
                pending,
                key=lambda memory_id: (
                    sum(distance_matrix[memory_id]),
                    memory_id,
                ),
            )
            selected.append(first_id)
            first_distance_sum = float(sum(distance_matrix[first_id]))
            scores[first_id] = {
                "value": first_distance_sum,
                "score_type": "faiss_l2_distance_sum",
                "higher_is_better": False,
                "selection_rank": 1,
            }
            best_distances = list(distance_matrix[first_id])

        target_count = min(capacity, len(pending))
        while len(selected) < target_count:
            remaining = pending - set(selected)
            marginal_gains = {
                memory_id: float(
                    sum(
                        max(
                            0.0,
                            best_distances[query_index]
                            - distance_matrix[memory_id][query_index],
                        )
                        for query_index in range(query_count)
                    )
                )
                for memory_id in remaining
            }
            next_id = min(
                remaining,
                key=lambda memory_id: (-marginal_gains[memory_id], memory_id),
            )
            selected.append(next_id)
            scores[next_id] = {
                "value": marginal_gains[next_id],
                "score_type": "faiss_l2_marginal_gain",
                "higher_is_better": True,
                "selection_rank": len(selected),
            }
            best_distances = [
                min(best_distance, distance_matrix[next_id][query_index])
                for query_index, best_distance in enumerate(best_distances)
            ]

        return ScheduleSelection(
            memory_ids=tuple(selected),
            oracle_scores=scores,
        )


def summarize_scheduling_intervals(
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_interval: dict[int, list[dict[str, Any]]] = {}
    for result in results:
        by_interval.setdefault(int(result["interval_id"]), []).append(result)
    summaries: list[dict[str, Any]] = []
    cumulative_count = 0
    cumulative_success = 0
    for interval_id in sorted(by_interval):
        rows = by_interval[interval_id]
        successes = sum(bool(row["reward"]) for row in rows)
        cumulative_count += len(rows)
        cumulative_success += successes
        summaries.append(
            {
                "interval_id": interval_id,
                "task_count": len(rows),
                "success_count": successes,
                "success_rate": successes / len(rows),
                "cumulative_success_rate": cumulative_success / cumulative_count,
                "average_steps": sum(int(row["steps"]) for row in rows) / len(rows),
                "available_memory_count": int(rows[0]["available_memory_count"]),
                "selected_memory_ids": list(rows[0]["selected_memory_ids"]),
            }
        )
    return summaries
