"""Raw-trajectory Edge memory for ALFWorld."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ProcedureMem.edge_subsets import (
    EDGE_SUBSET_SCHEMA_VERSION,
    classify_task_family,
    load_trajectories,
    validate_edge_subset_manifest,
)


EDGE_MEMORY_SCHEMA_VERSION = 1


def extract_task_episode(turns: Sequence[dict[str, Any]]) -> str:
    """Render the task episode without the repeated instruction and `OK` prefix."""
    start = None
    for index, turn in enumerate(turns):
        if turn.get("from") == "human" and "Your task is to:" in str(turn.get("value", "")):
            start = index
            break
    if start is None:
        raise ValueError("Trajectory has no Human turn containing 'Your task is to:'")

    rendered: list[str] = []
    for turn in turns[start:]:
        sender = turn.get("from")
        value = turn.get("value")
        if sender not in {"human", "gpt"} or not isinstance(value, str):
            raise ValueError("Trajectory contains an invalid turn")
        role = "Human" if sender == "human" else "Assistant"
        rendered.append(f"{role}:\n{value.strip()}")
    return "\n\n".join(rendered)


class RawTrajectoryMemory:
    """Index trajectory queries and return unabstracted task episodes."""

    def __init__(
        self,
        *,
        trajectory_file: str | Path,
        subset_manifest: str | Path,
        capacity: int,
        memory_dir: str | Path,
        top_k: int = 1,
        score_threshold: float | None = None,
        embedding: Any | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if score_threshold is not None and score_threshold < 0:
            raise ValueError("score_threshold must be non-negative")

        self.trajectory_file = Path(trajectory_file).expanduser().resolve()
        self.subset_manifest_path = Path(subset_manifest).expanduser().resolve()
        self.capacity = capacity
        self.memory_dir = Path(memory_dir).expanduser().resolve()
        self.top_k = top_k
        self.score_threshold = score_threshold
        if embedding is None:
            from ProcedureMem.llm_api import get_embedding_model

            embedding = get_embedding_model()
        self.embedding = embedding

        trajectories = load_trajectories(self.trajectory_file)
        manifest = json.loads(self.subset_manifest_path.read_text(encoding="utf-8"))
        validate_edge_subset_manifest(manifest, trajectory_count=len(trajectories))
        subset = manifest["subsets"].get(str(capacity))
        if not subset:
            raise ValueError(f"Edge subset manifest has no capacity {capacity}")
        self.trajectory_indices = list(subset["trajectory_indices"])

        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir = self.memory_dir / "faiss"
        self.library_path = self.memory_dir / "library.json"
        self.documents_path = self.memory_dir / "documents.json"
        self._descriptor = {
            "schema_version": EDGE_MEMORY_SCHEMA_VERSION,
            "memory_type": "raw_trajectory",
            "builder_llm_used": False,
            "trajectory_file": str(self.trajectory_file),
            "subset_manifest": str(self.subset_manifest_path),
            "subset_schema_version": EDGE_SUBSET_SCHEMA_VERSION,
            "capacity": self.capacity,
            "trajectory_indices": self.trajectory_indices,
        }

        existing = (self.library_path.exists(), self.documents_path.exists(), self.index_dir.exists())
        if any(existing) and not all(existing):
            raise RuntimeError(
                f"Incomplete Edge memory artifact in {self.memory_dir}; use a clean directory"
            )
        if all(existing):
            self._load()
        else:
            self._build(trajectories)

    @property
    def document_count(self) -> int:
        return len(self.trajectory_indices)

    def _build(self, trajectories: Sequence[dict[str, Any]]) -> None:
        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        documents: list[Any] = []
        serialized: list[dict[str, Any]] = []
        for trajectory_index in self.trajectory_indices:
            item = trajectories[trajectory_index]
            query = item["query"].strip()
            metadata = {
                "memory_type": "raw_trajectory",
                "trajectory_index": trajectory_index,
                "query": query,
                "trajectory": extract_task_episode(item["trajectory"]),
                "task_type": classify_task_family(query),
                "source": item.get("source", "alfworld"),
            }
            documents.append(Document(page_content=query, metadata=metadata))
            serialized.append({"page_content": query, "metadata": metadata})

        self.vector_store = FAISS.from_documents(documents, self.embedding)
        self.vector_store.save_local(str(self.index_dir))
        self.documents_path.write_text(
            json.dumps(serialized, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.library_path.write_text(
            json.dumps(self._descriptor, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] Built Edge-{self.capacity} raw trajectory memory in {self.memory_dir}")

    def _load(self) -> None:
        from langchain_community.vectorstores import FAISS

        descriptor = json.loads(self.library_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key, expected in self._descriptor.items()
            if descriptor.get(key) != expected
        ]
        if mismatches:
            raise RuntimeError(
                f"Edge memory artifact does not match requested configuration: "
                + ", ".join(mismatches)
            )
        documents = json.loads(self.documents_path.read_text(encoding="utf-8"))
        if not isinstance(documents, list) or len(documents) != self.capacity:
            raise RuntimeError(
                f"Edge memory documents count does not match capacity {self.capacity}"
            )
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embedding,
            allow_dangerous_deserialization=True,
        )
        print(f"[INFO] Loaded Edge-{self.capacity} raw trajectory memory from {self.memory_dir}")

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[tuple[Any, float]]:
        requested_k = self.top_k if top_k is None else top_k
        if requested_k < 1:
            raise ValueError("top_k must be at least 1")
        threshold = self.score_threshold if score_threshold is None else score_threshold
        if threshold is not None and threshold < 0:
            raise ValueError("score_threshold must be non-negative")

        results = self.vector_store.similarity_search_with_score(
            query,
            k=min(requested_k, self.document_count),
        )
        if threshold is not None:
            # FAISS returns L2 distance here: lower values are more similar.
            results = [(document, score) for document, score in results if score <= threshold]
        return [(document, float(score)) for document, score in results]
