"""Run the 10-task Cloud MemP reranker feasibility and latency benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

from ProcedureMem.benchmark_stats import latency_summary
from ProcedureMem.reranker import DEFAULT_MODEL, OpenMemReranker
from ProcedureMem.runtime_config import (
    DEFAULT_MEMORY_DIR,
    DEFAULT_RESULTS_DIR,
    configure_runtime,
    load_environment,
)


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "Alfworld"
    / "manifests"
    / "valid_unseen_seed42_n10.json"
)
DEFAULT_OUTPUT = DEFAULT_RESULTS_DIR / "reranker_smoke" / "valid_unseen_seed42_n10"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--memory-dir", type=Path, default=DEFAULT_MEMORY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alfworld-data")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rerank-model", default=DEFAULT_MODEL)
    parser.add_argument("--rerank-timeout", type=float)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    for name in ("candidate_k", "top_n", "repeats"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be non-negative")
    if args.top_n > args.candidate_k:
        parser.error("--top-n cannot exceed --candidate-k")
    if args.rerank_model != DEFAULT_MODEL:
        parser.error(f"--rerank-model must be {DEFAULT_MODEL}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_goal(data_root: Path, task_id: str) -> str:
    gamefile = data_root / Path(task_id)
    trajectory_path = gamefile.parent / "traj_data.json"
    if not trajectory_path.is_file():
        raise FileNotFoundError(
            f"Cannot find task metadata for {task_id}: {trajectory_path}"
        )
    trajectory = _load_json(trajectory_path)
    annotations = trajectory.get("turk_annotations", {}).get("anns", [])
    for annotation in annotations:
        goal = annotation.get("task_desc")
        if isinstance(goal, str) and goal.strip():
            return goal.strip()
    raise ValueError(f"No task_desc found in {trajectory_path}")


def _load_workflow_store(memory_dir: Path) -> tuple[Any, list[Any]]:
    from langchain.storage import LocalFileStore
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document

    from ProcedureMem.llm_api import get_embedding_model

    documents_path = memory_dir / "direct" / "documents.json"
    if not documents_path.is_file():
        raise FileNotFoundError(f"MemP documents not found: {documents_path}")
    documents = [Document(**item) for item in _load_json(documents_path)]
    if not documents:
        raise ValueError(f"MemP documents are empty: {documents_path}")

    embedding = get_embedding_model()
    model_name = getattr(embedding, "model", None) or embedding.__class__.__name__
    cached = CacheBackedEmbeddings.from_bytes_store(
        embedding,
        LocalFileStore(str(memory_dir / "vector_cache")),
        namespace=str(model_name),
    )
    return FAISS.from_documents(documents, cached), documents


def _rerank_text(document: Any) -> str:
    return (
        f"Task goal: {document.metadata.get('query', document.page_content)}\n"
        f"Reusable workflow: {document.metadata.get('workflow', '')}"
    )


def _candidate_record(
    document: Any,
    *,
    vector_rank: int,
    vector_score: float,
    rerank_rank: int | None = None,
    rerank_score: float | None = None,
) -> dict[str, Any]:
    task_name = document.metadata.get("query", document.page_content)
    workflow = document.metadata.get("workflow")
    memory_id = hashlib.sha256(
        f"{task_name}\0{workflow or ''}".encode("utf-8")
    ).hexdigest()
    return {
        "memory_id": memory_id,
        "source": document.metadata.get("source"),
        "task_name": task_name,
        "workflow": workflow,
        "vector_rank": vector_rank,
        "vector_score": float(vector_score),
        "rerank_rank": rerank_rank,
        "rerank_score": rerank_score,
    }


def _search(store: Any, query: str, k: int):
    started = time.perf_counter()
    items = store.similarity_search_with_score(query, k=k, score_threshold=0.5)
    return items, (time.perf_counter() - started) * 1000.0


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    settings = configure_runtime(
        alfworld_data=args.alfworld_data,
        require_embedding=True,
    )
    manifest = _load_json(args.task_manifest.resolve())
    if manifest.get("task_count") != 10 or len(manifest.get("tasks", [])) != 10:
        raise ValueError("The feasibility benchmark requires a fixed 10-task manifest")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    store, memory_documents = _load_workflow_store(args.memory_dir.resolve())
    reranker = OpenMemReranker(
        model=args.rerank_model,
        timeout=args.rerank_timeout,
    )
    tasks = [
        {
            "task_id": item["task_id"],
            "query": _task_goal(settings.alfworld_data, item["task_id"]),
        }
        for item in manifest["tasks"]
    ]

    # Warm both embedding/FAISS and the persistent HTTP connection. Warmup data is discarded.
    for _ in range(args.warmup_runs):
        candidates, _ = _search(store, tasks[0]["query"], args.candidate_k)
        if not candidates:
            raise RuntimeError("FAISS returned no candidates during warmup")
        reranker.rerank(
            query=tasks[0]["query"],
            documents=[_rerank_text(document) for document, _ in candidates],
            top_n=args.top_n,
        )

    task_results: list[dict[str, Any]] = []
    all_similarity: list[float] = []
    all_api: list[float] = []
    all_pipeline: list[float] = []
    all_added: list[float] = []

    for task in tasks:
        repeats: list[dict[str, float]] = []
        representative: dict[str, Any] | None = None
        for _ in range(args.repeats):
            baseline, similarity_latency = _search(store, task["query"], args.top_n)
            pipeline_started = time.perf_counter()
            candidates, candidate_search_latency = _search(
                store, task["query"], args.candidate_k
            )
            if not candidates:
                raise RuntimeError(f"FAISS returned no candidates for {task['task_id']}")
            response = reranker.rerank(
                query=task["query"],
                documents=[_rerank_text(document) for document, _ in candidates],
                top_n=args.top_n,
            )
            pipeline_latency = (time.perf_counter() - pipeline_started) * 1000.0
            added_latency = pipeline_latency - similarity_latency

            repeats.append(
                {
                    "similarity_latency_ms": similarity_latency,
                    "candidate_search_latency_ms": candidate_search_latency,
                    "rerank_api_latency_ms": response.latency_ms,
                    "rerank_pipeline_latency_ms": pipeline_latency,
                    "rerank_added_latency_ms": added_latency,
                }
            )
            all_similarity.append(similarity_latency)
            all_api.append(response.latency_ms)
            all_pipeline.append(pipeline_latency)
            all_added.append(added_latency)

            if representative is None:
                baseline_records = [
                    _candidate_record(doc, vector_rank=rank, vector_score=score)
                    for rank, (doc, score) in enumerate(baseline, start=1)
                ]
                reranked_records = []
                for rerank_rank, result in enumerate(response.results, start=1):
                    doc, score = candidates[result.index]
                    reranked_records.append(
                        _candidate_record(
                            doc,
                            vector_rank=result.index + 1,
                            vector_score=score,
                            rerank_rank=rerank_rank,
                            rerank_score=result.relevance_score,
                        )
                    )
                baseline_ids = [item["memory_id"] for item in baseline_records]
                reranked_ids = [item["memory_id"] for item in reranked_records]
                overlap = len(set(baseline_ids) & set(reranked_ids))
                representative = {
                    "top1_changed": bool(
                        baseline_ids
                        and reranked_ids
                        and baseline_ids[0] != reranked_ids[0]
                    ),
                    "top_n_overlap_count": overlap,
                    "top_n_overlap_rate": overlap / max(1, len(baseline_ids)),
                    "similarity_top_n": baseline_records,
                    "reranked_top_n": reranked_records,
                    # Kept only as raw API metadata, not a primary metric.
                    "rerank_request_id": response.request_id,
                    "rerank_prompt_tokens": response.prompt_tokens,
                    "rerank_total_tokens": response.total_tokens,
                }

        assert representative is not None
        task_results.append(
            {
                **task,
                **representative,
                "latency": {
                    key: latency_summary([item[key] for item in repeats])
                    for key in repeats[0]
                },
                "repeats": repeats,
            }
        )

    summary = {
        "schema_version": 1,
        "task_manifest": str(args.task_manifest.resolve()),
        "task_count": len(tasks),
        "memory_dir": str(args.memory_dir.resolve()),
        "memory_document_count": len(memory_documents),
        "embedding_model": settings.embedding_model,
        "rerank_model": args.rerank_model,
        "candidate_k": args.candidate_k,
        "top_n": args.top_n,
        "warmup_runs": args.warmup_runs,
        "repeats": args.repeats,
        "response_cache_enabled": False,
        "top1_changed_count": sum(item["top1_changed"] for item in task_results),
        "mean_top_n_overlap_rate": statistics.fmean(
            item["top_n_overlap_rate"] for item in task_results
        ),
        "latency": {
            "similarity": latency_summary(all_similarity),
            "rerank_api": latency_summary(all_api),
            "rerank_pipeline": latency_summary(all_pipeline),
            "rerank_added": latency_summary(all_added),
        },
    }
    (output_dir / "tasks.json").write_text(
        json.dumps(task_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    load_environment()
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    summary = run_benchmark(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
