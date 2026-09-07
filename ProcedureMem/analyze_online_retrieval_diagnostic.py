"""Compare FIFO and Coverage retrieval geometry from completed online runs.

This diagnostic does not run ALFWorld.  It reconstructs each policy's actual
memory pool from the persisted workflow documents and construction events,
then reuses MemP's embedding client and squared-L2 query-distance pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence


METRIC = "faiss_squared_l2_distance"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected one JSON object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def _documents_path(run_dir: Path, experiment: Mapping[str, Any]) -> Path:
    method = str(experiment.get("construction_method") or "direct")
    expected = run_dir / "memory" / method / "documents.json"
    if expected.is_file():
        return expected
    matches = list((run_dir / "memory").glob("*/documents.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot resolve workflow documents under {run_dir / 'memory'}"
    )


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load and validate the artifacts needed to reconstruct one run."""
    root = Path(run_dir).expanduser().resolve()
    experiment = _load_json(root / "experiment.json")
    results = _load_jsonl(root / "results.jsonl")
    construction_events = _load_jsonl(root / "construction_events.jsonl")
    documents_raw = json.loads(
        _documents_path(root, experiment).read_text(encoding="utf-8")
    )
    if not isinstance(documents_raw, list):
        raise ValueError(f"Workflow documents are not a list for {root}")

    results.sort(key=lambda row: int(row["task_index"]))
    indices = [int(row["task_index"]) for row in results]
    if indices != list(range(len(results))):
        raise ValueError(f"Task indices are incomplete or out of order in {root}")
    if not all(isinstance(row.get("query"), str) and row["query"].strip() for row in results):
        raise ValueError(f"One or more task queries are missing in {root}")

    documents: dict[str, dict[str, Any]] = {}
    for item in documents_raw:
        if not isinstance(item, dict) or not isinstance(item.get("metadata"), dict):
            raise ValueError(f"Invalid workflow document in {root}")
        metadata = item["metadata"]
        memory_id = metadata.get("memory_id")
        query = item.get("page_content")
        available_from = metadata.get("available_from_interval")
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError(f"Workflow document has no memory_id in {root}")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Workflow document {memory_id} has no query")
        if not isinstance(available_from, int) or available_from < 0:
            raise ValueError(
                f"Workflow document {memory_id} has no valid available_from_interval"
            )
        if memory_id in documents:
            raise ValueError(f"Duplicate workflow memory ID {memory_id} in {root}")
        documents[memory_id] = {
            "query": query,
            "available_from_interval": available_from,
            "metadata": metadata,
        }

    successful_events: dict[str, dict[str, Any]] = {}
    for event in construction_events:
        if event.get("construction_result") != "success":
            continue
        memory_id = event.get("constructed_memory_id")
        if not isinstance(memory_id, str) or memory_id not in documents:
            raise ValueError(
                f"Successful construction event has no persisted document: {memory_id!r}"
            )
        if int(event["available_from_interval"]) != documents[memory_id][
            "available_from_interval"
        ]:
            raise ValueError(f"Activation interval differs for {memory_id}")
        successful_events[memory_id] = event

    online_document_ids = {
        memory_id
        for memory_id, document in documents.items()
        if document["metadata"].get("memory_origin") == "online"
    }
    if online_document_ids != set(successful_events):
        missing_events = sorted(online_document_ids - set(successful_events))
        missing_documents = sorted(set(successful_events) - online_document_ids)
        raise ValueError(
            "Online documents and successful construction events differ: "
            f"missing_events={missing_events[:5]}, "
            f"missing_documents={missing_documents[:5]}"
        )

    interval_size = experiment.get("interval_size")
    if not isinstance(interval_size, int) or interval_size < 1:
        raise ValueError(f"Invalid interval_size in {root / 'experiment.json'}")
    return {
        "run_dir": root,
        "experiment": experiment,
        "results": results,
        "documents": documents,
        "interval_size": interval_size,
    }


def _pool_queries(
    documents: Mapping[str, Mapping[str, Any]], *, available_by: int
) -> dict[str, str]:
    return {
        memory_id: str(document["query"])
        for memory_id, document in documents.items()
        if int(document["available_from_interval"]) <= available_by
    }


def diagnose_run(run: Mapping[str, Any], embedding: Any) -> dict[str, Any]:
    """Compute per-interval before/after nearest squared-L2 statistics."""
    # This is the exact helper used by OnlineConstructionController for the
    # Oracle Coverage distance matrix: document embeddings + query embeddings
    # followed by float32 squared-L2 distance.
    from ProcedureMem.online_construction import _query_distance_matrix

    results = list(run["results"])
    documents = run["documents"]
    interval_size = int(run["interval_size"])
    task_count = len(results)
    interval_count = math.ceil(task_count / interval_size)
    rows: list[dict[str, Any]] = []
    gain_sum = 0.0
    gain_query_count = 0
    final_sum = 0.0
    final_query_count = 0

    for construction_interval in range(max(0, interval_count - 1)):
        next_start = (construction_interval + 1) * interval_size
        next_end = min(next_start + interval_size, task_count)
        next_tasks = results[next_start:next_end]
        if not next_tasks:
            continue
        task_queries = [str(task["query"]) for task in next_tasks]
        pre_pool = _pool_queries(
            documents, available_by=construction_interval
        )
        post_pool = _pool_queries(
            documents, available_by=construction_interval + 1
        )
        if not post_pool:
            raise ValueError(
                f"Post-construction pool is empty after interval {construction_interval} "
                f"in {run['run_dir']}"
            )
        new_ids = set(post_pool) - set(pre_pool)
        if not new_ids:
            raise ValueError(
                f"No memory became available after interval {construction_interval} "
                f"in {run['run_dir']}"
            )

        # Embed each next-interval query once.  Computing a single post-pool
        # matrix and slicing it for the pre-pool avoids numerical noise from
        # issuing duplicate embed_query calls for before and after.
        distances = _query_distance_matrix(post_pool, task_queries, embedding)
        final_best = [
            min(distances[memory_id][query_index] for memory_id in post_pool)
            for query_index in range(len(task_queries))
        ]
        mean_final = statistics.fmean(final_best)
        final_sum += sum(final_best)
        final_query_count += len(final_best)

        mean_pre: float | None
        mean_gain: float | None
        if pre_pool:
            pre_best = [
                min(distances[memory_id][query_index] for memory_id in pre_pool)
                for query_index in range(len(task_queries))
            ]
            gains = [before - after for before, after in zip(pre_best, final_best)]
            mean_pre = statistics.fmean(pre_best)
            mean_gain = statistics.fmean(gains)
            gain_sum += sum(gains)
            gain_query_count += len(gains)
        else:
            # The real retrieval pipeline has no finite best score for an
            # empty memory pool.  Do not invent a threshold-based baseline.
            mean_pre = None
            mean_gain = None

        rows.append(
            {
                "construction_interval": construction_interval,
                "affected_task_interval": construction_interval + 1,
                "affected_task_start": next_start,
                "affected_task_end_exclusive": next_end,
                "task_count": len(task_queries),
                "pre_memory_count": len(pre_pool),
                "new_memory_count": len(new_ids),
                "post_memory_count": len(post_pool),
                "mean_pre_best_similarity": mean_pre,
                "mean_retrieval_gain": mean_gain,
                "mean_final_best_similarity": mean_final,
            }
        )

    return {
        "intervals": rows,
        "overall_mean_retrieval_gain": (
            gain_sum / gain_query_count if gain_query_count else None
        ),
        "overall_mean_final_best_similarity": (
            final_sum / final_query_count if final_query_count else None
        ),
        "gain_query_count": gain_query_count,
        "final_query_count": final_query_count,
    }


def _validate_pair(fifo: Mapping[str, Any], coverage: Mapping[str, Any]) -> None:
    left = fifo["experiment"]
    right = coverage["experiment"]
    for key in ("split", "manifest_sha256", "embedding_model", "interval_size"):
        if left.get(key) != right.get(key):
            raise ValueError(
                f"FIFO and Coverage differ on {key}: "
                f"{left.get(key)!r} != {right.get(key)!r}"
            )
    if len(fifo["results"]) != len(coverage["results"]):
        raise ValueError("FIFO and Coverage task counts differ")
    for index, (fifo_task, coverage_task) in enumerate(
        zip(fifo["results"], coverage["results"])
    ):
        if fifo_task.get("task_id") != coverage_task.get("task_id"):
            raise ValueError(f"FIFO and Coverage task IDs differ at index {index}")
        if fifo_task.get("query") != coverage_task.get("query"):
            raise ValueError(f"FIFO and Coverage queries differ at index {index}")


def build_diagnostic(
    *, fifo_run_dir: str | Path, coverage_run_dir: str | Path
) -> dict[str, Any]:
    """Build the paired retrieval diagnostic using the configured endpoint."""
    from ProcedureMem.cloud_scheduling import load_cached_embedding
    from ProcedureMem.runtime_config import configure_runtime

    fifo = load_run(fifo_run_dir)
    coverage = load_run(coverage_run_dir)
    _validate_pair(fifo, coverage)

    settings = configure_runtime(require_embedding=True)
    expected_model = fifo["experiment"].get("embedding_model")
    if settings.embedding_model != expected_model:
        raise ValueError(
            "Configured embedding model differs from the experiment: "
            f"{settings.embedding_model!r} != {expected_model!r}"
        )

    fifo_embedding = load_cached_embedding(fifo["run_dir"] / "memory")
    coverage_embedding = load_cached_embedding(coverage["run_dir"] / "memory")
    fifo_result = diagnose_run(fifo, fifo_embedding)
    coverage_result = diagnose_run(coverage, coverage_embedding)
    if len(fifo_result["intervals"]) != len(coverage_result["intervals"]):
        raise ValueError("FIFO and Coverage diagnostic interval counts differ")

    intervals: list[dict[str, Any]] = []
    for fifo_row, coverage_row in zip(
        fifo_result["intervals"], coverage_result["intervals"]
    ):
        shared_keys = (
            "construction_interval",
            "affected_task_interval",
            "affected_task_start",
            "affected_task_end_exclusive",
            "task_count",
        )
        if any(fifo_row[key] != coverage_row[key] for key in shared_keys):
            raise ValueError("FIFO and Coverage interval alignment differs")
        intervals.append(
            {
                **{key: fifo_row[key] for key in shared_keys},
                "fifo_pre_memory_count": fifo_row["pre_memory_count"],
                "fifo_new_memory_count": fifo_row["new_memory_count"],
                "fifo_post_memory_count": fifo_row["post_memory_count"],
                "coverage_pre_memory_count": coverage_row["pre_memory_count"],
                "coverage_new_memory_count": coverage_row["new_memory_count"],
                "coverage_post_memory_count": coverage_row["post_memory_count"],
                "fifo_mean_retrieval_gain": fifo_row["mean_retrieval_gain"],
                "coverage_mean_retrieval_gain": coverage_row["mean_retrieval_gain"],
                "fifo_mean_final_best_similarity": fifo_row[
                    "mean_final_best_similarity"
                ],
                "coverage_mean_final_best_similarity": coverage_row[
                    "mean_final_best_similarity"
                ],
            }
        )

    return {
        "schema_version": 1,
        "metric": METRIC,
        "metric_semantics": (
            "Raw FAISS squared-L2 score from MemP; smaller means more similar. "
            "The requested final-best-similarity columns contain the mean "
            "minimum raw score and are therefore lower-is-better."
        ),
        "gain_definition": (
            "mean(pre-construction best squared-L2 - post-construction best "
            "squared-L2); larger is better"
        ),
        "empty_pre_pool_policy": (
            "Interval 0 pre score and gain are null; no artificial baseline is used."
        ),
        "embedding_model": expected_model,
        "embedding_base_url": settings.embedding_base_url,
        "fifo_run_dir": str(fifo["run_dir"]),
        "coverage_run_dir": str(coverage["run_dir"]),
        "task_count": len(fifo["results"]),
        "interval_size": fifo["interval_size"],
        "intervals": intervals,
        "overall": {
            "averaging": "query_weighted",
            "fifo_mean_retrieval_gain": fifo_result[
                "overall_mean_retrieval_gain"
            ],
            "coverage_mean_retrieval_gain": coverage_result[
                "overall_mean_retrieval_gain"
            ],
            "fifo_mean_final_best_similarity": fifo_result[
                "overall_mean_final_best_similarity"
            ],
            "coverage_mean_final_best_similarity": coverage_result[
                "overall_mean_final_best_similarity"
            ],
            "retrieval_gain_query_count": fifo_result["gain_query_count"],
            "final_best_similarity_query_count": fifo_result[
                "final_query_count"
            ],
        },
    }


def _write_csv(path: Path, result: Mapping[str, Any]) -> None:
    rows = list(result["intervals"])
    if not rows:
        raise ValueError("Diagnostic produced no interval rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as writer:
        table = csv.DictWriter(writer, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def _format(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def _print_summary(result: Mapping[str, Any]) -> None:
    headers = (
        "interval",
        "FIFO gain",
        "Coverage gain",
        "FIFO final",
        "Coverage final",
    )
    print(" | ".join(headers))
    print("-" * 82)
    for row in result["intervals"]:
        values = (
            str(row["construction_interval"]),
            _format(row["fifo_mean_retrieval_gain"]),
            _format(row["coverage_mean_retrieval_gain"]),
            _format(row["fifo_mean_final_best_similarity"]),
            _format(row["coverage_mean_final_best_similarity"]),
        )
        print(" | ".join(values))
    overall = result["overall"]
    print("-" * 82)
    print(
        "overall | "
        + " | ".join(
            _format(overall[key])
            for key in (
                "fifo_mean_retrieval_gain",
                "coverage_mean_retrieval_gain",
                "fifo_mean_final_best_similarity",
                "coverage_mean_final_best_similarity",
            )
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fifo-run-dir", type=Path, required=True)
    parser.add_argument("--coverage-run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_diagnostic(
        fifo_run_dir=args.fifo_run_dir,
        coverage_run_dir=args.coverage_run_dir,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.output_csv:
        _write_csv(args.output_csv, result)
    _print_summary(result)
    print(f"JSON: {args.output_json.resolve()}")
    if args.output_csv:
        print(f"CSV: {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
