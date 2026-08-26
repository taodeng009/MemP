"""Add mean task-query-to-nearest-memory distance to diversity coverage results."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ProcedureMem.alfworld_experiment import load_json, manifest_sha256, write_json
from ProcedureMem.cloud_scheduling import (
    ScheduledWorkflowMemory,
    load_cached_embedding,
    load_candidate_memories,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_CONFIG = PACKAGE_DIR / "config.yaml"
DEFAULT_MEMORY_DIR = PACKAGE_DIR / "memory" / "alfworld"


def compute_pool_query_metrics(
    pools: Sequence[dict[str, Any]],
    query_distances: Mapping[str, Sequence[float]],
    *,
    query_count: int,
    threshold: float,
) -> dict[str, dict[str, float]]:
    """Compute coverage and mean nearest squared-L2 distance for each pool."""
    metrics: dict[str, dict[str, float]] = {}
    for pool in pools:
        pool_id = pool["pool_id"]
        memory_ids = pool["memory_ids"]
        if not isinstance(memory_ids, list) or not memory_ids:
            raise ValueError(f"Pool {pool_id} has no memory IDs")
        missing = [memory_id for memory_id in memory_ids if memory_id not in query_distances]
        if missing:
            raise ValueError(
                f"Pool {pool_id} is missing distance rows: " + ", ".join(missing)
            )
        rows = [query_distances[memory_id] for memory_id in memory_ids]
        if any(len(row) != query_count for row in rows):
            raise ValueError(f"Pool {pool_id} has the wrong query distance count")
        nearest = [
            min(float(row[query_index]) for row in rows)
            for query_index in range(query_count)
        ]
        metrics[pool_id] = {
            "coverage": sum(distance <= threshold for distance in nearest)
            / query_count,
            "mean_nearest_query_memory_distance": statistics.fmean(nearest),
        }
    return metrics


def _resolve_candidate_path(recorded_path: str | Path) -> Path:
    path = Path(recorded_path).expanduser()
    if path.is_file():
        return path.resolve()
    fallback = DEFAULT_MEMORY_DIR / "direct" / path.name
    if fallback.is_file():
        return fallback.resolve()
    raise FileNotFoundError(f"Candidate memory file not found: {path}")


def _load_task_queries(
    results_dir: Path,
    pool_id: str,
    expected_task_ids: Sequence[str],
) -> list[str]:
    path = results_dir / f"diversity_{pool_id}" / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation results not found: {path}")
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if [task.get("task_id") for task in tasks] != list(expected_task_ids):
        raise ValueError(f"Task IDs or order differ in {path}")
    queries = [task.get("query") for task in tasks]
    if not all(isinstance(query, str) and query.strip() for query in queries):
        raise ValueError(f"Missing task query in {path}")
    return queries


def build_query_distance_results(
    *,
    diversity_pools: str | Path,
    diversity_results: str | Path,
    coverage_results: str | Path,
    results_dir: str | Path,
    task_manifest: str | Path,
    memory_config: str | Path = DEFAULT_MEMORY_CONFIG,
) -> dict[str, Any]:
    from ProcedureMem.runtime_config import configure_runtime, load_memory_config

    pool_manifest = load_json(Path(diversity_pools).expanduser().resolve())
    diversity = load_json(Path(diversity_results).expanduser().resolve())
    coverage = load_json(Path(coverage_results).expanduser().resolve())
    manifest_path = Path(task_manifest).expanduser().resolve()
    manifest = load_json(manifest_path)
    root = Path(results_dir).expanduser().resolve()

    pools = pool_manifest.get("pools")
    diversity_rows = diversity.get("pool_results")
    coverage_rows = coverage.get("pool_results")
    generation = pool_manifest.get("generation_parameters")
    evaluation = diversity.get("evaluation_parameters")
    if not isinstance(pools, list) or not pools:
        raise ValueError("Pool manifest has no pools")
    if not isinstance(diversity_rows, list) or not isinstance(coverage_rows, list):
        raise ValueError("Diversity or coverage summary has no pool_results")
    if not isinstance(generation, dict) or not isinstance(evaluation, dict):
        raise ValueError("Experiment parameters are missing")

    pool_ids = [pool["pool_id"] for pool in pools]
    diversity_by_id = {row["pool_id"]: row for row in diversity_rows}
    coverage_by_id = {row["pool_id"]: row for row in coverage_rows}
    if set(pool_ids) != set(diversity_by_id) or set(pool_ids) != set(coverage_by_id):
        raise ValueError("Pool IDs differ across pool, diversity, and coverage files")

    manifest_tasks = manifest.get("tasks")
    if not isinstance(manifest_tasks, list) or not manifest_tasks:
        raise ValueError(f"Task manifest has no tasks: {manifest_path}")
    task_ids = [task["task_id"] for task in manifest_tasks]
    if len(task_ids) != evaluation.get("task_count"):
        raise ValueError("Task manifest count differs from diversity evaluation")
    if manifest_sha256(manifest) != evaluation.get("manifest_sha256"):
        raise ValueError("Task manifest hash differs from diversity evaluation")
    queries = _load_task_queries(root, pool_ids[0], task_ids)

    settings = configure_runtime(require_embedding=True)
    expected_embedding_model = evaluation.get("embedding_model")
    if settings.embedding_model != expected_embedding_model:
        raise ValueError(
            f"Embedding model differs: expected {expected_embedding_model!r}, "
            f"got {settings.embedding_model!r}"
        )
    if generation.get("embedding_model") != expected_embedding_model:
        raise ValueError("Pool generation and evaluation embedding models differ")

    threshold = float(coverage["score_threshold"])
    if coverage.get("coverage_comparison_operator") != "<=":
        raise ValueError("Coverage summary does not use squared-L2 <= semantics")
    if threshold != float(evaluation["score_threshold"]):
        raise ValueError("Coverage and evaluation thresholds differ")

    candidate_path = _resolve_candidate_path(generation["candidate_memory_file"])
    candidates = load_candidate_memories(
        candidate_path,
        limit=int(generation["candidate_count"]),
    )
    unique_memory_ids = {
        memory_id for pool in pools for memory_id in pool["memory_ids"]
    }
    config = load_memory_config(memory_config)
    memory = ScheduledWorkflowMemory(
        candidates,
        embedding=load_cached_embedding(config["memory_dir"]),
        retrieve_num=1,
        score_threshold=threshold,
    )
    query_distances = memory.oracle_distance_matrix(queries, unique_memory_ids)
    metrics = compute_pool_query_metrics(
        pools,
        query_distances,
        query_count=len(queries),
        threshold=threshold,
    )

    output_rows = []
    for pool_id in pool_ids:
        recorded_coverage = float(coverage_by_id[pool_id]["coverage"])
        calculated_coverage = metrics[pool_id]["coverage"]
        if not math.isclose(
            recorded_coverage, calculated_coverage, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                f"Recomputed coverage differs for {pool_id}: "
                f"recorded={recorded_coverage}, calculated={calculated_coverage}"
            )
        output_rows.append(
            {
                "pool_id": pool_id,
                "diversity": float(diversity_by_id[pool_id]["diversity"]),
                "coverage": recorded_coverage,
                "mean_nearest_query_memory_distance": metrics[pool_id][
                    "mean_nearest_query_memory_distance"
                ],
                "success_rate": float(diversity_by_id[pool_id]["success_rate"]),
            }
        )

    from scipy.stats import spearmanr

    correlation = spearmanr(
        [row["mean_nearest_query_memory_distance"] for row in output_rows],
        [row["success_rate"] for row in output_rows],
    )
    rho = float(correlation.statistic)
    p_value = float(correlation.pvalue)
    updated = dict(coverage)
    updated.update(
        {
            "task_manifest": str(manifest_path),
            "candidate_memory_file": str(candidate_path),
            "query_distance_metric": "faiss_squared_l2_distance",
            "pool_results": output_rows,
            "mean_nearest_query_memory_distance_success_rate_spearman": {
                "rho": None if math.isnan(rho) else rho,
                "p_value": None if math.isnan(p_value) else p_value,
            },
        }
    )
    return updated


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diversity-pools", type=Path, required=True)
    parser.add_argument("--diversity-results", type=Path, required=True)
    parser.add_argument("--coverage-results", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task-manifest", type=Path, required=True)
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results = build_query_distance_results(
        diversity_pools=args.diversity_pools,
        diversity_results=args.diversity_results,
        coverage_results=args.coverage_results,
        results_dir=args.results_dir,
        task_manifest=args.task_manifest,
        memory_config=args.memory_config,
    )
    output = args.output or args.coverage_results
    write_json(output, results)
    correlation = results[
        "mean_nearest_query_memory_distance_success_rate_spearman"
    ]
    print(f"Updated results: {Path(output).resolve()}")
    print(f"Spearman: rho={correlation['rho']}, p={correlation['p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
