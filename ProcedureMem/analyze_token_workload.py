"""Compute task-to-construction workload ratios from canonical usage JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as reader:
        for line_number, line in enumerate(reader, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc


def complete_workloads(
    path: str | Path,
    *,
    record_type: str,
    gamma: float = 1.0,
) -> list[float]:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    workloads = []
    for record in _records(Path(path)):
        if record.get("record_type") != record_type:
            continue
        if record.get("usage_complete") is not True:
            continue
        prompt_tokens = record.get("prompt_tokens")
        completion_tokens = record.get("completion_tokens")
        if prompt_tokens is None or completion_tokens is None:
            continue
        workloads.append(float(prompt_tokens) + gamma * float(completion_tokens))
    return workloads


def workload_summary(
    task_usage_path: str | Path,
    memory_usage_path: str | Path,
    *,
    gamma: float = 1.0,
) -> dict[str, Any]:
    task_workloads = complete_workloads(
        task_usage_path, record_type="task_aggregate", gamma=gamma
    )
    memory_workloads = complete_workloads(
        memory_usage_path, record_type="memory_aggregate", gamma=gamma
    )
    task_median = statistics.median(task_workloads) if task_workloads else None
    memory_median = statistics.median(memory_workloads) if memory_workloads else None
    task_mean = statistics.fmean(task_workloads) if task_workloads else None
    memory_mean = statistics.fmean(memory_workloads) if memory_workloads else None
    return {
        "gamma": gamma,
        "task_sample_count": len(task_workloads),
        "memory_sample_count": len(memory_workloads),
        "task_workload_median": task_median,
        "memory_workload_median": memory_median,
        "median_ratio": (
            task_median / memory_median
            if task_median is not None and memory_median not in {None, 0}
            else None
        ),
        "task_workload_mean": task_mean,
        "memory_workload_mean": memory_mean,
        "mean_ratio": (
            task_mean / memory_mean
            if task_mean is not None and memory_mean not in {None, 0}
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-usage", type=Path, required=True)
    parser.add_argument("--memory-usage", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=1.0)
    args = parser.parse_args()
    print(
        json.dumps(
            workload_summary(
                args.task_usage,
                args.memory_usage,
                gamma=args.gamma,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
