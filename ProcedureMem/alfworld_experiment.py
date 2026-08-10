"""Task manifests and auditable result summaries for ALFWorld evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


MANIFEST_SCHEMA_VERSION = 1
RESULT_SCHEMA_VERSION = 1
CONDITIONS = ("no_memory", "memory")
SPLIT_NAMES = {
    "valid_seen": "eval_in_distribution",
    "valid_unseen": "eval_out_of_distribution",
}


def task_id_from_gamefile(gamefile: str | Path, data_root: str | Path) -> str:
    path = Path(gamefile).expanduser().resolve()
    root = Path(data_root).expanduser().resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"Gamefile {path} is outside ALFWorld data root {root}") from exc


def available_task_map(
    gamefiles: Iterable[str | Path], data_root: str | Path
) -> dict[str, str]:
    gamefile_list = list(gamefiles)
    tasks = {
        task_id_from_gamefile(gamefile, data_root): str(Path(gamefile).resolve())
        for gamefile in gamefile_list
    }
    if len(tasks) != len(gamefile_list):
        raise ValueError("ALFWorld returned duplicate task IDs")
    return tasks


def build_task_manifest(
    gamefiles: Sequence[str | Path],
    *,
    data_root: str | Path,
    split: str,
    seed: int,
    limit_tasks: int | None = None,
    task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    if split not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split: {split}")
    available = available_task_map(gamefiles, data_root)
    ordered_ids = sorted(available)

    if task_ids:
        selected_ids = list(task_ids)
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("task_ids contains duplicates")
        missing = [task_id for task_id in selected_ids if task_id not in available]
        if missing:
            raise ValueError("Unknown task IDs: " + ", ".join(missing[:5]))
        if limit_tasks is not None and limit_tasks != len(selected_ids):
            raise ValueError("limit_tasks must match the number of explicit task_ids")
    else:
        rng = random.Random(seed)
        rng.shuffle(ordered_ids)
        if limit_tasks is not None:
            if limit_tasks < 1:
                raise ValueError("limit_tasks must be at least 1")
            if limit_tasks > len(ordered_ids):
                raise ValueError(
                    f"Requested {limit_tasks} tasks, but split has {len(ordered_ids)}"
                )
            ordered_ids = ordered_ids[:limit_tasks]
        selected_ids = ordered_ids

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split": split,
        "alfworld_split": SPLIT_NAMES[split],
        "seed": seed,
        "task_count": len(selected_ids),
        "tasks": [{"index": index, "task_id": task_id} for index, task_id in enumerate(selected_ids)],
    }


def validate_task_manifest(
    manifest: dict[str, Any],
    *,
    split: str,
    available_gamefiles: Sequence[str | Path],
    data_root: str | Path,
) -> list[str]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported task manifest schema_version")
    if manifest.get("split") != split:
        raise ValueError(
            f"Manifest split is {manifest.get('split')!r}, requested {split!r}"
        )
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest has no tasks")
    task_ids = [task.get("task_id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id for task_id in task_ids):
        raise ValueError("Task manifest contains an invalid task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Task manifest contains duplicate task IDs")
    if manifest.get("task_count") != len(task_ids):
        raise ValueError("Task manifest task_count does not match tasks")

    available = available_task_map(available_gamefiles, data_root)
    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        raise ValueError("Manifest tasks missing from ALFWorld data: " + ", ".join(missing[:5]))
    return [available[task_id] for task_id in task_ids]


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(destination)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_sha256(manifest: dict[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_query(observation: str) -> str:
    marker = "Your task is to: "
    return observation.split(marker, 1)[1].strip() if marker in observation else observation.strip()


def retrieval_records(items: Sequence[Any]) -> list[dict[str, Any]]:
    records = []
    for rank, item in enumerate(items, start=1):
        if isinstance(item, tuple):
            document, score = item
        else:
            document, score = item, None
        metadata = document.metadata
        records.append(
            {
                "rank": rank,
                "task_name": metadata.get("query"),
                "workflow": metadata.get("workflow"),
                "score": float(score) if score is not None else None,
                "source": metadata.get("source"),
            }
        )
    return records


def inject_memory(observation: str, records: Sequence[dict[str, Any]]) -> str:
    if not records:
        return observation
    guidelines = [
        {"task_name": record["task_name"], "guidelines": record["workflow"]}
        for record in records
    ]
    return (
        observation
        + "\n\nHere are some guidelines for solving similar tasks:\n"
        + json.dumps(guidelines, indent=2, ensure_ascii=False)
        + "\n"
    )


def summarize_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Cannot summarize an empty result set")
    task_ids = [result["task_id"] for result in results]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Results contain duplicate task IDs")

    successes = [result for result in results if bool(result["reward"])]
    steps = [int(result["steps"]) for result in results]
    successful_steps = [int(result["steps"]) for result in successes]
    termination_counts = Counter(result["termination_reason"] for result in results)
    error_count = sum(bool(result.get("error")) for result in results)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "condition": results[0]["condition"],
        "split": results[0]["split"],
        "task_count": len(results),
        "success_count": len(successes),
        "failure_count": len(results) - len(successes),
        "success_rate": len(successes) / len(results),
        "average_steps": sum(steps) / len(steps),
        "average_success_steps": (
            sum(successful_steps) / len(successful_steps) if successful_steps else None
        ),
        "error_count": error_count,
        "termination_counts": dict(sorted(termination_counts.items())),
        "task_ids": task_ids,
    }


def write_results(
    output_dir: str | Path,
    results: Sequence[dict[str, Any]],
    *,
    summary_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = Path(output_dir)
    task_dir = directory / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    for index, result in enumerate(results):
        digest = hashlib.sha256(result["task_id"].encode("utf-8")).hexdigest()[:12]
        write_json(task_dir / f"{index:05d}_{digest}.json", result)

    jsonl = "".join(json.dumps(result, ensure_ascii=False) + "\n" for result in results)
    (directory / "results.jsonl").write_text(jsonl, encoding="utf-8")
    summary = summarize_results(results)
    if summary_metadata:
        summary.update(summary_metadata)
    write_json(directory / "summary.json", summary)
    _write_csv(directory / "summary.csv", [summary])
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                for key, value in row.items()
            }
        )
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(scalar_rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(scalar_rows)


def build_paired_comparison(
    no_memory: dict[str, Any], memory: dict[str, Any]
) -> dict[str, Any]:
    if no_memory.get("task_ids") != memory.get("task_ids"):
        raise ValueError("Cannot compare conditions with different task IDs or order")
    if no_memory.get("split") != memory.get("split"):
        raise ValueError("Cannot compare conditions from different splits")
    no_memory_parameters = no_memory.get("parameters", {})
    memory_parameters = memory.get("parameters", {})
    paired_keys = (
        "model",
        "agent_api_base_url",
        "split",
        "seed",
        "batch_size",
        "max_steps",
        "temperature",
        "top_p",
        "few_shot",
        "top_k",
        "manifest_sha256",
    )
    mismatches = [
        key
        for key in paired_keys
        if no_memory_parameters.get(key) != memory_parameters.get(key)
    ]
    if mismatches:
        raise ValueError("Paired experiment parameter mismatch: " + ", ".join(mismatches))
    no_memory_sr = float(no_memory["success_rate"])
    memory_sr = float(memory["success_rate"])
    absolute = memory_sr - no_memory_sr
    relative = absolute / no_memory_sr if no_memory_sr else None
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "split": no_memory["split"],
        "task_count": no_memory["task_count"],
        "no_memory_success_rate": no_memory_sr,
        "memory_success_rate": memory_sr,
        "absolute_improvement": absolute,
        "absolute_improvement_percentage_points": absolute * 100,
        "relative_improvement": relative,
        "task_ids": no_memory["task_ids"],
    }


def maybe_write_paired_comparison(experiment_dir: str | Path) -> dict[str, Any] | None:
    root = Path(experiment_dir)
    paths = {condition: root / condition / "summary.json" for condition in CONDITIONS}
    if not all(path.is_file() for path in paths.values()):
        return None
    comparison = build_paired_comparison(
        load_json(paths["no_memory"]), load_json(paths["memory"])
    )
    write_json(root / "comparison.json", comparison)
    _write_csv(root / "comparison.csv", [comparison])
    return comparison
