"""Summarize ALFWorld Edge-50/100/150 P0 evaluation results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


CONDITION_ORDER = (
    "no_memory",
    "edge_raw_50",
    "edge_raw_100",
    "edge_raw_150",
    "cloud_workflow_300",
)
EDGE_PAIRS = (("edge_raw_50", "edge_raw_100"), ("edge_raw_100", "edge_raw_150"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as writer:
        csv_writer = csv.DictWriter(writer, fieldnames=list(rows[0]))
        csv_writer.writeheader()
        csv_writer.writerows(rows)


def summarize_edge_p0(results_dir: str | Path) -> dict[str, Any]:
    root = Path(results_dir)
    available = [
        condition
        for condition in CONDITION_ORDER
        if (root / condition / "summary.json").is_file()
        and (root / condition / "results.jsonl").is_file()
    ]
    required_edges = {"edge_raw_50", "edge_raw_100", "edge_raw_150"}
    if not required_edges.issubset(available):
        missing = sorted(required_edges - set(available))
        raise FileNotFoundError("Missing Edge P0 result conditions: " + ", ".join(missing))

    summaries = {condition: _load_json(root / condition / "summary.json") for condition in available}
    results = {condition: _load_jsonl(root / condition / "results.jsonl") for condition in available}
    reference_ids = [item["task_id"] for item in results[available[0]]]
    for condition in available[1:]:
        if [item["task_id"] for item in results[condition]] != reference_ids:
            raise ValueError(f"Task IDs or order differ for condition {condition}")

    overview = []
    task_type_rows = []
    for condition in available:
        summary = summaries[condition]
        overview.append(
            {
                "condition": condition,
                "task_count": summary["task_count"],
                "success_count": summary["success_count"],
                "success_rate": summary["success_rate"],
                "average_steps": summary["average_steps"],
                "average_success_steps": summary["average_success_steps"],
                "error_count": summary["error_count"],
            }
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in results[condition]:
            grouped[item["task_type"]].append(item)
        for task_type in sorted(grouped):
            items = grouped[task_type]
            success_count = sum(bool(item["reward"]) for item in items)
            task_type_rows.append(
                {
                    "condition": condition,
                    "task_type": task_type,
                    "task_count": len(items),
                    "success_count": success_count,
                    "success_rate": success_count / len(items),
                }
            )

    transitions = []
    for source, target in EDGE_PAIRS:
        source_by_id = {item["task_id"]: bool(item["reward"]) for item in results[source]}
        target_by_id = {item["task_id"]: bool(item["reward"]) for item in results[target]}
        counts = Counter(
            (source_by_id[task_id], target_by_id[task_id]) for task_id in reference_ids
        )
        transitions.append(
            {
                "source": source,
                "target": target,
                "failure_to_success": counts[(False, True)],
                "success_to_failure": counts[(True, False)],
                "both_success": counts[(True, True)],
                "both_failure": counts[(False, False)],
            }
        )

    output = {
        "task_count": len(reference_ids),
        "available_conditions": available,
        "overview": overview,
        "edge_transitions": transitions,
        "task_type_results": task_type_rows,
    }
    (root / "capacity_comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(root / "capacity_comparison.csv", overview)
    _write_csv(root / "task_type_summary.csv", task_type_rows)
    _write_csv(root / "edge_transitions.csv", transitions)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = summarize_edge_p0(args.results_dir)
    for row in summary["overview"]:
        print(
            f"{row['condition']}: SR={row['success_rate']:.4f} "
            f"({row['success_count']}/{row['task_count']}) "
            f"avg_steps={row['average_steps']:.2f}"
        )
    print(f"Wrote P0 comparison to {args.results_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
