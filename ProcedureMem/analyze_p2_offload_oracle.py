"""Paired P2 oracle feasibility analysis across model and memory states.

For each memory state, a task has positive oracle offloading value when the
Edge model fails and the Cloud model succeeds.  The analysis aligns four
existing result logs by task ID and measures how that set changes after adding
offline memory to both models.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


STATE_NAMES = {
    (True, True): "both_success",
    (True, False): "edge_only_success",
    (False, True): "offload_value",
    (False, False): "neither_success",
}


def load_results(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    results_file = path / "results.jsonl" if path.is_dir() else path
    rows: dict[str, dict[str, Any]] = {}
    with results_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row["task_id"])
            if task_id in rows:
                raise ValueError(
                    f"Duplicate task_id in {results_file}:{line_number}: {task_id}"
                )
            rows[task_id] = row
    if not rows:
        raise ValueError(f"No task results found in {results_file}")
    first = next(iter(rows.values()))
    params = first.get("parameters") or {}
    metadata = {
        "path": str(path),
        "results_file": str(results_file),
        "experiment_name": first.get("experiment_name"),
        "model": first.get("model"),
        "condition": first.get("condition"),
        "task_count": len(rows),
        "success_count": sum(bool(row.get("reward")) for row in rows.values()),
        "batch_size": params.get("batch_size"),
        "seed": params.get("seed"),
        "max_steps": params.get("max_steps"),
        "top_k": params.get("top_k"),
        "score_threshold": params.get("score_threshold"),
        "manifest_sha256": params.get("manifest_sha256"),
    }
    return rows, metadata


def validate_task_sets(result_sets: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    names = list(result_sets)
    reference_name = names[0]
    reference = set(result_sets[reference_name])
    for name in names[1:]:
        current = set(result_sets[name])
        if current != reference:
            missing = sorted(reference - current)
            extra = sorted(current - reference)
            raise ValueError(
                f"Task IDs differ between {reference_name} and {name}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
    return sorted(reference, key=lambda task_id: result_sets[reference_name][task_id].get("task_index", 10**9))


def exact_mcnemar_p_value(changed_one_way: int, changed_other_way: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant pairs."""

    discordant = changed_one_way + changed_other_way
    if discordant == 0:
        return 1.0
    lower_tail = sum(
        math.comb(discordant, k) for k in range(min(changed_one_way, changed_other_way) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * lower_tail)


def task_family(task_type: str) -> str:
    prefixes = (
        "look_at_obj_in_light",
        "pick_and_place_simple",
        "pick_clean_then_place_in_recep",
        "pick_cool_then_place_in_recep",
        "pick_heat_then_place_in_recep",
        "pick_two_obj_and_place",
    )
    for prefix in prefixes:
        if task_type.startswith(prefix):
            return prefix
    return task_type.split("-", 1)[0]


def analyze(
    edge_no_memory: dict[str, dict[str, Any]],
    edge_memory: dict[str, dict[str, Any]],
    cloud_no_memory: dict[str, dict[str, Any]],
    cloud_memory: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    result_sets = {
        "edge_no_memory": edge_no_memory,
        "edge_memory": edge_memory,
        "cloud_no_memory": cloud_no_memory,
        "cloud_memory": cloud_memory,
    }
    task_ids = validate_task_sets(result_sets)
    rows: list[dict[str, Any]] = []

    for task_id in task_ids:
        source = edge_no_memory[task_id]
        e0 = bool(edge_no_memory[task_id].get("reward"))
        e1 = bool(edge_memory[task_id].get("reward"))
        c0 = bool(cloud_no_memory[task_id].get("reward"))
        c1 = bool(cloud_memory[task_id].get("reward"))
        o0 = (not e0) and c0
        o1 = (not e1) and c1
        if o0 and not o1:
            flip_type = (
                "offload_to_edge_success" if e1 else "offload_to_neither_success"
            )
        elif not o0 and o1:
            flip_type = (
                "edge_success_to_offload" if e0 else "neither_success_to_offload"
            )
        elif o0 and o1:
            flip_type = "offload_value_persisted"
        else:
            flip_type = "no_offload_value_in_either_state"
        raw_task_type = str(source.get("task_type") or "")
        rows.append(
            {
                "task_index": source.get("task_index"),
                "task_id": task_id,
                "task_family": task_family(raw_task_type),
                "task_type": raw_task_type,
                "query": source.get("query"),
                "edge_no_memory_success": int(e0),
                "cloud_no_memory_success": int(c0),
                "no_memory_state": STATE_NAMES[(e0, c0)],
                "no_memory_offload_value": int(o0),
                "edge_memory_success": int(e1),
                "cloud_memory_success": int(c1),
                "memory_state": STATE_NAMES[(e1, c1)],
                "memory_offload_value": int(o1),
                "flip_type": flip_type,
                "edge_no_memory_steps": edge_no_memory[task_id].get("steps"),
                "cloud_no_memory_steps": cloud_no_memory[task_id].get("steps"),
                "edge_memory_steps": edge_memory[task_id].get("steps"),
                "cloud_memory_steps": cloud_memory[task_id].get("steps"),
            }
        )

    total = len(rows)
    no_set = {row["task_id"] for row in rows if row["no_memory_offload_value"]}
    memory_set = {row["task_id"] for row in rows if row["memory_offload_value"]}
    removed = no_set - memory_set
    added = memory_set - no_set
    overlap = no_set & memory_set
    union = no_set | memory_set

    flip_counts = {
        label: sum(row["flip_type"] == label for row in rows)
        for label in (
            "offload_to_edge_success",
            "offload_to_neither_success",
            "edge_success_to_offload",
            "neither_success_to_offload",
            "offload_value_persisted",
            "no_offload_value_in_either_state",
        )
    }

    def condition_summary(edge_key: str, cloud_key: str) -> dict[str, Any]:
        edge_success = sum(row[edge_key] for row in rows)
        cloud_success = sum(row[cloud_key] for row in rows)
        prefix = "no_memory" if edge_key.startswith("edge_no_memory") else "memory"
        states = {
            name: sum(row[f"{prefix}_state"] == name for row in rows)
            for name in STATE_NAMES.values()
        }
        return {
            "edge_success_count": edge_success,
            "edge_success_rate": edge_success / total,
            "cloud_success_count": cloud_success,
            "cloud_success_rate": cloud_success / total,
            "oracle_union_success_count": total - states["neither_success"],
            "oracle_union_success_rate": (total - states["neither_success"]) / total,
            "state_counts": states,
            "offload_value_count": states["offload_value"],
            "offload_value_rate": states["offload_value"] / total,
        }

    edge_fail_to_success = sum(
        not row["edge_no_memory_success"] and row["edge_memory_success"] for row in rows
    )
    edge_success_to_fail = sum(
        row["edge_no_memory_success"] and not row["edge_memory_success"] for row in rows
    )
    summary = {
        "schema_version": 1,
        "definition": "offload_value = Edge failure and Cloud success for the same task and memory state",
        "task_count": total,
        "no_memory": condition_summary(
            "edge_no_memory_success", "cloud_no_memory_success"
        ),
        "memory": condition_summary("edge_memory_success", "cloud_memory_success"),
        "offload_set_comparison": {
            "no_memory_count": len(no_set),
            "memory_count": len(memory_set),
            "count_change": len(memory_set) - len(no_set),
            "overlap_count": len(overlap),
            "removed_count": len(removed),
            "added_count": len(added),
            "union_count": len(union),
            "jaccard_similarity": len(overlap) / len(union) if union else 1.0,
            "jaccard_distance": 1 - (len(overlap) / len(union)) if union else 0.0,
            "exact_mcnemar_p_value": exact_mcnemar_p_value(len(removed), len(added)),
        },
        "requested_transitions": {
            "offload_to_edge_success_count": flip_counts["offload_to_edge_success"],
            "edge_success_to_offload_count": flip_counts["edge_success_to_offload"],
            "net_edge_resolution_of_offload_value": (
                flip_counts["offload_to_edge_success"]
                - flip_counts["edge_success_to_offload"]
            ),
        },
        "offload_membership_transition_counts": flip_counts,
        "edge_success_transition": {
            "failure_to_success_count": edge_fail_to_success,
            "success_to_failure_count": edge_success_to_fail,
            "exact_mcnemar_p_value": exact_mcnemar_p_value(
                edge_fail_to_success, edge_success_to_fail
            ),
        },
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
) -> None:
    comparison = summary["offload_set_comparison"]
    transitions = summary["requested_transitions"]
    flips = summary["offload_membership_transition_counts"]
    no_memory = summary["no_memory"]
    memory = summary["memory"]
    batch_sizes = {item["batch_size"] for item in metadata.values()}
    lines = [
        "# ALFWorld P2 Oracle Offloading Feasibility Analysis",
        "",
        "## 定义与数据对齐",
        "",
        "对每个 memory state，将同一 task 上 `Edge fail ∧ Cloud success` 定义为具有正向 oracle offloading value。四组日志按完整 task ID 严格对齐，共 134 个共同任务；该指标只表示 outcome oracle 的可行空间，不是可部署 routing policy。",
        "",
        "| 条件 | Edge 成功 | Cloud 成功 | Edge fail / Cloud success | Edge∪Cloud oracle 成功 |",
        "|---|---:|---:|---:|---:|",
        f'| No memory | {no_memory["edge_success_count"]}/134 ({100 * no_memory["edge_success_rate"]:.2f}%) | {no_memory["cloud_success_count"]}/134 ({100 * no_memory["cloud_success_rate"]:.2f}%) | {no_memory["offload_value_count"]}/134 ({100 * no_memory["offload_value_rate"]:.2f}%) | {no_memory["oracle_union_success_count"]}/134 ({100 * no_memory["oracle_union_success_rate"]:.2f}%) |',
        f'| Offline 300 MemP | {memory["edge_success_count"]}/134 ({100 * memory["edge_success_rate"]:.2f}%) | {memory["cloud_success_count"]}/134 ({100 * memory["cloud_success_rate"]:.2f}%) | {memory["offload_value_count"]}/134 ({100 * memory["offload_value_rate"]:.2f}%) | {memory["oracle_union_success_count"]}/134 ({100 * memory["oracle_union_success_rate"]:.2f}%) |',
        "",
        "## Offload 集合变化",
        "",
        f'- No-memory offload set：{comparison["no_memory_count"]} 个任务。',
        f'- Memory offload set：{comparison["memory_count"]} 个任务，净变化 {comparison["count_change"]:+d}。',
        f'- 两集合重叠 {comparison["overlap_count"]}，移出 {comparison["removed_count"]}，新增 {comparison["added_count"]}；Jaccard similarity={comparison["jaccard_similarity"]:.4f}。',
        f'- 对 offload 标签变化做 paired exact McNemar test：p={comparison["exact_mcnemar_p_value"]:.6g}。',
        "",
        "变化分解：",
        "",
        "| 变化类型 | 数量 | 解释 |",
        "|---|---:|---|",
        f'| Offload → Edge success | {flips["offload_to_edge_success"]} | 原来 Edge fail / Cloud success，加入 memory 后 Edge 成功 |',
        f'| Offload → Neither success | {flips["offload_to_neither_success"]} | 原有 Cloud offload gain 在 memory 条件下消失，Edge 仍失败 |',
        f'| Edge success → Offload | {flips["edge_success_to_offload"]} | 原来 Edge 成功，加入 memory 后 Edge 失败而 Cloud 成功 |',
        f'| Neither success → Offload | {flips["neither_success_to_offload"]} | 原来 Edge/Cloud 都失败，加入 memory 后仅 Cloud 成功 |',
        f'| Offload value persisted | {flips["offload_value_persisted"]} | 两种 memory state 下均值得 offload |',
        "",
        "## 针对研究问题的结论",
        "",
        f'在原 no-memory offload set 中，有 {transitions["offload_to_edge_success_count"]} 个任务在加入 memory 后变为 Edge 可成功；反向有 {transitions["edge_success_to_offload_count"]} 个任务从 Edge 可成功变为需要 offload，净 Edge-resolution 为 {transitions["net_edge_resolution_of_offload_value"]:+d}。',
        "",
        "因此，不能只根据 offload task 总数的净变化判断 memory 的影响：集合的移出、新增和保留共同决定 offloading value。完整 task-level 明细见同目录 CSV。",
        "",
        "## 解释限制",
        "",
    ]
    if len(batch_sizes) > 1:
        lines.append(
            "- 4B no-memory 日志使用 `batch_size=1`，另外三组使用 `batch_size=2`。因此这里的 task-level flips 同时包含 memory effect、batch-size effect 和模型推理随机性，不能作为严格的因果效应估计。"
        )
    lines += [
        "- 这是单次 outcome oracle 分析；Cloud 成功并不代表实际 router 能事前识别该任务。",
        "- Memory 同时改变 Edge 与 Cloud outcome，所以 offload set 的变化必须按上述四类分解，不能全部归因于 Edge 端改善。",
        "- 要形成严格结论，应以相同 batch size、同一 inference service 条件运行重复实验，并报告集合稳定性与均值/方差。",
        "",
        "## 输入日志元数据",
        "",
        "| 角色 | 模型 | 条件 | 成功数 | Batch size | Manifest SHA-256 |",
        "|---|---|---|---:|---:|---|",
    ]
    for name, item in metadata.items():
        lines.append(
            f'| `{name}` | `{item["model"]}` | `{item["condition"]}` | '
            f'{item["success_count"]}/{item["task_count"]} | {item["batch_size"]} | '
            f'`{item["manifest_sha256"]}` |'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-no-memory", type=Path, required=True)
    parser.add_argument("--edge-memory", type=Path, required=True)
    parser.add_argument("--cloud-no-memory", type=Path, required=True)
    parser.add_argument("--cloud-memory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inputs = {
        "edge_no_memory": args.edge_no_memory,
        "edge_memory": args.edge_memory,
        "cloud_no_memory": args.cloud_no_memory,
        "cloud_memory": args.cloud_memory,
    }
    loaded = {name: load_results(path) for name, path in inputs.items()}
    result_sets = {name: value[0] for name, value in loaded.items()}
    metadata = {name: value[1] for name, value in loaded.items()}
    summary, rows = analyze(**result_sets)
    manifest_hashes = {item["manifest_sha256"] for item in metadata.values()}
    summary["inputs"] = metadata
    summary["validation"] = {
        "task_ids_aligned": True,
        "shared_manifest_sha256": next(iter(manifest_hashes))
        if len(manifest_hashes) == 1
        else None,
        "manifest_hashes_match": len(manifest_hashes) == 1,
        "batch_sizes_match": len({item["batch_size"] for item in metadata.values()})
        == 1,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "task_level_paired.csv", rows)
    changed = [
        row
        for row in rows
        if row["no_memory_offload_value"] != row["memory_offload_value"]
    ]
    write_csv(args.output_dir / "offload_value_flips.csv", changed)
    write_report(args.output_dir / "analysis.md", summary, metadata)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
