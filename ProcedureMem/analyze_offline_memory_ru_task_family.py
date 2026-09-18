"""Analyze retrieval utility versus success by ALFWorld task family."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean


RUN_DIRS = {
    "run1": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run1",
    "run2": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run2",
    "run3": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run3",
}

THRESHOLD = 0.5
BINS = ("Low", "Mid", "High")


def extended_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def load_rows(root: Path):
    rows = []
    signatures: dict[str, dict[str, tuple]] = {}
    for run, directory in RUN_DIRS.items():
        path = root / directory / "memory" / "results.jsonl"
        signatures[run] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                memories = row.get("retrieved_memories", [])
                ru = sum(
                    max(0.0, THRESHOLD - float(memory["score"]))
                    for memory in memories
                )
                family = row["task_type"].split("-", 1)[0]
                rows.append(
                    {
                        "run": run,
                        "task_id": row["task_id"],
                        "task_family": family,
                        "ru": ru,
                        "success": int(bool(row["reward"])),
                        "retrieved_count": int(row["retrieved_count"]),
                    }
                )
                signatures[run][row["task_id"]] = {
                    "names": tuple(memory.get("task_name") for memory in memories),
                    "workflows": tuple(memory.get("workflow") for memory in memories),
                    "scores": tuple(round(float(memory["score"]), 12) for memory in memories),
                    "ru": ru,
                }
    reference = signatures["run1"]
    diagnostics = {}
    for run in ("run2", "run3"):
        diagnostics[run] = {
            "name_order_mismatch_count": sum(
                signatures[run][task_id]["names"] != reference[task_id]["names"]
                for task_id in reference
            ),
            "workflow_mismatch_count": sum(
                signatures[run][task_id]["workflows"]
                != reference[task_id]["workflows"]
                for task_id in reference
            ),
            "ru_mismatch_count": sum(
                abs(signatures[run][task_id]["ru"] - reference[task_id]["ru"])
                > 1e-12
                for task_id in reference
            ),
            "max_abs_delta_ru": max(
                abs(signatures[run][task_id]["ru"] - reference[task_id]["ru"])
                for task_id in reference
            ),
        }
    return rows, diagnostics


def band(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "Low"
    if value <= q67:
        return "Mid"
    return "High"


def analyze(rows):
    families = sorted({row["task_family"] for row in rows})
    boundaries = {}
    for family in families:
        values = [row["ru"] for row in rows if row["task_family"] == family]
        boundaries[family] = (quantile(values, 1 / 3), quantile(values, 2 / 3))

    summaries = []
    bins = []
    for scope in [*RUN_DIRS, "pooled"]:
        scoped = rows if scope == "pooled" else [row for row in rows if row["run"] == scope]
        for family in families:
            family_rows = [row for row in scoped if row["task_family"] == family]
            successes = [row for row in family_rows if row["success"]]
            failures = [row for row in family_rows if not row["success"]]
            summaries.append(
                {
                    "scope": scope,
                    "task_family": family,
                    "task_count": len(family_rows),
                    "success_count": len(successes),
                    "success_rate": len(successes) / len(family_rows),
                    "mean_ru": mean(row["ru"] for row in family_rows),
                    "mean_ru_success": mean(row["ru"] for row in successes)
                    if successes
                    else None,
                    "mean_ru_failure": mean(row["ru"] for row in failures)
                    if failures
                    else None,
                    "ru_success_pearson": pearson(
                        [row["ru"] for row in family_rows],
                        [row["success"] for row in family_rows],
                    ),
                }
            )
            q33, q67 = boundaries[family]
            for bin_name in BINS:
                subset = [
                    row
                    for row in family_rows
                    if band(row["ru"], q33, q67) == bin_name
                ]
                success_count = sum(row["success"] for row in subset)
                bins.append(
                    {
                        "scope": scope,
                        "task_family": family,
                        "bin": bin_name,
                        "q33": q33,
                        "q67": q67,
                        "task_count": len(subset),
                        "success_count": success_count,
                        "success_rate": success_count / len(subset) if subset else None,
                        "mean_ru": mean(row["ru"] for row in subset) if subset else None,
                    }
                )
    return families, boundaries, summaries, bins


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def bin_cell(row) -> str:
    if not row or not row["task_count"]:
        return "0 / 0 / —"
    return f'{row["task_count"]} / {row["success_count"]} / {100 * row["success_rate"]:.1f}%'


def write_report(path, families, boundaries, summaries, bins, diagnostics):
    summary_index = {(row["scope"], row["task_family"]): row for row in summaries}
    bin_index = {
        (row["scope"], row["task_family"], row["bin"]): row for row in bins
    }
    lines = [
        "# ALFWorld offline memory：task family 下 RU 与 SR 的关系",
        "",
        "分析 run1–run3 的 `memory` condition。RU 使用实际 Top-3 retrieved squared-L2 distances 与固定 retrieval cutoff 0.5 计算：",
        "",
        "\\[RU_i=\\sum_{d\\in D_i^{topK}}\\max(0,0.5-d)\\]",
        "",
        "run1 与 run2 的 retrieved task-name/order、workflow 和 RU 完全一致。run3 的 retrieved task-name/order 仍完全一致，但有29个任务的 workflow 文本不同、34个任务的 RU 存在浮点级差异；最大绝对 ΔRU 仅为 "
        f"{diagnostics['run3']['max_abs_delta_ru']:.8f}。",
        "",
        "每个 family 使用该 family 的 pooled RU 三分位边界；各 run 复用相同边界。分档表单元为 `task数 / success数 / SR`。",
        "",
        "## Family-specific RU 边界",
        "",
        "| Task family | Q33 | Q67 |",
        "|---|---:|---:|",
    ]
    for family in families:
        q33, q67 = boundaries[family]
        lines.append(f"| `{family}` | {q33:.4f} | {q67:.4f} |")

    for scope in [*RUN_DIRS, "pooled"]:
        lines += [
            "",
            f"## {scope}",
            "",
            "| Task family | N | SR | RU(success) | RU(failure) | Pearson r | Low | Mid | High |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for family in families:
            row = summary_index[(scope, family)]
            cells = [bin_cell(bin_index[(scope, family, name)]) for name in BINS]
            lines.append(
                f'| `{family}` | {row["task_count"]} | {100 * row["success_rate"]:.1f}% | '
                f'{fmt(row["mean_ru_success"])} | {fmt(row["mean_ru_failure"])} | '
                f'{fmt(row["ru_success_pearson"])} | {cells[0]} | {cells[1]} | {cells[2]} |'
            )
    lines += [
        "",
        "## 结论",
        "",
        "1. `pick_cool_then_place_in_recep` 的 RU–SR 关系最强且跨 run 最稳定。Pooled Pearson r=0.487，Low/Mid/High SR 为47.6%/85.7%/100.0%；三个 run 的 High 档均为100%。",
        "2. `pick_heat_then_place_in_recep` 也存在较明确的正向关系。Pooled r=0.371，Low/Mid/High SR 为33.3%/33.3%/76.2%；三个 run 的 High 档分别为71.4%、71.4%、85.7%。",
        "3. `pick_clean_then_place_in_recep` 只有弱正相关（r=0.144），且分档关系非单调：59.4%/41.9%/80.0%。高 RU 有帮助，但 Mid 不优于 Low。",
        "4. `look_at_obj_in_light` 与 `pick_and_place_simple` 的相关性较弱且非单调。前者为52.4%/83.3%/40.0%；后者整体存在90.3%的成功率上限效应，使 RU 难以进一步区分成功和失败。",
        "5. `pick_two_obj_and_place` 几乎全部失败（pooled SR=2.0%），RU 与成功呈弱负相关。该 family 的困难主要不是 retrieval similarity 不足，RU 无法作为有效成功代理。",
        "6. Family 间不能直接用绝对 RU 比较 SR。例如 `pick_heat` 的平均 RU 明显高于 `pick_and_place_simple`，但 SR 分别只有46.4%和90.3%。Task difficulty 与操作结构是强混杂因素。",
        "7. run1 与 run2 的 retrieval 完全一致；run3 虽有29个 workflow 文本不同，但 retrieved task-name/order 相同，最大绝对 RU 差异仅9.6e-5。因此 pooled RU 趋势主要不是由 run3 的 retrieval distance 漂移造成的。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--bins-csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    rows, diagnostics = load_rows(extended_path(args.results_root))
    families, boundaries, summaries, bins = analyze(rows)
    write_csv(args.summary_csv, summaries)
    write_csv(args.bins_csv, bins)
    write_report(
        args.report, families, boundaries, summaries, bins, diagnostics
    )
    print(
        json.dumps(
            {
                "retrieval_diagnostics": diagnostics,
                "boundaries": boundaries,
                "pooled": [row for row in summaries if row["scope"] == "pooled"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
