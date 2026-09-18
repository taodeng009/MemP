"""Exact-Retrieval task-family diagnostic with global positive Delta-RU bins."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_delta_ru_continuous_paired import RUN_DIRS, collect_pairs, extended_path


BINS = ("Low", "Mid", "High")
SCOPES = [*RUN_DIRS, "pooled"]


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def band(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "Low"
    if value <= q67:
        return "Mid"
    return "High"


def analyze(pairs):
    exact = [row for row in pairs if row["strategy"] == "Exact"]
    q33 = quantile([float(row["delta_ru"]) for row in exact], 1 / 3)
    q67 = quantile([float(row["delta_ru"]) for row in exact], 2 / 3)
    prepared = []
    for row in exact:
        prepared.append(
            {
                **row,
                "bin": band(float(row["delta_ru"]), q33, q67),
                "delta_success": int(row["policy_success"])
                - int(row["fifo_success"]),
            }
        )

    families = sorted({str(row["task_family"]) for row in prepared})
    stats = []
    for scope in SCOPES:
        for family in families:
            for bin_name in BINS:
                subset = [
                    row
                    for row in prepared
                    if row["task_family"] == family
                    and row["bin"] == bin_name
                    and (scope == "pooled" or row["run"] == scope)
                ]
                up = sum(int(row["delta_success"]) == 1 for row in subset)
                down = sum(int(row["delta_success"]) == -1 for row in subset)
                stats.append(
                    {
                        "scope": scope,
                        "task_family": family,
                        "bin": bin_name,
                        "global_q33": q33,
                        "global_q67": q67,
                        "pair_count": len(subset),
                        "success_0_to_1": up,
                        "success_1_to_0": down,
                        "net_success": up - down,
                        "net_success_rate": (up - down) / len(subset)
                        if subset
                        else None,
                        "mean_delta_ru": mean(float(row["delta_ru"]) for row in subset)
                        if subset
                        else None,
                    }
                )

    run_bin_means = {
        (run, bin_name): mean(
            float(row["delta_success"])
            for row in prepared
            if row["run"] == run and row["bin"] == bin_name
        )
        for run in RUN_DIRS
        for bin_name in BINS
        if any(row["run"] == run and row["bin"] == bin_name for row in prepared)
    }
    residuals = defaultdict(list)
    for row in prepared:
        residuals[str(row["task_family"])].append(
            float(row["delta_success"])
            - run_bin_means[(str(row["run"]), str(row["bin"]))]
        )
    diagnostic = [
        {
            "task_family": family,
            "pair_count": len(values),
            "mean_run_bin_adjusted_delta_success": mean(values),
        }
        for family, values in sorted(residuals.items())
    ]
    return q33, q67, families, stats, diagnostic


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell(row) -> str:
    if not row or not row["pair_count"]:
        return "0 / 0 / 0 / —"
    return (
        f'{row["pair_count"]} / {row["success_0_to_1"]} / '
        f'{row["success_1_to_0"]} / {100 * row["net_success_rate"]:+.1f}%'
    )


def write_report(path: Path, q33, q67, families, stats, diagnostic) -> None:
    lookup = {
        (row["scope"], row["task_family"], row["bin"]): row for row in stats
    }
    lines = [
        "# Exact Retrieval：全局 ΔRU 分档下的 task-family diagnostic",
        "",
        "只分析 run4–run7 中 Exact Retrieval 相对 FIFO 的 `ΔRU > 0` task pairs。全部 family 共用由195个正 ΔRU pairs 计算的一组全局三分位边界：",
        "",
        f"- Low：`0 < ΔRU ≤ {q33:.6f}`",
        f"- Mid：`{q33:.6f} < ΔRU ≤ {q67:.6f}`",
        f"- High：`ΔRU > {q67:.6f}`",
        "",
        "表格单元依次为 `pair 数 / 0→1 / 1→0 / Net Success Rate`。",
        "",
    ]
    for scope in SCOPES:
        lines += [
            f"## {scope}",
            "",
            "| Task family | Low | Mid | High |",
            "|---|---:|---:|---:|",
        ]
        for family in families:
            values = [cell(lookup[(scope, family, name)]) for name in BINS]
            lines.append(
                f"| `{family}` | {values[0]} | {values[1]} | {values[2]} |"
            )
        lines.append("")

    lines += [
        "## 控制 run 与全局 ΔRU 档位后的 family residual",
        "",
        "先减去对应 `run + global ΔRU bin` 的平均 ΔSuccess，再在 family 内求平均。",
        "",
        "| Task family | Pair 数 | Run+bin-adjusted mean ΔSuccess |",
        "|---|---:|---:|",
    ]
    for row in diagnostic:
        lines.append(
            f'| `{row["task_family"]}` | {row["pair_count"]} | '
            f'{100 * row["mean_run_bin_adjusted_delta_success"]:+.2f} pp |'
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    pairs = collect_pairs(extended_path(args.results_root))
    q33, q67, families, stats, diagnostic = analyze(pairs)
    write_csv(args.csv, stats)
    write_report(args.report, q33, q67, families, stats, diagnostic)
    print(
        json.dumps(
            {"q33": q33, "q67": q67, "diagnostic": diagnostic},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
