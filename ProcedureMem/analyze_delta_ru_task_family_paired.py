"""Task-family diagnostic for positive paired Delta-RU in existing logs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_delta_ru_continuous_paired import (
    POLICIES,
    RUN_DIRS,
    collect_pairs,
    extended_path,
)


SCOPES = [*RUN_DIRS, "pooled"]
BINS = ("Low", "Mid", "High")


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + fraction * (ordered[high] - ordered[low])


def bin_name(value: float, q33: float, q67: float) -> str:
    if value <= q33:
        return "Low"
    if value <= q67:
        return "Mid"
    return "High"


def summarize(pairs):
    families = sorted({str(row["task_family"]) for row in pairs})
    thresholds = []
    rows = []
    pair_rows = []

    for strategy in POLICIES:
        for family in families:
            pooled = [
                row
                for row in pairs
                if row["strategy"] == strategy and row["task_family"] == family
            ]
            values = [float(row["delta_ru"]) for row in pooled]
            q33 = quantile(values, 1 / 3)
            q67 = quantile(values, 2 / 3)
            thresholds.append(
                {
                    "strategy": strategy,
                    "task_family": family,
                    "positive_pair_count": len(pooled),
                    "q33": q33,
                    "q67": q67,
                }
            )
            for row in pooled:
                pair_rows.append(
                    {
                        **row,
                        "bin": bin_name(float(row["delta_ru"]), q33, q67),
                        "delta_success": int(row["policy_success"])
                        - int(row["fifo_success"]),
                    }
                )

            for scope in SCOPES:
                scoped = pooled if scope == "pooled" else [r for r in pooled if r["run"] == scope]
                for band in BINS:
                    subset = [
                        row
                        for row in scoped
                        if bin_name(float(row["delta_ru"]), q33, q67) == band
                    ]
                    up = sum(
                        int(row["fifo_success"]) == 0
                        and int(row["policy_success"]) == 1
                        for row in subset
                    )
                    down = sum(
                        int(row["fifo_success"]) == 1
                        and int(row["policy_success"]) == 0
                        for row in subset
                    )
                    rows.append(
                        {
                            "strategy": strategy,
                            "task_family": family,
                            "scope": scope,
                            "bin": band,
                            "q33": q33,
                            "q67": q67,
                            "pair_count": len(subset),
                            "success_0_to_1": up,
                            "success_1_to_0": down,
                            "net_success": up - down,
                            "net_success_rate": (up - down) / len(subset)
                            if subset
                            else None,
                            "mean_delta_ru": mean(float(r["delta_ru"]) for r in subset)
                            if subset
                            else None,
                        }
                    )
    return families, thresholds, rows, pair_rows


def family_diagnostic(pair_rows):
    output = []
    for strategy in POLICIES:
        strategy_rows = [row for row in pair_rows if row["strategy"] == strategy]
        bin_means = {
            band: mean(
                float(row["delta_success"])
                for row in strategy_rows
                if row["bin"] == band
            )
            for band in BINS
        }
        run_bin_means = {
            (run, band): mean(
                float(row["delta_success"])
                for row in strategy_rows
                if row["run"] == run and row["bin"] == band
            )
            for run in RUN_DIRS
            for band in BINS
            if any(
                row["run"] == run and row["bin"] == band
                for row in strategy_rows
            )
        }
        bin_residuals = defaultdict(list)
        run_bin_residuals = defaultdict(list)
        for row in strategy_rows:
            family = str(row["task_family"])
            bin_residuals[family].append(
                float(row["delta_success"]) - bin_means[str(row["bin"])]
            )
            run_bin_residuals[family].append(
                float(row["delta_success"])
                - run_bin_means[(str(row["run"]), str(row["bin"]))]
            )
        for family, values in sorted(bin_residuals.items()):
            output.append(
                {
                    "strategy": strategy,
                    "task_family": family,
                    "pair_count": len(values),
                    "mean_bin_adjusted_delta_success": mean(values),
                    "mean_run_bin_adjusted_delta_success": mean(
                        run_bin_residuals[family]
                    ),
                }
            )
    return output


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


def write_markdown(path: Path, families, thresholds, rows, diagnostic) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# ALFWorld 正 ΔRU paired data 的 task-family diagnostic（run4–run7）",
        "",
        "只分析 `ΔRU > 0` 的 task pairs。每个 `strategy × task family` 使用 run4–run7 pooled 样本计算 ΔRU 三分位边界，再将相同边界用于各 run。表格单元依次为 `pair 数 / 0→1 / 1→0 / Net Success Rate`。",
        "",
        "## 分档边界",
        "",
        "| 策略 | Task family | 正 ΔRU pairs | Q33 | Q67 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in thresholds:
        lines.append(
            f'| {row["strategy"]} | `{row["task_family"]}` | '
            f'{row["positive_pair_count"]} | {row["q33"]:.4f} | {row["q67"]:.4f} |'
        )

    index = {
        (row["strategy"], row["task_family"], row["scope"], row["bin"]): row
        for row in rows
    }
    for strategy in POLICIES:
        lines += [
            "",
            f"## {strategy}",
            "",
            "| Scope | Task family | Low | Mid | High |",
            "|---|---|---:|---:|---:|",
        ]
        for scope in SCOPES:
            for family in families:
                values = [
                    cell(index[(strategy, family, scope, band)]) for band in BINS
                ]
                lines.append(
                    f"| {scope} | `{family}` | {values[0]} | {values[1]} | {values[2]} |"
                )

    lines += [
        "",
        "## 控制 ΔRU 档位后的 family residual",
        "",
        "`Bin-adjusted` 先减去每个策略对应 Low/Mid/High 档的 pooled 平均 ΔSuccess；`Run+bin-adjusted` 进一步减去对应 `strategy + run + ΔRU档位` 的平均 ΔSuccess。正值表示控制这些因素后，该 family 的 success conversion 仍高于策略总体；负值表示仍低于总体。",
        "",
        "| 策略 | Task family | Pair 数 | Bin-adjusted mean ΔSuccess | Run+bin-adjusted mean ΔSuccess |",
        "|---|---|---:|---:|---:|",
    ]
    for row in diagnostic:
        lines.append(
            f'| {row["strategy"]} | `{row["task_family"]}` | {row["pair_count"]} | '
            f'{100 * row["mean_bin_adjusted_delta_success"]:+.2f} pp | '
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
    families, thresholds, rows, pair_rows = summarize(pairs)
    diagnostic = family_diagnostic(pair_rows)
    write_csv(args.csv, rows)
    write_markdown(args.report, families, thresholds, rows, diagnostic)
    print(
        json.dumps(
            {"thresholds": thresholds, "diagnostic": diagnostic},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
