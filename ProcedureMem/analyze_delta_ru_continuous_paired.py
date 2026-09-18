"""Analyze positive paired Delta-RU with equal-frequency sliding windows.

This is an offline analysis over existing ALFWorld online-construction logs.  It
does not invoke ALFWorld, the agent model, the embedding model, or the memory
builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from statistics import mean, median


RUN_DIRS = {
    "run4": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run4",
    "run5": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run5",
    "run6": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run6",
    "run7": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run7",
}

POLICIES = {
    "Coverage": "online_construction_oracle_coverage",
    "Exact": "online_construction_oracle_exact_retrieval_h1",
    "HitQuality": "online_construction_oracle_hit_quality_ru_h1_alpha0.25",
}


def extended_path(path: Path) -> Path:
    """Use the Windows extended-length prefix for deeply nested result paths."""
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def load_tasks(path: Path) -> dict[str, dict[str, float | int | str]]:
    tasks: dict[str, dict[str, float | int | str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            threshold = float(row["parameters"]["score_threshold"])
            retrieval_utility = sum(
                max(0.0, threshold - float(memory["score"]))
                for memory in row.get("retrieved_memories", [])
            )
            tasks[row["task_id"]] = {
                "ru": retrieval_utility,
                "success": int(bool(row["reward"])),
                "task_family": row["task_type"].split("-", 1)[0],
            }
    return tasks


def collect_pairs(root: Path) -> list[dict[str, float | int | str]]:
    pairs: list[dict[str, float | int | str]] = []
    for run, run_dir in RUN_DIRS.items():
        run_root = root / run_dir
        fifo = load_tasks(
            run_root / "online_construction_fifo_shortest_first" / "results.jsonl"
        )
        for strategy, policy_dir in POLICIES.items():
            policy = load_tasks(run_root / policy_dir / "results.jsonl")
            if fifo.keys() != policy.keys():
                raise ValueError(f"Task IDs differ for {run} / {strategy}")
            for task_id, fifo_row in fifo.items():
                policy_row = policy[task_id]
                delta_ru = float(policy_row["ru"]) - float(fifo_row["ru"])
                if delta_ru <= 1e-12:
                    continue
                pairs.append(
                    {
                        "run": run,
                        "strategy": strategy,
                        "task_id": task_id,
                        "task_family": str(fifo_row["task_family"]),
                        "delta_ru": delta_ru,
                        "fifo_success": int(fifo_row["success"]),
                        "policy_success": int(policy_row["success"]),
                    }
                )
    return pairs


def starts_for_windows(size: int, window: int, step: int) -> list[int]:
    if size <= window:
        return [0]
    starts = list(range(0, size - window + 1, step))
    last = size - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_windows(
    rows: list[dict[str, float | int | str]], window: int, step: int
) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: float(row["delta_ru"]))
    output: list[dict[str, float | int]] = []
    for index, start in enumerate(starts_for_windows(len(ordered), window, step), 1):
        chunk = ordered[start : start + min(window, len(ordered))]
        values = [float(row["delta_ru"]) for row in chunk]
        up = sum(
            int(row["fifo_success"]) == 0 and int(row["policy_success"]) == 1
            for row in chunk
        )
        down = sum(
            int(row["fifo_success"]) == 1 and int(row["policy_success"]) == 0
            for row in chunk
        )
        output.append(
            {
                "window_index": index,
                "rank_start": start + 1,
                "rank_end": start + len(chunk),
                "n": len(chunk),
                "delta_ru_min": min(values),
                "delta_ru_max": max(values),
                "delta_ru_mean": mean(values),
                "delta_ru_median": median(values),
                "success_0_to_1": up,
                "success_1_to_0": down,
                "net_success": up - down,
                "net_success_rate": (up - down) / len(chunk),
            }
        )
    return output


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def persistent_positive_threshold(windows: list[dict[str, float | int]]) -> float | None:
    for index, window in enumerate(windows):
        if all(float(later["net_success_rate"]) > 0 for later in windows[index:]):
            return float(window["delta_ru_mean"])
    return None


def build_analysis(pairs: list[dict[str, float | int | str]]):
    windows: list[dict[str, float | int | str]] = []
    summaries: list[dict[str, float | int | str | None]] = []
    for strategy in POLICIES:
        for scope in [*RUN_DIRS, "pooled"]:
            subset = [
                row
                for row in pairs
                if row["strategy"] == strategy
                and (scope == "pooled" or row["run"] == scope)
            ]
            window_size, step = (30, 10) if scope == "pooled" else (15, 5)
            scoped_windows = make_windows(subset, window_size, step)
            for window in scoped_windows:
                windows.append({"strategy": strategy, "scope": scope, **window})
            xs = [float(row["delta_ru_mean"]) for row in scoped_windows]
            ys = [float(row["net_success_rate"]) for row in scoped_windows]
            summaries.append(
                {
                    "strategy": strategy,
                    "scope": scope,
                    "positive_pair_count": len(subset),
                    "window_size": min(window_size, len(subset)),
                    "step": step,
                    "window_count": len(scoped_windows),
                    "window_curve_pearson": pearson(xs, ys),
                    "persistent_positive_threshold": persistent_positive_threshold(
                        scoped_windows
                    ),
                    "first_rate": ys[0],
                    "last_rate": ys[-1],
                }
            )
    return windows, summaries


def write_csv(path: Path, windows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader()
        writer.writerows(windows)


def write_html(path: Path, windows, summaries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"windows": windows, "summaries": summaries},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    fragment = f"""<div id="delta-ru-continuous-root">
  <h1>Positive ΔRU vs Net Success Rate</h1>
  <div class="text-small text-muted">Equal-frequency sliding windows · per run: N=15, step=5 · pooled: N=30, step=10</div>
  <div id="delta-ru-strategies"></div>
  <div class="tooltip" role="tooltip" hidden></div>
</div>
<style>
#delta-ru-continuous-root {{ position: relative; width: 100%; color: var(--foreground); }}
#delta-ru-continuous-root h1 {{ margin-bottom: 4px; }}
#delta-ru-continuous-root .strategy-section {{ margin-top: 22px; }}
#delta-ru-continuous-root .strategy-section h2 {{ margin-bottom: 8px; }}
#delta-ru-continuous-root .chart-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }}
#delta-ru-continuous-root .chart-panel {{ min-width: 0; }}
#delta-ru-continuous-root .chart-panel h3 {{ margin: 0 0 4px; text-align: center; }}
#delta-ru-continuous-root .chart-panel svg {{ display: block; width: 100%; min-height: 220px; }}
#delta-ru-continuous-root .axis path,
#delta-ru-continuous-root .axis line {{ stroke: var(--border); }}
#delta-ru-continuous-root .axis text {{ fill: var(--foreground); font-size: 11px; }}
#delta-ru-continuous-root .axis-title {{ fill: var(--foreground); font-size: 12px; }}
#delta-ru-continuous-root .grid line {{ stroke: var(--border); stroke-opacity: 0.45; }}
#delta-ru-continuous-root .grid path {{ display: none; }}
#delta-ru-continuous-root .curve {{ fill: none; stroke: var(--viz-series-1); stroke-width: 2; }}
#delta-ru-continuous-root .point {{ fill: var(--viz-series-1); stroke: var(--background); stroke-width: 1.5; }}
#delta-ru-continuous-root .zero-line {{ stroke: var(--foreground); stroke-opacity: 0.65; stroke-width: 1; stroke-dasharray: 4 3; }}
#delta-ru-continuous-root .chart-frame {{ fill: none; stroke: var(--border); }}
#delta-ru-continuous-root .tooltip {{ position: absolute; pointer-events: none; z-index: 5; background: var(--popover); color: var(--popover-foreground); border: 1px solid var(--border); padding: 8px; border-radius: 6px; font-size: 12px; }}
@media (max-width: 900px) {{ #delta-ru-continuous-root .chart-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
@media (max-width: 520px) {{ #delta-ru-continuous-root .chart-grid {{ grid-template-columns: 1fr; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('delta-ru-continuous-root');
  const data = {payload};
  const strategies = ['Coverage', 'Exact', 'HitQuality'];
  const scopes = ['run4', 'run5', 'run6', 'run7', 'pooled'];
  const host = root.querySelector('#delta-ru-strategies');
  const tooltip = root.querySelector('.tooltip');

  strategies.forEach(strategy => {{
    const section = document.createElement('section');
    section.className = 'strategy-section';
    const heading = document.createElement('h2');
    heading.textContent = strategy;
    section.appendChild(heading);
    const grid = document.createElement('div');
    grid.className = 'chart-grid';
    section.appendChild(grid);
    host.appendChild(section);

    const strategyRows = data.windows.filter(d => d.strategy === strategy);
    const maxX = d3.max(strategyRows, d => d.delta_ru_mean) || 1;
    scopes.forEach(scope => {{
      const panel = document.createElement('div');
      panel.className = 'chart-panel';
      const title = document.createElement('h3');
      title.textContent = scope === 'pooled' ? 'Pooled' : scope;
      panel.appendChild(title);
      const svg = d3.select(panel).append('svg')
        .attr('role', 'img')
        .attr('aria-label', `${{strategy}} ${{scope}} ΔRU versus net success rate`);
      svg.append('title').text(`${{strategy}} ${{scope}}`);
      svg.append('desc').text('Equal-frequency sliding-window curve. Horizontal axis is mean ΔRU and vertical axis is net success rate.');
      grid.appendChild(panel);

      const draw = () => {{
        const width = Math.max(220, panel.getBoundingClientRect().width);
        const height = 230;
        const margin = {{top: 8, right: 12, bottom: 48, left: 60}};
        svg.attr('viewBox', `0 0 ${{width}} ${{height}}`);
        svg.selectAll('g, path, line, circle, rect, text').remove();
        const rows = strategyRows.filter(d => d.scope === scope);
        const x = d3.scaleLinear().domain([0, maxX * 1.05]).range([margin.left, width - margin.right]);
        const y = d3.scaleLinear().domain([-0.55, 0.75]).range([height - margin.bottom, margin.top]);
        svg.append('g').attr('class', 'grid').attr('transform', `translate(${{margin.left}},0)`)
          .call(d3.axisLeft(y).ticks(5).tickSize(-(width - margin.left - margin.right)).tickFormat(''));
        svg.append('rect').attr('class', 'chart-frame').attr('data-chart-frame', '')
          .attr('x', margin.left).attr('y', margin.top)
          .attr('width', width - margin.left - margin.right).attr('height', height - margin.top - margin.bottom);
        svg.append('line').attr('class', 'zero-line')
          .attr('x1', margin.left).attr('x2', width - margin.right)
          .attr('y1', y(0)).attr('y2', y(0));
        svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${{height - margin.bottom}})`)
          .call(d3.axisBottom(x).ticks(width < 260 ? 3 : 4).tickFormat(d3.format('.2f')));
        svg.append('g').attr('class', 'axis').attr('transform', `translate(${{margin.left}},0)`)
          .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.0%')));
        svg.append('text').attr('class', 'axis-title').attr('data-axis', 'x')
          .attr('x', (margin.left + width - margin.right) / 2).attr('y', height - 8)
          .attr('text-anchor', 'middle').text('Mean ΔRU');
        svg.append('text').attr('class', 'axis-title').attr('data-axis', 'y')
          .attr('transform', 'rotate(-90)').attr('x', -(margin.top + height - margin.bottom) / 2)
          .attr('y', 15).attr('text-anchor', 'middle').text('Net success rate');
        const line = d3.line().x(d => x(d.delta_ru_mean)).y(d => y(d.net_success_rate));
        svg.append('path').datum(rows).attr('class', 'curve').attr('d', line);
        svg.selectAll('.point').data(rows).join('circle').attr('class', 'point')
          .attr('cx', d => x(d.delta_ru_mean)).attr('cy', d => y(d.net_success_rate)).attr('r', 4)
          .on('mouseenter', (event, d) => {{
            tooltip.hidden = false;
            tooltip.innerHTML = `<strong>${{strategy}} · ${{scope}} · window ${{d.window_index}}</strong><br>` +
              `ΔRU: ${{d.delta_ru_min.toFixed(3)}}–${{d.delta_ru_max.toFixed(3)}} (mean ${{d.delta_ru_mean.toFixed(3)}})<br>` +
              `N=${{d.n}}, 0→1=${{d.success_0_to_1}}, 1→0=${{d.success_1_to_0}}<br>` +
              `Net rate=${{(100*d.net_success_rate).toFixed(1)}}%`;
            const box = root.getBoundingClientRect();
            const target = event.currentTarget.getBoundingClientRect();
            tooltip.style.left = `${{target.left - box.left + 10}}px`;
            tooltip.style.top = `${{target.top - box.top - 70}}px`;
          }})
          .on('mouseleave', () => {{ tooltip.hidden = true; }});
      }};
      draw();
      new ResizeObserver(draw).observe(panel);
    }});
  }});
}})();
</script>
"""
    path.write_text(fragment, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--html", type=Path, required=True)
    args = parser.parse_args()

    pairs = collect_pairs(extended_path(args.results_root))
    windows, summaries = build_analysis(pairs)
    write_csv(args.csv, windows)
    write_html(args.html, windows, summaries)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
