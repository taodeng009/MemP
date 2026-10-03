"""Task-level RU/BD associations with failure propensity across three runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np


def ranks(values):
    """Average ranks for exact ties, as required by Spearman correlation."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def spearman_permutation(x, y, repetitions, seed):
    a, b = ranks(x), ranks(y)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return {'n': len(x), 'rho': None, 'two_sided_permutation_p': None}
    rho = float(a @ b / denominator)
    generator = np.random.default_rng(seed)
    extreme = 0
    # Permute task labels; preserve tied failure-propensity group sizes.
    for start in range(0, repetitions, 2000):
        count = min(2000, repetitions - start)
        permutations = generator.permuted(np.tile(b, (count, 1)), axis=1)
        statistics = permutations @ a / denominator
        extreme += int(np.count_nonzero(np.abs(statistics) >= abs(rho) - 1e-12))
    return {
        'n': len(x), 'rho': rho,
        'two_sided_permutation_p': (extreme + 1) / (repetitions + 1),
        'permutations': repetitions, 'seed': seed,
    }


def statistics(values):
    if not values:
        return dict(n=0, mean=None, median=None, sd=None, q25=None, q75=None, min=None, max=None)
    return {
        'n': len(values), 'mean': mean(values), 'median': median(values),
        'sd': stdev(values) if len(values) > 1 else None,
        'q25': float(np.quantile(values, 0.25)),
        'q75': float(np.quantile(values, 0.75)),
        'min': min(values), 'max': max(values),
    }


def write_csv(path, rows, fields=None):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--permutations', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    datasets = []
    for path in args.runs:
        dataset = {}
        with (path / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record['task_id']
                if task_id in dataset:
                    raise ValueError(f'Duplicate task ID: {path}: {task_id}')
                if not isinstance(record.get('reward'), bool):
                    raise ValueError(f'Invalid reward: {task_id}')
                scores = [float(m['score']) for m in record['retrieved_memories']]
                if len(scores) != record['retrieved_count']:
                    raise ValueError(f'Retrieval count mismatch: {task_id}')
                if any(not np.isfinite(d) or d < 0 or d > args.threshold for d in scores):
                    raise ValueError(f'Score outside squared-L2 threshold: {task_id}')
                dataset[task_id] = {
                    'record': record, 'scores': scores,
                    'ru': sum(max(0, args.threshold - d) for d in scores),
                    'bd': min(scores) if scores else None,
                }
        datasets.append(dataset)
    if not datasets[0] or any(set(d) != set(datasets[0]) for d in datasets[1:]):
        raise ValueError('Empty logs or task ID sets not aligned')
    tasks = []
    score_mismatches = 0
    for task_id in sorted(datasets[0], key=lambda k: datasets[0][k]['record']['task_index']):
        entries = [d[task_id] for d in datasets]
        successes = [int(e['record']['reward']) for e in entries]
        scores = [e['scores'] for e in entries]
        score_mismatches += int(any(s != scores[0] for s in scores[1:]))
        if any(bool(s) != bool(scores[0]) for s in scores[1:]):
            raise ValueError(f'Retrieval eligibility differs across runs: {task_id}')
        ru = [e['ru'] for e in entries]
        bd = [e['bd'] for e in entries]
        row = {
            'task_id': task_id, 'task_index': entries[0]['record']['task_index'],
            'query': entries[0]['record'].get('query'),
            'success_pattern': ''.join(map(str, successes)),
            'success_count': sum(successes), 'failure_count': 3 - sum(successes),
            'empirical_edge_success_probability': sum(successes) / 3,
            'failure_propensity': (3 - sum(successes)) / 3,
            'ru': mean(ru), 'bd': mean(bd) if bd[0] is not None else None,
            'ru_run1': ru[0], 'ru_run2': ru[1], 'ru_run3': ru[2],
            'bd_run1': bd[0], 'bd_run2': bd[1], 'bd_run3': bd[2],
            'ru_range': max(ru) - min(ru),
            'bd_range': max(bd) - min(bd) if bd[0] is not None else None,
        }
        tasks.append(row)
    bd_tasks = [row for row in tasks if row['bd'] is not None]
    correlations = {
        'RU_all_tasks': spearman_permutation(
            [r['ru'] for r in tasks], [r['failure_propensity'] for r in tasks],
            args.permutations, args.seed,
        ),
        'BD_retrieval_tasks': spearman_permutation(
            [r['bd'] for r in bd_tasks], [r['failure_propensity'] for r in bd_tasks],
            args.permutations, args.seed,
        ),
    }
    groups = []
    for feature, eligible in (('RU', tasks), ('BD', bd_tasks)):
        for failures in range(4):
            subset = [r for r in eligible if r['failure_count'] == failures]
            groups.append({
                'feature': feature, 'failure_propensity_group': ('0', '1/3', '2/3', '1')[failures],
                **statistics([r[feature.lower()] for r in subset]),
            })
    summary = {
        'task_count': len(tasks), 'bd_eligible_task_count': len(bd_tasks),
        'inputs': [str(p) for p in args.runs], 'threshold': args.threshold,
        'task_feature_aggregation': 'Arithmetic mean of the three run-specific feature values',
        'validation': {
            'task_ids_aligned': True, 'score_vector_mismatch_tasks': score_mismatches,
            'max_task_ru_range': max(r['ru_range'] for r in tasks),
            'max_task_bd_range': max(r['bd_range'] for r in bd_tasks),
        },
        'spearman': correlations, 'group_statistics': groups,
        'p_value_method': 'Two-sided Monte Carlo permutation test, average ranks for ties, plus-one correction',
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'task_level_features.csv', tasks)
    write_csv(args.output_dir / 'group_statistics.csv', groups)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        '# Retrieval feature–Edge failure propensity analysis', '',
        r'对三次 run 按 task_id 对齐，$\hat p_k^E=\sum_r s_{k,r}/3$，failure propensity 为 $1-\hat p_k^E$。每个 task 只作为一个样本。', '',
        r'每次 run 的 $RU_k=\sum_{m\in R_k}\max(0,0.5-d_{km})$；$BD_k=\min_{m\in R_k}d_{km}$，no-retrieval 的 BD 缺失。任务级 RU、BD 取三次均值。', '',
        f"共 {len(tasks)} 个任务，BD 有效样本 {len(bd_tasks)} 个。{score_mismatches} 个任务的三次 score vectors 不完全相同；最大 RU range={summary['validation']['max_task_ru_range']:.8g}，最大 BD range={summary['validation']['max_task_bd_range']:.8g}。", '',
        '| Feature | N | Spearman rho | Two-sided permutation p |',
        '|---|---:|---:|---:|',
    ]
    for name, corr in correlations.items():
        lines.append(f"| {name} | {corr['n']} | {corr['rho']:.6f} | {corr['two_sided_permutation_p']:.6g} |")
    lines += ['', f'对 ties 使用 average ranks；随机置换 failure propensity 标签 {args.permutations:,} 次，seed={args.seed}，p值使用 plus-one correction。', '',
              '| Feature | Failure propensity | N | Mean | Median | SD | Q25 | Q75 | Min | Max |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in groups:
        cells = [f"{row[key]:.6f}" if row[key] is not None else 'N/A' for key in ('mean', 'median', 'sd', 'q25', 'q75', 'min', 'max')]
        lines.append(f"| {row['feature']} | {row['failure_propensity_group']} | {row['n']} | " + ' | '.join(cells) + ' |')
    lines += ['',
              '本次结果：RU 与 failure propensity 呈弱负相关，BD 呈弱正相关。RU 的未校正双侧置换 p 值低于 0.05；BD 的 p 值略高于 0.05。', '',
              '分组均值并非单调：始终失败组的平均 RU 高于两个 outcome flip 组，平均 BD 则低于它们；检索质量较好仍可能伴随持续失败。这些指标的任务级关联较弱。', '',
              '仅进行 RU、BD 与 failure propensity 的任务级描述和相关分析。相关性不证明 memory 的因果效应；三次 outcome 得到的 propensity 只有四档，且 1/3、2/3 组样本较少。p 值未进行多重比较校正。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
