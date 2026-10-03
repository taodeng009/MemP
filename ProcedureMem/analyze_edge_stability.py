"""Align repeated Edge execution logs and summarize empirical success patterns."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    datasets = []
    metadata = []
    parameter_keys = (
        'model', 'condition', 'split', 'seed', 'batch_size', 'max_steps',
        'temperature', 'top_p', 'few_shot', 'top_k', 'embedding_model',
        'manifest_sha256', 'memory_type', 'retrieval_pipeline',
    )
    for run in args.runs:
        records = {}
        with (run / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                task_id = row['task_id']
                if task_id in records:
                    raise ValueError(f'Duplicate task ID in {run}: {task_id}')
                if not isinstance(row.get('reward'), bool):
                    raise ValueError(f'Non-boolean reward in {run}: {task_id}')
                records[task_id] = row
        if not records:
            raise ValueError(f'Empty run: {run}')
        experiment = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
        parameters = {key: experiment.get(key) for key in parameter_keys}
        summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
        success_count = sum(row['reward'] for row in records.values())
        if summary['task_count'] != len(records) or summary['success_count'] != success_count:
            raise ValueError(f'Summary does not match execution logs: {run}')
        for row in records.values():
            row_parameters = row.get('parameters', {})
            for key, value in parameters.items():
                if key in row_parameters and row_parameters[key] != value:
                    raise ValueError(f'Inconsistent per-task parameter {key}: {run}')
        metadata.append({
            'path': str(run), 'parameters': parameters,
            'task_count': len(records), 'success_count': success_count,
            'success_rate': success_count / len(records),
            'error_count': sum(bool(row.get('error')) for row in records.values()),
        })
        datasets.append(records)
    if any(set(dataset) != set(datasets[0]) for dataset in datasets[1:]):
        raise ValueError('The three runs have different task ID sets')
    differences = {
        key: [item['parameters'][key] for item in metadata]
        for key in parameter_keys
        if any(item['parameters'][key] != metadata[0]['parameters'][key] for item in metadata[1:])
    }
    tasks = []
    for task_id, reference in sorted(datasets[0].items(), key=lambda item: item[1]['task_index']):
        outcomes = [int(dataset[task_id]['reward']) for dataset in datasets]
        count = sum(outcomes)
        tasks.append({
            'task_id': task_id, 'task_index': reference['task_index'],
            'query': reference.get('query'),
            'run1_success': outcomes[0], 'run2_success': outcomes[1],
            'run3_success': outcomes[2],
            'success_pattern': ''.join(map(str, outcomes)),
            'success_count': count, 'success_fraction': f'{count}/3',
            'empirical_edge_success_probability': count / 3,
            'outcome_flip': int(0 < count < 3),
        })
    total = len(tasks)
    buckets = Counter(task['success_count'] for task in tasks)
    distribution = [{
        'success_fraction': f'{count}/3',
        'empirical_edge_success_probability': count / 3,
        'task_count': buckets[count], 'task_proportion': buckets[count] / total,
    } for count in range(4)]
    flip_tasks = [task for task in tasks if task['outcome_flip']]
    summary = {
        'task_count': total, 'run_count': 3, 'inputs': metadata,
        'validation': {'task_ids_aligned': True, 'parameter_differences': differences},
        'success_count_distribution': distribution,
        'success_pattern_counts': {
            format(pattern, '03b'): sum(task['success_pattern'] == format(pattern, '03b') for task in tasks)
            for pattern in range(8)
        },
        'outcome_flip_definition': 'At least one success and at least one failure across three runs',
        'outcome_flip_task_count': len(flip_tasks),
        'outcome_flip_task_proportion': len(flip_tasks) / total,
        'stable_task_count': total - len(flip_tasks),
        'pooled_success_rate': sum(task['success_count'] for task in tasks) / (3 * total),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ('task_level_stability.csv', tasks), ('outcome_flip_tasks.csv', flip_tasks),
        ('success_count_summary.csv', distribution),
    ):
        fields = list(distribution[0]) if filename == 'success_count_summary.csv' else list(tasks[0])
        with (args.output_dir / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    lines = [
        '# 4B + offline 300 MemP Edge stability analysis', '',
        f'按完整 task_id 对齐 run1、run2、run3，共 {total} 个任务。pattern 位序为 run1/run2/run3；1 表示 reward=true，0 表示失败。', '',
        r'每个任务的经验成功概率为 $\hat p_k^E=(s_{k,1}+s_{k,2}+s_{k,3})/3$。', '',
        '| 成功次数 | 经验成功概率 | 任务数 | 比例 |',
        '|---|---:|---:|---:|',
    ]
    for row in distribution:
        lines.append(f"| {row['success_fraction']} | {row['empirical_edge_success_probability']:.4f} | {row['task_count']} | {100 * row['task_proportion']:.2f}% |")
    lines += ['', f'Outcome flip：{len(flip_tasks)}/{total}（{100 * len(flip_tasks) / total:.2f}%）。定义为三次结果中至少一次成功且至少一次失败，即 1/3 或 2/3。', '',
              '| Pattern | 任务数 |', '|---|---:|']
    for pattern, count in summary['success_pattern_counts'].items():
        lines.append(f'| {pattern} | {count} |')
    lines += ['', '| Run | 成功数 | SR | Error 数 |', '|---|---:|---:|---:|']
    for index, item in enumerate(metadata, start=1):
        lines.append(f"| run{index} | {item['success_count']}/{total} | {100 * item['success_rate']:.2f}% | {item['error_count']} |")
    lines += ['', f'参数差异：{json.dumps(differences, ensure_ascii=False)}。', '',
              '三次试验的经验概率仅取 0、1/3、2/3、1；0/3 和 3/3 表示本次重复中结果一致，不能据此断言真实成功概率为 0 或 1。', '',
              '全部任务的 pattern 与经验概率见 task_level_stability.csv；发生 flip 的 task ID 见 outcome_flip_tasks.csv。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
