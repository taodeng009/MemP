"""Align three memory runs per model and estimate offloading propensities."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np


def values(edge_count, cloud_count):
    return ((3 - edge_count) * cloud_count / 9,
            edge_count * (3 - cloud_count) / 9,
            (cloud_count - edge_count) / 3)


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--edge-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--cloud-runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    datasets, sources = [], []
    keys = ['condition', 'split', 'seed', 'batch_size', 'max_steps', 'temperature',
            'top_p', 'few_shot', 'top_k', 'embedding_model', 'manifest_sha256',
            'memory_type', 'retrieval_pipeline', 'score_threshold']
    for path in [*args.edge_runs, *args.cloud_runs]:
        file = path / 'results.jsonl'
        records = {}
        for line in file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['task_id'] in records or not isinstance(row['reward'], bool):
                raise ValueError(f'Duplicate ID or invalid reward: {path}')
            records[row['task_id']] = row
        if len(records) != 134:
            raise ValueError(f'Expected 134 records: {path}')
        summary = json.loads((path / 'summary.json').read_text(encoding='utf-8'))
        successes = sum(r['reward'] for r in records.values())
        if summary['task_count'] != 134 or summary['success_count'] != successes:
            raise ValueError(f'Summary mismatch: {path}')
        datasets.append(records)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest(),
                        'success_count': successes, 'success_rate': successes / 134,
                        'error_count': sum(bool(r.get('error')) for r in records.values()),
                        'models': sorted({r['model'] for r in records.values()})})
    reference = datasets[0]
    reference_parameters = next(iter(reference.values()))['parameters']
    for dataset in datasets:
        if set(dataset) != set(reference):
            raise ValueError('Task ID sets differ')
        for task_id, row in dataset.items():
            if row['query'] != reference[task_id]['query'] or row['task_index'] != reference[task_id]['task_index']:
                raise ValueError('Task query/index mismatch')
            for key in keys:
                if row['parameters'].get(key) != reference_parameters.get(key):
                    raise ValueError(f'Experiment parameter differs: {key}')
    tasks = []
    joint = Counter()
    for task_id, row in sorted(reference.items(), key=lambda item: item[1]['task_index']):
        e = [int(d[task_id]['reward']) for d in datasets[:3]]
        c = [int(d[task_id]['reward']) for d in datasets[3:]]
        es, cs = sum(e), sum(c)
        v, n, delta = values(es, cs)
        joint[es, cs] += 1
        assert abs(v - n - delta) < 1e-12
        tasks.append({'task_id': task_id, 'task_index': row['task_index'], 'query': row['query'],
                      'edge_success_pattern': ''.join(map(str, e)),
                      'cloud_success_pattern': ''.join(map(str, c)),
                      'edge_success_count': es, 'cloud_success_count': cs,
                      'p_edge': es / 3, 'p_cloud': cs / 3,
                      'rescue_value': v, 'negative_transfer_propensity': n,
                      'net_success_gain': delta})
    metrics, distributions = {}, []
    for name in ['rescue_value', 'negative_transfer_propensity', 'net_success_gain']:
        array = np.array([row[name] for row in tasks])
        metrics[name] = {'mean': float(array.mean()), 'median': float(np.median(array)),
                         'quantiles': {str(q): float(np.quantile(array, q, method='linear'))
                                       for q in [0, .05, .10, .25, .50, .75, .90, .95, 1]}}
        for value, count in sorted(Counter(array).items()):
            distributions.append({'metric': name, 'value_fraction': str(Fraction(float(value)).limit_denominator(9)),
                                  'value': float(value), 'task_count': count, 'task_proportion': count / 134})
    negative = [row for row in tasks if row['negative_transfer_propensity'] > 0]
    summary = {'task_count': 134, 'runs_per_model': 3, 'sources': sources,
               'parameters_checked': keys, 'quantile_method': 'linear',
               'mean_p_edge': float(np.mean([r['p_edge'] for r in tasks])),
               'mean_p_cloud': float(np.mean([r['p_cloud'] for r in tasks])),
               'metrics': metrics, 'distributions': distributions,
               'negative_transfer_positive_task_count': len(negative),
               'negative_transfer_positive_task_proportion': len(negative) / 134,
               'net_gain_sign_counts': {label: sum(test(r['net_success_gain']) for r in tasks)
                                       for label, test in [('positive', lambda x: x > 0),
                                                           ('zero', lambda x: x == 0), ('negative', lambda x: x < 0)]},
               'interpretation': 'Products of task-level marginal empirical probabilities; not matched-run joint event frequencies or causal effects.'}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'task_level_propensities.csv', tasks)
    write_csv(args.output_dir / 'negative_transfer_positive_tasks.csv', negative)
    write_csv(args.output_dir / 'distributions.csv', distributions)
    write_csv(args.output_dir / 'joint_success_counts.csv', [
        {'edge_success_count': e, 'cloud_success_count': c, 'task_count': joint[e, c]}
        for e in range(4) for c in range(4)])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
