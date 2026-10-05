"""Untrained -RU and BD priority ranking of empirical net offloading gain."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_task_semantics_linear_probe import average_ranks, write_csv
else:
    from analyze_task_semantics_linear_probe import average_ranks, write_csv


def spearman(a, b):
    a, b = average_ranks(a), average_ranks(b)
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--propensities', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with args.features.open(encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda r: int(r['task_index']))
    with args.propensities.open(encoding='utf-8') as handle:
        target_rows = list(csv.DictReader(handle))
    targets = {r['task_id']: r for r in target_rows}
    if len(rows) != 134 or len({r['task_id'] for r in rows}) != 134 or len(targets) != len(target_rows) or set(targets) != {r['task_id'] for r in rows}:
        raise ValueError('Expected aligned 134 unique task IDs')
    for row in rows:
        t = targets[row['task_id']]
        if row['query'] != t['query'] or not np.isclose(float(row['empirical_edge_success_probability']), float(t['p_edge']), atol=1e-12, rtol=0):
            raise ValueError('Task query or Edge probability mismatch')
        if not np.isclose(float(t['net_success_gain']), float(t['p_cloud']) - float(t['p_edge']), atol=1e-12, rtol=0):
            raise ValueError('Net gain mismatch')
    ru, bd = (np.array([float(r[key]) for r in rows]) for key in ['ru', 'bd'])
    pe, gain = (np.array([float(targets[r['task_id']][key]) for r in rows]) for key in ['p_edge', 'net_success_gain'])
    if not np.isfinite(np.column_stack([ru, bd, pe, gain])).all():
        raise ValueError('Nonfinite inputs')
    correlations = [{'feature': name, 'target': target, 'spearman': spearman(x, y)}
                    for name, x in [('RU', ru), ('BD', bd)]
                    for target, y in [('p_edge', pe), ('net_gain', gain)]]
    orders = {'minus_ru': np.argsort(ru, kind='stable'), 'bd': np.argsort(-bd, kind='stable'),
              'oracle': np.argsort(-gain, kind='stable')}
    ranking = []
    for b in [10, 20, 30, 40, 50]:
        for name, order in orders.items():
            ids = order[:b]
            ranking.append({'ranking': name, 'B': b, 'captured_gain': float(gain[ids].sum()),
                            'positive_tasks': int(np.sum(gain[ids] > 0)),
                            'negative_tasks': int(np.sum(gain[ids] < 0))})
        ranking.append({'ranking': 'random_expected', 'B': b, 'captured_gain': float(b * gain.mean()),
                        'positive_tasks': float(b * np.mean(gain > 0)),
                        'negative_tasks': float(b * np.mean(gain < 0))})
    ranks = {}
    for name, order in orders.items():
        rank = np.empty(len(rows), int)
        rank[order] = np.arange(1, len(rows) + 1)
        ranks[name] = rank
    summary = {'task_count': len(rows), 'features': 'Three-run mean RU and BD from previous analyses',
               'priority': {'minus_ru': '-RU descending (RU ascending)', 'bd': 'BD descending'},
               'ties': 'Original task_index ascending, independent of outcomes',
               'training': False, 'CV': False, 'correlations': correlations, 'ranking': ranking,
               'total_net_gain': float(gain.sum()),
               'random': 'Exact expectation over uniform size-B subsets without replacement',
               'oracle': 'Sort signed empirical net gain descending',
               'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.features, args.propensities]}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'correlations.csv', correlations)
    write_csv(args.output_dir / 'ranking.csv', ranking)
    write_csv(args.output_dir / 'task_priorities.csv', [
        {'task_id': r['task_id'], 'task_index': r['task_index'], 'query': r['query'],
         'ru': float(ru[i]), 'bd': float(bd[i]), 'priority_minus_ru': float(-ru[i]),
         'priority_bd': float(bd[i]), 'p_edge': float(pe[i]), 'net_gain': float(gain[i]),
         **{name + '_rank': int(rank[i]) for name, rank in ranks.items()}} for i, r in enumerate(rows)])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
