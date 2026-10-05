"""Task-only Ridge probe of empirical rescue value using frozen query CV splits."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import analyze_task_semantics_linear_probe as probe
else:
    import analyze_task_semantics_linear_probe as probe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--propensities', type=Path, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, failure, groups, original, model = probe.load_inputs(args.features, args.embedding_cache)
    with args.propensities.open(encoding='utf-8') as handle:
        targets = list(csv.DictReader(handle))
    by_id = {row['task_id']: row for row in targets}
    if len(by_id) != len(targets) or set(by_id) != {row['task_id'] for row in rows}:
        raise ValueError('Target task IDs mismatch or duplicated')
    y = []
    for i, row in enumerate(rows):
        target = by_id[row['task_id']]
        pe, pc = float(target['p_edge']), float(target['p_cloud'])
        value = float(target['rescue_value'])
        if row['query'] != target['query'] or not np.isclose(failure[i], 1 - pe, atol=1e-12, rtol=0):
            raise ValueError('Query or Edge propensity mismatch')
        if not 0 <= pc <= 1 or not np.isclose(value, (1 - pe) * pc, atol=1e-12, rtol=0):
            raise ValueError('Invalid rescue value')
        y.append(value)
    y = np.asarray(y)
    previous = json.loads((args.previous_cv_dir / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        old_oof = list(csv.DictReader(handle))
    if [r['task_id'] for r in rows] != [r['task_id'] for r in old_oof]:
        raise ValueError('Previous CV task order mismatch')
    designs = {'embedding': original['embedding']}
    performance, folds, tuning, actual_manifest, oof, fold_ids = probe.evaluate(
        y, groups, designs, previous['alpha_grid'], previous['seed'], fold_manifest=manifest)
    # Independently reproduce the old control before comparing targets.
    control = probe.evaluate(failure, groups, designs, previous['alpha_grid'], previous['seed'], fold_manifest=manifest)
    if not np.allclose(control[4]['embedding'], [float(r['embedding_prediction']) for r in old_oof], atol=1e-12, rtol=0):
        raise ValueError('Failure propensity control not reproduced')
    old = {r['model']: r for r in previous['performance']}
    new = {r['model']: r for r in performance}
    comparisons = []
    for target, index in [('failure_propensity', old), ('rescue_value', new)]:
        base, task = index['mean_baseline'], index['embedding']
        comparisons.append({'target': target,
                            'mae_reduction_vs_baseline': base['mae'] - task['mae'],
                            'relative_mae_reduction': 1 - task['mae'] / base['mae'],
                            'rmse_reduction_vs_baseline': base['rmse'] - task['rmse'],
                            'relative_rmse_reduction': 1 - task['rmse'] / base['rmse'],
                            'r2_increase_vs_baseline': task['r2'] - base['r2'],
                            'task_oof_spearman': task['spearman']})
    summary = {'task_count': len(rows), 'canonical_query_groups': len(set(groups)),
               'target': '(1 - empirical Edge success probability) * empirical Cloud success probability',
               'embedding_model': model, 'feature_dimensions': 768,
               'seed': previous['seed'], 'alpha_grid': previous['alpha_grid'],
               'protocol': 'Exact previous outer 5-fold / inner 3-fold groups; train-only standardization; pooled inner MSE tuning; raw unclipped Ridge; unpenalized intercept',
               'previous_failure_oof_reproduced': True, 'performance': performance,
               'previous_failure_performance': [old['mean_baseline'], old['embedding']],
               'baseline_comparisons': comparisons,
               'folds_with_lower_error_than_baseline': {
                   metric: sum(t[metric] < b[metric] for t, b in zip(
                       [r for r in folds if r['model'] == 'embedding'],
                       [r for r in folds if r['model'] == 'mean_baseline'])) for metric in ['mae', 'rmse']},
               'target_variance': {'rescue_value': float(np.var(y)), 'failure_propensity': float(np.var(failure))},
               'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                   args.features, args.embedding_cache, args.propensities, args.previous_cv_dir / 'fold_manifest.json']}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [('cv_summary.csv', performance), ('fold_metrics.csv', folds),
                          ('inner_alpha_selection.csv', tuning), ('target_comparisons.csv', comparisons)]:
        probe.write_csv(args.output_dir / name, records)
    probe.write_csv(args.output_dir / 'task_oof_predictions.csv', [
        {'task_id': r['task_id'], 'query': r['query'], 'canonical_query': groups[i],
         'fold': int(fold_ids[i]), 'rescue_value': float(y[i]),
         'p_edge': by_id[r['task_id']]['p_edge'], 'p_cloud': by_id[r['task_id']]['p_cloud'],
         **{f'{name}_prediction': float(v[i]) for name, v in oof.items()}} for i, r in enumerate(rows)])
    (args.output_dir / 'fold_manifest.json').write_text(json.dumps(actual_manifest, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
