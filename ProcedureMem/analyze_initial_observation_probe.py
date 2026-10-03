"""Grouped Ridge feasibility analysis using a server-generated observation cache."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import analyze_task_semantics_linear_probe as probe
    from .build_initial_observation_embeddings import load_inputs, MODEL, FORMAT_VERSION
else:
    import analyze_task_semantics_linear_probe as probe
    from build_initial_observation_embeddings import load_inputs, MODEL, FORMAT_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--task-embedding-cache', type=Path, required=True)
    parser.add_argument('--observation-inputs', type=Path, required=True)
    parser.add_argument('--observation-embedding-cache', type=Path, required=True)
    parser.add_argument('--distance-features', type=Path, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-inputs-only', action='store_true')
    args = parser.parse_args()
    rows, y, groups, original_designs, task_model = probe.load_inputs(args.features, args.task_embedding_cache)
    inputs = load_inputs(args.observation_inputs)
    inputs_by_id = {row['task_id']: row for row in inputs}
    if set(inputs_by_id) != {row['task_id'] for row in rows}:
        raise ValueError('Observation input task IDs do not match target data')
    ordered_inputs = [inputs_by_id[row['task_id']] for row in rows]
    if any(a['query'] != b['query'] for a, b in zip(rows, ordered_inputs)):
        raise ValueError('Observation input queries do not match target data')
    with args.distance_features.open(encoding='utf-8') as handle:
        distances_by_id = {row['task_id']: row for row in csv.DictReader(handle)}
    if set(distances_by_id) != {row['task_id'] for row in rows}:
        raise ValueError('Distance task IDs do not match target data')
    distances = np.asarray([[float(distances_by_id[row['task_id']][f'd{i}']) for i in (1, 2, 3)] for row in rows])
    if not np.isfinite(distances).all():
        raise ValueError('Non-finite Top-3 distance')
    if any(abs(float(distances_by_id[row['task_id']]['failure_propensity']) - y[i]) > 1e-12
           for i, row in enumerate(rows)):
        raise ValueError('Distance file target does not match target data')
    previous = json.loads((args.previous_cv_dir / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        old_predictions = list(csv.DictReader(handle))
    if [row['task_id'] for row in rows] != [row['task_id'] for row in old_predictions]:
        raise ValueError('Previous CV order mismatch')
    if task_model != MODEL:
        raise ValueError('Existing task cache is not the required BGE model')
    if args.validate_inputs_only:
        print(json.dumps({'tasks': len(rows), 'canonical_query_groups': len(set(groups)),
                          'prepared_texts_aligned': True, 'distance_features_aligned': True,
                          'observation_cache_available': args.observation_embedding_cache.is_file()}, indent=2))
        return
    if not args.observation_embedding_cache.is_file():
        parser.error('Observation embedding cache is missing. Run the server embedding command and sync the .npz first.')
    with np.load(args.observation_embedding_cache, allow_pickle=False) as cache:
        cache_ids = list(cache['task_ids'].astype(str))
        matrix = np.asarray(cache['embeddings'], dtype=float)
        cache_texts = list(cache['input_texts'].astype(str))
        cache_hashes = list(cache['text_sha256'].astype(str))
        if (str(cache['model'].item()) != task_model
                or str(cache['format_version'].item()) != FORMAT_VERSION
                or str(cache['input_sha256'].item()) != hashlib.sha256(args.observation_inputs.read_bytes()).hexdigest()
                or len(set(cache_ids)) != len(rows) or set(cache_ids) != set(inputs_by_id)
                or matrix.shape != (134, 768) or not np.isfinite(matrix).all()
                or len(cache_texts) != len(rows) or len(cache_hashes) != len(rows)):
            raise ValueError('Observation cache model, inputs or vector shape mismatch')
        for i, task_id in enumerate(cache_ids):
            if (cache_texts[i] != inputs_by_id[task_id]['embedding_text']
                    or cache_hashes[i] != inputs_by_id[task_id]['text_sha256']):
                raise ValueError(f'Observation cache text/hash mismatch: {task_id}')
    positions = {task_id: i for i, task_id in enumerate(cache_ids)}
    observation = matrix[[positions[row['task_id']] for row in rows]]
    designs = {
        'task_embedding': original_designs['embedding'],
        'task_initial_observation_embedding': observation,
        'task_initial_observation_embedding_top3': np.column_stack([observation, distances]),
    }
    summaries, fold_rows, tuning_rows, actual_manifest, oof, fold_ids = probe.evaluate(
        y, groups, designs, previous['alpha_grid'], previous['seed'], fold_manifest=manifest)
    if not np.allclose(oof['task_embedding'], [float(row['embedding_prediction']) for row in old_predictions], atol=1e-12, rtol=0):
        raise ValueError('Task-only control does not reproduce previous OOF predictions')
    index = {row['model']: row for row in summaries}
    comparisons = []
    for candidate, reference in (
        ('task_initial_observation_embedding', 'task_embedding'),
        ('task_initial_observation_embedding_top3', 'task_embedding'),
        ('task_initial_observation_embedding_top3', 'task_initial_observation_embedding'),
        ('task_initial_observation_embedding', 'mean_baseline'),
    ):
        cf = [row for row in fold_rows if row['model'] == candidate]
        rf = [row for row in fold_rows if row['model'] == reference]
        comparisons.append({
            'candidate': candidate, 'reference': reference,
            'oof_mae_reduction': index[reference]['mae'] - index[candidate]['mae'],
            'oof_rmse_reduction': index[reference]['rmse'] - index[candidate]['rmse'],
            'oof_r2_increase': index[candidate]['r2'] - index[reference]['r2'],
            'oof_spearman_increase': index[candidate]['spearman'] - index[reference]['spearman'],
            'folds_with_lower_mae': sum(c['mae'] < r['mae'] for c, r in zip(cf, rf)),
            'folds_with_lower_rmse': sum(c['rmse'] < r['rmse'] for c, r in zip(cf, rf)),
        })
    input_files = [args.features, args.task_embedding_cache, args.observation_inputs,
                   args.observation_embedding_cache, args.distance_features,
                   args.previous_cv_dir / 'fold_manifest.json']
    summary = {
        'task_count': len(rows), 'canonical_query_count': len(set(groups)), 'embedding_model': task_model,
        'feature_dimensions': {name: x.shape[1] for name, x in designs.items()},
        'target': 'failure_propensity = three-run failure count / 3',
        'seed': previous['seed'], 'alpha_grid': previous['alpha_grid'],
        'folds': 'Reuse exact previous outer 5-fold and inner 3-fold query assignments',
        'scaling': 'Fit column means/std only on each training split',
        'prediction': 'Raw Ridge outputs without clipping; task-level equal weights',
        'distance_imputation': 'Tasks 8 and 48: missing d2/d3 replaced by 0.5 as in prior 134-task sensitivity analysis',
        'task_only_control_reproduced': True,
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files},
        'performance': summaries, 'comparisons': comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe.write_csv(args.output_dir / 'cv_summary.csv', summaries)
    probe.write_csv(args.output_dir / 'fold_metrics.csv', fold_rows)
    probe.write_csv(args.output_dir / 'inner_alpha_selection.csv', tuning_rows)
    probe.write_csv(args.output_dir / 'model_comparisons.csv', comparisons)
    probe.write_csv(args.output_dir / 'task_oof_predictions.csv', [{
        'task_id': row['task_id'], 'task_index': row['task_index'], 'query': row['query'],
        'canonical_query': groups[i], 'fold': int(fold_ids[i]), 'failure_propensity': float(y[i]),
        **{f'{name}_prediction': float(pred[i]) for name, pred in oof.items()},
    } for i, row in enumerate(rows)])
    (args.output_dir / 'fold_manifest.json').write_text(json.dumps(actual_manifest, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Initial observation feasibility analysis', '',
             '134 tasks / 78 canonical query groups。query与initial observation拼接为单个文本后生成embedding；memory guidelines与重复task goal已移除，输入不包含执行轨迹或outcome。', '',
             '复用原外层5-fold和内层3-fold，三组使用同一task集合。原Task-only OOF已精确复现。两个task的缺失d2/d3补为0.5，与之前134-task sensitivity设置一致。', '',
             '| Input | OOF MAE | OOF RMSE | OOF R² | OOF Spearman |', '|---|---:|---:|---:|---:|']
    for row in summaries:
        values = [f'{row[key]:.6f}' if row[key] is not None else 'N/A' for key in ('mae', 'rmse', 'r2', 'spearman')]
        lines.append(f"| {row['model']} | " + ' | '.join(values) + ' |')
    lines += ['', '| Input | Fold MAE mean ± SD | Fold RMSE mean ± SD | Fold R² mean ± SD | Fold Spearman mean ± SD |', '|---|---:|---:|---:|---:|']
    for row in summaries:
        values = [f"{row[f'fold_mean_{key}']:.6f} ± {row[f'fold_sd_{key}']:.6f}" if row[f'fold_sd_{key}'] is not None else 'N/A'
                  for key in ('mae', 'rmse', 'r2', 'spearman')]
        lines.append(f"| {row['model']} | " + ' | '.join(values) + ' |')
    lines += ['', '| Comparison | RMSE reduction | MAE reduction | R² increase | Spearman increase | Lower RMSE folds | Lower MAE folds |', '|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['candidate']} vs {row['reference']} | {row['oof_rmse_reduction']:+.6f} | {row['oof_mae_reduction']:+.6f} | {row['oof_r2_increase']:+.6f} | {row['oof_spearman_increase']:+.6f} | {row['folds_with_lower_rmse']}/5 | {row['folds_with_lower_mae']}/5 |")
    gain = comparisons[0]
    lines += ['', f"Initial observation相对Task-only的RMSE变化（下降为正）为 {gain['oof_rmse_reduction']:+.6f}，MAE变化为 {gain['oof_mae_reduction']:+.6f}；RMSE改善 {gain['folds_with_lower_rmse']}/5 folds，MAE改善 {gain['folds_with_lower_mae']}/5 folds。", '',
              '仅一次分组划分；改善大小及fold一致性用于判断额外信号，未执行显著性检验。拼接文本也改变query在embedding中的表示，因此差异反映整个拼接表示的增益，不能单独归因于某一观察字段。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
