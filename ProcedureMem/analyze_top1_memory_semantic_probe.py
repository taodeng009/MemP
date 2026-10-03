"""Compare separately encoded task and retrieved Top-1 workflow semantics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

if __package__:
    from . import analyze_task_semantics_linear_probe as probe
    from .build_top1_memory_embeddings import load_inputs, validate_cache, MODEL
else:
    import analyze_task_semantics_linear_probe as probe
    from build_top1_memory_embeddings import load_inputs, validate_cache, MODEL


def task_memory_matrix(rows, inputs, vectors):
    positions = {row['record_id']: i for i, row in enumerate(inputs)}
    return np.asarray([
        np.mean([vectors[positions[f"{row['task_id']}::run{run}"]] for run in (1, 2, 3)], axis=0)
        for row in rows
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--task-embedding-cache', type=Path, required=True)
    parser.add_argument('--memory-inputs', type=Path, required=True)
    parser.add_argument('--memory-embedding-cache', type=Path, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--validate-inputs-only', action='store_true')
    args = parser.parse_args()
    rows, y, groups, original, task_model = probe.load_inputs(args.features, args.task_embedding_cache)
    inputs = load_inputs(args.memory_inputs)
    by_task = {row['task_id']: row for row in rows}
    if set(by_task) != {row['task_id'] for row in inputs} or any(row['query'] != by_task[row['task_id']]['query'] for row in inputs):
        raise ValueError('Memory inputs do not match feature task IDs/queries')
    if task_model != MODEL:
        raise ValueError('Task embedding cache uses a different model')
    previous = json.loads((args.previous_cv_dir / 'summary.json').read_text(encoding='utf-8'))
    manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        old_predictions = list(csv.DictReader(handle))
    if [row['task_id'] for row in rows] != [row['task_id'] for row in old_predictions]:
        raise ValueError('Original CV order mismatch')
    if args.validate_inputs_only:
        print(json.dumps({'tasks': len(rows), 'canonical_query_groups': len(set(groups)),
                          'memory_run_records': len(inputs), 'unique_workflow_texts': len({row['embedding_text'] for row in inputs}),
                          'memory_cache_available': args.memory_embedding_cache.is_file()}, indent=2))
        return
    if not args.memory_embedding_cache.is_file():
        parser.error('Top-1 workflow embedding cache missing. Generate it on the server and sync the .npz first.')
    input_hash = hashlib.sha256(args.memory_inputs.read_bytes()).hexdigest()
    validate_cache(args.memory_embedding_cache, inputs, input_hash)
    with np.load(args.memory_embedding_cache, allow_pickle=False) as cache:
        vectors = np.asarray(cache['embeddings'], dtype=float)
    memory = task_memory_matrix(rows, inputs, vectors)
    task = original['embedding']
    designs = {
        'task_embedding': task, 'top1_memory_embedding': memory,
        'task_embedding_top1_memory_embedding': np.column_stack([task, memory]),
    }
    summaries, fold_rows, tuning_rows, actual_manifest, oof, fold_ids = probe.evaluate(
        y, groups, designs, previous['alpha_grid'], previous['seed'], fold_manifest=manifest)
    if not np.allclose(oof['task_embedding'], [float(row['embedding_prediction']) for row in old_predictions], atol=1e-12, rtol=0):
        raise ValueError('Task-only control does not reproduce previous OOF predictions')
    index = {row['model']: row for row in summaries}
    comparisons = []
    for candidate, reference in (
        ('top1_memory_embedding', 'task_embedding'),
        ('task_embedding_top1_memory_embedding', 'task_embedding'),
        ('task_embedding_top1_memory_embedding', 'top1_memory_embedding'),
        ('top1_memory_embedding', 'mean_baseline'),
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
    input_files = [args.features, args.task_embedding_cache, args.memory_inputs,
                   args.memory_embedding_cache, args.previous_cv_dir / 'fold_manifest.json']
    summary = {
        'task_count': len(rows), 'canonical_query_count': len(set(groups)), 'embedding_model': MODEL,
        'memory_representation': 'Mean three separately encoded actual Top-1 workflow vectors per task, without post-mean normalization',
        'feature_dimensions': {name: x.shape[1] for name, x in designs.items()},
        'input_combination': 'Concatenate task and memory vectors; never concatenate their raw texts',
        'seed': previous['seed'], 'alpha_grid': previous['alpha_grid'],
        'target': 'three-run failure count / 3',
        'folds': 'Reuse exact previous query-grouped outer 5-fold and inner 3-fold assignments',
        'scaling': 'Train-only per-column mean/std in every split',
        'prediction': 'Raw Ridge without clipping, unpenalized intercept, each task equally weighted',
        'task_only_control_reproduced': True,
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_files},
        'performance': summaries, 'comparisons': comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, records in (('cv_summary.csv', summaries), ('fold_metrics.csv', fold_rows),
                              ('inner_alpha_selection.csv', tuning_rows), ('model_comparisons.csv', comparisons)):
        probe.write_csv(args.output_dir / filename, records)
    probe.write_csv(args.output_dir / 'task_oof_predictions.csv', [{
        'task_id': row['task_id'], 'task_index': row['task_index'], 'query': row['query'],
        'canonical_query': groups[i], 'fold': int(fold_ids[i]), 'failure_propensity': float(y[i]),
        **{f'{name}_prediction': float(pred[i]) for name, pred in oof.items()},
    } for i, row in enumerate(rows)])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'fold_manifest.json').write_text(json.dumps(actual_manifest, indent=2) + '\n', encoding='utf-8')
    lines = ['# Top-1 memory semantic feasibility analysis', '',
             '134 tasks / 78 canonical query groups。目标为三次run失败比例，Top-1 memory文本取实际retrieved workflow正文，不使用memory query替代。逐run编码后取三次向量均值。', '',
             'Task与memory分别BGE编码后拼接向量；不拼接原始文本，不加入RU、BD、distance、family或其他特征。三组复用原内外层CV分组，Task-only OOF已精确复现。', '',
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
    gain = comparisons[1]
    lines += ['', f"在Task embedding上追加Top-1 memory向量后，OOF RMSE下降（正值为改善）{gain['oof_rmse_reduction']:+.6f}，MAE下降{gain['oof_mae_reduction']:+.6f}，RMSE改善{gain['folds_with_lower_rmse']}/5 folds，MAE改善{gain['folds_with_lower_mae']}/5 folds。", '',
              '仅一次grouped CV，不进行显著性检验。相同memory可能服务于不同query；分组沿用canonical task query，不要求memory文本互斥。结果衡量真实重复run的平均memory表示对平均failure propensity的关联，不估计单条workflow的因果效应。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
