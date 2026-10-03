"""Grouped nested CV Ridge probes for task-level Edge failure propensity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


MODEL_NAMES = ('mean_baseline', 'ru_bd', 'embedding', 'embedding_ru_bd')


def canonical_query(query):
    return re.sub(r'\s+', ' ', query.strip().casefold()).rstrip('.!?').rstrip()


def grouped_folds(groups, fold_count, seed):
    """Balance task counts, breaking ties randomly, without consulting labels."""
    members = defaultdict(list)
    for index, group in enumerate(groups):
        members[group].append(index)
    if len(members) < fold_count:
        raise ValueError('Too few independent query groups for CV')
    rng = np.random.default_rng(seed)
    ordered = sorted(members)
    rng.shuffle(ordered)
    ordered.sort(key=lambda group: len(members[group]), reverse=True)
    counts = np.zeros(fold_count, dtype=int)
    group_counts = np.zeros(fold_count, dtype=int)
    tests = [[] for _ in range(fold_count)]
    for group in ordered:
        eligible = np.flatnonzero(counts == counts.min())
        eligible = eligible[group_counts[eligible] == group_counts[eligible].min()]
        fold = int(rng.choice(eligible))
        tests[fold].extend(members[group])
        counts[fold] += len(members[group])
        group_counts[fold] += 1
    return [np.asarray(sorted(test), dtype=int) for test in tests]


def ridge_predictions(x_train, y_train, x_test, alphas):
    """Train-only column standardization; centered unpenalized intercept.

    The dual eigensystem is shared across the small Ridge alpha grid.
    Objective: ||y - intercept - X beta||^2 + alpha ||beta||^2.
    """
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train = (x_train - center) / scale
    test = (x_test - center) / scale
    y_center = float(y_train.mean())
    eigenvalues, eigenvectors = np.linalg.eigh(train @ train.T)
    eigenvalues = np.maximum(eigenvalues, 0)
    projected_y = eigenvectors.T @ (y_train - y_center)
    cross = test @ train.T @ eigenvectors
    return {
        alpha: y_center + cross @ (projected_y / (eigenvalues + alpha))
        for alpha in alphas
    }


def average_ranks(values):
    order = np.argsort(values, kind='stable')
    result = np.empty(len(values))
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (start + 1 + end) / 2
        start = end
    return result


def metrics(y, predictions):
    error = predictions - y
    denominator = float(np.sum((y - y.mean()) ** 2))
    a, b = average_ranks(y), average_ranks(predictions)
    rho = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None
    return {
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'r2': 1 - float(error @ error) / denominator if denominator > 0 else None,
        'spearman': rho,
    }


def load_inputs(feature_file, embedding_file):
    with feature_file.open(encoding='utf-8') as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r['task_index']))
    ids = [row['task_id'] for row in rows]
    if len(ids) != 134 or len(set(ids)) != len(ids):
        raise ValueError('Expected 134 unique task IDs')
    with np.load(embedding_file, allow_pickle=False) as cached:
        cache_ids = list(cached['task_ids'].astype(str))
        vectors = np.asarray(cached['embeddings'], dtype=float)
        model = str(cached['model'].item())
    if len(cache_ids) != len(set(cache_ids)) or set(cache_ids) != set(ids):
        raise ValueError('Embedding cache IDs do not match feature task IDs')
    lookup = {task_id: index for index, task_id in enumerate(cache_ids)}
    embedding = vectors[[lookup[task_id] for task_id in ids]]
    retrieval = np.asarray([[float(row['ru']), float(row['bd'])] for row in rows])
    y = np.asarray([float(row['failure_propensity']) for row in rows])
    if not all(np.isfinite(a).all() for a in (embedding, retrieval, y)):
        raise ValueError('Input contains missing/non-finite values')
    if any(abs(float(row['empirical_edge_success_probability']) + y[i] - 1) > 1e-12
           or abs((3 - int(row['success_count'])) / 3 - y[i]) > 1e-12
           for i, row in enumerate(rows)):
        raise ValueError('Target does not match three-run empirical propensity')
    groups = [canonical_query(row['query']) for row in rows]
    if any(not group for group in groups):
        raise ValueError('Empty canonical query')
    designs = {
        'ru_bd': retrieval, 'embedding': embedding,
        'embedding_ru_bd': np.column_stack([embedding, retrieval]),
    }
    return rows, y, groups, designs, model


def evaluate(y, groups, designs, alphas, seed, fold_manifest=None):
    all_indices = np.arange(len(y))
    model_names = ('mean_baseline', *designs)
    outer_tests = (grouped_folds(groups, 5, seed) if fold_manifest is None
                   else [np.asarray(fold['test_indices'], dtype=int) for fold in fold_manifest])
    if sorted(np.concatenate(outer_tests).tolist()) != all_indices.tolist():
        raise ValueError('Outer test folds must cover each task exactly once')
    oof = {name: np.full(len(y), np.nan) for name in model_names}
    fold_ids = np.full(len(y), -1, dtype=int)
    fold_rows, tuning_rows, manifest = [], [], []
    for fold, test in enumerate(outer_tests, start=1):
        train = np.setdiff1d(all_indices, test)
        train_groups = {groups[i] for i in train}
        test_groups = {groups[i] for i in test}
        if train_groups & test_groups:
            raise ValueError('Outer query leakage')
        if fold_manifest is None:
            inner_tests = grouped_folds([groups[i] for i in train], 3, seed + fold)
        else:
            local_index = {int(index): position for position, index in enumerate(train)}
            inner_tests = [np.asarray([local_index[index] for index in inner['validation_indices']], dtype=int)
                           for inner in fold_manifest[fold - 1]['inner_folds']]
        if sorted(np.concatenate(inner_tests).tolist()) != list(range(len(train))):
            raise ValueError('Inner validation folds must cover each training task exactly once')
        inner_manifest = []
        for held_out in inner_tests:
            inner_train = np.setdiff1d(np.arange(len(train)), held_out)
            if {groups[i] for i in train[inner_train]} & {groups[i] for i in train[held_out]}:
                raise ValueError('Inner query leakage')
            inner_manifest.append({'train_indices': train[inner_train].tolist(),
                                   'validation_indices': train[held_out].tolist()})
        manifest.append({
            'fold': fold, 'train_indices': train.tolist(), 'test_indices': test.tolist(),
            'train_group_count': len(train_groups), 'test_group_count': len(test_groups),
            'inner_folds': inner_manifest,
        })
        fold_ids[test] = fold
        oof['mean_baseline'][test] = y[train].mean()
        selected = {'mean_baseline': None}
        for name, x in designs.items():
            squared_errors = {alpha: 0.0 for alpha in alphas}
            for local_test in inner_tests:
                local_train = np.setdiff1d(np.arange(len(train)), local_test)
                predictions = ridge_predictions(x[train[local_train]], y[train[local_train]],
                                                x[train[local_test]], alphas)
                for alpha, pred in predictions.items():
                    squared_errors[alpha] += float(np.sum((pred - y[train[local_test]]) ** 2))
            best = min(alphas, key=lambda a: (squared_errors[a], -a))
            selected[name] = best
            for alpha in alphas:
                tuning_rows.append({'model': name, 'outer_fold': fold, 'alpha': alpha,
                                    'inner_oof_mse': squared_errors[alpha] / len(train),
                                    'selected': int(alpha == best)})
            oof[name][test] = ridge_predictions(x[train], y[train], x[test], [best])[best]
        for name in model_names:
            fold_rows.append({
                'model': name, 'fold': fold, 'train_tasks': len(train), 'test_tasks': len(test),
                'train_query_groups': len(train_groups), 'test_query_groups': len(test_groups),
                'selected_alpha': selected[name], **metrics(y[test], oof[name][test]),
            })
    if any(not np.isfinite(pred).all() for pred in oof.values()) or np.any(fold_ids < 1):
        raise ValueError('Incomplete out-of-fold predictions')
    summaries = []
    for name in model_names:
        row = {'model': name, **metrics(y, oof[name])}
        for metric in ('mae', 'rmse', 'r2', 'spearman'):
            values = [fold[metric] for fold in fold_rows if fold['model'] == name and fold[metric] is not None]
            row[f'fold_mean_{metric}'] = float(np.mean(values)) if values else None
            row[f'fold_sd_{metric}'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        row['prediction_min'] = float(oof[name].min())
        row['prediction_max'] = float(oof[name].max())
        row['predictions_outside_0_1'] = int(np.sum((oof[name] < 0) | (oof[name] > 1)))
        summaries.append(row)
    return summaries, fold_rows, tuning_rows, manifest, oof, fold_ids


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 1, 10, 100, 1000, 10000, 100000, 1000000])
    args = parser.parse_args()
    alphas = sorted(set(args.alphas))
    if any(not np.isfinite(alpha) or alpha <= 0 for alpha in alphas):
        raise ValueError('Ridge alphas must be finite and positive')
    rows, y, groups, designs, embedding_model = load_inputs(args.features, args.embedding_cache)
    summaries, fold_rows, tuning_rows, manifest, oof, fold_ids = evaluate(y, groups, designs, alphas, args.seed)
    index = {row['model']: row for row in summaries}
    comparisons = []
    for candidate, reference in (('embedding', 'mean_baseline'), ('embedding', 'ru_bd'),
                                 ('embedding_ru_bd', 'ru_bd'), ('embedding_ru_bd', 'embedding')):
        candidate_folds = [row for row in fold_rows if row['model'] == candidate]
        reference_folds = [row for row in fold_rows if row['model'] == reference]
        comparisons.append({
            'candidate': candidate, 'reference': reference,
            'oof_rmse_reduction': index[reference]['rmse'] - index[candidate]['rmse'],
            'oof_mae_reduction': index[reference]['mae'] - index[candidate]['mae'],
            'oof_r2_increase': index[candidate]['r2'] - index[reference]['r2'],
            'folds_with_lower_rmse': sum(c['rmse'] < r['rmse'] for c, r in zip(candidate_folds, reference_folds)),
            'folds_with_lower_mae': sum(c['mae'] < r['mae'] for c, r in zip(candidate_folds, reference_folds)),
        })
    summary = {
        'task_count': len(rows), 'canonical_query_count': len(set(groups)),
        'raw_unique_query_count': len({row['query'] for row in rows}),
        'embedding_model': embedding_model, 'feature_dimensions': {name: x.shape[1] for name, x in designs.items()},
        'target': 'failure_propensity = 1 - empirical_edge_success_probability',
        'canonicalization': 'Unicode casefold, trim/collapse whitespace, strip terminal . ! ?; preserve query contents',
        'outer_cv': '5 grouped folds, greedily balance task counts; seed-based ties; no target/family stratification',
        'inner_cv': '3 grouped folds within each outer train; select alpha by pooled inner validation MSE',
        'seed': args.seed, 'alpha_grid': alphas,
        'standardization': 'Per-column population mean/std fitted within each training split only',
        'prediction': 'Raw linear outputs, no clipping; intercept not penalized',
        'weighting': 'Each task receives equal weight, including tasks sharing canonical query',
        'input_files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.features, args.embedding_cache)},
        'performance': summaries, 'comparisons': comparisons,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'cv_summary.csv', summaries)
    write_csv(args.output_dir / 'fold_metrics.csv', fold_rows)
    write_csv(args.output_dir / 'inner_alpha_selection.csv', tuning_rows)
    write_csv(args.output_dir / 'model_comparisons.csv', comparisons)
    task_rows = [{
        'task_id': row['task_id'], 'task_index': row['task_index'], 'query': row['query'],
        'canonical_query': groups[i], 'fold': int(fold_ids[i]), 'failure_propensity': float(y[i]),
        **{f'{name}_prediction': float(oof[name][i]) for name in MODEL_NAMES},
    } for i, row in enumerate(rows)]
    write_csv(args.output_dir / 'task_oof_predictions.csv', task_rows)
    (args.output_dir / 'fold_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    lines = ['# Task semantics feasibility: grouped 5-fold Ridge linear probe', '',
             f'134 tasks；{len(set(groups))} canonical query groups；cached {embedding_model} embeddings。', '',
             '目标为三次运行的失败比例。RU、BD使用三次均值；embedding按task ID读取既有缓存。', '',
             f'外层grouped 5-fold，seed={args.seed}，按组大小平衡任务数，不使用目标或task-family分层。同一canonical query不跨训练/测试。内层grouped 3-fold只在外层训练数据选择alpha。标准化在每个训练split内拟合，原始预测不截断。', '',
             f'Alpha grid: {alphas}。常数基线为外层训练fold目标均值。每个task等权。', '',
             '| Model | Dimensions | OOF MAE ↓ | OOF RMSE ↓ | OOF R² ↑ | OOF Spearman ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summaries:
        cells = [f'{row[key]:.4f}' if row[key] is not None else 'N/A' for key in ('mae', 'rmse', 'r2', 'spearman')]
        lines.append(f"| {row['model']} | {designs[row['model']].shape[1] if row['model'] in designs else 0} | " + ' | '.join(cells) + ' |')
    lines += ['', '| Model | Fold MAE mean ± SD | Fold RMSE mean ± SD | Fold R² mean ± SD | Fold Spearman mean ± SD |',
              '|---|---:|---:|---:|---:|']
    for row in summaries:
        cells = [f"{row[f'fold_mean_{key}']:.4f} ± {row[f'fold_sd_{key}']:.4f}"
                 if row[f'fold_sd_{key}'] is not None else 'N/A'
                 for key in ('mae', 'rmse', 'r2', 'spearman')]
        lines.append(f"| {row['model']} | " + ' | '.join(cells) + ' |')
    lines += ['', '| Fold | Train tasks/groups | Test tasks/groups |', '|---|---:|---:|']
    for fold in manifest:
        lines.append(f"| {fold['fold']} | {len(fold['train_indices'])}/{fold['train_group_count']} | {len(fold['test_indices'])}/{fold['test_group_count']} |")
    lines += ['', '| Candidate vs reference | OOF RMSE reduction | OOF MAE reduction | R² increase | Lower RMSE folds | Lower MAE folds |',
              '|---|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['candidate']} vs {row['reference']} | {row['oof_rmse_reduction']:+.4f} | {row['oof_mae_reduction']:+.4f} | {row['oof_r2_increase']:+.4f} | {row['folds_with_lower_rmse']}/5 | {row['folds_with_lower_mae']}/5 |")
    semantic_gain = comparisons[1]
    retrieval_gain = comparisons[3]
    lines += ['',
              f"Embedding相对RU+BD的OOF RMSE下降 {semantic_gain['oof_rmse_reduction']:.6f}，五fold中 {semantic_gain['folds_with_lower_rmse']} 个RMSE改善。Embedding的OOF R²={index['embedding']['r2']:.6f}；这项结果用于判断额外语义信号的方向和大小，不构成显著性检验。", '',
              f"在embedding上追加RU+BD的OOF RMSE下降 {retrieval_gain['oof_rmse_reduction']:.6f}，MAE下降 {retrieval_gain['oof_mae_reduction']:.6f}。", '']
    lines += ['', 'OOF指标按134个任务整体计算；fold均值/SD描述五个fold的波动，不能视为独立重复实验的置信区间。常数基线的不同fold均值会使pooled Spearman非零。', '',
              '目标仅来自三次outcome，样本规模小，grouped CV允许不同但语义相近的query跨fold；结果检验未见canonical query上的线性预测信号，不代表对全新task family的泛化或真实offloading收益。未加入family、cluster、Hit、BD gain或复杂模型。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
