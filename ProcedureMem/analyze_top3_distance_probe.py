"""Complete-case Top-3 distance Ridge probes with boundary-imputation sensitivity."""

from __future__ import annotations

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


def load_top3(run_paths, rows, threshold=0.5):
    datasets = []
    expected = {row['task_id'] for row in rows}
    for path in run_paths:
        dataset = {}
        with (path / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_id = record['task_id']
                if task_id in dataset:
                    raise ValueError(f'Duplicate task ID: {path}: {task_id}')
                scores = [float(memory['score']) for memory in record['retrieved_memories']]
                if len(scores) != record['retrieved_count'] or not 1 <= len(scores) <= 3:
                    raise ValueError(f'Unexpected retrieval count: {task_id}')
                if any(not np.isfinite(d) or d < 0 or d > threshold for d in scores):
                    raise ValueError(f'Invalid score: {task_id}')
                if scores != sorted(scores):
                    raise ValueError(f'Scores are not in nearest-neighbor order: {task_id}')
                dataset[task_id] = record, scores
        if set(dataset) != expected:
            raise ValueError(f'Task set mismatch: {path}')
        datasets.append(dataset)
    distances, complete, task_features = [], [], []
    for row in rows:
        task_id = row['task_id']
        records = [dataset[task_id][0] for dataset in datasets]
        scores = [dataset[task_id][1] for dataset in datasets]
        if any(record['query'] != row['query'] for record in records):
            raise ValueError(f'Query mismatch: {task_id}')
        if ''.join(str(int(record['reward'])) for record in records) != row['success_pattern']:
            raise ValueError(f'Outcome mismatch: {task_id}')
        full = all(len(values) == 3 for values in scores)
        # Boundary values are sensitivity imputations, not recovered raw distances.
        padded = np.asarray([values + [threshold] * (3 - len(values)) for values in scores])
        averaged = padded.mean(axis=0)
        ru = np.mean([sum(threshold - d for d in values) for values in scores])
        bd = np.mean([values[0] for values in scores])
        if abs(ru - float(row['ru'])) > 1e-12 or abs(bd - float(row['bd'])) > 1e-12:
            raise ValueError(f'Existing RU/BD values do not match logs: {task_id}')
        distances.append(averaged)
        complete.append(full)
        output = {
            'task_id': task_id, 'task_index': row['task_index'], 'query': row['query'],
            'canonical_query': probe.canonical_query(row['query']),
            'failure_propensity': float(row['failure_propensity']),
            'complete_top3_all_runs': int(full),
            'ru': ru, 'bd': bd,
            **{f'd{i+1}': float(value) for i, value in enumerate(averaged)},
        }
        for run, values in enumerate(scores, start=1):
            output[f'run{run}_retrieved_count'] = len(values)
            for index in range(3):
                output[f'run{run}_d{index+1}_observed'] = values[index] if index < len(values) else None
        task_features.append(output)
    return np.asarray(distances), np.asarray(complete, dtype=bool), task_features


def restrict_manifest(previous_manifest, indices):
    """Preserve original query fold assignment in both outer and inner CV."""
    mapping = {int(old): new for new, old in enumerate(indices)}
    def mapped(old):
        return [mapping[index] for index in old if index in mapping]
    return [{
        'fold': fold['fold'], 'train_indices': mapped(fold['train_indices']),
        'test_indices': mapped(fold['test_indices']),
        'inner_folds': [{
            'train_indices': mapped(inner['train_indices']),
            'validation_indices': mapped(inner['validation_indices']),
        } for inner in fold['inner_folds']],
    } for fold in previous_manifest]


def run_analysis(name, indices, rows, y, groups, old_designs, distances,
                 alphas, seed, previous_manifest, output_dir):
    selected_rows = [rows[i] for i in indices]
    selected_groups = [groups[i] for i in indices]
    designs = {
        'ru_bd': old_designs['ru_bd'][indices],
        'top3_distances': distances[indices],
        'embedding_top3_distances': np.column_stack([old_designs['embedding'][indices], distances[indices]]),
    }
    plan = restrict_manifest(previous_manifest, indices)
    summaries, fold_rows, tuning_rows, manifest, oof, fold_ids = probe.evaluate(
        y[indices], selected_groups, designs, alphas, seed, fold_manifest=plan,
    )
    comparisons = []
    metrics_by_model = {row['model']: row for row in summaries}
    for candidate, reference in (
        ('top3_distances', 'ru_bd'), ('embedding_top3_distances', 'ru_bd'),
        ('embedding_top3_distances', 'top3_distances'),
        ('embedding_top3_distances', 'mean_baseline'),
    ):
        candidate_folds = [row for row in fold_rows if row['model'] == candidate]
        reference_folds = [row for row in fold_rows if row['model'] == reference]
        comparisons.append({
            'candidate': candidate, 'reference': reference,
            'oof_mae_reduction': metrics_by_model[reference]['mae'] - metrics_by_model[candidate]['mae'],
            'oof_rmse_reduction': metrics_by_model[reference]['rmse'] - metrics_by_model[candidate]['rmse'],
            'oof_r2_increase': metrics_by_model[candidate]['r2'] - metrics_by_model[reference]['r2'],
            'folds_with_lower_mae': sum(c['mae'] < r['mae'] for c, r in zip(candidate_folds, reference_folds)),
            'folds_with_lower_rmse': sum(c['rmse'] < r['rmse'] for c, r in zip(candidate_folds, reference_folds)),
        })
    summary = {
        'analysis': name, 'task_count': len(indices), 'canonical_query_count': len(set(selected_groups)),
        'feature_dimensions': {model: values.shape[1] for model, values in designs.items()},
        'performance': summaries, 'comparisons': comparisons,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    probe.write_csv(output_dir / 'cv_summary.csv', summaries)
    probe.write_csv(output_dir / 'fold_metrics.csv', fold_rows)
    probe.write_csv(output_dir / 'inner_alpha_selection.csv', tuning_rows)
    probe.write_csv(output_dir / 'model_comparisons.csv', comparisons)
    probe.write_csv(output_dir / 'task_oof_predictions.csv', [{
        'task_id': row['task_id'], 'task_index': row['task_index'], 'query': row['query'],
        'canonical_query': selected_groups[i], 'fold': int(fold_ids[i]),
        'failure_propensity': float(y[indices[i]]),
        **{f'{model}_prediction': float(pred[i]) for model, pred in oof.items()},
    } for i, row in enumerate(selected_rows)])
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'fold_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return summary, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, y, groups, designs, model = probe.load_inputs(args.features, args.embedding_cache)
    previous_summary = json.loads((args.previous_cv_dir / 'summary.json').read_text(encoding='utf-8'))
    previous_manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        previous_predictions = list(csv.DictReader(handle))
    if [row['task_id'] for row in rows] != [row['task_id'] for row in previous_predictions]:
        raise ValueError('Original CV manifest order does not match current task IDs')
    distances, complete, task_features = load_top3(args.runs, rows)
    if int(complete.sum()) != 132:
        raise ValueError('Expected 132 complete cases')
    seed, alphas = previous_summary['seed'], previous_summary['alpha_grid']
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe.write_csv(args.output_dir / 'task_distance_features.csv', task_features)
    analyses = []
    manifests = []
    for name, indices in (('main_132', np.flatnonzero(complete)),
                          ('sensitivity_134', np.arange(len(rows)))):
        summary, manifest = run_analysis(name, indices, rows, y, groups, designs,
                                         distances, alphas, seed, previous_manifest, args.output_dir / name)
        analyses.append(summary)
        manifests.append(manifest)
    # Reproduces the previous RU+BD control on all 134 tasks with identical CV.
    old_control = next(row for row in previous_summary['performance'] if row['model'] == 'ru_bd')
    new_control = next(row for row in analyses[1]['performance'] if row['model'] == 'ru_bd')
    for metric in ('mae', 'rmse', 'r2', 'spearman'):
        if abs(old_control[metric] - new_control[metric]) > 1e-12:
            raise ValueError(f'Previous 134-task RU+BD control not reproduced: {metric}')
    semantic_comparisons = [next(row for row in analysis['comparisons']
                                 if row['candidate'] == 'embedding_top3_distances'
                                 and row['reference'] == 'top3_distances') for analysis in analyses]
    distance_comparisons = [analysis['comparisons'][0] for analysis in analyses]
    interpretation = {
        'embedding_improves_top3_oof_mae_and_rmse_in_both_analyses': all(
            row['oof_mae_reduction'] > 0 and row['oof_rmse_reduction'] > 0
            for row in semantic_comparisons),
        'embedding_improves_top3_in_every_fold_in_both_analyses': all(
            row['folds_with_lower_mae'] == 5 and row['folds_with_lower_rmse'] == 5
            for row in semantic_comparisons),
        'top3_vs_ru_bd_mae_direction_consistent': (
            np.sign(distance_comparisons[0]['oof_mae_reduction'])
            == np.sign(distance_comparisons[1]['oof_mae_reduction'])).item(),
    }
    inputs = [args.features, args.embedding_cache, args.previous_cv_dir / 'fold_manifest.json',
              *(path / 'results.jsonl' for path in args.runs)]
    combined = {
        'embedding_model': model, 'seed': seed, 'alpha_grid': alphas,
        'target': 'failure_propensity = three-run failure count / 3',
        'aggregation': 'Mean run-specific RU/BD and ordered d1/d2/d3; impute missing distances before averaging',
        'sensitivity_imputation': 'Missing d2/d3 set to 0.5; boundary surrogate, not recovered raw distances',
        'fold_policy': 'Reuse previous outer AND inner group assignments; restrict original folds to complete cases for main analysis',
        'excluded_or_imputed_tasks': [row for row in task_features if not row['complete_top3_all_runs']],
        'previous_134_ru_bd_control_reproduced': True,
        'sensitivity_interpretation': interpretation,
        'input_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        'analyses': analyses,
    }
    (args.output_dir / 'summary.json').write_text(json.dumps(combined, indent=2) + '\n', encoding='utf-8')
    lines = ['# Top-3 distance supplementary analysis', '',
             '目标为134-task三次运行的failure propensity。主分析保留所有run均有完整Top-3的132个task；敏感性分析将剩余2个task缺失的d2/d3逐run补为0.5，再取三次均值。', '',
             '复用之前的canonical query、外层5-fold和内层3-fold分组归属。主分析从原fold删除2个task；敏感性分析使用完全相同的原134-task folds。标准化仅在训练split内拟合；Ridge alpha由训练内CV选择；原始预测不截断，各task等权。', '',
             f'Alpha grid={alphas}；seed={seed}。敏感性分析的RU+BD控制已精确复现之前指标。', '',
             '缺失任务为task_index 8与48，query均为“put a vase in safe.”，属于同一canonical query组；主分析为77组，敏感性分析为78组。', '',
             '在完整Top-3任务上，BD=d1，RU=1.5-(d1+d2+d3)。因此Top-3允许分别使用d2/d3信息，而RU+BD只保留d1及距离总和。各模型使用标准化后的Ridge，不同表示也改变正则化几何；性能差异不能完全归因于新增信息。', '']
    for analysis, manifest in zip(analyses, manifests):
        lines += [f"## {analysis['analysis']}：{analysis['task_count']} tasks / {analysis['canonical_query_count']} groups", '',
                  '| Input | OOF MAE | OOF RMSE | OOF R² | OOF Spearman |', '|---|---:|---:|---:|---:|']
        for row in analysis['performance']:
            cells = [f"{row[key]:.6f}" if row[key] is not None else 'N/A' for key in ('mae', 'rmse', 'r2', 'spearman')]
            lines.append(f"| {row['model']} | " + ' | '.join(cells) + ' |')
        lines += ['', '| Input | Fold MAE mean ± SD | Fold RMSE mean ± SD | Fold R² mean ± SD | Fold Spearman mean ± SD |',
                  '|---|---:|---:|---:|---:|']
        for row in analysis['performance']:
            cells = [f"{row[f'fold_mean_{key}']:.6f} ± {row[f'fold_sd_{key}']:.6f}"
                     if row[f'fold_sd_{key}'] is not None else 'N/A' for key in ('mae', 'rmse', 'r2', 'spearman')]
            lines.append(f"| {row['model']} | " + ' | '.join(cells) + ' |')
        lines += ['', '| Candidate vs reference | RMSE reduction | MAE reduction | Lower RMSE folds | Lower MAE folds |',
                  '|---|---:|---:|---:|---:|']
        for row in analysis['comparisons']:
            lines.append(f"| {row['candidate']} vs {row['reference']} | {row['oof_rmse_reduction']:+.6f} | {row['oof_mae_reduction']:+.6f} | {row['folds_with_lower_rmse']}/5 | {row['folds_with_lower_mae']}/5 |")
        lines += ['', '| Fold | Train tasks/groups | Test tasks/groups |', '|---|---:|---:|']
        for fold in manifest:
            lines.append(f"| {fold['fold']} | {len(fold['train_indices'])}/{fold['train_group_count']} | {len(fold['test_indices'])}/{fold['test_group_count']} |")
        lines.append('')
    lines += ['## 结果解释', '',
              '两次分析中，单独Top-3距离和RU+BD都接近训练均值基线，OOF R²均为负；Top-3相对RU+BD的RMSE变化很小，MAE变化方向在主分析与敏感性分析间反转。因此没有明确证据表明分别输入三个距离能明显改善预测。', '',
              'Embedding+Top-3在两次分析中都优于两个检索特征模型，五fold的MAE、RMSE均改善，支持额外的较弱语义信号；OOF R²仍仅约0.025，绝对预测能力有限。', '',
              '0.5补值是阈值边界代理，真实被过滤距离大于0.5。补值不能提供其真实幅度。OOF是任务级整体指标；fold SD不是独立重复实验的置信区间。仅一次分组划分，不进行性能改善显著性检验。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'analyses': analyses, 'previous_control_reproduced': True}, indent=2))


if __name__ == '__main__':
    main()
