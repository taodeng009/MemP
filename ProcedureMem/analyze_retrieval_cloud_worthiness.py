"""Retrieval-feature logistic OOF ranking: complete cases and imputation sensitivity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

if __package__:
    from . import analyze_task_semantics_linear_probe as probe
    from .analyze_cloud_worthiness import logistic_predictions, classification_metrics
    from .analyze_top3_distance_probe import load_top3, restrict_manifest
else:
    import analyze_task_semantics_linear_probe as probe
    from analyze_cloud_worthiness import logistic_predictions, classification_metrics
    from analyze_top3_distance_probe import load_top3, restrict_manifest


def evaluate(rows, groups, gain, designs, manifest, output):
    y = (gain > 0).astype(float)
    cs = [1e-5, 1e-4, 1e-3, .01, .1, 1., 10., 100.]
    scores = {name: np.full(len(y), np.nan) for name in designs}
    fold_ids = np.zeros(len(y), int)
    tuning, fold_rows = [], []
    if sorted(i for fold in manifest for i in fold['test_indices']) != list(range(len(y))):
        raise ValueError('Invalid test coverage')
    for fold in manifest:
        train, test = np.array(fold['train_indices']), np.array(fold['test_indices'])
        if {groups[i] for i in train} & {groups[i] for i in test}:
            raise ValueError('Outer leakage')
        fold_ids[test] = fold['fold']
        for name, x in designs.items():
            loss = {c: 0. for c in cs}
            if sorted(i for inner in fold['inner_folds'] for i in inner['validation_indices']) != sorted(train.tolist()):
                raise ValueError('Invalid inner coverage')
            for inner in fold['inner_folds']:
                a, b = np.array(inner['train_indices']), np.array(inner['validation_indices'])
                if {groups[i] for i in a} & {groups[i] for i in b} or set(a) | set(b) != set(train):
                    raise ValueError('Inner leakage or partition mismatch')
                for c, p in logistic_predictions(x[a], y[a], x[b], cs).items():
                    p = np.clip(p, 1e-15, 1 - 1e-15)
                    loss[c] += float(np.sum(-y[b] * np.log(p) - (1 - y[b]) * np.log1p(-p)))
            best = min(cs, key=lambda c: (loss[c], c))
            tuning.extend({'model': name, 'fold': fold['fold'], 'C': c,
                           'inner_log_loss': loss[c] / len(train), 'selected': int(c == best)} for c in cs)
            scores[name][test] = logistic_predictions(x[train], y[train], x[test], [best])[best]
            fold_rows.append({'model': name, 'fold': fold['fold'], 'selected_C': best,
                              **classification_metrics(y[test], scores[name][test])})
    performance, ranking = [], []
    for name, score in scores.items():
        if not np.isfinite(score).all():
            raise ValueError('Incomplete OOF')
        performance.append({'model': name, **classification_metrics(y, score)})
    orders = {name: np.argsort(-s, kind='stable') for name, s in scores.items()}
    orders['oracle'] = np.argsort(-gain, kind='stable')
    for b in [10, 20, 30, 40, 50]:
        for name, order in orders.items():
            ids = order[:b]
            ranking.append({'ranking': name, 'B': b, 'precision': float(y[ids].mean()),
                            'recall': float(y[ids].sum() / y.sum()), 'captured_gain': float(gain[ids].sum())})
        ranking.append({'ranking': 'random_expected', 'B': b, 'precision': float(y.mean()),
                        'recall': b / len(y), 'captured_gain': float(b * gain.mean())})
    summary = {'task_count': len(y), 'canonical_query_groups': len(set(groups)),
               'positive_tasks': int(y.sum()), 'positive_proportion': float(y.mean()),
               'feature_dimensions': {name: x.shape[1] for name, x in designs.items()},
               'performance': performance, 'ranking': ranking, 'total_net_gain': float(gain.sum()),
               'C_grid': cs, 'protocol': 'Same outer/inner query folds, train-only scaling, pooled inner log loss, summed logistic loss + L2/(2C), unpenalized intercept, no class weighting',
               'PR_AUC_definition': 'Trapezoidal precision-recall area; average precision separately reported',
               'random_definition': 'Exact uniform subset expectation; not a random realization',
               'oracle_definition': 'Sort signed empirical net gain descending; ties preserve task-index order'}
    output.mkdir(parents=True, exist_ok=True)
    for filename, records in [('cv_summary.csv', performance), ('ranking.csv', ranking),
                              ('fold_metrics.csv', fold_rows), ('inner_tuning.csv', tuning)]:
        probe.write_csv(output / filename, records)
    probe.write_csv(output / 'task_oof_scores.csv', [
        {'task_id': row['task_id'], 'query': row['query'], 'fold': int(fold_ids[i]),
         'net_gain': float(gain[i]), 'label': int(y[i]),
         **{name + '_score': float(s[i]) for name, s in scores.items()}} for i, row in enumerate(rows)])
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (output / 'fold_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--propensities', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, _, groups, original, model = probe.load_inputs(args.features, args.embedding_cache)
    distance, complete, feature_rows = load_top3(args.runs, rows)
    with args.propensities.open(encoding='utf-8') as handle:
        records = list(csv.DictReader(handle))
    targets = {r['task_id']: r for r in records}
    if len(targets) != len(records) or set(targets) != {r['task_id'] for r in rows}:
        raise ValueError('Target IDs mismatch')
    for row in rows:
        target = targets[row['task_id']]
        if target['query'] != row['query'] or not np.isclose(float(target['p_edge']), float(row['empirical_edge_success_probability'])):
            raise ValueError('Target query or Edge probability mismatch')
        if not np.isclose(float(target['net_success_gain']), float(target['p_cloud']) - float(target['p_edge'])):
            raise ValueError('Invalid net gain')
    gain = np.array([float(targets[r['task_id']]['net_success_gain']) for r in rows])
    manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        previous = list(csv.DictReader(handle))
    if [r['task_id'] for r in previous] != [r['task_id'] for r in rows]:
        raise ValueError('CV row order mismatch')
    result = {'embedding_model': model, 'distance_representation': 'Three-run mean observed squared-L2; missing ranks imputed 0.5 only in sensitivity',
              'excluded_main_tasks': [r['task_id'] for i, r in enumerate(rows) if not complete[i]]}
    for name, indices in [('main_132', np.flatnonzero(complete)), ('sensitivity_134', np.arange(len(rows)))]:
        designs = {'ru_bd': original['ru_bd'][indices], 'top3_distances': distance[indices],
                   'embedding_top3_distances': np.column_stack([original['embedding'][indices], distance[indices]])}
        result[name] = evaluate([rows[i] for i in indices], [groups[i] for i in indices], gain[indices],
                                designs, restrict_manifest(manifest, indices), args.output_dir / name)
    paths = [args.features, args.embedding_cache, args.propensities, args.previous_cv_dir / 'fold_manifest.json',
             *[r / 'results.jsonl' for r in args.runs]]
    result['input_hashes'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    probe.write_csv(args.output_dir / 'task_distance_features.csv', feature_rows)
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    lines = ['# Retrieval-feature Cloud-worthiness feasibility', '',
             'Label: empirical net success gain > 0. Task embeddings only in the third feature set. No extra features.', '',
             'Exact old outer/inner query fold assignments retained, removing excluded tasks without regrouping in main analysis. Same Logistic C grid and pooled inner-log-loss tuning as Task-only classification. Scaling fits training rows only. Three-run mean RU/BD and distances; sensitivity imputes missing d2/d3 as 0.5, not recovered raw distances.', '']
    for name in ['main_132', 'sensitivity_134']:
        item = result[name]
        lines += [f'## {name}', '', f"{item['task_count']} tasks, {item['canonical_query_groups']} query groups; {item['positive_tasks']} positives ({item['positive_proportion']:.2%}).", '',
                  '| Input | ROC-AUC | PR-AUC trapezoidal | Average Precision |', '|---|---:|---:|---:|']
        for p in item['performance']:
            lines.append(f"| {p['model']} | {p['roc_auc']:.6f} | {p['pr_auc_trapezoidal']:.6f} | {p['average_precision']:.6f} |")
        lines += ['', '| B | Ranking | Precision | Recall | Captured Gain |', '|---|---|---:|---:|---:|']
        for r in item['ranking']:
            lines.append(f"| {r['B']} | {r['ranking']} | {r['precision']:.2%} | {r['recall']:.2%} | {r['captured_gain']:.6f} |")
    lines += ['', '## Conclusion', '',
              '三组在132-task主分析和134-task敏感性分析均未显示有效pooled OOF分类/排序收益：ROC-AUC均低于0.5；全部Top-B的Precision、Recall、Captured Gain均低于各自Random期望。Top-3距离相对RU+BD的主分析AUC略高，但仍未达随机参考；追加Task embedding未改善整体AUC。敏感性分析结论一致。', '',
              'Random为无放回均匀选择B项的精确期望；Oracle按真实signed net gain排序。Captured Gain包含负项，单位是经验概率差之和，不是观测救援任务数，也不计推理/通信成本。分类概率不直接预测收益幅度。', '',
              '这是合并五个OOF模型的ranking，不是单一部署模型。不同fold概率偏移可影响pooled排序，不能从AUC<0.5推断稳定的反向信号。单次grouped CV、三次outcome、小样本；未检验统计显著性，不能断言这些特征完全无信息。PR-AUC使用梯形积分，AP另列；正类比例是no-skill precision参考，不是有限样本PR-AUC/AP精确期望。', '',
              '完整数值、fold调参、task-level OOF scores、fold manifest和输入hash见各分析子目录及summary.json。', '']
    (args.output_dir / 'analysis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
