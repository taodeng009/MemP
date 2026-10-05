"""Query-grouped task-only logistic classification and budgeted OOF ranking."""
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


def sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def logistic_predictions(train_x, y, test_x, cs):
    mean, std = train_x.mean(0), train_x.std(0)
    std[std < 1e-12] = 1
    x, xt = (train_x - mean) / std, (test_x - mean) / std
    eig, u = np.linalg.eigh(x @ x.T)
    keep = eig > max(1e-10, eig.max() * 1e-12)
    root, u = np.sqrt(eig[keep]), u[:, keep]
    a = np.column_stack([np.ones(len(y)), u * root])
    at = np.column_stack([np.ones(len(test_x)), (xt @ x.T @ u) / root])
    predictions = {}
    for c in cs:
        penalty = np.full(a.shape[1], 1 / c)
        penalty[0] = 0
        w = np.zeros(a.shape[1])
        w[0] = np.log(y.mean() / (1 - y.mean()))
        def objective(w):
            z = a @ w
            return np.sum(np.logaddexp(0, z) - y * z) + .5 * np.sum(penalty * w * w)
        for iteration in range(100):
            p = sigmoid(a @ w)
            gradient = a.T @ (p - y) + penalty * w
            if np.max(np.abs(gradient)) < 1e-7:
                break
            hessian = (a.T * (p * (1 - p))) @ a + np.diag(penalty)
            step = np.linalg.solve(hessian, gradient)
            rate, loss = 1., objective(w)
            while objective(w - rate * step) > loss - 1e-4 * rate * gradient.dot(step):
                rate *= .5
                if rate < 1e-12:
                    raise ValueError('Logistic line search failed')
            w -= rate * step
        else:
            raise ValueError('Logistic did not converge')
        predictions[c] = sigmoid(at @ w)
    return predictions


def classification_metrics(y, scores):
    positive, negative = scores[y == 1], scores[y == 0]
    auc = np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative))
    order = np.argsort(-scores, kind='stable')
    labels, s = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(labels)[ends]
    recall, precision = tp / y.sum(), tp / (ends + 1)
    rec, prec = np.r_[0., recall], np.r_[1., precision]
    return {'roc_auc': float(auc), 'pr_auc_trapezoidal': float(np.sum(np.diff(rec) * (prec[:-1] + prec[1:]) / 2)),
            'average_precision': float(np.sum(np.diff(rec) * prec[1:]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--embedding-cache', type=Path, required=True)
    parser.add_argument('--propensities', type=Path, required=True)
    parser.add_argument('--previous-cv-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    rows, failure, groups, designs, model = probe.load_inputs(args.features, args.embedding_cache)
    with args.propensities.open(encoding='utf-8') as handle:
        target_rows = list(csv.DictReader(handle))
    targets = {r['task_id']: r for r in target_rows}
    if len(targets) != len(target_rows) or set(targets) != {r['task_id'] for r in rows}:
        raise ValueError('Target ID mismatch')
    for i, r in enumerate(rows):
        t = targets[r['task_id']]
        if t['query'] != r['query'] or not np.isclose(1 - float(t['p_edge']), failure[i]):
            raise ValueError('Query or Edge probability mismatch')
        if not np.isclose(float(t['net_success_gain']), float(t['p_cloud']) - float(t['p_edge'])):
            raise ValueError('Net gain mismatch')
    gain = np.array([float(targets[r['task_id']]['net_success_gain']) for r in rows])
    y = (gain > 0).astype(float)
    x = designs['embedding']
    manifest = json.loads((args.previous_cv_dir / 'fold_manifest.json').read_text(encoding='utf-8'))
    with (args.previous_cv_dir / 'task_oof_predictions.csv').open(encoding='utf-8') as handle:
        old = list(csv.DictReader(handle))
    if [r['task_id'] for r in old] != [r['task_id'] for r in rows]:
        raise ValueError('CV task order mismatch')
    cs = [.00001, .0001, .001, .01, .1, 1., 10., 100.]
    oof, baseline, fold_ids = np.full(len(y), np.nan), np.full(len(y), np.nan), np.zeros(len(y), int)
    tuning, folds = [], []
    for fold in manifest:
        train, test = np.array(fold['train_indices']), np.array(fold['test_indices'])
        if {groups[i] for i in train} & {groups[i] for i in test}:
            raise ValueError('Outer query leakage')
        losses = {c: 0. for c in cs}
        count = 0
        for inner in fold['inner_folds']:
            a, b = np.array(inner['train_indices']), np.array(inner['validation_indices'])
            if {groups[i] for i in a} & {groups[i] for i in b} or not set(a) | set(b) <= set(train):
                raise ValueError('Inner query leakage')
            count += len(b)
            for c, p in logistic_predictions(x[a], y[a], x[b], cs).items():
                p = np.clip(p, 1e-15, 1 - 1e-15)
                losses[c] += float(np.sum(-y[b] * np.log(p) - (1 - y[b]) * np.log1p(-p)))
        best = min(cs, key=lambda c: (losses[c], c))
        tuning.extend({'fold': fold['fold'], 'C': c, 'inner_log_loss': losses[c] / count,
                       'selected': int(c == best)} for c in cs)
        oof[test] = logistic_predictions(x[train], y[train], x[test], [best])[best]
        baseline[test], fold_ids[test] = y[train].mean(), fold['fold']
        folds.append({'fold': fold['fold'], 'selected_C': best, 'train_tasks': len(train),
                      'test_tasks': len(test), **classification_metrics(y[test], oof[test])})
    if not np.isfinite(oof).all() or np.any(fold_ids == 0):
        raise ValueError('Incomplete OOF scores')
    order = np.argsort(-oof, kind='stable')
    oracle = np.argsort(-gain, kind='stable')
    ranking = []
    for b in [10, 20, 30, 40, 50]:
        for name, indices in [('logistic_oof', order[:b]), ('oracle', oracle[:b])]:
            ranking.append({'ranking': name, 'B': b, 'precision': float(y[indices].mean()),
                            'recall': float(y[indices].sum() / y.sum()), 'captured_gain': float(gain[indices].sum())})
        ranking.append({'ranking': 'random_expected', 'B': b, 'precision': float(y.mean()),
                        'recall': b / len(y), 'captured_gain': float(b * gain.mean())})
    summary = {'task_count': len(y), 'query_groups': len(set(groups)), 'positive_tasks': int(y.sum()),
               'label': 'net_success_gain > 0', 'embedding_model': model, 'features': 'Task embedding only (768)',
               'C_grid': cs, 'regularization': 'sum logistic loss + ||w||²/(2C), intercept unpenalized',
               'protocol': 'Exact prior grouped outer 5-fold and inner 3-fold; train-only standardization; pooled inner log loss; no class weighting',
               'classification': classification_metrics(y, oof),
               'training_prevalence_oof': classification_metrics(y, baseline),
               'random_population_reference': {'roc_auc': .5, 'average_precision': float(y.mean())},
               'ranking': ranking, 'total_positive_gain': float(gain[gain > 0].sum()),
               'total_net_gain': float(gain.sum()),
               'fold_metrics': folds,
               'input_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.features, args.embedding_cache, args.propensities, args.previous_cv_dir / 'fold_manifest.json']}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    probe.write_csv(args.output_dir / 'ranking.csv', ranking)
    probe.write_csv(args.output_dir / 'fold_metrics.csv', folds)
    probe.write_csv(args.output_dir / 'inner_tuning.csv', tuning)
    ranks = np.empty(len(y), int)
    ranks[order] = np.arange(1, len(y) + 1)
    probe.write_csv(args.output_dir / 'task_oof_scores.csv', [
        {'task_id': r['task_id'], 'task_index': r['task_index'], 'query': r['query'],
         'fold': int(fold_ids[i]), 'label': int(y[i]), 'net_gain': float(gain[i]),
         'score': float(oof[i]), 'oof_rank': int(ranks[i])} for i, r in enumerate(rows)])
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    (args.output_dir / 'fold_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
