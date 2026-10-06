"""Instruction-summary embedding and grouped OOF Top-K capability retrieval.

Embedding mode needs only NumPy and the existing server embedding endpoint.
Offline evaluation additionally requires scikit-learn (GroupKFold).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

import numpy as np

if __package__:
    from .difficulty_batch import parse_summary
else:
    from difficulty_batch import parse_summary

HERE = Path(__file__).resolve().parent
MODEL = 'BAAI/bge-base-en-v1.5'
KS = (3, 10)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def embedding_text(record):
    """Only permit bracket-only normalization; never change source records."""
    raw = record['response_text']
    parsed = parse_summary(raw)
    normalized = False
    if parsed['status'] == 'invalid':
        repaired, count = re.subn(r'(?m)^primary_dimensions:[ \t]*([^\[\]\n]+?)[ \t]*$',
                                 r'primary_dimensions: [\1]', raw)
        candidate = parse_summary(repaired)
        if count == 1 and candidate['status'] == 'success':
            parsed, normalized = candidate, True
    if parsed['status'] != 'success':
        raise ValueError(f"Summary is not usable: {record['task_id']}")
    if not normalized and record['difficulty_summary'] != parsed['difficulty_summary']:
        raise ValueError(f"Cached summary differs from raw response: {record['task_id']}")
    return parsed['difficulty_summary'], normalized


def load_inputs(dataset, profiles):
    with dataset.open(encoding='utf-8-sig', newline='') as handle:
        tasks = list(csv.DictReader(handle))
    records = [json.loads(line) for line in profiles.read_text(encoding='utf-8').splitlines() if line.strip()]
    for name, rows in [('dataset', tasks), ('profiles', records)]:
        if len(rows) != 134 or len({row['task_id'] for row in rows}) != 134:
            raise ValueError(f'Expected exactly 134 unique task IDs in {name}')
    lookup = {r['task_id']: r for r in records}
    if set(lookup) != {t['task_id'] for t in tasks}:
        raise ValueError('Dataset/profile task sets differ')
    inputs = []
    for task in tasks:
        record = lookup[task['task_id']]
        instruction = task['task_instruction']
        canonical = re.sub(r'\s+', ' ', instruction.strip().casefold()).rstrip('.!?').rstrip()
        if not instruction.strip() or canonical != task['canonical_query'] or record['task_instruction'] != instruction:
            raise ValueError('Instruction/canonical query mismatch')
        successes = [int(task[f'success_run{i}']) for i in (1, 2, 3)]
        p = float(task['p_edge'])
        if any(s not in (0, 1) for s in successes) or not np.isfinite(p) or abs(p-sum(successes)/3) > 1e-10:
            raise ValueError('Invalid empirical Edge success probability')
        text, normalized = embedding_text(record)
        inputs.append({'task_id': task['task_id'], 'text': text,
                       'text_sha256': digest(text.encode()), 'bracket_normalized': normalized})
        task['p_edge'] = p
    # Deliberately excludes outcome labels: they are never embedding inputs.
    identity = {'model': MODEL, 'format_version': 'difficulty_summary_v1',
                'inputs': inputs, 'normalization': 'bracket-only repair if strict validation then succeeds'}
    return tasks, identity


def validate_vector(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (768,) or not np.isfinite(array).all() or np.linalg.norm(array) <= 0:
        raise ValueError('Expected a nonzero finite 768-dimensional BGE embedding')
    return array


def request_embeddings(endpoint, key, texts, timeout, retries):
    body = json.dumps({'model': MODEL, 'input': texts}, ensure_ascii=False).encode()
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=body, headers={
            'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                response = json.load(handle)
            if response.get('model', MODEL) != MODEL:
                raise ValueError('Embedding response model mismatch')
            data = sorted(response['data'], key=lambda x: int(x['index']))
            if [int(x['index']) for x in data] != list(range(len(texts))):
                raise ValueError('Embedding batch indices mismatch')
            return [validate_vector(x['embedding']).tolist() for x in data]
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            if (code != 429 and code < 500) or attempt == retries:
                raise RuntimeError(f'Embedding HTTP error {code}; checkpoint preserved') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise RuntimeError('Embedding service unreachable/timed out; checkpoint preserved') from None
        time.sleep(min(2 ** attempt, 8))


def load_embeddings(path, identity):
    metadata = json.loads(path.with_suffix('.manifest.json').read_text(encoding='utf-8'))
    if metadata['identity'] != identity or metadata['npy_sha256'] != digest(path.read_bytes()):
        raise ValueError('Embedding identity/hash mismatch; use a new output directory')
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (134, 768) or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 134 x 768 embedding cache')
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-6, rtol=0):
        raise ValueError('Expected unit-normalized embedding rows')
    return matrix.astype(np.float64)


def embed(args, identity):
    path = args.output_dir / 'difficulty_embeddings.npy'
    if path.exists():
        load_embeddings(path, identity)
        print('Validated existing 134 x 768 cache; no API calls.')
        return
    values = {}
    if args.env_file.exists():
        for raw in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            line = raw.strip()
            if line.startswith('export '):
                line = line[7:].strip()
            if line and not line.startswith('#') and '=' in line:
                name, value = line.split('=', 1)
                values[name.strip()] = value.strip().strip('\"\'')
    def configured(name):
        return os.environ.get(name) or values.get(name)
    if configured('EMBEDDING_MODEL_NAME') != MODEL:
        raise ValueError(f'EMBEDDING_MODEL_NAME must be {MODEL}')
    base, key = configured('EMBEDDING_MODEL_BASE_URL'), configured('EMBEDDING_MODEL_KEY')
    if not base or not key:
        raise ValueError('Missing EMBEDDING_MODEL_BASE_URL / EMBEDDING_MODEL_KEY')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = path.with_suffix('.checkpoint.json')
    checkpoint = {'identity': identity, 'vectors': {}}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['identity'] != identity:
            raise ValueError('Embedding checkpoint identity mismatch')
    unique = {r['text_sha256']: r['text'] for r in identity['inputs']}
    if not set(checkpoint['vectors']) <= set(unique):
        raise ValueError('Unknown embedding checkpoint inputs')
    for vector in checkpoint['vectors'].values():
        validate_vector(vector)
    pending = [h for h in unique if h not in checkpoint['vectors']]
    for start in range(0, len(pending), args.batch_size):
        hashes = pending[start:start+args.batch_size]
        vectors = request_embeddings(base.rstrip('/') + '/embeddings', key,
                                     [unique[h] for h in hashes], args.timeout, args.retries)
        checkpoint['vectors'].update(zip(hashes, vectors))
        atomic_json(checkpoint_path, checkpoint)
        print(f"Embedded {len(checkpoint['vectors'])}/{len(unique)} unique summaries", flush=True)
    matrix = np.asarray([checkpoint['vectors'][r['text_sha256']] for r in identity['inputs']], dtype=np.float64)
    matrix = (matrix / np.linalg.norm(matrix, axis=1, keepdims=True)).astype(np.float32)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as handle:
        np.save(handle, matrix, allow_pickle=False)
    atomic_json(path.with_suffix('.manifest.json'), {'identity': identity,
                'npy_sha256': digest(temporary.read_bytes()), 'shape': list(matrix.shape), 'unit_normalized': True})
    os.replace(temporary, path)
    print(f'Saved {path}; bracket-normalized inputs: {sum(r["bracket_normalized"] for r in identity["inputs"])}')


def rank_values(values):
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return ranks


def metrics(target, prediction):
    error = prediction - target
    a, b = rank_values(target), rank_values(prediction)
    return {'spearman': float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else None,
            'mae': float(np.mean(np.abs(error))), 'rmse': float(np.sqrt(np.mean(error ** 2)))}


def retrieve_oof(tasks, matrix, splits):
    n = len(tasks)
    p = np.asarray([t['p_edge'] for t in tasks])
    groups = np.asarray([t['canonical_query'] for t in tasks])
    predictions = {k: np.full(n, np.nan) for k in KS}
    fold_ids = np.full(n, -1, dtype=int)
    neighbors, folds = [], []
    ids = np.asarray([t['task_id'] for t in tasks])
    for fold, (train, test) in enumerate(splits, 1):
        train, test = np.asarray(train), np.asarray(test)
        if len(train) < max(KS) or set(groups[train]) & set(groups[test]):
            raise ValueError('Insufficient training tasks or query leakage')
        if set(train) & set(test) or set(train) | set(test) != set(range(n)) or np.any(fold_ids[test] != -1):
            raise ValueError('Invalid CV partition')
        fold_ids[test] = fold
        folds.append({'fold': fold, 'train_tasks': len(train), 'test_tasks': len(test),
                      'train_query_groups': len(set(groups[train])), 'test_query_groups': len(set(groups[test])),
                      'test_task_ids': ids[test].tolist()})
        for i in test:
            distances = np.sum((matrix[train] - matrix[i]) ** 2, axis=1)
            # Stable, label-free tie-break by task ID, independent of outcomes.
            order = np.lexsort((ids[train], distances))[:max(KS)]
            selected = train[order]
            ds = distances[order]
            sims = 1 / (1 + ds)
            for k in KS:
                predictions[k][i] = float(np.mean(sims[:k] * p[selected[:k]]))
                for rank, (j, distance, similarity) in enumerate(zip(selected[:k], ds[:k], sims[:k]), 1):
                    neighbors.append({'task_id': tasks[i]['task_id'], 'fold': fold, 'k': k, 'rank': rank,
                                      'neighbor_task_id': tasks[j]['task_id'], 'neighbor_canonical_query': tasks[j]['canonical_query'],
                                      'distance': float(distance), 'similarity': float(similarity),
                                      'neighbor_p_edge': float(p[j]), 'contribution': float(similarity * p[j] / k)})
    if np.any(fold_ids < 0) or any(not np.isfinite(v).all() for v in predictions.values()):
        raise ValueError('Not every task has exactly one OOF prediction')
    return predictions, fold_ids, neighbors, folds


def failure_curves(tasks, predictions):
    failure = 1 - np.asarray([t['p_edge'] for t in tasks])
    ids = np.asarray([t['task_id'] for t in tasks])
    total = float(failure.sum())
    curves = {}
    for k in KS:
        order = np.lexsort((ids, predictions[k]))
        curves[f'vdar_k{k}'] = np.r_[0, np.cumsum(failure[order])].tolist()
    curves['random_expectation'] = (np.arange(len(tasks)+1) * total / len(tasks)).tolist()
    curves['oracle'] = np.r_[0, np.cumsum(np.sort(failure)[::-1])].tolist()
    return {'budget': list(range(len(tasks)+1)), 'total_failure_mass': total,
            'captured_failure_mass': curves,
            'failure_capture_fraction': {name: [v/total if total > 0 else None for v in curve] for name, curve in curves.items()}}


def write_csv(path, rows):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_curve_svg(path, curves):
    # Dependency-free static scientific curve: B=0..134, fractional captured failure.
    width, height, left, top, plot_w, plot_h = 760, 470, 70, 45, 650, 340
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial" font-size="12" fill="#222">',
             '<text x="70" y="25" font-size="16">OOF failure-capture curve</text>']
    n = curves['budget'][-1]
    for tick in range(6):
        f = tick/5
        y = top + plot_h*(1-f)
        lines += [f'<path d="M {left} {y} H {left+plot_w}" stroke="#e5e5e5"/>',
                  f'<text x="{left-10}" y="{y+4}" text-anchor="end">{f:.1f}</text>']
    for b in [0, 20, 40, 60, 80, 100, 120, n]:
        x = left+plot_w*b/n
        lines.append(f'<text x="{x}" y="{top+plot_h+22}" text-anchor="middle">{b}</text>')
    lines += [f'<path d="M {left} {top} V {top+plot_h} H {left+plot_w}" fill="none" stroke="#222"/>',
              '<text x="395" y="430" text-anchor="middle">Budget B (tasks selected)</text>',
              '<text transform="translate(18,215) rotate(-90)" text-anchor="middle">Fraction of empirical failure mass captured</text>']
    for index, (name, color, label) in enumerate([('vdar_k3', '#2166ac', 'VDAR K=3'), ('vdar_k10', '#b2182b', 'VDAR K=10'),
                                                ('random_expectation', '#777777', 'Random expectation'), ('oracle', '#238b45', 'Oracle')]):
        values = curves['failure_capture_fraction'][name]
        points = ' '.join(f'{left+plot_w*b/n:.3f},{top+plot_h*(1-(v or 0)):.3f}' for b, v in zip(curves['budget'], values))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        x = 70+index*175
        lines += [f'<path d="M {x} 455 h 20" stroke="{color}" stroke-width="2"/>', f'<text x="{x+26}" y="459">{label}</text>']
    lines.append('</g></svg>')
    path.write_text('\n'.join(lines), encoding='utf-8')


def evaluate(args, tasks, identity):
    try:
        from sklearn.model_selection import GroupKFold
        import sklearn
    except ImportError:
        raise RuntimeError('Offline evaluation requires scikit-learn: pip install scikit-learn') from None
    matrix = load_embeddings(args.output_dir / 'difficulty_embeddings.npy', identity)
    groups = np.asarray([t['canonical_query'] for t in tasks])
    splits = list(GroupKFold(n_splits=5).split(matrix, groups=groups))
    predictions, fold_ids, neighbors, folds = retrieve_oof(tasks, matrix, splits)
    p = np.asarray([t['p_edge'] for t in tasks])
    rows = [{'task_id': t['task_id'], 'task_instruction': t['task_instruction'], 'canonical_query': t['canonical_query'],
             'fold': int(fold_ids[i]), 'p_edge': float(p[i]), 'failure_propensity': float(1-p[i]),
             'c_edge_k3': float(predictions[3][i]), 'c_edge_k10': float(predictions[10][i]),
             'failure_priority_k3': float(1-predictions[3][i]), 'failure_priority_k10': float(1-predictions[10][i])}
            for i, t in enumerate(tasks)]
    curves = failure_curves(tasks, predictions)
    summary = {'task_total': len(tasks), 'unique_canonical_queries': len(set(groups)), 'cv': 'GroupKFold(n_splits=5), shuffle=False',
               'sklearn_version': sklearn.__version__, 'folds': folds, 'embedding_model': MODEL,
               'embedding_input': 'difficulty_summary only; no task text, outcome or retrieval features appended',
               'bracket_normalized_task_ids': [r['task_id'] for r in identity['inputs'] if r['bracket_normalized']],
               'distance': 'squared L2 on unit-normalized BGE embeddings', 'similarity': '1/(1+d)',
               'c_edge': '(1/K) * sum(similarity_j * p_edge_j); not divided by sum(similarity)',
               'neighbor_scope': 'training-fold task records, not unique query averages',
               'tie_break': 'ascending task_id, never labels',
               'metrics_target': 'p_edge', 'metrics': {f'k{k}': metrics(p, predictions[k]) for k in KS},
               'failure_capture_definition': 'rank ascending C_edge; cumulative sum(1-p_edge), not offloading gain',
               'random_definition': 'exact uniform random selection expectation B/N * sum(1-p_edge); no CI',
               'failure_capture_curve': curves,
               'input_sha256': {str(path): digest(path.read_bytes()) for path in [args.dataset, args.profiles]},
               'embedding_sha256': digest((args.output_dir / 'difficulty_embeddings.npy').read_bytes())}
    write_csv(args.output_dir / 'oof_predictions.csv', rows)
    write_csv(args.output_dir / 'neighbors.csv', neighbors)
    atomic_json(args.output_dir / 'summary.json', summary)
    write_curve_svg(args.output_dir / 'failure_capture_curve.svg', curves)
    print(json.dumps({'task_total': len(tasks), 'metrics': summary['metrics']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--embed-only', action='store_true')
    parser.add_argument('--dataset', type=Path, default=HERE / 'outputs/edge_capability_dataset.csv')
    parser.add_argument('--profiles', type=Path, default=HERE / 'outputs/difficulty_profiles.jsonl')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs')
    parser.add_argument('--env-file', type=Path, default=HERE.parents[1] / '.env')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.retries < 0 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Invalid batch size/retries/timeout')
    tasks, identity = load_inputs(args.dataset, args.profiles)
    if args.embed_only:
        embed(args, identity)
    else:
        evaluate(args, tasks, identity)


if __name__ == '__main__':
    main()
