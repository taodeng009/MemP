"""Server BGE workflow embeddings; resume and deduplicate exact text hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .build_initial_observation_embeddings import MODEL, load_env, request_batch, atomic_json, validate_vector
    from .prepare_top1_memory_inputs import FORMAT_VERSION
else:
    from build_initial_observation_embeddings import MODEL, load_env, request_batch, atomic_json, validate_vector
    from prepare_top1_memory_inputs import FORMAT_VERSION


def load_inputs(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 402 or len({row['record_id'] for row in rows}) != 402:
        raise ValueError('Expected 402 unique task-run records')
    by_task = defaultdict(list)
    for row in rows:
        text = row['workflow']
        if (not isinstance(text, str) or not text.strip() or row['embedding_text'] != text
                or row['text_sha256'] != hashlib.sha256(text.encode('utf-8')).hexdigest()
                or row['format_version'] != FORMAT_VERSION
                or row['record_id'] != f"{row['task_id']}::run{row['run']}"):
            raise ValueError(f"Invalid memory input: {row['record_id']}")
        by_task[row['task_id']].append(row['run'])
    if len(by_task) != 134 or any(sorted(runs) != [1, 2, 3] for runs in by_task.values()):
        raise ValueError('Each of 134 tasks must have run1/run2/run3')
    return rows


def validate_cache(path, rows, input_hash):
    with np.load(path, allow_pickle=False) as cache:
        if (str(cache['model'].item()) != MODEL
                or str(cache['input_sha256'].item()) != input_hash
                or str(cache['format_version'].item()) != FORMAT_VERSION
                or list(cache['record_ids'].astype(str)) != [row['record_id'] for row in rows]
                or list(cache['task_ids'].astype(str)) != [row['task_id'] for row in rows]
                or list(cache['runs']) != [row['run'] for row in rows]
                or list(cache['input_texts'].astype(str)) != [row['embedding_text'] for row in rows]
                or list(cache['text_sha256'].astype(str)) != [row['text_sha256'] for row in rows]
                or cache['embeddings'].shape != (402, 768) or not np.isfinite(cache['embeddings']).all()):
            raise ValueError('Existing memory cache does not match inputs/model')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    args = parser.parse_args()
    if args.batch_size < 1 or args.timeout <= 0 or args.retries < 0:
        raise ValueError('Invalid batch/timeout/retries')
    rows = load_inputs(args.inputs)
    input_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    if args.output.exists():
        validate_cache(args.output, rows, input_hash)
        print('Validated existing 402 x 768 memory cache; no API calls.')
        return
    values = load_env(args.env_file)
    def configured(key):
        return os.environ.get(key) or values.get(key)
    if configured('EMBEDDING_MODEL_NAME') != MODEL:
        raise ValueError(f'EMBEDDING_MODEL_NAME must equal {MODEL}')
    base_url, api_key = configured('EMBEDDING_MODEL_BASE_URL'), configured('EMBEDDING_MODEL_KEY')
    if not base_url or not api_key:
        raise ValueError('Missing embedding service base URL or key')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output.with_name(args.output.name + '.checkpoint.json')
    checkpoint = {'model': MODEL, 'input_sha256': input_hash, 'vectors': {}}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['model'] != MODEL or checkpoint['input_sha256'] != input_hash:
            raise ValueError('Memory checkpoint model/input mismatch')
    unique = {row['text_sha256']: row['embedding_text'] for row in rows}
    if not set(checkpoint['vectors']).issubset(unique):
        raise ValueError('Checkpoint has unknown text hashes')
    for vector in checkpoint['vectors'].values():
        validate_vector(vector)
    pending = [key for key in unique if key not in checkpoint['vectors']]
    for start in range(0, len(pending), args.batch_size):
        keys = pending[start:start + args.batch_size]
        vectors = request_batch(base_url.rstrip('/') + '/embeddings', api_key,
                                [unique[key] for key in keys], args.timeout, args.retries)
        checkpoint['vectors'].update(zip(keys, vectors))
        atomic_json(checkpoint_path, checkpoint)
        print(f"Cached {len(checkpoint['vectors'])}/{len(unique)} unique workflows", flush=True)
    matrix = np.asarray([checkpoint['vectors'][row['text_sha256']] for row in rows], dtype=np.float32)
    temporary = args.output.with_name(args.output.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, embeddings=matrix, model=np.asarray(MODEL),
                            record_ids=np.asarray([row['record_id'] for row in rows]),
                            task_ids=np.asarray([row['task_id'] for row in rows]),
                            runs=np.asarray([row['run'] for row in rows]),
                            input_texts=np.asarray([row['embedding_text'] for row in rows]),
                            text_sha256=np.asarray([row['text_sha256'] for row in rows]),
                            input_sha256=np.asarray(input_hash), format_version=np.asarray(FORMAT_VERSION))
    os.replace(temporary, args.output)
    validate_cache(args.output, rows, input_hash)
    print(f'Saved 402 x 768 workflow vectors to {args.output}')


if __name__ == '__main__':
    main()
