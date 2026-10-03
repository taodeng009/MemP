"""Server-side batched embeddings with token validation and resumable checkpoints.

Only this script, the prepared inputs, numpy/transformers, a local tokenizer,
and the existing OpenAI-compatible embedding service are required on the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


MODEL = 'BAAI/bge-base-en-v1.5'
FORMAT_VERSION = 'task_initial_observation_v1'


def load_env(path):
    values = {}
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if line.startswith('export '):
            line = line[7:].strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_inputs(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) != 134 or len({row['task_id'] for row in rows}) != 134:
        raise ValueError('Expected 134 unique task IDs')
    for row in rows:
        text = f"Task: {row['query']}\nInitial observation: {row['initial_observation']}"
        if (row['format_version'] != FORMAT_VERSION or row['embedding_text'] != text
                or row['text_sha256'] != hashlib.sha256(text.encode('utf-8')).hexdigest()):
            raise ValueError(f"Input text/hash mismatch: {row['task_id']}")
    return rows


def validate_vector(vector):
    array = np.asarray(vector, dtype=float)
    if array.shape != (768,) or not np.isfinite(array).all():
        raise ValueError('Expected a finite 768-dimensional embedding')
    return array.tolist()


def request_batch(endpoint, api_key, texts, timeout, retries):
    body = json.dumps({'model': MODEL, 'input': texts}).encode('utf-8')
    for attempt in range(retries + 1):
        request = urllib.request.Request(endpoint, data=body, method='POST', headers={
            'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}',
        })
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode('utf-8'))
            data = sorted(result['data'], key=lambda item: int(item['index']))
            if [int(item['index']) for item in data] != list(range(len(texts))):
                raise ValueError('Embedding response indices do not match batch')
            return [validate_vector(item['embedding']) for item in data]
        except urllib.error.HTTPError as error:
            retryable = error.code == 429 or error.code >= 500
            error.close()
            if not retryable or attempt == retries:
                raise RuntimeError(f'Embedding request failed: HTTP {error.code}; checkpoint preserved') from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == retries:
                raise RuntimeError('Embedding service is unreachable or timed out; checkpoint preserved') from None
        time.sleep(min(2 ** attempt, 8))


def atomic_json(path, data):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--tokenizer-path', type=str, required=True,
                        help='Local tokenizer directory used by the deployed BGE model')
    parser.add_argument('--max-input-tokens', type=int, default=512,
                        help='Set to the deployed embedding service input token limit')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--retries', type=int, default=3)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_input_tokens < 1 or args.retries < 0:
        raise ValueError('Invalid batch size, token limit or retry count')
    rows = load_inputs(args.inputs)
    input_hash = hashlib.sha256(args.inputs.read_bytes()).hexdigest()
    if args.output.exists():
        with np.load(args.output, allow_pickle=False) as cache:
            if (str(cache['model'].item()) != MODEL
                    or str(cache['input_sha256'].item()) != input_hash
                    or list(cache['task_ids'].astype(str)) != [row['task_id'] for row in rows]
                    or list(cache['text_sha256'].astype(str)) != [row['text_sha256'] for row in rows]
                    or cache['embeddings'].shape != (134, 768)
                    or not np.isfinite(cache['embeddings']).all()):
                raise ValueError('Existing output cache does not match inputs; use a new output path')
        print('Validated existing 134 x 768 cache; no embedding calls needed.')
        return
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    token_counts = [len(tokenizer.encode(row['embedding_text'], add_special_tokens=True, truncation=False)) for row in rows]
    if max(token_counts) > args.max_input_tokens:
        raise ValueError(f'Input exceeds configured token limit: max={max(token_counts)}, limit={args.max_input_tokens}; no requests sent')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token_summary = {
        'model': MODEL, 'tokenizer_path': args.tokenizer_path, 'max_input_tokens': args.max_input_tokens,
        'min_tokens': min(token_counts), 'max_tokens': max(token_counts),
        'task_token_counts': token_counts, 'input_sha256': input_hash,
    }
    atomic_json(args.output.with_name(args.output.name + '.tokens.json'), token_summary)
    if args.validate_only:
        print(json.dumps({'tasks': len(rows), 'min_tokens': min(token_counts), 'max_tokens': max(token_counts)}))
        return
    values = load_env(args.env_file)
    def configured(key):
        return os.environ.get(key) or values.get(key)
    if configured('EMBEDDING_MODEL_NAME') != MODEL:
        raise ValueError(f'EMBEDDING_MODEL_NAME must equal {MODEL}')
    base_url, api_key = configured('EMBEDDING_MODEL_BASE_URL'), configured('EMBEDDING_MODEL_KEY')
    if not base_url or not api_key:
        raise ValueError('Missing EMBEDDING_MODEL_BASE_URL or EMBEDDING_MODEL_KEY')
    checkpoint_path = args.output.with_name(args.output.name + '.checkpoint.json')
    checkpoint = {'model': MODEL, 'input_sha256': input_hash, 'vectors': {}}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8'))
        if checkpoint['model'] != MODEL or checkpoint['input_sha256'] != input_hash:
            raise ValueError('Checkpoint inputs/model mismatch; use a new output path')
    unique_texts = {row['text_sha256']: row['embedding_text'] for row in rows}
    if not set(checkpoint['vectors']).issubset(unique_texts):
        raise ValueError('Checkpoint has unknown input hashes')
    for vector in checkpoint['vectors'].values():
        validate_vector(vector)
    pending = [key for key in unique_texts if key not in checkpoint['vectors']]
    for start in range(0, len(pending), args.batch_size):
        keys = pending[start:start + args.batch_size]
        vectors = request_batch(base_url.rstrip('/') + '/embeddings', api_key,
                                [unique_texts[key] for key in keys], args.timeout, args.retries)
        checkpoint['vectors'].update(zip(keys, vectors))
        atomic_json(checkpoint_path, checkpoint)
        print(f"Cached {len(checkpoint['vectors'])}/{len(unique_texts)} unique texts", flush=True)
    embeddings = np.asarray([checkpoint['vectors'][row['text_sha256']] for row in rows], dtype=np.float32)
    temporary = args.output.with_name(args.output.name + '.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, embeddings=embeddings, model=np.asarray(MODEL),
                            task_ids=np.asarray([row['task_id'] for row in rows]),
                            input_texts=np.asarray([row['embedding_text'] for row in rows]),
                            text_sha256=np.asarray([row['text_sha256'] for row in rows]),
                            input_sha256=np.asarray(input_hash), format_version=np.asarray(FORMAT_VERSION),
                            token_counts=np.asarray(token_counts))
    os.replace(temporary, args.output)
    print(f'Saved {len(rows)} x 768 embeddings to {args.output}')


if __name__ == '__main__':
    main()
