"""Extract memory-free initial environment text from three ALFWorld runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT_VERSION = 'task_initial_observation_v1'
MEMORY_MARKER = '\n\nHere are some guidelines for solving similar tasks:\n'
TASK_MARKER = '\nYour task is to: '


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def extract_initial_observation(record):
    first = record['trajectory'][0]
    if first.get('from') != 'human' or not isinstance(first.get('value'), str):
        raise ValueError(f"Initial trajectory item is not a human observation: {record['task_id']}")
    initial = first['value'].split(MEMORY_MARKER, 1)[0].strip()
    if TASK_MARKER not in initial:
        raise ValueError(f"Cannot identify initial task boundary: {record['task_id']}")
    observation, query = initial.rsplit(TASK_MARKER, 1)
    if query.strip() != record['query'].strip() or not observation.strip():
        raise ValueError(f"Initial query or observation mismatch: {record['task_id']}")
    return observation.strip()


def prepare(run_paths):
    datasets = []
    for path in run_paths:
        dataset = {}
        with (path / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record['task_id'] in dataset:
                    raise ValueError(f'Duplicate task ID: {path}')
                dataset[record['task_id']] = record
        datasets.append(dataset)
    if len(datasets[0]) != 134 or any(set(d) != set(datasets[0]) for d in datasets[1:]):
        raise ValueError('Three logs must contain the same 134 task IDs')
    inputs = []
    for task_id, reference in sorted(datasets[0].items(), key=lambda item: item[1]['task_index']):
        observations = [extract_initial_observation(dataset[task_id]) for dataset in datasets]
        if any(observation != observations[0] for observation in observations[1:]):
            raise ValueError(f'Initial observations differ across runs: {task_id}')
        if any(dataset[task_id]['query'] != reference['query'] for dataset in datasets):
            raise ValueError(f'Queries differ across runs: {task_id}')
        text = f"Task: {reference['query']}\nInitial observation: {observations[0]}"
        inputs.append({
            'task_id': task_id, 'task_index': reference['task_index'],
            'query': reference['query'], 'initial_observation': observations[0],
            'embedding_text': text, 'text_sha256': text_hash(text),
            'format_version': FORMAT_VERSION,
        })
    return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = prepare(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in inputs), encoding='utf-8')
    summary = {
        'task_count': len(inputs), 'unique_embedding_texts': len({row['embedding_text'] for row in inputs}),
        'unique_initial_observations': len({row['initial_observation'] for row in inputs}),
        'initial_observation_identical_across_runs': True,
        'extraction': 'trajectory[0] human text; remove appended memory guidelines and separate task goal',
        'format_version': FORMAT_VERSION,
        'input_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'min_text_characters': min(len(row['embedding_text']) for row in inputs),
        'max_text_characters': max(len(row['embedding_text']) for row in inputs),
        'sources': {str(path): hashlib.sha256((path / 'results.jsonl').read_bytes()).hexdigest() for path in args.runs},
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
