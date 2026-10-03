"""Prepare actual Top-1 workflow texts from three repeated ALFWorld runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FORMAT_VERSION = 'top1_memory_workflow_v1'


def prepare(run_paths):
    datasets = []
    for path in run_paths:
        records = {}
        with (path / 'results.jsonl').open(encoding='utf-8') as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row['task_id'] in records:
                    raise ValueError(f'Duplicate task ID: {path}')
                records[row['task_id']] = row
        datasets.append(records)
    if len(datasets[0]) != 134 or any(set(d) != set(datasets[0]) for d in datasets[1:]):
        raise ValueError('Expected the same 134 task IDs in all three runs')
    inputs, differing = [], []
    for task_id, reference in sorted(datasets[0].items(), key=lambda item: item[1]['task_index']):
        workflows = []
        for run, dataset in enumerate(datasets, start=1):
            row = dataset[task_id]
            if row['query'] != reference['query'] or not row['retrieved_memories']:
                raise ValueError(f'Query mismatch or missing retrieval: {task_id}')
            memory = row['retrieved_memories'][0]
            if memory['rank'] != 1 or float(memory['score']) != min(float(m['score']) for m in row['retrieved_memories']):
                raise ValueError(f'First record is not Top-1: {task_id}')
            text = memory['workflow']
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f'Empty workflow: {task_id}')
            workflows.append(text)
            inputs.append({
                'record_id': f'{task_id}::run{run}', 'task_id': task_id,
                'task_index': reference['task_index'], 'run': run, 'query': reference['query'],
                'memory_query': memory['task_name'], 'top1_distance': float(memory['score']),
                'workflow': text, 'embedding_text': text,
                'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'format_version': FORMAT_VERSION,
            })
        if len(set(workflows)) > 1:
            differing.append({'task_id': task_id, 'task_index': reference['task_index'],
                              'distinct_workflow_count': len(set(workflows))})
    return inputs, differing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs, differing = prepare(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in inputs), encoding='utf-8')
    summary = {
        'task_count': 134, 'task_run_record_count': len(inputs),
        'unique_workflow_text_count': len({row['embedding_text'] for row in inputs}),
        'top1_workflow_differs_across_runs_task_count': len(differing),
        'differing_tasks': differing,
        'representation': 'BGE encode workflow body only; mean three actual run-specific vectors per task',
        'format_version': FORMAT_VERSION,
        'min_text_characters': min(len(row['embedding_text']) for row in inputs),
        'max_text_characters': max(len(row['embedding_text']) for row in inputs),
        'input_sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'sources': {str(path): hashlib.sha256((path / 'results.jsonl').read_bytes()).hexdigest() for path in args.runs},
    }
    args.output.with_suffix('.manifest.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
