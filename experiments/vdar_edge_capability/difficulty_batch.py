"""Instruction-only 134-task JSONL generation and validated resume."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.request

if __package__:
    from .difficulty_prompts import PROMPT_REGISTRY
else:
    from difficulty_prompts import PROMPT_REGISTRY

HERE = Path(__file__).resolve().parent
DIMENSIONS = {'object_localization', 'navigation', 'object_manipulation', 'state_transformation',
              'action_ordering', 'multi_object_handling', 'receptacle_interaction'}


def load_tasks(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 134 or len({r['task_id'] for r in rows}) != 134 or any(not r['task_instruction'].strip() for r in rows):
        raise ValueError('Expected 134 unique tasks with nonempty instructions')
    return [{'task_id': r['task_id'], 'task_instruction': r['task_instruction']} for r in rows]


def parse_summary(text):
    if not isinstance(text, str) or not text.strip():
        return {'status': 'empty', 'difficulty_summary': '', 'validation_error': 'Empty response'}
    match = re.fullmatch(r'\s*<summary>\s*(.*?)\s*</summary>\s*', text, re.S)
    if not match:
        return {'status': 'invalid', 'difficulty_summary': '', 'validation_error': 'Expected one summary block with no outside text'}
    summary = match.group(1).strip()
    if not summary:
        return {'status': 'empty', 'difficulty_summary': '', 'validation_error': 'Empty summary'}
    fields = re.fullmatch(r'overall_difficulty:\s*(low|medium|high)\s*\nprimary_dimensions:\s*\[([^\]\n]*)\]\s*\ndifficulty_profile:\s*(.+)', summary, re.S)
    if fields:
        dimensions = [d.strip().strip('\"\'') for d in fields.group(2).split(',') if d.strip()]
        profile = fields.group(3).strip()
        if 1 <= len(dimensions) <= 3 and len(set(dimensions)) == len(dimensions) and set(dimensions) <= DIMENSIONS and profile:
            return {'status': 'success', 'difficulty_summary': summary, 'overall_difficulty': fields.group(1),
                    'primary_dimensions': dimensions, 'difficulty_profile': profile}
    return {'status': 'invalid', 'difficulty_summary': summary, 'validation_error': 'Invalid fields, level, dimensions or profile'}


def save_records(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_existing(path, tasks, config):
    if not path.exists():
        return []
    expected = {t['task_id']: t['task_instruction'] for t in tasks}
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    if len({r['task_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate cached task IDs')
    for r in rows:
        if r['task_id'] not in expected or r['task_instruction'] != expected[r['task_id']] or r['configuration'] != config:
            raise ValueError('Cache task/model/prompt/settings mismatch')
        if any(r.get(k) != v for k, v in parse_summary(r['response_text']).items()):
            raise ValueError('Cached validation/summary mismatch')
    return rows


def summarize(rows):
    good = [r for r in rows if r['status'] == 'success']
    levels = Counter(r['overall_difficulty'] for r in good)
    dims = Counter(d for r in good for d in r['primary_dimensions'])
    repeats = Counter(r['difficulty_summary'] for r in good)
    return {'task_total': 134, 'recorded_tasks': len(rows), 'remaining_tasks': 134-len(rows),
            'successful_generation_count': len(good),
            'empty_summary_count': sum(r['status'] == 'empty' for r in rows),
            'invalid_summary_count': sum(r['status'] == 'invalid' for r in rows),
            'overall_difficulty': {k: levels[k] for k in ['low', 'medium', 'high']},
            'primary_dimensions': {k: dims[k] for k in sorted(DIMENSIONS)},
            'identical_summary_duplicate_extra_records': sum(v-1 for v in repeats.values()),
            'identical_summary_duplicate_groups': sum(v>1 for v in repeats.values()),
            'identical_summary_participating_records': sum(v for v in repeats.values() if v>1),
            'statistics_scope': 'Level/dimension/duplicate counts use valid summaries; duplicates compare exact extracted summary text'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'outputs/edge_capability_dataset.csv')
    parser.add_argument('--env-file', type=Path, default=HERE.parents[1] / '.env')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs/difficulty_profiles.jsonl')
    parser.add_argument('--prompt-version', choices=['alfworld'], default='alfworld')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--stats-only', action='store_true')
    parser.add_argument('--retry-invalid', action='store_true',
                        help='Retry recorded invalid summaries once, preserving previous responses; valid records are skipped.')
    args = parser.parse_args()
    if args.env_file.exists():
        for line in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('\"\''))
    tasks = load_tasks(args.input)
    system = PROMPT_REGISTRY['alfworld']
    settings = {'temperature': float(os.environ.get('MEMORY_BUILD_TEMPERATURE', '0')),
                'seed': int(os.environ.get('MEMORY_BUILD_SEED', '42')), 'top_p': 1, 'max_tokens': 4096,
                'top_k': int(os.environ.get('MEMORY_BUILD_TOP_K', '1'))}
    thinking = os.environ.get('MEMORY_BUILD_ENABLE_THINKING')
    if thinking is not None:
        if thinking.lower().strip() not in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']:
            raise ValueError('Invalid thinking setting')
        settings['enable_thinking'] = thinking.lower().strip() in ['true', '1', 'yes', 'on']
    if args.timeout <= 0 or not math.isfinite(settings['temperature']) or settings['temperature'] < 0 or settings['top_k'] < 1:
        raise ValueError('Invalid request settings')
    config = {'model': os.environ.get('MEMORY_BUILD_MODEL_NAME'), 'prompt_version': 'alfworld',
              'prompt_sha256': hashlib.sha256(system.encode()).hexdigest(), 'request_settings': settings}
    records = load_existing(args.output, tasks, config)
    known = {r['task_id'] for r in records}
    retry_ids = {r['task_id'] for r in records if args.retry_invalid and r['status'] == 'invalid'}
    pending = [t for t in tasks if t['task_id'] not in known or t['task_id'] in retry_ids]
    if args.dry_run or args.stats_only or not pending:
        print(json.dumps({'pending_api_calls': len(pending), **summarize(records)}, indent=2))
        return
    key = os.environ.get('MEMORY_BUILD_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base = os.environ.get('MEMORY_BUILD_API_BASE_URL') or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')
    if not config['model'] or not key or not base:
        raise ValueError('Missing memory-build model/key/base URL')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for task in pending:
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': '**Query to Analyze:**\n' + task['task_instruction']}]
            req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
                data=json.dumps({'model': config['model'], 'messages': messages, **settings}, ensure_ascii=False).encode(),
                headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as response:
                    value = json.load(response)
                text = value['choices'][0]['message']['content']
            except Exception as exc:
                raise RuntimeError(f'Memory-build request failed ({type(exc).__name__}); completed records preserved, rerun to resume.') from None
            new_record = {**task, **parse_summary(text), 'configuration': config, 'response_text': text,
                          'messages': messages, 'usage': value.get('usage')}
            if task['task_id'] in retry_ids:
                index = next(i for i, r in enumerate(records) if r['task_id'] == task['task_id'])
                previous = dict(records[index])
                history = list(previous.pop('previous_attempts', []))
                new_record['previous_attempts'] = history + [previous]
                records[index] = new_record
            else:
                records.append(new_record)
            save_records(args.output, records)
            print(f"Recorded {len(records)}/134: {new_record['status']}", flush=True)
    finally:
        stats = summarize(records)
        args.output.with_suffix('.stats.json').write_text(json.dumps(stats, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(stats, indent=2))
