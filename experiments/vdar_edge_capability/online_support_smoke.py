"""Twelve outcome-blind, manually selected online-memory support cases.

No embedding/retrieval/capability evaluation. Labels stay in a separate file
and are read for display only after all twelve generation records exist.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import urllib.request

if __package__:
    from .difficulty_batch import build_messages, load_existing, parse_summary, save_records, SMOKE_FAMILIES
    from .difficulty_prompts import PROMPT_REGISTRY
else:
    from difficulty_batch import build_messages, load_existing, parse_summary, save_records, SMOKE_FAMILIES
    from difficulty_prompts import PROMPT_REGISTRY

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / ('ProcedureMem/Alfworld/results/paired/'
    'online_construction_valid_unseen_seed42_n134_b2_i20_c5_mbQwen3.6-27B_agentbi1_run2/'
    'online_construction_fifo_shortest_first/results.jsonl')
# Chosen after inspecting ONLY instructions and logged workflow bodies.
# "good" means stronger applicable procedural support, not guaranteed quality.
CASES = (
    ('pick_and_place_simple', 'good', 128, 'put a soapbottle in toilet.',
     'Two memories directly cover soapbottle-to-toilet placement; a third is mismatched.'),
    ('pick_and_place_simple', 'bad', 45, 'put a soapbottle in toilet.',
     'Only clean-soapbar-to-cabinet procedure: wrong object, transformation and destination.'),
    ('pick_cool_then_place_in_recep', 'good', 121, 'cool some bread and put it in countertop.',
     'First memory covers bread cooling with a fridge and countertop placement; other memories concern cleaning.'),
    ('pick_cool_then_place_in_recep', 'bad', 43, 'cool some bread and put it in countertop.',
     'Only knife cleaning at sinkbasin; cooling procedure is missing.'),
    ('look_at_obj_in_light', 'good', 122, 'examine the cd with the desklamp.',
     'CD/desklamp and related lamp-examination procedures; stronger support but command-level correctness is not certified.'),
    ('look_at_obj_in_light', 'bad', 13, 'look at cd under the desklamp.',
     'No retrieved memory, hence no procedural support.'),
    ('pick_clean_then_place_in_recep', 'good', 131, 'put a clean spatula in drawer.',
     'First memory covers spatula cleaning and drawer placement; others transfer cleaning procedures.'),
    ('pick_clean_then_place_in_recep', 'bad', 129, 'put a clean mug in coffeemachine.',
     'Mug heating rather than cleaning; remaining memories only place a mug on a desk.'),
    ('pick_heat_then_place_in_recep', 'good', 94, 'put a hot cup in cabinet.',
     'Partial stronger support: one memory covers microwave heating of a mug; another covers opening a cabinet. No exact full-task match.'),
    ('pick_heat_then_place_in_recep', 'bad', 68, 'put a hot cup in cabinet.',
     'Only mug-to-desk placement; heating and cabinet placement are absent.'),
    ('pick_two_obj_and_place', 'good', 117, 'put two peppershaker in drawer.',
     'First memory explicitly retrieves and places the first and second peppershakers in the same drawer.'),
    ('pick_two_obj_and_place', 'bad', 35, 'find two peppershaker and put them in drawer.',
     'Both memories describe one saltshaker; two-object handling is not covered.'),
)


def readable(path):
    # The supplied online directory exceeds classic Windows MAX_PATH locally.
    return Path('\\\\?\\' + str(path.resolve())) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def select_inputs(source):
    # Strip all outcomes, actions, observations and retrieval scores BEFORE selection.
    fields = ('task_id', 'task_index', 'query', 'interval_id', 'policy', 'condition',
              'available_memory_count', 'retrieved_count', 'retrieved_memory_ids')
    candidates = []
    for line in readable(source).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        row = {k: raw[k] for k in fields}
        row['memories'] = [{'rank': m['rank'], 'workflow': m['workflow']} for m in raw['retrieved_memories']]
        candidates.append(row)
    if len(candidates) != 134 or len({r['task_id'] for r in candidates}) != 134:
        raise ValueError('Expected 134 distinct online task records')
    indexed = {r['task_index']: r for r in candidates}
    if len(indexed) != 134:
        raise ValueError('Duplicate online task indices')
    selected = []
    for family, group, index, instruction, reason in CASES:
        row = indexed[index]
        if (row['query'] != instruction or row['task_id'].split('/')[2].split('-', 1)[0] != family
                or row['condition'] != 'online_construction_fifo_shortest_first'):
            raise ValueError('Selection does not match the reviewed source run')
        memories = row['memories']
        if row['retrieved_count'] != len(memories) or len(row['retrieved_memory_ids']) != len(memories):
            raise ValueError('Online retrieval count/IDs mismatch')
        for rank, m in enumerate(memories, 1):
            if m['rank'] != rank or not isinstance(m['workflow'], str) or not m['workflow'].strip():
                raise ValueError('Invalid logged memory body/rank')
        body = '\n\n'.join(f"Memory {rank}:\n{m['workflow']}" for rank, m in enumerate(memories, 1))
        selected.append({'task_id': row['task_id'], 'task_index': index, 'task_family': family,
                         'support_group': group, 'selection_reason': reason, 'task_instruction': instruction,
                         'memory_state': {'interval_id': row['interval_id'], 'policy': row['policy'],
                                          'available_memory_count': row['available_memory_count'],
                                          'retrieved_count': len(memories), 'retrieved_memory_ids': row['retrieved_memory_ids']},
                         'retrieved_memories': body or '(No procedural memories were retrieved.)'})
    return selected


def prepare(source, output_dir, tasks):
    manifest = {'source_log': str(source), 'selection_uses_outcomes': False,
                'support_definition': 'Manual relative procedural applicability, not agent success or certified memory correctness.',
                'tasks': [{**t, 'messages': build_messages(t, 'alfworld_memory')} for t in tasks]}
    path = output_dir / 'inputs.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Prepared inputs changed; use a new output directory')
    write_json(path, manifest)  # Freeze selection and exact LLM inputs first.
    # Only now reopen labels; these are never read by score().
    selected_ids = {t['task_id'] for t in tasks}
    labels = {}
    for line in readable(source).read_text(encoding='utf-8').splitlines():
        if line.strip():
            raw = json.loads(line)
            if raw['task_id'] in selected_ids:
                if type(raw['reward']) is not bool:
                    raise ValueError('Expected boolean final outcome')
                labels[raw['task_id']] = 'success' if raw['reward'] else 'failure'
    value = {'selection_task_ids': [t['task_id'] for t in tasks], 'labels': labels}
    label_path = output_dir / 'display_labels.json'
    if label_path.exists() and json.loads(label_path.read_text(encoding='utf-8')) != value:
        raise ValueError('Display labels changed; use a new output directory')
    write_json(label_path, value)
    print(json.dumps([{'task_family': t['task_family'], 'support_group': t['support_group'],
                       'task_index': t['task_index'], 'task_instruction': t['task_instruction'],
                       'interval': t['memory_state']['interval_id'], 'retrieved_count': t['memory_state']['retrieved_count']}
                      for t in tasks], ensure_ascii=False, indent=2), flush=True)


def score(args, tasks):
    if args.env_file.exists():
        for line in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                k, value = line.split('=', 1)
                os.environ.setdefault(k.strip(), value.strip().strip('\"\''))
    settings = {'temperature': float(os.environ.get('MEMORY_BUILD_TEMPERATURE', '0')),
                'seed': int(os.environ.get('MEMORY_BUILD_SEED', '42')), 'top_p': 1, 'max_tokens': 4096,
                'top_k': int(os.environ.get('MEMORY_BUILD_TOP_K', '1'))}
    thinking = os.environ.get('MEMORY_BUILD_ENABLE_THINKING')
    if thinking is not None:
        if thinking.lower().strip() not in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']:
            raise ValueError('Invalid thinking setting')
        settings['enable_thinking'] = thinking.lower().strip() in ['true', '1', 'yes', 'on']
    if not math.isfinite(settings['temperature']) or settings['temperature'] < 0 or settings['top_k'] < 1:
        raise ValueError('Invalid generation settings')
    config = {'model': os.environ.get('MEMORY_BUILD_MODEL_NAME'), 'prompt_version': 'alfworld_memory',
              'prompt_sha256': hashlib.sha256(PROMPT_REGISTRY['alfworld_memory'].encode()).hexdigest(), 'request_settings': settings}
    path = args.output_dir / 'summaries.jsonl'
    records = load_existing(path, tasks, config)
    known = {r['task_id'] for r in records}
    pending = [t for t in tasks if t['task_id'] not in known]
    if not pending:
        return
    key = os.environ.get('MEMORY_BUILD_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base = os.environ.get('MEMORY_BUILD_API_BASE_URL') or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')
    if not config['model'] or not key or not base:
        raise ValueError('Missing memory-build model/key/base URL')
    for task in pending:
        messages = build_messages(task, 'alfworld_memory')
        req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
            data=json.dumps({'model': config['model'], 'messages': messages, **settings}, ensure_ascii=False).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as handle:
                value = json.load(handle)
            text = value['choices'][0]['message']['content']
        except Exception as exc:
            raise RuntimeError(f'Request failed ({type(exc).__name__}); resume preserves completed records. No outcome merge performed.') from None
        records.append({**task, **parse_summary(text), 'configuration': config, 'response_text': text,
                        'messages': messages, 'usage': value.get('usage')})
        save_records(path, records)
        print(f"Generated {len(records)}/12: {task['task_family']} {task['support_group']} {records[-1]['status']}", flush=True)


def display(output_dir, tasks):
    records = [json.loads(line) for line in (output_dir / 'summaries.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {r['task_id']: r for r in records}
    if len(records) != 12 or len(by_id) != 12 or set(by_id) != {t['task_id'] for t in tasks}:
        raise ValueError('All 12 responses are required before outcome display')
    for task in tasks:
        r = by_id[task['task_id']]
        if any(r.get(k) != value for k, value in task.items()) or r['messages'] != build_messages(task, 'alfworld_memory'):
            raise ValueError('Display inputs mismatch')
    labels = json.loads((output_dir / 'display_labels.json').read_text(encoding='utf-8'))
    if labels['selection_task_ids'] != [t['task_id'] for t in tasks] or set(labels['labels']) != set(by_id):
        raise ValueError('Display label alignment mismatch')
    combined = []
    for task in tasks:
        r = by_id[task['task_id']]
        row = {**task, 'difficulty_summary': r['difficulty_summary'] or r['response_text'],
               'validation_status': r['status'], 'actual_outcome': labels['labels'][task['task_id']]}
        combined.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
    write_json(output_dir / 'display_results.json', {'cases': combined, 'validation_counts': dict(Counter(r['status'] for r in records)),
               'interpretation': 'Qualitative support smoke test only, not a success predictor or statistical test.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-log', type=Path, default=SOURCE)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'outputs/online_memory_support_smoke_run2')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('Timeout must be positive and finite')
    tasks = select_inputs(args.source_log)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare(args.source_log, args.output_dir, tasks)
    if not args.prepare_only:
        score(args, tasks)
        display(args.output_dir, tasks)


if __name__ == '__main__':
    main()
