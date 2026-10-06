"""Five task types only; unchanged official VDAR V2 prompt and query-only input."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def select_tasks(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    selected, seen = [], set()
    for row in rows:
        task_type = row['task_id'].split('/')[2].split('-', 1)[0]
        if task_type not in seen:
            selected.append({'task_id': row['task_id'], 'task_type': task_type,
                             'task_instruction': row['task_instruction']})
            seen.add(task_type)
        if len(selected) == 5:
            return selected
    raise ValueError('Need at least five different ALFWorld task types')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=HERE / 'outputs/edge_capability_dataset.csv')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--output', type=Path, default=HERE / 'outputs/difficulty_5_tasks.csv')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--timeout', type=float, default=120)
    args = parser.parse_args()
    if args.env_file.exists():
        for line in args.env_file.read_text(encoding='utf-8-sig').splitlines():
            if line.strip() and not line.lstrip().startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip().strip('\"\''))
    tasks = select_tasks(args.input)
    system = (HERE / 'prompts/vdar_v2_system.txt').read_text(encoding='utf-8').strip()
    prompt_hash = hashlib.sha256(system.encode()).hexdigest()
    provenance = json.loads((HERE / 'prompts/vdar_v2_provenance.json').read_text(encoding='utf-8'))
    if prompt_hash != provenance['v2_prompt_sha256']:
        raise ValueError('Official V2 prompt changed')
    messages = [[{'role': 'system', 'content': system},
                 {'role': 'user', 'content': '**Query to Analyze:**\n' + t['task_instruction']}]
                for t in tasks]
    if args.dry_run:
        print(json.dumps({'tasks': tasks, 'prompt_sha256': prompt_hash,
                          'model': os.environ.get('MEMORY_BUILD_MODEL_NAME'),
                          'model_input': 'task_instruction only; exact official system/user templates'}, indent=2))
        return
    model = os.environ.get('MEMORY_BUILD_MODEL_NAME')
    key = os.environ.get('MEMORY_BUILD_API_KEY') or os.environ.get('OPENAI_API_KEY')
    base = os.environ.get('MEMORY_BUILD_API_BASE_URL') or os.environ.get('OPENAI_API_BASE') or os.environ.get('OPENAI_BASE_URL')
    if not model or not key or not base:
        raise ValueError('Missing memory-build model/key/base URL configuration')
    if args.output.exists():
        raise ValueError('Output exists; use another --output to avoid overwriting')
    extra = {'top_k': int(os.environ.get('MEMORY_BUILD_TOP_K', '1'))}
    thinking = os.environ.get('MEMORY_BUILD_ENABLE_THINKING')
    if thinking is not None:
        if thinking.strip().lower() not in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off']:
            raise ValueError('Invalid MEMORY_BUILD_ENABLE_THINKING')
        extra['enable_thinking'] = thinking.strip().lower() in ['true', '1', 'yes', 'on']
    results, audit = [], []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for task, prompt in zip(tasks, messages):
        request = {'model': model, 'messages': prompt, 'temperature': float(os.environ.get('MEMORY_BUILD_TEMPERATURE', '0')),
                   'seed': int(os.environ.get('MEMORY_BUILD_SEED', '42')), 'top_p': 1, 'max_tokens': 4096, **extra}
        req = urllib.request.Request(base.rstrip('/') + '/chat/completions',
              data=json.dumps(request, ensure_ascii=False).encode(),
              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key})
        try:
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                value = json.load(response)
        except Exception as exc:
            raise RuntimeError(f'Memory-build request failed ({type(exc).__name__}); check configured service. No summary fabricated.') from None
        text = value['choices'][0]['message']['content']
        blocks = [s.strip() for s in re.findall(r'<summary>\s*(.*?)\s*</summary>', text or '', flags=re.S)]
        if not blocks or not any(blocks):
            raise ValueError('Model response has no nonempty <summary> block')
        summary = max(blocks, key=len)
        results.append({'task_instruction': task['task_instruction'], 'difficulty_summary': summary})
        audit.append({**task, 'messages': prompt, 'response_text': text, 'usage': value.get('usage'), 'model': model})
        # Persist after every successful task; never include outcomes in model input.
        with args.output.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['task_instruction', 'difficulty_summary'])
            writer.writeheader()
            writer.writerows(results)
        args.output.with_suffix('.audit.json').write_text(json.dumps({'v2_prompt_sha256': prompt_hash, 'records': audit}, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
