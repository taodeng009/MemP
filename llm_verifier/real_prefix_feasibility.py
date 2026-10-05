"""Prepare 30 real records, score label-free prefixes, then evaluate completed scores."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import time

from alfworld_capability import prompt_for, official_score, load_env, save, CRITERION

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = Path(__file__).parent / 'results/real_prefix_feasibility_seed42'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_three(record):
    trajectory = record['trajectory']
    first = trajectory[0]
    if first['from'] != 'human':
        raise ValueError('Initial item must be environment text')
    initial = first['value'].split('\n\nHere are some guidelines for solving similar tasks:\n', 1)[0].strip()
    observation, task = initial.rsplit('\nYour task is to: ', 1)
    if task.strip() != record['query'].strip() or not observation.strip():
        raise ValueError('Initial task/observation mismatch')
    prefix = []
    for j in range(3):
        agent, env = trajectory[2*j+1:2*j+3]
        if agent['from'] != 'gpt' or env['from'] != 'human' or not env['value'].startswith('Observation: '):
            raise ValueError('Expected action then resulting environment observation')
        action = agent['value'].rsplit('Action:', 1)[-1].strip()
        if 'Action:' not in agent['value'] or action != record['actions'][j]:
            raise ValueError('Logged action inconsistent')
        prefix.append({'action': action, 'observation': env['value'][len('Observation: '):]})
    return observation.strip(), prefix


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    output = args.directory / 'prepared'
    if output.exists():
        raise ValueError('Prepared directory exists; use a new --directory')
    datasets, sources = [], []
    for i in (1, 2, 3):
        path = ROOT / f'ProcedureMem/Alfworld/results/paired/valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run{i}/memory/results.jsonl'
        records = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
        if len(records) != 134 or len({r['task_id'] for r in records}) != 134:
            raise ValueError('Expected 134 unique tasks per run')
        if any(not isinstance(r['reward'], bool) or r['model'] != 'Qwen/Qwen3-4B-Instruct-2507' or r['condition'] != 'memory' for r in records):
            raise ValueError('Invalid outcome/model/condition')
        datasets.append({r['task_id']: r for r in records})
        sources.append({'run_id': f'run{i}', 'path': str(path), 'sha256': digest(path)})
    if any(set(d) != set(datasets[0]) for d in datasets):
        raise ValueError('Run task sets differ')
    pools = {True: [], False: []}
    for i, dataset in enumerate(datasets, 1):
        for task_id, record in sorted(dataset.items()):
            pools[record['reward']].append((f'run{i}', record))
    rng = random.Random(args.seed)
    selected = rng.sample(pools[True], 15) + rng.sample(pools[False], 15)
    rng.shuffle(selected)
    inputs, labels = [], []
    for i, (run, record) in enumerate(selected, 1):
        sample = f'sample_{i:02d}'
        initial, steps = first_three(record)
        labels.append({'sample_id': sample, 'task_id': record['task_id'], 'run_id': run,
                       'final_success': int(record['reward'])})
        for length in (0, 1, 2, 3):
            inputs.append({'sample_id': sample, 'L': length, 'task': record['query'],
                           'initial_observation': initial, 'prefix': steps[:length]})
    output.mkdir(parents=True)
    save(output / 'scoring_inputs.json', inputs)
    save(output / 'evaluation_labels.json', labels)
    write_csv(output / 'sample_manifest.csv', labels)
    save(output / 'sampling_audit.json', {'seed': args.seed, 'sample_unit': '(task_id, run_id)',
         'pool_success': len(pools[True]), 'pool_failure': len(pools[False]),
         'sample_success': 15, 'sample_failure': 15, 'sources': sources,
         'unique_sampled_task_ids': len({r['task_id'] for r in labels}),
         'scoring_input_sha256': digest(output / 'scoring_inputs.json'),
         'label_sha256': digest(output / 'evaluation_labels.json'),
         'prompt_adapter_sha256': digest(Path(__file__).with_name('alfworld_capability.py'))})
    print(f'Prepared 30 records, 120 prefix inputs. Labels kept separate: {output}')


def load_inputs(path):
    rows = json.loads(path.read_text(encoding='utf-8'))
    if len(rows) != 120 or len({(r['sample_id'], r['L']) for r in rows}) != 120:
        raise ValueError('Expected 120 unique sample/L inputs')
    by_sample = {}
    for row in rows:
        if set(row) != {'sample_id', 'L', 'task', 'initial_observation', 'prefix'} or row['L'] not in (0, 1, 2, 3) or len(row['prefix']) != row['L']:
            raise ValueError('Unexpected input fields/length')
        if any(set(s) != {'action', 'observation'} for s in row['prefix']):
            raise ValueError('Unexpected step fields')
        by_sample.setdefault(row['sample_id'], {})[row['L']] = row
    if len(by_sample) != 30:
        raise ValueError('Expected 30 samples')
    for sample in by_sample.values():
        if set(sample) != {0, 1, 2, 3}:
            raise ValueError('Missing L')
        for length in (0, 1, 2):
            if sample[length]['task'] != sample[3]['task'] or sample[length]['initial_observation'] != sample[3]['initial_observation'] or sample[length]['prefix'] != sample[3]['prefix'][:length]:
                raise ValueError('Non-nested prefix')
    return rows


def score(args):
    # Deliberately never open labels or sampling manifest in scoring mode.
    inputs_path = args.directory / 'prepared/scoring_inputs.json'
    rows = load_inputs(inputs_path)
    output = args.directory / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        save(output / 'exact_prompts.json', [{'sample_id': r['sample_id'], 'L': r['L'],
             'prompt': prompt_for(r)} for r in rows])
        print('120 exact prompts saved; no scoring and no label access.')
        return
    load_env(args.env_file)
    import llm_verifier.fine_grained_reward as api
    model = args.model or os.environ.get('VERIFIER_MODEL')
    if not model:
        raise ValueError('Explicit verifier model required')
    client = (api.create_openai_client(base_url=os.environ['VERIFIER_BASE_URL'], api_key=os.environ.get('VERIFIER_API_KEY', 'EMPTY'))
              if os.environ.get('VERIFIER_BASE_URL') else api.create_client())
    manifest = {'model': model, 'n_evaluations': 3, 'criterion': CRITERION,
                'input_sha256': digest(inputs_path), 'adapter_sha256': digest(Path(__file__).with_name('alfworld_capability.py')),
                'package_version': importlib.metadata.version('llm-verifier'),
                'package_source_sha256': digest(Path(api.__file__))}
    file = output / 'manifest.json'
    if file.exists() and json.loads(file.read_text(encoding='utf-8')) != manifest:
        raise ValueError('Resume configuration mismatch')
    save(file, manifest)
    summaries = []
    for row in rows:
        filename = output / f"{row['sample_id']}_L{row['L']}.json"
        prompt = prompt_for(row)  # EXACT unchanged smoke prompt builder.
        if filename.exists():
            record = json.loads(filename.read_text(encoding='utf-8'))
            if record['prompt'] != prompt:
                raise ValueError('Cached prompt mismatch')
        else:
            record = {'sample_id': row['sample_id'], 'L': row['L'], 'prompt': prompt,
                      'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'evaluations': []}
        for rep in range(len(record['evaluations']), 3):
            start = time.monotonic()
            result = official_score(api, prompt, client, model)
            result.update({'repeat': rep + 1, 'elapsed_seconds': time.monotonic() - start})
            record['evaluations'].append(result)
            save(filename, record)
        scores = [r['score'] for r in record['evaluations']]
        summaries.append({'sample_id': row['sample_id'], 'L': row['L'],
                          'repeated_scores': scores, 'mean_score': statistics.mean(scores)})
        print(f"{row['sample_id']} L={row['L']}: {scores}, mean={statistics.mean(scores):.4f}", flush=True)
    save(output / 'completed_scores.json', {'status': 'complete', 'n_evaluations': 3,
                                           'input_sha256': digest(inputs_path), 'scores': summaries})


def evaluate(args):
    # Label merge is allowed ONLY once all 120 x 3 evaluations are complete.
    import numpy as np
    inputs = load_inputs(args.directory / 'prepared/scoring_inputs.json')
    output = args.directory / args.run_name
    done = json.loads((output / 'completed_scores.json').read_text(encoding='utf-8'))
    if done['status'] != 'complete' or done['input_sha256'] != digest(args.directory / 'prepared/scoring_inputs.json'):
        raise ValueError('Scoring incomplete/input mismatch')
    scores = done['scores']
    if len(scores) != 120 or {(r['sample_id'], r['L']) for r in scores} != {(r['sample_id'], r['L']) for r in inputs} or any(len(r['repeated_scores']) != 3 or not all(0 <= s <= 1 for s in r['repeated_scores']) or not np.isclose(r['mean_score'], np.mean(r['repeated_scores'])) for r in scores):
        raise ValueError('Invalid completed scores')
    labels = json.loads((args.directory / 'prepared/evaluation_labels.json').read_text(encoding='utf-8'))
    audit = json.loads((args.directory / 'prepared/sampling_audit.json').read_text(encoding='utf-8'))
    if digest(args.directory / 'prepared/evaluation_labels.json') != audit['label_sha256']:
        raise ValueError('Label file changed')
    by_id = {r['sample_id']: r for r in labels}
    if len(by_id) != 30 or sum(r['final_success'] for r in labels) != 15:
        raise ValueError('Label/sample mismatch')
    merged = [{**by_id[r['sample_id']], 'L': r['L'], 'score_1': r['repeated_scores'][0],
               'score_2': r['repeated_scores'][1], 'score_3': r['repeated_scores'][2], 'mean_score': r['mean_score']} for r in scores]
    metrics = []
    for length in (0, 1, 2, 3):
        subset = [r for r in merged if r['L'] == length]
        y = np.array([r['final_success'] for r in subset])
        s = np.array([r['mean_score'] for r in subset])
        pos, neg = s[y == 1], s[y == 0]
        auc = float(np.mean((pos[:, None] > neg) + .5 * (pos[:, None] == neg)))
        order = np.argsort(-s, kind='stable')
        ends = np.r_[np.flatnonzero(np.diff(s[order]) != 0), len(s)-1]
        tp = np.cumsum(y[order])[ends]
        recall = tp / y.sum()
        ap = float(np.sum(np.diff(np.r_[0, recall]) * tp / (ends + 1)))
        metrics.append({'L': length, 'AUROC': auc, 'Average_Precision': ap,
                        'success_mean': float(pos.mean()), 'success_median': float(np.median(pos)),
                        'failure_mean': float(neg.mean()), 'failure_median': float(np.median(neg))})
    samples = []
    for sample in by_id:
        seq = sorted((r for r in merged if r['sample_id'] == sample), key=lambda r: r['L'])
        ss = [r['mean_score'] for r in seq]
        label = by_id[sample]['final_success']
        change = (ss[-1] - ss[0]) * (1 if label else -1)
        samples.append({**by_id[sample], 'scores_L0_to_L3': ss,
                        'label_consistent_change': change,
                        'threshold_05_crossings': [(i, i+1) for i in range(3) if (ss[i] >= .5) != (ss[i+1] >= .5)]})
    # Always list descriptive extremes; do not imply each exceeds a high/low cutoff.
    examples = {'high_score_failures': sorted((r for r in samples if not r['final_success']), key=lambda r: -r['scores_L0_to_L3'][-1])[:3],
                'low_score_successes': sorted((r for r in samples if r['final_success']), key=lambda r: r['scores_L0_to_L3'][-1])[:3],
                'judgment_corrected': sorted((r for r in samples if r['label_consistent_change'] > 0), key=lambda r: -r['label_consistent_change'])[:3],
                'judgment_worsened': sorted((r for r in samples if r['label_consistent_change'] < 0), key=lambda r: r['label_consistent_change'])[:3]}
    for examples_list in examples.values():
        for example in examples_list:
            example['observed_prefixes'] = [r for r in inputs if r['sample_id'] == example['sample_id']]
    write_csv(output / 'evaluation_task_scores.csv', merged)
    write_csv(output / 'metrics_by_L.csv', metrics)
    save(output / 'human_review_examples.json', examples)
    save(output / 'evaluation_summary.json', {'metrics': metrics, 'examples': examples,
         'notes': '30 stratified records, 15/15. AP prevalence reference .5; not representative deployment prevalence. Repeated task IDs may occur across runs; no independence/significance claims. Score is not calibrated success probability. Correction/worsening is L3 vs L0 movement toward/away from observed label, not necessarily monotonic at every intermediate L.'})
    # Dependency-free scientific line chart.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="720" height="450"><rect width="100%" height="100%" fill="white"/>', '<g font-family="Arial" font-size="14">']
    for value in [0, .25, .5, .75, 1]:
        yy = 370 - value*300
        svg.append(f'<line x1="70" y1="{yy}" x2="660" y2="{yy}" stroke="#ddd"/><text x="30" y="{yy+5}">{value}</text>')
    for key, color in [('AUROC', '#2563eb'), ('Average_Precision', '#ea580c')]:
        points = ' '.join(f'{90+r["L"]*180},{370-r[key]*300}' for r in metrics)
        svg.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="3"/>')
    for length in range(4):
        svg.append(f'<text x="{90+length*180}" y="400">L={length}</text>')
    svg += ['<text x="70" y="30">Real prefix feasibility (30 records)</text><text x="70" y="52" fill="#2563eb">AUROC</text><text x="180" y="52" fill="#ea580c">Average Precision</text></g></svg>']
    (output / 'metrics_by_L.svg').write_text('\n'.join(svg), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'score', 'evaluate'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--run-name', default='real_run1')
    parser.add_argument('--model')
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    {'prepare': prepare, 'score': score, 'evaluate': evaluate}[args.mode](args)


if __name__ == '__main__':
    main()
