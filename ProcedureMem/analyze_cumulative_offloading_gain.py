"""Full gain curves with finite-population random-permutation envelopes."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
import numpy as np

if __package__:
    from .analyze_task_semantics_linear_probe import write_csv
else:
    from analyze_task_semantics_linear_probe import write_csv


def intervals(mask):
    budgets = np.flatnonzero(mask) + 1
    result = []
    for b in budgets:
        if result and b == result[-1][1] + 1:
            result[-1][1] = int(b)
        else:
            result.append([int(b), int(b)])
    return result


def analyze(gain, priorities, permutations=50000, seed=42):
    # Integer units preserve ties and deterministic total exactly.
    units = np.rint(gain * 3).astype(int)
    if not np.allclose(units / 3, gain, atol=1e-12, rtol=0):
        raise ValueError('Expected net gains in thirds')
    n = len(gain)
    b = np.arange(1, n + 1)
    curves = {name: np.cumsum(units[np.argsort(-s, kind='stable')]) / 3
              for name, s in priorities.items()}
    curves['oracle'] = np.cumsum(np.sort(units)[::-1]) / 3
    mean = b * gain.mean()
    sd = np.sqrt(b * (n - b) / (n - 1) * np.var(gain))
    rng = np.random.default_rng(seed)
    random = np.empty((permutations, n))
    for i in range(permutations):
        random[i] = np.cumsum(rng.permutation(units)) / 3
    low, high = np.quantile(random, [.025, .975], axis=0)
    active = sd > 1e-12
    z = (random[:, active] - mean[active]) / sd[active]
    maximum = z.max(axis=1)
    # Budgetwise max-T with Bonferroni over two independently specified rules.
    critical = float(np.quantile(maximum, 1 - .05 / len(priorities)))
    simultaneous_upper = mean + critical * sd
    mc_mean = random.mean(0)
    mc_se = random.std(0, ddof=1) / np.sqrt(permutations)
    records, summary = [], {}
    for name, curve in curves.items():
        if name == 'oracle':
            continue
        raw = (1 + np.sum(random >= curve - 1e-12, axis=0)) / (permutations + 1)
        observed_z = np.zeros(n)
        observed_z[active] = (curve[active] - mean[active]) / sd[active]
        adjusted = np.ones(n)
        for j in np.flatnonzero(active):
            adjusted[j] = min(1., len(priorities) * (1 + np.sum(maximum >= observed_z[j] - 1e-12)) / (permutations + 1))
        summary[name] = {'above_random_expectation_budgets': intervals(curve > mean + 1e-12),
                         'pointwise_p_lt_005_budgets': intervals(raw < .05),
                         'familywise_p_lt_005_budgets': intervals(adjusted < .05),
                         'minimum_pointwise_p': float(raw.min()), 'minimum_familywise_p': float(adjusted.min()),
                         'largest_excess_gain': float((curve - mean).max()),
                         'largest_excess_budget': int(np.argmax(curve - mean) + 1)}
        for j in range(n):
            if len(records) <= j:
                records.append({'B': j + 1, 'random_expectation': float(mean[j]),
                                'random_permutation_mean': float(mc_mean[j]),
                                'random_pointwise_95_low': float(low[j]), 'random_pointwise_95_high': float(high[j]),
                                'random_mean_MC_95_low': float(mc_mean[j] - 1.96 * mc_se[j]),
                                'random_mean_MC_95_high': float(mc_mean[j] + 1.96 * mc_se[j]),
                                'random_familywise_one_sided_upper': float(simultaneous_upper[j]),
                                'oracle_gain': float(curves['oracle'][j])})
            records[j].update({name + '_gain': float(curve[j]), name + '_pointwise_p': float(raw[j]),
                               name + '_familywise_p': float(adjusted[j])})
    return records, summary, critical


def svg_plot(path, records):
    width, height = 1000, 620
    left, right, top, bottom = 80, 960, 70, 535
    n = len(records)
    ymin, ymax = -5., 50.
    x = lambda b: left + (b - 1) / (n - 1) * (right - left)
    y = lambda g: bottom - (g - ymin) / (ymax - ymin) * (bottom - top)
    points = lambda rows, key: ' '.join(f'{x(r["B"]):.2f},{y(r[key]):.2f}' for r in rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="14" fill="#222">',
             '<text x="80" y="30" font-size="21">Cumulative captured net gain — 134 tasks</text>',
             '<text x="80" y="52">50,000 random permutations; shaded region: pointwise 95% random interval</text>']
    for g in range(0, 51, 10):
        parts += [f'<line x1="{left}" y1="{y(g)}" x2="{right}" y2="{y(g)}" stroke="#ddd"/>',
                  f'<text x="65" y="{y(g)+5}" text-anchor="end">{g}</text>']
    band = points(records, 'random_pointwise_95_high') + ' ' + points(records[::-1], 'random_pointwise_95_low')
    parts.append(f'<polygon points="{band}" fill="#cbd5e1" opacity="0.7"/>')
    series = [('oracle_gain', 'Oracle', '#111827', ''), ('random_expectation', 'Random expectation', '#64748b', '6 4'),
              ('random_familywise_one_sided_upper', 'FWER-adjusted upper threshold', '#9333ea', '3 4'),
              ('minus_ru_gain', '-RU priority', '#2563eb', ''), ('bd_gain', 'BD priority', '#ea580c', '')]
    for key, label, color, dash in series:
        parts.append(f'<polyline points="{points(records,key)}" fill="none" stroke="{color}" stroke-width="2.3" stroke-dasharray="{dash}"/>')
    for i, (_, label, color, _) in enumerate(series):
        parts += [f'<rect x="95" y="{80+i*23}" width="17" height="3" fill="{color}"/>',
                  f'<text x="120" y="{86+i*23}">{label}</text>']
    for b in [1, 20, 40, 60, 80, 100, 120, 134]:
        parts.append(f'<text x="{x(b)}" y="560" text-anchor="middle">{b}</text>')
    parts += ['<text x="520" y="590" text-anchor="middle">Offloading budget B</text>',
              '<text transform="translate(23,310) rotate(-90)" text-anchor="middle">Sum of empirical net success gains</text>',
              '</g></svg>']
    path.write_text('\n'.join(parts), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    with args.inputs.open(encoding='utf-8') as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r['task_index']))
    if len(rows) != 134 or len({r['task_id'] for r in rows}) != 134:
        raise ValueError('Expected 134 distinct tasks')
    gain = np.array([float(r['net_gain']) for r in rows])
    priorities = {'minus_ru': -np.array([float(r['ru']) for r in rows]),
                  'bd': np.array([float(r['bd']) for r in rows])}
    records, rules, critical = analyze(gain, priorities)
    summary = {'task_count': 134, 'permutations': 50000, 'seed': 42, 'rules': rules,
               'total_net_gain': float(gain.sum()), 'critical_standardized_max': critical,
               'inference': 'Fixed-task random-ranking null; max standardized deviation over B=1..133; Bonferroni across two prespecified priorities; one-sided familywise alpha .05',
               'intervals': 'Pointwise 95% permutation random-selection intervals, not confidence intervals for population performance. Monte Carlo mean 95% intervals describe simulation error only.',
               'ties': 'task_index ascending; B=134 deterministic for all rules',
               'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / 'cumulative_gain.csv', records)
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    svg_plot(args.output_dir / 'cumulative_gain.svg', records)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
