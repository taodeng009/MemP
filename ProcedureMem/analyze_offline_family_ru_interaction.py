"""Compare common and task-family-specific RU slopes on three offline runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_DIRS = {
    "run1": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run1",
    "run2": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run2",
    "run3": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run3",
}
THRESHOLD = 0.5


def extended_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def load_tasks(root: Path):
    tasks: dict[str, dict] = {}
    for run, directory in RUN_DIRS.items():
        path = root / directory / "memory" / "results.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                record = tasks.setdefault(
                    row["task_id"],
                    {
                        "task_id": row["task_id"],
                        "task_family": row["task_type"].split("-", 1)[0],
                        "success_count": 0,
                        "run_ru": [],
                    },
                )
                contributions = [
                    max(0.0, THRESHOLD - float(memory["score"]))
                    for memory in row.get("retrieved_memories", [])[:3]
                ]
                record["run_ru"].append(sum(contributions))
                record["success_count"] += int(bool(row["reward"]))

    output = []
    for record in tasks.values():
        run_ru = np.asarray(record.pop("run_ru"), dtype=float)
        if run_ru.shape != (3,):
            raise ValueError(f"Expected three runs for {record['task_id']}")
        output.append(
            {
                **record,
                "ru": float(run_ru.mean()),
                "ru_run_range": float(np.ptp(run_ru)),
            }
        )
    return sorted(output, key=lambda row: row["task_id"])


def design_matrices(tasks):
    families = sorted({row["task_family"] for row in tasks})
    baseline = families[0]
    others = families[1:]
    family_dummy = np.asarray(
        [[float(row["task_family"] == family) for family in others] for row in tasks]
    )
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)[:, None]
    intercept = np.ones((len(tasks), 1), dtype=float)
    common = np.hstack([intercept, family_dummy])
    x1 = np.hstack([common, ru])
    x2 = np.hstack([common, ru, family_dummy * ru])
    names1 = ["Intercept", *[f"family[{f}]" for f in others], "RU"]
    names2 = [
        "Intercept",
        *[f"family[{f}]" for f in others],
        "RU",
        *[f"family[{f}]:RU" for f in others],
    ]
    return x1, x2, names1, names2, families, baseline


def sigmoid(values):
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def log_likelihood(x, y, trials, beta, include_constant=True):
    eta = x @ beta
    value = np.sum(
        y * (-np.logaddexp(0.0, -eta))
        + (trials - y) * (-np.logaddexp(0.0, eta))
    )
    if include_constant:
        value += sum(
            math.lgamma(int(n) + 1)
            - math.lgamma(int(k) + 1)
            - math.lgamma(int(n - k) + 1)
            for k, n in zip(y, trials)
        )
    return float(value)


def fit_logistic(x, y, trials, ridge=0.0, max_iter=500):
    beta = np.zeros(x.shape[1], dtype=float)
    penalty_mask = np.ones(x.shape[1], dtype=float)
    penalty_mask[0] = 0.0
    previous = -math.inf
    for _ in range(max_iter):
        probability = sigmoid(x @ beta)
        weights = trials * probability * (1.0 - probability)
        gradient = x.T @ (y - trials * probability) - ridge * penalty_mask * beta
        information = x.T @ (weights[:, None] * x) + np.diag(ridge * penalty_mask)
        information += np.eye(x.shape[1]) * 1e-10
        step = np.linalg.solve(information, gradient)
        objective = log_likelihood(x, y, trials, beta, False) - 0.5 * ridge * np.sum(
            (penalty_mask * beta) ** 2
        )
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            candidate_objective = log_likelihood(
                x, y, trials, candidate, False
            ) - 0.5 * ridge * np.sum((penalty_mask * candidate) ** 2)
            if candidate_objective >= objective - 1e-12:
                beta = candidate
                break
            scale *= 0.5
        current = log_likelihood(x, y, trials, beta, False)
        if max(abs(scale * step)) < 1e-9 or abs(current - previous) < 1e-11:
            break
        previous = current

    probability = sigmoid(x @ beta)
    weights = trials * probability * (1.0 - probability)
    information = x.T @ (weights[:, None] * x) + np.diag(ridge * penalty_mask)
    information += np.eye(x.shape[1]) * 1e-10
    covariance = np.linalg.pinv(information)
    return beta, covariance


def gammaincc(shape, value):
    """Regularized upper incomplete gamma Q(shape, value)."""
    if value < 0.0 or shape <= 0.0:
        raise ValueError("Invalid incomplete-gamma arguments")
    if value == 0.0:
        return 1.0
    eps = 3e-14
    tiny = 1e-300
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        ap = shape
        for _ in range(10000):
            ap += 1.0
            term *= value / ap
            total += term
            if abs(term) < abs(total) * eps:
                break
        lower = total * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
        return max(0.0, min(1.0, 1.0 - lower))

    b = value + 1.0 - shape
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for index in range(1, 10000):
        an = -index * (index - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    upper = h * math.exp(-value + shape * math.log(value) - math.lgamma(shape))
    return max(0.0, min(1.0, upper))


def make_folds(tasks, seed, fold_count=5):
    strata = defaultdict(list)
    for index, row in enumerate(tasks):
        strata[(row["task_family"], row["success_count"])].append(index)
    rng = random.Random(seed)
    folds = [[] for _ in range(fold_count)]
    for indices in strata.values():
        rng.shuffle(indices)
        offset = rng.randrange(fold_count)
        for position, index in enumerate(indices):
            folds[(offset + position) % fold_count].append(index)
    return [sorted(fold) for fold in folds]


def prediction_metrics(y, trials, probability):
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    total = float(np.sum(trials))
    log_loss = -float(
        np.sum(y * np.log(probability) + (trials - y) * np.log(1.0 - probability))
    ) / total
    brier = float(
        np.sum(y * (1.0 - probability) ** 2 + (trials - y) * probability**2)
    ) / total
    return log_loss, brier


def family_slopes(beta, covariance, families, baseline):
    ru_index = len(families)
    slopes = {}
    for family_index, family in enumerate(families):
        contrast = np.zeros(len(beta), dtype=float)
        contrast[ru_index] = 1.0
        if family != baseline:
            interaction_index = ru_index + 1 + (family_index - 1)
            contrast[interaction_index] = 1.0
        estimate = float(contrast @ beta)
        standard_error = float(math.sqrt(max(contrast @ covariance @ contrast, 0.0)))
        slopes[family] = {
            "estimate": estimate,
            "standard_error": standard_error,
            "ci95_low": estimate - 1.96 * standard_error,
            "ci95_high": estimate + 1.96 * standard_error,
        }
    return slopes


def cross_validate(tasks, x1, x2, y, trials, families, baseline, repeats=20, folds=5):
    all_indices = np.arange(len(tasks))
    metrics = []
    slope_rows = []
    for repeat in range(repeats):
        predictions1 = np.zeros(len(tasks), dtype=float)
        predictions2 = np.zeros(len(tasks), dtype=float)
        for fold_index, test_indices in enumerate(make_folds(tasks, 42 + repeat, folds)):
            test = np.asarray(test_indices, dtype=int)
            train = np.setdiff1d(all_indices, test)
            beta1, _ = fit_logistic(x1[train], y[train], trials[train], ridge=1e-4)
            beta2, covariance2 = fit_logistic(
                x2[train], y[train], trials[train], ridge=1e-4
            )
            predictions1[test] = sigmoid(x1[test] @ beta1)
            predictions2[test] = sigmoid(x2[test] @ beta2)
            for family, values in family_slopes(
                beta2, covariance2, families, baseline
            ).items():
                slope_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold_index,
                        "task_family": family,
                        "ru_slope": values["estimate"],
                    }
                )
        ll1, brier1 = prediction_metrics(y, trials, predictions1)
        ll2, brier2 = prediction_metrics(y, trials, predictions2)
        metrics.append(
            {
                "repeat": repeat,
                "m1_log_loss": ll1,
                "m2_log_loss": ll2,
                "delta_log_loss_m2_minus_m1": ll2 - ll1,
                "m1_brier": brier1,
                "m2_brier": brier2,
                "delta_brier_m2_minus_m1": brier2 - brier1,
            }
        )
    return metrics, slope_rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--cv-csv", type=Path, required=True)
    parser.add_argument("--slope-csv", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(extended_path(args.results_root))
    x1, x2, names1, names2, families, baseline = design_matrices(tasks)
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)

    beta1, covariance1 = fit_logistic(x1, y, trials)
    beta2, covariance2 = fit_logistic(x2, y, trials)
    ll1 = log_likelihood(x1, y, trials, beta1)
    ll2 = log_likelihood(x2, y, trials, beta2)
    lr = max(0.0, 2.0 * (ll2 - ll1))
    degrees_freedom = len(families) - 1
    p_value = gammaincc(degrees_freedom / 2.0, lr / 2.0)

    cv_rows, slope_rows = cross_validate(
        tasks, x1, x2, y, trials, families, baseline
    )
    metric_names = [
        "m1_log_loss",
        "m2_log_loss",
        "delta_log_loss_m2_minus_m1",
        "m1_brier",
        "m2_brier",
        "delta_brier_m2_minus_m1",
    ]
    cv_summary = {
        name: {
            "mean": float(np.mean([row[name] for row in cv_rows])),
            "sd": float(np.std([row[name] for row in cv_rows], ddof=1)),
            "min": float(np.min([row[name] for row in cv_rows])),
            "max": float(np.max([row[name] for row in cv_rows])),
        }
        for name in metric_names
    }

    family_diagnostics = {}
    full_slopes = family_slopes(beta2, covariance2, families, baseline)
    for family in families:
        family_tasks = [row for row in tasks if row["task_family"] == family]
        ru_values = [row["ru"] for row in family_tasks]
        success_values = [row["success_count"] for row in family_tasks]
        cv_slopes = [
            row["ru_slope"] for row in slope_rows if row["task_family"] == family
        ]
        family_diagnostics[family] = {
            "task_count": len(family_tasks),
            "success_trials": int(sum(success_values)),
            "total_trials": 3 * len(family_tasks),
            "ru_min": float(min(ru_values)),
            "ru_max": float(max(ru_values)),
            "ru_sd": float(np.std(ru_values, ddof=1)),
            **full_slopes[family],
            "cv_slope_mean": float(np.mean(cv_slopes)),
            "cv_slope_sd": float(np.std(cv_slopes, ddof=1)),
            "cv_slope_min": float(np.min(cv_slopes)),
            "cv_slope_max": float(np.max(cv_slopes)),
            "cv_positive_fraction": float(np.mean(np.asarray(cv_slopes) > 0.0)),
        }

    summary = {
        "task_count": len(tasks),
        "trial_count": int(np.sum(trials)),
        "success_count_distribution": dict(
            sorted(Counter(row["success_count"] for row in tasks).items())
        ),
        "family_baseline": baseline,
        "max_ru_range_across_runs": max(row["ru_run_range"] for row in tasks),
        "m1": {
            "log_likelihood": ll1,
            "coefficients": dict(zip(names1, map(float, beta1))),
            "standard_errors": dict(
                zip(names1, map(float, np.sqrt(np.maximum(np.diag(covariance1), 0.0))))
            ),
        },
        "m2": {
            "log_likelihood": ll2,
            "coefficients": dict(zip(names2, map(float, beta2))),
            "standard_errors": dict(
                zip(names2, map(float, np.sqrt(np.maximum(np.diag(covariance2), 0.0))))
            ),
        },
        "likelihood_ratio_test": {
            "statistic": lr,
            "df": degrees_freedom,
            "p_value": p_value,
        },
        "cross_validation": {
            "folds": 5,
            "repeats": 20,
            "weak_ridge_for_separated_training_folds": 1e-4,
            "summary": cv_summary,
            "m2_better_log_loss_repeats": sum(
                row["delta_log_loss_m2_minus_m1"] < 0 for row in cv_rows
            ),
            "m2_better_brier_repeats": sum(
                row["delta_brier_m2_minus_m1"] < 0 for row in cv_rows
            ),
        },
        "family_ru_slopes": family_diagnostics,
    }

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.cv_csv, cv_rows)
    write_csv(args.slope_csv, slope_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
