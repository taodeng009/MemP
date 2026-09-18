"""Compare equal-weight RU and rank-specific Top-3 binomial logistic models."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
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
                        "contributions": [],
                    },
                )
                contributions = [
                    max(0.0, THRESHOLD - float(memory["score"]))
                    for memory in row.get("retrieved_memories", [])[:3]
                ]
                contributions.extend([0.0] * (3 - len(contributions)))
                record["contributions"].append(contributions)
                record["success_count"] += int(bool(row["reward"]))

    output = []
    for record in tasks.values():
        values = np.asarray(record.pop("contributions"), dtype=float)
        if values.shape != (3, 3):
            raise ValueError(f"Expected three runs and three ranks for {record['task_id']}")
        means = values.mean(axis=0)
        output.append(
            {
                **record,
                "u1": float(means[0]),
                "u2": float(means[1]),
                "u3": float(means[2]),
                "ru": float(means.sum()),
                "max_rank_range": float(np.max(np.ptp(values, axis=0))),
            }
        )
    return sorted(output, key=lambda row: row["task_id"])


def design_matrices(tasks):
    families = sorted({row["task_family"] for row in tasks})
    baseline = families[0]
    family_columns = families[1:]
    common = []
    for row in tasks:
        common.append(
            [1.0]
            + [float(row["task_family"] == family) for family in family_columns]
        )
    common = np.asarray(common, dtype=float)
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)[:, None]
    ranks = np.asarray([[row["u1"], row["u2"], row["u3"]] for row in tasks])
    x1 = np.hstack([common, ru])
    x2 = np.hstack([common, ranks])
    names1 = ["Intercept", *[f"family[{name}]" for name in family_columns], "RU"]
    names2 = [
        "Intercept",
        *[f"family[{name}]" for name in family_columns],
        "u1",
        "u2",
        "u3",
    ]
    return x1, x2, names1, names2, baseline


def sigmoid(values):
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def log_likelihood(x, y, trials, beta, include_constant=True):
    eta = x @ beta
    core = np.sum(y * (-np.logaddexp(0.0, -eta)) + (trials - y) * (-np.logaddexp(0.0, eta)))
    if include_constant:
        core += sum(
            math.lgamma(int(n) + 1)
            - math.lgamma(int(k) + 1)
            - math.lgamma(int(n - k) + 1)
            for k, n in zip(y, trials)
        )
    return float(core)


def fit_logistic(x, y, trials, ridge=0.0, max_iter=300):
    beta = np.zeros(x.shape[1], dtype=float)
    penalty_mask = np.ones(x.shape[1], dtype=float)
    penalty_mask[0] = 0.0
    previous = -math.inf
    for _ in range(max_iter):
        eta = x @ beta
        probability = sigmoid(eta)
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
            candidate_objective = log_likelihood(x, y, trials, candidate, False) - 0.5 * ridge * np.sum(
                (penalty_mask * candidate) ** 2
            )
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
    information = x.T @ (weights[:, None] * x) + np.eye(x.shape[1]) * 1e-10
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return beta, standard_errors


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


def cross_validate(tasks, x1, x2, y, trials, repeats=20, folds=5):
    results = []
    coefficients = []
    all_indices = np.arange(len(tasks))
    for repeat in range(repeats):
        split = make_folds(tasks, 42 + repeat, folds)
        predictions1 = np.zeros(len(tasks), dtype=float)
        predictions2 = np.zeros(len(tasks), dtype=float)
        for fold_index, test_indices in enumerate(split):
            test = np.asarray(test_indices, dtype=int)
            train = np.setdiff1d(all_indices, test)
            beta1, _ = fit_logistic(x1[train], y[train], trials[train], ridge=1e-4)
            beta2, _ = fit_logistic(x2[train], y[train], trials[train], ridge=1e-4)
            predictions1[test] = sigmoid(x1[test] @ beta1)
            predictions2[test] = sigmoid(x2[test] @ beta2)
            coefficients.append(
                {
                    "repeat": repeat,
                    "fold": fold_index,
                    "beta1": float(beta2[-3]),
                    "beta2": float(beta2[-2]),
                    "beta3": float(beta2[-1]),
                }
            )
        ll1, brier1 = prediction_metrics(y, trials, predictions1)
        ll2, brier2 = prediction_metrics(y, trials, predictions2)
        results.append(
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
    return results, coefficients


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
    parser.add_argument("--coefficient-csv", type=Path, required=True)
    args = parser.parse_args()

    tasks = load_tasks(extended_path(args.results_root))
    x1, x2, names1, names2, baseline = design_matrices(tasks)
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)

    full_beta1, full_se1 = fit_logistic(x1, y, trials)
    full_beta2, full_se2 = fit_logistic(x2, y, trials)
    ll1 = log_likelihood(x1, y, trials, full_beta1)
    ll2 = log_likelihood(x2, y, trials, full_beta2)
    lr = 2.0 * (ll2 - ll1)
    lr_p = math.exp(-max(lr, 0.0) / 2.0)  # chi-square survival for df=2

    cv_rows, coefficient_rows = cross_validate(tasks, x1, x2, y, trials)
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
        }
        for name in metric_names
    }
    coefficient_summary = {
        name: {
            "mean": float(np.mean([row[name] for row in coefficient_rows])),
            "sd": float(np.std([row[name] for row in coefficient_rows], ddof=1)),
            "min": float(np.min([row[name] for row in coefficient_rows])),
            "max": float(np.max([row[name] for row in coefficient_rows])),
        }
        for name in ("beta1", "beta2", "beta3")
    }
    order_count = sum(
        row["beta1"] > row["beta2"] > row["beta3"] for row in coefficient_rows
    )

    summary = {
        "task_count": len(tasks),
        "trial_count": int(np.sum(trials)),
        "family_baseline": baseline,
        "max_feature_range_across_runs": max(row["max_rank_range"] for row in tasks),
        "m1": {
            "log_likelihood": ll1,
            "coefficients": dict(zip(names1, map(float, full_beta1))),
            "standard_errors": dict(zip(names1, map(float, full_se1))),
        },
        "m2": {
            "log_likelihood": ll2,
            "coefficients": dict(zip(names2, map(float, full_beta2))),
            "standard_errors": dict(zip(names2, map(float, full_se2))),
        },
        "likelihood_ratio_test": {"statistic": lr, "df": 2, "p_value": lr_p},
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
        "m2_cv_coefficients": coefficient_summary,
        "m2_beta_order": {
            "beta1_gt_beta2_gt_beta3_count": order_count,
            "fit_count": len(coefficient_rows),
            "fraction": order_count / len(coefficient_rows),
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(args.cv_csv, cv_rows)
    write_csv(args.coefficient_csv, coefficient_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
