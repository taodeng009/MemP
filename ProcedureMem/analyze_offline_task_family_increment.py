"""Test the incremental predictive value of TaskFamily beyond RU."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


WEAK_RIDGE = 1e-4


def load_shared(path: Path):
    spec = importlib.util.spec_from_file_location("family_ru_shared", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def design_matrices(tasks):
    families = sorted({row["task_family"] for row in tasks})
    baseline = families[0]
    others = families[1:]
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)[:, None]
    intercept = np.ones((len(tasks), 1), dtype=float)
    family_dummy = np.asarray(
        [[float(row["task_family"] == family) for family in others] for row in tasks],
        dtype=float,
    )
    x0 = np.hstack([intercept, ru])
    x1 = np.hstack([intercept, family_dummy, ru])
    return x0, x1, families, baseline


def ridge_penalty(parameter_count):
    penalty = np.eye(parameter_count, dtype=float) * WEAK_RIDGE
    penalty[0, 0] = 0.0
    return penalty


def repeated_cv(shared, tasks, x0, x1, y, trials):
    all_indices = np.arange(len(tasks))
    rows = []
    for repeat in range(20):
        prediction0 = np.zeros(len(tasks), dtype=float)
        prediction1 = np.zeros(len(tasks), dtype=float)
        for test_indices in shared.make_folds(tasks, 42 + repeat, 5):
            test = np.asarray(test_indices, dtype=int)
            train = np.setdiff1d(all_indices, test)
            beta0 = shared.fit_logistic(
                x0[train], y[train], trials[train], ridge=WEAK_RIDGE
            )[0]
            beta1 = shared.fit_logistic(
                x1[train], y[train], trials[train], ridge=WEAK_RIDGE
            )[0]
            prediction0[test] = shared.sigmoid(x0[test] @ beta0)
            prediction1[test] = shared.sigmoid(x1[test] @ beta1)
        ll0, brier0 = shared.prediction_metrics(y, trials, prediction0)
        ll1, brier1 = shared.prediction_metrics(y, trials, prediction1)
        rows.append(
            {
                "repeat": repeat,
                "m0_log_loss": ll0,
                "m1_log_loss": ll1,
                "delta_log_loss_m1_minus_m0": ll1 - ll0,
                "m0_brier": brier0,
                "m1_brier": brier1,
                "delta_brier_m1_minus_m0": brier1 - brier0,
            }
        )
    return rows


def summarize(rows, names):
    return {
        name: {
            "mean": float(np.mean([row[name] for row in rows])),
            "sd": float(np.std([row[name] for row in rows], ddof=1)),
            "min": float(np.min([row[name] for row in rows])),
            "max": float(np.max([row[name] for row in rows])),
        }
        for name in names
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shared-script", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--cv-csv", type=Path, required=True)
    args = parser.parse_args()

    shared = load_shared(args.shared_script.resolve())
    tasks = shared.load_tasks(shared.extended_path(args.results_root))
    x0, x1, families, baseline = design_matrices(tasks)
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)

    beta0, covariance0 = shared.fit_logistic(x0, y, trials)
    beta1, covariance1 = shared.fit_logistic(x1, y, trials)
    ll0 = shared.log_likelihood(x0, y, trials, beta0)
    ll1 = shared.log_likelihood(x1, y, trials, beta1)
    lr = max(0.0, 2.0 * (ll1 - ll0))
    degrees_freedom = len(families) - 1
    p_value = shared.gammaincc(degrees_freedom / 2.0, lr / 2.0)

    cv_rows = repeated_cv(shared, tasks, x0, x1, y, trials)
    metric_names = [
        "m0_log_loss",
        "m1_log_loss",
        "delta_log_loss_m1_minus_m0",
        "m0_brier",
        "m1_brier",
        "delta_brier_m1_minus_m0",
    ]
    family_trial_counts = {}
    for family in families:
        selected = [row for row in tasks if row["task_family"] == family]
        family_trial_counts[family] = {
            "task_count": len(selected),
            "success_trials": sum(row["success_count"] for row in selected),
            "total_trials": 3 * len(selected),
        }

    summary = {
        "task_count": len(tasks),
        "trial_count": int(trials.sum()),
        "family_baseline": baseline,
        "m0": {
            "log_likelihood": ll0,
            "coefficients": {
                "Intercept": float(beta0[0]),
                "RU": float(beta0[1]),
            },
        },
        "m1": {
            "log_likelihood": ll1,
            "coefficients": {
                "Intercept": float(beta1[0]),
                **{
                    f"family[{family}]": float(beta1[index + 1])
                    for index, family in enumerate(families[1:])
                },
                "RU": float(beta1[-1]),
            },
        },
        "likelihood_ratio_test": {
            "statistic": lr,
            "df": degrees_freedom,
            "p_value": p_value,
        },
        "cross_validation": {
            "folds": 5,
            "repeats": 20,
            "weak_ridge": WEAK_RIDGE,
            "summary": summarize(cv_rows, metric_names),
            "m1_better_log_loss_repeats": sum(
                row["delta_log_loss_m1_minus_m0"] < 0 for row in cv_rows
            ),
            "m1_better_brier_repeats": sum(
                row["delta_brier_m1_minus_m0"] < 0 for row in cv_rows
            ),
        },
        "family_trial_counts": family_trial_counts,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.cv_csv, cv_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
