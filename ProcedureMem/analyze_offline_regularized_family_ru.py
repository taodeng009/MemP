"""Nested-CV comparison of common and L2-shrunk family-specific RU slopes."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


LAMBDA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
WEAK_RIDGE = 1e-4


def load_shared_module(path: Path):
    spec = importlib.util.spec_from_file_location("family_ru_shared", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def design_matrices(tasks):
    families = sorted({row["task_family"] for row in tasks})
    baseline = families[0]
    others = families[1:]
    family_dummy = np.asarray(
        [[float(row["task_family"] == family) for family in others] for row in tasks],
        dtype=float,
    )
    family_onehot = np.asarray(
        [[float(row["task_family"] == family) for family in families] for row in tasks],
        dtype=float,
    )
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)[:, None]
    common = np.hstack([np.ones((len(tasks), 1)), family_dummy])
    x1 = np.hstack([common, ru])
    x3 = np.hstack([common, family_onehot * ru])
    return x1, x3, families, baseline


def sigmoid(values):
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def log_likelihood(x, y, trials, beta):
    eta = x @ beta
    return float(
        np.sum(
            y * (-np.logaddexp(0.0, -eta))
            + (trials - y) * (-np.logaddexp(0.0, eta))
        )
    )


def fit_penalized(x, y, trials, penalty, max_iter=500):
    beta = np.zeros(x.shape[1], dtype=float)
    previous = -math.inf
    for _ in range(max_iter):
        probability = sigmoid(x @ beta)
        weights = trials * probability * (1.0 - probability)
        gradient = x.T @ (y - trials * probability) - penalty @ beta
        information = x.T @ (weights[:, None] * x) + penalty
        information += np.eye(x.shape[1]) * 1e-10
        step = np.linalg.solve(information, gradient)
        objective = log_likelihood(x, y, trials, beta) - 0.5 * beta @ penalty @ beta
        scale = 1.0
        while scale > 1e-8:
            candidate = beta + scale * step
            candidate_objective = (
                log_likelihood(x, y, trials, candidate)
                - 0.5 * candidate @ penalty @ candidate
            )
            if candidate_objective >= objective - 1e-12:
                beta = candidate
                break
            scale *= 0.5
        current = log_likelihood(x, y, trials, beta)
        if max(abs(scale * step)) < 1e-9 or abs(current - previous) < 1e-11:
            break
        previous = current
    return beta


def m1_penalty(parameter_count):
    penalty = np.eye(parameter_count) * WEAK_RIDGE
    penalty[0, 0] = 0.0
    return penalty


def m3_penalty(family_count, lambda_value):
    common_count = family_count
    parameter_count = common_count + family_count
    penalty = np.zeros((parameter_count, parameter_count), dtype=float)
    # Same weak numerical ridge as M1 on non-intercept nuisance parameters.
    penalty[1:common_count, 1:common_count] += np.eye(common_count - 1) * WEAK_RIDGE
    # Family slopes are decomposed into an unpenalized mean plus penalized deviations.
    centering = np.eye(family_count) - np.ones((family_count, family_count)) / family_count
    penalty[common_count:, common_count:] = lambda_value * centering
    # A negligible common-direction ridge matches the numerical stabilization of M1's RU.
    penalty[common_count:, common_count:] += np.ones(
        (family_count, family_count)
    ) * (WEAK_RIDGE / family_count)
    return penalty


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


def subset_folds(shared, tasks, global_indices, seed, fold_count=5):
    subset_tasks = [tasks[index] for index in global_indices]
    local_folds = shared.make_folds(subset_tasks, seed, fold_count)
    return [[global_indices[index] for index in fold] for fold in local_folds]


def select_lambda(
    shared, tasks, x3, y, trials, train_indices, family_count, seed
):
    inner_folds = subset_folds(shared, tasks, list(train_indices), seed)
    train_set = np.asarray(train_indices, dtype=int)
    scores = {}
    for lambda_value in LAMBDA_GRID:
        predictions = np.zeros(len(tasks), dtype=float)
        evaluated = []
        penalty = m3_penalty(family_count, lambda_value)
        for validation_indices in inner_folds:
            validation = np.asarray(validation_indices, dtype=int)
            inner_train = np.setdiff1d(train_set, validation)
            beta = fit_penalized(
                x3[inner_train], y[inner_train], trials[inner_train], penalty
            )
            predictions[validation] = sigmoid(x3[validation] @ beta)
            evaluated.extend(validation_indices)
        evaluated = np.asarray(sorted(evaluated), dtype=int)
        log_loss, brier = prediction_metrics(
            y[evaluated], trials[evaluated], predictions[evaluated]
        )
        scores[lambda_value] = {"log_loss": log_loss, "brier": brier}
    selected = min(LAMBDA_GRID, key=lambda value: (scores[value]["log_loss"], -value))
    return selected, scores


def nested_cross_validate(shared, tasks, x1, x3, y, trials, families):
    all_indices = np.arange(len(tasks))
    m1_pen = m1_penalty(x1.shape[1])
    rows = []
    slope_rows = []
    lambda_score_rows = []
    for repeat in range(20):
        predictions1 = np.zeros(len(tasks), dtype=float)
        predictions3 = np.zeros(len(tasks), dtype=float)
        outer_folds = shared.make_folds(tasks, 42 + repeat, 5)
        for fold_index, test_indices in enumerate(outer_folds):
            test = np.asarray(test_indices, dtype=int)
            train = np.setdiff1d(all_indices, test)
            selected, inner_scores = select_lambda(
                shared,
                tasks,
                x3,
                y,
                trials,
                train,
                len(families),
                10000 + repeat * 10 + fold_index,
            )
            for lambda_value, values in inner_scores.items():
                lambda_score_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold_index,
                        "lambda": lambda_value,
                        "inner_log_loss": values["log_loss"],
                        "inner_brier": values["brier"],
                        "selected": int(lambda_value == selected),
                    }
                )
            beta1 = fit_penalized(x1[train], y[train], trials[train], m1_pen)
            beta3 = fit_penalized(
                x3[train],
                y[train],
                trials[train],
                m3_penalty(len(families), selected),
            )
            predictions1[test] = sigmoid(x1[test] @ beta1)
            predictions3[test] = sigmoid(x3[test] @ beta3)
            slopes = beta3[-len(families) :]
            for family, slope in zip(families, slopes):
                slope_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": fold_index,
                        "selected_lambda": selected,
                        "task_family": family,
                        "ru_slope": float(slope),
                    }
                )
        ll1, brier1 = prediction_metrics(y, trials, predictions1)
        ll3, brier3 = prediction_metrics(y, trials, predictions3)
        rows.append(
            {
                "repeat": repeat,
                "m1_log_loss": ll1,
                "m3_log_loss": ll3,
                "delta_log_loss_m3_minus_m1": ll3 - ll1,
                "m1_brier": brier1,
                "m3_brier": brier3,
                "delta_brier_m3_minus_m1": brier3 - brier1,
            }
        )
    return rows, slope_rows, lambda_score_rows


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
    parser.add_argument("--slope-csv", type=Path, required=True)
    parser.add_argument("--lambda-csv", type=Path, required=True)
    args = parser.parse_args()

    shared = load_shared_module(args.shared_script.resolve())
    tasks = shared.load_tasks(shared.extended_path(args.results_root))
    x1, x3, families, baseline = design_matrices(tasks)
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)

    cv_rows, slope_rows, lambda_rows = nested_cross_validate(
        shared, tasks, x1, x3, y, trials, families
    )
    selected_lambdas = [row["selected_lambda"] for row in slope_rows[:: len(families)]]
    lambda_counts = Counter(selected_lambdas)
    modal_lambda = max(LAMBDA_GRID, key=lambda value: (lambda_counts[value], value))
    full_beta = fit_penalized(
        x3, y, trials, m3_penalty(len(families), modal_lambda)
    )
    full_slopes = full_beta[-len(families) :]

    metric_names = [
        "m1_log_loss",
        "m3_log_loss",
        "delta_log_loss_m3_minus_m1",
        "m1_brier",
        "m3_brier",
        "delta_brier_m3_minus_m1",
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
    slope_summary = {}
    for family, full_slope in zip(families, full_slopes):
        values = np.asarray(
            [row["ru_slope"] for row in slope_rows if row["task_family"] == family]
        )
        slope_summary[family] = {
            "full_data_slope_at_modal_lambda": float(full_slope),
            "outer_training_slope_mean": float(values.mean()),
            "outer_training_slope_sd": float(values.std(ddof=1)),
            "outer_training_slope_min": float(values.min()),
            "outer_training_slope_max": float(values.max()),
            "positive_fraction": float(np.mean(values > 0.0)),
        }

    summary = {
        "task_count": len(tasks),
        "trial_count": int(trials.sum()),
        "family_baseline": baseline,
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_selection_criterion": "inner 5-fold binomial log loss",
        "outer_cv": {"folds": 5, "repeats": 20},
        "inner_cv_folds": 5,
        "interaction_parameterization": (
            "family slopes penalized toward their unweighted common mean"
        ),
        "weak_numerical_ridge": WEAK_RIDGE,
        "selected_lambda_counts_across_outer_folds": {
            str(value): lambda_counts[value] for value in LAMBDA_GRID
        },
        "modal_selected_lambda": modal_lambda,
        "cross_validation": {
            "summary": cv_summary,
            "m3_better_log_loss_repeats": sum(
                row["delta_log_loss_m3_minus_m1"] < 0 for row in cv_rows
            ),
            "m3_better_brier_repeats": sum(
                row["delta_brier_m3_minus_m1"] < 0 for row in cv_rows
            ),
        },
        "family_ru_slopes": slope_summary,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.cv_csv, cv_rows)
    write_csv(args.slope_csv, slope_rows)
    write_csv(args.lambda_csv, lambda_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
