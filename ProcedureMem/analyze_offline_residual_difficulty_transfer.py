"""Test nearest-query transfer of cross-fitted execution residuals."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


K_VALUES = (3, 5, 10)
WEAK_RIDGE = 1e-4
EPSILON = 1e-8


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def subset_query_folds(recovery, tasks, indices, seed, fold_count=5):
    local_tasks = [tasks[index] for index in indices]
    local_groups = recovery.build_query_groups(local_tasks)
    query_folds = recovery.make_query_group_folds(
        local_tasks, local_groups, seed, fold_count
    )
    output = []
    for query_set in query_folds:
        output.append(
            np.asarray(
                [index for index in indices if tasks[index]["query"] in query_set],
                dtype=int,
            )
        )
    return output


def m0_matrix(tasks, indices):
    return np.asarray([[1.0, tasks[index]["ru"]] for index in indices], dtype=float)


def mf_matrix(tasks, indices, families):
    return np.asarray(
        [
            [1.0, tasks[index]["ru"]]
            + [
                float(tasks[index]["task_family"] == family)
                for family in families[1:]
            ]
            for index in indices
        ],
        dtype=float,
    )


def md_matrix(tasks, indices, difficulty_scores):
    return np.asarray(
        [
            [1.0, tasks[index]["ru"], float(difficulty_scores[position])]
            for position, index in enumerate(indices)
        ],
        dtype=float,
    )


def cross_fitted_residuals(
    shared, recovery, tasks, indices, y, trials, seed, fold_count=5
):
    predictions = {}
    folds = subset_query_folds(recovery, tasks, list(indices), seed, fold_count)
    index_array = np.asarray(indices, dtype=int)
    for held_out in folds:
        reference = np.setdiff1d(index_array, held_out)
        beta = shared.fit_logistic(
            m0_matrix(tasks, reference),
            y[reference],
            trials[reference],
            ridge=WEAK_RIDGE,
        )[0]
        probability = shared.sigmoid(m0_matrix(tasks, held_out) @ beta)
        for index, value in zip(held_out, probability):
            predictions[int(index)] = float(value)
    if set(predictions) != set(map(int, indices)):
        raise ValueError("Cross-fitting did not predict every reference task")
    return {
        int(index): float(y[index] / trials[index] - predictions[int(index)])
        for index in indices
    }


def difficulty_scores(
    cluster_module, tasks, target_indices, reference_indices, residuals, k
):
    reference_indices = np.asarray(reference_indices, dtype=int)
    target_indices = np.asarray(target_indices, dtype=int)
    reference_embeddings = np.asarray(
        [tasks[index]["embedding"] for index in reference_indices], dtype=float
    )
    target_embeddings = np.asarray(
        [tasks[index]["embedding"] for index in target_indices], dtype=float
    )
    distances = cluster_module.squared_distances(
        target_embeddings, reference_embeddings
    )
    scores = []
    neighbor_distance_rows = []
    for target_position, target_index in enumerate(target_indices):
        same_query = np.asarray(
            [
                tasks[index]["query"] == tasks[target_index]["query"]
                for index in reference_indices
            ]
        )
        eligible_positions = np.flatnonzero(~same_query)
        if len(eligible_positions) < k:
            raise ValueError(f"Fewer than {k} eligible neighbors")
        eligible_distances = distances[target_position, eligible_positions]
        nearest_local = np.argpartition(eligible_distances, k - 1)[:k]
        nearest_positions = eligible_positions[nearest_local]
        nearest_positions = nearest_positions[
            np.argsort(distances[target_position, nearest_positions])
        ]
        nearest_distances = distances[target_position, nearest_positions]
        weights = 1.0 / (nearest_distances + EPSILON)
        neighbor_residuals = np.asarray(
            [residuals[int(reference_indices[position])] for position in nearest_positions]
        )
        scores.append(float(np.dot(weights, neighbor_residuals) / weights.sum()))
        neighbor_distance_rows.append(
            {
                "mean": float(nearest_distances.mean()),
                "minimum": float(nearest_distances.min()),
                "maximum": float(nearest_distances.max()),
            }
        )
    return np.asarray(scores, dtype=float), neighbor_distance_rows


def outer_training_difficulty(
    shared,
    recovery,
    cluster_module,
    tasks,
    train_indices,
    y,
    trials,
    seed,
):
    scores = {k: {} for k in K_VALUES}
    inner_folds = subset_query_folds(
        recovery, tasks, list(train_indices), seed, fold_count=5
    )
    train_array = np.asarray(train_indices, dtype=int)
    for inner_fold_index, held_out in enumerate(inner_folds):
        reference = np.setdiff1d(train_array, held_out)
        reference_residuals = cross_fitted_residuals(
            shared,
            recovery,
            tasks,
            reference,
            y,
            trials,
            seed + 100 + inner_fold_index,
        )
        for k in K_VALUES:
            values, _ = difficulty_scores(
                cluster_module,
                tasks,
                held_out,
                reference,
                reference_residuals,
                k,
            )
            for index, value in zip(held_out, values):
                scores[k][int(index)] = float(value)
    for k in K_VALUES:
        if set(scores[k]) != set(map(int, train_indices)):
            raise ValueError(f"Missing outer-training DifficultyScore for K={k}")
    return {
        k: np.asarray([scores[k][int(index)] for index in train_indices], dtype=float)
        for k in K_VALUES
    }


def repeated_cv(shared, recovery, cluster_module, tasks, seed):
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)
    families = sorted({row["task_family"] for row in tasks})
    groups = recovery.build_query_groups(tasks)
    all_indices = np.arange(len(tasks))
    rows = []
    fold_rows = []
    coefficient_rows = []

    for repeat in range(20):
        outer_query_folds = recovery.make_query_group_folds(
            tasks, groups, seed + repeat, fold_count=5
        )
        prediction0 = np.zeros(len(tasks), dtype=float)
        predictionf = np.zeros(len(tasks), dtype=float)
        predictiond = {k: np.zeros(len(tasks), dtype=float) for k in K_VALUES}

        for outer_fold_index, test_queries in enumerate(outer_query_folds):
            test = np.asarray(
                [index for index, row in enumerate(tasks) if row["query"] in test_queries],
                dtype=int,
            )
            train = np.setdiff1d(all_indices, test)
            train_queries = {tasks[index]["query"] for index in train}
            if train_queries & test_queries:
                raise ValueError("Outer unique-query leakage detected")

            beta0 = shared.fit_logistic(
                m0_matrix(tasks, train), y[train], trials[train], ridge=WEAK_RIDGE
            )[0]
            betaf = shared.fit_logistic(
                mf_matrix(tasks, train, families),
                y[train],
                trials[train],
                ridge=WEAK_RIDGE,
            )[0]
            prediction0[test] = shared.sigmoid(m0_matrix(tasks, test) @ beta0)
            predictionf[test] = shared.sigmoid(
                mf_matrix(tasks, test, families) @ betaf
            )

            training_scores = outer_training_difficulty(
                shared,
                recovery,
                cluster_module,
                tasks,
                train,
                y,
                trials,
                seed + repeat * 10000 + outer_fold_index * 1000,
            )
            train_residuals = cross_fitted_residuals(
                shared,
                recovery,
                tasks,
                train,
                y,
                trials,
                seed + repeat * 10000 + outer_fold_index * 1000 + 900,
            )
            for k in K_VALUES:
                test_scores, distance_rows = difficulty_scores(
                    cluster_module,
                    tasks,
                    test,
                    train,
                    train_residuals,
                    k,
                )
                xd_train = md_matrix(tasks, train, training_scores[k])
                xd_test = md_matrix(tasks, test, test_scores)
                betad = shared.fit_logistic(
                    xd_train, y[train], trials[train], ridge=WEAK_RIDGE
                )[0]
                predictiond[k][test] = shared.sigmoid(xd_test @ betad)
                coefficient_rows.append(
                    {
                        "repeat": repeat,
                        "fold": outer_fold_index,
                        "k": k,
                        "difficulty_coefficient": float(betad[-1]),
                    }
                )
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": outer_fold_index,
                        "k": k,
                        "train_unique_query_count": len(train_queries),
                        "test_unique_query_count": len(test_queries),
                        "train_task_count": len(train),
                        "test_task_count": len(test),
                        "query_overlap_count": 0,
                        "test_difficulty_mean": float(test_scores.mean()),
                        "test_difficulty_sd": float(test_scores.std(ddof=1)),
                        "neighbor_distance_mean": float(
                            np.mean([row["mean"] for row in distance_rows])
                        ),
                    }
                )

        m0_log, m0_brier = shared.prediction_metrics(
            y, trials, prediction0
        )
        mf_log, mf_brier = shared.prediction_metrics(
            y, trials, predictionf
        )
        for k in K_VALUES:
            md_log, md_brier = shared.prediction_metrics(
                y, trials, predictiond[k]
            )
            rows.append(
                {
                    "repeat": repeat,
                    "k": k,
                    "m0_log_loss": m0_log,
                    "md_log_loss": md_log,
                    "mf_log_loss": mf_log,
                    "m0_brier": m0_brier,
                    "md_brier": md_brier,
                    "mf_brier": mf_brier,
                }
            )
    return rows, fold_rows, coefficient_rows, len(groups)


def metric(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize(rows, coefficient_rows):
    output = {}
    for k in K_VALUES:
        selected = [row for row in rows if row["k"] == k]
        selected_coefficients = [
            row["difficulty_coefficient"]
            for row in coefficient_rows
            if row["k"] == k
        ]
        result = {
            name: metric([row[name] for row in selected])
            for name in (
                "m0_log_loss",
                "md_log_loss",
                "mf_log_loss",
                "m0_brier",
                "md_brier",
                "mf_brier",
            )
        }
        result["mean_log_loss_improvement_md_vs_m0"] = (
            result["m0_log_loss"]["mean"] - result["md_log_loss"]["mean"]
        )
        result["mean_brier_improvement_md_vs_m0"] = (
            result["m0_brier"]["mean"] - result["md_brier"]["mean"]
        )
        result["md_better_log_loss_repeats"] = sum(
            row["md_log_loss"] < row["m0_log_loss"] for row in selected
        )
        result["md_better_brier_repeats"] = sum(
            row["md_brier"] < row["m0_brier"] for row in selected
        )
        result["difficulty_coefficient"] = metric(selected_coefficients)
        result["difficulty_coefficient_positive_fraction"] = float(
            np.mean(np.asarray(selected_coefficients) > 0.0)
        )
        output[str(k)] = result
    return output


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--shared-script", type=Path, required=True)
    parser.add_argument("--recovery-script", type=Path, required=True)
    parser.add_argument("--cluster-script", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--cv-csv", type=Path, required=True)
    parser.add_argument("--fold-csv", type=Path, required=True)
    parser.add_argument("--coefficient-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shared = load_module("family_ru_shared", args.shared_script.resolve())
    recovery = load_module("cluster_recovery_shared", args.recovery_script.resolve())
    cluster_module = load_module("task_cluster_shared", args.cluster_script.resolve())
    tasks, embedding_model, dimension = recovery.prepare_tasks(
        shared,
        args.results_root,
        args.results_jsonl,
        args.embedding_cache.resolve(),
    )
    rows, fold_rows, coefficient_rows, unique_query_count = repeated_cv(
        shared, recovery, cluster_module, tasks, args.seed
    )
    summary = {
        "task_count": len(tasks),
        "unique_query_count": unique_query_count,
        "trial_count": 3 * len(tasks),
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "k_values": list(K_VALUES),
        "distance": "squared Euclidean on normalized BGE query embeddings",
        "weight": "1 / (distance + 1e-8)",
        "residual": "actual success_count/3 - cross-fitted M0 probability",
        "cv": {
            "folds": 5,
            "repeats": 20,
            "outer_group": "exact query text",
            "outer_query_overlap": max(
                row["query_overlap_count"] for row in fold_rows
            ),
            "training_score": (
                "nested out-of-fold DifficultyScore; held-out query outcomes are not "
                "used by its reference residual pool"
            ),
            "test_score": "outer-training cross-fitted residual pool only",
            "weak_logistic_ridge": WEAK_RIDGE,
        },
        "results": summarize(rows, coefficient_rows),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.cv_csv, rows)
    write_csv(args.fold_csv, fold_rows)
    write_csv(args.coefficient_csv, coefficient_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
