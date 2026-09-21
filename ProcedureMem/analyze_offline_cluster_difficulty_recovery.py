"""Evaluate how much TaskFamily difficulty signal query clusters recover."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


K_VALUES = (4, 5, 6, 7, 8)
WEAK_RIDGE = 1e-4


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_queries(results_path: Path):
    output = {}
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output[row["task_id"]] = {
                "query": row["query"],
                "task_family": row["task_type"].split("-", 1)[0],
            }
    return output


def load_embedding_cache(path: Path):
    with np.load(path, allow_pickle=False) as cached:
        task_ids = list(cached["task_ids"].astype(str))
        embeddings = np.asarray(cached["embeddings"], dtype=np.float64)
        model = str(cached["model"].item())
    if embeddings.shape[0] != len(task_ids):
        raise ValueError("Embedding cache task IDs and vectors differ in length")
    return dict(zip(task_ids, embeddings)), model, embeddings.shape[1]


def prepare_tasks(shared, results_root, results_path, embedding_path):
    aggregated = shared.load_tasks(shared.extended_path(results_root))
    queries = load_queries(shared.extended_path(results_path))
    embedding_by_id, model, dimension = load_embedding_cache(embedding_path)
    tasks = []
    for row in aggregated:
        task_id = row["task_id"]
        if task_id not in queries or task_id not in embedding_by_id:
            raise ValueError(f"Missing query or embedding for {task_id}")
        metadata = queries[task_id]
        if metadata["task_family"] != row["task_family"]:
            raise ValueError(f"Task-family mismatch for {task_id}")
        tasks.append(
            {
                **row,
                "query": metadata["query"],
                "embedding": embedding_by_id[task_id],
            }
        )
    if len(tasks) != 134:
        raise ValueError(f"Expected 134 tasks, found {len(tasks)}")
    return tasks, model, dimension


def build_query_groups(tasks):
    groups = defaultdict(list)
    for index, row in enumerate(tasks):
        groups[row["query"]].append(index)
    for query, indices in groups.items():
        families = {tasks[index]["task_family"] for index in indices}
        if len(families) != 1:
            raise ValueError(f"Repeated query spans multiple families: {query}")
        reference = tasks[indices[0]]["embedding"]
        if not all(np.array_equal(reference, tasks[index]["embedding"]) for index in indices):
            raise ValueError(f"Repeated query has different embeddings: {query}")
    return dict(groups)


def make_query_group_folds(tasks, groups, seed, fold_count=5):
    by_family = defaultdict(list)
    for query, indices in groups.items():
        family = tasks[indices[0]]["task_family"]
        success_trials = sum(tasks[index]["success_count"] for index in indices)
        by_family[family].append(
            {
                "query": query,
                "task_count": len(indices),
                "success_trials": success_trials,
            }
        )
    rng = random.Random(seed)
    fold_queries = [set() for _ in range(fold_count)]
    for family_groups in by_family.values():
        rng.shuffle(family_groups)
        family_groups.sort(
            key=lambda row: (row["task_count"], row["success_trials"]), reverse=True
        )
        task_totals = [0] * fold_count
        success_totals = [0] * fold_count
        group_totals = [0] * fold_count
        for group in family_groups:
            minimum_tasks = min(task_totals)
            candidates = [
                fold for fold in range(fold_count) if task_totals[fold] == minimum_tasks
            ]
            minimum_success = min(success_totals[fold] for fold in candidates)
            candidates = [
                fold
                for fold in candidates
                if success_totals[fold] == minimum_success
            ]
            minimum_groups = min(group_totals[fold] for fold in candidates)
            candidates = [
                fold for fold in candidates if group_totals[fold] == minimum_groups
            ]
            selected = rng.choice(candidates)
            fold_queries[selected].add(group["query"])
            task_totals[selected] += group["task_count"]
            success_totals[selected] += group["success_trials"]
            group_totals[selected] += 1
    return fold_queries


def base_designs(tasks, families):
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)[:, None]
    intercept = np.ones((len(tasks), 1), dtype=float)
    family_dummy = np.asarray(
        [
            [float(row["task_family"] == family) for family in families[1:]]
            for row in tasks
        ],
        dtype=float,
    )
    return np.hstack([intercept, ru]), np.hstack([intercept, ru, family_dummy])


def cluster_design(ru, labels, k):
    dummy = np.asarray(
        [[float(label == cluster) for cluster in range(1, k)] for label in labels],
        dtype=float,
    )
    return np.hstack([np.ones((len(labels), 1)), ru[:, None], dummy])


def prediction_metrics(shared, y, trials, probability):
    return shared.prediction_metrics(y, trials, probability)


def repeated_grouped_cv(shared, cluster_module, tasks, n_init, seed):
    families = sorted({row["task_family"] for row in tasks})
    groups = build_query_groups(tasks)
    x0, xf = base_designs(tasks, families)
    y = np.asarray([row["success_count"] for row in tasks], dtype=float)
    trials = np.full(len(tasks), 3.0)
    ru = np.asarray([row["ru"] for row in tasks], dtype=float)
    all_indices = np.arange(len(tasks))
    rows = []
    fold_rows = []

    for repeat in range(20):
        fold_queries = make_query_group_folds(tasks, groups, seed + repeat, 5)
        prediction0 = np.zeros(len(tasks), dtype=float)
        predictionf = np.zeros(len(tasks), dtype=float)
        predictionc = {k: np.zeros(len(tasks), dtype=float) for k in K_VALUES}

        for fold_index, test_queries in enumerate(fold_queries):
            test = np.asarray(
                [index for index, row in enumerate(tasks) if row["query"] in test_queries],
                dtype=int,
            )
            train = np.setdiff1d(all_indices, test)
            train_queries = {tasks[index]["query"] for index in train}
            if train_queries & test_queries:
                raise ValueError("Unique-query leakage detected")

            beta0 = shared.fit_logistic(
                x0[train], y[train], trials[train], ridge=WEAK_RIDGE
            )[0]
            betaf = shared.fit_logistic(
                xf[train], y[train], trials[train], ridge=WEAK_RIDGE
            )[0]
            prediction0[test] = shared.sigmoid(x0[test] @ beta0)
            predictionf[test] = shared.sigmoid(xf[test] @ betaf)

            unique_train_queries = sorted(train_queries)
            train_embedding = np.asarray(
                [tasks[groups[query][0]]["embedding"] for query in unique_train_queries]
            )
            test_embedding = np.asarray([tasks[index]["embedding"] for index in test])
            for k in K_VALUES:
                _, centroids, inertia = cluster_module.kmeans(
                    train_embedding,
                    k,
                    seed=seed + repeat * 1000 + fold_index * 100 + k,
                    n_init=n_init,
                )
                train_labels = np.argmin(
                    cluster_module.squared_distances(
                        np.asarray([tasks[index]["embedding"] for index in train]),
                        centroids,
                    ),
                    axis=1,
                )
                test_labels = np.argmin(
                    cluster_module.squared_distances(test_embedding, centroids), axis=1
                )
                xc_train = cluster_design(ru[train], train_labels, k)
                xc_test = cluster_design(ru[test], test_labels, k)
                betac = shared.fit_logistic(
                    xc_train, y[train], trials[train], ridge=WEAK_RIDGE
                )[0]
                predictionc[k][test] = shared.sigmoid(xc_test @ betac)
                fold_rows.append(
                    {
                        "repeat": repeat,
                        "fold": fold_index,
                        "k": k,
                        "train_unique_query_count": len(train_queries),
                        "test_unique_query_count": len(test_queries),
                        "train_task_count": len(train),
                        "test_task_count": len(test),
                        "query_overlap_count": 0,
                        "kmeans_train_inertia": inertia,
                    }
                )

        m0_log, m0_brier = prediction_metrics(
            shared, y, trials, prediction0
        )
        mf_log, mf_brier = prediction_metrics(
            shared, y, trials, predictionf
        )
        for k in K_VALUES:
            mc_log, mc_brier = prediction_metrics(
                shared, y, trials, predictionc[k]
            )
            log_denominator = m0_log - mf_log
            brier_denominator = m0_brier - mf_brier
            rows.append(
                {
                    "repeat": repeat,
                    "k": k,
                    "m0_log_loss": m0_log,
                    "mc_log_loss": mc_log,
                    "mf_log_loss": mf_log,
                    "log_loss_recovery": (m0_log - mc_log) / log_denominator,
                    "m0_brier": m0_brier,
                    "mc_brier": mc_brier,
                    "mf_brier": mf_brier,
                    "brier_recovery": (m0_brier - mc_brier) / brier_denominator,
                }
            )
    return rows, fold_rows, len(groups)


def summarize(rows):
    output = {}
    for k in K_VALUES:
        selected = [row for row in rows if row["k"] == k]
        metrics = {}
        for name in (
            "m0_log_loss",
            "mc_log_loss",
            "mf_log_loss",
            "log_loss_recovery",
            "m0_brier",
            "mc_brier",
            "mf_brier",
            "brier_recovery",
        ):
            values = np.asarray([row[name] for row in selected], dtype=float)
            metrics[name] = {
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        # Primary recovery requested by the user: ratio of mean CV losses.
        metrics["ratio_of_mean_log_loss_improvements"] = (
            metrics["m0_log_loss"]["mean"] - metrics["mc_log_loss"]["mean"]
        ) / (
            metrics["m0_log_loss"]["mean"] - metrics["mf_log_loss"]["mean"]
        )
        metrics["ratio_of_mean_brier_improvements"] = (
            metrics["m0_brier"]["mean"] - metrics["mc_brier"]["mean"]
        ) / (
            metrics["m0_brier"]["mean"] - metrics["mf_brier"]["mean"]
        )
        metrics["mc_better_than_m0_log_loss_repeats"] = sum(
            row["mc_log_loss"] < row["m0_log_loss"] for row in selected
        )
        metrics["mc_better_than_mf_log_loss_repeats"] = sum(
            row["mc_log_loss"] < row["mf_log_loss"] for row in selected
        )
        output[str(k)] = metrics
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
    parser.add_argument("--cluster-script", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--cv-csv", type=Path, required=True)
    parser.add_argument("--fold-csv", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=100)
    args = parser.parse_args()

    shared = load_module("family_ru_shared", args.shared_script.resolve())
    cluster_module = load_module("task_cluster_shared", args.cluster_script.resolve())
    tasks, embedding_model, dimension = prepare_tasks(
        shared,
        args.results_root,
        args.results_jsonl,
        args.embedding_cache.resolve(),
    )
    cv_rows, fold_rows, unique_query_count = repeated_grouped_cv(
        shared, cluster_module, tasks, args.n_init, args.seed
    )
    summary = {
        "task_count": len(tasks),
        "unique_query_count": unique_query_count,
        "trial_count": 3 * len(tasks),
        "embedding_model": embedding_model,
        "embedding_dimension": dimension,
        "k_values": list(K_VALUES),
        "cv": {
            "folds": 5,
            "repeats": 20,
            "group": "exact query text",
            "same_query_overlap": max(row["query_overlap_count"] for row in fold_rows),
            "kmeans_fit_scope": "unique query embeddings in each training fold only",
            "test_assignment": "nearest training-fold centroid by squared Euclidean distance",
            "kmeans_n_init": args.n_init,
            "weak_logistic_ridge": WEAK_RIDGE,
        },
        "results": summarize(cv_rows),
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.cv_csv, cv_rows)
    write_csv(args.fold_csv, fold_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
