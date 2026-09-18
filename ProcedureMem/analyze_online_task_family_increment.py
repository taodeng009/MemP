"""Incremental value of TaskFamily in online run4-run7 success prediction."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_DIRS = {
    "run4": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run4",
    "run5": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run5",
    "run6": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run6",
    "run7": "online_construction_valid_unseen_seed42_n134_b2_i20_c3_mbQwen3.6-27B_agentbi1_run7",
}
POLICIES = {
    "FIFO": "online_construction_fifo_shortest_first",
    "Coverage": "online_construction_oracle_coverage",
    "Exact": "online_construction_oracle_exact_retrieval_h1",
    "HitQuality": "online_construction_oracle_hit_quality_ru_h1_alpha0.25",
}
WEAK_RIDGE = 1e-4


def load_shared(path: Path):
    spec = importlib.util.spec_from_file_location("family_ru_shared", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extended_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def load_records(root: Path):
    records = []
    for run, run_directory in RUN_DIRS.items():
        for policy, policy_directory in POLICIES.items():
            path = root / run_directory / policy_directory / "results.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    threshold = float(row["parameters"]["score_threshold"])
                    ru = sum(
                        max(0.0, threshold - float(memory["score"]))
                        for memory in row.get("retrieved_memories", [])
                    )
                    records.append(
                        {
                            "task_id": row["task_id"],
                            "task_family": row["task_type"].split("-", 1)[0],
                            "policy": policy,
                            "run": run,
                            "interval": str(int(row["interval_id"])),
                            "ru": ru,
                            "success": int(bool(row["reward"])),
                        }
                    )
    return records


def dummy_matrix(records, field, levels):
    return np.asarray(
        [[float(row[field] == level) for level in levels[1:]] for row in records],
        dtype=float,
    )


def design_matrices(records):
    families = sorted({row["task_family"] for row in records})
    policies = list(POLICIES)
    runs = list(RUN_DIRS)
    intervals = sorted({row["interval"] for row in records}, key=int)
    intercept = np.ones((len(records), 1), dtype=float)
    ru = np.asarray([row["ru"] for row in records], dtype=float)[:, None]
    controls = np.hstack(
        [
            dummy_matrix(records, "policy", policies),
            dummy_matrix(records, "run", runs),
            dummy_matrix(records, "interval", intervals),
        ]
    )
    family = dummy_matrix(records, "task_family", families)
    x0 = np.hstack([intercept, ru, controls])
    x1 = np.hstack([intercept, ru, family, controls])
    metadata = {
        "families": families,
        "policies": policies,
        "runs": runs,
        "intervals": intervals,
        "m0_parameter_count": x0.shape[1],
        "m1_parameter_count": x1.shape[1],
    }
    return x0, x1, metadata


def make_group_folds(records, seed, fold_count=5):
    by_task = defaultdict(list)
    for index, row in enumerate(records):
        by_task[row["task_id"]].append(index)
    by_family = defaultdict(list)
    for task_id, indices in by_task.items():
        first = records[indices[0]]
        successes = sum(records[index]["success"] for index in indices)
        by_family[first["task_family"]].append((task_id, successes))
    rng = random.Random(seed)
    fold_tasks = [set() for _ in range(fold_count)]
    # Balance task count and task-level success totals separately within each family.
    # Random shuffling before sorting randomizes ties across repeated CV partitions.
    for family_tasks in by_family.values():
        rng.shuffle(family_tasks)
        family_tasks.sort(key=lambda item: item[1], reverse=True)
        family_counts = [0] * fold_count
        family_successes = [0] * fold_count
        for task_id, successes in family_tasks:
            minimum_count = min(family_counts)
            candidates = [
                fold for fold in range(fold_count) if family_counts[fold] == minimum_count
            ]
            minimum_success = min(family_successes[fold] for fold in candidates)
            candidates = [
                fold for fold in candidates if family_successes[fold] == minimum_success
            ]
            selected_fold = rng.choice(candidates)
            fold_tasks[selected_fold].add(task_id)
            family_counts[selected_fold] += 1
            family_successes[selected_fold] += successes
    folds = []
    for selected in fold_tasks:
        folds.append(
            [index for index, row in enumerate(records) if row["task_id"] in selected]
        )
    return folds


def repeated_grouped_cv(shared, records, x0, x1, y, trials):
    all_indices = np.arange(len(records))
    rows = []
    leakage_checks = []
    for repeat in range(20):
        prediction0 = np.zeros(len(records), dtype=float)
        prediction1 = np.zeros(len(records), dtype=float)
        folds = make_group_folds(records, 42 + repeat, 5)
        for fold_index, test_indices in enumerate(folds):
            test = np.asarray(test_indices, dtype=int)
            train = np.setdiff1d(all_indices, test)
            train_tasks = {records[index]["task_id"] for index in train}
            test_tasks = {records[index]["task_id"] for index in test}
            overlap = train_tasks & test_tasks
            if overlap:
                raise ValueError(f"Task leakage in repeat {repeat}, fold {fold_index}")
            leakage_checks.append(
                {
                    "repeat": repeat,
                    "fold": fold_index,
                    "train_task_count": len(train_tasks),
                    "test_task_count": len(test_tasks),
                    "train_row_count": len(train),
                    "test_row_count": len(test),
                    "task_overlap_count": len(overlap),
                }
            )
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
    return rows, leakage_checks


def metric_summary(rows, names):
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
    parser.add_argument("--fold-csv", type=Path, required=True)
    args = parser.parse_args()

    shared = load_shared(args.shared_script.resolve())
    records = load_records(extended_path(args.results_root))
    x0, x1, metadata = design_matrices(records)
    y = np.asarray([row["success"] for row in records], dtype=float)
    trials = np.ones(len(records), dtype=float)

    task_row_counts = Counter(row["task_id"] for row in records)
    if len(task_row_counts) != 134 or set(task_row_counts.values()) != {16}:
        raise ValueError(
            f"Expected 134 tasks with 16 rows each; got {len(task_row_counts)} tasks "
            f"and row-count values {sorted(set(task_row_counts.values()))}"
        )

    beta0, _ = shared.fit_logistic(x0, y, trials)
    beta1, _ = shared.fit_logistic(x1, y, trials)
    ll0 = shared.log_likelihood(x0, y, trials, beta0)
    ll1 = shared.log_likelihood(x1, y, trials, beta1)
    lr = max(0.0, 2.0 * (ll1 - ll0))
    degrees_freedom = len(metadata["families"]) - 1
    p_value = shared.gammaincc(degrees_freedom / 2.0, lr / 2.0)

    cv_rows, fold_rows = repeated_grouped_cv(
        shared, records, x0, x1, y, trials
    )
    names = [
        "m0_log_loss",
        "m1_log_loss",
        "delta_log_loss_m1_minus_m0",
        "m0_brier",
        "m1_brier",
        "delta_brier_m1_minus_m0",
    ]
    family_counts = {}
    for family in metadata["families"]:
        selected = [row for row in records if row["task_family"] == family]
        family_counts[family] = {
            "task_count": len({row["task_id"] for row in selected}),
            "row_count": len(selected),
            "success_count": sum(row["success"] for row in selected),
            "success_rate": sum(row["success"] for row in selected) / len(selected),
        }

    summary = {
        "record_count": len(records),
        "task_count": len(task_row_counts),
        "rows_per_task": 16,
        "design": {
            **metadata,
            "interval_treated_as_categorical": True,
            "baselines": {
                "task_family": metadata["families"][0],
                "policy": metadata["policies"][0],
                "run": metadata["runs"][0],
                "interval": metadata["intervals"][0],
            },
        },
        "m0": {"log_likelihood": ll0, "ru_coefficient": float(beta0[1])},
        "m1": {"log_likelihood": ll1, "ru_coefficient": float(beta1[1])},
        "likelihood_ratio_test": {
            "statistic": lr,
            "df": degrees_freedom,
            "p_value": p_value,
        },
        "cross_validation": {
            "folds": 5,
            "repeats": 20,
            "group": "task_id",
            "stratification": (
                "greedy balance of task count and total success within task_family"
            ),
            "weak_ridge": WEAK_RIDGE,
            "summary": metric_summary(cv_rows, names),
            "m1_better_log_loss_repeats": sum(
                row["delta_log_loss_m1_minus_m0"] < 0 for row in cv_rows
            ),
            "m1_better_brier_repeats": sum(
                row["delta_brier_m1_minus_m0"] < 0 for row in cv_rows
            ),
            "max_task_overlap": max(row["task_overlap_count"] for row in fold_rows),
            "test_task_count_min": min(row["test_task_count"] for row in fold_rows),
            "test_task_count_max": max(row["test_task_count"] for row in fold_rows),
        },
        "family_outcomes": family_counts,
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
