"""Chronological prequential replay of online RU/difficulty adaptation.

Each run-policy directory is treated as an isolated stream.  The script never
shares outcomes, residuals, or fitted online state between streams.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
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

OFFLINE_DIRS = {
    "offline_run1": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run1",
    "offline_run2": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run2",
    "offline_run3": "valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run3",
}

METHODS = ("Static", "Online", "Prior+Online")
EPSILON = 1e-8
PROBABILITY_EPSILON = 1e-6
ONLINE_PRIOR_PRECISION = 0.04  # N(0, 5^2), only stabilizes sparse early fits.


def extended_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    output = np.empty_like(values)
    positive = values >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def fit_map_logistic(x, y, prior_mean, prior_precision, initial=None):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    prior_precision = np.asarray(prior_precision, dtype=float)
    beta = prior_mean.copy() if initial is None else np.asarray(initial, dtype=float).copy()
    for _ in range(100):
        probability = sigmoid(x @ beta)
        weights = np.maximum(probability * (1.0 - probability), 1e-9)
        gradient = x.T @ (y - probability) - prior_precision @ (beta - prior_mean)
        information = (x.T * weights) @ x + prior_precision
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def retrieval_utility(row):
    configured_threshold = row.get("parameters", {}).get("score_threshold")
    threshold = 0.5 if configured_threshold is None else float(configured_threshold)
    return float(
        sum(
            max(0.0, threshold - float(memory["score"]))
            for memory in row.get("retrieved_memories", [])
        )
    )


def load_offline_rows(results_root):
    rows = []
    for run, directory in OFFLINE_DIRS.items():
        path = results_root / directory / "memory" / "results.jsonl"
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append(
                    {
                        "run": run,
                        "task_id": row["task_id"],
                        "query": row["query"],
                        "ru": retrieval_utility(row),
                        "success": int(bool(row["reward"])),
                    }
                )
    return rows


def load_online_streams(results_root):
    streams = {}
    for run, run_directory in RUN_DIRS.items():
        for policy, policy_directory in POLICIES.items():
            path = results_root / run_directory / policy_directory / "results.jsonl"
            rows = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    rows.append(
                        {
                            "run": run,
                            "policy": policy,
                            "task_id": row["task_id"],
                            "task_index": int(row["task_index"]),
                            "query": row["query"],
                            "interval": int(row["interval_id"]),
                            "ru": retrieval_utility(row),
                            "success": int(bool(row["reward"])),
                        }
                    )
            rows.sort(key=lambda item: item["task_index"])
            if len(rows) != 134 or [row["task_index"] for row in rows] != list(range(134)):
                raise ValueError(f"Unexpected task order in {run}/{policy}")
            streams[(run, policy)] = rows
    return streams


def load_embeddings(path):
    cache = np.load(path, allow_pickle=False)
    embeddings = np.asarray(cache["embeddings"], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, EPSILON)
    return {
        str(task_id): embedding
        for task_id, embedding in zip(cache["task_ids"], embeddings)
    }, str(cache["model"])


def topk_difficulty(current, history, embedding_by_task, k=3):
    eligible = [
        item
        for item in history
        if item["task_id"] != current["task_id"] and item["query"] != current["query"]
    ]
    if not eligible:
        return 0.0, 0, None
    current_embedding = embedding_by_task[current["task_id"]]
    distances = np.asarray(
        [
            float(np.sum((current_embedding - embedding_by_task[item["task_id"]]) ** 2))
            for item in eligible
        ],
        dtype=float,
    )
    count = min(k, len(eligible))
    positions = np.argpartition(distances, count - 1)[:count]
    positions = positions[np.argsort(distances[positions])]
    nearest_distances = distances[positions]
    weights = 1.0 / (nearest_distances + EPSILON)
    residuals = np.asarray([eligible[position]["residual"] for position in positions])
    score = float(np.dot(weights, residuals) / weights.sum())
    return score, count, float(nearest_distances.mean())


def probability_metrics(rows):
    if not rows:
        return {"count": 0, "log_loss": None, "brier": None}
    outcomes = np.asarray([row["success"] for row in rows], dtype=float)
    probabilities = np.clip(
        np.asarray([row["probability"] for row in rows], dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return {
        "count": len(rows),
        "log_loss": float(
            np.mean(-outcomes * np.log(probabilities) - (1.0 - outcomes) * np.log(1.0 - probabilities))
        ),
        "brier": float(np.mean((probabilities - outcomes) ** 2)),
    }


def fit_difficulty_coefficient(base_logits, scores, outcomes, initial=0.0):
    """Fit a one-parameter logit correction from past stream records only."""
    coefficient = float(initial)
    base_logits = np.asarray(base_logits, dtype=float)
    scores = np.asarray(scores, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    for _ in range(100):
        probability = sigmoid(base_logits + coefficient * scores)
        gradient = float(
            np.dot(scores, outcomes - probability)
            - ONLINE_PRIOR_PRECISION * coefficient
        )
        information = float(
            np.dot(scores * scores, probability * (1.0 - probability))
            + ONLINE_PRIOR_PRECISION
        )
        step = gradient / information
        coefficient += step
        if abs(step) < 1e-10:
            break
    return coefficient


def replay_stream(rows, embedding_by_task, offline_beta, offline_precision):
    predictions = []
    state = {
        "Online": {
            "beta": np.zeros(2),
            "difficulty_coefficient": 0.0,
            "history": [],
            "calibration": [],
        },
        "Prior+Online": {
            "beta": offline_beta.copy(),
            "difficulty_coefficient": 0.0,
            "history": [],
            "calibration": [],
        },
    }
    observed_x = []
    observed_y = []

    for task in rows:
        x_current = np.asarray([1.0, task["ru"]], dtype=float)
        static_probability = float(sigmoid(x_current @ offline_beta))
        predictions.append(
            {
                **task,
                "method": "Static",
                "base_probability": static_probability,
                "difficulty_score": 0.0,
                "difficulty_coefficient": 0.0,
                "neighbor_count": 0,
                "mean_neighbor_distance": None,
                "probability": static_probability,
            }
        )

        current_dynamic = {}
        for method in ("Online", "Prior+Online"):
            base_logit = float(x_current @ state[method]["beta"])
            base_probability = float(sigmoid(base_logit))
            difficulty, neighbor_count, mean_distance = topk_difficulty(
                task, state[method]["history"], embedding_by_task, k=3
            )
            probability = float(
                sigmoid(
                    base_logit
                    + state[method]["difficulty_coefficient"] * difficulty
                )
            )
            predictions.append(
                {
                    **task,
                    "method": method,
                    "base_probability": base_probability,
                    "difficulty_score": difficulty,
                    "difficulty_coefficient": state[method]["difficulty_coefficient"],
                    "neighbor_count": neighbor_count,
                    "mean_neighbor_distance": mean_distance,
                    "probability": probability,
                }
            )
            current_dynamic[method] = {
                "residual": float(task["success"] - base_probability),
                "base_logit": base_logit,
                "difficulty_score": difficulty,
            }

        # Observe once, then update both isolated method states.
        observed_x.append(x_current)
        observed_y.append(float(task["success"]))
        x_history = np.asarray(observed_x, dtype=float)
        y_history = np.asarray(observed_y, dtype=float)
        online_precision = np.eye(2, dtype=float) * ONLINE_PRIOR_PRECISION
        state["Online"]["beta"] = fit_map_logistic(
            x_history,
            y_history,
            prior_mean=np.zeros(2),
            prior_precision=online_precision,
            initial=state["Online"]["beta"],
        )
        state["Prior+Online"]["beta"] = fit_map_logistic(
            x_history,
            y_history,
            prior_mean=offline_beta,
            prior_precision=offline_precision,
            initial=state["Prior+Online"]["beta"],
        )
        for method in ("Online", "Prior+Online"):
            state[method]["history"].append(
                {
                    "task_id": task["task_id"],
                    "query": task["query"],
                    "residual": current_dynamic[method]["residual"],
                }
            )
            state[method]["calibration"].append(
                {
                    "base_logit": current_dynamic[method]["base_logit"],
                    "difficulty_score": current_dynamic[method]["difficulty_score"],
                    "success": float(task["success"]),
                }
            )
            state[method]["difficulty_coefficient"] = fit_difficulty_coefficient(
                [item["base_logit"] for item in state[method]["calibration"]],
                [item["difficulty_score"] for item in state[method]["calibration"]],
                [item["success"] for item in state[method]["calibration"]],
                initial=state[method]["difficulty_coefficient"],
            )
    return predictions


def summarize(predictions):
    overall = []
    interval = []
    stream = []
    for method in METHODS:
        selected = [row for row in predictions if row["method"] == method]
        overall.append({"method": method, **probability_metrics(selected)})
        for interval_id in sorted({row["interval"] for row in selected}):
            subset = [row for row in selected if row["interval"] == interval_id]
            interval.append(
                {"method": method, "interval": interval_id, **probability_metrics(subset)}
            )
    for run in RUN_DIRS:
        for policy in POLICIES:
            for method in METHODS:
                subset = [
                    row
                    for row in predictions
                    if row["run"] == run and row["policy"] == policy and row["method"] == method
                ]
                stream.append(
                    {"run": run, "policy": policy, "method": method, **probability_metrics(subset)}
                )
    return overall, interval, stream


def difficulty_ablation(predictions):
    """Compare each dynamic method with its same prequential RU state at gamma=0."""
    overall = []
    interval = []
    for method in ("Online", "Prior+Online"):
        selected = [row for row in predictions if row["method"] == method]
        with_score = probability_metrics(selected)
        base_rows = [{**row, "probability": row["base_probability"]} for row in selected]
        without_score = probability_metrics(base_rows)
        overall.append(
            {
                "method": method,
                "without_difficulty_log_loss": without_score["log_loss"],
                "with_difficulty_log_loss": with_score["log_loss"],
                "delta_log_loss_with_minus_without": (
                    with_score["log_loss"] - without_score["log_loss"]
                ),
                "without_difficulty_brier": without_score["brier"],
                "with_difficulty_brier": with_score["brier"],
                "delta_brier_with_minus_without": (
                    with_score["brier"] - without_score["brier"]
                ),
            }
        )
        for interval_id in sorted({row["interval"] for row in selected}):
            subset = [row for row in selected if row["interval"] == interval_id]
            subset_base = [
                {**row, "probability": row["base_probability"]} for row in subset
            ]
            with_interval = probability_metrics(subset)
            without_interval = probability_metrics(subset_base)
            interval.append(
                {
                    "method": method,
                    "interval": interval_id,
                    "delta_log_loss_with_minus_without": (
                        with_interval["log_loss"] - without_interval["log_loss"]
                    ),
                    "delta_brier_with_minus_without": (
                        with_interval["brier"] - without_interval["brier"]
                    ),
                }
            )
    return overall, interval


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--interval-csv", type=Path, required=True)
    parser.add_argument("--stream-csv", type=Path, required=True)
    args = parser.parse_args()

    results_root = extended_path(args.results_root)
    offline_rows = load_offline_rows(results_root)
    offline_x = np.asarray([[1.0, row["ru"]] for row in offline_rows], dtype=float)
    offline_y = np.asarray([row["success"] for row in offline_rows], dtype=float)
    weak_precision = np.eye(2, dtype=float) * 1e-4
    offline_beta = fit_map_logistic(
        offline_x, offline_y, np.zeros(2), weak_precision
    )
    offline_probability = sigmoid(offline_x @ offline_beta)
    offline_precision = (
        offline_x.T * np.maximum(offline_probability * (1.0 - offline_probability), 1e-9)
    ) @ offline_x + weak_precision

    embedding_by_task, embedding_model = load_embeddings(args.embedding_cache)
    streams = load_online_streams(results_root)
    all_task_ids = {row["task_id"] for rows in streams.values() for row in rows}
    missing_embeddings = sorted(all_task_ids - set(embedding_by_task))
    if missing_embeddings:
        raise ValueError(f"Missing {len(missing_embeddings)} online task embeddings")

    predictions = []
    for rows in streams.values():
        predictions.extend(
            replay_stream(rows, embedding_by_task, offline_beta, offline_precision)
        )
    overall, interval, stream = summarize(predictions)
    ablation_overall, ablation_interval = difficulty_ablation(predictions)

    stream_by_key = {
        (row["run"], row["policy"], row["method"]): row for row in stream
    }
    win_counts = {}
    for metric in ("log_loss", "brier"):
        win_counts[metric] = {}
        for method in ("Online", "Prior+Online"):
            win_counts[metric][f"{method}_better_than_Static"] = sum(
                stream_by_key[(run, policy, method)][metric]
                < stream_by_key[(run, policy, "Static")][metric]
                for run in RUN_DIRS
                for policy in POLICIES
            )
        win_counts[metric]["Prior+Online_better_than_Online"] = sum(
            stream_by_key[(run, policy, "Prior+Online")][metric]
            < stream_by_key[(run, policy, "Online")][metric]
            for run in RUN_DIRS
            for policy in POLICIES
        )

    dynamic_predictions = [
        row for row in predictions if row["method"] in ("Online", "Prior+Online")
    ]
    summary = {
        "stream_count": len(streams),
        "tasks_per_stream": 134,
        "prediction_count": len(predictions),
        "offline_prior_trial_count": len(offline_rows),
        "offline_prior_coefficients": {
            "intercept": float(offline_beta[0]),
            "ru": float(offline_beta[1]),
        },
        "embedding_model": embedding_model,
        "difficulty": {
            "k": 3,
            "distance": "squared Euclidean on normalized query embeddings",
            "weight": "1 / (distance + 1e-8)",
            "exact_same_query_excluded": True,
            "correction": (
                "sigmoid(base_logit + gamma * DifficultyScore); gamma is fitted "
                "only from earlier observations in the same stream"
            ),
            "dynamic_rows_with_three_neighbors": sum(
                row["neighbor_count"] == 3 for row in dynamic_predictions
            ),
            "dynamic_row_count": len(dynamic_predictions),
        },
        "online_update": {
            "protocol": "predict -> observe success -> refit RU logistic state",
            "online_only_prior": "N([0,0], 5^2 I)",
            "prior_plus_online_prior": "offline RU-SR Laplace posterior",
            "state_shared_across_streams": False,
        },
        "overall": overall,
        "by_interval": interval,
        "difficulty_ablation": {
            "interpretation": "positive delta means DifficultyScore worsened the metric",
            "overall": ablation_overall,
            "by_interval": ablation_interval,
        },
        "stream_win_counts_out_of_16": win_counts,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.predictions_csv, predictions)
    write_csv(args.overall_csv, overall)
    write_csv(args.interval_csv, interval)
    write_csv(args.stream_csv, stream)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
