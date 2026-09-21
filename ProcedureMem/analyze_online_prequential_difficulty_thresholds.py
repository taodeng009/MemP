"""Test squared-L2 thresholds for prequential Top-3 residual transfer."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np


THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
VARIANTS = ("no_difficulty", "top3_all", "tau_0.1", "tau_0.2", "tau_0.3", "tau_0.5")


def load_base(path: Path):
    spec = importlib.util.spec_from_file_location("prequential_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def threshold_for_variant(variant):
    if variant == "top3_all":
        return None
    if variant.startswith("tau_"):
        return float(variant.split("_", 1)[1])
    raise ValueError(variant)


def topk_difficulty(base, current, history, embedding_by_task, threshold):
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
    top_count = min(3, len(eligible))
    positions = np.argpartition(distances, top_count - 1)[:top_count]
    positions = positions[np.argsort(distances[positions])]
    if threshold is not None:
        positions = positions[distances[positions] <= threshold]
    if len(positions) == 0:
        return 0.0, 0, None
    nearest_distances = distances[positions]
    weights = 1.0 / (nearest_distances + base.EPSILON)
    residuals = np.asarray([eligible[position]["residual"] for position in positions])
    score = float(np.dot(weights, residuals) / weights.sum())
    return score, len(positions), float(nearest_distances.mean())


def replay_stream(base, rows, embedding_by_task, offline_beta, offline_precision):
    predictions = []
    state = {}
    for method, beta in (
        ("Online", np.zeros(2)),
        ("Prior+Online", offline_beta.copy()),
    ):
        state[method] = {
            "beta": beta,
            "history": [],
            "calibration": {
                variant: [] for variant in VARIANTS if variant != "no_difficulty"
            },
            "gamma": {
                variant: 0.0 for variant in VARIANTS if variant != "no_difficulty"
            },
        }
    observed_x = []
    observed_y = []

    for task in rows:
        x_current = np.asarray([1.0, task["ru"]], dtype=float)
        current = {}
        for method in ("Online", "Prior+Online"):
            base_logit = float(x_current @ state[method]["beta"])
            base_probability = float(base.sigmoid(base_logit))
            current[method] = {}
            for variant in VARIANTS:
                if variant == "no_difficulty":
                    score, neighbor_count, mean_distance = 0.0, 0, None
                    gamma = 0.0
                else:
                    score, neighbor_count, mean_distance = topk_difficulty(
                        base,
                        task,
                        state[method]["history"],
                        embedding_by_task,
                        threshold_for_variant(variant),
                    )
                    gamma = state[method]["gamma"][variant]
                probability = float(base.sigmoid(base_logit + gamma * score))
                predictions.append(
                    {
                        **task,
                        "method": method,
                        "variant": variant,
                        "threshold": threshold_for_variant(variant)
                        if variant != "no_difficulty"
                        else None,
                        "base_probability": base_probability,
                        "difficulty_score": score,
                        "difficulty_coefficient": gamma,
                        "neighbor_count": neighbor_count,
                        "mean_neighbor_distance": mean_distance,
                        "probability": probability,
                    }
                )
                current[method][variant] = {
                    "base_logit": base_logit,
                    "difficulty_score": score,
                }
            current[method]["residual"] = float(task["success"] - base_probability)

        observed_x.append(x_current)
        observed_y.append(float(task["success"]))
        x_history = np.asarray(observed_x, dtype=float)
        y_history = np.asarray(observed_y, dtype=float)
        online_precision = np.eye(2, dtype=float) * base.ONLINE_PRIOR_PRECISION
        state["Online"]["beta"] = base.fit_map_logistic(
            x_history,
            y_history,
            prior_mean=np.zeros(2),
            prior_precision=online_precision,
            initial=state["Online"]["beta"],
        )
        state["Prior+Online"]["beta"] = base.fit_map_logistic(
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
                    "residual": current[method]["residual"],
                }
            )
            for variant in VARIANTS:
                if variant == "no_difficulty":
                    continue
                state[method]["calibration"][variant].append(
                    {
                        **current[method][variant],
                        "success": float(task["success"]),
                    }
                )
                calibration = state[method]["calibration"][variant]
                state[method]["gamma"][variant] = base.fit_difficulty_coefficient(
                    [item["base_logit"] for item in calibration],
                    [item["difficulty_score"] for item in calibration],
                    [item["success"] for item in calibration],
                    initial=state[method]["gamma"][variant],
                )
    return predictions


def summarize(base, predictions):
    overall = []
    streams = []
    for method in ("Online", "Prior+Online"):
        for variant in VARIANTS:
            selected = [
                row
                for row in predictions
                if row["method"] == method and row["variant"] == variant
            ]
            metrics = base.probability_metrics(selected)
            eligible = [row for row in selected if variant != "no_difficulty"]
            coverage = (
                sum(row["neighbor_count"] >= 1 for row in eligible) / len(eligible)
                if eligible
                else None
            )
            mean_neighbors = (
                float(np.mean([row["neighbor_count"] for row in eligible]))
                if eligible
                else None
            )
            overall.append(
                {
                    "method": method,
                    "variant": variant,
                    **metrics,
                    "coverage": coverage,
                    "mean_neighbor_count": mean_neighbors,
                }
            )
            for run in base.RUN_DIRS:
                for policy in base.POLICIES:
                    subset = [
                        row
                        for row in selected
                        if row["run"] == run and row["policy"] == policy
                    ]
                    streams.append(
                        {
                            "run": run,
                            "policy": policy,
                            "method": method,
                            "variant": variant,
                            **base.probability_metrics(subset),
                            "coverage": (
                                sum(row["neighbor_count"] >= 1 for row in subset)
                                / len(subset)
                                if variant != "no_difficulty"
                                else None
                            ),
                        }
                    )

    keyed_streams = {
        (row["run"], row["policy"], row["method"], row["variant"]): row
        for row in streams
    }
    for row in overall:
        if row["variant"] == "no_difficulty":
            row["better_log_loss_streams_vs_no_difficulty"] = None
            row["better_brier_streams_vs_no_difficulty"] = None
            row["delta_log_loss_vs_no_difficulty"] = 0.0
            row["delta_brier_vs_no_difficulty"] = 0.0
            continue
        baseline = next(
            item
            for item in overall
            if item["method"] == row["method"]
            and item["variant"] == "no_difficulty"
        )
        row["delta_log_loss_vs_no_difficulty"] = row["log_loss"] - baseline["log_loss"]
        row["delta_brier_vs_no_difficulty"] = row["brier"] - baseline["brier"]
        row["better_log_loss_streams_vs_no_difficulty"] = sum(
            keyed_streams[(run, policy, row["method"], row["variant"])]["log_loss"]
            < keyed_streams[(run, policy, row["method"], "no_difficulty")]["log_loss"]
            for run in base.RUN_DIRS
            for policy in base.POLICIES
        )
        row["better_brier_streams_vs_no_difficulty"] = sum(
            keyed_streams[(run, policy, row["method"], row["variant"])]["brier"]
            < keyed_streams[(run, policy, row["method"], "no_difficulty")]["brier"]
            for run in base.RUN_DIRS
            for policy in base.POLICIES
        )
    return overall, streams


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-script", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--overall-csv", type=Path, required=True)
    parser.add_argument("--stream-csv", type=Path, required=True)
    parser.add_argument("--predictions-csv", type=Path, required=True)
    args = parser.parse_args()

    base = load_base(args.base_script.resolve())
    results_root = base.extended_path(args.results_root)
    offline_rows = base.load_offline_rows(results_root)
    offline_x = np.asarray([[1.0, row["ru"]] for row in offline_rows], dtype=float)
    offline_y = np.asarray([row["success"] for row in offline_rows], dtype=float)
    weak_precision = np.eye(2, dtype=float) * 1e-4
    offline_beta = base.fit_map_logistic(
        offline_x, offline_y, np.zeros(2), weak_precision
    )
    offline_probability = base.sigmoid(offline_x @ offline_beta)
    offline_precision = (
        offline_x.T
        * np.maximum(offline_probability * (1.0 - offline_probability), 1e-9)
    ) @ offline_x + weak_precision

    embedding_by_task, embedding_model = base.load_embeddings(args.embedding_cache)
    source_streams = base.load_online_streams(results_root)
    predictions = []
    for rows in source_streams.values():
        predictions.extend(
            replay_stream(
                base, rows, embedding_by_task, offline_beta, offline_precision
            )
        )
    overall, streams = summarize(base, predictions)

    summary = {
        "stream_count": len(source_streams),
        "tasks_per_stream": 134,
        "embedding_model": embedding_model,
        "distance": "squared Euclidean on normalized query embeddings",
        "thresholds": list(THRESHOLDS),
        "neighbor_rule": "take three nearest eligible past queries, then keep distance <= tau",
        "exact_same_query_excluded": True,
        "state_shared_across_streams": False,
        "overall": overall,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.overall_csv, overall)
    write_csv(args.stream_csv, streams)
    write_csv(args.predictions_csv, predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
