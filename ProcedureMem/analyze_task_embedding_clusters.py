"""Cluster existing ALFWorld task queries with the configured embedding model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np


def extended_path(path: Path) -> Path:
    resolved = path.resolve()
    if os.name == "nt" and not str(resolved).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(resolved))
    return resolved


def load_env(path: Path):
    values = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[key.strip()] = value
    return values


def load_tasks(path: Path):
    tasks = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = row["task_id"]
            if task_id in seen:
                raise ValueError(f"Duplicate task_id: {task_id}")
            seen.add(task_id)
            tasks.append(
                {
                    "task_id": task_id,
                    "task_index": int(row["task_index"]),
                    "query": row["query"],
                    "task_family": row["task_type"].split("-", 1)[0],
                }
            )
    tasks.sort(key=lambda row: row["task_index"])
    if len(tasks) != 134:
        raise ValueError(f"Expected 134 tasks, found {len(tasks)}")
    return tasks


def embedding_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base + "/embeddings"


def request_embeddings(endpoint, api_key, model, texts, timeout):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Embedding endpoint returned HTTP {error.code}: {detail}")
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot reach embedding endpoint {endpoint}: {error.reason}. "
            "Start the configured embedding service or point "
            "EMBEDDING_MODEL_BASE_URL to a reachable server."
        ) from error
    rows = sorted(result["data"], key=lambda row: int(row["index"]))
    if len(rows) != len(texts):
        raise ValueError(f"Requested {len(texts)} embeddings, received {len(rows)}")
    return [row["embedding"] for row in rows]


def obtain_embeddings(tasks, env_values, cache_path, batch_size, timeout):
    model = env_values.get("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    base_url = env_values.get("EMBEDDING_MODEL_BASE_URL") or env_values.get(
        "OPENAI_API_BASE"
    )
    api_key = env_values.get("EMBEDDING_MODEL_KEY") or env_values.get(
        "OPENAI_API_KEY"
    )
    if not base_url or not api_key:
        raise RuntimeError("Embedding base URL or API key is missing from the env file")

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_model = str(cached["model"].item())
            cached_ids = list(cached["task_ids"].astype(str))
            vectors = np.asarray(cached["embeddings"], dtype=np.float64)
        expected_ids = [row["task_id"] for row in tasks]
        if cached_model != model or cached_ids != expected_ids:
            raise ValueError("Embedding cache does not match configured model/task IDs")
        return vectors, model, True

    endpoint = embedding_endpoint(base_url)
    unique_texts = list(dict.fromkeys(row["query"] for row in tasks))
    unique_vectors = []
    for start in range(0, len(unique_texts), batch_size):
        texts = unique_texts[start : start + batch_size]
        unique_vectors.extend(
            request_embeddings(endpoint, api_key, model, texts, timeout)
        )
    vector_by_text = dict(zip(unique_texts, unique_vectors))
    matrix = np.asarray(
        [vector_by_text[row["query"]] for row in tasks], dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.shape[0] != len(tasks):
        raise ValueError(f"Invalid embedding matrix shape: {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Embedding matrix contains non-finite values")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=matrix.astype(np.float32),
        task_ids=np.asarray([row["task_id"] for row in tasks]),
        model=np.asarray(model),
    )
    return matrix, model, False


def squared_distances(points, centroids):
    point_norm = np.sum(points * points, axis=1)[:, None]
    centroid_norm = np.sum(centroids * centroids, axis=1)[None, :]
    return np.maximum(point_norm + centroid_norm - 2.0 * points @ centroids.T, 0.0)


def kmeans_plus_plus(points, k, rng):
    centroids = [points[rng.integers(len(points))].copy()]
    closest = squared_distances(points, np.asarray(centroids))[:, 0]
    for _ in range(1, k):
        total = closest.sum()
        if total <= 0:
            choices = rng.choice(len(points), size=k - len(centroids), replace=False)
            centroids.extend(points[index].copy() for index in choices)
            break
        selected = rng.choice(len(points), p=closest / total)
        centroids.append(points[selected].copy())
        new_distance = squared_distances(points, np.asarray([centroids[-1]]))[:, 0]
        closest = np.minimum(closest, new_distance)
    return np.asarray(centroids[:k])


def single_kmeans(points, k, seed, max_iter=500, tolerance=1e-10):
    rng = np.random.default_rng(seed)
    centroids = kmeans_plus_plus(points, k, rng)
    labels = np.full(len(points), -1, dtype=int)
    for _ in range(max_iter):
        distances = squared_distances(points, centroids)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        new_centroids = np.empty_like(centroids)
        closest = distances[np.arange(len(points)), labels]
        for cluster in range(k):
            members = points[labels == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(closest))
                new_centroids[cluster] = points[farthest]
                labels[farthest] = cluster
                closest[farthest] = -1.0
        shift = float(np.sum((new_centroids - centroids) ** 2))
        centroids = new_centroids
        if shift <= tolerance:
            break
    final_distances = squared_distances(points, centroids)
    labels = np.argmin(final_distances, axis=1)
    inertia = float(np.sum(final_distances[np.arange(len(points)), labels]))
    return labels, centroids, inertia


def kmeans(points, k, seed=42, n_init=100):
    best = None
    for initialization in range(n_init):
        result = single_kmeans(points, k, seed + 104729 * initialization)
        if best is None or result[2] < best[2] - 1e-12:
            best = result
    return best


def contingency(labels, families):
    family_levels = sorted(set(families))
    cluster_levels = sorted(set(map(int, labels)))
    table = np.zeros((len(cluster_levels), len(family_levels)), dtype=int)
    family_index = {value: index for index, value in enumerate(family_levels)}
    cluster_index = {value: index for index, value in enumerate(cluster_levels)}
    for cluster, family in zip(labels, families):
        table[cluster_index[int(cluster)], family_index[family]] += 1
    return table, cluster_levels, family_levels


def entropy(counts):
    probabilities = np.asarray(counts, dtype=float)
    probabilities = probabilities[probabilities > 0]
    probabilities /= probabilities.sum()
    return -float(np.sum(probabilities * np.log(probabilities)))


def clustering_metrics(table):
    total = int(table.sum())
    rows = table.sum(axis=1)
    columns = table.sum(axis=0)
    mutual_information = 0.0
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            count = int(table[row, column])
            if count:
                mutual_information += (count / total) * math.log(
                    count * total / (rows[row] * columns[column])
                )
    row_entropy = entropy(rows)
    column_entropy = entropy(columns)
    denominator = (row_entropy + column_entropy) / 2.0
    nmi = mutual_information / denominator if denominator else 1.0

    choose2 = lambda values: np.sum(values * (values - 1) / 2.0)
    index = float(choose2(table))
    row_pairs = float(choose2(rows))
    column_pairs = float(choose2(columns))
    total_pairs = total * (total - 1) / 2.0
    expected = row_pairs * column_pairs / total_pairs
    maximum = (row_pairs + column_pairs) / 2.0
    ari = (index - expected) / (maximum - expected) if maximum != expected else 1.0
    purity = float(np.sum(np.max(table, axis=1)) / total)
    return {"nmi": nmi, "ari": ari, "purity": purity}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--assignments-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-init", type=int, default=100)
    args = parser.parse_args()

    tasks = load_tasks(extended_path(args.results_jsonl))
    env_values = load_env(args.env_file.resolve())
    embeddings, model, cache_reused = obtain_embeddings(
        tasks,
        env_values,
        args.embedding_cache.resolve(),
        args.batch_size,
        args.timeout,
    )
    families = [row["task_family"] for row in tasks]
    results = {}
    assignment_rows = []
    for k in (4, 5, 6, 7, 8):
        labels, _, inertia = kmeans(
            embeddings, k, seed=args.seed, n_init=args.n_init
        )
        table, cluster_levels, family_levels = contingency(labels, families)
        metrics = clustering_metrics(table)
        results[str(k)] = {
            **metrics,
            "inertia": inertia,
            "cluster_sizes": [int(value) for value in table.sum(axis=1)],
            "cluster_levels": cluster_levels,
            "family_levels": family_levels,
            "contingency": table.tolist(),
        }
        for task, label in zip(tasks, labels):
            assignment_rows.append(
                {
                    "k": k,
                    "task_index": task["task_index"],
                    "task_id": task["task_id"],
                    "query": task["query"],
                    "task_family": task["task_family"],
                    "cluster": int(label),
                }
            )

    summary = {
        "task_count": len(tasks),
        "unique_query_count": len({row["query"] for row in tasks}),
        "embedding_model": model,
        "embedding_dimension": int(embeddings.shape[1]),
        "embedding_cache_reused": cache_reused,
        "clustering_uses_task_family": False,
        "metric_definitions": {
            "nmi_normalization": "arithmetic mean of cluster and family entropy",
            "ari": "adjusted Rand index",
            "purity": "sum of per-cluster majority-family counts / N",
        },
        "kmeans": {
            "distance": "squared Euclidean on raw embedding vectors",
            "initialization": "k-means++",
            "seed": args.seed,
            "n_init": args.n_init,
            "max_iter": 500,
        },
        "results": results,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(args.assignments_csv, assignment_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
