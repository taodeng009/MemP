"""Summarize diversity-pool ALFWorld runs and compute Spearman correlation."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Sequence

from ProcedureMem.alfworld_experiment import load_json, write_json


CONTROLLED_PARAMETER_KEYS = (
    "model",
    "agent_api_base_url",
    "embedding_model",
    "split",
    "seed",
    "batch_size",
    "max_steps",
    "temperature",
    "top_p",
    "few_shot",
    "top_k",
    "score_threshold",
    "manifest_sha256",
)


def _load_run_summaries(results_dir: Path) -> dict[str, dict[str, Any]]:
    runs: dict[str, dict[str, Any]] = {}
    for summary_path in sorted(results_dir.glob("*/summary.json")):
        summary = load_json(summary_path)
        parameters = summary.get("parameters", {})
        if parameters.get("condition_mode") != "diversity_pool":
            continue
        pool_id = parameters.get("pool_id")
        if not isinstance(pool_id, str) or not pool_id:
            raise ValueError(f"Diversity run has no pool_id: {summary_path}")
        if pool_id in runs:
            raise ValueError(f"Multiple completed runs found for pool {pool_id}")
        runs[pool_id] = summary
    return runs


def build_diversity_results(
    results_dir: str | Path,
    diversity_pools: str | Path,
) -> dict[str, Any]:
    root = Path(results_dir).expanduser().resolve()
    pool_path = Path(diversity_pools).expanduser().resolve()
    pool_manifest = load_json(pool_path)
    pools = pool_manifest.get("pools")
    if not isinstance(pools, list) or not pools:
        raise ValueError(f"No formal pools found in {pool_path}")

    runs = _load_run_summaries(root)
    expected_ids = [pool["pool_id"] for pool in pools]
    missing = [pool_id for pool_id in expected_ids if pool_id not in runs]
    if missing:
        raise ValueError("Missing completed diversity runs: " + ", ".join(missing))

    reference = runs[expected_ids[0]]
    reference_parameters = reference["parameters"]
    reference_task_ids = reference.get("task_ids")
    for pool_id in expected_ids[1:]:
        summary = runs[pool_id]
        if summary.get("task_ids") != reference_task_ids:
            raise ValueError(f"Task IDs or order differ for pool {pool_id}")
        parameters = summary["parameters"]
        mismatches = [
            key
            for key in CONTROLLED_PARAMETER_KEYS
            if parameters.get(key) != reference_parameters.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Controlled evaluation parameters differ for pool {pool_id}: "
                + ", ".join(mismatches)
            )

    pool_results = [
        {
            "pool_id": pool["pool_id"],
            "quantile_bin": pool["quantile_bin"],
            "quantile_range": pool["quantile_range"],
            "diversity": float(pool["diversity"]),
            "success_rate": float(runs[pool["pool_id"]]["success_rate"]),
        }
        for pool in pools
    ]

    from scipy.stats import spearmanr

    correlation = spearmanr(
        [row["diversity"] for row in pool_results],
        [row["success_rate"] for row in pool_results],
    )
    rho = float(correlation.statistic)
    p_value = float(correlation.pvalue)
    trend = []
    for bin_index in sorted({row["quantile_bin"] for row in pool_results}):
        rows = [row for row in pool_results if row["quantile_bin"] == bin_index]
        trend.append(
            {
                "quantile_bin": bin_index,
                "mean_diversity": statistics.fmean(row["diversity"] for row in rows),
                "mean_success_rate": statistics.fmean(
                    row["success_rate"] for row in rows
                ),
            }
        )

    evaluation_parameters = {
        key: reference_parameters.get(key) for key in CONTROLLED_PARAMETER_KEYS
    }
    evaluation_parameters["task_count"] = reference.get("task_count")
    evaluation_parameters["task_manifest"] = reference_parameters.get("manifest")
    return {
        "schema_version": 1,
        "diversity_metric": pool_manifest.get("diversity_metric"),
        "generation_parameters": pool_manifest.get("generation_parameters"),
        "evaluation_parameters": evaluation_parameters,
        "pool_results": pool_results,
        "quantile_trend": trend,
        "spearman": {
            "rho": None if math.isnan(rho) else rho,
            "p_value": None if math.isnan(p_value) else p_value,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--diversity-pools", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    output = args.output or results_dir / "diversity_results.json"
    results = build_diversity_results(results_dir, args.diversity_pools)
    write_json(output, results)
    print(f"Wrote diversity results to {Path(output).resolve()}")
    print(
        "Spearman: "
        f"rho={results['spearman']['rho']}, p={results['spearman']['p_value']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
