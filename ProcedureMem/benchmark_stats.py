"""Dependency-free statistics helpers for retrieval benchmarks."""

from __future__ import annotations

import statistics
from typing import Sequence


def percentile(values: Sequence[float], percentile_value: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize empty latency values")
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
    }

