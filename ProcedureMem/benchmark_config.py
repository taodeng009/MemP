"""Dependency-free configuration helpers for retrieval benchmarks."""

from __future__ import annotations

import os


RERANK_CANDIDATE_THRESHOLD_ENV = "MEMP_RERANK_CANDIDATE_SCORE_THRESHOLD"


def candidate_score_threshold(value: float | None = None) -> float | None:
    """Resolve CLI/environment candidate threshold; an empty env means disabled."""
    if value is not None:
        if value < 0:
            raise ValueError("Reranker candidate score threshold must be non-negative")
        return value
    raw = os.getenv(RERANK_CANDIDATE_THRESHOLD_ENV)
    if raw is None or not raw.strip():
        return None
    try:
        parsed = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{RERANK_CANDIDATE_THRESHOLD_ENV} must be a non-negative number or empty"
        ) from exc
    if parsed < 0:
        raise ValueError(
            f"{RERANK_CANDIDATE_THRESHOLD_ENV} must be non-negative or empty"
        )
    return parsed

