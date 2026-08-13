"""Small OpenMem reranker client used by the feasibility benchmark."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Sequence

try:
    import requests
except ModuleNotFoundError:  # Allows dependency-light unit tests with an injected session.
    requests = None  # type: ignore[assignment]


DEFAULT_BASE_URL = "https://memos.memtensor.cn/api/openmem/v1"
DEFAULT_MODEL = "memos-reranker-4b"


class RerankerError(RuntimeError):
    """Raised when OpenMem cannot provide a usable rerank response."""


@dataclass(frozen=True)
class RerankResult:
    index: int
    relevance_score: float


@dataclass(frozen=True)
class RerankResponse:
    results: tuple[RerankResult, ...]
    latency_ms: float
    request_id: str | None = None
    prompt_tokens: int | None = None
    total_tokens: int | None = None


class OpenMemReranker:
    """Call OpenMem's ``POST /rerank`` endpoint with a persistent HTTP session."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        session: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MEMOS_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing MEMOS_API_KEY for OpenMem reranker")
        self.base_url = (
            base_url or os.getenv("MEMOS_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.getenv("MEMOS_RERANK_MODEL") or DEFAULT_MODEL
        if self.model != DEFAULT_MODEL:
            raise ValueError(
                f"This benchmark requires {DEFAULT_MODEL!r}, got {self.model!r}"
            )
        configured_timeout = os.getenv("MEMOS_RERANK_TIMEOUT")
        self.timeout = float(timeout or configured_timeout or 30.0)
        if self.timeout <= 0:
            raise ValueError("Reranker timeout must be positive")
        if session is None and requests is None:
            raise RuntimeError("The 'requests' package is required for OpenMem reranking")
        self.session = session or requests.Session()

    def rerank(
        self,
        *,
        query: str,
        documents: Sequence[str],
        top_n: int,
    ) -> RerankResponse:
        if not documents:
            return RerankResponse(results=(), latency_ms=0.0)
        if top_n < 1:
            raise ValueError("top_n must be at least 1")

        started = time.perf_counter()
        try:
            response = self.session.post(
                f"{self.base_url}/rerank",
                headers={
                    "Authorization": f"Token {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "query": query,
                    "documents": list(documents),
                    "top_n": min(top_n, len(documents)),
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except Exception as exc:
            raise RerankerError(f"OpenMem rerank request failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise RerankerError("OpenMem response has no results list")
        parsed: list[RerankResult] = []
        for item in raw_results:
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankerError("OpenMem returned an invalid rerank item") from exc
            if not 0 <= index < len(documents):
                raise RerankerError(f"OpenMem returned out-of-range index {index}")
            parsed.append(RerankResult(index=index, relevance_score=score))

        usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
        return RerankResponse(
            results=tuple(parsed),
            latency_ms=latency_ms,
            request_id=payload.get("id"),
            prompt_tokens=usage.get("prompt_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
