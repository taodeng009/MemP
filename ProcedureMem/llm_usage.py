"""Canonical token-usage types and JSONL helpers for LLM calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None

    @property
    def available(self) -> bool:
        return (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
        )


@dataclass(frozen=True)
class LLMCallResult:
    content: str
    usage: LLMUsage
    model: str | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class ParsedLLMCallResult:
    value: Any
    call: LLMCallResult


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _token_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        count = int(value)
    except (TypeError, ValueError):
        return None
    return count if count >= 0 else None


def extract_usage(response: Any) -> LLMUsage:
    """Extract only the three provider-reported token totals we persist."""
    usage = _field(response, "usage")
    if usage is None:
        return LLMUsage(None, None, None)
    return LLMUsage(
        prompt_tokens=_token_count(_field(usage, "prompt_tokens")),
        completion_tokens=_token_count(_field(usage, "completion_tokens")),
        total_tokens=_token_count(_field(usage, "total_tokens")),
    )


def aggregate_usage(calls: Sequence[LLMUsage]) -> dict[str, Any]:
    reported = [call for call in calls if call.available]
    complete = bool(calls) and len(reported) == len(calls)
    return {
        "call_count": len(calls),
        "reported_call_count": len(reported),
        "usage_complete": complete,
        "prompt_tokens": (
            sum(call.prompt_tokens for call in calls) if complete else None
        ),
        "completion_tokens": (
            sum(call.completion_tokens for call in calls) if complete else None
        ),
        "total_tokens": (
            sum(call.total_tokens for call in calls) if complete else None
        ),
        "reported_prompt_tokens": sum(call.prompt_tokens for call in reported),
        "reported_completion_tokens": sum(
            call.completion_tokens for call in reported
        ),
        "reported_total_tokens": sum(call.total_tokens for call in reported),
    }


def call_usage_fields(call: LLMCallResult) -> dict[str, Any]:
    return {
        "model": call.model,
        "request_id": call.request_id,
        "prompt_tokens": call.usage.prompt_tokens,
        "completion_tokens": call.usage.completion_tokens,
        "total_tokens": call.usage.total_tokens,
        "usage_available": call.usage.available,
    }


def append_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Append complete JSON lines and flush before returning."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as writer:
        for record in records:
            writer.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        writer.flush()


def task_usage_records(
    *,
    task_index: int,
    task_id: str,
    calls: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    identity = {
        "schema_version": 1,
        "scope": "task_execution",
        "task_index": task_index,
        "task_id": task_id,
    }
    records = [
        {**identity, "record_type": "call", **dict(call)} for call in calls
    ]
    records.append(
        {**identity, "record_type": "task_aggregate", **dict(aggregate)}
    )
    return records


def memory_usage_records(
    *,
    source_index: int | None,
    memory_id: str | None,
    call: LLMCallResult,
) -> list[dict[str, Any]]:
    identity = {
        "schema_version": 1,
        "scope": "memory_construction",
        "source_index": source_index,
        "memory_id": memory_id,
    }
    return [
        {
            **identity,
            "record_type": "call",
            "call_index": 1,
            **call_usage_fields(call),
        },
        {
            **identity,
            "record_type": "memory_aggregate",
            **aggregate_usage([call.usage]),
        },
    ]
