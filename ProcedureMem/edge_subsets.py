"""Deterministic stratified, nested trajectory subsets for ALFWorld Edge memory."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


EDGE_SUBSET_SCHEMA_VERSION = 1
TASK_FAMILIES = (
    "pick_and_place",
    "clean_then_place",
    "cool_then_place",
    "heat_then_place",
    "examine_in_light",
    "pick_two_then_place",
)


def classify_task_family(query: str) -> str:
    """Map an ALFWorld natural-language goal to one of its six task families."""
    normalized = " ".join(query.lower().split())
    if normalized.startswith("examine "):
        return "examine_in_light"
    if normalized.startswith(("find two ", "put two ")):
        return "pick_two_then_place"
    if re.search(r"\bclean\b", normalized):
        return "clean_then_place"
    if re.search(r"\bcool\b", normalized):
        return "cool_then_place"
    if re.search(r"\bheat\b", normalized):
        return "heat_then_place"
    return "pick_and_place"


def load_trajectories(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("Trajectory file must contain a non-empty JSON list")
    for index, item in enumerate(data):
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError(f"Trajectory item {index} has no valid query")
        if not isinstance(item.get("trajectory"), list):
            raise ValueError(f"Trajectory item {index} has no valid trajectory")
    return data


def _balanced_nested_order(
    grouped_indices: dict[str, list[int]],
    family_totals: dict[str, int],
    *,
    count: int,
) -> list[int]:
    """Create one order whose prefixes stay close to the source proportions."""
    selected_counts = Counter()
    cursors = Counter()
    order: list[int] = []
    total = sum(family_totals.values())

    for position in range(1, count + 1):
        candidates = [
            family
            for family in TASK_FAMILIES
            if cursors[family] < len(grouped_indices[family])
        ]
        if not candidates:
            raise ValueError("Not enough trajectories to satisfy requested capacities")

        # Pick the family with the largest proportional deficit. TASK_FAMILIES
        # supplies a deterministic tie break.
        family = max(
            candidates,
            key=lambda name: (
                position * family_totals[name] / total - selected_counts[name],
                -TASK_FAMILIES.index(name),
            ),
        )
        order.append(grouped_indices[family][cursors[family]])
        cursors[family] += 1
        selected_counts[family] += 1
    return order


def build_edge_subset_manifest(
    trajectories: Sequence[dict[str, Any]],
    *,
    source_count: int = 300,
    capacities: Sequence[int] = (50, 100, 150),
    seed: int = 42,
    trajectory_file: str | Path | None = None,
) -> dict[str, Any]:
    if source_count < 1 or source_count > len(trajectories):
        raise ValueError(
            f"source_count must be between 1 and {len(trajectories)}, got {source_count}"
        )
    normalized_capacities = sorted(set(int(value) for value in capacities))
    if not normalized_capacities or normalized_capacities[0] < 1:
        raise ValueError("capacities must contain positive integers")
    if normalized_capacities[-1] > source_count:
        raise ValueError("Largest capacity cannot exceed source_count")

    grouped: dict[str, list[int]] = {family: [] for family in TASK_FAMILIES}
    for index, item in enumerate(trajectories[:source_count]):
        grouped[classify_task_family(item["query"])].append(index)

    rng = random.Random(seed)
    for family in TASK_FAMILIES:
        rng.shuffle(grouped[family])

    family_totals = {family: len(grouped[family]) for family in TASK_FAMILIES}
    nested_order = _balanced_nested_order(
        grouped,
        family_totals,
        count=normalized_capacities[-1],
    )

    subsets: dict[str, Any] = {}
    for capacity in normalized_capacities:
        indices = nested_order[:capacity]
        counts = Counter(
            classify_task_family(trajectories[index]["query"]) for index in indices
        )
        subsets[str(capacity)] = {
            "capacity": capacity,
            "trajectory_indices": indices,
            "task_family_counts": {
                family: counts[family] for family in TASK_FAMILIES
            },
            "unique_query_count": len(
                {trajectories[index]["query"].strip() for index in indices}
            ),
        }

    manifest = {
        "schema_version": EDGE_SUBSET_SCHEMA_VERSION,
        "sampling_strategy": "task_type_stratified_nested",
        "seed": seed,
        "trajectory_file": str(Path(trajectory_file)) if trajectory_file else None,
        "source_start": 0,
        "source_count": source_count,
        "capacities": normalized_capacities,
        "source_task_family_counts": family_totals,
        "subsets": subsets,
    }
    validate_edge_subset_manifest(manifest, trajectory_count=len(trajectories))
    return manifest


def validate_edge_subset_manifest(
    manifest: dict[str, Any], *, trajectory_count: int
) -> None:
    if manifest.get("schema_version") != EDGE_SUBSET_SCHEMA_VERSION:
        raise ValueError("Unsupported Edge subset manifest schema_version")
    if manifest.get("sampling_strategy") != "task_type_stratified_nested":
        raise ValueError("Unsupported Edge subset sampling strategy")
    source_count = manifest.get("source_count")
    if not isinstance(source_count, int) or source_count < 1:
        raise ValueError("Edge subset manifest has invalid source_count")
    if source_count > trajectory_count:
        raise ValueError("Edge subset source_count exceeds trajectory file length")

    capacities = manifest.get("capacities")
    subsets = manifest.get("subsets")
    if not isinstance(capacities, list) or not capacities or not isinstance(subsets, dict):
        raise ValueError("Edge subset manifest has no capacities or subsets")

    previous: set[int] = set()
    for capacity in capacities:
        subset = subsets.get(str(capacity), {})
        indices = subset.get("trajectory_indices")
        if not isinstance(indices, list) or len(indices) != capacity:
            raise ValueError(f"Edge-{capacity} does not contain {capacity} indices")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Edge-{capacity} contains duplicate indices")
        if any(not isinstance(index, int) or index < 0 or index >= source_count for index in indices):
            raise ValueError(f"Edge-{capacity} contains an out-of-range index")
        current = set(indices)
        if not previous.issubset(current):
            raise ValueError("Edge subsets are not nested")
        previous = current


def write_edge_subset_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
