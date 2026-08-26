"""Build equal-size workflow-memory pools across semantic-diversity quantiles."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from ProcedureMem.alfworld_experiment import write_json
from ProcedureMem.cloud_scheduling import (
    ScheduledWorkflowMemory,
    load_cached_embedding,
    load_candidate_memories,
)

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MEMORY_CONFIG = PACKAGE_DIR / "config.yaml"
DEFAULT_CANDIDATE_FILE = (
    PACKAGE_DIR / "memory" / "alfworld" / "direct" / "documents.json"
)
DEFAULT_OUTPUT = (
    PACKAGE_DIR
    / "Alfworld"
    / "diversity_pools"
    / "workflow_pool20.json"
)


def mean_nearest_neighbor_squared_l2(
    memory_ids: Sequence[str],
    distance_matrix: Mapping[str, Mapping[str, float]],
) -> float:
    """Return mean within-pool nearest-neighbor squared-L2 distance."""
    ids = tuple(memory_ids)
    if len(ids) < 2:
        raise ValueError("A diversity pool must contain at least two memories")
    if len(ids) != len(set(ids)):
        raise ValueError("A diversity pool cannot contain duplicate memory IDs")
    try:
        nearest = [
            min(
                float(distance_matrix[memory_id][other])
                for other in ids
                if other != memory_id
            )
            for memory_id in ids
        ]
    except KeyError as exc:
        raise ValueError(f"Distance matrix is missing memory ID {exc.args[0]}") from exc
    return sum(nearest) / len(nearest)


def generate_unique_subsets(
    candidate_ids: Sequence[str],
    *,
    pool_size: int,
    count: int,
    seed: int,
) -> list[tuple[str, ...]]:
    """Generate deterministic random subsets, with no repeated subset."""
    ids = tuple(candidate_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate memory IDs must be unique")
    if pool_size < 2 or pool_size > len(ids):
        raise ValueError(f"pool_size must be between 2 and {len(ids)}")
    if count < 1:
        raise ValueError("candidate_pool_count must be at least 1")
    possible = math.comb(len(ids), pool_size)
    if count > possible:
        raise ValueError(f"Requested {count} unique pools, but only {possible} exist")

    order = {memory_id: index for index, memory_id in enumerate(ids)}
    rng = random.Random(seed)
    selected: set[tuple[str, ...]] = set()
    while len(selected) < count:
        subset = tuple(
            sorted(rng.sample(ids, pool_size), key=order.__getitem__)
        )
        selected.add(subset)
    return sorted(selected)


def select_across_quantile_bins(
    candidates: Sequence[dict[str, Any]],
    *,
    bin_count: int,
    pools_per_bin: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Randomly select the same number of pools from equal-frequency bins."""
    if bin_count < 2:
        raise ValueError("quantile_bin_count must be at least 2")
    if pools_per_bin < 1:
        raise ValueError("pools_per_bin must be at least 1")
    if len(candidates) < bin_count * pools_per_bin:
        raise ValueError(
            "candidate_pool_count must be at least quantile_bin_count * pools_per_bin"
        )

    ordered = sorted(
        candidates,
        key=lambda item: (item["diversity"], tuple(item["memory_ids"])),
    )
    bins: list[list[dict[str, Any]]] = [[] for _ in range(bin_count)]
    for rank, item in enumerate(ordered):
        bin_index = min(rank * bin_count // len(ordered), bin_count - 1)
        bins[bin_index].append(item)

    rng = random.Random(seed)
    formal: list[dict[str, Any]] = []
    for bin_index, rows in enumerate(bins):
        if len(rows) < pools_per_bin:
            raise ValueError(
                f"Quantile bin {bin_index} contains only {len(rows)} candidate pools"
            )
        chosen = sorted(
            rng.sample(rows, pools_per_bin),
            key=lambda item: (item["diversity"], tuple(item["memory_ids"])),
        )
        for within_bin_index, item in enumerate(chosen):
            formal.append(
                {
                    "pool_id": f"q{bin_index:02d}_p{within_bin_index:02d}",
                    "quantile_bin": bin_index,
                    "quantile_range": [
                        bin_index / bin_count,
                        (bin_index + 1) / bin_count,
                    ],
                    "diversity": float(item["diversity"]),
                    "memory_ids": list(item["memory_ids"]),
                }
            )
    subsets = [tuple(item["memory_ids"]) for item in formal]
    if len(subsets) != len(set(subsets)):
        raise RuntimeError("Formal pools unexpectedly contain duplicate subsets")
    return formal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-memory-file", type=Path, default=DEFAULT_CANDIDATE_FILE
    )
    parser.add_argument("--candidate-count", type=int, default=300)
    parser.add_argument("--pool-size", type=int, default=20)
    parser.add_argument("--candidate-pool-count", type=int, default=1000)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--quantile-bin-count", type=int, default=10)
    parser.add_argument("--pools-per-bin", type=int, default=2)
    parser.add_argument("--memory-config", default=str(DEFAULT_MEMORY_CONFIG))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    from ProcedureMem.runtime_config import configure_runtime, load_memory_config

    args = build_parser().parse_args(argv)
    if args.candidate_count < 2:
        raise ValueError("candidate_count must be at least 2")

    settings = configure_runtime(require_embedding=True)
    config = load_memory_config(args.memory_config)
    candidate_path = args.candidate_memory_file.expanduser().resolve()
    candidates = load_candidate_memories(candidate_path, limit=args.candidate_count)
    candidate_ids = tuple(item.memory_id for item in candidates)
    subsets = generate_unique_subsets(
        candidate_ids,
        pool_size=args.pool_size,
        count=args.candidate_pool_count,
        seed=args.sampling_seed,
    )

    memory = ScheduledWorkflowMemory(
        candidates,
        embedding=load_cached_embedding(config["memory_dir"]),
        retrieve_num=1,
        score_threshold=None,
    )
    distance_matrix = memory.candidate_query_distance_matrix()
    scored = [
        {
            "memory_ids": list(subset),
            "diversity": mean_nearest_neighbor_squared_l2(subset, distance_matrix),
        }
        for subset in subsets
    ]
    pools = select_across_quantile_bins(
        scored,
        bin_count=args.quantile_bin_count,
        pools_per_bin=args.pools_per_bin,
        seed=args.selection_seed,
    )

    output = args.output.expanduser().resolve()
    manifest = {
        "schema_version": 1,
        "diversity_metric": "mean_nearest_neighbor_squared_l2",
        "generation_parameters": {
            "candidate_memory_file": str(candidate_path),
            "candidate_count": len(candidates),
            "embedding_model": settings.embedding_model,
            "distance_metric": "mean_nearest_neighbor_squared_l2",
            "pool_size": args.pool_size,
            "candidate_pool_count": args.candidate_pool_count,
            "sampling_seed": args.sampling_seed,
            "selection_seed": args.selection_seed,
            "quantile_bin_count": args.quantile_bin_count,
            "pools_per_bin": args.pools_per_bin,
        },
        "pools": pools,
    }
    write_json(output, manifest)
    print(f"Wrote {len(pools)} unique formal pools to {output}")
    print(
        "D_NN range: "
        f"{min(item['diversity'] for item in pools):.6f} .. "
        f"{max(item['diversity'] for item in pools):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
