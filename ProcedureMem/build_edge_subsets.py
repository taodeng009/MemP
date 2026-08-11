"""Build deterministic stratified, nested ALFWorld Edge trajectory subsets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ProcedureMem.edge_subsets import (
    build_edge_subset_manifest,
    load_trajectories,
    write_edge_subset_manifest,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAJECTORY_PATH = PACKAGE_DIR / "Alfworld" / "alfworld_format_traj.json"
DEFAULT_OUTPUT = (
    PACKAGE_DIR
    / "Alfworld"
    / "edge_subsets"
    / "stratified_nested_seed42.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--source-count", type=int, default=300)
    parser.add_argument("--capacities", type=int, nargs="+", default=(50, 100, 150))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    trajectory_path = args.trajectory_file.expanduser().resolve()
    try:
        trajectory_label = trajectory_path.relative_to(Path.cwd().resolve())
    except ValueError:
        trajectory_label = trajectory_path
    trajectories = load_trajectories(trajectory_path)
    manifest = build_edge_subset_manifest(
        trajectories,
        source_count=args.source_count,
        capacities=args.capacities,
        seed=args.seed,
        trajectory_file=trajectory_label,
    )
    write_edge_subset_manifest(args.output, manifest)
    print(f"Wrote Edge subset manifest: {args.output.resolve()}")
    for capacity in manifest["capacities"]:
        subset = manifest["subsets"][str(capacity)]
        print(
            f"Edge-{capacity}: unique_queries={subset['unique_query_count']} "
            f"families={subset['task_family_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
