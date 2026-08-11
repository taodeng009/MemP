"""Build or validate an ALFWorld raw-trajectory Edge memory index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from ProcedureMem.build_edge_subsets import DEFAULT_OUTPUT as DEFAULT_SUBSET_MANIFEST
from ProcedureMem.edge_memory import RawTrajectoryMemory
from ProcedureMem.runtime_config import (
    DEFAULT_MEMORY_DIR,
    DEFAULT_TRAJECTORY_PATH,
    configure_runtime,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-file", type=Path, default=DEFAULT_TRAJECTORY_PATH)
    parser.add_argument("--subset-manifest", type=Path, default=DEFAULT_SUBSET_MANIFEST)
    parser.add_argument("--capacity", type=int, required=True)
    parser.add_argument("--memory-dir", type=Path)
    parser.add_argument("--smoke-query", default="put a clean apple in fridge")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_runtime(require_embedding=True)
    memory_dir = args.memory_dir or DEFAULT_MEMORY_DIR / f"edge_raw_{args.capacity}"
    memory = RawTrajectoryMemory(
        trajectory_file=args.trajectory_file,
        subset_manifest=args.subset_manifest,
        capacity=args.capacity,
        memory_dir=memory_dir,
        top_k=1,
        score_threshold=None,
    )
    retrieved = memory.retrieve(args.smoke_query)
    if len(retrieved) != 1:
        raise RuntimeError("Edge memory smoke retrieval did not return exactly one item")
    document, score = retrieved[0]
    print(
        f"Edge-{args.capacity} ready: documents={memory.document_count} "
        f"top1_index={document.metadata['trajectory_index']} raw_score={score:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
