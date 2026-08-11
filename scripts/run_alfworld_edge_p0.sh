#!/usr/bin/env bash

set -e

# Edge capacity options: 50, 100, or 150.
EDGE_CAPACITY=50

# Edit these settings before running the experiment.
MANIFEST="ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n134.json"
SUBSET_MANIFEST="ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json"
TRAJECTORY_FILE="ProcedureMem/Alfworld/alfworld_format_traj.json"
EXPERIMENT_NAME="edge_raw_capacity_v1"

python -m ProcedureMem.eval_alfworld \
  --condition edge_raw \
  --edge-capacity "$EDGE_CAPACITY" \
  --edge-subset-manifest "$SUBSET_MANIFEST" \
  --edge-memory-dir "ProcedureMem/memory/alfworld/edge_raw_${EDGE_CAPACITY}" \
  --trajectory-file "$TRAJECTORY_FILE" \
  --split valid_unseen \
  --seed 42 \
  --task-manifest "$MANIFEST" \
  --batch-size 2 \
  --max-steps 30 \
  --temperature 0 \
  --top-k 1 \
  --experiment-name "$EXPERIMENT_NAME"

echo "Results: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}/edge_raw_${EDGE_CAPACITY}"
