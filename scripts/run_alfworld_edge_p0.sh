#!/usr/bin/env bash

set -e

# Edit these settings before running the experiment.
MANIFEST="ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n134.json"
SUBSET_MANIFEST="ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json"
TRAJECTORY_FILE="ProcedureMem/Alfworld/alfworld_format_traj.json"
EXPERIMENT_NAME="edge_raw_capacity_v1"

# Edge-50
python -m ProcedureMem.eval_alfworld \
  --condition edge_raw \
  --edge-capacity 50 \
  --edge-subset-manifest "$SUBSET_MANIFEST" \
  --edge-memory-dir ProcedureMem/memory/alfworld/edge_raw_50 \
  --trajectory-file "$TRAJECTORY_FILE" \
  --split valid_unseen \
  --seed 42 \
  --task-manifest "$MANIFEST" \
  --batch-size 2 \
  --max-steps 30 \
  --temperature 0 \
  --top-k 1 \
  --experiment-name "$EXPERIMENT_NAME"

# Edge-100
python -m ProcedureMem.eval_alfworld \
  --condition edge_raw \
  --edge-capacity 100 \
  --edge-subset-manifest "$SUBSET_MANIFEST" \
  --edge-memory-dir ProcedureMem/memory/alfworld/edge_raw_100 \
  --trajectory-file "$TRAJECTORY_FILE" \
  --split valid_unseen \
  --seed 42 \
  --task-manifest "$MANIFEST" \
  --batch-size 2 \
  --max-steps 30 \
  --temperature 0 \
  --top-k 1 \
  --experiment-name "$EXPERIMENT_NAME"

# Edge-150
python -m ProcedureMem.eval_alfworld \
  --condition edge_raw \
  --edge-capacity 150 \
  --edge-subset-manifest "$SUBSET_MANIFEST" \
  --edge-memory-dir ProcedureMem/memory/alfworld/edge_raw_150 \
  --trajectory-file "$TRAJECTORY_FILE" \
  --split valid_unseen \
  --seed 42 \
  --task-manifest "$MANIFEST" \
  --batch-size 2 \
  --max-steps 30 \
  --temperature 0 \
  --top-k 1 \
  --experiment-name "$EXPERIMENT_NAME"

echo "Results: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}"
