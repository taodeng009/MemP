#!/usr/bin/env bash

set -euo pipefail

# P0 settings.
SPLIT="valid_unseen"
SEED=42
TASK_COUNT=134
BATCH_SIZE=2
MAX_STEPS=30
TEMPERATURE=0
TOP_K=1
EXPERIMENT_NAME="edge_raw_capacity_v1"

TRAJECTORY_FILE="ProcedureMem/Alfworld/alfworld_format_traj.json"
SUBSET_MANIFEST="ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json"
TASK_MANIFEST="ProcedureMem/Alfworld/manifests/${SPLIT}_seed${SEED}_n${TASK_COUNT}.json"
RESULTS_DIR="ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}"
MEMORY_ROOT="ProcedureMem/memory/alfworld"

# Set RUN_REFERENCES=1 only when No Memory and Cloud MemP-300 must be rerun
# under the same experiment directory. The default runs Edge P0 only.
RUN_REFERENCES="${RUN_REFERENCES:-0}"

python -m ProcedureMem.build_edge_subsets \
  --trajectory-file "$TRAJECTORY_FILE" \
  --source-count 300 \
  --capacities 50 100 150 \
  --seed "$SEED" \
  --output "$SUBSET_MANIFEST"

# Create or validate the fixed 134-task evaluation manifest.
python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$TASK_MANIFEST" \
  --create-manifest-only

for CAPACITY in 50 100 150; do
  python -m ProcedureMem.build_edge_memory \
    --trajectory-file "$TRAJECTORY_FILE" \
    --subset-manifest "$SUBSET_MANIFEST" \
    --capacity "$CAPACITY" \
    --memory-dir "${MEMORY_ROOT}/edge_raw_${CAPACITY}"

  python -m ProcedureMem.eval_alfworld \
    --condition edge_raw \
    --condition-name "edge_raw_${CAPACITY}" \
    --edge-capacity "$CAPACITY" \
    --edge-subset-manifest "$SUBSET_MANIFEST" \
    --edge-memory-dir "${MEMORY_ROOT}/edge_raw_${CAPACITY}" \
    --trajectory-file "$TRAJECTORY_FILE" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --task-manifest "$TASK_MANIFEST" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --experiment-name "$EXPERIMENT_NAME"
done

if [[ "$RUN_REFERENCES" == "1" ]]; then
  python -m ProcedureMem.eval_alfworld \
    --condition no_memory \
    --split "$SPLIT" \
    --seed "$SEED" \
    --task-manifest "$TASK_MANIFEST" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --experiment-name "$EXPERIMENT_NAME"

  python -m ProcedureMem.eval_alfworld \
    --condition memory \
    --condition-name cloud_workflow_300 \
    --split "$SPLIT" \
    --seed "$SEED" \
    --task-manifest "$TASK_MANIFEST" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --temperature "$TEMPERATURE" \
    --top-k 10 \
    --experiment-name "$EXPERIMENT_NAME"
fi

python -m ProcedureMem.summarize_edge_p0 --results-dir "$RESULTS_DIR"

echo "Results: ${RESULTS_DIR}/capacity_comparison.json"
