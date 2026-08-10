#!/usr/bin/env bash

set -e

# Experiment settings. Edit these values before starting a new run.
SPLIT="valid_unseen"
SEED=42
TASK_COUNT=10
BATCH_SIZE=1
MAX_STEPS=30
TEMPERATURE=1.0
TOP_K=3

MANIFEST="ProcedureMem/Alfworld/manifests/${SPLIT}_seed${SEED}_n${TASK_COUNT}.json"
EXPERIMENT_NAME="${SPLIT}_seed${SEED}_n${TASK_COUNT}"

# Create the fixed task list. An existing compatible manifest is reused.
python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$MANIFEST" \
  --create-manifest-only

# Run the baseline.
python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --task-manifest "$MANIFEST" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --experiment-name "$EXPERIMENT_NAME" \
  --overwrite

# Run workflow memory on the same tasks with the same inference settings.
python -m ProcedureMem.eval_alfworld \
  --condition memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --task-manifest "$MANIFEST" \
  --batch-size "$BATCH_SIZE" \
  --max-steps "$MAX_STEPS" \
  --temperature "$TEMPERATURE" \
  --top-k "$TOP_K" \
  --experiment-name "$EXPERIMENT_NAME" \
  --overwrite

echo "Results: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}/comparison.json"
