#!/usr/bin/env bash

set -euo pipefail

# Construction-scheduling feasibility experiment.
SPLIT="${SPLIT:-valid_unseen}"
SEED="${SEED:-42}"
TASK_COUNT="${TASK_COUNT:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
INTERVAL_SIZE="${INTERVAL_SIZE:-10}"
CONSTRUCTION_CAPACITY="${CONSTRUCTION_CAPACITY:-5}"
MAX_STEPS="${MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-3}"
read -r -a RANDOM_SEEDS_ARRAY <<< "${RANDOM_SEEDS:-1 2 3}"

MANIFEST="ProcedureMem/Alfworld/manifests/${SPLIT}_seed${SEED}_n${TASK_COUNT}.json"
CANDIDATES="ProcedureMem/memory/alfworld/direct/documents.json"
EXPERIMENT_NAME="cloud_scheduling_${SPLIT}_seed${SEED}_n${TASK_COUNT}_b${INTERVAL_SIZE}_c${CONSTRUCTION_CAPACITY}"

python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$MANIFEST" \
  --create-manifest-only

COMMON_ARGS=(
  --condition cloud_scheduled
  --split "$SPLIT"
  --seed "$SEED"
  --task-manifest "$MANIFEST"
  --candidate-memory-file "$CANDIDATES"
  --interval-size "$INTERVAL_SIZE"
  --construction-capacity "$CONSTRUCTION_CAPACITY"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --experiment-name "$EXPERIMENT_NAME"
  --overwrite
)

for scheduler_seed in "${RANDOM_SEEDS_ARRAY[@]}"; do
  python -m ProcedureMem.eval_alfworld \
    --schedule-policy random \
    --scheduler-seed "$scheduler_seed" \
    --condition-name "cloud_scheduled_random_seed${scheduler_seed}" \
    "${COMMON_ARGS[@]}"
done

python -m ProcedureMem.eval_alfworld \
  --schedule-policy oracle_high \
  --condition-name cloud_scheduled_oracle_high \
  "${COMMON_ARGS[@]}"

echo "Results: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}"
echo "Comparison: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}/scheduling_comparison.json"
