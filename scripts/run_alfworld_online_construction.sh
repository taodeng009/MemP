#!/usr/bin/env bash

set -euo pipefail

# Queue-based online Cloud workflow construction.
SPLIT="${SPLIT:-valid_unseen}"
SEED="${SEED:-42}"
TASK_COUNT="${TASK_COUNT:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
INTERVAL_SIZE="${INTERVAL_SIZE:-10}"
CONSTRUCTION_CAPACITY="${CONSTRUCTION_CAPACITY:-2}"
WARM_START_COUNT="${WARM_START_COUNT:-0}"
WARM_START_SEED="${WARM_START_SEED:-42}"
WARM_START_MEMORY_FILE="${WARM_START_MEMORY_FILE:-ProcedureMem/memory/alfworld/direct/documents.json}"
MAX_STEPS="${MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-3}"
POLICIES="${POLICIES:-fifo greedy_novelty}"
RUN_RANDOM="${RUN_RANDOM:-0}"
SCHEDULER_SEED="${SCHEDULER_SEED:-42}"

MANIFEST="ProcedureMem/Alfworld/manifests/${SPLIT}_seed${SEED}_n${TASK_COUNT}.json"
WARM_SUFFIX=""
if [[ "$WARM_START_COUNT" != "0" ]]; then
  WARM_SUFFIX="_warm${WARM_START_COUNT}_ws${WARM_START_SEED}"
fi
DEFAULT_EXPERIMENT_NAME="online_construction_${SPLIT}_seed${SEED}_n${TASK_COUNT}_b${INTERVAL_SIZE}_c${CONSTRUCTION_CAPACITY}${WARM_SUFFIX}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"

python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$MANIFEST" \
  --create-manifest-only

COMMON_ARGS=(
  --condition online_construction
  --split "$SPLIT"
  --seed "$SEED"
  --task-manifest "$MANIFEST"
  --interval-size "$INTERVAL_SIZE"
  --construction-capacity "$CONSTRUCTION_CAPACITY"
  --warm-start-count "$WARM_START_COUNT"
  --warm-start-seed "$WARM_START_SEED"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --experiment-name "$EXPERIMENT_NAME"
  --overwrite
)

if [[ "$WARM_START_COUNT" != "0" ]]; then
  COMMON_ARGS+=(--warm-start-memory-file "$WARM_START_MEMORY_FILE")
fi

for policy in $POLICIES; do
  python -m ProcedureMem.eval_alfworld \
    --schedule-policy "$policy" \
    --condition-name "online_construction_${policy}" \
    "${COMMON_ARGS[@]}"
done

if [[ "$RUN_RANDOM" == "1" ]]; then
  python -m ProcedureMem.eval_alfworld \
    --schedule-policy random \
    --scheduler-seed "$SCHEDULER_SEED" \
    --condition-name "online_construction_random_seed${SCHEDULER_SEED}" \
    "${COMMON_ARGS[@]}"
fi

RESULT_ROOT="ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}"
echo "Results: ${RESULT_ROOT}"
echo "Comparison: ${RESULT_ROOT}/online_construction_comparison.json"
