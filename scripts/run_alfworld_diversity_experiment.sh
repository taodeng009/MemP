#!/usr/bin/env bash

set -euo pipefail

# Fixed-capacity memory semantic-diversity experiment.
# Every setting can be overridden through an environment variable, for example:
#   TASK_COUNT=3 QUANTILE_BIN_COUNT=2 POOLS_PER_BIN=1 \
#     bash scripts/run_alfworld_diversity_experiment.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
SPLIT="${SPLIT:-valid_unseen}"
SEED="${SEED:-42}"
TASK_COUNT="${TASK_COUNT:-50}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-3}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-0.5}"

CANDIDATE_MEMORY_FILE="${CANDIDATE_MEMORY_FILE:-ProcedureMem/memory/alfworld/direct/documents.json}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-300}"
POOL_SIZE="${POOL_SIZE:-20}"
CANDIDATE_POOL_COUNT="${CANDIDATE_POOL_COUNT:-1000}"
SAMPLING_SEED="${SAMPLING_SEED:-42}"
SELECTION_SEED="${SELECTION_SEED:-42}"
QUANTILE_BIN_COUNT="${QUANTILE_BIN_COUNT:-5}"
POOLS_PER_BIN="${POOLS_PER_BIN:-3}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-diversity_${SPLIT}_seed${SEED}_n${TASK_COUNT}_pool${POOL_SIZE}}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-ProcedureMem/Alfworld/results/diversity/${EXPERIMENT_NAME}}"
POOL_FILE="${POOL_FILE:-${EXPERIMENT_ROOT}/pools.json}"
TASK_MANIFEST="${TASK_MANIFEST:-${EXPERIMENT_ROOT}/tasks.json}"
RESULTS_DIR="${RESULTS_DIR:-${EXPERIMENT_ROOT}/results}"
OVERWRITE="${OVERWRITE:-0}"

if [[ "$OVERWRITE" != "0" && "$OVERWRITE" != "1" ]]; then
  echo "OVERWRITE must be 0 or 1" >&2
  exit 2
fi

mkdir -p "$EXPERIMENT_ROOT"

"$PYTHON_BIN" -m ProcedureMem.build_diversity_pools \
  --candidate-memory-file "$CANDIDATE_MEMORY_FILE" \
  --candidate-count "$CANDIDATE_COUNT" \
  --pool-size "$POOL_SIZE" \
  --candidate-pool-count "$CANDIDATE_POOL_COUNT" \
  --sampling-seed "$SAMPLING_SEED" \
  --selection-seed "$SELECTION_SEED" \
  --quantile-bin-count "$QUANTILE_BIN_COUNT" \
  --pools-per-bin "$POOLS_PER_BIN" \
  --output "$POOL_FILE"

"$PYTHON_BIN" -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$TASK_MANIFEST" \
  --create-manifest-only

mapfile -t POOL_IDS < <(
  "$PYTHON_BIN" -c \
    'import json, sys; print(*[pool["pool_id"] for pool in json.load(open(sys.argv[1], encoding="utf-8"))["pools"]], sep="\n")' \
    "$POOL_FILE"
)

if [[ "${#POOL_IDS[@]}" -eq 0 ]]; then
  echo "No formal pools found in $POOL_FILE" >&2
  exit 2
fi

COMMON_ARGS=(
  --condition diversity_pool
  --diversity-pools "$POOL_FILE"
  --split "$SPLIT"
  --seed "$SEED"
  --limit-tasks "$TASK_COUNT"
  --task-manifest "$TASK_MANIFEST"
  --batch-size "$BATCH_SIZE"
  --max-steps "$MAX_STEPS"
  --temperature "$TEMPERATURE"
  --top-k "$TOP_K"
  --score-threshold "$SCORE_THRESHOLD"
  --output-dir "$RESULTS_DIR"
)

if [[ "$OVERWRITE" == "1" ]]; then
  COMMON_ARGS+=(--overwrite)
fi

for pool_id in "${POOL_IDS[@]}"; do
  echo "Running diversity pool: $pool_id"
  "$PYTHON_BIN" -m ProcedureMem.eval_alfworld \
    --pool-id "$pool_id" \
    "${COMMON_ARGS[@]}"
done

"$PYTHON_BIN" -m ProcedureMem.summarize_diversity_experiment \
  --results-dir "$RESULTS_DIR" \
  --diversity-pools "$POOL_FILE"

echo "Pool definitions: $POOL_FILE"
echo "Task manifest: $TASK_MANIFEST"
echo "Final results: $RESULTS_DIR/diversity_results.json"
