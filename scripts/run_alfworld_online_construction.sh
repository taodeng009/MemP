#!/usr/bin/env bash

set -euo pipefail

# Queue-based online Cloud workflow construction.
#
# Warm-start-only initial-pool-size sweep (serial execution, no construction):
#   WARM_START_ONLY=1 WARM_START_COUNTS="0 5 10 20" \
#     bash scripts/run_alfworld_online_construction.sh
#
# FIFO vs Oracle-Coverage smoke test (three intervals):
#   TASK_COUNT=30 WARM_START_ONLY=0 WARM_START_COUNTS=0 \
#   POLICIES="fifo oracle_coverage" EXPERIMENT_NAME="online_fifo_oracle_smoke_n30" \
#     bash scripts/run_alfworld_online_construction.sh
#
# FIFO vs Oracle-Coverage 134-task cold-start baseline:
#   TASK_COUNT=134 WARM_START_ONLY=0 WARM_START_COUNTS=0 \
#   POLICIES="fifo oracle_coverage" \
#   EXPERIMENT_NAME="online_construction_valid_unseen_seed42_n134_b10_c2_cold_fifo_oracle" \
#     bash scripts/run_alfworld_online_construction.sh
SPLIT="${SPLIT:-valid_unseen}"
SEED="${SEED:-42}"
TASK_COUNT="${TASK_COUNT:-50}"
BATCH_SIZE="${BATCH_SIZE:-2}"
INTERVAL_SIZE="${INTERVAL_SIZE:-10}"
CONSTRUCTION_CAPACITY="${CONSTRUCTION_CAPACITY:-2}"
WARM_START_COUNT="${WARM_START_COUNT:-5 10 15 20}"
WARM_START_COUNTS="${WARM_START_COUNTS:-$WARM_START_COUNT}"
WARM_START_ONLY="${WARM_START_ONLY:-1}"
WARM_START_SEED="${WARM_START_SEED:-42}"
WARM_START_MEMORY_FILE="${WARM_START_MEMORY_FILE:-ProcedureMem/memory/alfworld/direct/documents.json}"
MAX_STEPS="${MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-3}"
POLICIES="${POLICIES:-fifo greedy_novelty}"
RUN_RANDOM="${RUN_RANDOM:-0}"
SCHEDULER_SEED="${SCHEDULER_SEED:-42}"

if [[ "$WARM_START_ONLY" != "0" && "$WARM_START_ONLY" != "1" ]]; then
  echo "WARM_START_ONLY must be 0 or 1" >&2
  exit 2
fi

read -r -a WARM_COUNTS <<< "$WARM_START_COUNTS"
if [[ "${#WARM_COUNTS[@]}" -eq 0 ]]; then
  echo "WARM_START_COUNTS must contain at least one non-negative integer" >&2
  exit 2
fi
for warm_count in "${WARM_COUNTS[@]}"; do
  if [[ ! "$warm_count" =~ ^[0-9]+$ ]]; then
    echo "Invalid warm-start count: $warm_count" >&2
    exit 2
  fi
done

MANIFEST="ProcedureMem/Alfworld/manifests/${SPLIT}_seed${SEED}_n${TASK_COUNT}.json"

python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split "$SPLIT" \
  --seed "$SEED" \
  --limit-tasks "$TASK_COUNT" \
  --task-manifest "$MANIFEST" \
  --create-manifest-only

RESULT_ROOTS=()
for warm_count in "${WARM_COUNTS[@]}"; do
  effective_capacity="$CONSTRUCTION_CAPACITY"
  run_policies="$POLICIES"
  condition_prefix="online_construction"
  mode_suffix=""
  if [[ "$WARM_START_ONLY" == "1" ]]; then
    effective_capacity=0
    run_policies="fifo"
    condition_prefix="warm_start_only"
    mode_suffix="_initial_pool_only"
  fi

  warm_suffix=""
  if [[ "$warm_count" != "0" ]]; then
    warm_suffix="_warm${warm_count}_ws${WARM_START_SEED}"
  fi
  default_experiment_name="online_construction_${SPLIT}_seed${SEED}_n${TASK_COUNT}_b${INTERVAL_SIZE}_c${effective_capacity}${warm_suffix}${mode_suffix}"
  if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
    if [[ "${#WARM_COUNTS[@]}" -gt 1 ]]; then
      experiment_name="${EXPERIMENT_NAME}_warm${warm_count}"
    else
      experiment_name="$EXPERIMENT_NAME"
    fi
  else
    experiment_name="$default_experiment_name"
  fi

  common_args=(
    --condition online_construction
    --split "$SPLIT"
    --seed "$SEED"
    --task-manifest "$MANIFEST"
    --interval-size "$INTERVAL_SIZE"
    --construction-capacity "$effective_capacity"
    --warm-start-count "$warm_count"
    --warm-start-seed "$WARM_START_SEED"
    --batch-size "$BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --temperature "$TEMPERATURE"
    --top-k "$TOP_K"
    --experiment-name "$experiment_name"
    --overwrite
  )

  if [[ "$warm_count" != "0" ]]; then
    common_args+=(--warm-start-memory-file "$WARM_START_MEMORY_FILE")
  fi

  for policy in $run_policies; do
    condition_name="online_construction_${policy}"
    if [[ "$WARM_START_ONLY" == "1" ]]; then
      condition_name="${condition_prefix}_w${warm_count}"
    fi
    python -m ProcedureMem.eval_alfworld \
      --schedule-policy "$policy" \
      --condition-name "$condition_name" \
      "${common_args[@]}"
  done

  if [[ "$WARM_START_ONLY" != "1" && "$RUN_RANDOM" == "1" ]]; then
    python -m ProcedureMem.eval_alfworld \
      --schedule-policy random \
      --scheduler-seed "$SCHEDULER_SEED" \
      --condition-name "online_construction_random_seed${SCHEDULER_SEED}" \
      "${common_args[@]}"
  fi

  result_root="ProcedureMem/Alfworld/results/paired/${experiment_name}"
  RESULT_ROOTS+=("$result_root")
  echo "Completed warm-start count ${warm_count}: ${result_root}"
done

echo "Serial sweep complete. Result roots:"
printf '  %s\n' "${RESULT_ROOTS[@]}"
