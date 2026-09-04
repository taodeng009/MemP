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
#
# FIFO vs intra-interval shortest-first, 134-task cold-start:
#   TASK_COUNT=134 CONSTRUCTION_CAPACITY=3 WARM_START_ONLY=0 WARM_START_COUNTS=0 \
#   POLICIES="fifo fifo_shortest_first" \
#   EXPERIMENT_NAME="online_construction_valid_unseen_seed42_n134_b10_c3_fifo_shortest" \
#     bash scripts/run_alfworld_online_construction.sh
#
# Long-horizon exact-retrieval Oracle smoke test:
#   TASK_COUNT=30 CONSTRUCTION_CAPACITY=3 WARM_START_COUNTS=0 \
#   POLICIES="fifo oracle_exact_retrieval" \
#   ORACLE_LOOKAHEAD_HORIZONS="1 3 all_remaining" \
#   EXPERIMENT_NAME="online_exact_oracle_smoke_n30" \
#     bash scripts/run_alfworld_online_construction.sh
#
# Exact-H1 vs Exact-H1 + Historical Utility:
#   TASK_COUNT=250 BATCH_SIZE=1 INTERVAL_SIZE=20 CONSTRUCTION_CAPACITY=5 \
#   WARM_START_COUNTS=0 \
#   POLICIES="oracle_exact_retrieval oracle_exact_retrieval_historical_utility" \
#   ORACLE_LOOKAHEAD_HORIZONS=1 \
#   HISTORICAL_UTILITY_MIN_COUNT=5 HISTORICAL_UTILITY_LAMBDA=1.0 \
#   HISTORICAL_UTILITY_EPSILON=1e-8 \
#   EXPERIMENT_NAME="online_exact_hu_train_seed42_n250_b1_i20_c5" \
#     bash scripts/run_alfworld_online_construction.sh
#
# Normalized Gain + Historical Utility V2 (Coverage and Exact-H1):
#   TASK_COUNT=250 BATCH_SIZE=1 INTERVAL_SIZE=20 CONSTRUCTION_CAPACITY=5 \
#   WARM_START_COUNTS=0 \
#   POLICIES="oracle_coverage_historical_utility_v2 oracle_exact_retrieval_historical_utility_v2" \
#   ORACLE_LOOKAHEAD_HORIZONS=1 ORACLE_RETRIEVAL_THRESHOLD=0.5 \
#   HISTORICAL_UTILITY_MIN_COUNT=5 HISTORICAL_UTILITY_ALPHA=1.0 \
#   HISTORICAL_UTILITY_EPSILON=1e-8 GAIN_NORMALIZATION_EPSILON=1e-8 \
#   EXPERIMENT_NAME="online_normalized_hu_v2_train_seed42_n250_b1_i20_c5" \
#     bash scripts/run_alfworld_online_construction.sh
#
# Normalized Gain + local Historical Top-K V2 patch:
#   TASK_COUNT=134 BATCH_SIZE=1 INTERVAL_SIZE=20 CONSTRUCTION_CAPACITY=5 \
#   WARM_START_COUNTS=0 \
#   POLICIES="oracle_coverage_historical_utility_v2_topk oracle_exact_retrieval_historical_utility_v2_topk" \
#   ORACLE_LOOKAHEAD_HORIZONS=1 ORACLE_RETRIEVAL_THRESHOLD=0.5 \
#   HISTORICAL_UTILITY_MIN_COUNT=5 HISTORICAL_UTILITY_ALPHA=0.5 \
#   HISTORICAL_UTILITY_TOP_K=5 \
#   HISTORICAL_UTILITY_EPSILON=1e-8 GAIN_NORMALIZATION_EPSILON=1e-8 \
#     bash scripts/run_alfworld_online_construction.sh
SPLIT="${SPLIT:-valid_unseen}"
SEED="${SEED:-42}"
TASK_COUNT="${TASK_COUNT:-134}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INTERVAL_SIZE="${INTERVAL_SIZE:-20}"
CONSTRUCTION_CAPACITY="${CONSTRUCTION_CAPACITY:-5}"
WARM_START_COUNT="${WARM_START_COUNT:-0}"
WARM_START_COUNTS="${WARM_START_COUNTS:-$WARM_START_COUNT}"
WARM_START_ONLY="${WARM_START_ONLY:-0}"
WARM_START_SEED="${WARM_START_SEED:-42}"
WARM_START_MEMORY_FILE="${WARM_START_MEMORY_FILE:-ProcedureMem/memory/alfworld/direct/documents.json}"
MAX_STEPS="${MAX_STEPS:-30}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_K="${TOP_K:-3}"
POLICIES="${POLICIES:-oracle_coverage_historical_utility_v2 oracle_exact_retrieval_historical_utility_v2}"
ORACLE_LOOKAHEAD_HORIZONS="${ORACLE_LOOKAHEAD_HORIZONS:-1}"
ORACLE_RETRIEVAL_THRESHOLD="${ORACLE_RETRIEVAL_THRESHOLD:-0.5}"
HISTORICAL_UTILITY_MIN_COUNT="${HISTORICAL_UTILITY_MIN_COUNT:-5}"
HISTORICAL_UTILITY_LAMBDA="${HISTORICAL_UTILITY_LAMBDA:-1.0}"
HISTORICAL_UTILITY_EPSILON="${HISTORICAL_UTILITY_EPSILON:-1e-8}"
HISTORICAL_UTILITY_ALPHA="${HISTORICAL_UTILITY_ALPHA:-0.5}"
HISTORICAL_UTILITY_TOP_K="${HISTORICAL_UTILITY_TOP_K:-5}"
GAIN_NORMALIZATION_EPSILON="${GAIN_NORMALIZATION_EPSILON:-1e-8}"
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

read -r -a ORACLE_HORIZONS <<< "$ORACLE_LOOKAHEAD_HORIZONS"
if [[ "${#ORACLE_HORIZONS[@]}" -eq 0 ]]; then
  echo "ORACLE_LOOKAHEAD_HORIZONS must contain at least one horizon" >&2
  exit 2
fi
for horizon in "${ORACLE_HORIZONS[@]}"; do
  if [[ "$horizon" != "all_remaining" && ! "$horizon" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid Oracle lookahead horizon: $horizon" >&2
    exit 2
  fi
done

for configured_policy in $POLICIES; do
  if [[ "$configured_policy" == "oracle_coverage_historical_utility_v2_topk" \
    || "$configured_policy" == "oracle_exact_retrieval_historical_utility_v2_topk" ]]; then
    if [[ ! "$HISTORICAL_UTILITY_TOP_K" =~ ^[1-9][0-9]*$ ]]; then
      echo "HISTORICAL_UTILITY_TOP_K must be a positive integer" >&2
      exit 2
    fi
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
  default_experiment_name="online_construction_${SPLIT}_seed${SEED}_n${TASK_COUNT}_b${BATCH_SIZE}_i${INTERVAL_SIZE}_c${effective_capacity}${warm_suffix}${mode_suffix}"
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
    policy_horizons=("")
    if [[ "$policy" == "oracle_exact_retrieval" \
      || "$policy" == "oracle_exact_retrieval_historical_utility" \
      || "$policy" == "oracle_exact_retrieval_historical_utility_v2" \
      || "$policy" == "oracle_exact_retrieval_historical_utility_v2_topk" ]]; then
      policy_horizons=("${ORACLE_HORIZONS[@]}")
    fi
    for horizon in "${policy_horizons[@]}"; do
      condition_name="online_construction_${policy}"
      oracle_args=()
      historical_utility_args=()
      if [[ "$policy" == "oracle_exact_retrieval" \
        || "$policy" == "oracle_exact_retrieval_historical_utility" \
        || "$policy" == "oracle_exact_retrieval_historical_utility_v2" \
        || "$policy" == "oracle_exact_retrieval_historical_utility_v2_topk" ]]; then
        horizon_suffix="$horizon"
        if [[ "$horizon" == "all_remaining" ]]; then
          horizon_suffix="all"
        fi
        condition_name="online_construction_${policy}_h${horizon_suffix}"
        oracle_args+=(
          --oracle-lookahead-horizon "$horizon"
          --oracle-retrieval-threshold "$ORACLE_RETRIEVAL_THRESHOLD"
        )
      fi
      if [[ "$policy" == "oracle_exact_retrieval_historical_utility" ]]; then
        historical_utility_args+=(
          --historical-utility-min-count "$HISTORICAL_UTILITY_MIN_COUNT"
          --historical-utility-lambda "$HISTORICAL_UTILITY_LAMBDA"
          --historical-utility-epsilon "$HISTORICAL_UTILITY_EPSILON"
        )
      elif [[ "$policy" == "oracle_coverage_historical_utility_v2" \
        || "$policy" == "oracle_coverage_historical_utility_v2_topk" \
        || "$policy" == "oracle_exact_retrieval_historical_utility_v2" \
        || "$policy" == "oracle_exact_retrieval_historical_utility_v2_topk" ]]; then
        condition_name="${condition_name}_alpha${HISTORICAL_UTILITY_ALPHA}"
        historical_utility_args+=(
          --historical-utility-min-count "$HISTORICAL_UTILITY_MIN_COUNT"
          --historical-utility-epsilon "$HISTORICAL_UTILITY_EPSILON"
          --historical-utility-alpha "$HISTORICAL_UTILITY_ALPHA"
          --gain-normalization-epsilon "$GAIN_NORMALIZATION_EPSILON"
        )
        if [[ "$policy" == "oracle_coverage_historical_utility_v2_topk" \
          || "$policy" == "oracle_exact_retrieval_historical_utility_v2_topk" ]]; then
          condition_name="${condition_name}_htop${HISTORICAL_UTILITY_TOP_K}"
          historical_utility_args+=(
            --historical-utility-top-k "$HISTORICAL_UTILITY_TOP_K"
          )
        fi
      fi
      if [[ "$WARM_START_ONLY" == "1" ]]; then
        condition_name="${condition_prefix}_w${warm_count}"
      fi
      python -m ProcedureMem.eval_alfworld \
        --schedule-policy "$policy" \
        --condition-name "$condition_name" \
        "${oracle_args[@]}" \
        "${historical_utility_args[@]}" \
        "${common_args[@]}"
    done
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
