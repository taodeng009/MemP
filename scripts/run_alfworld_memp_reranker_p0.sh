#!/usr/bin/env bash

set -e

# Fixed P0 comparison: existing MemP-300 baseline vs OpenMem 4B reranking.
MANIFEST="ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n50.json"
EXPERIMENT_NAME="valid_unseen_seed42_n50_memp_reranker_p0"

COMMON_ARGS=(
  --split valid_unseen
  --seed 42
  --task-manifest "$MANIFEST"
  --batch-size 2
  --max-steps 30
  --temperature 0
  --experiment-name "$EXPERIMENT_NAME"
  --overwrite
)

# Historical workflow-memory baseline: Top-10 with its existing L2 threshold 0.5.
python -m ProcedureMem.eval_alfworld \
  --condition memory \
  --top-k 10 \
  "${COMMON_ARGS[@]}"

# P0 Cloud retrieval: FAISS Top-20 candidate pool, OpenMem 4B reranker, inject Top-10.
# MEMP_RERANK_CANDIDATE_SCORE_THRESHOLD is read from the repository .env.
python -m ProcedureMem.eval_alfworld \
  --condition memory_rerank \
  --rerank-candidate-k 20 \
  --rerank-top-n 10 \
  --rerank-model memos-reranker-4b \
  "${COMMON_ARGS[@]}"

echo "Results: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}"
echo "Comparison: ProcedureMem/Alfworld/results/paired/${EXPERIMENT_NAME}/memory_vs_memory_rerank_comparison.json"
