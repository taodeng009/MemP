#!/usr/bin/env bash

set -e

# Fixed feasibility benchmark: MemP-300, 10 ALFWorld tasks, OpenMem reranker 4B.
python -m ProcedureMem.benchmark_memp_reranker \
  --task-manifest ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n10.json \
  --memory-dir ProcedureMem/memory/alfworld \
  --output-dir ProcedureMem/Alfworld/results/reranker_smoke/valid_unseen_seed42_n10 \
  --candidate-k 20 \
  --top-n 10 \
  --warmup-runs 1 \
  --repeats 3 \
  --rerank-model memos-reranker-4b \
  --overwrite

echo "Summary: ProcedureMem/Alfworld/results/reranker_smoke/valid_unseen_seed42_n10/summary.json"
