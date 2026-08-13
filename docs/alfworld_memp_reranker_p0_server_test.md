# ALFWorld MemP + Reranker P0：服务器测试说明

更新时间：2026-08-13

## 1. 测试目标

在固定的 50-task manifest 上公平比较：

- `memory`：现有 MemP-300 similarity search，Top-10，历史 L2 threshold 0.5；
- `memory_rerank`：MemP-300 FAISS Top-20 候选，经 OpenMem 4B reranker 重排后注入 Top-10。

Agent 模型、任务、顺序、seed、temperature、batch size、max steps 和 few-shot 设置保持一致。核心指标为 SR、positive/negative flips 和 retrieval latency。

## 2. 更新与配置

在服务器仓库根目录执行：

```bash
pip install -r requirements.txt
```

确认已有 MemP-300：

```text
ProcedureMem/memory/alfworld/direct/documents.json
ProcedureMem/memory/alfworld/direct/manifest.json
ProcedureMem/memory/alfworld/vector_cache/
```

在 `.env` 中保留现有 Agent、embedding、ALFWorld 和 memory builder 配置，并增加：

```dotenv
MEMOS_API_KEY=<openmem-api-key>
MEMOS_BASE_URL=https://memos.memtensor.cn/api/openmem/v1
MEMOS_RERANK_MODEL=memos-reranker-4b
MEMOS_RERANK_TIMEOUT=30

# 本轮采用已完成 smoke test 的 0.8；只作用于 reranker 候选池。
MEMP_RERANK_CANDIDATE_SCORE_THRESHOLD=0.8
```

不要将真实 API key 写入 Git、命令行日志或结果目录。

## 3. 运行测试

先运行相关单元测试：

```bash
python -m unittest \
  tests.test_benchmark_config \
  tests.test_reranker \
  tests.test_memory_rerank \
  tests.test_alfworld_experiment \
  tests.test_runtime_config
```

确认已有 baseline 位于：

```text
ProcedureMem/Alfworld/results/paired/valid_unseen_seed42_n50/memory/
```

再运行 50-task reranker 条件：

```bash
bash scripts/run_alfworld_memp_reranker_p0.sh
```

脚本只运行 `memory_rerank`，不会重新运行或覆盖现有 `memory` baseline。P0 脚本显式传入 `--measure-baseline-retrieval-latency`，因此每个任务会额外计时一次原 Top-10 similarity search；该结果不注入 Agent，仅用于与 reranker pipeline 比较时延。这个开关默认关闭，以后普通 `memory_rerank` 运行不会产生额外检索。两个条件使用同一份：

```text
ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n50.json
```

`memory_rerank` 默认 fail-fast：OpenMem 超时、非 2xx 或非法响应会终止该条件，不会静默回退到 similarity 排序。

## 4. 结果检查

结果目录：

```text
ProcedureMem/Alfworld/results/paired/valid_unseen_seed42_n50/
```

主要文件：

```text
memory/summary.json
memory/results.jsonl
memory_rerank/summary.json
memory_rerank/results.jsonl
memory_vs_memory_rerank_comparison.json
memory_vs_memory_rerank_comparison.csv
```

首先检查 `memory_rerank/summary.json`：

- `task_count` 为 50；
- `error_count` 为 0；
- `parameters.rerank_model` 为 `memos-reranker-4b`；
- `parameters.rerank_candidate_k` 为 20；
- `parameters.rerank_top_n` 为 10；
- `parameters.rerank_candidate_score_threshold` 为 0.8；
- `rerank_summary.candidate_count_min` 最好为 20；若小于 20，需要报告实际分布；
- rerank API 和 pipeline latency 均有合法均值。

然后检查 `memory_vs_memory_rerank_comparison.json`：

- `baseline_success_rate`；
- `rerank_success_rate`；
- `absolute_improvement_percentage_points`；
- `failure_to_success`；
- `success_to_failure`；
- `both_success` 和 `both_failure`。
- `baseline_retrieval_latency_ms_mean`、`rerank_pipeline_latency_ms_mean` 和 `rerank_added_latency_ms_mean`。

逐任务 `rerank` 字段记录候选数、FAISS Top-10、rerank Top-10 overlap、Top-1 是否改变、候选搜索/API/pipeline 时延和原始 request ID。`retrieved_memories` 记录最终注入 Agent 的 workflow，以及各自的 vector/rerank rank 和 score。

## 5. 需要回传的内容

完成后提供：

1. 整个 `memory_vs_memory_rerank_comparison.json`；
2. 两个条件的 `summary.json`；
3. `memory_rerank/results.jsonl`；
4. 控制台输出；
5. 如果失败，提供脱敏后的错误栈，不得包含任何 API key。

根据这些结果判断 reranker 的 SR 增益是否足以抵消约 0.4 秒/任务的新增 retrieval latency，并决定是否扩展到完整 134-task manifest。
