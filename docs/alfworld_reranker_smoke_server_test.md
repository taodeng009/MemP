# ALFWorld MemP Reranker 前置测试：服务器运行说明

更新时间：2026-08-13

## 1. 测试目的

使用固定的 `valid_unseen_seed42_n10` manifest，对比现有 MemP similarity search 与 `FAISS Top-20 + OpenMem 4B reranker Top-10` 的排序变化和 retrieval latency。本测试不运行 ALFWorld Agent，不评价 SR。

## 2. 服务器准备

在仓库根目录安装或更新依赖：

```bash
pip install -r requirements.txt
```

确认以下文件已经同步到服务器：

- `ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n10.json`
- `ProcedureMem/memory/alfworld/direct/documents.json`
- `ProcedureMem/memory/alfworld/direct/manifest.json`
- `ProcedureMem/memory/alfworld/vector_cache/`

本测试仍需 embedding 服务，因为每个真实 task goal 都要执行 similarity search。Agent LLM 和 memory builder LLM 不会被调用。

在仓库根目录 `.env` 中配置：

```dotenv
EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_KEY=<embedding-api-key>
EMBEDDING_MODEL_BASE_URL=<embedding-openai-compatible-base-url>

ALFWORLD_DATA=<alfworld-data-root>

MEMOS_API_KEY=<openmem-api-key>
MEMOS_BASE_URL=https://memos.memtensor.cn/api/openmem/v1
MEMOS_RERANK_MODEL=memos-reranker-4b
MEMOS_RERANK_TIMEOUT=30
```

`ALFWORLD_DATA` 中需要包含 manifest 对应的 `traj_data.json`，benchmark 会从中读取真实 task goal。不要把 `MEMOS_API_KEY` 提交到 Git 或复制进结果文件。

## 3. 先运行单元测试

```bash
python -m unittest tests.test_reranker tests.test_alfworld_experiment tests.test_runtime_config
```

预期所有可用测试通过；依赖本地 ALFWorld 数据的测试若按原有条件 skip，不影响本次 smoke test。

## 4. 运行真实 10-task benchmark

```bash
bash scripts/run_alfworld_reranker_smoke.sh
```

脚本固定使用：

- MemP-300；
- 10-task manifest；
- `memos-reranker-4b`；
- FAISS `candidate_k=20`；
- reranker `top_n=10`；
- 预热 1 次；
- 每个 task 正式重复 3 次；
- 不使用 reranker response cache。

预热会产生 1 次 OpenMem 请求，正式测试会产生 30 次请求，共 31 次。若需要先验证密钥和连通性，可临时手动运行 `--repeats 1`，但正式报告仍使用脚本中的 3 次重复。

## 5. 检查结果

输出目录：

```text
ProcedureMem/Alfworld/results/reranker_smoke/valid_unseen_seed42_n10/
```

主要文件：

- `summary.json`：整体排序变化和 latency 的 mean、median、P95；
- `tasks.json`：每个任务的 query、原 Top-10、rerank Top-10、逐次 latency 和原始 API 元数据。

重点检查 `summary.json`：

- `task_count` 等于 10；
- `rerank_model` 等于 `memos-reranker-4b`；
- `top1_changed_count` 或 Top-N overlap 表明排序确实发生变化；
- `latency.similarity`、`latency.rerank_api`、`latency.rerank_pipeline` 和 `latency.rerank_added` 均有合法统计值；
- 运行过程中无超时、非 2xx 或非法响应。

`tasks.json` 不应包含 `MEMOS_API_KEY`。request ID 和 token usage 仅作为原始 API 元数据，不属于主要分析指标。

## 6. 需要回传的结果

完成后提供：

1. `summary.json`；
2. `tasks.json`；
3. benchmark 控制台输出；
4. 若失败，提供错误栈以及 embedding/OpenMem HTTP 状态，但必须删除所有 API key。

拿到结果后，主要回答三个问题：reranker 是否稳定、是否改变 Top-1/Top-N，以及相对 similarity search 增加多少 retrieval latency。
