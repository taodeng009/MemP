# ALFWorld Cloud MemP + Reranker 简洁实现计划

更新时间：2026-08-13

## 1. 目标

在现有 Cloud MemP workflow memory 的 FAISS similarity search 后增加 OpenMem reranker，形成“两阶段检索”：

```text
任务 query → FAISS 初筛 candidate-k → OpenMem rerank → 选取 top-n → 注入 Agent prompt
```

目标是在增加 Cloud retrieval 时延的前提下，验证 reranker 是否能够提高 ALFWorld 任务成功率。现有 `memory` 条件保持不变，新方案使用独立条件 `memory_rerank`。

## 2. API 配置

使用 OpenMem Rerank Memory API：

- 接口：`POST ${MEMOS_BASE_URL}/rerank`
- 默认地址：`https://memos.memtensor.cn/api/openmem/v1`
- 鉴权：`Authorization: Token ${MEMOS_API_KEY}`
- 本实验使用模型：`memos-reranker-4b`
- 输入：`query`、`documents`、`top_n`
- 限制：候选文档总计不超过 8k tokens
- 输出：按 `relevance_score` 降序排列的原候选 `index`

在 `.env.example` 中增加：

```dotenv
MEMOS_API_KEY=
MEMOS_BASE_URL=https://memos.memtensor.cn/api/openmem/v1
MEMOS_RERANK_MODEL=memos-reranker-4b
MEMOS_RERANK_TIMEOUT=30
```

真实 API key 不写入代码、配置文件或实验结果。

## 3. 前置可行性测试

在完整接入评测框架之前，先编写一个独立、最小化的 smoke benchmark，使用当前 MemP-300、固定的 10-task manifest 和真实 ALFWorld task goal 测试 reranker 在本系统中的实际表现。

测试流程：

1. 创建并固定一份 10-task manifest，尽量覆盖主要 task family，后续重复测试均使用相同任务和顺序；
2. 对每个 task goal 使用现有 FAISS similarity search 召回固定数量的 workflow；
3. 将相同候选交给 OpenMem reranker，候选文本包含历史 task goal 和 workflow；
4. 对比 rerank 前后的 Top-1、Top-N 排名和集合变化；
5. 确认 reranker 能稳定返回、排序确实发生变化且没有明显错误。

该测试只验证接口可用性和基本排序行为，不进行系统性的 workflow 人工标注，也不运行完整 Agent。reranker 是否有效最终由后续 SR 实验判断。

### 3.1 时延对比

在相同 query 和候选设置下分别测量：

- `similarity_latency_ms`：一次 FAISS similarity search 的端到端时延；
- `rerank_api_latency_ms`：OpenMem `/rerank` 请求时延；
- `rerank_pipeline_latency_ms`：FAISS 初筛加 reranker 的总时延；
- `rerank_added_latency_ms`：相对于纯 similarity search 增加的时延。

测试要求：

- 使用同一组固定 task goal；
- 先预热 embedding、FAISS 和 HTTP 连接，再进行正式计时；
- 每个配置重复运行多次，分别报告 mean、median 和 P95；
- 区分本地 FAISS 时间、网络/API 时间和完整检索流水线时间；
- 记录 reranker 模型、`candidate_k` 和 `top_n`；
- 正式 latency 测试关闭 reranker response cache。

前置测试通过标准：reranker API 能稳定返回合法排序，Top-1 或 Top-N 存在变化，没有明显错误，并获得真实的新增时延数据。该阶段不要求 reranker 一定优于 similarity search。

## 4. P0 实现范围

### 4.1 新增 reranker 客户端

新增 `ProcedureMem/reranker.py`，负责：

- 构造 `/rerank` 请求；
- 设置鉴权、模型和超时；
- 校验 HTTP 状态、响应字段和候选 index；
- 返回 rerank score 和请求延迟。

P0 默认采用 fail-fast：API 超时、非 2xx 或非法响应直接记录为 retrieval error，不静默回退到原 FAISS 排序。

### 4.2 扩展 Cloud MemP 检索

在 `ProcedureMem/memory.py` 中保留现有 `retrieve()`，新增 rerank 检索路径：

1. FAISS 召回 `candidate_k` 条 workflow memory；
2. 将每条候选格式化为包含历史任务和 workflow 的文本；
3. 调用 OpenMem reranker；
4. 根据返回的原始 `index` 映射回 LangChain `Document`；
5. 返回 rerank 后的前 `top_n` 条。

候选文本建议使用：

```text
Task goal: <historical query>
Reusable workflow: <workflow>
```

reranker 必须看到 workflow 内容，避免仅重复比较 task-goal similarity。

### 4.3 扩展评测入口

在 `ProcedureMem/eval_alfworld.py` 和 condition 定义中增加：

```text
memory_rerank
```

新增参数：

- `--rerank-model`
- `--rerank-candidate-k`，默认 20
- `--rerank-top-n`，默认 10
- `--rerank-timeout`，默认 30 秒

`candidate_k` 表示 FAISS 候选数，`top_n` 表示最终注入数量，两者不得继续共用一个 `top-k` 含义。`inject_memory()` 的 prompt 格式保持不变。

## 5. 结果与成本记录

每条被选中的 memory 记录：

- `vector_rank`
- `vector_score`，类型为 `faiss_l2_distance`，越低越好
- `rerank_rank`
- `rerank_score`，类型为 `openmem_relevance_score`，越高越好
- task name、workflow 和 source

每个任务额外记录：

- `rerank_latency_ms`
- `rerank_changed_top1`
- rerank 前后 Top-N 集合重叠率

`experiment.json` 记录 retrieval pipeline、reranker 模型、`candidate_k`、`top_n`、timeout 和 base URL，但不记录 API key。

API 返回的 request ID 和 token usage 可保留在原始日志中，但不作为 P0 核心记录项或主要分析指标。

## 6. 缓存与可复现性

P0 不要求实现完整的响应缓存和精确回放机制。可以暂不开启 cache，或只提供简单的本地 cache；正式 latency 测试必须关闭 response cache。更完整的候选 hash、响应审计和可复现缓存机制后置到正式系统工程阶段。

## 7. 测试

新增单元测试，至少覆盖：

- 正常响应的 index 映射和排序；
- 空候选不调用 API；
- 超时和非 2xx 响应；
- API key 不进入结果文件；
- `memory` 原有检索行为不变。

先使用 mock HTTP 响应完成单元测试；真实 API smoke benchmark 按第 3 节执行。

## 8. 第一轮实验

在同一份 50-task manifest、MemP-300、相同 Agent 和推理参数下比较：

| 条件 | FAISS 候选 | 最终注入 | Reranker |
|---|---:|---:|---|
| MemP baseline | 10 | 10 | 无 |
| MemP + reranker | 20 | 10 | 4B |

若 10-task 前置测试和 50-task 实验出现正向信号，再运行完整 134-task 实验。主要评价指标为 SR、positive/negative flips 和 retrieval latency；平均步数可随结果记录，但不作为第一轮核心指标。

## 9. 完成标准

P0 在满足以下条件后完成：

1. 完成小规模真实 reranker 排序测试和 similarity/reranker 时延对比；
2. `memory` baseline 行为和历史结果格式不被破坏；
3. `memory_rerank` 可独立运行并正确注入 rerank 后的 workflow；
4. 检索排名、延迟和错误可逐任务追踪；
5. mock 测试及真实 API smoke benchmark 通过；
6. 完成固定 50-task 的公平配对实验并生成汇总结果。
