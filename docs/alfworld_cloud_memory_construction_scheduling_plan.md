# ALFWorld Cloud Memory Construction Scheduling Feasibility Experiment

## 1. 第一阶段目标

第一阶段只实现一个最小的 **construction scheduling feasibility experiment**，验证以下问题：

> 在完全相同的 candidate memories、evaluation tasks、interval size `B`、construction capacity `C` 和 retrieval 配置下，比较 Random 与 Oracle-High，判断仅改变 workflow memory 的 construction/activation order 是否能够显著影响后续 ALFWorld Agent performance。

本阶段不实现在线 memory builder、复杂优化算法或通用实验基础设施。所有候选 workflow memories 在实验开始前已经构建完成，实验运行过程中只模拟它们按照不同顺序逐步进入 available memory pool。

## 2. 固定数据划分

```text
Candidate workflow memories: ALFWorld train set
Evaluation tasks: ALFWorld valid_unseen
```

两者来自不同的固定 ALFWorld 数据集划分。本阶段不实现逐 task leakage detection，也不建设复杂的数据 provenance 系统。

## 3. 核心实验设置

实验维护两个逻辑集合：

- `candidate_memories`：完整的 300 条预构建 workflow memories；
- `available_ids`：当前真正允许 Agent retrieval 的 memory IDs；
- `pending_ids`：尚未加入 available pool 的 candidate memory IDs。

实验按 logical interval 运行：

- 每执行 `B` 个 ALFWorld tasks 构成一个 interval；
- 同一个 interval 内使用固定的 available memory snapshot；
- interval 结束后，scheduler 从 `pending_ids` 中选择最多 `C` 条 memory；
- 新选择的 memories 从下一个 interval 开始加入 `available_ids`；
- candidate pool 耗尽后，不再激活 memory，但剩余 evaluation tasks 继续正常执行；
- 每个 task 只在开始时进行一次 retrieval；
- batch 不能跨越 logical interval 边界。

第 0 个 interval 开始时 `available_ids` 为空，因此第一批 `B` 个 tasks 不使用 workflow memory。

例如 `B=10`、`C=3`：

```text
tasks 0-9     available memory count = 0
interval 0 结束：scheduler 选择最多 3 条 memory

tasks 10-19   available memory count = 3
interval 1 结束：scheduler 再选择最多 3 条 memory

tasks 20-29   available memory count = 6
```

最后一个不足 `B` 个任务的 partial interval 正常执行。如果其后没有任务，则不再进行没有实际作用的 memory activation。

## 4. Candidate Memory Loader

第一阶段直接读取当前已有的 300 条 ALFWorld train workflow memories，不新增独立的 candidate memory 构建工具。

每条 memory 只需要以下字段：

```json
{
  "memory_id": "mem_0001",
  "query": "put a clean potato in microwave",
  "workflow": "...",
  "trajectory_index": 1
}
```

其中：

- `memory_id` 按读取后的固定列表位置生成，例如 `mem_0000`、`mem_0001`；
- `query` 是 retrieval key；
- `workflow` 是注入 Agent prompt 的 workflow guideline；
- `trajectory_index` 为可选字段，已有可靠值时保留，没有时不强制恢复。

可以对规范化后的 300 条 `(memory_id, query, workflow)` 计算一个简单 candidate pool hash，用于确认 Random 与 Oracle-High 使用同一候选池。除此之外，不增加 source trajectory SHA、prompt SHA、build manifest SHA、复杂 `memory_id` hash 或迁移校验流程。

## 5. Available Memory Wrapper

实现一个轻量的 scheduled/available memory wrapper，维护：

```python
candidate_memories: dict[str, MemoryDocument]
available_ids: set[str]
pending_ids: set[str]
```

最小接口如下：

```python
class ScheduledWorkflowMemory:
    def activate(self, memory_ids: list[str]) -> None:
        ...

    def rebuild_available_index(self) -> None:
        ...

    def retrieve(self, query: str):
        ...
```

必须保证：

- Agent retrieval 只能访问 `available_ids` 对应的 documents；
- available pool 为空时，retrieval 返回空列表；
- scheduler 不能重复激活同一 memory；
- scheduler 不能激活 candidate pool 之外的 memory；
- 新激活 memory 在当前 interval 内不可见，只能在下一 interval 生效。

第一阶段不要求 incremental FAISS。每个 interval 开始时，可以直接根据当前 `available_ids` 重新建立 FAISS index。candidate pool 规模仅为 300，这种实现更容易验证 available/pending 隔离是否正确。

retrieval 继续复用现有 Cloud workflow memory 配置：

- 相同 embedding model；
- memory `query` 作为索引 key；
- 相同 Top-K；
- 相同 score threshold；
- 相同 workflow prompt 注入格式。

## 6. Scheduling Policies

第一阶段只实现 Random 和 Oracle-High。

### 6.1 Random

实验开始时，Random scheduler 使用独立的 `scheduler_seed` 对全部 candidate memory IDs 生成一次固定 permutation。每个 interval 结束后，依次从该 permutation 中取出最多 `C` 条尚未激活的 memory。

要求：

- 相同 candidate pool 和 scheduler seed 必须产生相同 activation order；
- scheduler 使用自己的 random generator，不依赖 Agent、ALFWorld 或全局 random state；
- Random scheduler 不读取 future evaluation queries；
- 正式实验运行多个 Random seeds，用于观察随机 construction order 的均值和方差。

### 6.2 Oracle-High

Oracle-High 仅用于验证 construction scheduling 是否存在可优化空间，是使用 privileged future-task information 的 **upper-bound baseline**，不是可部署策略。

interval `t` 结束后，Oracle-High 读取下一个 interval `t+1` 的最多 `B` 个冻结 evaluation task queries，记为 $\mathcal I_{t+1}$。当前实现的 FAISS retrieval score 是 L2 distance，数值越小表示越相似。因此，对每条尚未 available 的 candidate memory $w_j$，直接计算累计距离：

$$
D_{j,t}^{\mathrm{oracle}}
=
\sum_{q_i \in \mathcal I_{t+1}}
d_{\mathrm{FAISS}}(q_i,w_j)
$$

其中，$d_{\mathrm{FAISS}}(q_i,w_j)$ 使用与 Cloud retrieval 相同的 embedding model，根据 evaluation task query 与 candidate memory `query` 计算 L2 distance。

Oracle-High 按 $D_{j,t}^{\mathrm{oracle}}$ 从小到大排序，选择 Bottom-`C` unavailable memories；累计距离相同时使用 `memory_id` 作为稳定 tie-breaker。选中的 memories 从下一 interval 开始生效。这里名称中的 `High` 表示预期 relevance 高，并不表示原始 FAISS distance 数值高。

等价地，也可以定义 $S(q_i,w_j)=-d_{\mathrm{FAISS}}(q_i,w_j)$，再选择累计 similarity 最大的 Top-`C`。第一阶段统一采用“累计原始 distance、越小越好、选择 Bottom-`C`”的写法，避免实现时误将最不相关的 memories 排在前面。

参考实现：

```python
oracle_distance[memory_id] = sum(
    faiss_distance(task_query, candidate_memory.query)
    for task_query in next_interval_queries
)

selected_ids = sorted(
    pending_ids,
    key=lambda memory_id: (oracle_distance[memory_id], memory_id),
)[:construction_capacity]
```

Oracle-High 必须满足：

- 每个 interval 使用下一 interval queries 重新计算 ranking；
- 不能为每条 memory 使用一个全局固定 Oracle distance；
- 累计 FAISS L2 distance 必须按升序排列并选择 Bottom-`C`，不能选择数值最大的 memories；
- 只能读取下一个 interval 的 task queries，不能读取 reward、success、steps、termination reason 或 Agent trajectory；
- 不能根据已经完成的 interval results 调整 score；
- Oracle scoring 只影响 activation order，不改变 Agent retrieval 配置。

evaluation task query 在实验前随固定 task manifest 一起冻结。Random runner 使用相同的 evaluation task manifest，但不把 future query 传给 Random scheduler。

## 7. Interval-aware ALFWorld Runner

在现有 `ProcedureMem/eval_alfworld.py` 上增加 `cloud_scheduled` condition，并尽量复用当前 task manifest、ALFWorld environment、Agent runner、workflow retrieval record 和 prompt injection 代码。

建议增加以下参数：

```text
--condition cloud_scheduled
--schedule-policy random|oracle_high
--interval-size B
--construction-capacity C
--scheduler-seed SEED
--candidate-memory-file PATH
```

Oracle-High 从冻结 evaluation task manifest 中读取 task queries，不需要额外的全局 Oracle score 文件。

任务循环按照 interval 边界执行。每次实际 batch 大小为：

```python
min(batch_size, remaining_tasks, remaining_tasks_in_interval)
```

每个 interval 的执行顺序如下：

1. 固定当前 `available_ids` snapshot；
2. 根据该 snapshot 重建 available FAISS index；
3. 执行本 interval 内最多 `B` 个 tasks；
4. 保存 task results，并计算 interval SR；
5. 如果还有下一个 interval，调用 scheduler 选择最多 `C` 条 pending memories；
6. 保存本次 selection；
7. 激活 selected memories，使其从下一 interval 生效。

同一个 interval 内，即使包含多个 batches，也不能修改 available memory snapshot。

## 8. 最小日志与指标

第一阶段不建立完整的 artifact、审计和通用 comparison 系统。只保存能够验证 scheduling correctness 和性能差异的字段。

### 8.1 Task result

每个 task 记录：

```text
task_id
interval_id
policy
selected_memory_ids
available_memory_count
retrieved_memory_ids
retrieval_scores
success/reward
steps
oracle_scores
```

说明：

- `selected_memory_ids` 表示在该 interval 开始时新生效的 memories；第 0 个 interval 为空；
- `oracle_scores` 仅 Oracle-High 使用，记录本 interval 开始时激活 memories 在上一个 interval 边界计算得到的累计 distance；Random 保存为 `null`；
- `oracle_scores` 明确标注 `score_type = faiss_l2_distance_sum` 和 `higher_is_better = false`；
- `retrieval_scores` 保持现有单次 FAISS retrieval score 语义，不能和跨 query 求和后的 Oracle distance 直接比较。

### 8.2 Summary

每个 policy 输出：

- overall success rate；
- average execution steps；
- 每个 interval 的 success rate；
- 每个 interval 结束后的 cumulative success rate。

完成 Random 与 Oracle-High 后，输出一个简单比较：

- Random 各 seed 的 SR 和 average steps；
- Random mean/std；
- Oracle-High SR 和 average steps；
- Oracle-High 相对 Random mean 的 SR 差值；
- Oracle-High 相对 Random mean 的 average steps 差值。

不实现 candidate pool snapshot、多层 pool SHA、next-interval query subset hash、Top-1 frequency、retrieval overlap、activation-frequency statistics 或独立的通用 policy comparison framework。

## 9. 最小 Correctness Tests

第一阶段只增加一个聚焦的测试文件，例如：

```text
tests/test_cloud_scheduling.py
```

覆盖以下六项：

1. 未激活的 candidate memory 不能被 retrieve；
2. 新激活 memory 只能从下一 interval 生效；
3. Random 在相同 scheduler seed 下产生相同 activation order；
4. Oracle-High 每个 interval 使用下一 interval queries 重新计算累计 FAISS L2 distance，并选择 Bottom-`C`；
5. scheduler 每次选择数量不超过 `C`，且不能重复激活 memory；
6. batch 不跨 logical interval 边界。

测试使用简单 fake documents、fake distance function 和轻量 runner stub，不建立三套测试体系或独立 mock benchmark framework，也不调用真实 Agent LLM、embedding API 或 memory builder。

## 10. 第一阶段代码结构

核心逻辑尽量压缩为：

```text
candidate memory loader
scheduled/available memory wrapper
Random scheduler
Oracle-High scheduler
interval-aware ALFWorld runner
minimal result logger
```

建议只新增一个小型核心模块：

```text
ProcedureMem/cloud_scheduling.py
```

其中包含 candidate loader、available memory wrapper、Random scheduler 和 Oracle-High scheduler。其余修改集中在：

```text
ProcedureMem/eval_alfworld.py
ProcedureMem/alfworld_experiment.py
tests/test_cloud_scheduling.py
scripts/run_alfworld_cloud_scheduling.sh
```

不新增独立 candidate builder、迁移工具、通用 scheduler framework、复杂 provenance validator 或通用 policy comparison subsystem。

## 11. 实施顺序

1. 从现有 300 条 workflow documents 加载并分配稳定 `memory_id`。
2. 实现 candidate/available/pending 集合和按 interval 重建的 available FAISS index。
3. 实现 deterministic Random scheduler。
4. 实现 interval-dependent Oracle-High scheduler。
5. 将现有 ALFWorld task loop 改为 interval-aware loop，并保证 batch 不跨 interval。
6. 增加最小 task log、interval/cumulative summary 和 Random vs Oracle-High 比较。
7. 完成六项 correctness tests。
8. 使用同一 valid_unseen task manifest 运行小规模 pilot。

建议 pilot 配置：

```text
Candidate memories = 300 ALFWorld train workflows
Evaluation tasks = fixed ALFWorld valid_unseen manifest
B = 10
C = 5
Top-K = 3
temperature = 0
Policies = Random(seed 1/2/3), Oracle-High
```

## 12. 第一阶段验收标准

1. Random 与 Oracle-High 使用完全相同的 300 条 candidate memories。
2. Random 与 Oracle-High 使用完全相同的 valid_unseen tasks、顺序、`B`、`C` 和 retrieval 配置。
3. Agent 只能检索当前 `available_ids` 中的 memory。
4. 每完成 `B` 个 tasks，scheduler 最多激活 `C` 条 pending memory。
5. 新 memory 只从下一 interval 开始生效。
6. 同一 interval 使用固定 available memory snapshot。
7. batch 不跨 logical interval 边界。
8. candidate pool 耗尽后，剩余 tasks 继续正常执行。
9. Random activation order 在相同 scheduler seed 下可复现。
10. Oracle-High 针对每个 next interval 动态重新计算累计 FAISS L2 distance，按升序选择 Bottom-`C`，并且不读取任务执行结果。
11. 能输出 overall SR、average steps、interval/cumulative SR 和简单的 Random vs Oracle-High 比较。

第一阶段完成后，只需要回答一个问题：在固定 construction capacity 下，Oracle-High 是否能够相对 Random 获得可观察的 Agent performance 改善，从而证明 Cloud workflow memory construction scheduling 存在值得进一步优化的空间。
