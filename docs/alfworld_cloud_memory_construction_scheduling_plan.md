# ALFWorld Cloud Memory Construction Scheduling Feasibility Experiment

## 1. 第一阶段目标

第一阶段只实现一个最小的 **construction scheduling feasibility experiment**，验证以下问题：

> 在完全相同的 candidate memories、evaluation tasks、interval size `B`、construction capacity `C` 和 retrieval 配置下，比较 Random、`oracle_sum` 与 `oracle_coverage`，判断仅改变 workflow memory 的 construction/activation order 是否能够显著影响后续 ALFWorld Agent performance，并验证 set-level coverage 是否能够减少逐 memory 排序造成的冗余 selection。

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

可以对规范化后的 300 条 `(memory_id, query, workflow)` 计算一个简单 candidate pool hash，用于确认 Random、`oracle_sum` 与 `oracle_coverage` 使用同一候选池。除此之外，不增加 source trajectory SHA、prompt SHA、build manifest SHA、复杂 `memory_id` hash 或迁移校验流程。

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

第一阶段实现 Random，并同时保留原 Oracle 与新增 Set-Coverage Oracle，分别命名为 `oracle_sum` 和 `oracle_coverage`。

### 6.1 Random

实验开始时，Random scheduler 使用独立的 `scheduler_seed` 对全部 candidate memory IDs 生成一次固定 permutation。每个 interval 结束后，依次从该 permutation 中取出最多 `C` 条尚未激活的 memory。

要求：

- 相同 candidate pool 和 scheduler seed 必须产生相同 activation order；
- scheduler 使用自己的 random generator，不依赖 Agent、ALFWorld 或全局 random state；
- Random scheduler 不读取 future evaluation queries；
- 正式实验运行多个 Random seeds，用于观察随机 construction order 的均值和方差。

### 6.2 Oracle 的使用边界

两种 Oracle 仅用于验证 construction scheduling 是否存在可优化空间，是使用 privileged future-task information 的 **upper-bound baselines**，不是可部署策略。

两种 Oracle 都在 interval `t` 结束后读取下一个 interval `t+1` 的最多 `B` 个冻结 evaluation task queries，记为 $\mathcal I_{t+1}$。当前 FAISS score 是 L2 distance，数值越小表示越相似。距离使用与 Cloud retrieval 相同的 embedding model，根据 evaluation task query 与 candidate memory `query` 计算。

Oracle 只能使用下一 interval 的冻结 task queries，不能读取 reward、success、steps、termination reason 或 Agent trajectory，也不能根据已经完成的 interval results 调整选择。Oracle scoring 只影响 activation order，不改变 Agent retrieval 配置。

evaluation task query 在实验前随固定 task manifest 一起冻结。Random runner 使用相同的 evaluation task manifest，但不把 future query 传给 Random scheduler。

### 6.3 `oracle_sum`：原逐 Memory 累计距离方法

`oracle_sum` 保留原 Oracle-High 的计算方法，作为对照。对每条尚未 available 的 candidate memory $w_j$，独立计算累计距离：

$$
D_{j,t}^{\mathrm{sum}}
=
\sum_{q_i \in \mathcal I_{t+1}}
d_{\mathrm{FAISS}}(q_i,w_j)
$$

`oracle_sum` 按 $D_{j,t}^{\mathrm{sum}}$ 从小到大排序，选择 Bottom-`C` unavailable memories；累计距离相同时使用 `memory_id` 作为稳定 tie-breaker。选中的 memories 从下一 interval 开始生效。

该方法不考虑 selected memories 之间的互补性，可能在同一 interval 中选出 exact-query duplicates，或把多条 construction slots 集中到同一 task family。

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

### 6.4 `oracle_coverage`：Greedy Marginal Set-Coverage

`oracle_coverage` 使用下一 interval queries 与当前 available memories、pending candidate memories 之间的完整 distance matrix 进行集合级贪心选择。其优化目标是 **newly selected memories 相对于 already-available memory pool 的 marginal semantic coverage**，而不仅仅是本轮 selected memories 之间的互补性。

在 interval `t` 结束、准备为 interval `t+1` 选择新 memory 时，先读取当前已经 available 的 memory 集合 $\mathcal M_t^C$。

当 $\mathcal M_t^C$ 非空时，对下一 interval 的每个 query $q_i$，首先计算 available pool 已经提供的最佳距离：

$$
d_i^{best}
=
\min_{w\in\mathcal M_t^C}
d_{\mathrm{FAISS}}(q_i,w)
$$

然后对每条 pending candidate memory $w_j$，计算它相对于当前 available pool 以及本轮已选 memories 的边际改善：

$$
\Delta_j
=
\sum_{q_i\in\mathcal I_{t+1}}
\max\left(0,\ d_i^{best}-d_{\mathrm{FAISS}}(q_i,w_j)\right)
$$

选择 $\Delta_j$ 最大的 memory，并更新：

$$
d_i^{best}
=
\min\left(d_i^{best},d_{\mathrm{FAISS}}(q_i,w_j)\right)
$$

只有当当前 available pool 为空，即 $\mathcal M_t^C=\varnothing$ 时，才使用累计 distance 最小的 pending candidate 作为第一条 memory：

$$
w_{(1)}
=
\arg\min_{w_j}
\sum_{q_i\in\mathcal I_{t+1}}
d_{\mathrm{FAISS}}(q_i,w_j)
$$

并以该 memory 初始化：

$$
d_i^{best}=d_{\mathrm{FAISS}}(q_i,w_{(1)})
$$

之后进入相同的 greedy marginal selection。重复计算 marginal gain、选择和更新，直到选满 `C` 条 memory，或 pending candidate pool 已耗尽。当 available pool 非空时，本轮第一条 memory 也按 $\Delta_j$ 最大原则选择；仅在 available pool 为空时，第一条 memory 才按累计 distance 最小原则选择。所有 score tie 均使用 `memory_id` 作为稳定 tie-breaker；即使所有剩余 $\Delta_j=0$，只要 candidate pool 尚未耗尽，也继续选择直到达到 `C`。

参考实现：

```python
pending_ids = sorted(pending_ids)

if available_ids:
    best_distance = {
        query_id: min(
            distance[query_id][memory_id]
            for memory_id in available_ids
        )
        for query_id in next_query_ids
    }
    selected_ids = []
else:
    first_id = min(
        pending_ids,
        key=lambda memory_id: (
            sum(distance[query_id][memory_id] for query_id in next_query_ids),
            memory_id,
        ),
    )
    selected_ids = [first_id]
    best_distance = {
        query_id: distance[query_id][first_id]
        for query_id in next_query_ids
    }

while len(selected_ids) < min(construction_capacity, len(pending_ids)):
    remaining_ids = [mid for mid in pending_ids if mid not in selected_ids]
    marginal_gain = {
        memory_id: sum(
            max(0.0, best_distance[query_id] - distance[query_id][memory_id])
            for query_id in next_query_ids
        )
        for memory_id in remaining_ids
    }
    next_id = min(remaining_ids, key=lambda mid: (-marginal_gain[mid], mid))
    selected_ids.append(next_id)
    for query_id in next_query_ids:
        best_distance[query_id] = min(
            best_distance[query_id],
            distance[query_id][next_id],
        )
```

`oracle_coverage` 必须满足：

- 每个 interval 使用下一 interval queries 重新计算 distance matrix 和 greedy selection；
- available pool 非空时，必须先用全部 `available_ids` 初始化每个 query 的 $d_i^{best}$；
- available pool 为空时，才允许用累计 distance 最小的 pending candidate 初始化第一条选择；
- 每加入一条 memory，都必须更新每个 query 的 $d_i^{best}$，再重新计算剩余 candidates 的 $\Delta_j$；
- selection 仅排除已经 available 或本轮已经选中的 IDs，不执行 duplicate-query filtering；
- set-coverage 只改变 activation order，不改变 retrieval threshold、Top-K、`B`、`C`、candidate pool 或 initial available pool。

这里的 set coverage 定义为 newly selected memories 相对于 already-available memory pool，对 future queries 带来的连续 best-distance improvement，不是基于 retrieval threshold 的二值覆盖。candidate pool 中的 duplicate queries 保持原样，以便单独验证集合级目标是否能减少原 `oracle_sum` 的冗余 selection。

## 7. Interval-aware ALFWorld Runner

在现有 `ProcedureMem/eval_alfworld.py` 上增加 `cloud_scheduled` condition，并尽量复用当前 task manifest、ALFWorld environment、Agent runner、workflow retrieval record 和 prompt injection 代码。

建议增加以下参数：

```text
--condition cloud_scheduled
--schedule-policy random|oracle_sum|oracle_coverage
--interval-size B
--construction-capacity C
--scheduler-seed SEED
--candidate-memory-file PATH
```

`oracle_sum` 和 `oracle_coverage` 都从冻结 evaluation task manifest 中读取 task queries，不需要额外的全局 Oracle score 文件。

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
- `oracle_scores` 仅 `oracle_sum` 和 `oracle_coverage` 使用，Random 保存为 `null`；
- `oracle_sum` 记录每条 selected memory 的 `faiss_l2_distance_sum`，并标注 `higher_is_better = false`；
- `oracle_coverage` 按 greedy selection rank 记录 score：available pool 非空时，本轮所有 selected memories 都记录相对于 available pool 加本轮已选集合的 `marginal_gain`；available pool 为空时，第一条记录累计 distance，后续记录 `marginal_gain`。累计 distance 标注 `higher_is_better = false`，marginal gain 标注 `higher_is_better = true`；
- `retrieval_scores` 保持现有单次 FAISS retrieval score 语义，不能和跨 query 求和后的 Oracle distance 直接比较。

### 8.2 Summary

每个 policy 输出：

- overall success rate；
- average execution steps；
- 每个 interval 的 success rate；
- 每个 interval 结束后的 cumulative success rate。

完成 Random、`oracle_sum` 与 `oracle_coverage` 后，输出一个简单比较：

- Random 各 seed 的 SR 和 average steps；
- Random mean/std；
- `oracle_sum` 和 `oracle_coverage` 各自的 SR 和 average steps；
- 两种 Oracle 分别相对 Random mean 的 SR 差值；
- 两种 Oracle 分别相对 Random mean 的 average steps 差值；
- `oracle_coverage` 相对 `oracle_sum` 的结果差异。

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
4. `oracle_sum` 每个 interval 重新计算累计 FAISS L2 distance；`oracle_coverage` 使用当前 `available_ids` 初始化 $d_i^{best}$，仅在 available pool 为空时以累计 distance 最小的 candidate 初始化，并在每轮选择后更新 $d_i^{best}$、重新计算 marginal gains；
5. scheduler 每次选择数量不超过 `C`，且不能重复激活 memory；
6. batch 不跨 logical interval 边界。

测试使用简单 fake documents、fake distance function 和轻量 runner stub，不建立三套测试体系或独立 mock benchmark framework，也不调用真实 Agent LLM、embedding API 或 memory builder。

## 10. 第一阶段代码结构

核心逻辑尽量压缩为：

```text
candidate memory loader
scheduled/available memory wrapper
Random scheduler
Oracle Sum scheduler
Oracle Coverage scheduler
interval-aware ALFWorld runner
minimal result logger
```

建议只新增一个小型核心模块：

```text
ProcedureMem/cloud_scheduling.py
```

其中包含 candidate loader、available memory wrapper、Random scheduler、Oracle Sum scheduler 和 Oracle Coverage scheduler。其余修改集中在：

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
4. 保留并命名 interval-dependent `oracle_sum`，实现 greedy marginal `oracle_coverage` scheduler。
5. 将现有 ALFWorld task loop 改为 interval-aware loop，并保证 batch 不跨 interval。
6. 增加最小 task log、interval/cumulative summary 和 Random vs `oracle_sum` vs `oracle_coverage` 比较。
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
Policies = Random(seed 1/2/3), oracle_sum, oracle_coverage
```

## 12. 第一阶段验收标准

1. Random、`oracle_sum` 与 `oracle_coverage` 使用完全相同的 300 条 candidate memories。
2. 三种 policy 使用完全相同的 valid_unseen tasks、顺序、`B`、`C` 和 retrieval 配置。
3. Agent 只能检索当前 `available_ids` 中的 memory。
4. 每完成 `B` 个 tasks，scheduler 最多激活 `C` 条 pending memory。
5. 新 memory 只从下一 interval 开始生效。
6. 同一 interval 使用固定 available memory snapshot。
7. batch 不跨 logical interval 边界。
8. candidate pool 耗尽后，剩余 tasks 继续正常执行。
9. Random activation order 在相同 scheduler seed 下可复现。
10. `oracle_sum` 针对每个 next interval 动态重新计算累计 FAISS L2 distance 并按升序选择；`oracle_coverage` 以 already-available pool 为 coverage baseline，按 greedy marginal set-coverage 逐条选择，且两者都不读取任务执行结果。
11. `oracle_coverage` 不修改 retrieval threshold、Top-K、`B`、`C`、candidate pool 或 initial available pool，也不加入 duplicate-query filtering。
12. 能输出 overall SR、average steps、interval/cumulative SR，以及 Random、`oracle_sum`、`oracle_coverage` 的简单比较。

第一阶段完成后，需要回答两个紧密相关的问题：在固定 construction capacity 下，future-query-aware Oracle 是否能够相对 Random 获得可观察的 Agent performance 改善；在不改变其他实验设置的条件下，`oracle_coverage` 是否能相对于 already-available memory pool 选择边际覆盖更高的新 memories，并比 `oracle_sum` 产生更低冗余、更互补的 activation order，从而改善后续 retrieval 或 Agent performance。

## 13. 第二阶段：Warm-Start Construction Scheduling

### 13.1 实验目标

第一阶段使用 cold start，即 interval 0 开始时 `available_ids` 为空。第二阶段增加 warm-start 实验，用于比较 Agent 已经持有一组固定 workflow memories 时，Random 与 `oracle_coverage` 对后续 construction capacity 的使用效率。

warm-start 实验开始时，从相同的 300 条 ALFWorld train candidate workflows 中，使用固定 `warm_start_seed` 随机抽取固定数量 `W` 的 memories，作为所有 policy 完全相同的 initial available pool。

必须保持：

```text
same candidate pool
same warm_start_count W
same warm_start_seed
same initial_available_ids
same evaluation task manifest and order
same B and C
same retrieval Top-K and threshold
```

warm-start pool 是实验开始前已经 available 的 memories，不属于任何 interval 的 scheduler construction，因此不消耗每个 interval 的 construction capacity `C`。

### 13.2 初始 Pool 定义

新增参数：

```text
--warm-start-count W
--warm-start-seed SEED
```

建议默认值：

```text
warm_start_count = 0
warm_start_seed = 42
```

其中 `warm_start_count=0` 必须完全保留现有 cold-start 行为和旧命令语义。

warm-start IDs 使用独立的 local random generator 生成：

```python
def select_warm_start_ids(candidate_ids, *, count, seed):
    if count < 0 or count > len(candidate_ids):
        raise ValueError("Invalid warm-start count")
    ids = list(candidate_ids)
    random.Random(seed).shuffle(ids)
    selected = set(ids[:count])
    return tuple(memory_id for memory_id in candidate_ids if memory_id in selected)
```

输出重新按照稳定 candidate order 排列，使不同 policy 的日志、hash 和 FAISS document order 可直接比较。

初始化完成后：

```text
available_ids = initial_available_ids
pending_ids = candidate_ids - initial_available_ids
```

参数语义必须分离：

- `warm_start_seed` 只决定所有 policy 共享的 initial available pool；
- `scheduler_seed` 只决定 Random policy 后续 pending memories 的 activation order；
- warm-start sampling 不能读取或修改全局 random state。

### 13.3 Interval 语义

warm-start 实验执行顺序：

```text
生成并激活 W 条 initial memories
        ↓
重建 interval 0 available FAISS index
        ↓
执行 interval 0 的 B 个 tasks
        ↓
scheduler 从 pending_ids 中选择最多 C 条 memories
        ↓
新 memories 从 interval 1 开始生效
```

interval 0 必须满足：

```text
selected_memory_ids = []
available_memory_count = W
```

`selected_memory_ids` 继续只表示 scheduler 在上一个 interval 边界选择、并于当前 interval 开始生效的新 memories。initial available IDs 单独记录，不能把它们记作 interval 0 的 scheduler selection。

若 candidate pool 尚未耗尽，则 interval `t` 的 available memory count 为：

$$
\left|\mathcal M_t^C\right|
=
\min\left(300,\ W+tC\right)
$$

其余逻辑保持不变：同一 interval 使用固定 snapshot、batch 不跨 interval、新 construction memory 只从下一 interval 生效、candidate pool 耗尽后继续执行剩余 tasks。

### 13.4 Random Policy

Random scheduler 继续对完整 candidate ID list 使用独立 `scheduler_seed` 生成固定 permutation，并在 selection 时根据 `pending_ids` 过滤。由于 initial available IDs 已经从 pending pool 排除，Random 不会重复激活 warm-start memory。

Random 的 construction capacity 仍为每个 interval 最多 `C` 条，warm-start count `W` 不影响该容量。

### 13.5 Oracle-Coverage Policy

现有 `oracle_coverage` 已经接收当前 `available_ids`。warm-start 下，interval 0 结束时首先使用 initial available pool $\mathcal M_0^C$ 计算：

$$
d_i^{best}
=
\min_{w\in\mathcal M_0^C}d(q_i,w)
$$

然后对每条 pending candidate memory 计算：

$$
\Delta_j
=
\sum_i
\max\left(0,d_i^{best}-d(q_i,w_j)\right)
$$

因此当 `warm_start_count > 0` 时，Oracle-Coverage 的第一次 scheduler selection 也必须全部使用：

```text
score_type = faiss_l2_marginal_gain
higher_is_better = true
```

只有 cold start，即 `warm_start_count=0` 时，第一次 selection 的第一条 memory 才使用累计 distance 最小原则初始化。

本阶段不修改 coverage objective、retrieval threshold、Top-K、`B`、`C`、candidate pool，也不加入 duplicate-query filtering。

### 13.6 Runner 修改

在 candidate memories 加载完成后、创建 scheduler 和执行 interval 0 前：

```python
initial_available_ids = select_warm_start_ids(
    memory.candidate_order,
    count=args.warm_start_count,
    seed=args.warm_start_seed,
)

memory.activate(initial_available_ids, interval_id=0)
```

随后直接复用现有 runner：

- interval 0 开始时根据 warm-start pool 重建 FAISS index；
- `pending_ids` 自动排除 initial memories；
- Random 从剩余 pending IDs 中选择；
- Oracle-Coverage 以 warm-start pool 作为 marginal coverage baseline；
- cold start 在 `W=0` 时保持原样。

不新增在线 memory builder、warm-start migration tool 或复杂 pool infrastructure。

### 13.7 日志与 Comparison

在 `experiment.json` 和 summary parameters 中增加：

```text
warm_start_count
warm_start_seed
initial_available_memory_ids
initial_available_pool_sha256
```

`initial_available_pool_sha256` 只对稳定排序后的 initial IDs 计算，用于快速确认不同 policy 使用相同 warm-start pool。

task-level scheduling 日志继续记录：

```text
interval_id
selected_memory_ids
available_memory_count
retrieved_memory_ids
retrieval_scores
oracle_scores
```

scheduling comparison 的 controlled keys 增加：

```text
warm_start_count
warm_start_seed
initial_available_pool_sha256
```

comparison 还应直接检查 `initial_available_memory_ids` 完全一致。若 Random 与 Oracle-Coverage 的 initial pool 不同，必须拒绝生成比较结果。

### 13.8 运行脚本

在 `scripts/run_alfworld_cloud_scheduling.sh` 中增加：

```bash
WARM_START_COUNT="${WARM_START_COUNT:-0}"
WARM_START_SEED="${WARM_START_SEED:-42}"
```

并加入公共 CLI 参数：

```bash
--warm-start-count "$WARM_START_COUNT"
--warm-start-seed "$WARM_START_SEED"
```

warm-start 实验名必须包含 count 和 seed，避免覆盖 cold-start 结果，例如：

```text
cloud_scheduling_valid_unseen_seed42_n50_b10_c5_warm20_ws7
```

第一轮 warm-start 对比只需运行：

```text
Random(seed 1/2/3)
oracle_coverage
```

保留 `oracle_sum` 和旧 `oracle_high` CLI，不删除或改变现有 cold-start policy。

### 13.9 Correctness Tests

在现有聚焦测试中增加：

1. 相同 candidate IDs、`warm_start_count` 和 `warm_start_seed` 生成完全相同的 initial IDs；
2. warm-start IDs 唯一且全部属于 candidate pool；
3. `warm_start_count=0` 保持现有 cold-start 行为；
4. warm-start memories 在 interval 0 即可 retrieve；
5. warm-start memories 从 `pending_ids` 中排除；
6. Random 和 Oracle-Coverage 在相同 warm-start 参数下使用完全相同的 initial IDs；
7. scheduler 不会再次选择 warm-start memory；
8. interval 0 的 `selected_memory_ids=[]` 且 `available_memory_count=W`；
9. pool 未耗尽时，interval 1 的 available count 为 `W+C`；
10. warm-start 下 Oracle-Coverage 第一次 selection 使用 available pool 初始化 $d_i^{best}$，所有新选择均记录 marginal-gain score；
11. comparison 在 initial IDs 或 warm-start pool hash 不一致时拒绝生成；
12. `W=300` 时 pending pool 为空，scheduler 不激活新 memory，剩余 tasks 仍正常执行。

### 13.10 实验配置

Smoke test 至少需要三个 intervals，以覆盖相对于 warm-start pool 的第一次 scheduling 和相对于扩展 available pool 的第二次 scheduling：

```text
N = 21
B = 10
C = 5
W = 10 or 20
Random seeds = 1
Policies = Random, oracle_coverage
```

正式实验建议：

```text
N = 50
B = 10
C = 5
W = fixed warm-start count
warm_start_seed = fixed seed
Random scheduler seeds = 1, 2, 3
Policies = Random, oracle_coverage
```

由于 Agent inference 存在非确定性，每种 deterministic activation order 后续应运行至少 3–5 次 inference repeats，并使用相同 task order、batch size、模型和服务端条件。

### 13.11 Warm-Start 验收标准

1. 默认 `warm_start_count=0` 时，现有 cold-start 行为和结果命名保持兼容；
2. 所有 warm-start policies 使用完全相同的 initial IDs；
3. interval 0 可以检索 warm-start memories；
4. initial memories 不消耗每 interval 的 construction capacity `C`；
5. Random 只随机决定剩余 pending memories 的 activation order；
6. Oracle-Coverage 从 warm-start pool 已提供的 coverage 出发计算 marginal gain；
7. warm-start memory 不能被 scheduler 重复激活；
8. 不修改 retrieval threshold、Top-K、`B`、`C` 或 candidate pool；
9. comparison 能验证 initial pool 完全一致；
10. cold-start 和 warm-start 使用不同结果目录，不发生覆盖。
