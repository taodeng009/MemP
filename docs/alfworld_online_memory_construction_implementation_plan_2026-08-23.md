# ALFWorld Online Cloud Memory Construction Scheduling 实现计划

日期：2026-08-23

## 1. 研究目标

当前 `cloud_scheduled` 实验使用已经完整构建好的 300 条 workflow candidate memories，scheduler 只决定哪些 memory 变为 available，并不在任务执行过程中真正调用 builder。

下一阶段进入 queue-based online construction，研究：

> ALFWorld Agent 持续产生 successful trajectories，而 Cloud workflow builder 每个 interval 只能构建有限数量的 memory。当 trajectory arrival rate 接近或超过 construction service rate 时，FIFO 和 Greedy Novelty 会优先构建哪些 pending trajectories，以及不同调度策略是否会影响后续任务成功率和执行效率？

首轮实现聚焦最小、完整、可验证的 scheduling 主流程：

```text
successful trajectory arrival
        ↓
persistent pending queue
        ↓
FIFO / Greedy Novelty
(Random 为可选诊断 baseline)
        ↓
limited direct workflow construction
        ↓
next-interval memory retrieval
```

首轮固定 construction method、arrival policy 和 construction capacity，只比较 scheduling policy。

## 2. 首轮范围

### 2.1 本轮实现

- success-only trajectory arrival；
- 跨 interval 持续存在的 pending queue；
- 固定 direct workflow builder；
- 每 interval 固定 construction capacity `C`；
- FIFO；
- 可选 Random 诊断 baseline；
- Greedy Novelty；
- cold-start 正式实验；
- warm-start CLI 和同一 initial pool 加载能力；
- 基础 task、queue、construction 和 retrieval 日志；
- 基础结果比较。

### 2.2 本轮暂不实现

- Pending-Queue Coverage；
- Oracle Future Coverage；
- `C ∈ {1,2,3}` capacity sensitivity；
- 3–5 次 independent inference repeats；
- confidence interval 和复杂统计；
- raw trajectory memory；
- round construction；
- failed trajectory admission；
- reflect；
- memory eviction；
- 复杂 builder retry/terminal state；
- 完整 safe-resume；
- 全量 hash 和 provenance audit；
- Edge–Cloud routing。

这些内容只作为后续扩展，不进入首轮实现和验收范围。

## 3. 与现有代码的关系

### 3.1 可直接复用

仓库中已经存在 MemP online 基础路径：

```text
ProcedureMem/run_memp_online.py
ProcedureMem/alfworld_run_update.py
ProcedureMem/memory.py::Memory.update()
```

以下能力可以直接复用：

- `Memory.build()` 的 direct builder；
- ALFWorld direct construction prompt；
- builder model 和 API 配置；
- workflow Document 基本格式；
- documents JSON 保存格式；
- embedding model 与 embedding cache；
- FAISS workflow retrieval；
- workflow prompt injection；
- `eval_alfworld.py` 的 task manifest 和固定 task order；
- `build_interval_batches()`；
- per-task JSON/JSONL 和 summary 框架；
- `select_warm_start_ids()`；
- 现有 Random 和 Greedy Novelty 的核心选择逻辑。

### 3.2 需要重构后复用

#### `Memory.process_trajectory_item()`

保留 direct workflow 构建和 Document metadata 生成，但需要：

- 接收 queue ID、source task ID 和 arrival interval；
- 返回构建结果，不直接修改共享 documents；
- 不再只按 query 去重；
- 记录 memory 的 source queue item 和 available interval。

#### `Memory._save_documents()`

保留 documents 保存、embedding cache 和 FAISS，但拆分为：

```text
save_documents()
rebuild_index()
```

同时支持空 documents：

- cold-start interval 0 允许 memory 为空；
- memory 为空时 `retrieve()` 返回空列表；
- memory 为空时不调用 `FAISS.from_documents()`。

#### `Memory.update()`

不原样调用。当前函数混合了：

- workflow construction；
- hit/success statistics；
- low-success memory deletion；
- validation；
- reflection；
- documents 保存；
- FAISS 重建。

首轮只抽取并使用：

```text
build_document()
append_documents()
save_documents()
rebuild_index()
```

statistics、eviction、validation 和 reflection 均不进入首轮路径。

#### Random 与 Greedy Novelty

复用现有 scheduler 的核心逻辑，但输入从预构建 `CandidateMemory` 改为 pending queue item：

```text
queue_id
query
query embedding
arrival metadata
```

### 3.3 必须新增或修改

- persistent `OnlineConstructionQueue`；
- `OnlineConstructionController`；
- FIFO scheduler；
- success-only arrival admission；
- 纯净 trajectory capture；
- explicit interval 和 construction capacity；
- staged workflow 的 next-interval activation；
- warm-start 与 online update 解耦；
- queue、construction 和 retrieval 基础日志；
- `online_construction` evaluation condition；
- queue correctness tests。

### 3.4 不复用到新实验主路径

- `ScheduledWorkflowMemory` 的完整预构建 candidate activation 状态；
- 原样的 `Memory.update()`；
- `run_memp_online.py` 中“每个 batch 全量更新”的实验入口；
- `result["messages"]` 直接作为 builder trajectory；
- 只按 query 判断重复 memory；
- 失败 trajectory 默认进入 memory construction 的行为。

## 4. 系统时间语义

### 4.1 Logical Interval

任务按照固定 task manifest 和顺序划分为 logical intervals，每个 interval 包含 `B` 个任务。

同一 interval 内：

- 所有任务使用相同 available memory snapshot；
- batch 不能跨 interval；
- 当前 interval 新产生或新构建的 memory 均不能立即参与检索。

### 4.2 Success-Only Arrival

只有成功任务的 trajectory 才进入 Cloud construction queue。

设 interval `t` 的 successful arrivals 为：

```text
A_t
```

失败 trajectory：

- 保存在 task result 中；
- 不上传到 queue；
- 不参与 scheduler selection；
- 不占用 construction capacity。

### 4.3 Persistent Pending Queue

系统维护跨 interval 持续存在的 pending queue：

```text
Q_t
```

interval `t` 结束时：

```text
Q_t_plus = Q_t ∪ A_t
S_t = Scheduler(Q_t_plus, M_t, C)
```

其中：

- `S_t` 最多包含 `C` 条 trajectory；
- scheduler 从完整 pending queue 中选择，而不是只从当前 interval arrivals 中选择；
- 未被选择的 trajectory 保留到后续 interval；
- 后续 arrivals 继续追加到相同 queue；
- 成功构建的 trajectory 从 pending queue 移除；
- 构建失败的 trajectory 保留在 pending queue。

### 4.4 Construction 与 Memory 生效

被选 trajectory 使用固定 direct builder：

```text
successful trajectory
        ↓
direct workflow builder
        ↓
workflow Document
```

成功构建的 workflow：

- 标记 `constructed_after_interval=t`；
- 标记 `available_from_interval=t+1`；
- 先进入 staged documents；
- 只在下一 interval 开始时追加到 available memory 并重建 FAISS。

完整时序：

```text
interval t 开始
    ↓
激活上一 interval staged workflows
    ↓
固定 available memory snapshot
    ↓
执行 B 个任务
    ↓
successful trajectories 加入 persistent queue
    ↓
scheduler 从整个 queue 选择最多 C 条
    ↓
调用 direct builder
    ↓
成功 workflows 从 interval t+1 生效
```

最后一个 interval 的 successful arrivals 仍加入 queue 并记录，但默认不再调用 builder，因为其结果不会影响本次 evaluation 的任务成功率。

## 5. Queue 数据结构

建议定义 `OnlineTrajectoryCandidate`：

```text
queue_id
task_id
task_index
task_type
query
trajectory
arrival_interval
arrival_order
selected_count
last_selected_interval
last_construction_result
```

首轮 queue 只维护两类 item：

```text
pending
constructed
```

不实现复杂的 retry、terminal-failed 或优先级状态机。

Queue ID 不能只由 query 决定。建议基于以下内容生成稳定 ID：

```text
task_id + task_index + trajectory content
```

相同 query、不同环境或 trajectory 必须能够同时保留在 queue 中。

### 5.1 Builder Failure

首轮 failure 处理保持简单：

1. 记录 builder 调用失败和错误信息；
2. 本次 construction slot 已被使用；
3. 对应 trajectory 保留在 pending queue；
4. 下一 interval 继续由正常 scheduler 决定是否再次选择；
5. 不设置 retry 上限；
6. 不实现 terminal-failed 状态；
7. 不实现额外 retry scheduler。

Builder API 内部已有的请求级 retry 可以保留，但 queue 层不增加复杂恢复逻辑。

## 6. Scheduling Policies

### 6.1 FIFO

FIFO 按稳定 arrival order 选择最早到达的 pending trajectories。

排序键：

```text
(arrival_interval, task_index, queue_id)
```

FIFO 是首轮主要 baseline。

### 6.2 Random（可选诊断 Baseline）

Random 从当前完整 pending queue 中选择最多 `C` 条 trajectory。由于首轮核心问题可以通过 FIFO 与 Greedy Novelty 直接比较，Random 不属于正式实验必跑 policy，只在需要检查结果是否落入随机波动范围时运行。

要求：

- 使用独立 `scheduler_seed`；
- 不修改全局 random state；
- 相同 queue snapshot 和 seed 产生相同选择；
- 未选项保留在 queue；
- 如运行 Random，只固定一个 seed，不做多 seed 统计。

### 6.3 Greedy Novelty

Greedy Novelty 复用当前 `GreedyNoveltyScheduler` 的 farthest-first 思路：

- pending trajectory query 为 candidates；
- available workflow queries 为初始 references；
- 本轮已经选择的 trajectory queries 也加入 references；
- 每一步选择到 reference set 最近距离最大的 pending candidate；
- cold-start 且 available pool 为空时使用稳定 queue ID tie-break 初始化。

首轮保留该策略，是为了判断已在预构建 candidate 实验中观察到的 low-density outlier 问题，在 persistent online queue 中是否仍然存在。

### 6.4 首轮 Policy 集合

首轮正式必跑：

```text
fifo
greedy_novelty
```

可选诊断：

```text
random
```

Pending-Queue Coverage 和 Oracle Future Coverage 均后置，不为其预先增加额外实现复杂度。

## 7. Capacity 配置

### 7.1 Pipeline Smoke Test

可以使用：

```text
N=20, B=5, C=5
```

该配置只用于验证：

- arrivals；
- queue；
- builder；
- staged activation；
- retrieval。

它不用于形成 scheduling 结论。

### 7.2 固定 Scheduling 配置

首轮正式 scheduling 只使用一组能形成 backlog 的固定配置，不做 capacity sensitivity。

初始建议：

```text
N=50
B=10
C=2
W=0
```

先用 FIFO 运行一次 pilot，检查：

- 多个 interval 是否出现 `queue_length > C`；
- 是否存在 trajectory 等待超过一个 interval；
- construction slots 是否大部分被使用。

如果该配置不能形成 backlog，只在正式 policy comparison 前调整一次 `B` 或 `C`。配置一旦确定，FIFO、Greedy Novelty 以及任何可选 Random 运行必须使用完全相同的 `N/B/C`，不再做 `C ∈ {1,2,3}` sensitivity。

## 8. Cold Start 与 Warm Start

### 8.1 Cold Start

首轮正式实验先只运行 cold start：

```text
W=0
```

Cold start 下：

- interval 0 的 workflow memory 为空；
- interval 0 不进行 memory retrieval；
- interval 0 successful trajectories 成为第一批 arrivals；
- interval 0 结束后进行第一次 scheduling 和 construction；
- 新 workflow 从 interval 1 开始参与检索。

### 8.2 保留 Warm-Start 接口

虽然首轮先跑 cold start，但实现必须保留：

```text
--warm-start-count W
--warm-start-seed SEED
--warm-start-memory-file PATH
```

Warm-start 实现要求：

- 从 `warm_start_memory_file` 加载预构建 workflow documents；
- 使用固定 `warm_start_seed` 选择 `W` 条 initial memories；
- 所有 policy 使用相同的 selection function；
- 记录 `initial_available_memory_ids`；
- comparison 检查不同 policy 的 initial IDs 完全相同；
- initial memory 从 interval 0 开始可检索；
- initial memory 不进入 pending queue；
- initial memory 不占用 construction capacity；
- warm start 不关闭后续 online arrivals 或 construction。

首轮不要求正式运行 warm-start policy comparison，但必须通过同池加载 correctness test。

### 8.3 初始化与 Online Update 解耦

不继续使用原 `is_cold_start` 同时控制初始化和更新。新模式使用：

```text
initialization_mode = empty | warm
online_update_enabled = true
```

Cold/warm 只决定 interval 0 初始 memory，不决定后续 queue 和 builder 是否工作。

## 9. 纯净 Trajectory Capture

当前 `run_memp_online.py` 使用 `result["messages"]` 作为 builder trajectory，其中包含：

- system prompt；
- few-shot examples；
- retrieved workflows；
- 当前任务交互。

这会污染 online construction input。

需要修改 `ProcedureMem/alfworld_agent.py`，在 `TaskState` 中独立维护：

```text
trajectory_initial_observation
trajectory_events
```

Runner 同时传入：

```text
agent_observation
    允许包含 retrieved workflow

trajectory_initial_observation
    原始环境 observation，不含 retrieved workflow
```

每一步记录：

```text
Agent response
parsed action
environment observation
```

最终 builder trajectory 必须排除 system prompt、few-shot example 和 retrieved workflow。

## 10. Online Construction Controller

建议新增：

```text
ProcedureMem/online_construction.py
```

该模块包含：

```text
OnlineTrajectoryCandidate
OnlineConstructionQueue
OnlineConstructionController
FIFOScheduler
```

并适配现有：

```text
RandomScheduler
GreedyNoveltyScheduler
```

Controller 负责：

- logical interval 生命周期；
- success-only arrival admission；
- persistent queue；
- construction capacity accounting；
- scheduler 调用；
- builder failure logging；
- staged workflow activation；
- cold/warm initialization；
- 必要的 queue 和 construction 日志。

Controller 不重复实现 builder 或 vector store，而是调用重构后的 `Memory` primitives。

### 10.1 Interval 结束处理

每个非最终 interval 结束时：

1. 从 task results 中提取 successful trajectories；
2. 分配稳定 queue IDs；
3. 将 arrivals 追加到 persistent queue；
4. 记录 scheduling 前 queue length 和 IDs；
5. scheduler 从完整 queue 中选择最多 `C` 条；
6. 记录 selected IDs 和 selection order；
7. 逐条调用 direct builder；
8. 成功项从 queue 移除并进入 staged documents；
9. 失败项保留在 queue；
10. 记录 construction result；
11. 下一 interval 开始时激活 staged workflows 并重建 FAISS。

最终 interval 只执行 arrivals admission 和日志保存，不再调用 builder。

## 11. CLI 设计

在 `ProcedureMem/eval_alfworld.py` 中新增：

```text
--condition online_construction
--schedule-policy fifo|random|greedy_novelty
--interval-size B
--construction-capacity C
--scheduler-seed SEED
--warm-start-count W
--warm-start-seed SEED
--warm-start-memory-file PATH
--online-memory-dir PATH
--builder-workers 1
```

首轮固定并写入 experiment parameters：

```text
construction_method = direct
arrival_policy = success_only
```

参数验证：

1. `interval_size >= 1`；
2. `construction_capacity >= 1`；
3. `warm_start_count >= 0`；
4. `warm_start_count > 0` 时必须提供有效 memory file；
5. 如果运行 Random，则使用独立 scheduler seed；
6. cold 和 warm 结果目录分离；
7. `builder_workers=1` 作为首轮默认值；
8. online construction 参数不能用于其他 condition。

## 12. 日志与持久化

首轮不实现完整 safe-resume 和全量 audit，只保存复现实验及诊断主流程所需的信息。

建议保存：

```text
experiment.json
online_trajectories.jsonl
queue_events.jsonl
construction_events.jsonl
results.jsonl
summary.json
```

Warm start 时额外保存：

```text
initial_available_memory_ids
warm_start_count
warm_start_seed
warm_start_memory_file
```

### 12.1 `experiment.json`

至少记录：

- task manifest path；
- task IDs/order；
- Agent model 和主要推理参数；
- builder model；
- direct prompt identity；
- `B` 和 `C`；
- scheduling policy；
- scheduler seed；
- warm-start 参数；
- retrieval Top-K 和 threshold。

### 12.2 Queue Event

至少记录：

```text
interval_id
arrived_queue_ids
queue_length_before_selection
pending_queue_ids_before_selection
selected_queue_ids
queue_length_after_construction
pending_queue_ids_after_construction
```

### 12.3 Construction Event

每个 selected item 记录：

```text
interval_id
queue_id
source_task_id
construction_result: success|failure
workflow or error
constructed_memory_id
available_from_interval
```

## 13. 首轮指标

指标只保留：

### 13.1 Agent Performance

- success rate；
- success count；
- average steps；
- per-interval SR 和 steps。

### 13.2 Queue

- queue length before selection；
- queue length after construction；
- arrivals per interval；
- selected IDs；
- 每条 selected trajectory 的 waiting intervals；
- final queue length。

Waiting time 定义为：

```text
selected_interval - arrival_interval
```

### 13.3 Construction

- 每 interval selected count；
- construction success/failure result；
- constructed memory count。

### 13.4 Retrieval Utilization

- 每任务 retrieved memory IDs；
- retrieved memory 的来源 queue ID；
- online constructed memory retrieval count；
- 被检索过的 constructed memory 数量；
- 从未被检索的 constructed memory 数量。

首轮不计算 confidence interval、复杂公平性指标、复杂 queueing statistics 或多次 inference 汇总。

## 14. Comparison 正确性约束

FIFO 与 Greedy Novelty 生成正式 comparison 前必须确认；如果同时运行 Random，则对 Random 应用相同检查：

1. task manifest、task IDs 和 task order 相同；
2. Agent model、temperature、seed、batch size、max steps 和 few-shot 相同；
3. interval size `B` 和 construction capacity `C` 相同；
4. construction method 均为 direct；
5. arrival policy 均为 success-only；
6. builder model 和 prompt identity 相同；
7. retrieval Top-K、threshold 和 embedding model 相同；
8. warm start 时 initial available memory IDs 完全相同。

Online closed-loop 中，不同 policy 可能改变 Agent success，进而改变后续 arrivals。这属于真实系统反馈，不要求不同 policy 在所有 interval 具有相同 queue contents，但必须保留 arrivals、pending queue 和 selected IDs 日志。

## 15. 测试计划

建议新增：

```text
tests/test_online_construction.py
```

### 15.1 Queue

1. 只有 successful trajectories 进入 queue；
2. failure 不进入 queue；
3. 未选 item 跨 interval 保留；
4. 新 arrivals 追加到已有 queue；
5. scheduler 每 interval 最多选择 `C` 条；
6. 成功构建 item 从 pending queue 移除；
7. 构建失败 item 保留在 queue；
8. 相同 query、不同 trajectory 不会错误去重；
9. waiting time 计算正确。

### 15.2 Interval

1. cold-start interval 0 memory 为空；
2. 当前 interval 构建的 workflow 不能在当前 interval 使用；
3. staged workflow 只从下一 interval 生效；
4. 同一 interval 使用固定 memory snapshot；
5. batch 不跨 interval；
6. 最后 interval arrivals 被记录但不构建。

### 15.3 Scheduler

1. FIFO 使用稳定 arrival order；
2. 如果启用 Random，其在相同 queue 和 seed 下确定；
3. Greedy Novelty 使用 available memory 和本轮已选 item 作为 references；
4. Greedy Novelty cold-start tie-break 稳定；
5. 所有启用的 scheduler 均不读取未来 task query、reward 或 trajectory。

### 15.4 Construction 与 Retrieval

1. 固定使用 direct builder；
2. builder trajectory 不含 retrieved workflow；
3. builder trajectory 不含 few-shot；
4. builder failure 不终止实验；
5. builder failure item 保留在 queue；
6. construction 路径不执行 reflect 或 eviction；
7. empty memory retrieve 返回空；
8. retrieved online memory 可追溯到 source queue ID。

### 15.5 Warm-Start Interface

1. `W=0` 正确表示 cold start；
2. `W>0` 从指定 file 加载 initial pool；
3. 相同 W、seed 和 file 产生相同 initial IDs；
4. FIFO、Greedy Novelty 以及可选 Random 可加载完全相同的 initial IDs；
5. initial memories 在 interval 0 可检索；
6. initial memories 不进入 queue；
7. warm start 不关闭 online arrivals 和 construction；
8. comparison 在 initial IDs 不一致时拒绝生成。

## 16. 实施阶段

### P0：最小 Online Queue Pipeline

实现：

1. 纯净 trajectory capture；
2. 拆分并复用 `Memory` direct builder、Document、save 和 FAISS；
3. persistent queue；
4. success-only arrivals；
5. FIFO；
6. construction capacity；
7. staged workflow activation；
8. cold-start；
9. warm-start CLI 和同池加载接口；
10. 必要日志与测试。

运行：

```text
N=20, B=5, C=5, W=0
```

该运行只验证 pipeline。

### P1：Persistent Backlog 与核心 Scheduler

实现：

1. Greedy Novelty queue adapter；
2. waiting time；
3. FIFO vs Greedy Novelty comparison；
4. fixed backlog-forming configuration；
5. 可选 Random queue adapter，不作为 P1 完成的阻塞项。

先运行 FIFO pilot：

```text
N=50, B=10, C=2, W=0
```

确认形成 backlog 后，固定该配置运行：

```text
fifo
greedy_novelty
```

如需要诊断，再额外运行一次固定 seed 的 `random`。

### P2：首轮 Cold-Start 结果分析

只分析：

- SR；
- steps；
- queue length；
- arrivals；
- selected IDs；
- waiting time；
- construction results；
- retrieval utilization。

不做多 inference repeats、capacity sensitivity 或复杂统计。

### P3：后续工作

主流程稳定后再决定是否加入：

- warm-start 正式实验；
- inference repeats；
- capacity sensitivity；
- Pending-Queue Coverage；
- Oracle Future Coverage；
- raw/round construction；
- failed trajectory admission；
- reflect/eviction；
- safe-resume 和更完整 audit。

## 17. 首轮验收标准

实现完成需要满足：

1. 只有成功 trajectory 进入 queue；
2. pending item 未被选择时不会丢失；
3. scheduler 始终从完整 persistent queue 中选择；
4. 每 interval construction attempts 不超过 `C`；
5. 构建失败被记录且 item 保留在 queue；
6. 新 workflow 只从下一 interval 生效；
7. builder input 不含 retrieved memory 或 few-shot；
8. FIFO 与 Greedy Novelty 使用同一 direct builder；可选 Random 也必须使用相同 builder；
9. 固定正式配置能够形成 backlog；
10. 所有正式比较 policy 的关键实验参数一致；
11. warm-start 三个接口存在且所有 policy 可加载同一 initial pool；
12. 可以输出约定的八类核心指标；
13. 现有 `cloud_scheduled` 行为和测试不受破坏。

首轮结论只回答：

> 在 success-only arrivals、固定 direct construction 和单一有限 capacity 配置下，FIFO 与 Greedy Novelty 是否会产生不同的 Agent performance、queue behavior 和 memory retrieval utilization；Random 仅在需要时作为辅助诊断基线。
