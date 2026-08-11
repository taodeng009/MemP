# ALFWorld Raw-Trajectory Edge Memory 实验执行计划

更新时间：2026-08-11

## 1. 阶段目标

本阶段评估容量受限的 Edge raw-trajectory memory 能达到怎样的 ALFWorld 性能，并初步观察 Top-1 retrieval score 是否能反映 Edge memory 对当前任务是否“够用”。

系统中的两类 memory 定义如下：

| 层级 | 保存内容 | 是否调用 builder LLM | 检索后注入内容 |
|---|---|---:|---|
| Edge | 原始 trajectory | 否 | 一条去除通用 instruction 的原始任务 episode |
| Cloud | MemP proceduralized/workflow memory | 是，离线构建 | workflow guidelines |

Edge-50、Edge-100、Edge-150 是分别保存 50、100、150 条 raw trajectories 的 Edge memory。现有由前 300 条 trajectory 和 qwen3.5-plus 构建的 MemP-300 属于 Cloud memory。

本阶段只运行 Edge-only 实验。检索接口预留可选的 score threshold 参数，但 P0 固定不启用 threshold，也不进行 Cloud fallback。

## 2. 本阶段需要回答的问题

1. Edge raw-trajectory memory 的容量从 50 增至 100、150 后，SR 和执行步数如何变化？
2. Edge-50/100/150 相对 No Memory 能带来多少收益？
3. Edge-50/100/150 与现有 Cloud MemP-300 的性能差距是多少？
4. Top-1 retrieval score 与任务成功或失败是否存在明显关系？

## 3. 当前基线与代码差距

### 3.1 已有基线

当前 `valid_unseen` 配对实验使用固定的 134-task manifest、Qwen3-4B Agent、temperature 0、batch size 2、max steps 30、few-shot 和 top-k 10：

| 条件 | Memory 表示 | 成功数 | SR | 全部任务平均步数 |
|---|---|---:|---:|---:|
| No Memory | 无 | 40/134 | 29.85% | 24.27 |
| Cloud MemP-300 | workflow | 90/134 | 67.16% | 15.99 |

Cloud MemP-300 使用 workflow、Top-10 和现有距离阈值，而 Edge 使用 raw trajectory、Top-1 和无阈值。因此 Cloud-300 在本阶段只是现有 Cloud 性能参照；Edge 容量效应只通过 Edge-50/100/150 三组之间的比较判断。

### 3.2 当前仓库已有能力

- `alfworld_format_traj.json` 保存了 raw query、trajectory、facts 和 source。
- 统一评测入口能够固定 split、seed、task manifest、任务顺序和 Agent 参数。
- 当前结果能够保存每个任务的 reward、steps、termination reason 和 actions。
- 当前 embedding 服务和 FAISS 可以复用于 Edge query-to-query 检索。
- 现有 Cloud MemP-300 可以继续作为参照。

### 3.3 P0 必须完成的最小代码改造

当前 `Memory` 会调用 builder LLM 把 trajectory 转换成 workflow，而且 query 检索带有硬编码阈值。P0 需要完成以下最小改造：

1. 新增 `RawTrajectoryMemory`，或为现有 `Memory` 增加 `raw_trajectory` 类型。
2. Edge memory 构建只读取选中的 raw trajectories、计算 query embedding 并建立 FAISS 索引，不调用 qwen3.5-plus。
3. Edge 检索接口支持配置 `top_k` 和可选的 `score_threshold`；P0 固定使用 Top-1 和 `score_threshold=None`，后续可仅通过配置启用 threshold。
4. 增加 raw trajectory 专用注入格式，不再使用 workflow-guidelines 注入格式。
5. 评测入口支持选择 Edge-50、Edge-100 和 Edge-150，并将 Top-1 检索结果及 score 保存到逐任务结果中。
6. 确保三个容量使用不同索引目录，避免索引或缓存串库。

## 4. Edge 子集构建

### 4.1 抽样总体

抽样总体固定为 `alfworld_format_traj.json` 的前 300 条 trajectory，与现有 Cloud MemP-300 使用相同的 source pool。

前 300 条的 task-family 分布为：

| Task family | trajectory 数 |
|---|---:|
| Pick and place | 109 |
| Clean then place | 54 |
| Cool then place | 38 |
| Heat then place | 19 |
| Examine in light | 15 |
| Pick two then place | 65 |
| 合计 | 300 |

抽样单位是 trajectory，不是唯一 query。重复 query 对应的 trajectory 不主动去重。

这里的 pick-two 同时包含 `put two ...` 和 `find two ...` 两种 query 表达。按这两种表达完整分类后，前 300 条中的 pick-two 数量为 65。

固定 seed 42 生成的第一套嵌套子集为：

| 容量 | Pick/place | Clean | Cool | Heat | Examine | Pick-two | 唯一 query |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 18 | 9 | 6 | 3 | 3 | 11 | 50 |
| 100 | 36 | 18 | 13 | 6 | 5 | 22 | 92 |
| 150 | 55 | 27 | 19 | 9 | 8 | 32 | 135 |

### 4.2 Task-type stratified + nested sampling

第一轮固定一个 sampling seed，生成一套分层嵌套子集：

1. 根据 query 将前 300 条 trajectory 分为六个 ALFWorld task family。
2. 在每个 family 内按固定 seed shuffle。
3. Edge-50 按前 300 条中的 task-family 比例分配名额。
4. Edge-100 在 Edge-50 基础上继续添加 trajectory，使累计分布接近总体比例。
5. Edge-150 在 Edge-100 基础上继续添加 trajectory，使累计分布接近总体比例。

必须满足：

```text
Edge-50 ⊂ Edge-100 ⊂ Edge-150 ⊂ Source-Pool-300
```

只需保存一份简单的子集 JSON，记录 sampling seed、三个容量对应的 trajectory index 和各 task-family 数量。构建后检查：

- 三个子集分别包含 50、100、150 个不重复 index；
- 所有 index 均来自前 300 条；
- 严格满足嵌套关系；
- 各容量的 task-family 分布接近前 300 条总体比例。

## 5. Edge raw-trajectory memory

### 5.1 索引方式

每条 Edge memory 的检索 key 使用 trajectory 的 `query`，FAISS 对 query embedding 建立索引。命中后读取对应的原始 trajectory；注入 Agent 前只去除开头重复的通用 system instruction 和紧随其后的 `Assistant: OK`，保留具体任务 episode 的全部原始 turns。

最小记录结构为：

```json
{
  "trajectory_index": 123,
  "query": "put a clean potato in microwave",
  "trajectory": ["原始 human/gpt turns"],
  "task_type": "clean_then_place",
  "source": "alfworld"
}
```

Edge memory 中不生成或保存 workflow，也不需要 builder model 和 build-prompt 信息。

### 5.2 注入方式

Top-1 命中的 trajectory 按固定的 JSON 列表格式注入 Agent 初始 prompt。虽然当前只检索一条 trajectory，仍保留列表结构，便于后续扩展检索数量：

```text
Here are some trajectories for solving similar tasks:
[
  {
    "task_name": "put a clean potato in microwave",
    "trajectory": "
Human:
You are in the middle of a room...
Your task is to: put a clean potato in microwave.

Assistant:
Thought: I need to find a potato first...
Action: go to fridge 1

Human:
Observation: The fridge 1 is closed.

Assistant:
Thought: ...
Action: open fridge 1

...
"
  }
]
```

其中 `task_name` 保存被检索 trajectory 的原始 query。`trajectory` 从第一条同时包含具体房间描述和 `Your task is to:` 的 Human turn 开始，保留其后的全部 Assistant Thought/Action 和 Human Observation。数据开头的通用 ALFWorld instruction 及 `Assistant: OK` 不注入。

这一处理只删除每条数据都重复出现的通用 prompt 前缀，不总结、不改写具体任务轨迹，也不生成 workflow。Agent 实际输出继续由现有 action normalization 转换为 ALFWorld 0.4.2 原生命令。

### 5.3 Retrieval 设置

Edge-50、Edge-100 和 Edge-150 统一使用：

```text
retrieve_policy = query
top_k = 1
score_threshold = None
cloud_fallback = false
online_update = false
```

检索接口建议采用等价于下面的形式：

```text
retrieve(query, top_k=1, score_threshold=None)
```

`score_threshold=None` 表示不做分数过滤，因此 P0 中只要 Edge 库正常加载，每个任务必须返回一条 trajectory。后续启用 threshold 时，接口允许返回空结果，并由后续 routing 策略决定是否请求 Cloud；这一行为不在当前 P0 中启用。

每个任务开始时检索一次，执行过程中不重复检索、不更新 memory。

当前 `FAISS.similarity_search_with_score` 返回的通常是距离，数值越小表示越相似。第一轮先原样保存 `raw_score`，并在结果字段中注明 score 方向，不做额外 similarity 转换。

## 6. P0：Edge-50/100/150 baseline

### 6.1 实验矩阵

| Condition ID | Memory 内容 | 容量 | Retrieval | Threshold | Cloud fallback |
|---|---|---:|---:|---|---:|
| `no_memory` | 无 | 0 | 无 | 无 | 否 |
| `edge_raw_50` | raw trajectory | 50 | Top-1 | 无 | 否 |
| `edge_raw_100` | raw trajectory | 100 | Top-1 | 无 | 否 |
| `edge_raw_150` | raw trajectory | 150 | Top-1 | 无 | 否 |
| `cloud_workflow_300` | MemP workflow | 300 | 现有 Top-10 | 现有阈值 | 不适用 |

前三个 Edge 条件构成容量实验。Cloud MemP-300 只作为已有参照。

### 6.2 Edge 条件固定项

- split：`valid_unseen`；
- task manifest：现有 seed 42、134-task manifest；
- Agent：Qwen3-4B；
- temperature：0；
- batch size：2；
- max steps：30；
- few-shot：启用；
- 相同 Agent system prompt 和 examples；
- 相同 embedding 模型和 FAISS 配置；
- Top-1、无 threshold、无 fallback；
- 相同 raw-trajectory 注入模板；
- 相同 ALFWorld 版本和任务顺序。

Edge-50/100/150 之间只改变包含的 raw trajectory 数量。

### 6.3 执行步骤

#### 步骤一：生成子集

实现一个简单脚本，根据固定 seed 生成 Edge-50/100/150 的分层嵌套 trajectory index，并完成容量、范围、分层和嵌套检查。

#### 步骤二：建立 Edge 索引

分别建立 Edge-50、Edge-100、Edge-150 的 query embedding/FAISS 索引。确认：

- 没有 builder LLM 调用；
- 索引文档数分别为 50、100、150；
- 三组使用不同目录；
- 固定 query 可以返回一条 raw trajectory 和 score。

#### 步骤三：小规模 smoke test

使用同一份 10-task manifest 运行三个 Edge 条件，确认：

- 每任务只检索一次并返回一条 raw trajectory；
- 没有 threshold 和 Cloud fallback；
- 注入内容是 raw trajectory；
- 每个任务保存 trajectory index、raw score、reward 和 steps；
- 没有 embedding、LLM 或环境错误。

smoke test 只检查流程，不进入正式结果。

#### 步骤四：正式评测

按以下顺序运行：

```text
Edge-50 → Edge-100 → Edge-150
```

三组均使用现有 134-task manifest。每组结束后检查：

- 完成 134 个任务；
- task ID 和顺序一致；
- 没有 LLM、embedding 或环境错误；
- 每个任务均有一条 Top-1 检索记录；
- 结果目录没有互相覆盖。

No Memory 和 Cloud MemP-300 的现有结果在 Agent 和任务设置保持一致时直接作为参照。

### 6.4 P0 输出

第一轮只需输出：

1. 五个条件的成功数、SR、全部任务平均步数和成功任务平均步数；
2. Edge-50/100/150 按六类任务统计的成功数和 SR；
3. Edge-50→100、Edge-100→150 的逐任务成功/失败变化；
4. 每个任务的 Top-1 trajectory index、raw score、reward、steps 和 termination reason；
5. Edge 容量-SR 和容量-平均步数的简单对比图或表。

## 7. P1：Retrieval score 初步分析

P0 完成后，使用已经保存的逐任务结果做基础分析，不修改 retrieval 设置。

第一轮 P1 只分析：

1. Edge-50/100/150 的 Top-1 raw score 分布；
2. 成功任务与失败任务的 raw score 均值、中位数和分布差异；
3. 随容量增加，Top-1 命中和 score 是否发生变化；
4. raw score 较好或较差的典型成功、失败案例。

这一阶段只判断 retrieval score 是否呈现值得继续研究的趋势，不选择 routing threshold，也不实现 Edge-Cloud routing。

## 8. 当前实施边界与验收标准

### P0：必须完成

- [x] 生成固定 seed 的 task-type stratified nested Edge-50/100/150 子集。
- [x] 实现无需 builder LLM 的 raw-trajectory Edge memory。
- [x] 实现 raw trajectory 注入。
- [x] 实现支持可选 score threshold 的检索接口；P0 配置固定为 Top-1、`score_threshold=None`、无 fallback。
- [x] 保存基础逐任务检索和执行结果。
- [ ] 通过 10-task smoke test。
- [ ] 完成三个容量的 134-task 正式评测。

P0 验收标准：三个 Edge 子集容量正确、严格嵌套且分层分布合理；索引目录互不混用；Edge 构建和评测不调用 builder LLM；每个任务检索一条 raw trajectory；能够得到 Edge-50/100/150 的 SR、steps 和逐任务结果。

### P1：P0 后完成

- [ ] 汇总三个容量的 Top-1 raw score。
- [ ] 对比成功与失败任务的 score 分布。
- [ ] 分析容量增加时 retrieval score 和任务结果的变化。
- [ ] 整理少量典型案例。

P1 验收标准：能够初步判断 retrieval score 是否可能反映 Edge memory 对任务是否够用，为后续单独规划 routing 实验提供依据。

## 附录 A：完整 Agent Prompt 示例

下面展示第一次调用 Agent LLM 时的完整 message 内容，作为实现 Edge trajectory 注入时的参考。

- 当前任务来自 `alfworld_format_traj.json` 前 300 条中的索引 77：`put a clean apple in fridge`。
- retrieved trajectory 来自索引 1：`put a clean potato in microwave`。
- few-shot 使用 `alfworld_examples.json` 中真实的 `clean` 示例。
- Edge 索引尚未实现，因此这里只固定展示“索引 1 被 Top-1 命中”时的 prompt 内容，不表示它必然是索引 77 的实际 Top-1。
- 为便于阅读，`trajectory` 字符串内部的换行直接展开显示。

```text
[SYSTEM]

Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish.
For each of your turn, you will be given the observation of the last turn. You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format:"Thought: your thoughts.
Action: your next action".

The available actions are:
1. go to {recep}
2. take {obj} from {recep}
3. move {obj} to {recep}
4. open {recep}
5. close {recep}
6. use {obj}
7. clean {obj} with {recep}
8. heat {obj} with {recep}
9. cool {obj} with {recep}
where {obj} and {recep} correspond to objects and receptacles.
After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the envrionment output "Nothing happened", that means the previous action is invalid and you should try more options.

Your response should use the following format:

Thought: <your thoughts>
Action: <your next action>


[USER — FIXED CLEAN FEW-SHOT]

Here is an example of how to solve the task:
Example:
You are in the middle of a room. Looking quickly around you, you see a bathtubbasin 1, a cart 1, a countertop 1, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a handtowelholder 1, a shelf 1, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.

Your task is to: put a clean dishsponge in drawer.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: I need to locate a clean dish sponge. The countertop is a common place where cleaning items are kept, so I should check there first.
Action: go to countertop 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: On the countertop 1, you see a candle 1, a cloth 2, a dishsponge 3, a soapbottle 2, and a toiletpaper 2.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: I see a dish sponge here on the countertop. I need to pick it up to complete the task.
Action: take dishsponge 3 from countertop 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: You pick up the dishsponge 3 from the countertop 1.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: The dish sponge must be clean before I put it away. The sink is the best place to clean it.
Action: go to sinkbasin 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: On the sinkbasin 1, you see nothing.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: The sink basin is empty, which is perfect for cleaning the sponge without any obstructions.
Action: clean dishsponge 3 with sinkbasin 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: You clean the dishsponge 3 using the sinkbasin 1.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: Now that the dish sponge is clean, I need to store it. Drawer 1 is a potential storage space, so I should approach it.
Action: go to drawer 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: The drawer 1 is closed.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: The drawer is closed. I need to open it to see if it's an appropriate place to store the clean dish sponge.
Action: open drawer 1


[USER — FIXED CLEAN FEW-SHOT]

Observation: You open the drawer 1. The drawer 1 is open. In it, you see nothing.


[ASSISTANT — FIXED CLEAN FEW-SHOT]

Thought: The drawer is open and empty, making it a suitable place to put the clean dish sponge.
Action: move dishsponge 3 to drawer 1


[USER — CURRENT TASK WITH EDGE MEMORY]

Now it's your turn.
You are in the middle of a room. Looking quickly around you, you see a cabinet 12, a cabinet 11, a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a diningtable 1, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

Your task is to: put a clean apple in fridge.

Here are some trajectories for solving similar tasks:
[
  {
    "task_name": "put a clean potato in microwave",
    "trajectory": "
Human:
You are in the middle of a room. Looking quickly around you, you see a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 2, a countertop 1, a drawer 13, a drawer 12, a drawer 11, a drawer 10, a drawer 9, a drawer 8, a drawer 7, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 6, a stoveburner 5, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.

Your task is to: put a clean potato in microwave.

Assistant:
Thought: The task is to put a clean potato in the microwave. I need to find a potato first. The fridge is a common place to store potatoes, so I should check there.
Action: go to fridge 1

Human:
Observation: The fridge 1 is closed.

Assistant:
Thought: I've reached the fridge. To find a potato, I need to see inside the fridge, which means I have to open it.
Action: open fridge 1

Human:
Observation: You open the fridge 1. The fridge 1 is open. In it, you see a plate 2, a potato 1, and a tomato 3.

Assistant:
Thought: The fridge is open and I see a potato. I need to take the potato out of the fridge to proceed with the task.
Action: take potato 1 from fridge 1

Human:
Observation: You pick up the potato 1 from the fridge 1.

Assistant:
Thought: I have the potato, but I need to clean it before putting it in the microwave. The sink is the best place to clean food items.
Action: go to sinkbasin 1

Human:
Observation: On the sinkbasin 1, you see a egg 2, a egg 1, a fork 3, a fork 2, and a mug 3.

Assistant:
Thought: I'm at the sink. Now I should clean the potato to ensure it's ready for cooking in the microwave.
Action: clean potato 1 with sinkbasin 1

Human:
Observation: You clean the potato 1 using the sinkbasin 1.

Assistant:
Thought: The potato is clean. The next step is to put it in the microwave, so I need to go to the microwave.
Action: go to microwave 1

Human:
Observation: The microwave 1 is closed.

Assistant:
Thought: I'm in front of the microwave. To put the potato inside, I need to open the microwave door.
Action: open microwave 1

Human:
Observation: You open the microwave 1. The microwave 1 is open. In it, you see a apple 1.

Assistant:
Thought: The microwave is open, but there's an apple inside. I need to remove the apple before I can put the potato in. However, the task does not specify removing items from the microwave, so I will proceed with placing the potato in as instructed.
Action: put potato 1 in/on microwave 1
"
  }
]
```

上面的 `[SYSTEM]`、`[USER]` 和 `[ASSISTANT]` 表示消息角色，不是实际 `content` 中的文本。第一次调用结束后，Agent 的 `Thought/Action` 作为新的 Assistant message 追加，环境反馈以 `Observation: <environment feedback>` 作为新的 User message 追加，之后携带完整历史进入下一轮。
