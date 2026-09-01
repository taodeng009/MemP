# ALFWorld Historical Utility Corrected Exact Retrieval 实现计划

**日期：2026-08-31**

## 1. 研究目标

在现有 online memory construction 的 Exact Retrieval Oracle 基础上，引入历史 memory 的实际使用效果，对 pending trajectories 的 construction priority 进行修正。

核心评分为：

$$
\boxed{
Score_j
=
R_j^{retrieval}
\left(1+\lambda\hat H_j\right)
}
$$

其中：

- \(R_j^{retrieval}\) 是现有 Exact Retrieval Oracle 对 pending trajectory \(\tau_j\) 计算的 greedy marginal retrieval gain；
- \(\hat H_j\) 是 source task name/query 与当前 pending task name/query 相似的历史 memories 的使用效果；
- \(\lambda\) 控制 historical utility correction 的强度。

策略优先构建：

1. 未来更可能进入真实 Top-K threshold retrieval 的 trajectories；
2. task name/query 与过去实际使用效果较好的 memories 的 source task name/query 更相似的 trajectories。

本阶段新增独立策略，不改变现有 `oracle_exact_retrieval`、`oracle_coverage`、FIFO 或其他 policy 的语义。

## 2. 新策略名称

新增 online construction policy：

```text
oracle_exact_retrieval_historical_utility
```

该策略继承现有 Exact Retrieval Oracle 的 future-query window、Top-K、threshold 和 greedy marginal selection 逻辑，仅在每一步 greedy selection 中加入 historical utility correction。

第一版默认与 Exact-H1 比较：

```text
--oracle-lookahead-horizon 1
```

## 3. 因果时间边界

Historical utility 必须只使用当前 construction 时刻之前已经完成的 tasks。

每个 interval 的执行顺序为：

```text
使用当前 available memory 执行 interval t 的 tasks
                    ↓
记录 interval t 的 retrieval 和 reward
                    ↓
更新 historical memory retrieval/success counts
                    ↓
接收 interval t 新产生的 successful trajectories
                    ↓
为 interval t 结束后的 construction 计算新策略分数
                    ↓
选择并构建 memories，下一 interval 生效
```

因此：

- interval \(t\) 的 outcomes 可以用于 interval \(t\) 结束后的 construction；
- 不能读取尚未执行的 task reward、Agent trajectory 或 future retrieval outcome；
- Exact Retrieval 部分仍按现有 Oracle 设计读取 configured future query window；
- Historical Utility Correction 本身只读取历史执行结果。

## 4. 在线 Historical Utility 统计

### 4.1 Memory-level counters

`OnlineConstructionController` 为每条已经 constructed 且 available 的 online memory 维护：

```text
retrieval_count
success_count
```

对于每个已完成 task：

- 若 memory \(m_i\) 出现在该 task 的实际 retrieval context 中，则 `retrieval_count += 1`；
- 若该 task 同时成功，则 `success_count += 1`；
- 同一 task 对同一 memory 最多计数一次；
- 未被实际 retrieve 的 memory 不更新。

Historical utility 定义为：

$$
H_i
=
\frac{S_i}{R_i}
$$

其中：

- \(R_i\) 为该 run 内截至当前时刻的 retrieval count；
- \(S_i\) 为这些 retrieval tasks 中的 success count。

不同 runs 或实验之间不共享 counters 和 historical reference pool。

### 4.2 可靠历史 memory 门槛

所有 available memories（包括 warm-start 和 online constructed memories）只要满足以下条件，就可以作为 historical references：

$$
R_i \ge R_{min}
$$

第一版默认：

```text
R_min = 5
```

低于门槛的 memory 仍继续累计历史数据，但不参与当前 correction。

Warm-start memory 与 online constructed memory 使用相同规则：正常参与 Agent retrieval，并在被实际 retrieve 后累计 `retrieval_count` 和 `success_count`；达到门槛后正常进入 HU reference pool，不要求可追溯的 source trajectory。

## 5. Task Name/Query 表示

Historical utility transfer 只比较 task name/task query，不比较 raw trajectory，也不比较已经构建好的 workflow。该信息在 construction scheduling 时已经可观测。

### 5.1 Pending task query

Pending queue 中的 `OnlineTrajectoryCandidate.query` 保存当前 candidate 的 task goal/name。第一版直接使用经过 `strip` 等基础规范化后的 query 文本作为 pending transfer text，不加入 trajectory steps、Thought/Action、Observation、workflow content 或 workflow embedding。

### 5.2 Historical reference task query

Online constructed memory 通过 `source_queue_id` 关联 source candidate 的原始 task query；warm-start memory 直接使用其已有的 task `query` metadata。Controller 对两者统一形成：

```text
memory_id -> reference_task_query
```

Online memory 即使已经从 pending queue 移除，其 source task query 仍保留供后续 HU transfer 使用。Warm-start memory 不需要 source trajectory；其已有 task query 就是 HU transfer text。不同 runs 不共享该映射。

### 5.3 Query embedding 与距离

Pending 和 historical source tasks 使用同一个 query normalization 和 embedding model。计算 squared-L2 distance：

$$
d_{ji}
=
\lVert e(q_j)-e(q_i)\rVert_2^2
$$

其中 \(q_j\) 是 pending candidate 的 task query，\(q_i\) 是 historical memory 的 reference task query。同一 selection interval 内批量 embedding pending queries 与可靠 historical reference queries，避免重复调用 embedding 服务。

## 6. Historical Utility Estimation

设可靠历史 memories 集合为：

$$
\mathcal H_t
=
\{m_i \mid R_i \ge R_{min}\}
$$

对 pending candidate \(\tau_j\) 的 task query，计算：

$$
w_{ji}
=
\frac{1}{d_{ji}+\epsilon}
$$

$$
\hat H_j
=
\frac{\sum_{i\in\mathcal H_t}w_{ji}H_i}
{\sum_{i\in\mathcal H_t}w_{ji}}
$$

第一版固定：

```text
epsilon = 1e-8
```

由于 \(H_i\in[0,1]\)，加权结果同样满足：

$$
\hat H_j\in[0,1]
$$

如果当前不存在可靠 historical references：

$$
\hat H_j=0
$$

此时 correction factor 为 1，新策略退化为原始 Exact Retrieval Oracle。

第一版使用全部满足门槛的可靠历史 memories，不加入 historical nearest-neighbor Top-K、额外 distance threshold、confidence weighting 或 smoothing。

## 7. Corrected Greedy Selection

现有 Exact Retrieval Oracle 会根据真实 FAISS squared-L2、Top-K 和 retrieval threshold，在每一步 greedy selection 中计算 pending candidate 的 marginal retrieval gain：

$$
R_j^{retrieval}
$$

新策略对每条 remaining candidate 计算：

$$
C_j=1+\lambda\hat H_j
$$

$$
Score_j=R_j^{retrieval}C_j
$$

第一版固定：

```text
lambda = 1.0
```

每一步选择 `Score_j` 最大的 candidate，并使用稳定 memory/queue ID 作为 tie-break。

选择一条 candidate 后：

1. 按现有 Exact Retrieval 逻辑更新每个 future query 的模拟 Top-K retrieval state；
2. 重新计算 remaining candidates 的 marginal retrieval gain；
3. \(\hat H_j\) 在同一 construction interval 内保持不变；
4. 重新计算 adjusted score 并选择下一条 candidate。

如果所有 candidates 的 base marginal retrieval gain 都为 0，则 adjusted scores 仍为 0，并按稳定 ID tie-break；historical correction 不绕过 base retrieval objective。

## 8. CLI 与参数校验

新增 policy：

```text
--schedule-policy oracle_exact_retrieval_historical_utility
```

第一版不把 historical hyperparameters 暴露为 CLI 参数，内部固定：

```text
R_min = 5
lambda = 1.0
epsilon = 1e-8
```

只保留已有 Exact Retrieval 所需的基本合法性检查，例如 lookahead horizon、retrieval Top-K 和 threshold。内部 scoring 接口仍允许测试时传入 `lambda=0`，用于验证能够退化为原始 Exact Retrieval。

## 9. 日志与结果记录

只在现有 construction event 中增加判断结果所必需的最小字段：

```text
adjusted_score
base_retrieval_value
historical_utility_estimate
historical_reference_count
selection_rank
```

不新增 detailed interval-level historical 汇总，不记录 `historical_weight_sum`，也不设计复杂 `score_type` 或完整 provenance。固定的 `R_min`、`lambda` 和 `epsilon` 只需在实现注释或简单 run metadata 中注明一次。

## 10. 代码修改范围

预计修改：

```text
ProcedureMem/online_construction.py
ProcedureMem/cloud_scheduling.py
ProcedureMem/eval_alfworld.py
tests/test_online_construction.py
```

主要职责：

- `online_construction.py`：维护因果 historical counters、reference task query 映射、query distances 和 controller 集成；
- `cloud_scheduling.py`：实现 corrected Exact Retrieval greedy scoring；
- `eval_alfworld.py`：新增 policy 入口、interval outcome 更新和最小日志；
- tests：只覆盖核心数学正确性、因果 counter 更新和退化行为。

第一版不扩展完整 comparison framework，也不增加复杂 shell orchestration。Exact-H1 和 Exact-H1+HU 可以分别手动运行并比较结果。

## 11. 测试计划

第一版只保留四组核心 correctness tests：

1. **Counter 更新**：被 retrieve 的 memory 增加 `retrieval_count`；成功 task 同时增加 `success_count`；失败 task 不增加 success；同一 task 重复 memory ID 只计一次。
2. **门槛过滤**：\(R_i<5\) 不参与，\(R_i=5\) 开始参与；达到门槛的 warm-start memory 正常进入 reference pool；无可靠 reference 时 \(\hat H_j=0\)。
3. **\(\hat H_j\) 计算**：用一个简单 fixture 验证：

$$
\hat H_j
=
\frac{\sum_i H_i/(d_{ji}+\epsilon)}
{\sum_i1/(d_{ji}+\epsilon)}
$$

4. **退化行为**：把内部 `lambda` 设为 0 时，selection 与相同输入下的原始 Exact Retrieval 完全一致。

不为 near-zero distance、多种 reference 组合、复杂 CLI 互斥、comparison framework 或完整 provenance 增加额外测试。

## 12. 第一轮实验设计

在相同 train250 配置下比较：

```text
oracle_exact_retrieval_h1
oracle_exact_retrieval_historical_utility_h1
```

固定：

```text
Split: train
Tasks: 250
Batch size: 1
Interval size: 20
Construction capacity: 5
Top-K: 3
Retrieval threshold: 0.5
Lookahead horizon: 1
Historical min count: 5
Historical lambda: 1.0
Historical epsilon: 1e-8
Warm start: 0
```

第一轮只运行这组固定 historical 参数，不进行 \(\lambda\) 或 \(R_{min}\) sweep，也不自动生成多策略 comparison 汇总。

主要结果：

- final success rate；
- task-level gained/lost flips；
- 两个 runs 的 selected candidate 是否发生变化；
- 发生变化时对应的 base retrieval value、\(\hat H_j\) 和 adjusted score。

第一版不分析 average steps、constructed memory count 或 historical reliable memory 数随 interval 的曲线。

## 13. 预期结论边界

若新策略优于 Exact-H1，可以说明 historical memory outcomes 为 retrieval-based construction priority 提供了额外有效信号。

若 adjusted score 明显改变 selection，但 task SR 不提高，则说明历史 observational utility 仍不能可靠近似 candidate downstream utility，可能受到以下因素影响：

- historical utility 的时间预测能力较弱；
- task-query similarity 与 constructed workflow utility 不一致；
- Top-K co-retrieval credit assignment；
- task distribution 随时间变化；
- builder 和 Agent inference 波动。

该策略仍然是 privileged baseline，因为 \(R_j^{retrieval}\) 来自 future queries。Historical correction 本身是 causal、只使用过去 outcomes，但整个 policy 不能描述为完全 deployable scheduler。
