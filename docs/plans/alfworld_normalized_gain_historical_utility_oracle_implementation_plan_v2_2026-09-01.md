# ALFWorld Normalized Gain + Historical Utility Oracle V2 实现计划

**日期：2026-09-01**

## 1. 研究目标

在现有 Historical Utility Corrected Exact Retrieval V1 的基础上，增加第二版独立策略，并同时支持：

- Oracle Coverage；
- Oracle Exact Retrieval。

V2 不再使用 V1 的乘法 correction：

$$
G_j\left(1+\lambda\hat H_j\right)
$$

而是先把当前 greedy selection step 的 base gain 归一化到与 Historical Utility 可比较的数值尺度，再进行加性组合：

$$
\boxed{
Score_j
=
\operatorname{Norm}(G_j)+\alpha\hat H_j
}
$$

其中：

$$
\operatorname{Norm}(G_j)
=
\frac{G_j}{G_{\max}+\epsilon},
\qquad
G_{\max}=\max_{k\in\mathcal C}G_k
$$

- \(G_j\) 是当前 candidate 的 Oracle Coverage 或 Exact Retrieval marginal gain；
- \(\mathcal C\) 是当前 greedy step 尚未选择的 candidates；
- \(\hat H_j\in[0,1]\) 是由历史 memory outcomes 迁移得到的 Historical Utility；
- \(\alpha\ge0\) 直接表示 HU 相对于一个完整 normalized base-gain 单位的影响强度；
- \(\epsilon\) 仅用于数值稳定。

这个归一化不是新的 coverage/retrieval 机制，而是把 base gain 和 HU 放到可比较的数值尺度上。V1 保持不变，V2 通过新增 policy 实现，不改变已有实验语义。

## 2. V2 与 V1 的关系

当 \(G_{\max}>0\) 时，对 V2 score 整体乘以正数 \(G_{\max}+\epsilon\)，其排序与下面的尺度自适应表达完全相同：

$$
G_j+\alpha(G_{\max}+\epsilon)\hat H_j
$$

忽略只用于数值稳定的 \(\epsilon\) 后，可写成：

$$
G_j+\alpha G_{\max}\hat H_j
$$

因此，除以 \(G_{\max}\) 的目的只是让 \(\alpha\) 具有稳定且容易解释的尺度。

需要明确的是，V2 与 V1 的乘法公式并不排序等价：

$$
\underbrace{G_j(1+\lambda\hat H_j)}_{\text{V1}}
\neq
\underbrace{\operatorname{Norm}(G_j)+\alpha\hat H_j}_{\text{V2}}
$$

V1 中 HU 只能按比例放大已有 gain，\(G_j=0\) 时永远保持 0；V2 中 HU 是独立的加性信号，即使 \(G_j=0\)，可靠的 \(\hat H_j\) 仍然可以决定候选顺序。这是 V2 的核心行为差异。

## 3. 新策略名称

新增两个 online construction policies：

```text
oracle_coverage_historical_utility_v2
oracle_exact_retrieval_historical_utility_v2
```

保留且不修改：

```text
oracle_coverage
oracle_exact_retrieval
oracle_exact_retrieval_historical_utility
```

其中最后一个仍表示 V1 乘法 correction，避免新实验覆盖已有结果或改变旧命令的含义。

## 4. Historical Utility 统计与迁移

V2 完全复用 V1 已实现的 run-local Historical Utility 数据和因果时间边界，不重新定义 utility。

### 4.1 Memory-level historical utility

每条 available memory 独立维护：

```text
retrieval_count
success_count
```

并计算：

$$
H_i=\frac{S_i}{R_i}
$$

规则保持不变：

- 每个 task 对同一 retrieved memory 最多计数一次；
- task 成功时，retrieval context 中的每条 memory 都增加一次 success；
- counters、reference pool 均不跨 run 共享；
- warm-start 和 online constructed memories 正常参与 retrieval、统计与 reference pool；
- 只使用当前 construction 时刻之前已经完成的 task outcomes。

### 4.2 Reliable reference pool

只有满足下式的 available memory 进入 HU reference pool：

$$
R_i\ge R_{\min}
$$

默认：

```text
R_min = 5
```

### 4.3 Pending candidate utility estimate

继续只使用 pre-construction 可观测的 task name/query，不使用 workflow、raw trajectory 或 workflow embedding。

对 pending candidate \(\tau_j\) 与 historical reference memory \(m_i\) 的 task-query embedding 计算 squared-L2 distance：

$$
d_{ji}=\lVert e(q_j)-e(q_i)\rVert_2^2
$$

然后计算：

$$
w_{ji}=\frac{1}{d_{ji}+\epsilon_H}
$$

$$
\hat H_j=
\frac{\sum_i w_{ji}H_i}
{\sum_i w_{ji}}
$$

默认：

```text
historical epsilon = 1e-8
```

若没有可靠 historical reference，则所有 pending candidates 的 \(\hat H_j=0\)。第一版 V2 不加入 HU Top-K、distance threshold、retrieval-count confidence weighting 或 smoothing。

## 5. 通用 V2 Greedy Scoring

在 greedy selection 的每一个 rank，执行：

1. 根据当前已 available memories 和本轮已经选中的 candidates，计算每条 remaining candidate 的最新 marginal gain \(G_j\)；
2. 在当前 remaining candidate 集合上计算 \(G_{\max}\)；
3. 计算 \(\operatorname{Norm}(G_j)=G_j/(G_{\max}+\epsilon_G)\)；
4. 使用固定的 interval-level \(\hat H_j\) 计算 \(Score_j\)；
5. 选择 \(Score_j\) 最大的 candidate；
6. 更新 Oracle 状态，进入下一个 selection rank并重新计算 gains 和 \(G_{\max}\)。

默认：

```text
gain normalization epsilon = 1e-8
alpha = 1.0
```

同一个 construction interval 内，historical counters、reference pool 和每条 candidate 的 \(\hat H_j\) 保持不变；base marginal gain、\(G_{\max}\)、normalized gain 和 final score 随 greedy rank 重新计算。

稳定 queue ID 继续作为 final tie-break。

### 5.1 \(\alpha\) 的解释

因为 normalized gain 和 \(\hat H_j\) 都位于 \([0,1]\) 的可比较尺度：

- \(\alpha=0\)：忽略 HU，退化为对应的原始 Oracle 排序；
- \(\alpha=0.25\)：HU 从最低到最高的完整差异，最多抵消 0.25 个 normalized gain；
- \(\alpha=1\)：HU 与 normalized base gain 具有相同的最大数值跨度；
- \(\alpha>1\)：允许 HU 主导更大的 base-gain 差异。

第一版先固定或手动设置 \(\alpha\)，不实现自动调参。

### 5.2 \(G_{\max}=0\) 的行为

当所有 remaining candidates 的 marginal gain 都为 0 时：

$$
\operatorname{Norm}(G_j)=0
$$

因此：

$$
Score_j=\alpha\hat H_j
$$

处理规则为：

- 若存在可靠 HU signal 且 \(\alpha>0\)，选择 \(\hat H_j\) 最大的 candidate；
- 若没有可靠 references，所有 \(\hat H_j=0\)，继续按稳定 queue ID tie-break；
- 若 \(\alpha=0\)，同样退化为稳定 ID tie-break。

这使 V2 能在 Oracle objective 对 candidates 完全无区分能力时，使用过去实际效果作为 secondary construction signal。

## 6. Oracle Coverage + HU V2

对 Oracle Coverage，base gain 沿用现有 marginal distance improvement：

$$
G_j^{cov}
=
\sum_q
\max\left(0,
d_q^{best}-d_{jq}
\right)
$$

其中 \(d_q^{best}\) 是 query \(q\) 相对于当前 available memories 和本轮已选 candidates 的最小距离。

V2 score 为：

$$
\boxed{
Score_j^{cov}
=
\frac{G_j^{cov}}{G_{\max}^{cov}+\epsilon_G}
+\alpha\hat H_j
}
$$

每选择一条 candidate 后，更新每个 future query 的 best distance，然后重新计算 remaining candidates 的 coverage marginal gains 和 normalization scale。

现有 Oracle Coverage 在 available pool 为空时，会先选择 distance sum 最小的 candidate 作为 coverage bootstrap。此时也不可能存在 available historical references，因此所有 \(\hat H_j=0\)。V2 保留原 bootstrap 规则；从建立第一个 coverage anchor 后开始使用上述 normalized marginal-gain scoring。这样可以保持 \(\alpha=0\) 时与原 Oracle Coverage 完全一致，并避免人为定义无 memory baseline distance。

Oracle Coverage 继续读取现有实现定义的 next-interval task-query window，不扩大 oracle 信息范围。

## 7. Oracle Exact Retrieval + HU V2

对 Oracle Exact Retrieval，base gain 继续使用真实 squared-L2、Top-K 和 retrieval threshold 下的 marginal retrieval utility gain：

$$
G_j^{ret}
=
U(\mathcal M\cup\mathcal S\cup\{j\})
-
U(\mathcal M\cup\mathcal S)
$$

其中：

- \(\mathcal M\) 是当前 available memory pool；
- \(\mathcal S\) 是本轮 greedy selection 已选 candidates；
- \(U\) 是现有 Exact Retrieval Oracle 的 Top-K thresholded retrieval utility。

V2 score 为：

$$
\boxed{
Score_j^{ret}
=
\frac{G_j^{ret}}{G_{\max}^{ret}+\epsilon_G}
+\alpha\hat H_j
}
$$

选择后继续按现有 Exact Retrieval 逻辑更新模拟 Top-K state，并在下一 rank 重新计算 marginal gains 与 \(G_{\max}^{ret}\)。lookahead horizon、Top-K 和 threshold 语义均保持不变。

## 8. CLI 和运行脚本

新增 policy choices：

```text
--schedule-policy oracle_coverage_historical_utility_v2
--schedule-policy oracle_exact_retrieval_historical_utility_v2
```

V2 的核心控制参数全部暴露为 CLI，不只保留在 Python constructor 或硬编码常量中：

```text
--historical-utility-alpha 1.0
--historical-utility-min-count 5
--historical-utility-epsilon 1e-8
--gain-normalization-epsilon 1e-8
```

参数含义如下：

| CLI 参数 | 作用 | 适用模式 |
| --- | --- | --- |
| `--historical-utility-alpha` | 控制加性 HU 项的相对影响强度 | 两个 V2 policies |
| `--historical-utility-min-count` | 控制 memory 进入可靠 reference pool 的最小 retrieval count | 两个 V2 policies |
| `--historical-utility-epsilon` | 控制 task-query inverse-distance weighting 的数值稳定项 | 两个 V2 policies |
| `--gain-normalization-epsilon` | 控制 base marginal gain normalization 的数值稳定项 | 两个 V2 policies |

Oracle 本身会影响 base gain 的关键参数也必须由 CLI 保持可配置：

```text
--top-k 3
--oracle-retrieval-threshold 0.5
--oracle-lookahead-horizon 1
```

其中：

- `--top-k` 和 `--oracle-retrieval-threshold` 用于 Exact Retrieval 及其 V2 模式；
- `--oracle-lookahead-horizon` 用于控制 Exact Retrieval future-query window；
- Coverage 及 Coverage+HU V2 继续使用下一 interval queries，不额外引入 lookahead 参数；
- `--oracle-retrieval-threshold` 使用独立名称，避免与其他 evaluation conditions 已有的 `--score-threshold` 混淆。

已有 `--historical-utility-lambda` 只用于 V1，不用于 V2。基本合法性检查：

```text
alpha >= 0
gain-normalization-epsilon > 0
historical-utility-min-count >= 1
historical-utility-epsilon > 0
top-k >= 1
oracle-retrieval-threshold >= 0
oracle-lookahead-horizon >= 1 或 all
```

在 `scripts/run_alfworld_online_construction.sh` 中增加对应 policy 和环境变量映射：

```text
HISTORICAL_UTILITY_ALPHA
HISTORICAL_UTILITY_MIN_COUNT
HISTORICAL_UTILITY_EPSILON
GAIN_NORMALIZATION_EPSILON
TOP_K
ORACLE_RETRIEVAL_THRESHOLD
ORACLE_LOOKAHEAD_HORIZONS
```

shell script 必须把这些值显式传给对应 CLI，便于服务器端直接修改实验配置；不增加复杂 orchestration 或自动 sweep。

## 9. 最小日志

在 V2 construction event 中记录：

```text
base_gain
normalized_base_gain
historical_utility_estimate
historical_reference_count
historical_utility_alpha
adjusted_score
selection_rank
```

这里的 `base_gain` 分别表示 coverage marginal gain 或 exact retrieval marginal gain。记录 normalized gain 是为了直接检查 score 是否严格满足：

$$
adjusted\_score
=
normalized\_base\_gain
+\alpha\times historical\_utility\_estimate
$$

第一版不增加完整 interval-level historical provenance、weight sums、reference 明细或自动 comparison framework。

## 10. 代码修改范围

预计修改：

```text
ProcedureMem/cloud_scheduling.py
ProcedureMem/online_construction.py
ProcedureMem/eval_alfworld.py
scripts/run_alfworld_online_construction.sh
tests/test_online_construction.py
```

职责划分：

- `cloud_scheduling.py`：为 Coverage 和 Exact Retrieval greedy scheduler 增加 V2 normalized additive scoring；
- `online_construction.py`：复用 HU counters/transfer，接入两个 V2 policies；
- `eval_alfworld.py`：增加 policy、全部核心 CLI 参数、validation 和 run metadata；
- shell script：暴露两个 V2 modes，并配置 \(\alpha\)、两个 epsilon、\(R_{min}\) 以及 Exact Retrieval 的 Top-K、threshold 和 lookahead；
- tests：覆盖归一化、零 gain、两个 scheduler 的退化行为和核心集成。

建议抽取一个共享的小函数计算：

```text
normalized_gain + alpha * historical_utility
```

但 Coverage 和 Exact Retrieval 保留各自已有的 marginal-gain 状态更新逻辑，不合并两个 scheduler。

## 11. 核心测试

第一版 V2 保留以下 correctness tests：

1. **Normalization**：给定 gains `{2, 1, 0}`，验证 normalized gains 接近 `{1, 0.5, 0}`。
2. **Additive score**：验证 final score 等于 normalized gain 加 \(\alpha\hat H\)，并能由 HU 改变候选排序。
3. **Zero-gain behavior**：所有 \(G_j=0\) 时，\(\alpha>0\) 按 \(\hat H_j\) 选择；无 reference 或 \(\alpha=0\) 时按稳定 ID。
4. **Coverage degeneration**：`alpha=0` 时，V2 Coverage 与原 `oracle_coverage` 在相同输入下 selection 完全一致，包括 available pool 为空的 bootstrap。
5. **Exact Retrieval degeneration**：`alpha=0` 时，V2 Exact Retrieval 与原 `oracle_exact_retrieval` 在相同输入下 selection 完全一致。
6. **Greedy recomputation**：选择第一条 candidate 后，验证下一 rank 使用更新后的 marginal gains 和新的 \(G_{\max}\)，而不是复用第一次 normalization。
7. **HU reuse**：继续通过已有 tests 验证 counter 更新、\(R_{min}\) 门槛和 task-query distance weighted \(\hat H_j\) 计算。

不扩展复杂边角 fixtures、参数 sweep 或多策略自动统计。

## 12. 第一轮实验设计

分别对两类 Oracle 做成对比较：

```text
oracle_coverage
oracle_coverage_historical_utility_v2

oracle_exact_retrieval
oracle_exact_retrieval_historical_utility_v2
```

首先加入 placebo/退化检查：

```text
V2 alpha = 0
```

它应在相同 scheduler 输入下复现对应原始 Oracle 的 selection。端到端结果仍可能受到 builder 或 Agent nondeterminism 影响，因此 selection equality 是主要 correctness criterion，不能只比较最终 SR。

确认退化正确后，第一轮使用：

```text
alpha = 1.0
R_min = 5
historical epsilon = 1e-8
gain normalization epsilon = 1e-8
```

主要比较：

- selected source task IDs；
- base gain、normalized gain、\(\hat H_j\) 和 adjusted score；
- task-level gained/lost flips；
- final success rate。

如果需要判断 \(\alpha\) 是否过强，再手动补跑 `0.25` 和 `0.5`；第一版实现不提供自动 sweep。

## 13. 结论边界

如果 V2 优于对应原始 Oracle，只能说明在当前实验条件下，query-similarity transferred historical outcomes 为 normalized Oracle gain 提供了额外有效的排序信息。

如果 V2 改变 selection 但没有提高 SR，需要区分：

- HU 的时间预测能力不足；
- task-query similarity 无法稳定迁移 memory effectiveness；
- Top-K co-retrieval 的共享成功归因存在噪声；
- normalized additive HU 权重过强；
- online closed-loop 导致后续 pending pool 分叉；
- builder 和 Agent inference nondeterminism。

两个 V2 policies 仍然是 privileged Oracle baselines，因为 Coverage 和 Exact Retrieval base gain 都读取 future task queries。HU 部分只读取过去 outcomes，但整个 policy 不能描述为完全 deployable scheduler。
