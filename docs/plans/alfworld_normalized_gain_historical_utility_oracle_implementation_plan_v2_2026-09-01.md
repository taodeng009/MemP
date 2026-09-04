# ALFWorld Normalized Gain + Historical Utility Oracle V2 实现计划

**日期：2026-09-01**
**Top-K 补丁修订：2026-09-02**

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

---

## 14. V2 补丁：Local Historical Top-K References

### 14.1 补丁目标

本补丁不新增 V3，也不改变 normalized gain + additive HU 的核心公式。为严格保留原 V2 的 all-reference 语义，新增两个显式的 Top-K policy 变体：

```text
oracle_coverage_historical_utility_v2_topk
oracle_exact_retrieval_historical_utility_v2_topk
```

原有两个 policy 不做任何语义修改，继续始终使用全部 reliable references：

```text
oracle_coverage_historical_utility_v2
oracle_exact_retrieval_historical_utility_v2
```

Top-K 补丁只在新增 policy 中把 HU reference aggregation 从“使用全部 reliable historical references”改为“只使用与当前 pending task query 最相似的 K 条 reliable historical references”。

原 V2：

$$
\hat H_j^{all}
=
\frac{
\sum_{i\in\mathcal R_t}
\frac{1}{d_{ji}+\epsilon_H}H_i
}{
\sum_{i\in\mathcal R_t}
\frac{1}{d_{ji}+\epsilon_H}
}
$$

Top-K 补丁：

$$
\boxed{
\hat H_j^{TopK}
=
\frac{
\sum_{i\in\operatorname{KNN}(j,\mathcal R_t)}
\frac{1}{d_{ji}+\epsilon_H}H_i
}{
\sum_{i\in\operatorname{KNN}(j,\mathcal R_t)}
\frac{1}{d_{ji}+\epsilon_H}
}
}
$$

其中：

- \(\mathcal R_t\) 是当前 construction 时刻所有满足 \(R_i\ge R_{min}\) 的 available historical memories；
- `KNN` 按 pending task query 与 historical source task query 的 squared-L2 distance 从小到大选择；
- 默认 \(K=5\)；
- 当 \(|\mathcal R_t|<K\) 时，使用全部 reliable references；
- 当 \(|\mathcal R_t|=0\) 时，保持 \(\hat H_j=0\)；
- reference utility、run-local counters、时间边界、warm-start 行为均不改变。

最终 V2 score 仍为：

$$
Score_j
=
\operatorname{Norm}(G_j)
+\alpha\hat H_j^{TopK}
$$

该补丁的目的不是放大 HU 的绝对数值，而是减少远距离 historical tasks 对所有 candidates 的共同平均化，使 \(\hat H_j\) 更接近 candidate-specific local utility estimate。

### 14.2 与原计划第4.3节的关系

原计划第4.3节规定“第一版 V2 不加入 HU Top-K”。该规定对原有两个 V2 policies 继续完整有效。本补丁实施后：

- 原 `oracle_coverage_historical_utility_v2` 和 `oracle_exact_retrieval_historical_utility_v2` 的默认值、all-reference HU 和运行语义均不改变；
- 新增的 `_v2_topk` policies 默认使用 `historical_top_k=5`；
- all-reference 对照直接使用原 V2 policies，不通过参数把 Top-K policy 切换成另一种算法语义；
- Exact Retrieval base objective 自身的 retrieval Top-K 与 Historical Top-K 是两个完全不同的参数，不能共用 CLI 名称或变量。

## 15. 触发补丁的实验观测

### 15.1 已完成 V2 实验

分析使用以下 `valid_unseen` 134-task 结果：

```text
ProcedureMem/Alfworld/results/paired/
online_construction_valid_unseen_seed42_n134_b1_i20_c5/
```

相关 runs：

```text
online_construction_oracle_coverage
online_construction_oracle_coverage_historical_utility_v2
online_construction_oracle_exact_retrieval_h1
online_construction_oracle_exact_retrieval_historical_utility_h1
online_construction_oracle_exact_retrieval_historical_utility_v2_h1
```

总体结果：

| 方法 | 成功数 | 成功率 | 平均步数 |
| --- | ---: | ---: | ---: |
| Oracle Coverage | 59/134 | 44.03% | 21.45 |
| Coverage + HU V2 | 62/134 | 46.27% | 20.84 |
| Exact Retrieval H1 | 54/134 | 40.30% | 21.95 |
| Exact Retrieval H1 + HU V2 | 62/134 | 46.27% | 21.07 |
| Exact Retrieval H1 + HU V1 | 66/134 | 49.25% | 20.58 |

这些端到端差异不能直接归因于 HU：Coverage+HU V2 的3个净提升中已有2个在 HU-built memories 可用前形成；Exact+HU V2 的8个净提升中已有5个在 HU 生效前形成。运行还受到 Agent execution、workflow generation 和 `success_only` arrival feedback 的非确定性影响。

### 15.2 HU 已计算，但很少改变 selection

原 all-reference V2 日志显示：

- Coverage V2 的27次非 bootstrap scored selections 中，没有一次所选 candidate 的 `normalized_base_gain < 0.999`；
- Exact V2 只有一次明确由 HU 选中非 base-gain-max candidate；
- 该 Exact candidate 的 \(\operatorname{Norm}(G)=0.9209\)、\(\hat H=0.8\)、adjusted score 为1.3209，后续实际 retrieval utility 为4/5。

因此，HU 数值进入了 score，但除一个明确案例外，没有表现出足够强的 candidate-level 排序区分度。

### 15.3 H 随 reference pool 增长趋向平均化

Reliable reference pool 随 interval 增长：

| Construction interval | Coverage references | Exact references |
| ---: | ---: | ---: |
| 2 | 3 | 4 |
| 3 | 5 | 6 |
| 4 | 10 | 8 |
| 5 | 15 | 14 |

多数 interval 中，selected candidate H 的均值几乎等于 reference utilities 的全局均值。例如：

| Run/interval | Reference utility mean | Selected candidate H mean |
| --- | ---: | ---: |
| Coverage / 2 | 0.476 | 0.479 |
| Coverage / 4 | 0.725 | 0.733 |
| Coverage / 5 | 0.613 | 0.603 |
| Exact / 2 | 0.611 | 0.613 |
| Exact / 4 | 0.572 | 0.579 |
| Exact / 5 | 0.622 | 0.644 |

原公式对每条 reliable reference 都赋予严格正的 inverse-distance weight。ALFWorld task queries 又高度模板化；当没有极近的 reference 时，多条中等权重共同参与，使每个 candidate 的 H 收缩到 pool mean。随着 pool 增长到10—15条，该效应更加明显。

在 \(\alpha=0.5\) 下，后期 selected candidates 之间由 HU 提供的最大 adjusted-score 差异通常只有约0.02—0.04，难以推翻 normalized base gain 排序。

## 16. Historical Top-K=5 Offline 反事实结果

### 16.1 Offline 重放方法与正确性

Offline 分析直接读取两个 V2 run 保存的 BGE-base-en-v1.5 vector cache，并按每个 interval 当时的真实状态重建：

- pending queue；
- available constructed memories；
- 截至该时刻的 retrieval/success counters；
- `R_min=5` reliable reference pool；
- pending/reference task-query squared-L2 distances。

随后同时计算 \(\hat H^{all}\) 和 \(\hat H^{Top5}\)。重放得到的 \(\hat H^{all}\) 与 construction log 中原值的最大误差小于 \(6\times10^{-9}\)，验证了 offline 重算与服务器实现一致。

### 16.2 Top-5 的有效时间窗口

| Construction interval | Coverage pool | Exact pool | 与 all-reference 的差异 |
| ---: | ---: | ---: | --- |
| 2 | 3 | 4 | 无差异 |
| 3 | 5 | 6 | Coverage 无差异；Exact 仅排除1条 |
| 4 | 10 | 8 | 明显差异 |
| 5 | 15 | 14 | 明显差异 |

因此：

- Coverage Top-5 主要影响 construction interval 4、5，对应最多直接影响最后34个 tasks；
- Exact Top-5 从 interval 3 开始不同，对应最多影响最后54个 tasks，但最明显的 locality 变化仍集中在 interval 4、5；
- Top-5 不可能改变或解释两个 V2 runs 的早期结果分叉。

### 16.3 Coverage pending pool 的变化

| Interval | Ref count | All H mean | Top-5 H mean | All H SD | Top-5 H SD | Mean absolute change | Max absolute change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 0.477 | 0.477 | 0.004 | 0.004 | 0 | 0 |
| 3 | 5 | 0.711 | 0.711 | 0.085 | 0.085 | 0 | 0 |
| 4 | 10 | 0.751 | 0.753 | 0.078 | **0.105** | 0.042 | 0.105 |
| 5 | 15 | 0.664 | 0.683 | 0.115 | **0.154** | 0.059 | 0.164 |

在 Top-5 生效后，全部 pending candidates 的 H 标准差提高约33%—36%。在 \(\alpha=0.5\) 下，单条 candidate adjusted score 的最大变化约为0.052和0.082。

对原 all-reference policy 实际选中的 candidates：

| Interval | All H SD | Top-5 H SD | All H range | Top-5 H range |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.024 | **0.080** | 0.060 | **0.203** |
| 5 | 0.021 | **0.050** | 0.054 | **0.136** |

selected-candidate H 离散度分别扩大约3.3倍和2.4倍，说明 Top-5 能够移除较远 references 造成的公共抬升，并显著增强 local differentiation。

### 16.4 Exact pending pool 的变化

| Interval | Ref count | All H mean | Top-5 H mean | All H SD | Top-5 H SD | Mean absolute change | Max absolute change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 4 | 0.613 | 0.613 | 0.013 | 0.013 | 0 | 0 |
| 3 | 6 | 0.618 | 0.612 | 0.049 | **0.062** | 0.025 | 0.037 |
| 4 | 8 | 0.579 | 0.560 | 0.045 | **0.066** | 0.033 | 0.076 |
| 5 | 14 | 0.641 | 0.635 | 0.057 | **0.077** | 0.028 | 0.096 |

Exact 的 pending-candidate H 标准差提高约27%—48%。在 \(\alpha=0.5\) 下，单条 candidate adjusted score 的最大变化约为0.019、0.038和0.048。

原 Exact interval 3 中具有同类 Apple-to-GarbageCan historical reference 的 candidate 在 Top-5 下仍是 H 最大者。Top-5 保留了已观察到的强近邻信号，同时主要重排中间 candidates。

### 16.5 HU 排名与探索性预测关系

| Run/interval | All-H 与 Top5-H rank correlation | 两种 H Top-5 candidates overlap | H winner 是否变化 |
| --- | ---: | ---: | --- |
| Coverage / 4 | 0.917 | 3/5 | 否 |
| Coverage / 5 | 0.944 | 4/5 | 否 |
| Exact / 3 | 0.821 | 5/5 | 否 |
| Exact / 4 | 0.854 | 3/5 | 否 |
| Exact / 5 | 0.842 | 4/5 | 否 |

Top-5 没有改变任何 interval 的最高-H candidate，但对中间候选产生了1—2条 Top-5 membership 替换。

将原策略选出的20条 HU-active memories 与其后续实际 retrieval utility 关联：

| Run | All-reference H Pearson correlation | Top-5 H Pearson correlation |
| --- | ---: | ---: |
| Coverage V2 | 0.100 | **0.201** |
| Exact V2 | 0.111 | **0.180** |

Top-5 后相关大致提高一倍，方向上支持 locality；但样本只有20条 memory，相关仍然较弱，而且分析对象仍是原 all-reference policy 选出的 memories，因此不能据此预测最终成功率提升。

### 16.6 Offline 分析的结论边界

现有日志没有保存每个 greedy rank 下所有未选 candidates 的 base gain/normalized base gain。Offline 分析可以精确重算全部 pending candidates 的 Top-5 H，但不能完整重放：

$$
\operatorname{Norm}(G_j)+\alpha\hat H_j^{Top5}
$$

对所有 candidates 的最终排序。因此，上述结果证明的是“Top-5 使 H 更 local、更有区分度”，而不是“Top-5 一定改变多少条 construction selection”或“最终 success count 一定提高”。

## 17. Top-K 补丁实现方案

### 17.1 配置语义

新增 policy choices：

```text
--schedule-policy oracle_coverage_historical_utility_v2_topk
--schedule-policy oracle_exact_retrieval_historical_utility_v2_topk
```

新增独立参数，仅供上述两个 Top-K policies 使用：

```text
--historical-utility-top-k 5
```

语义：

- 正整数：每个 pending candidate 只使用最近 K 条 reliable historical references；
- 默认值：`5`；
- K 大于当前 reliable reference 数时自动 clamp 到 pool size；
- 不允许0或负数；
- 原两个 V2 policies 不读取该值，并继续走 all-reference 路径。

Python 内部统一表示为：

```python
historical_utility_top_k: int | None
```

其中 `None` 表示 all-reference，仅作为内部规范化表示。

Controller 对原 V2 policies 固定传入 `None`；只有新增 `_v2_topk` policies 才传入 CLI 中的正整数 K。这样 policy 名称本身即可唯一确定 HU aggregation 语义，不依赖隐藏默认值。

该参数不能复用现有 `--top-k`：

- `--top-k` 控制在线 retrieval/Exact Oracle base objective 的 retrieved memory 数；
- `--historical-utility-top-k` 控制 HU transfer 使用的 historical neighbors 数。

### 17.2 HU estimator 修改

修改 `estimate_historical_utilities(...)`：

```python
def estimate_historical_utilities(
    pending_queries,
    reference_queries,
    reference_utilities,
    embedding,
    *,
    epsilon=1e-8,
    top_k: int | None = None,
):
    ...
```

计算步骤：

1. 保留当前一次性 embedding 和 squared-distance matrix 计算；
2. 对每个 pending candidate 的 distance row 选择最近 `min(K, N_ref)` 个 indices；
3. 仅在这些 indices 上计算 inverse-distance weights；
4. 对对应 historical utilities 做归一化加权平均；
5. `top_k=None` 时沿用现有 vectorized all-reference 路径，供原 V2 和 V1 使用；
6. reference pool 为空时仍直接返回每个 pending candidate 的0值。

距离相同的 references 必须使用稳定 reference ID/order 作为 tie-break，保证同一输入下结果可复现。不要让默认 `np.argsort` 的非稳定 tie 行为决定 neighbor membership；可使用 stable sort，或明确按 `(distance, reference_id)` 排序。

补丁不加入：

- distance threshold；
- retrieval-count confidence weighting；
- smoothing；
- dynamic K；
- task-family hard filter；
- workflow/content feature；
- 跨 run reference sharing。

### 17.3 Controller 接入

在 `OnlineConstructionController` 中新增：

```python
self.historical_utility_top_k
```

不同 policies 对 estimator 参数的使用必须明确：

- 原 `oracle_coverage_historical_utility_v2` 和 `oracle_exact_retrieval_historical_utility_v2` 固定传入 `top_k=None`，保持 all-reference；
- 新 `oracle_coverage_historical_utility_v2_topk` 和 `oracle_exact_retrieval_historical_utility_v2_topk` 传入配置的正整数 K；
- `oracle_exact_retrieval_historical_utility` V1 继续传入 `top_k=None`，保持 all-reference；
- 如果以后需要 V1+Top-K，应另行设计，不在本补丁中隐式启用。

Warm-start memory 和 online constructed memory 继续按完全相同的 reliable-reference 规则进入 neighbor search。

### 17.4 CLI、shell 与 metadata

修改 `ProcedureMem/eval_alfworld.py`：

- 增加两个 `_v2_topk` policy choices；
- 增加 `--historical-utility-top-k`；
- 只接受正整数；
- 原 V2 policies 在 controller 中固定使用 `None/all-reference`，不因该 CLI 默认值而改变；
- Top-K policies 在 run parameters/summary metadata 中记录正整数 K；
- 原 V2 policies 的 `historical_utility_top_k` 记录为 `null` 或不新增该字段，不能记录一个实际未生效的默认值。

修改 `scripts/run_alfworld_online_construction.sh`，增加：

```text
HISTORICAL_UTILITY_TOP_K="${HISTORICAL_UTILITY_TOP_K:-5}"
```

并只对两个新增 `_v2_topk` policies 传递：

```text
--historical-utility-top-k "$HISTORICAL_UTILITY_TOP_K"
```

服务器运行目录建议显式增加后缀：

```text
_htop5
```

避免覆盖已有 all-reference V2 结果。All-reference 复现实验使用：

```text
oracle_coverage_historical_utility_v2
oracle_exact_retrieval_historical_utility_v2
```

### 17.5 最小日志补充

只在两个 `_v2_topk` construction event/oracle score 中增加：

```text
historical_utility_top_k
historical_effective_reference_count
```

其中：

```text
historical_effective_reference_count
= min(historical_reference_count, configured_top_k)
```

原 V2 events 保持现有字段和语义。第一版补丁不记录每个 candidate 的完整 neighbor IDs、distances、weights 或 weight entropy，避免日志膨胀。

Run-level metadata 必须记录 Top-K 配置，确保 `_htop5` 不是唯一 provenance。

### 17.6 代码修改范围

补丁继续只修改原 V2 涉及的文件：

```text
ProcedureMem/online_construction.py
ProcedureMem/cloud_scheduling.py
ProcedureMem/eval_alfworld.py
scripts/run_alfworld_online_construction.sh
tests/test_online_construction.py
```

预计职责：

- `online_construction.py`：Top-K HU estimator、controller 参数、新 policies 和调用点，同时保留原 V2 的 all-reference 调用；
- `cloud_scheduling.py`：不改变 scoring 公式，仅透传/记录 configured/effective historical K；
- `eval_alfworld.py`：CLI parse、validation 和 metadata；
- shell：暴露 `HISTORICAL_UTILITY_TOP_K`；
- tests：验证 Top-K correctness、原 V2 语义不变和 Top-K-policy-only scope。

如果 estimator 层已经能够把 configured/effective K 写回 controller，scheduler 不需要重新选择 neighbors；Top-K neighbor selection 必须只实现一次，避免 Coverage 和 Exact 出现不一致逻辑。

## 18. Top-K 补丁核心测试

保留原第11节测试，并增加以下最小 correctness tests：

1. **Top-K selection**：构造6条距离和 utilities 已知的 references，验证 `K=5` 精确排除最远一条。
2. **Local weighted utility**：验证 H 只由最近K条 references 的 inverse-distance weighted utility 得到。
3. **K clamp**：当 reliable reference 数小于K时，Top-K 与 all-reference 结果完全一致。
4. **All compatibility**：`top_k=None` 与补丁前 estimator 数值完全一致。
5. **Empty reference**：reference pool 为空时，不调用无意义 neighbor selection，所有 H 严格为0。
6. **Stable distance tie**：相同距离下使用稳定 reference ordering/ID，结果可复现。
7. **Policy semantic preservation**：原两个 V2 policies 和 V1 继续使用 all-reference；只有两个 `_v2_topk` policies 使用 Top-K。
8. **CLI/metadata**：正整数 K 正确解析并写入 Top-K run metadata；0和负数被拒绝；原 V2 metadata/selection 不因 CLI 默认 K 而改变。
9. **Score invariance**：除 \(\hat H_j\) 输入变化外，normalized gain 和 additive score 公式保持不变。

不增加多 K sweep、distance-threshold fixtures 或完整 offline comparison framework。

## 19. 补丁实验设计

### 19.1 第一轮配置

为了只隔离 locality 的影响，第一轮 Top-5 实验保持现有 V2 实测参数不变：

```text
historical_utility_top_k = 5
historical_utility_alpha = 0.5
historical_utility_min_count = 5
historical_utility_epsilon = 1e-8
gain_normalization_epsilon = 1e-8
```

分别运行：

```text
oracle_coverage_historical_utility_v2_topk + htop5
oracle_exact_retrieval_historical_utility_v2_topk + htop5 + h1
```

不要同时改变 alpha、R_min、Exact lookahead horizon 或 retrieval threshold，否则无法判断变化来自 Top-K 还是其他参数。

### 19.2 对照组

直接使用当前已完成的 all-reference V2 结果作为探索性对照：

```text
oracle_coverage_historical_utility_v2:
    alpha=0.5, R_min=5, all-reference

oracle_exact_retrieval_historical_utility_v2:
    alpha=0.5, R_min=5, all-reference, H1
```

但端到端 run 仍受非确定性影响。更可靠的机制对照是在同一个 Top-5 run 中同时计算 shadow scores：

```text
H_all
H_top5
Score_alpha0
Score_top5_alpha0.5
```

第一版补丁可以不自动采用 shadow selection，但至少应能够在相同 scheduler state 下确认：

- Top-5 是否改变 H；
- Top-5 是否改变最终 selected queue IDs；
- 改变发生在第几个 interval/rank；
- 被改变选择的 memory 后续是否被 retrieve、成功多少次。

### 19.3 主要观测指标

第一轮只保留直接回答机制问题的指标：

1. 每个 interval 的 reliable reference count 和 effective Top-K count；
2. pending/selected candidate H 的 mean、SD、range；
3. `H_top5 - H_all` 的 mean absolute/max absolute change；
4. all-reference 与 Top-5 的 selected queue ID overlap；
5. 明确由 HU 选中 `normalized_base_gain < 1` candidate 的次数；
6. 被 Top-5 改变选择的 constructed memories 的后续 retrieval count 和 utility；
7. task-level success flips 和 final success rate。

不把单次最终 SR 差异单独作为 HU 有效的证据。

### 19.4 成功判据

补丁首先按机制而非最终 SR 判断：

- **实现正确**：Top-K offline/online H 一致；K小于 pool时只使用最近K条；K大于等于pool时与 all-reference 一致；
- **locality 生效**：reference pool大于5时，candidate H 离散度或 ranking 相比 all-reference 出现可见变化；
- **调度生效**：在完全相同的 scheduler state 下，Top-5 至少改变部分 construction selections，而不是只改变公共 score offset；
- **效果候选成立**：发生 selection change 的 memories 后续 utility 方向与 Top-5 H 偏好一致；
- **端到端有效**：需要多 run/seed 或严格 shadow/counterfactual evidence，不能由一次 SR 上升直接认定。

## 20. 补丁风险与结论边界

Top-K 缓解的是 all-reference bias，但会增加 local estimate variance。当前 `R_min=5` 只保证每条 memory 至少有5次 retrieval，并不能保证 \(H_i\) 已经稳定。Top-5 让少数 references 获得更高影响，因此可能同时放大单条 historical utility 的采样噪声。

本补丁暂不加入 confidence weighting 或 smoothing，目的是先单独验证 locality 是否改善 utility transfer。若 Top-5 明显改变 selection 但方向不稳定，再单独考虑基于 retrieval count 的 confidence correction，不能与 Top-K 首次实验同时加入。

Offline 结果支持以下有限结论：

$$
\boxed{
\text{Top-5 使 H 更 local、离散度更高，且探索性预测相关有所改善}
}
$$

但目前不能声称：

$$
\boxed{
\text{Top-5 已被证明会提高最终 ALFWorld success rate}
}
$$

该结论必须由补丁实现后的同状态 selection audit 和新的端到端实验验证。
