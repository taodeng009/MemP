# ALFWorld 固定容量 Memory Semantic Diversity 实验实现计划

## 1. 实验目标

本实验验证：

> 在同一次实验中 memory pool 大小保持不变的情况下，MemP workflow memory 的 semantic diversity 是否与 ALFWorld task success rate 存在稳定正相关关系。

实验从当前已有的 workflow memory 中随机生成大量不同的等容量子池，按实际 semantic diversity 排序，并从覆盖整个分布的多个 quantile level 中均匀抽取正式 pool。所有正式 pool 在完全相同的 ALFWorld 任务和 retrieval 配置下测量成功率。第一版推荐 pool size 为 20，但该值通过参数设置，不写死在代码中。

如果 memory diversity 与 task success rate 呈稳定正相关，则可以为 Novelty-based construction scheduling 提供直接依据。

## 2. 实验范围

### 2.1 Candidate Memory

使用当前已有的 workflow memory：

```text
ProcedureMem/memory/alfworld/direct/documents.json
```

通过现有 `load_candidate_memories(path, limit=candidate_count)` 加载候选 memory。第一版推荐 `candidate_count=300`。实验不重新构建 workflow，也不使用 raw trajectory memory。

### 2.2 固定 Pool Size

同一次正式实验中的所有 pool 使用相同容量。第一版推荐 `pool_size=20`，但由命令行参数指定。正式实验开始后冻结该值，不在不同 diversity condition 间改变。

### 2.3 唯一因变量

每个 pool 只比较 ALFWorld task success rate。第一版不分析 execution steps 或 retrieval 机制指标。

## 3. Semantic Diversity 指标

主指标为 mean nearest-neighbor squared-L2 distance：

$$
D_{NN}(M)
=
\frac{1}{|M|}
\sum_{i\in M}
\min_{j\in M,j\ne i}
\lVert e_i-e_j\rVert_2^2
$$

其中 $e_i$ 是 memory query 的 embedding。

该指标越大，表示 pool 中每条 memory 与其最近语义邻居之间的距离越大，即局部重复程度越低。

距离矩阵复用现有 `ScheduledWorkflowMemory.candidate_query_distance_matrix()`。该实现与当前 MemP FAISS retrieval 和 Greedy Novelty scheduler 使用的 squared-L2 distance 一致。

第一版只使用 $D_{NN}$，不计算其他 diversity 指标。

## 4. 正式 Pool 构造

### 4.1 随机候选子池

所有候选子池使用完全相同的 uniform random sampling 生成：

1. 从 `candidate_count` 条 candidate memory 中随机选择 `pool_size` 条；
2. pool 内不允许重复 memory；
3. 重复出现的相同 subset 只保留一次；
4. 对每个不同的 subset 计算 $D_{NN}$。

候选子池数量、pool size、candidate count 和 sampling seed 均作为命令行参数。随机生成过程使用指定 seed，保证相同配置可以复现。

### 4.2 在整个 Quantile 分布上均匀抽取

将所有不同的随机候选子池按照 $D_{NN}$ 从小到大排序，计算其在候选池集合中的经验 quantile，并将完整的 0%–100% 分布划分为 `quantile_bin_count` 个等宽 quantile bins。

例如 `quantile_bin_count=10` 时，bins 为：

```text
[0%,10%), [10%,20%), ..., [90%,100%]
```

使用配置中的 selection seed，在每个 bin 内随机、无放回抽取相同数量的 `pools_per_bin` 个 pool。这样正式 pool 均匀覆盖整个 diversity 分布，而不是只集中在少数区间。

正式 pool 总数为 `quantile_bin_count × pools_per_bin`，并满足：

- 每个 pool 恰好包含 `pool_size` 条 memory；
- 每个 pool 内 memory ID 不重复；
- 所有正式 pool 对应彼此不同的 subset；
- 每个 pool 都位于其记录的 quantile bin；
- 每个 bin 抽取相同数量的正式 pool。

所有候选池均来自同一个随机生成过程，因此主实验不引入不同 selection algorithm 的影响。

## 5. Pool 定义文件

新增一个简单 JSON 文件，只保存评测所需信息：

```json
{
  "generation_parameters": {
    "candidate_memory_file": "ProcedureMem/memory/alfworld/direct/documents.json",
    "candidate_count": 300,
    "pool_size": 20,
    "candidate_pool_count": 1000,
    "sampling_seed": 42,
    "selection_seed": 42,
    "quantile_bin_count": 10,
    "pools_per_bin": 2,
    "embedding_model": "...",
    "distance_metric": "mean_nearest_neighbor_squared_l2"
  },
  "pools": [
    {
      "pool_id": "q00_01",
      "quantile_bin": 0,
      "quantile_range": [0.0, 0.1],
      "memory_ids": ["mem_0001", "mem_0002"],
      "diversity": 0.0
    }
  ]
}
```

`generation_parameters` 保存 pool 构造时实际生效的参数。每个正式 pool 只记录 `pool_id`、`quantile_bin`、`quantile_range`、`memory_ids` 和实测 $D_{NN}$。

## 6. 代码实现

### 6.1 Pool 构造入口

新增：

```text
ProcedureMem/build_diversity_pools.py
```

该模块负责：

1. 复用 `load_candidate_memories()` 加载配置数量的 workflow memory；
2. 按现有方式创建 cached embedding；
3. 复用 `ScheduledWorkflowMemory.candidate_query_distance_matrix()` 计算 squared-L2 distance matrix；
4. 按配置的 pool size 随机生成不同子池；
5. 计算每个子池的 $D_{NN}$；
6. 计算候选 pool 的经验 quantile；
7. 将完整 quantile 分布划分为配置数量的等宽 bins；
8. 按 selection seed 在每个 bin 内随机、无放回抽取相同数量的正式 pool；
9. 写入 pool JSON。

命令示例：

```bash
python -m ProcedureMem.build_diversity_pools \
  --candidate-memory-file ProcedureMem/memory/alfworld/direct/documents.json \
  --candidate-count 300 \
  --pool-size 20 \
  --candidate-pool-count 1000 \
  --sampling-seed 42 \
  --selection-seed 42 \
  --quantile-bin-count 10 \
  --pools-per-bin 2 \
  --output ProcedureMem/Alfworld/diversity_pools/workflow_pool20.json
```

以上数值是推荐配置示例。参数通过 CLI 或当前 runtime environment 配置，并将实际生效值写入 pool JSON，不作为代码常量写死。

### 6.2 ALFWorld Evaluation

扩展 `ProcedureMem.eval_alfworld`：

```text
--condition diversity_pool
--diversity-pools PATH
--pool-id POOL_ID
--score-threshold FLOAT
```

新增的加载逻辑只负责：

1. 从 JSON 找到指定 pool；
2. 复用 `load_candidate_memories()` 加载 candidate memories；
3. 创建 `ScheduledWorkflowMemory`；
4. 调用 `activate(memory_ids, interval_id=0)`；
5. 调用 `rebuild_available_index()`；
6. 进入现有 workflow retrieval、`inject_memory()` 和 ALFWorld evaluation 流程。

不新增新的 memory class、prompt、retrieval pipeline 或持久化 FAISS index。

Pool size 和 pool selection 参数从 pool JSON 读取；ALFWorld 与 retrieval 参数继续使用 `eval_alfworld.py` 的 CLI。现有 `--score-threshold` 参数校验需要扩展为允许 `diversity_pool` 使用。新 condition 不在代码中覆盖用户传入的 model、temperature、seed、batch size、max steps、few-shot、top-k 或 score threshold。

### 6.3 复用现有评测能力

直接复用：

- `build_task_manifest()`；
- `validate_task_manifest()`；
- `ScheduledWorkflowMemory`；
- `retrieval_records()`；
- `inject_memory()`；
- `run_alfworld_batch()`；
- `write_results()`；
- `summarize_results()`。

### 6.4 统一运行入口

新增：

```text
scripts/run_alfworld_diversity_experiment.sh
```

该脚本统一完成 pool 构造、固定 task manifest、逐 pool 评测和最终汇总。实验参数通过环境变量设置，脚本默认生成 5 个 quantile bins、每个 bin 3 个 pool，共 15 个正式 pool。

## 7. 实验控制

同一次正式实验中的所有 pool 必须使用完全相同的：

- ALFWorld split；
- task manifest、task IDs 和任务顺序；
- Agent model；
- temperature 和 seed；
- batch size；
- max steps；
- few-shot 设置；
- embedding model；
- retrieval top-k；
- retrieval score threshold；
- workflow prompt injection。

不同条件之间只允许 pool ID、pool 中的 memory IDs 和实测 $D_{NN}$ 变化。

开始正式评测前确定一组 evaluation parameters，全部 pool 复用该配置。实际生效参数必须与最终结果一起保存，而不是只记录启动命令。

## 8. 结果保存与分析

所有结果保存到一个文件：

```text
diversity_results.json
```

文件采用简洁结构，同时保存实际生效的 pool generation parameters、evaluation parameters，以及每个 pool 的 diversity 和 SR：

```json
{
  "generation_parameters": {},
  "evaluation_parameters": {
    "split": "valid_unseen",
    "task_manifest": "...",
    "model": "...",
    "temperature": 0,
    "seed": 42,
    "batch_size": 1,
    "max_steps": 30,
    "few_shot": true,
    "embedding_model": "...",
    "top_k": 3,
    "score_threshold": 0.5
  },
  "pool_results": [
    {
      "pool_id": "q00_01",
      "quantile_bin": 0,
      "diversity": 0.0,
      "task_success_rate": 0.0
    }
  ],
  "spearman": {
    "rho": 0.0,
    "p_value": 1.0
  }
}
```

上述值只是结构示例。结果文件记录运行时实际值。

最终计算：

- 所有正式 pool 的 $D_{NN}$ 与 task success rate 的 Spearman correlation；
- 每个 quantile bin 的平均 diversity 和平均 task success rate。

核心判断是：

1. $D_{NN}$ 与 SR 的相关方向是否为正；
2. 各 quantile bin 的平均 SR 是否整体随 diversity 增加而上升；
3. 该方向是否在多个 pool 上保持一致，而不是由单个 pool 决定。

## 9. 最小测试范围

只实现与核心实验正确性直接相关的测试：

1. 相同 seed 能复现相同的正式 pool；
2. 每个正式 pool 的 size 等于配置的 `pool_size`；
3. 全部正式 pool 是不同 subset；
4. $D_{NN}$ 计算正确；
5. 正式 pool 分别落在其记录的 quantile bins 中，且每个 bin 数量相同；
6. `ScheduledWorkflowMemory` 只激活指定 pool 的 memory；
7. 结果汇总正确保存 pool diversity、SR 和 Spearman correlation；
8. 统一 Shell 入口能够完成 pool 构造、评测和汇总。

## 10. 实施顺序

1. 实现 `build_diversity_pools.py`；
2. 在完整 quantile 分布的各等宽 bin 中均匀抽取并检查正式 pool；
3. 将 `diversity_pool` condition 接入 `eval_alfworld.py`；
4. 使用同一个 task manifest 完成小规模 smoke test；
5. 运行全部正式 pool；
6. 保存 `diversity_results.json` 并计算 Spearman correlation。

## 11. 验收标准

实现完成必须满足：

1. 使用当前 workflow memory，candidate count 可配置；
2. Pool size 可配置，并在同一次实验的所有正式 pool 中保持一致；
3. 所有正式 pool 彼此不同；
4. 所有候选 pool 都由同一种随机抽样方式产生；
5. Candidate count、pool size、候选池数量、sampling seed、selection seed、quantile bin 数量和每个 bin 的 pool 数量均可配置并被记录；
6. 正式 pool 在完整 quantile 分布的各个等宽 bin 中随机、无放回、等量抽取；
7. 主 diversity 指标为 mean nearest-neighbor squared-L2 distance；
8. 所有 pool 使用完全相同、可配置且有记录的 ALFWorld 任务和 retrieval 参数；
9. 最终结果包含每个 pool 的 diversity、task success rate 和 Spearman correlation；
10. 提供统一、参数可修改的 Shell 运行入口；
11. 可以据此判断 diversity 与 SR 是否存在稳定正相关关系。
