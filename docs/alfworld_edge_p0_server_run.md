# ALFWorld Edge Raw-Trajectory P0 服务器执行清单

更新时间：2026-08-11

## 1. 服务器需要具备的服务和数据

在仓库根目录准备 Python 环境并安装：

```bash
pip install -r requirements.txt
```

确认 `.env` 至少配置：

```text
AGENT_MODEL_NAME=<Qwen3-4B served model name>
AGENT_API_KEY=<agent API key or EMPTY>
AGENT_API_BASE_URL=<OpenAI-compatible vLLM endpoint>

EMBEDDING_MODEL_NAME=<BGE embedding model name>
EMBEDDING_MODEL_KEY=<embedding API key>
EMBEDDING_MODEL_BASE_URL=<OpenAI-compatible embedding endpoint>

ALFWORLD_DATA=<complete ALFWorld 0.4.2 data directory>
```

Edge-50/100/150 不调用 qwen3.5-plus，不需要 builder LLM 服务。只有选择重跑 Cloud MemP-300 且服务器上不存在既有 Cloud artifact 时，才需要 `MEMORY_BUILD_*` 配置。

## 2. 拉取代码后先运行测试

```bash
python -m unittest discover -s tests -v
```

服务器已安装 LangChain 和 FAISS，因此 `test_index_build_reload_and_optional_threshold` 不应跳过。所有测试应通过。

## 3. 生成并检查固定 Edge 子集

```bash
python -m ProcedureMem.build_edge_subsets \
  --trajectory-file ProcedureMem/Alfworld/alfworld_format_traj.json \
  --source-count 300 \
  --capacities 50 100 150 \
  --seed 42 \
  --output ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json
```

预期输出：

```text
Edge-50:  unique_queries=50,  families=18/9/6/3/3/11
Edge-100: unique_queries=92,  families=36/18/13/6/5/22
Edge-150: unique_queries=135, families=55/27/19/9/8/32
```

family 顺序为 pick/place、clean、cool、heat、examine、pick-two。

## 4. 构建三个 Edge FAISS 索引

确保 embedding 服务已经启动，然后运行：

```bash
for CAPACITY in 50 100 150; do
  python -m ProcedureMem.build_edge_memory \
    --trajectory-file ProcedureMem/Alfworld/alfworld_format_traj.json \
    --subset-manifest ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json \
    --capacity "$CAPACITY" \
    --memory-dir "ProcedureMem/memory/alfworld/edge_raw_${CAPACITY}"
done
```

此步骤只调用 embedding 服务，不调用 Agent LLM 或 builder LLM。每个容量应输出 `documents=<capacity>`、一个 Top-1 trajectory index 和 raw score。

生成目录：

```text
ProcedureMem/memory/alfworld/edge_raw_50/
ProcedureMem/memory/alfworld/edge_raw_100/
ProcedureMem/memory/alfworld/edge_raw_150/
```

## 5. 运行 10-task smoke test

先生成固定任务 manifest：

```bash
python -m ProcedureMem.eval_alfworld \
  --condition no_memory \
  --split valid_unseen \
  --seed 42 \
  --limit-tasks 10 \
  --task-manifest ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n10.json \
  --create-manifest-only
```

然后运行三个容量：

```bash
for CAPACITY in 50 100 150; do
  python -m ProcedureMem.eval_alfworld \
    --condition edge_raw \
    --condition-name "edge_raw_${CAPACITY}" \
    --edge-capacity "$CAPACITY" \
    --edge-subset-manifest ProcedureMem/Alfworld/edge_subsets/stratified_nested_seed42.json \
    --edge-memory-dir "ProcedureMem/memory/alfworld/edge_raw_${CAPACITY}" \
    --trajectory-file ProcedureMem/Alfworld/alfworld_format_traj.json \
    --split valid_unseen \
    --seed 42 \
    --task-manifest ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n10.json \
    --batch-size 2 \
    --max-steps 30 \
    --temperature 0 \
    --top-k 1 \
    --experiment-name edge_raw_smoke_n10
done
```

检查每组均完成 10 个任务、error count 为 0，并从任意 task JSON 确认：

- `condition_mode` 为 `edge_raw`；
- `edge_capacity` 正确；
- `retrieved_memories` 恰好包含一项；
- 记录包含 `trajectory_index`、`trajectory`、`raw_score`、`score_type=faiss_l2_distance` 和 `higher_is_better=false`；
- Agent 第一条当前任务 User message 包含 `Here are some trajectories for solving similar tasks:`；
- retrieved trajectory 中没有通用 system instruction 和 `Assistant: OK`。

## 6. 运行正式 134-task P0

确认 smoke test 正常后，从仓库根目录运行：

```bash
bash scripts/run_alfworld_edge_p0.sh
```

该脚本只依次运行 Edge-50、Edge-100 和 Edge-150 三个 `edge_raw` 条件。它假设以下内容已经存在：

- `ProcedureMem/Alfworld/manifests/valid_unseen_seed42_n134.json`；
- 固定 Edge subset manifest；
- 三个已经构建好的 Edge FAISS 索引。

脚本不生成 task manifest、不构建索引、不运行 No Memory 或 Cloud MemP-300，也不自动做结果汇总。需要调试某个容量时，可以直接从脚本中复制对应的一段命令单独执行。

不要在已有非空正式结果目录上直接重复运行；为新运行修改脚本顶部的 `EXPERIMENT_NAME`，避免覆盖旧结果。

## 7. 正式结果检查

结果目录：

```text
ProcedureMem/Alfworld/results/paired/edge_raw_capacity_v1/
```

重点文件：

```text
edge_raw_50/summary.json
edge_raw_100/summary.json
edge_raw_150/summary.json
capacity_comparison.json
capacity_comparison.csv
task_type_summary.csv
edge_transitions.csv
```

三个 Edge 条件完成后，如需生成这些汇总文件，再单独运行：

```bash
python -m ProcedureMem.summarize_edge_p0 \
  --results-dir ProcedureMem/Alfworld/results/paired/edge_raw_capacity_v1
```

正式验收要求：

- 三组任务数均为 134；
- 三组 task ID 和顺序完全一致；
- error count 均为 0；
- 每个 Edge 任务在无 threshold 模式下恰好检索一条 trajectory；
- experiment 参数为 Top-1、`score_threshold=null`、无 Cloud fallback；
- 汇总文件包含 SR、平均 steps、成功任务平均 steps、六类任务 SR 和相邻容量成功/失败转移。
