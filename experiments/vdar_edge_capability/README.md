# ALFWorld instruction-only difficulty profiles

Run on the server from the MemP root:

```bash
python experiments/vdar_edge_capability/generate_difficulty.py
```

Default prompt_version is alfworld. Reads all 134 unique tasks from outputs/edge_capability_dataset.csv. Only task_instruction enters the model input; no outcomes, p_edge, environment, memory, embedding or retrieval. The supplied ALFWorld system prompt and frozen official V2 remain unchanged.

Uses .env MEMORY_BUILD_MODEL_NAME and memory-build endpoint/key and generation settings. Output: outputs/difficulty_profiles.jsonl, one record per task_id, with summary, parsed fields, validation status, raw response, exact messages, configuration and usage.

Resume by rerunning the same command. Each response is atomically checkpointed before the next request. All recorded task IDs are skipped, including empty/invalid summaries: no automatic re-evaluation of existing tasks. A failed network request is not recorded, so it is retried on the next invocation. Configuration, instructions, duplicate IDs and cached validation results are checked before resuming. Changing model/prompt/settings requires a separate --output. Do not run concurrent writers on the same output.

Completion prints and writes outputs/difficulty_profiles.stats.json:
- Successful valid summary count.
- Empty summary count and nonempty/malformed invalid count.
- low/medium/high distribution.
- Count of each of seven primary_dimensions.
- Exact duplicate summary extra-record count, duplicate groups and participating records.

A valid response has exactly one summary block with no outside text, low/medium/high, 1–3 distinct allowed dimensions, and a nonempty profile. Prompt requests 1–3 profile sentences; no NLP sentence-length classifier is added. Invalid responses are preserved, never repaired or fabricated. Level/dimension/duplicate statistics use valid summaries only. Duplicate count means sum(group size - 1), not number of groups.

```bash
python experiments/vdar_edge_capability/generate_difficulty.py --dry-run
python experiments/vdar_edge_capability/generate_difficulty.py --stats-only
```

Explicitly retry only recorded invalid summaries (and any not-yet-recorded tasks):

```bash
python experiments/vdar_edge_capability/generate_difficulty.py --retry-invalid --dry-run
python experiments/vdar_edge_capability/generate_difficulty.py --retry-invalid
```

Each invalid task receives one new API call per invocation, with the same prompt and settings. Valid records remain unchanged. The replacement keeps the same task position and stores the complete previous response record in `previous_attempts`. Network failure preserves the old record; a new invalid response remains invalid and can be retried explicitly again. Empty records are not retried by this flag. With deterministic generation, the same malformed response may recur; success is not guaranteed.

Dry-run/statistics modes make no API calls. Older five-task CSV files are not overwritten or silently converted into the JSONL cache. No real 134-task generation has been completed locally; use the configured server service and return the JSONL/statistics file for reporting.

## Memory-conditioned difficulty generation (optional)

`alfworld_memory` uses the supplied memory-conditioned system prompt without changing the existing `v2` or `alfworld` prompts. It still uses MemP's independent HTTP API call, not the official VDAR Agent. Select an explicit memory-condition `results.jsonl` so the source run is unambiguous:

```bash
# Validation only; no API calls and no output files written.
python experiments/vdar_edge_capability/generate_difficulty.py \
  --prompt-version alfworld_memory \
  --memory-log ProcedureMem/Alfworld/results/paired/valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run1/memory/results.jsonl \
  --dry-run
```

When generation is authorized, remove `--dry-run`. Default output for this mode is `outputs/difficulty_profiles_alfworld_memory.jsonl` and its `.stats.json`, separate from instruction-only outputs. `--output` remains available. Rerunning resumes without repeating recorded tasks; `--retry-invalid` retains its existing behavior.

Inputs are aligned by task ID and exact instruction. Only the logged `retrieved_memories[*].workflow` bodies, in recorded rank order, enter the user message:

```text
**Task Instruction:**
{task_instruction}

**Retrieved Procedural Memories:**
Memory 1:
{first workflow body}

Memory 2:
{second workflow body}

Memory 3:
{third workflow body}
```

The script uses all actually retrieved memories (including fewer than K on a miss/filtered result); it does not re-retrieve, re-rank, truncate or fabricate bodies. Zero-memory tasks use `(No procedural memories were retrieved.)`. Missing tasks, duplicates, query mismatch, non-memory condition, count/rank inconsistencies or empty workflow bodies stop the run. No reward, final outcome, trajectory, RU, BD, distance, similarity or other memory metadata is added. The source memory bodies are passed verbatim, not rewritten or supplemented with historical labels.

API request settings, summary validation and atomic saving are unchanged. Memory text is saved alongside the exact messages and checked during resume; changed support cannot silently reuse a cached summary. No embedding, retrieval or scoring code is modified by this mode.

### Six-task memory smoke test

```bash
python experiments/vdar_edge_capability/generate_difficulty.py \
  --prompt-version alfworld_memory \
  --memory-log ProcedureMem/Alfworld/results/paired/valid_unseen_seed42_n134_b2_qwen36_27b_fp8_top3_run1/memory/results.jsonl \
  --smoke-test
```

This selects the first dataset CSV-order task in each of the six families, in the order simple/cool/light/clean/heat/two-object. Selection never consults outcomes. At most six generation calls are made; there is no embedding, KNN or capability evaluation. It prints each task family, instruction, actual retrieved workflow bodies and generated difficulty summary. Add `--dry-run` to inspect inputs without API calls.

Outputs are separate from both full-task caches: `outputs/difficulty_profiles_alfworld_memory_smoke.jsonl`, `.stats.json`, and `.inputs.json` (selected IDs, source log, exact messages and manual review checklist). Reruns resume and print cached results without regenerating them. Review each real response for the three-part structure **task requirements → applicable procedural memory support → remaining effective difficulty**. Format validation alone does not establish that this semantic requirement is met; no automatic semantic score or fabricated verdict is added.

## VDAR retrieval feasibility

From the MemP root on the server:

```bash
python -m pip install -r experiments/vdar_edge_capability/requirements.txt
python experiments/vdar_edge_capability/vdar_feasibility.py --embed-only
python experiments/vdar_edge_capability/vdar_feasibility.py
```

The first mode uses the existing `.env` settings `EMBEDDING_MODEL_NAME=BAAI/bge-base-en-v1.5`, `EMBEDDING_MODEL_BASE_URL`, and `EMBEDDING_MODEL_KEY`; environment variables take precedence. The endpoint is the configured base URL plus `/embeddings`. No tokenizer directory or local model is needed. Only the full extracted `difficulty_summary` is encoded: no task instruction, canonical query, labels or metadata are appended. Identical summary texts share a request; the resulting `.npy` still has all 134 task rows in dataset CSV order. Batched requests are checkpointed and resumable. Inputs must have exactly the same 134 unique task IDs and matching instructions.

A malformed unbracketed dimension list is normalized **only in embedding input**, and only if adding brackets produces a fully valid summary. The original JSONL and raw response remain unchanged. Other malformed/empty summaries stop the run. Normalized task IDs and exact input texts are recorded in the embedding manifest and evaluation summary; this is not a claim that the original generation was valid.

The ordinary mode is entirely offline and requires the `.npy` and its `.manifest.json` sidecar. It uses actual scikit-learn `GroupKFold(n_splits=5)` with canonical-query groups and no shuffle (not the previous randomly tie-broken Ridge splitter). Each fold retrieves only from its training tasks; same-query tasks cannot cross train/test. Neighbors are task records, not deduplicated query averages. No labels are used to construct embeddings, folds, distances or tie-breaking.

Vectors are unit-normalized; distance `d` is squared Euclidean distance, similarity is `1/(1+d)`. For K=3/10, `C_edge=mean(sim_j*p_edge_j)` exactly, **not** a similarity-normalized weighted mean. This score is not a calibrated probability. Distance and ranking ties use ascending task ID, not outcomes. No fitted predictor, feature standardization, hyperparameter tuning, RU or BD is used.

Metrics compare OOF `C_edge` to `p_edge`: Spearman, MAE and RMSE. Failure ranking sorts ascending `C_edge`; captured failure mass is the cumulative sum of `1-p_edge` for budgets 0..134. This is empirical failure mass, not offloading gain (no Cloud outcome is used). Oracle sorts descending actual failure propensity. Random is the exact uniform-random expectation `B/134 * sum(1-p_edge)`, with no permutation confidence interval or significance claim.

Default outputs under this directory's `outputs/`:

- `difficulty_embeddings.npy`: 134 x 768 float32 unit-normalized BGE vectors.
- `difficulty_embeddings.manifest.json`: row IDs, exact texts/hashes, bracket-normalization audit, array hash.
- `difficulty_embeddings.checkpoint.json`: resumable unique-text embedding requests.
- `oof_predictions.csv`: 134 tasks, fold, p_edge, K=3/10 capability scores and failure priorities.
- `neighbors.csv`: 1,742 rows (134 x (3+10)), including rank, fold, neighbor ID/query, squared-L2 distance, similarity, neighbor p_edge and score contribution.
- `summary.json`: metrics, fold manifest, full failure-capture curves and provenance.
- `failure_capture_curve.svg`: VDAR K=3/K=10, Random expectation and Oracle curves.

Use `--output-dir` for a new cache if embedding input changes. A mismatched existing cache is rejected instead of silently reused. Do not run concurrent writers. Evaluation never calls the embedding service or reads `.env`.

Offline tests (mock embedding responses only):

```bash
python -m unittest discover -s tests -p test_vdar_feasibility.py
```
