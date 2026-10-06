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
