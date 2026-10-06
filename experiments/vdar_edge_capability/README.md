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
