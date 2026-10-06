# Minimal task difficulty generation

From the repository root:

```bash
python experiments/vdar_edge_capability/generate_difficulty.py
```

Reads `outputs/edge_capability_dataset.csv`, selects the first task of each new task type in dataset order until exactly five different types are selected. Selection does not use success or p_edge. No full134 batch, embedding or retrieval.

Uses the exact official VDAR `V2_SYSTEM_PROMPT` frozen in `prompts/vdar_v2_system.txt` (hash checked), and the original user template `**Query to Analyze:**\n{task_instruction}`. Official source remains unchanged. Dynamic model input is ONLY task_instruction; task_id/type and success metadata are not sent.

Reads MemP `.env` with environment variables taking precedence. Requires MEMORY_BUILD_MODEL_NAME; resolves MEMORY_BUILD_API_KEY/base URL with MemP's OpenAI fallbacks, and uses existing memory build temperature/seed/top_k/enable_thinking settings. Standard-library HTTP client sends the equivalent OpenAI-compatible request; no embedding dependencies needed.

Outputs `outputs/difficulty_5_tasks.csv` with exactly task_instruction and difficulty_summary columns, plus `.audit.json` with task selection, exact messages, raw responses and usage. The summary is the content inside `<summary>...</summary>`; missing summary fails rather than fabricating content. Outputs checkpoint after each successful request; existing CSV is not overwritten, so an interrupted partial batch must use another --output or be dealt with explicitly.

`--dry-run` validates prompt and prints the selected five tasks without API calls. Real generation needs network access to the configured memory-build server. Local request on 2026-10-06 failed with URLError; no actual summaries have been generated.
