# ALFWorld prefix capability smoke test

Independent MemP adapter. Does not copy, modify, monkeypatch or inject a path to official llm_verifier source. Run using the server Python environment in which llm_verifier is already installed. Sync this directory to `~/project/MemP/llm_verifier/`. This adapter directory deliberately has no `__init__.py`: do not add one, as it could shadow the installed official package. Run the file directly as below, not via `python -m llm_verifier`.

Official mechanism inspected: `progress.py`, `fine_grained_reward.py`, `prompts.py`, `criteria/TEMPLATE.md`. Progress scale A=0/T=1 assesses current satisfaction; its offline track sees future steps. This adapter instead supplies the requested future-capability criterion, one observed prefix at a time, and uses official fine_grained_reward scale A=1/T=0. `call_verifier` obtains top-20 logprobs; on compatible vLLM/SGLang it generates analysis then prefills `<score_A>` with constrained A–T sampling. `extract_score` renormalizes recognized score-token probabilities and computes the expected normalized value, then we average repeated evaluations. One logical repeat may incur two backend requests. Strict audit refuses missing score logprobs, text-only parsing or silent .5 defaults.

Six manually constructed synthetic prefix cases, three matched task pairs: grounded object acquisition vs repeated nonexistent-object actions; heating vs cooling; successful cleaning vs repeating invalid cleaning attempts. They are not actual ALFWorld environment rollouts; action/observation strings are illustrative. No final reward, future steps, termination, total rollout length or expected-group label enters prompts. Good cases remain incomplete. Smoke expected-group labels are descriptive fixtures, not true future outcomes. Scores are verifier judgments, not calibrated probabilities. Failure to exceed a predeclared .15 mean-score gap in every pair is reported, never rewritten as a pass.

Dry preparation (no installed package/API needed):

```bash
cd ~/project/MemP
python llm_verifier/alfworld_capability.py --dry-run \
  --output-dir llm_verifier/results/alfworld_capability_verifier_smoke_2026-10-05/prompts_only
```

Real evaluation, using your chosen backend/model configuration:

```bash
export VERIFIER_MODEL='your-served-verifier-model'
export VERIFIER_BASE_URL='http://your-server:port/v1'
export VERIFIER_API_KEY='your-key'
python llm_verifier/alfworld_capability.py --repeats 3 \
  --output-dir llm_verifier/results/alfworld_capability_verifier_smoke_2026-10-05/real_run1
```

Alternatively `--env-file` reads these three variables without printing secrets. Without VERIFIER_BASE_URL the official client resolver uses OPENAI_BASE_URL, DEEPSEEK_API_KEY or VERTEX_API_KEY. Specify VERIFIER_MODEL explicitly, never implicitly use the Edge model. OpenAI-compatible service must support the official score prefill/structured_outputs mechanism; a generic endpoint supporting only sampled text may fail. Model/package must actually be accessible on server; no package installations are performed by the runner.

Outputs: exact .prompt.txt files and prompts.json; per-repeat response, tokens, logprob alternatives and score; case repeated_scores/mean/SD; matched-pair gaps and pass/fail; model/package provenance and usage. Group labels are included in analysis artifacts only. Use a fresh output directory; partial scores persist if a request fails. The official package may perform its own fallback/retries, but this adapter never substitutes missing results with a score. No full logs are processed.
