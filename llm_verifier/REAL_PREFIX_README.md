# 30-record real-prefix feasibility

Local preparation complete at `llm_verifier/results/real_prefix_feasibility_seed42/prepared/`. Sample unit is a real `(task_id, run_id)` record from the three confirmed 4B+offline300 runs, not a distinct task. Python Random seed42 samples 15 of 235 successes and 15 of 167 failures without replacement within each stratum, then shuffles sample order. The same task can occur in different runs. All 402 records have at least three actions; none excluded for short trajectory. No full 402-record verifier batch is performed.

Sync the following to `~/project/MemP/`:

```text
llm_verifier/alfworld_capability.py
llm_verifier/real_prefix_feasibility.py
llm_verifier/results/real_prefix_feasibility_seed42/prepared/scoring_inputs.json
llm_verifier/results/real_prefix_feasibility_seed42/prepared/evaluation_labels.json
llm_verifier/results/real_prefix_feasibility_seed42/prepared/sampling_audit.json
llm_verifier/results/real_prefix_feasibility_seed42/prepared/sample_manifest.csv
```

Only `scoring_inputs.json` is opened by score mode. Labels and sample manifest are separate and never injected. Prefix fields are allowlisted; 120 inputs = 30 records × L0–3. First human message loses appended workflow guidelines; task instruction is separated into task. Each step is the logged executed action (checked against `Action:` in the assistant message) plus the immediately resulting environment observation; assistant thoughts are excluded. Only first three pairs are extracted, not future messages, final reward, total steps, termination, max-step settings or few-shot system messages. A resulting observation is allowed evidence even if it itself indicates goal completion.

The smoke-tested `prompt_for` is imported UNCHANGED; the entire template, criterion, A=best/T=worst scale, official `call_verifier`/`extract_score`, top-20 logprobs and strict fallback refusal are reused. L0 uses the same template with an empty observed-prefix body. Exactly 3 evaluations per prefix: 360 logical verifier evaluations, normally 720 backend calls on the smoke backend (analysis+prefill). No model training/RU/BD. Package is imported from the installed server environment, never copied or modified.

After setting the same verifier configuration as smoke, run:

```bash
cd ~/project/MemP
python llm_verifier/real_prefix_feasibility.py score
```

Default output: `llm_verifier/results/real_prefix_feasibility_seed42/real_run1/`. Interrupted scoring can resume with the same command/configuration; each successful repeat is saved immediately. No synthetic/default scores on failure. Exact prompt and all repeated responses/logprobs stored per sample/L, completed_scores.json generated only after full completion.

Then evaluate separately:

```bash
python llm_verifier/real_prefix_feasibility.py evaluate
```

Evaluation requires numpy, opens final labels only after validating all120×3 scores complete, and verifies label hash. Outputs: evaluation_task_scores.csv (task_id/run_id/L, scores1–3 and mean, label), metrics_by_L.csv (AUROC, Average Precision, success/failure mean/median), metrics_by_L.svg, evaluation_summary.json, human_review_examples.json.

Review examples: highest L3-score failures; lowest L3-score successes; largest label-consistent score change L0→L3; largest label-inconsistent change. Full L0–3 visible prefixes and score sequence attached. Improvement/worsening is an endpoint comparison, not guaranteed monotonic; .5 threshold crossings also listed. Empty category remains empty, never fabricated. Extremes are descriptive ranks, not claims of a calibrated cutoff.

Balanced case-control sample has prevalence .5: AP baseline .5, not deployment prevalence. Repeated task IDs may introduce dependence. Three repetitions reflect verifier variability, not independent task samples. No significance claims or calibrated-success-probability interpretation. No scores/results available locally until the server evaluation is returned.

Optional local dry-run (does not access labels): `python llm_verifier/real_prefix_feasibility.py score --dry-run --run-name prompts_only`.
