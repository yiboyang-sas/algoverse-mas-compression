# Run 2 design review

No paid calls or switch of the primary design has been approved. The configured primary remains `--design independent`, with GPT-4o, `MAS_DELEGATE=1`, temperature 0.7, the frozen 18 tasks, and the unchanged `scorer-fix` scorer. The shared-handoff option is a reviewable alternative. Selecting it does not establish team approval to replace the primary experiment. Credential-source and budget approval are still required before either real design runs.

## Which question does each design answer?

| Design | Agent 1 across six conditions | Interpretation | Main limitation |
|---|---|---|---|
| `independent` (current default) | A fresh draw for every task × repeat × condition | End-to-end utility of the whole stochastic workflow under each compression condition | A difference may arise before compression because the arms receive different Agent 1 handoffs |
| `shared-handoff` (optional) | One draw per task × repeat, reused by all six conditions | Compression comparison conditional on the same source handoff, including the same delegated actions | Conditions share Agent 1 successes and mistakes; this alone does not establish robustness across independent upstream executions |

For the question “what does compression change when the source handoff is held fixed?”, recommend shared handoffs as the scientific primary **after explicit team approval**. Keep independent draws as an end-to-end confirmation. For a question about whole-workflow reliability, independent draws remain directly relevant. No measured result currently supports a performance claim for either design.

With representative, unselected Agent 1 draws and a stable provider, both designs can estimate the same marginal utility for a condition. They differ in the finite handoff panel and cross-arm correlation: shared draws enable a within-handoff comparison, while independent draws sample more upstream executions (324 rather than 54 in the full sweep). The controlled comparison is conditional on its observed panel; sharing does not make the aggregate a purely downstream score or remove Agent 1 failures from the outcome.

Holding Agent 1 fixed removes one source of between-condition variation. Prompted compressors and Agent 2 still make separate stochastic requests; the six arms do not share their random draws. The fixed baseline-first order can also expose conditions to different service times or provider drift. Shared handoffs do not by themselves identify whether a particular downstream failure was caused by information loss or instruction misunderstanding.

Use all planned task/repeat handoffs, including weak handoffs and baseline failures. Selecting only cases where the baseline succeeds would change the estimand and can bias the main comparison. If examined, such a subset must be a separate exploratory analysis. The three repeat indices sample Agent 1 again; within an index all six arms must share exactly the same prepared handoff.

## Request counts and accounting

Counts exclude retries, interruptions, and rerunning a smoke after a configuration change. The full experiment remains 18 tasks × 3 repeats × 6 conditions = 324 scored trials; smoke remains 2 tasks × 1 repeat × 6 conditions = 12 trials.

| Scope / design | Agent 1 requests | Agent 2 requests | Prompted-compressor requests | Total requests |
|---|---:|---:|---:|---:|
| Full independent | 324 | 324 | 216 | 864 |
| Full shared handoffs | 54 | 324 | 216 | 594 |
| Smoke independent | 12 | 12 | 8 | 32 |
| Smoke shared handoffs | 2 | 12 | 8 | 22 |

`none` and `sliding_window` make no compressor request. Request-count savings are not a measured dollar saving: actual lengths, cached-input billing, retries, and unknown charges matter. The same per-attempt reservation ledger remains the spending authority under either design.

In shared mode, `handoffs.jsonl` is the persistent Agent 1 cache for one run. Cache keys are task ID and repeat index, bound to the immutable manifest, task content, and configuration. Trial records link the source using `shared_handoff_id`, `shared_handoff_sha256`, and `prepared_agent1_sha256`. All conditions must consume the identical augmented handoff; delegation happens before it is cached and before any compression.

Agent 1 usage is authoritative once in the cache. Arm-level `provider_usage` counts only new compressor and Agent 2 requests; a reused Agent 1 trace is retained as provenance and marked reused. Do not sum that retained trace six times when reporting cost or token consumption. Reconcile unique HTTP reservations in `budget.jsonl`, including partial attempts in `errors.jsonl`, with cache and trial links. Unknown or interrupted HTTP charges retain their full reservation.

Resume must reuse a completed cache record rather than regenerating a more favorable Agent 1 draw. Corruption, identity changes, or cross-design reuse must fail closed. A valid smoke must use the same design as the full run. Changing the design requires new output directories and a new smoke; independent and shared results must not be pooled into one condition table.

## Scoring and uncertainty

The primary outcome stays the existing required-write success proxy. `environment.py`, reference `required_calls`, and their match modes remain unchanged. The benchmark correction remains exactly four `allowed_tools` additions, preserving all other fields. Original dataset SHA256: `348b8ab77d10bb96ecc6cb4ca1b03e035edd679167d7f9b13ac028bbfb91f469`; corrected dataset SHA256: `203bb3c7bc21e73c701ee06a06857aa82800a6bd6f0612b898602cdd0c5a5c8e`.

The stricter evaluator is a separately versioned secondary analysis with separate references in `tasks/run2_secondary_references.json` (`run2-secondary-reference-conformity-v1`) and separate outputs. A strict pass requires an unchanged primary pass plus the additional checks. More exact call checks detect extra actions, recipient supersets, reused calls, and explicit create-before-share dependencies; independent TODO email actions have no invented order constraint. Lexical content checks can miss valid paraphrases and cannot establish useful suggestions or a faithful summary. Eleven of the 18 tasks are eligible for aggregate strict reference conformity under this policy; the others retain null strict-conformity values and explicit manual-review requirements. Neither a primary pass nor a secondary pass establishes full AgentDojo execution utility.

The supplemental analyzer reuses the frozen primary analyzer's validation, task-cluster bootstrap, and paired task/repeat mathematics. Its separate report labels the actual design, shows the all-task primary rate beside the 11-task covered-subset primary and strict rates, and records unique cached Agent 1 usage once. Avoid comparing a strict subset rate with an all-task primary rate as though their denominators were identical. Use this entry point for shared-mode reporting; the unchanged legacy primary report describes independent upstream draws.

```sh
.venv/bin/python scripts/analyze_run2_secondary.py \
  --run-dir experiments/selected-design --out-dir experiments/secondary-analysis
```

This command calls no model. It creates `secondary_summary.csv`, `secondary_per_task.csv`, `secondary_details.jsonl`, `secondary_analysis.json`, `secondary_results.md`, `secondary_provenance.json`, and `secondary_usage.json`, without overwriting primary analysis artifacts. In shared mode the pairing fixes Agent 1; preserve all repeats and all condition arms within each resampled task. Eighteen related tasks and three repeats support exploratory uncertainty estimates, not strong generalization. Homogeneous observations can yield degenerate empirical intervals without implying certainty. Report the design beside each table, distinguish measured from unavailable values, and retain manual failure-review notes.

## Local verification before review

Run the offline tests and mock smoke for both flags in distinct directories. Check that all six shared arms link to the same cache record, independent arms make separate Agent 1 requests, a resumed shared run reuses its cache, and design/source/task changes are rejected. Verify that unique request counts and cost aggregation match the table above. Test the secondary evaluator with primary-pass counterexamples and valid variations; inspect its coverage exclusions. These checks validate implementation only. Real-model success, effects of compression, and causal failure diagnoses remain unverified until approved API experiments and review are complete.
