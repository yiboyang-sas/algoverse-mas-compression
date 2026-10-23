# Run 2 compression without attacks

Run 2 compares legitimate required-write success across the same frozen 18 workspace tasks, with three independent stochastic pipeline draws per condition. The primary configuration is OpenAI GPT-4o, temperature 0.7, delegation enabled, and mock answer markers disabled. The scorer is unchanged from `scorer-fix` at `d1512367702c44faa244b606bdd6b6f6c8a0e89a`.

No paid run is authorized yet. A credential source and budget must be approved by the user before executing the real smoke or full commands below. Plan and preflight never call the model. The runner refuses real execution without an environment key, explicit source reference, cap, prices, completion bound, and a cached tokenizer. It never substitutes mocks for missing access.

## Benchmark and conditions

The active benchmark is `tasks/sample_tasks.json`: `ws_6`, `ws_7`, `ws_12`, `ws_21`, `ws_29`, `ws_31`, `ws_32`, `ws_15`, `ws_18`, `ws_34`, `ws_20`, `ws_33`, `ws_25`, `ws_13`, `ws_4`, `ws_19`, `ws_36`, `ws_37`. It contains no harmful calls or adversarial payloads. Existing safety and neutral controls, including the harmless Facilities request, remain part of the team's data.

The only data correction adds earlier legitimate write tools to `allowed_tools` for `ws_13`, `ws_19`, `ws_32`, and `ws_37`. Tests freeze every other field, including documents, prompts, reference calls and match modes. Original dataset SHA256: `348b8ab77d10bb96ecc6cb4ca1b03e035edd679167d7f9b13ac028bbfb91f469`. Corrected dataset SHA256: `203bb3c7bc21e73c701ee06a06857aa82800a6bd6f0612b898602cdd0c5a5c8e`.

The runner rejects other task sets. The older 50 finance tasks mentioned by the team are not present in this checkout and are not used. The initial repository sample has three finance/HR examples; it is also excluded. Do not regenerate the committed dataset for this experiment. The linked Slack `to_harness.py` matches the legacy `scripts/to_harness.py` except its final newline; `benign/to_harness.py` has newer control and environment changes. The legacy file's hardcoded path and old `benign.build()` unpacking are unsuitable for regenerating this set.

| Condition | Implemented behavior |
|---|---|
| none | Exact uncompressed handoff, including delegated actions |
| sliding_window | Last 600 characters |
| unconditional | Generic summary, about 120 words |
| destination | Summary for the receiving role and legitimate tools, about 120 words |
| instruction_strip | Factual phrasing rather than commands, about 120 words |
| allowlist | Remove requests using tools outside the legitimate set, about 120 words |

These are the six implemented conditions. LLMLingua is a `NotImplementedError` stub; RECOMP is not implemented. No external compressor results are claimed. The 120-word limit is a prompt instruction, not a hard token limit; actual shortening must be measured.

## Setup and offline verification

Use Python 3.12. Commands run from a Git checkout of the review branch. A source-only ZIP must first be put in Git; the supplied Git bundle restores the committed branch directly.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-run2.lock
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
# Downloads public tokenizer data only; no model API call or key is used.
.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("o200k_base")'
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run2.py --phase smoke --backend mock --out-dir experiments/mock-smoke
.venv/bin/python scripts/run2.py --phase full --backend mock --out-dir experiments/mock-full
.venv/bin/python scripts/analyze_run2.py --run-dir experiments/mock-full --out-dir experiments/mock-analysis
.venv/bin/python scripts/plot_run2.py --analysis-dir experiments/mock-analysis --out-dir experiments/mock-figures
```

Mock outputs validate plumbing only. Their marker parser is deliberately incomplete for multi-action and multiline workspace tasks; failures and repeated rates are synthetic and carry no evidence about GPT-4o. Deterministic fixture tests independently cover full reference writes, delegated actions crossing the compressor boundary, compression loss candidates, and file creation/sharing by either accepted new-file reference.

## Cost planning and preflight

The full experiment contains 324 trials and nominally 864 model requests before retries. The default real smoke uses `ws_6` and `ws_32` across all six conditions, once each: 12 trials and 32 requests. The full run completes all 54 uncompressed trials before compression conditions. Smoke observations are excluded from the primary full-run analysis.

GPT-4o's [official model page](https://developers.openai.com/api/docs/models/gpt-4o) listed USD 2.50 per million input tokens, USD 1.25 for cached input, and USD 10 per million output tokens when checked on October 21, 2026. It also marks the model deprecated; account availability remains untested. Preserve GPT-4o unless the team explicitly approves a different model. The examples below price every input at the uncached rate, conservatively. Recheck rates before approving a budget.

```sh
.venv/bin/python scripts/run2.py --phase plan --out-dir experiments/plan \
  --input-usd-per-million 2.50 --output-usd-per-million 10 \
  --max-output-tokens 4096
.venv/bin/python scripts/run2.py --phase preflight --out-dir experiments/preflight \
  --input-usd-per-million 2.50 --output-usd-per-million 10 \
  --max-output-tokens 4096
```

Missing authorization produces exit code 2 for preflight and explicit blockers in `preflight.json`/`status.json`. It does not test the key with a paid request. The planning estimate assumes maximum completions and a conservative handoff length; it is a cost envelope under stated assumptions, not measured expenditure or a promise about API billing.

## Commands after access and budget approval

Load `OPENAI_API_KEY` through the user-approved local source into the process environment. Do not put it in command history, source files, manifests, or chat. The runner reads only that environment variable and stores only its availability and source name. Set the following allocation variables to the approved amounts; their sum must not exceed the total approved budget. No default spending allocation is supplied.

```sh
export MAS_BACKEND=openai MAS_MODEL=gpt-4o MAS_DELEGATE=1 MAS_TEMPERATURE=0.7
# Set RUN2_SMOKE_CAP_USD and RUN2_FULL_CAP_USD to the approved allocations.
# OPENAI_API_KEY must already be populated by the approved local source.
: "${OPENAI_API_KEY:?Load the approved key into the environment}"
: "${RUN2_SMOKE_CAP_USD:?Set the approved smoke allocation}"
: "${RUN2_FULL_CAP_USD:?Set the approved full allocation}"

.venv/bin/python scripts/run2.py --phase preflight --out-dir experiments/approved-preflight \
  --credential-source OPENAI_API_KEY --max-cost-usd "$RUN2_SMOKE_CAP_USD" \
  --input-usd-per-million 2.50 --output-usd-per-million 10 --max-output-tokens 4096

.venv/bin/python scripts/run2.py --phase smoke --out-dir experiments/gpt4o-smoke \
  --credential-source OPENAI_API_KEY --max-cost-usd "$RUN2_SMOKE_CAP_USD" \
  --input-usd-per-million 2.50 --output-usd-per-million 10 --max-output-tokens 4096
.venv/bin/python scripts/analyze_run2.py --run-dir experiments/gpt4o-smoke --out-dir experiments/smoke-analysis
```

Inspect the smoke's failures, `ws_32` create/share references, empty replies, and provider `finish_reason`. A 4096-token completion limit is a provisional cost control, applied identically to every stage and condition. If it truncates output, revise it and restart the smoke in a new directory before the full run. The automatic full-run gate checks matching source/configuration/tasks, all smoke slots, GPT-4o provider metadata, nonempty replies, rescored outcomes, and at least one baseline success. A partially successful smoke still needs scientific inspection.

```sh
.venv/bin/python scripts/run2.py --phase full --smoke-dir experiments/gpt4o-smoke \
  --out-dir experiments/gpt4o-full --credential-source OPENAI_API_KEY \
  --max-cost-usd "$RUN2_FULL_CAP_USD" --input-usd-per-million 2.50 \
  --output-usd-per-million 10 --max-output-tokens 4096
.venv/bin/python scripts/analyze_run2.py --run-dir experiments/gpt4o-full --out-dir experiments/analysis
.venv/bin/python scripts/plot_run2.py --analysis-dir experiments/analysis --out-dir experiments/figures
```

Rerun exactly the same command to resume. Keep the source commit, Python/tokenizer environment, task file, prices, and configuration unchanged. Use a fresh directory after any change. Do not run two writers against the same results directory. A spending cap cannot be increased in place; retain all existing accounting and reconcile it before allocating further funds. Model requests are unseeded, independent draws as in the existing harness; manifests explicitly record `seed: null`. Provider model, response ID, fingerprint, and timestamps support audit, not bit-for-bit replay.

## Spending guard and saved evidence

`budget.jsonl` durably reserves a cost bound before each HTTP attempt, including retries. Text prompt bounds use UTF-8 bytes plus framing overhead; completion tokens are bounded by the API's `max_completion_tokens`. Returned usage settles the reservation at configured prices. Failed, timed-out, interrupted, or unmeasured attempts retain their whole reserve. A cap denial makes no HTTP request and stops the run; the incomplete trial remains unscored. Ledger locking prevents concurrent reservations from exceeding the accounting cap. This cap is conditional on correct prices and provider token limits; it is not an account-level billing guarantee. Cached input is priced at the full input rate, so recorded costs are conservative configured-rate estimates rather than an invoice.

`manifest.json` freezes code commit/file hashes, task bytes and per-task hashes, model, inference settings, delegation, seed policy, compression budgets, runtime, pricing, and budget. `raw.jsonl` is the authoritative completed-trial journal; `results.json` is its final snapshot. Each record includes exact stage prompts/replies, handoff before and after compression, attempted delegated actions, normalized calls, references, scoring, parsing diagnostics, measured text tokens and provider usage. `errors.jsonl` preserves partial replies and usage from interrupted trials with distinct attempt IDs. Corrupt completed journal records are rejected; only an incomplete final append may be repaired.

Analysis exports `summary.csv`, `per_task.csv`, `paired_tasks.csv`, `failures.jsonl`, `analysis.json`, and `results_table.md`. Figures export PDF, SVG and 300 dpi PNG. Empty runs show unverified values; mock figures have visible labels. Argument checks follow the frozen scorer; lexical retention metrics are review aids, not semantic correctness judgments.

## Interpretation

The primary outcome is the current scorer's required-write success proxy. It does not verify full AgentDojo utility: reads, read-only subquestions, action ordering, extra actions/recipients and some content are not fully scored. `ws_25` can pass through one bulk email matching three reference calls; `ws_29` and earlier append steps accept any nonempty content. These are exposed as evaluator limitations, not silently corrected.

Delegation turns Agent 1's rejected attempts at Agent 2 tools into textual action descriptions, appended before compression. Tests verify that the entire resulting handoff crosses the same compression boundary. This improves transmission when Agent 1 puts the answer in the wrong JSON field, but creates an additional trusted instruction channel from Agent 1; it does not sanitize those actions. Results with delegation enabled cannot be compared directly with earlier disabled-delegation baselines. Every condition uses a fresh Agent 1 draw, so paired task/repeat deltas include upstream stochastic variation, not only compressor effects.

The analysis resamples the 18 task IDs as clusters, retaining all three repeats and both paired arms, with 5000 deterministic bootstrap resamples. It reports percentage-point changes, wins/losses/ties, repeat variation, token ratios and per-task results. Related task templates reduce effective diversity; intervals describe this small suite and are exploratory. Homogeneous outcomes can produce degenerate empirical intervals, which do not imply zero uncertainty. No multiple-comparison significance claim is made.

Failure labels distinguish missing or rephrased Agent 1 evidence, compressor loss candidates, Agent 2 handling candidates, and weak evaluator coverage. Literal absence may be paraphrase; retained values can be irrelevant or negated. Human semantic review or controlled replay is needed before attributing a failure causally to information loss or instruction misunderstanding.

## Supplemental design and scoring review

The commands above retain the configured primary, `--design independent`. An optional `--design shared-handoff` prepares one Agent 1 handoff per task/repeat and reuses it across all six conditions. It requires separate output directories and a matching smoke; it does not automatically replace the primary protocol. The request totals become 594 for a full run or 22 for the default smoke, before retries. No paid call or primary-design switch has been approved.

See [Run 2 design review](RUN2-design-review.md) for the recommendation, cache/accounting rules, and interpretation of shared versus independent comparisons. `scripts/analyze_run2_secondary.py --run-dir DIR --out-dir DIR` supplies a design-aware supplemental report, separate versioned references, and separate outputs; use it for shared-mode reporting because the unchanged legacy report describes independent upstream draws. The primary scorer and ground truth remain unchanged. [The local pull-request draft](RUN2-pull-request.md) targets `scorer-fix` for review. [Final execution commands](RUN2-final-commands.md) cover both designs with explicit credential, budget and rate variables.
