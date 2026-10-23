# Run 2 execution commands for review

The existing protocol remains `independent`. Shared handoffs are an optional controlled design proposed for team review; the recommendation to make them primary is not an approved protocol change. Both designs require new run directories and a matching smoke. No paid calls are authorized until the credential source and budgets have been approved.

The primary scorer, benchmark reference calls, and `scripts/analyze_run2.py` are unchanged. Use `scripts/analyze_run2_secondary.py` for the design-aware supplemental report and strict secondary checks. It reports primary results using the frozen analysis mathematics, labels the design, exposes the secondary coverage denominator, and validates shared-cache links. It does not replace the original analysis or its saved outputs.

## Setup and offline verification

Run from the repository root on `codex/run2-compression-no-attack`, using Python 3.12. These commands require no API credential. Installing packages and warming the tokenizer download public dependencies; the tests and mock runs make no paid API calls.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-run2.lock
export TIKTOKEN_CACHE_DIR="$PWD/.cache/tiktoken"
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
.venv/bin/python -c 'import tiktoken; tiktoken.get_encoding("o200k_base")'
.venv/bin/python -m pytest -q

.venv/bin/python scripts/run2.py --phase smoke --backend mock --design independent \
  --out-dir experiments/review-v2-independent-mock-smoke
.venv/bin/python scripts/run2.py --phase smoke --backend mock --design shared-handoff \
  --out-dir experiments/review-v2-shared-mock-smoke
.venv/bin/python scripts/analyze_run2_secondary.py \
  --run-dir experiments/review-v2-independent-mock-smoke \
  --out-dir experiments/review-v2-independent-mock-secondary
.venv/bin/python scripts/analyze_run2_secondary.py \
  --run-dir experiments/review-v2-shared-mock-smoke \
  --out-dir experiments/review-v2-shared-mock-secondary
```

The full mock sweep is available by replacing `--phase smoke` with `--phase full` and choosing a fresh output directory. Its repeats are deterministic infrastructure checks, not independent scientific observations. Never use mock success rates to characterize GPT-4o.

## Planning with no paid calls

The following uses the original provisional prices only for planning. Recheck and approve the input/output rates before executing real requests. The completion limit remains 4096 in all arms and stages. Planning does not confer budget or protocol approval.

```sh
.venv/bin/python scripts/run2.py --phase plan --design independent \
  --out-dir experiments/review-v2-independent-plan \
  --input-usd-per-million 2.50 --output-usd-per-million 10 --max-output-tokens 4096
.venv/bin/python scripts/run2.py --phase plan --design shared-handoff \
  --out-dir experiments/review-v2-shared-plan \
  --input-usd-per-million 2.50 --output-usd-per-million 10 --max-output-tokens 4096
```

Nominal request counts before retries are 864 independent or 594 shared for the full 324-trial sweep; smoke uses 32 or 22 requests for 12 trials. Shared mode makes 54 full-run Agent 1 draws, one for each of 18 tasks and three repeat indices. Every one is used, including poor handoffs and baseline failures.

## Authorized real smoke

Only after user approval, load the key from the approved local source into `OPENAI_API_KEY`, and set the approved per-run smoke/full USD allocations and per-million token rates in the variables below. Do not paste credentials into commands, source, manifests, or chat. No spending allocation is provided by this guide. Smoke plus full allocations must fit the approved total budget.

This shell block works in Bash or Zsh. Keep `DESIGN=independent` for the existing protocol. Change it to `shared-handoff` only if the team explicitly approves that design for the run. Using the same label and flags on resume is required.

```sh
export MAS_BACKEND=openai MAS_MODEL=gpt-4o MAS_DELEGATE=1 MAS_TEMPERATURE=0.7
DESIGN=independent
RUN_ROOT="experiments/review-v2-${DESIGN}"
: "${OPENAI_API_KEY:?Load the approved credential source into the environment}"
: "${RUN2_SMOKE_CAP_USD:?Set the approved smoke allocation}"
: "${RUN2_FULL_CAP_USD:?Set the approved full allocation}"
: "${RUN2_INPUT_USD_PER_MILLION:?Set the approved input token rate}"
: "${RUN2_OUTPUT_USD_PER_MILLION:?Set the approved output token rate}"
COMMON=(--design "$DESIGN" --credential-source OPENAI_API_KEY
  --input-usd-per-million "$RUN2_INPUT_USD_PER_MILLION"
  --output-usd-per-million "$RUN2_OUTPUT_USD_PER_MILLION"
  --max-output-tokens 4096)

.venv/bin/python scripts/run2.py --phase preflight "${COMMON[@]}" \
  --max-cost-usd "$RUN2_SMOKE_CAP_USD" --out-dir "${RUN_ROOT}-preflight"
.venv/bin/python scripts/run2.py --phase smoke "${COMMON[@]}" \
  --max-cost-usd "$RUN2_SMOKE_CAP_USD" --out-dir "${RUN_ROOT}-smoke"
.venv/bin/python scripts/analyze_run2_secondary.py \
  --run-dir "${RUN_ROOT}-smoke" --out-dir "${RUN_ROOT}-smoke-secondary"
```

Inspect smoke outcomes and raw traces, the secondary disagreements and manual-review gaps, `ws_32` create/share order and filename handling, model identity, refusals and truncation. A valid real smoke from the other design cannot unlock the full run. If the completion limit or any source/configuration changes, use a new smoke and new directories.

## Authorized full sweep after smoke inspection

Use the same shell variables, source commit and environment as the smoke. This schedules three independent repeat indices for all 18 tasks and six conditions, with all baseline arms first. In shared mode, the repeat indices have separate upstream draws while the six conditions within an index reuse the same saved handoff.

```sh
.venv/bin/python scripts/run2.py --phase full "${COMMON[@]}" \
  --max-cost-usd "$RUN2_FULL_CAP_USD" --smoke-dir "${RUN_ROOT}-smoke" \
  --out-dir "${RUN_ROOT}-full"
.venv/bin/python scripts/analyze_run2_secondary.py \
  --run-dir "${RUN_ROOT}-full" --out-dir "${RUN_ROOT}-secondary"
```

For the existing independent protocol, also run the unchanged primary analysis and figures separately:

```sh
.venv/bin/python scripts/analyze_run2.py \
  --run-dir experiments/review-v2-independent-full \
  --out-dir experiments/review-v2-independent-primary
.venv/bin/python scripts/plot_run2.py \
  --analysis-dir experiments/review-v2-independent-primary \
  --out-dir experiments/review-v2-independent-figures
```

Do not use the frozen independent report's prose to describe shared sampling. The supplement labels the shared design and counts unique upstream usage once from `handoffs.jsonl`; arm-level usage excludes cached Agent 1. The existing spending ledger still covers every actual HTTP attempt, including retries and interrupted work. Durable upstream artifacts are reused on resume; unknown upstream outcomes may need a retry and retain their cost reservation. No client can guarantee exactly one network completion across a crash between a provider response and its local durable write.

Rerun the identical command to resume. Do not edit or delete the cache, journal or ledger to obtain a new draw or reset a cap. Keep independent and shared results in separate directories and tables. Preserve the earlier `e5eb587` manifests and outputs; their source identity is intentionally different from this review revision.
