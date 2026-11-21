# Compression redesign — implementation plan

Written after inspecting the repository, before any implementation. **Phase 1
is now built and tested; see §6. Phases 2 onwards have not started.**

- snapshot tag: `pre-compression-redesign` → `7608c47`
- archive: `archive/pre_compression_redesign/` (+ README recording config,
  baseline, known problems, restore instructions)
- working branch: `compression-redesign`, forked from `9cb4883`
- previous branch `multi-suite-and-plan-regime` left untouched at `9cb4883`

---

## 1. What is actually here

### Entry points

There are **two runners** and the distinction matters.

| file | role |
|---|---|
| `run.py` | the team's original runner. Sweeps compression conditions. Not what produced any current result. |
| `baseline.py` | **the real entry point.** 40 KB. Run 1 only, condition hard-coded to `"none"`. |

`baseline.py` is not a script that uses `pipeline.py` — it is a **monkey-patch
layer over it**. At import it reassigns seven module attributes:

```
pipeline.TOOL_REGISTRY        = environment.TOOL_REGISTRY     (line  88)
pipeline.score_run            = score_checkable               (line 233)
pipeline._build_agent1_input  = build_without_markers         (line 266)
pipeline._parse               = _parse_tolerant               (line 353)
pipeline.run_pipeline         = run_with_schema               (line 567)
pipeline._replay              = _replay_safe                  (line 588)
pipeline.run_pipeline         = _run_capturing                (line 938)
```

Note `run_pipeline` is replaced **twice** — `_run_capturing` wraps
`run_with_schema` wraps the original. Any change to the pipeline contract has
to respect that chain, and the order of those assignments is load-bearing.

### Module map against Part 0.1

| requested | file | symbol |
|---|---|---|
| pipeline | `pipeline.py` | `run_pipeline`, `_build_agent1_input`, `_replay`, `_parse` |
| Agent 1 prompt | `agent1_prompts.py` | `REGIMES` dict: summarise / plan / verbatim / scoped / defended; `brief` lives in `baseline.py` as `_BRIEF_SYSTEM` |
| compressor | `compressors.py` | `compress()`, `BUDGET_TOKENS`, six conditions |
| Agent 2 prompt | `baseline.py` | `_AGENT2_TEMPLATE`, `_PerThreadSystem`, `_signatures`, `TOOL_PARAMS`, `TOOL_DESC` |
| task loading / tiering | `benign_all.py` | `build_suite`, `tier_of`, `THIN_BELOW`, `OVERSIZED_ABOVE` |
| scoring | `baseline.py` | `score_checkable`, `_args_match`, `_norm` |
| benign critical fields | `obtainable.py` | `prune`, `findable`, `PROSE_KEYS`, `DATE_KEYS` |
| adversarial scoring | `baseline.py` | inside `score_checkable`; `harmful_critical` is **null on all 51 tasks** |
| synthetic expansion | `filler.py`, `filler_suites.py`, `suites.py` | `inflate_*`, `Gen`, `transaction`, `unit_*` |
| control markers | `to_harness_all.py` | `C_TEXT`, `N1_TEXT`, `N2_TEXT` — **fields only, never planted** |
| raw capture | `baseline.py` | patch 11, `_run_capturing`, `_cap` thread-local |
| budgets | `compressors.py` | `BUDGET_TOKENS` (env `MAS_BUDGET_TOKENS`), default 150 |
| tests | — | **none. Zero test functions in the repository.** |
| outputs | `baseline_results.json`, `run1_log.txt` | archived |

### Where the compressor sits

`pipeline.py:77` — one call:

```python
compressed = compress(condition, handoff, compressor_backend, task)
```

**Nothing in `pipeline.py` or `compressors.py` counts a token.** Every metric
in Parts 3, 19 and 20 has to be added; none can be derived from stored data.

---

## 2. Conflicts between the brief and the code

These are Part 31 stop conditions. Three need a decision from you; the rest I
can work around.

### 2.1 Competing implementations — Part 31 item 6 — **DECIDED: leave them**

The repository contains duplicated pipelines:

| | |
|---|---|
| `to_harness.py` vs `scripts/to_harness.py` | **different files**, same name, both build task JSON |
| `benign.py` vs `benign_all.py` | workspace-only vs all four suites |
| `tasks/sample_tasks.json` vs `tasks/sample_tasks_all.json` | 18-task legacy vs 51-task current |
| `run.py` vs `baseline.py` | two runners |

**Decision: nothing moves.** Files stay where they are; provenance is written
down instead, which is what was actually missing.

**Every result quoted so far — the 82% gate and the 73% build after the
harder banking distractors — came from:**

```
python baseline.py --tasks tasks/sample_tasks_all.json --repeats 3
```

that is, **`baseline.py` + `tasks/sample_tasks_all.json`**, 51 tasks, four
suites, `MAS_AGENT1=plan`, condition hard-coded to `"none"`, budget 150
tokens, `gpt-4o-mini`. `run1_log.txt` and the archived
`baseline_results.json` are from that path.

**Nothing has ever been run from the other half.** These are the legacy
paths, kept but not used:

| path | status |
|---|---|
| `benign.py` | workspace-only task builder. Superseded by `benign_all.py`. |
| `to_harness.py`, `scripts/to_harness.py` | two files, same name, both superseded by `to_harness_all.py` |
| `tasks/sample_tasks.json` | the 18-task workspace-only set. No current result uses it. |
| `run.py` | the team's original sweep runner. **Has never produced a reported number.** |
| `tasks/attack_tasks.json` | 3.6 KB of the team's original mock scaffolding, not AgentDojo injections |

One consequence worth stating: `baseline.py`'s `--tasks` still **defaults to
`tasks/sample_tasks.json`**, the legacy 18-task file. Every real run has
passed `--tasks` explicitly. Anyone who forgets gets the old set and no
warning, so always pass it.

### 2.2 Agent 2 does not see the task — Part 31 item 2 — **DECIDED: Option A**

Verified in `pipeline.py:84`:

```python
a2_input = f"Handoff from Agent 1:\n{compressed}"
```

That is the whole input. Agent 2 receives no task text, which is why
`obtainable.prune` forces the PLAN to carry task-stated values: measured, **40
of 117 required values exist only in the task prompt**.

Part 8 wants Agent 2 to do the reasoning — infer "March amount = 50, so
difference = 5". But Agent 2 cannot know the task says *"pay the March
difference"* unless something tells it. Three options:

| option | effect on the instrument |
|---|---|
| A. Agent 1 writes a one-line objective into the handoff | objective is compressible, so it can be lost — consistent with the current design |
| B. Agent 2 is given the task text directly | Agent 2 can recover from a lossy handoff; compression damage becomes invisible. **Breaks the instrument.** |
| C. Agent 2 gets a fixed abstract goal, no task specifics | weakest; most tasks become unanswerable |

**Decision: Option A.** Agent 1 writes a one-line objective into the handoff,
and that line goes through the compressor like everything else.

The reason this is the right one and not just the convenient one: the
objective being compressible is not a weakness of the design, it is the
measurement. If the objective can be lost, then losing it is a result —
"compression removed the reason for the action and Agent 2 acted anyway" is
one of the failure modes the paper is looking for. Option B hands Agent 2 the
task text directly, which lets it recover from a lossy handoff, and then
compression damage becomes invisible no matter how bad it gets. That is not a
weaker instrument, it is a broken one.

**Recorded, not implemented.** Phase 1 does not touch Agent 1's prompt.
Option A lands with the evidence regime in Phase 5 (§3, Part 7/8), where the
prompt is being rewritten anyway, so the objective line and the regime change
can be measured together instead of confounded.

### 2.3 `harmful_critical` cannot be populated yet — Part 31 item 5 — **CONFIRMED blocked**

`harmful_critical` is null on all 51 tasks because **no attack specs exist
yet**. `tasks/attack_tasks.json` is 3.6 KB of the team's original mock
scaffolding, not AgentDojo injections. Sourayan owns the attack layer and has
not delivered.

Part 12 asks for symmetric critical-field scoring. I can build the
**mechanism** and test it against synthetic attack specs. I cannot populate
`harmful_critical` for real attacks that do not exist. Flagged rather than
invented, per Part 32.

**Decision: confirmed.** `harmful_critical` stays null until Sourayan's
attack specs exist. The mechanism may be built and tested against **clearly
labelled synthetic mock specs only**, which live in `tests/` and never in
`tasks/`. No attack-critical field gets invented, and nothing is written into
the 51 production tasks. `RunConfig.attack_enabled` and
`RunConfig.controls_enabled` both default to `False` and
`tests/test_config.py` asserts it, so a config on disk cannot claim an attack
ran when none did.

### 2.4 Not blocking, but worth stating

- **No tests exist.** Part 33's test suite starts from zero, not from
  extending something.
- **`baseline.py` is a patch layer, not a program.** Part 28's centralised
  config is a genuine refactor of 40 KB of monkey-patching, not a config file
  drop-in. I propose introducing config alongside the patches and migrating
  incrementally rather than rewriting.
- **`tier_of` already filters tasks** on raw context size (`THIN_BELOW=1000`).
  That is a second, older eligibility mechanism that predates CER and will
  interact with it. Parts 4 and 24 need to know it exists.

---

## 3. Where each requested change goes

### Part 3 — CER and token accounting

| | |
|---|---|
| files | new `metrics.py`; `pipeline.py`; `baseline.py` |
| change | `count_tokens()` using tiktoken with a chars/4 fallback. `run_pipeline` records `source_context_tokens`, `precompression_handoff_tokens`, `compressor_budget`, `compressed_tokens`, `cer`, `realized_ratio`, `fraction_removed` |
| old behaviour | preserved — additive fields only |
| risk | tiktoken needs a network fetch for its vocab; the fallback must be explicit in the record so nobody quotes an estimate as measured |
| validation | unit tests on known strings; zero/empty handoff must not divide by zero |

### Part 4 — strata

| | |
|---|---|
| files | new `config.py`; `metrics.py` |
| change | `CER_STRATA = {"unexposed": (0,1.0), "weak": (1.0,1.5), "moderate": (1.5,3.0), "strong": (3.0,inf)}`, `PRIMARY_CER_MIN = 1.5` |
| note | strata are an **analysis label**, never a filter in the runner. Part 24. |

### Part 10/11 — controls planted and tracked per stage

| | |
|---|---|
| files | `benign_all.py` (planting), new `controls.py` (detection), `pipeline.py` (capture) |
| change | insert C/N1/N2 into the documents each task actually reads, reusing `filler.block_boundaries`; record `present_in_documents / handoff / compressed / expressed_in_action` |
| risk | **a planted control must not change a ground-truth answer.** `verify_all.py` already catches that and must pass after planting |
| validation | assert each control string appears in `documents` for every task; `verify_all` still 57 pass / 0 broken |

### Part 12 — symmetric scoring

| | |
|---|---|
| files | `baseline.py` (`score_checkable`), `obtainable.py` |
| change | `harmful_success` matches `harmful_critical` the way benign matches `handoff_critical`; prose excluded unless an attack declares it critical |
| blocked on | §2.3 |

### Part 13 — failure taxonomy

| | |
|---|---|
| files | new `failure_stage.py` |
| change | classify from raw traces only; `unknown` when evidence is absent |
| key rule | if the correct record is in the handoff and the PLAN already names a distractor, that is `AGENT1_SELECTION`, **never** `COMPRESSOR_LOSS` (Part 13, bk_5) |

### Part 15 — shared mutable state

Audit beyond the one known bug. Current per-task writes to module globals:

- `pipeline.AGENT2_SYSTEM = _PerThreadSystem()` — written per task inside
  `run_with_schema`. Reads thread-local, so it is safe, but it is still a
  global write in a concurrent path and should become explicit passing.
- `_a2.prompt`, `_cap.buf` — thread-locals, correct, but untested.

Proposed: a `TaskRunContext` passed explicitly; globals kept as a shim so the
old path still runs. Add a concurrency test asserting serial and 8-worker runs
produce identical per-task configuration.

### Part 7/8 — evidence regime

| | |
|---|---|
| files | `agent1_prompts.py` (new `EVIDENCE` entry), `baseline.py` |
| change | `handoff_regime = plan | evidence` via existing `MAS_AGENT1` |
| **`plan` is not removed** | Part 32, and old results stay reproducible |
| depends on | §2.2 decision |

### Part 6/18 — native vs stress_test

| | |
|---|---|
| files | `config.py`, `benign_all.py`, `suites.py` |
| change | `experiment_mode`; native disables `INFLATE`; stress_test keeps it with levels. Record `native_tokens`, `synthetic_tokens`, `synthetic_fraction` per task |
| note | **native will produce mostly unexposed tasks** — slack is 87 native tokens median. That is the finding, not a bug (Part 26) |

### Part 17 — budget sweep

`compressors.BUDGET_TOKENS` becomes a parameter rather than a module global
read at import. Today it is read once, so a sweep in one process is impossible.

### Part 20/21 — analysis exports

New `export_runs.py` → JSONL + CSV with the Part 20 columns. No plots until
the columns exist.

---

## 4. Phase 1, which is what I propose to do next

Per Part 30, and nothing beyond it.

1. `config.py` — `RunConfig` dataclass, saved with every run
2. `metrics.py` — token counting, CER, realized ratio, fraction removed
3. `pipeline.py` — record sizes at every stage; **`source_context_tokens` and
   `precompression_handoff_tokens` as separate fields** (Part 19)
4. strata labelling in `metrics.py`, thresholds in `config.py`
5. raw-trace capture hardened and written as JSONL to `results/redesign/<run_id>/`
6. API-failure handling audited against Part 14 — already excludes, needs
   per-call `requested/succeeded/error/retries/parse_status` records
7. shared-state audit + concurrency test
8. run-id, git commit, dirty flag, prompt hashes persisted

Then: unit tests, and a **4-task smoke run** — `bk_5` (unexposed), `ws_19`
(strong), `sl_15` (prose answer), `bk_9` (qualifier) — roughly 8 API calls.

**No full 51-task run.**

---

## 5. Decisions taken

All three answered. See §2.1, §2.2 and §2.3 above.

1. **§2.2** — Option A. Agent 1 carries the objective; it stays compressible.
   Recorded now, implemented in Phase 5.
2. **§2.1** — nothing moves. Provenance written down instead.
3. **§2.3** — confirmed. `harmful_critical` stays null; the mechanism may be
   tested against labelled mock specs in `tests/` only.

---

## 6. Phase 1 as built

Everything below runs with no API key and no agentdojo install.

### New modules

| file | what it does |
|---|---|
| `config.py` | `RunConfig`, written to `results/redesign/<run_id>/config.json` with the git commit, the dirty flag and a hash of each prompt used. `CER_STRATA`, `PRIMARY_CER_MIN = 1.5`, `BUDGET_SWEEP`. |
| `metrics.py` | `count_tokens() -> (n, method)`. `CompressionMetrics` with source, Agent 1 input, handoff and compressed sizes kept apart. `stratum_of`, `pooling_is_safe`. |
| `tracing.py` | `TraceWriter`: thread-safe append-only JSONL, flushed per line. `calls.jsonl` (one line per model call) and `runs.jsonl` (one line per task run). |
| `instrument.py` | `TaskRunContext` in a thread-local, the traced-backend wrapper, stage labelling, and `SHARED_STATE_AUDIT`. |
| `suite_tools.py` | `ALL`/`READ`/`WRITE`, extracted so the runner no longer imports the task builder. |
| `run_tests.py` + `tests/` | the first tests in this repository. |

### Token measurement

Every count records how it was obtained. `tiktoken:<encoding>` is a
measurement; `estimated_chars_div_4` is an estimate, and the runner prints a
warning beside it. `--require-measured-tokens` refuses to start on an
estimate, for anything whose CER gets published. `metrics.pooling_is_safe`
exists so the analysis layer can refuse to average the two.

The estimate path is not hypothetical: `tiktoken` fetches its vocabulary from
`openaipublic.blob.core.windows.net` on first use, and that host is blocked
from both sandboxes this repo gets inspected in. It works in an ordinary
terminal with `pip install tiktoken`.

### The budget

`compressors.BUDGET_TOKENS` was read once at import and baked into the four
compressor prompt strings, so a sweep inside one process was impossible and a
saved result carried no record of which budget produced it. `compress()` now
takes `budget_tokens`, the prompts are built per call, and the value reaches
it through the run context. It is recorded on every run, because CER divides
by it.

### Shared mutable state

Audited in full; the list is in `instrument.py` under `SHARED_STATE_AUDIT`.
One per-task global remains, `pipeline.AGENT2_SYSTEM`, and it holds a
`_PerThreadSystem` whose `format()` answers from a thread-local.
`tests/test_concurrency.py` holds all eight workers at a barrier until every
one of them has written that global before any of them reads it, which turns
the race from an occasional failure into a certain one. Verified by
reintroducing the old plain-string assignment: two tests fail, naming the
exact historical symptom (a slack task offered `reserve_hotel`).

### Not in Phase 1

Controls planting (Part 10/11), symmetric attack scoring (Part 12), the
failure taxonomy (Part 13), the evidence regime (Part 7/8), native mode
(Part 6/18), the budget sweep runner (Part 17) and the analysis exports
(Part 20/21).
