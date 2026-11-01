# Shared-handoff primary design: compatibility and review report

This revision incorporates Asra's feedback into PR #5. The primary comparison now generates **one Agent 1 draw per task/repeat and reuses its exact raw reply and handoff across all six conditions**. Historical independent sampling remains available through the team's original runner and an explicit legacy analysis option. No paid experiments were run; every supplied validation result is **mock, non-empirical**.

The comparison is against PR [4](https://github.com/yiboyang-sas/algoverse-mas-compression/pull/4) at `b6c5eb66d3380aa5bd1613fc5caff0673bd151a8` and the team's live **`compression-redesign`** branch at `52958c146d849aa60f50d9c3a78d4e54ecd6f474` (hyphenated remote name). This revision supersedes PR #5's earlier independent-only analysis and removes its duplicate field-retention implementation and tests.

## What is preserved and what is added

| Area | Team implementation preserved | Missing functionality added here |
|---|---|---|
| Benchmark | `tasks/sample_tasks_63.json`: 18 workspace, 15 Slack, 24 banking, 6 travel | Freeze the canonical file hash, selected IDs and repeats in a shared manifest; no task edits |
| Inference | GPT-4o-mini, temperature 0, PLAN handoff, nominal 150-token budget, six native conditions | Explicit primary defaults in `shared_run.py`; override unrelated sweep defaults with these explicit primary settings |
| Primary scoring | `baseline.score_checkable`, write denominator, distinct call consumption, critical aliases and prose exclusions | Read saved scores; never rescore, repair ground truth, or remove schema-warning rows |
| Logging | Existing `instrument.py` / `tracing.py`, verbatim `calls.jsonl`, native metrics | One upstream trace stream plus immutable handoff IDs/hashes linked from each arm; no second model wrapper |
| Retention and mismatches | Existing document → handoff → compressed C/N1/N2 controls and `failure_stage.check` | Carry their already-computed records into shared arm outcomes and analysis annotations; no new presence detector |
| Execution | Original independent runner remains available | Minimal optional prepared-Agent-1 hook in `pipeline.py` / `baseline.py`, native parse/replay/scoring in every arm |
| Failure interpretation | Existing PLAN/schema warning | Exclusive observable profiles for malformed/empty replies, missing actions, downstream schema/access problems and saved argument misses |
| Statistics | PR #4 task-cluster bootstrap mathematics | Validate identical upstream draws, report all 15 pairwise contrasts, discordance and per-task patterns; five none-reference contrasts marked primary |

The canonical dataset SHA256 is `cc9aed2091594034d68c7205225b810179139f1993cde9a11df4f241f22b51a9`. It is not the old 18-task benchmark, the 51-task file, or `baseline.py`'s legacy 14-task default. `ws_32` and `ws_37` are absent from the 63-task dataset because their sequential file-ID dependency is incompatible with this one-turn setup; this contribution does not reintroduce them.

The nominal budget preserves upstream semantics: prompted compressors request approximately 111 words, while `sliding_window` retains 600 characters. It is not a hard 150-token output ceiling. Report realized lengths and ratios using native metrics; do not reinterpret long outputs as a different configuration. The upstream prompts, compression implementations, control planting and benchmark/scoring functions are unchanged.

## Shared execution and reproducibility

`shared_run.py` defaults to mock execution, three repeats and all six conditions: `none`, `sliding_window`, `unconditional`, `destination`, `instruction_strip`, `allowlist`. It freezes the dataset, task selection, settings, source/prompt fingerprints and backend identity before execution. Each upstream draw uses the team's native Agent 1 prompt, parser and simulated read replay. The immutable cache preserves the raw reply, parsed value, input, read log, handoff, hashes and stable handoff ID before any downstream condition runs.

A cached draw is parsed and replayed through the native pipeline in every arm without calling Agent 1 again. This preserves parser evidence and task-local simulated tool state. The upstream model call is logged only once in `shared_agent1/calls.jsonl`; arm streams contain only their new compressor/Agent 2 calls. Native per-arm token metrics still describe each arm's full context; actual shared model-call totals must count upstream calls once, as the analysis does. Local token counts are not provider-billed usage.

Resume validates manifest/cache integrity and reuses completed or failed upstream draws. Empty, malformed or failed draws are not selectively regenerated. A native call without a durable outcome/cache is ambiguous; the runner stops rather than silently redrawing or double-spending. Such interruptions require human review. The experiment uses a local exclusive writer lock; it does not support multiple processes writing the same root.

The analyzer requires `--shared-run` for primary analysis. It checks model, temperature, budget, task snapshot, native stage boundaries, cache identity and both raw/handoff hashes. It rejects contradictory or duplicated evidence. Every matched comparison uses the same task, repeat and cached handoff. Missing/error slots are reported as coverage gaps, never filled with zero or silently discarded from the planned denominator. Available scored outcomes form the success estimand; transport/pipeline errors remain visible separately. Missingness can still bias that estimand.

## Failure taxonomy without duplicate instrumentation

`failure_taxonomy.py` assigns **one category per observed outcome**, with explicit precedence:

1. Native pipeline/transport error.
2. Saved primary success.
3. Unusable Agent 1 reply or empty upstream handoff.
4. Empty compressed output.
5. Unparseable downstream reply.
6. Downstream schema/access error (for example unknown tool, tool not held, malformed arguments).
7. Downstream reply with no attempted action.
8. Too few accepted calls of a required tool.
9. Saved argument misses despite sufficient same-named accepted calls.
10. Unclassified or missing evidence.

Co-occurring warnings are annotations, not additional failed trials. Raw attempted actions are distinguished from accepted tool-log actions. The argument-miss profile uses the **saved scorer miss and action counts**; it does not implement another argument matcher. Extra action counts, attempted Agent 1 writes, native parse events, native control retention and native tool mismatches remain review evidence even for primary successes.

Agent 2 receives only the treated handoff as its user input. PLAN action descriptions and EVIDENCE pass through the same compression boundary; the cache does not add a privileged uncompressed side channel. Native simulated read replay is retained for scoring consistency, not exposed as extra Agent 2 context.

These profiles are observations, not causal diagnoses. The existing three-stage control measurements distinguish an observed upstream marker omission from a subsequent boundary loss. They measure C/N1/N2 controls, not every legitimate task fact. A wrong PLAN tool does not prove that a required value was lost: EVIDENCE can still contain it and Agent 2 can recover it. Neither marker loss nor schema warnings automatically blame the compressor or exclude a task. Preserved text plus a wrong action does not establish instruction misunderstanding. Those stronger claims need manual review of the native verbatim traces; unsupported cases remain unresolved. No replacement critical-information detector is introduced.

Eight upstream definitions have required earlier writes absent from the compressor's `allowed_tools` list while present in Agent 2's held tools: `ws_13`, `ws_19`, `sl_11`, `sl_16`, `sl_18`, `sl_19`, `sl_20`, `bk_15`. That mismatch can affect destination/allowlist prompts. It is documented, not repaired or used to change the cohort. The original team tool-mismatch and control checks are reused unchanged.

## Paired estimates and scientific interpretation

For each common scored task/repeat/handoff, the analyzer reports treatment minus reference success in {-1, 0, +1}, including concordant failures/successes and discordant improvements (`01`) / regressions (`10`). It averages paired repeat differences within each task, then weights tasks equally. Percentile 95% intervals resample **whole tasks with all their repeats**, using a recorded seed; fewer than two task clusters have no interval. The bootstrap mathematics from PR #4 is preserved.

All 15 pairwise contrasts are exported. The five contrasts against `none` are marked primary; comparisons among compressors are descriptive. Per-task outputs retain repeat outcomes, failed repeat indices, discordance, and improved-only/regressed-only/mixed/concordant patterns. Estimates, coverage, configurations and input/code hashes are saved in `analysis.json`; unaltered scores/metrics in `trials.csv`; exclusive profiles and native annotations in `diagnostics.jsonl`; a concise table in `report.md`.

**Recommendation: shared handoffs are the primary design**, as requested by Asra. They control Agent 1 variability even at temperature 0, making within-handoff compression comparisons interpretable. Independent end-to-end sampling answers a different question about the whole pipeline including new upstream draws; it remains a secondary sensitivity analysis and is explicitly labelled `legacy-independent`. Matching its repeat labels alone cannot isolate compression. The fixed native condition order completes the baseline first; shared handoffs do not control downstream time/order effects or provider drift. Neither design guarantees deterministic model inference. Seeds are recorded only where supported, not claimed as determinism.

The 63 benchmark items are a fixed, heterogeneous sample. Repeats are not independent tasks, uniform mock outcomes can yield degenerate intervals, all contrasts are exploratory without multiplicity-adjusted significance claims, and post-treatment retention strata should not be used as a newly selected primary cohort. Shared handoffs isolate upstream sampling, not semantic mediation. No compression benefit/harm or paper result is established by offline fixtures.

## Exact offline commands

Use Python 3.10 or newer from the repository root. No API credential is needed or used by these commands.

```sh
PYTHONDONTWRITEBYTECODE=1 MAS_BACKEND=mock python3 -S run_tests.py

# Small six-condition smoke, three shared draws (one task x three repeats).
PYTHONDONTWRITEBYTECODE=1 python3 -S shared_run.py \
  --out-dir /tmp/algoverse-shared-smoke --limit 1
python3 -S analyze_redesign.py \
  --shared-run /tmp/algoverse-shared-smoke \
  --tasks tasks/sample_tasks_63.json --out-dir /tmp/algoverse-shared-smoke-analysis \
  --bootstrap-draws 2000 --seed 1729

# Full offline bookkeeping validation: 189 A1 draws and 1,134 arm outcomes.
PYTHONDONTWRITEBYTECODE=1 python3 -S shared_run.py \
  --out-dir /tmp/algoverse-shared-mock --repeats 3
python3 -S analyze_redesign.py \
  --shared-run /tmp/algoverse-shared-mock \
  --tasks tasks/sample_tasks_63.json --out-dir /tmp/algoverse-shared-mock-analysis \
  --bootstrap-draws 2000 --seed 1729

# Historical independent sampling: explicit non-primary opt-in.
python3 -S analyze_redesign.py --legacy-independent \
  PATH_TO_NONE_RUN PATH_TO_COMPRESSION_RUN \
  --tasks tasks/sample_tasks_63.json --out-dir /tmp/algoverse-legacy-analysis
```

Shared mock manifests automatically label analysis as non-empirical. Use `--fixture` for authored historical fixtures; legacy metadata alone cannot verify a real backend. Re-running a shared runner command resumes the same experiment after integrity checks; analysis output directories must be fresh and empty.

Real execution remains blocked on approved credentials, spending authorization and review of the revised design. The explicit `--execute-paid` path is not authorization and was not invoked. Production requires the team's measured tokenizer and a normal secure environment-variable credential source; no secret is committed. This PR is for review, not a merge or an empirical paper claim.

## Offline validation record

The final offline suite ran 306 tests: **305 passed, one skipped** because the optional measured tokenizer was unavailable. Tests cover native concurrency/scoring behavior, immutable reuse, empty/error draws, resume integrity, writer exclusion, stage boundaries, mutually exclusive taxonomy, same-handoff provenance, discordance, missing-pair coverage and task-cluster intervals. Authored fixtures include a known discordant outcome and an upstream timeout; neither is empirical data. Full mock and reproducibility artifacts are supplied separately from Git source to avoid committing generated traces.
