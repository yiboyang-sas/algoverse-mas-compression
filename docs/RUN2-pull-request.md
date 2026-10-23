# Proposed pull request

Target branch: `scorer-fix`
Working branch: `codex/run2-compression-no-attack`
Status: local review draft; no new push, pull-request creation, or merge is requested by this document.

## Proposed title

Prepare benign Run 2 with reproducible sweeps and optional shared handoffs

## Description

Run 2 needs a reproducible comparison of compression conditions on the team's current benign benchmark. This change prepares the frozen 18-task, three-repeat experiment with GPT-4o, temperature 0.7, delegation enabled, no attack content, incremental raw evidence, and per-attempt budget reservations. It fixes a task-configuration error that otherwise tells the allowlist compressor to remove earlier legitimate write steps.

The independent full-pipeline design remains the default. Optional shared handoffs reuse one durable Agent 1 draw per task/repeat across all six conditions, reducing upstream variation in compression comparisons. The manifest freezes the design and rejects incompatible resumes or smoke/full pairs. Shared cache links preserve the prepared handoff and account for Agent 1 requests once. Expected full request counts are 864 independent or 594 shared, before retries; the scored trial count stays 324. The shared design is proposed for team review and does not automatically replace the primary protocol.

Primary scoring remains unchanged. A separately versioned secondary evaluator uses `tasks/run2_secondary_references.json` (`run2-secondary-reference-conformity-v1`) to expose stricter call/content checks, coverage limits, and disagreement with the primary scorer. Paired task-level analysis, repeat variation, task-cluster intervals, and figures support review without treating 54 repeated observations as 54 independent tasks. Failure categories remain evidence-based hypotheses requiring semantic review.

## Benchmark scope

- Base: `scorer-fix` commit `d1512367702c44faa244b606bdd6b6f6c8a0e89a`.
- `tasks/sample_tasks.json` changes only four allowlists: `ws_13`/`ws_19` add `append_to_file`; `ws_32`/`ws_37` add `create_file`.
- The 18 IDs, documents, prompts, primary reference calls, and match modes are frozen. `environment.py` is unchanged from the base.
- Original task-file SHA256: `348b8ab77d10bb96ecc6cb4ca1b03e035edd679167d7f9b13ac028bbfb91f469`.
- Corrected task-file SHA256: `203bb3c7bc21e73c701ee06a06857aa82800a6bd6f0612b898602cdd0c5a5c8e`.
- Secondary references are separate artifacts; they do not rewrite the primary ground truth. No adversarial tasks, payloads, or compressor architecture changes are introduced.

## Validation and limitations

Offline validation covers the frozen dataset, scorer contract, delegation boundary, tracing, interrupted trials, incremental journals, source/configuration identity, shared-cache reuse and linkage, smoke gating, budget reservations including retries/concurrency, paired analysis, and empty/mock figure labels. A golden transcript test compares the independent pipeline against the frozen `e5eb587` implementation for all six conditions. The secondary suite includes primary-pass counterexamples for bulk emails, extra recipients/actions, omitted content and reversed create/share order.

Real GPT-4o smoke and full experiments have not run. Approved credentials, budget allocations, current pricing, and model access remain prerequisites. Neither mocks nor fixture results are scientific API measurements. The primary metric omits parts of semantic task utility, and stricter secondary checks still require manual coverage review. See `RUN2.md` for the execution protocol and `RUN2-design-review.md` for design tradeoffs.

## Reviewer decisions

Review the four allowlist corrections, the proposed role of shared handoffs versus independent confirmation, and the secondary reference coverage before approving a protocol change. Approve a credential source and separate smoke/full budget allocations before any paid call. No teammate work should be overwritten or merged as part of this preparation.

Offline validation: `python -m pytest -q` passed all **248 tests**. `git diff --check` is clean. The primary scorer, dataset and frozen analyzer retain their recorded SHA256 digests. The secondary evaluator also rejects inconsistent accepted-action and Agent 2 stage logs. See `RUN2-final-commands.md` for exact setup, preflight, smoke, full and supplemental-analysis commands; paid execution remains blocked pending credential and budget approval.
