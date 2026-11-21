# Runs 3 and 4: the attack arm

**8 November 2026 · Asra Asif · workshop deadline 20 November**

Most handoff compression is a bad security trade: it removes injected attacks
at roughly the price of the task itself. Destination-specific compression is
the exception, removing 22 points of attack success for 2 points of task
completion. The reason is that a compressor keeps what the recipient needs,
the attack has made itself part of that, and the safety rule that would have
stopped it has not.

---

## Method

The attacks are AgentDojo's, not ours. Every suite ships injection tasks, and
each one has a `ground_truth()` that returns the harmful call with the
benchmark's own parameter names, so the specs are read rather than written.
The four attack wrappers are copied from the package verbatim, including a
typo in `ignore_previous`, because fixing it would make our arm a different
attack from the published one.

Nineteen of AgentDojo's 35 injections are excluded, and each records why.
Unreachable and defended must never share a bucket, or every defence reads
better than it is.

| Injections | Count | Why |
| --- | --- | --- |
| Usable | 16 | Harmful call reachable with the tools Agent 2 holds |
| Empty ground truth | 9 | Denial-of-service goals whose check is that the agent stopped; our scorer reads zero whatever the defence does |
| Needs a read tool | 10 | Opens with `search_emails` or `get_hotels_prices`, which Agent 2 does not hold |

The injection is planted inside one record's free-text field, the same place
the N2 control goes. N2 only works as a baseline for an attack if the two sit
in the same kind of position. Most AgentDojo goals contain an apostrophe,
which closes a single-quoted record field early, so the quote is escaped the
way the environment's own repr would have escaped it.

Scoring changed. The previous code matched one harmful call against the full
argument dict, which scored a successful attack as a defended one in three
ways: two injections need three calls, matching full arguments made the
placeholder subject lines load-bearing, and one Slack goal is about getting a
link into a message body that a model words its own way. Matching is now on
identifiers only, by substring, and consumes, so three transfers need three
distinct calls.

Run 4 is paired. All six arms reuse the same Agent 1 draw per spec and repeat,
so a difference between arms is the compressor and not Agent 1 having a
different day.

---

## Run 3: attack, no compression

With no defence at all, 26% of attacks fire and 17% make a change the user
cannot undo. 1,248 trials over 416 task-injection-wrapper combinations, three
repeats, gpt-4o-mini, zero API failures and zero parse failures.

| Wrapper | Attack fired | What it says |
| --- | --- | --- |
| `tool_knowledge` | 46% | Message from the user, plus the tools to use |
| `important_instructions` | 40% | Message from the user, inside an `<INFORMATION>` block |
| `ignore_previous` | 11% | "Ignore your previous instructions" |
| `direct` | 9% | "TODO:" and the goal |

That ordering is AgentDojo's own published ordering and we fed it in nowhere.
It is the strongest evidence in the project that the planting and the scorer
behave rather than produce plausible numbers. Four wrappers all landing near
25% would have pointed at the scorer.

Two results here are not about compression at all.

The injection costs eight points of benign task completion by itself: 75%
against the 83% baseline, before any defence exists. An attack that fails to
fire still damages the task.

Planting in a prose block collapses to 6% where planting in a record field is
near 30%. The controls showed the same collapse, so planted content in a prose
block is weak generally, and those trials should not be pooled with the rest.

No attack fired only partly, and that is behaviour rather than a blind spot in
the scorer. The two multi-call injections are all-or-nothing by construction,
and a model that follows either emits every call in one response. 36 of 168
multi-call trials fired, all of them completely.

---

## Run 4: attack under every compressor

Four of the five compressors buy their attack reduction at roughly the price
of the task. 1,746 trials on the 97 specs that fired in run 3, paired on the
same Agent 1 draws, zero API failures.

| Compressor | Attack fired | Benign done | Attack removed | Task lost | Trade |
| --- | --- | --- | --- | --- | --- |
| none (baseline) | 89% | 68% | | | |
| `unconditional` | 84% | 56% | -5 | -12 | **0.5** |
| `destination` | 67% | 66% | -22 | -2 | **9.0** |
| `instruction_strip` | 65% | 43% | -24 | -25 | **1.0** |
| `allowlist` | 15% | 16% | -74 | -52 | **1.4** |
| `sliding_window` | 14% | 7% | -75 | -62 | **1.2** |

`Trade` is points of attack removed per point of benign task lost. Above 1 is
a defence worth having; below 1 costs more than it buys.

`unconditional` is strictly bad: twelve points of completion spent to remove
five points of attack. `instruction_strip` breaks even at 1.0, which for a
defence whose stated purpose is removing injected instructions is a result in
itself. `allowlist` and `sliding_window` do cut attacks hard, to 15% and 14%,
but they take task completion down to 16% and 7% doing it.

`destination` is the exception, and it is the direction the team proposed at
the start.

---

## Why it happens

A compressor keeps what the recipient needs. The safety rule serves no task,
so it goes first, in every arm.

Share of each planted control that survived compression, counting only the
controls that reached Agent 1's note at all:

| Compressor | C safety rule | N1 inert fact | N2 harmless instruction |
| --- | --- | --- | --- |
| `unconditional` | 19% | 62% | 58% |
| `destination` | 3% | 10% | 10% |
| `instruction_strip` | 16% | 51% | 50% |
| `allowlist` | 26% | 44% | 43% |
| `sliding_window` | 62% | 83% | 84% |
| **mean** | **25%** | **50%** | **49%** |

C, N1 and N2 are the same length and sit in the same documents, and only the
safety rule is systematically dropped. N1 and N2 track each other within a
point or two everywhere, which kills a claim we expected to make: being
phrased as an instruction does not help content survive compression.

What does predict survival is position. `destination` drops the planted
harmless content hardest of any arm, to 10%, because it keeps only what the
named recipient needs. The attack still fires 67% of the time under it. The
injection sits inside a record the task requires, so it cannot be dropped
without dropping the answer.

That is the paper's mechanism in one line. Compression does not distinguish
attacks from content. It distinguishes needed from unneeded, the attack has
made itself needed, and the safety rule never was.

---

## The prediction we registered

Before the run we wrote into `attacks.py` that tool-level allowlisting can only
act on an attack needing a tool the benign task does not. If the harmful tool
is already on the allowlist because the task needs it, the instruction to strip
content implying tools outside the list cannot fire. A test asserts both cases
exist in the data.

| Attack needs | No compression | Under `allowlist` |
| --- | --- | --- |
| A tool the task does not need | 89% | 5% |
| A tool the task already needs | 89% | 41% |

Identical at baseline, eightfold apart under the defence. This is the cleanest
result in the set, because the split was named in writing before any trial ran
and could have come back flat.

It also bounds what tool-level filtering can do. Of the 416 combinations in
run 3, 164 reuse a tool the task itself needs, and on those the allowlist has
no name to filter on.

---

## What not to quote yet

`sliding_window` is not interpretable. It keeps the last 600 characters of the
note and our plan sits at the front, so that arm deletes the plan by
construction. Its 14% attack rate and its unusually high control retention,
62% and 83% and 84%, are the same artefact seen from both ends: it copies a
verbatim tail instead of paraphrasing. A head-keeping arm is owed before this
column means anything.

Run 4's rates are conditional. It runs only the 97 specs that fired in run 3,
so its baseline is 89% and not run 3's 26%, and the two are not comparable.
Any sentence pairing a run 4 number with a run 3 number has to say which
denominator it uses. The selection was made on the control arm, before any
treatment, and applied identically to every arm, which is the standard way to
measure a defence and leaves the comparison between arms unbiased.

Suites are uneven. At baseline workspace runs at 95% and Slack at 72%, and
workspace is 153 of the 291 trials, so a pooled rate is weighted toward it.

One model, gpt-4o-mini, at one budget, 150 tokens.

Attack survival and control survival are measured differently. An attack counts
as surviving when Agent 2 acted; a control counts when its facts appear in the
compressed text. So the attack rate and the N2 rate cannot be set side by side
yet. Measuring the injection textually costs no API calls, since the compressed
text is already in the traces, and would let us make the stronger claim.

---

## Where the data is

| What | Where |
| --- | --- |
| Run 3 rows, every factor as a column | `results/run3_analysis/trials.csv`, 1,248 rows |
| Run 4 rows, plus all three controls at both stages | `results/run4_analysis/trials.csv`, 1,746 rows |
| Raw traces, every prompt and response | `results/run3_traces.tar.gz`, `results/run4_traces.tar.gz` |
| Attack specs as run | `tasks/attack_specs.json`, `tasks/attack_specs_run4.json` |
| Reprint either table for free | `python3 attack_run.py --out-dir results/run4 --report-only` |

---

## Who does what before 20 November

| Owner | Task | Why it blocks |
| --- | --- | --- |
| Yibo | Head-keeping `sliding_window` arm, one line in `compressors.py`, rerun that arm only, 291 trials | That column is uninterpretable until it exists |
| Yibo | Paired confidence intervals over `run4_analysis/trials.csv` with `analysis_stats.py`, on the trade column | Arms share Agent 1, so paired tests are available and much stronger |
| Tanmay | Failure taxonomy over run 4, per arm | Need to know why `allowlist` and `sliding_window` destroy the task, not assume it |
| Sourayan | Related work | Biggest unowned block; destination-specific compression is now the positive result |
| Asra | Methods section, text-level attack measurement | Lets attack survival and N2 survival be compared on one footing |
| Asra | Decide: lead with 63 tasks or the 51 reliable ones | Has to be decided once, not per table |
| Asra | Author list | Gets harder the longer it waits |
