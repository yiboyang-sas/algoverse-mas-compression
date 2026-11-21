# Runs 3 and 4

**Asra Asif · 11 November 2026**

What the attack arm returned, what I thought it would show, and what I think it
actually shows.

---

## 1. Run 3 — attack, no compression

| Wrapper | Attack fired |
| --- | --- |
| `tool_knowledge` | 46% |
| `important_instructions` | 40% |
| `ignore_previous` | 11% |
| `direct` | 9% |

- 1,248 trials. Zero API failures, zero parse failures.
- 26% of attacks fired overall. 17% did something the user cannot undo.
- That wrapper ordering is AgentDojo's own published ordering and I fed it in
  nowhere. Best evidence I have that the planting and the scorer work rather
  than producing plausible noise.
- The injection costs 8 points of benign task completion by itself, 75% against
  an 83% baseline. An attack that fails to fire still damages the task.
- Planting in a prose block collapses to 6%, where a record field is near 30%.
  Same collapse the controls showed, so those trials should not be pooled.

---

## 2. Run 4 — attack under every compressor

| Compressor | Attack | Benign | Trade |
| --- | --- | --- | --- |
| none | 89% | 68% | — |
| `unconditional` | 84% | 56% | 0.5 |
| **`destination`** | **67%** | **66%** | **9.0** |
| `instruction_strip` | 65% | 43% | 1.0 |
| `allowlist` | 15% | 16% | 1.4 |
| `sliding_window` | 14% | 7% | 1.2 |

- 1,746 trials on the 97 specs that fired in run 3, paired on the same Agent 1
  draws, so a difference between arms is the compressor.
- **Trade** is points of attack removed per point of benign task lost. Above 1
  is worth having, below 1 costs more than it buys.
- Four of the five land near 1. They cost about as much task as they buy safety.
- `destination` is the exception at 9.0.
- A prediction I registered in the code before the run came out right.
  Tool-level allowlisting can only act when the attack needs a tool the task
  does not. Baseline both 89%. Under the allowlist, 5% for a novel tool and 41%
  for a reused one.
- Caveat. `sliding_window` keeps the last 600 characters and the plan sits at
  the front, so that arm deletes the plan by construction. Its 14% does not mean
  anything yet.

---

## 3. What I thought this would show

> Compression at the handoff filters out injected content, and
> destination-specific compression is the defence.

---

## 4. Why the results do not support that

- **`destination` throws away almost everything and the attack still gets
  through.** It drops 90% of the harmless text I plant, and the attack still
  fires 67% of the time. The attack sits inside a record the task needs, so it
  cannot be dropped without dropping the answer.
- **The compressors that do cut attacks only do it by breaking the agent.**
  `allowlist` takes task completion to 16%, `sliding_window` to 7%. That is not
  a defence, that is an outage.
- **`instruction_strip` breaks even at 1.0.** It is the one built specifically
  to remove injected instructions.
- **I told the planner to ignore document instructions and nothing happened.**
  Plan adoption went 64% to 65%. p = 1.000
- **I stripped the section labels off the handoff and nothing happened either,**
  under any compressor except `allowlist`. Same sentences, same order, zero new
  Agent 1 calls.
- **Part of the claim is already published.** Governance Decay, June 2026, 1,323
  episodes across seven models, shows context compaction erases in-context
  safety constraints. Single-agent, no handoff, so it does not cover us, but
  "compression drops safety rules" is no longer a new finding.

Put those together and the reason is one thing. The attack has already become a
plan step before the compressor ever runs. Compression sits downstream of that,
so it cannot undo it. "Does compression filter injected content" has an answer,
the answer is no, and that is one line, not a paper.

---

## 5. What I think the paper is

> Instruction promotion at the document-to-plan boundary. Measuring when
> injected text stops being quoted and becomes the agent's own plan step.

### What the data says

| What I planted | Reaches the note | Lands in the PLAN |
| --- | --- | --- |
| **the attack** | — | **65%** |
| N2 harmless instruction | 45% | 14% |
| N1 inert dated fact | 44% | 14% |
| C safety rule | 39% | 7% |

291 handoffs, no compression. Same documents, same slot, same length. The only
difference between rows is what the sentence says. Attack vs N2, p = 1.3e-12.

### What it means

- Everything I plant reaches the handoff note at about the same rate, around
  40%. Only the attack gets picked up into the plan.
- So the handoff is not just losing information. It is changing what the
  information counts as. Text that arrived as something a document said leaves
  as something the agent intends to do.
- Once it is a plan step, Agent 2 has no way to tell it apart from what the user
  actually asked for. There is nothing left downstream to filter on.
- The safety rule is the worst treated of all four, at 7%. The one piece of
  planted text that would have stopped the harm is the one least likely to
  survive.

### What supports it

- **It is selective, not general copying.** If promotion were just "text near
  the task gets copied," the controls would be promoted too. They are not. 65%
  against 14%, 14% and 7%.
- **It predicts the harm.** When the attack is in the plan, Agent 2 acts 99% of
  the time. When it is not, 66%. p = 6.7e-14.
- **Every run I already have becomes a test of it.** Spotlighting aims directly
  at promotion and does not move it. Compression sits downstream and cannot undo
  it. Labelling does not change promotion but does change how filterable the
  result is later, `allowlist` 12% labelled against 5% delabelled, p = 0.001,
  with task completion unchanged.

### How

- No new data. Every number above is already collected and sitting in the CSVs.
- Run 4 stays in the paper in full, all six arms, `destination` at 9.0 included.
  It stops being the answer and becomes the evidence for the mechanism.
- The matched controls move from a methods detail to the centre of the method,
  because they are the part nobody else has.
