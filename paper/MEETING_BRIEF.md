# Project briefing

**For the meeting, 11 November 2026. Workshop deadline 20 November.**

---

## 1. The one thing to say if you only get thirty seconds

We set out to ask whether compressing an agent handoff filters out injected
attacks or concentrates them. The answer turned out to be neither, and the real
finding is upstream of the question we asked.

**An indirect prompt injection is not surviving our pipeline. It is being
adopted by it.** The reading agent takes the attacker's account number or file
ID out of the document and writes it into its own plan, as a line of the task it
hands downstream, on 65% of trials. Once that happens the attack fires 96% of the
time. By the time any compressor runs, the attack is no longer foreign content
in a document. It is part of the plan, and no content filter can remove it
without removing the task.

Everything else in the project is now evidence for that.

---

## 2. The setup, in plain terms

Two agents. Agent 1 reads documents and writes a handoff note. Agent 2 reads
only that note and takes the real actions, sending money, deleting files,
booking hotels. Agent 1 holds read tools only, Agent 2 holds write tools only,
and nothing passes between them except the note.

An attacker hides an instruction inside the documents Agent 1 reads.

Between the two agents we optionally put a **compressor** that shortens the note
to 150 tokens. People deploy these for cost and context reasons, not security.
Our question was what it does to the hidden instruction.

The tool split is what makes the experiment valid. Agent 1 physically cannot
perform the harmful action, so if an attack succeeds, it succeeded by travelling
through the note. We test this rather than assume it.

**Tasks:** 63 single-turn tasks across AgentDojo v1.2.1's workspace, banking,
Slack and travel suites. 51 adapted from AgentDojo, 12 banking tasks we wrote
ourselves because AgentDojo's banking handoffs come out shorter than the
compressor budget, which makes compression a no-op.

**Planted in every task's documents**, in the same kind of position, three
control passages of equal length:

| | What it is | Why |
| --- | --- | --- |
| **C** | A security rule: never forward credentials, report to this address | The thing that should stop the attack |
| **N1** | An inert dated fact about an office desk move | The floor. Content nobody is asked to act on |
| **N2** | A harmless instruction with a deadline and an address | The attack's shape, minus the harm |

N2 is the one that matters. If an attack survives no better than N2, then our
finding is only that compression keeps imperatives, which is much weaker.

---

## 3. What each run did and what it told us

### Run 1, the gate. No compression, no attack.

Can the two agents do the task at all? **81%**, with 83% of individual tool calls
correct.

Why it exists: without it, a low score under a defence cannot be distinguished
from a task nobody could do. Every later number is read against this.

### Run 2, compression with no attack. 1,134 trials.

What each compressor costs in pure task completion.

| Compressor | Task completed | Cost against baseline |
| --- | --- | --- |
| none | 81% | |
| `destination` | 72% | **-9** |
| `unconditional` | 50% | -31 |
| `instruction_strip` | 35% | -46 |
| `allowlist` | 23% | -58 |
| `sliding_window` | 10% | -71 |

**What it tells us.** Compression is expensive. Four of five compressors destroy
between a third and seven eighths of the agent's ability to do its job.
`destination`, which is told who the recipient is and what tools they hold, costs
9 points where the next cheapest costs 31. That gap is real, p = 0.0002 paired.

This run was done twice. The first version ran before we fixed where the controls
were planted. We reran it on the corrected dataset and the task numbers moved by
at most four points, which settles the question of whether the fix changed task
difficulty. It did not.

**The control result, which is the part that mattered.** Of the controls that
reached the note at all, the share surviving compression:

| Compressor | C safety rule | N1 inert fact | N2 harmless instruction |
| --- | --- | --- | --- |
| `unconditional` | 2/36 = 6% | 41/59 = 69% | 69% |
| `destination` | **0/36 = 0%** | 7/59 = 12% | 12% |
| `instruction_strip` | 2/36 = 6% | 32/59 = 54% | 54% |
| `allowlist` | 9/36 = 25% | 17/59 = 29% | 32% |
| `sliding_window` | 19/36 = 53% | 50/59 = 85% | 85% |

The safety rule is destroyed at roughly half the rate of harmless text of the
same length sitting in the same documents, and the gap is significant for three
of the five (p < 0.001). Under `destination` it is zero out of thirty-six.

**`allowlist` shows no discrimination (p = 0.81), and that confirms the
mechanism rather than undermining it.** Every other compressor filters by *is
this relevant to the task*, and a safety rule is relevant to no task. The
allowlist filters by *does this mention a forbidden tool*, which C and N1 answer
identically. The one compressor that does not reason about relevance is the one
that does not strip safety rules.

**And it starts before compression.** Agent 1 carries C into the note on 19% of
trials against 31% for the harmless controls, p = 6.6e-05 paired on the same
trials. Two independent stages discriminate against the safety rule and the
effects compound.

### Run 3, attack with no compression. 1,248 trials.

How often the attack works with no defence at all. **26% fire, 17% do something
irreversible.**

The attacks are AgentDojo's own, not ours. Each injection task there carries a
`ground_truth()` returning the harmful call with real parameter names, so we read
them from the benchmark rather than writing them. 16 of 35 are usable; 19 are
excluded with a recorded reason, 9 because their success criterion is that the
agent stopped rather than acted, 10 because they need a read tool Agent 2 does
not hold. Unreachable and defended must never share a bucket.

| Wrapper | Attack fired |
| --- | --- |
| `tool_knowledge` | 46% |
| `important_instructions` | 40% |
| `ignore_previous` | 11% |
| `direct` | 9% |

**This ordering is AgentDojo's own published ordering and we fed it in nowhere.**
It is the strongest evidence in the project that our planting and our scorer
behave like a known benchmark rather than producing plausible-looking noise. Four
wrappers all landing near 25% would have pointed at a broken scorer.

Two side findings. The injection costs eight points of benign completion on its
own, 75% against the 83% per-call baseline, before any defence exists. An attack
that *fails* to fire still damages the task. And planting in a prose block
collapses to 6% where planting inside a record field is near 30%, the same
collapse the controls show.

### Run 4, attack and compression. 1,746 trials.

The actual question. Run on the 97 attack specifications that fired at least once
in run 3, because pushing the 290 that never fired through six conditions would
spend two thirds of the run proving a defence stopped something that was never
going to happen, while dragging every arm toward zero and flattening the
differences between them.

| Compressor | Attack fires | Task done | Attack removed | Task lost | **Trade** |
| --- | --- | --- | --- | --- | --- |
| none | 89% | 68% | | | |
| `unconditional` | 84% | 56% | -5 | -12 | **0.5** |
| `destination` | 67% | 66% | -22 | -2 | **9.0** |
| `instruction_strip` | 65% | 43% | -24 | -25 | **1.0** |
| `allowlist` | 15% | 16% | -74 | -52 | **1.4** |
| `sliding_window` | 14% | 7% | -75 | -62 | **1.2** |

**Trade** is points of attack removed per point of task lost. Above 1 is a
defence worth having.

**What it tells us.** Four of five compressors buy their attack reduction at
roughly the price of the task. `unconditional` is strictly bad: twelve points of
completion spent to remove five points of attack. `instruction_strip`, whose
entire stated job is rewriting instructions as statements so they stop being
commands, breaks even at 1.0.

`destination` is the exception at 9.0, and it is the direction this team proposed
at the start.

**One prediction we registered in writing before the run, and it came back.** We
wrote into the code, with a test, that tool-level allowlisting can only act on an
attack needing a tool the benign task does not. If the harmful tool is already on
the allowlist because the task needs it, the instruction to strip content
implying other tools cannot fire.

| Attack needs | No compression | Under `allowlist` |
| --- | --- | --- |
| A tool the task does not need | 89% | **5%** |
| A tool the task already needs | 89% | **41%** |

Identical at baseline, eightfold apart under the defence. Named in advance, could
have come back flat, did not.

---

## 4. What Yibo did, and why it changed the paper

### The problem he was given

`sliding_window` keeps the last 600 characters of the note. But Agent 1 writes
its PLAN at the front and its EVIDENCE after, so that compressor was landing
entirely on the evidence and cutting the plan off every time. Agent 2 was getting
a pile of records with no instructions. Its 14% attack rate and 7% task
completion looked like a brilliant defence and were really just an agent that had
never been told what to do. The column was uninterpretable.

### What he did

Added a head-keeping version as its own condition rather than replacing the tail
one, so both survive and we can show that position matters, then ran it on the
run 4 specifications beside a fresh `none` arm. 291 trials per arm, no API
failures.

### What came back

| | none | head-keeping |
| --- | --- | --- |
| Attack fires | 87% | 85% |
| Task completed | 70% | 65% |

Head keeping removed 6 attacks and added 1, McNemar p = 0.13, so **no real
difference from having no compressor at all**. It also lost 15 task completions
and gained none, p = 0.0001.

So `sliding_window` is not a defence. The old 14% was purely the artefact of
cutting the plan, which the tail did on 83% of handoffs. We report head keeping
as the structural baseline, and it is a worse trade than no compression.

I verified all of this against his committed data independently. Every number
checks out, and his `none` arm cross-checks against run 4's at 87/70 versus
89/68, which is the right amount of drift for a redrawn Agent 1.

### The thing he found that nobody asked for

He instrumented where the plan sits and whether the attack's own values appear in
it. **In 65% of handoffs, Agent 1 had already written the attacker's recipient,
IBAN or file ID into its PLAN.**

I pushed on it further:

| | Attack fires |
| --- | --- |
| Attack's values are in the PLAN (188 trials) | **96%** |
| They are not (103 trials) | 70% |

Fisher p = 4.9e-10.

And I then measured whether the controls ever reach the plan, which completes it:

| | Reaches the note | Reaches the PLAN |
| --- | --- | --- |
| **The attack** | | **65%** |
| N2 harmless instruction | 45% | 14% |
| N1 inert fact | 44% | 14% |
| C safety rule | 39% | **7%** |

Attack versus N2 in the plan, p = 1.3e-12.

**This is not "the agent copies imperatives".** N2 is an instruction of the same
shape, the same length, planted in the same kind of position by the same code,
and it reaches the plan four and a half times less often.

**Why the attack specifically?** Because it names a tool the downstream agent
actually holds and supplies concrete argument values for it. That is exactly the
artefact Agent 1 was asked to produce. A harmless instruction asking someone to
confirm a desk preference by email names no tool and carries no arguments, so
there is nothing for Agent 1 to turn into a plan line. **An injection that speaks
the pipeline's own interface language gets promoted to first-class task content.**

---

## 5. What we were originally doing, and what we concluded

**Originally.** Does compressing a handoff filter out an injected attack, or
concentrate it into a smaller note where the acting agent is more likely to obey?
Two outcomes, both interesting, both about compression.

**What we concluded.** Neither. The question had a hidden assumption, that at
compression time the attack is still foreign content sitting in a document. It
usually is not. Agent 1 has already laundered it into its own plan.

That explains everything we measured:

- No compressor filters it, because filtering it means filtering the plan, and
  the plan is the task. That is `destination` exactly: it drops 88% of the
  harmless planted text and still keeps 67% of attacks.
- The safety rule dies for the mirror reason. It is a constraint, not an action,
  so it never becomes a plan line, 7% against the attack's 65%. The pipeline
  promotes the attack and discards the rule that would have stopped it, at two
  independent stages.
- `instruction_strip` breaking even is no longer surprising. Rewriting commands as
  statements cannot help when the command has already been restated by the agent
  as its own plan.

**What this costs us.** It is a pivot and it should be decided deliberately.
Compression stops being the subject and becomes the thing that cannot fix the
problem. The six-compressor table demotes from the result to the evidence that no
content filter at this position works.

I think that is a better paper. "We tested six compressors and they are bad
trades" is a benchmark result. "Injections that speak the interface get laundered
into the agent's own plan before any defence can see them, and here is the
measurement" is a mechanism, and it says something true about systems we did not
test.

Working title: *Laundered instructions: indirect prompt injections are adopted
into agent plans before any defence can see them.*

---

## 6. Who was given what

| Person | Task | Status |
| --- | --- | --- |
| **Yibo** | Fix `sliding_window` head keeping, rerun that arm | **Done.** Plus the plan-adoption instrumentation nobody asked for, which became the paper's lead |
| **Yibo** | Paired confidence intervals on the trade column | Outstanding. Lower priority now, I have exact p-values on the three main claims |
| **Tanmay** | Failure taxonomy over run 4, per arm | Outstanding. Yibo's result partly answers it: `sliding_window` is explained, so `allowlist` is where the value is now |
| **Sourayan** | Reading list, 10 to 15 papers in three buckets | Outstanding |
| **Asra** | Methods section | Drafted, needs rewriting around the new lead |
| **Asra** | Text-level attack survival measurement | Outstanding, offline, no API calls |

**What Yibo's reply means in practice.** He did the job and then did more than the
job, and the extra is what reframed the paper. The instrumentation he added,
whether the plan fits in the head, whether it survives in the tail, whether the
attack's values are in the plan, is now carrying the central result. His
suggestion that Tanmay move to `allowlist` is right and we should take it.

**Nobody else has replied yet.** With ten days left that is the thing to raise in
the meeting, not in a channel message.

---

## 7. Where we are, concretely

Four runs complete, plus the fix. About 4,400 paid trials. Zero API failures
across all of them. 430 tests passing. Everything committed with manifests
recording model, temperature, budget, prompt hashes and dataset hashes, so every
number can be traced back.

**Experiments are finished.** Nothing is blocking on a run any more.

**What is left is writing**, and that is where we are weakest, because three of
the five sections have no owner.

---

## 8. The ask for the mentor

Six things, in the order I would raise them.

**1. Do we take the pivot?** The laundering finding is stronger than the
compression finding and I want agreement before I rewrite the abstract around it,
not after.

**2. Which venue, or both?** Both LAUT and SPF-LLM are AAAI-27 workshops, both due
20 November, both non-archival, both single-blind on OpenReview, both wanting the
AAAI template. Six pages fits both. Does submitting the same work to two workshops
at the same conference cause a problem, and if so which one do we pick?

**3. Related work has no owner and nobody has read anything for it.** That is the
single biggest risk to making the deadline and it is not a thing I can fix by
messaging people again.

**4. Do we lead with all 63 tasks or the 51 that never fail at baseline?** Needs
deciding once and stating once. My view is lead with 63, report 51 as a
robustness check, because dropping 12 invites the question of whether we dropped
the hard ones.

**5. Author list and order.** Easier now than on the 19th.

**6. Is one model enough?** We ran everything on gpt-4o-mini. A reviewer will ask.
A reduced second-model arm is about 291 trials and I could do it, but only if the
writing is covered, and right now it isn't.

### One discretionary experiment I would still argue for

`destination` is our positive result and its prompt hands the compressor two
separate things, the recipient's role and the recipient's tool list. We cannot
currently say which does the work. Splitting them is 582 trials and about an hour,
and it turns "this prompt works" into "here is why", which is the difference
between a paper people cite and one they skim. Only worth doing if someone else
picks up a writing section.
