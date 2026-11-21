# Tomorrow's meeting

**The one rule.** Do not open by saying the framing changed. Lead with the
number, say nothing about titles, and let somebody in the room tell you what it
means. By the time they get there the reframe is theirs and all you do is agree.

Roughly 20 minutes of you, then questions.

---

## 0. Open, 30 seconds

"Runs 3 and 4 are done, plus three follow-ups nobody has seen yet. I want to
show you one number first and have you tell me what you think it means. Then
I'll fill in how we got there."

Don't preview the conclusion.

---

## 1. The number. Show it, then stop talking

```
                          reaches the      appears in
                          handoff note     Agent 1's PLAN
the attack                      -               65%
N2  harmless instruction       45%              14%
N1  inert dated fact           44%              14%
C   safety rule                39%               7%
```

291 handoffs, no compression. Same documents, same slot, same length. The only
thing that differs between rows is what the planted sentence says.

Then one sentence, and stop.

"Everything we plant reaches the note at about the same rate, around 40%. Only
the attack gets picked up into Agent 1's own plan. Four and a half times the
next closest thing."

Let someone react. Somebody will say "so the agent is adopting it as its own
step." That's the paper. Agree and move on.

If nobody speaks in twenty seconds, prompt them. "Why would the attack be the
one thing that gets promoted?"

**The answer if you have to give it.** Agent 1 writes a plan of what Agent 2
should do. The attack is the only planted sentence phrased as a task that fits
that plan. The safety rule is a restriction, the fact is a fact, the harmless
order is an order about nothing the task needs. The attack is an order about
exactly what Agent 1 is already working on, so it slots in. And once it's in
the plan, Agent 2 cannot tell it apart from what the user actually asked for.

**The analogy if people look lost.** A junior staffer is reading a customer
file. Someone has stuck a note on page 4 saying "also wire 5,000 to this
account." A careful staffer flags it upward as a weird note in the file. Ours
copies it into their own to-do list instead. Now it sits next to the real
tasks, in the same handwriting, and whoever gets that list cannot tell which
items came from the manager and which came off a sticky note. The note stopped
being a note and became a job. That's the whole paper.

**Second number**, once they're with you.

```
attack is in the PLAN       Agent 2 acts   99%
attack is not in the PLAN                  66%
                            Fisher p = 6.7e-14
```

So promotion isn't a curiosity about the text. It's what predicts whether the
harm actually happens.

---

## 2. Backfill runs 3 and 4, three minutes

They don't need the method. They need to believe the numbers.

"Run 3 is the attack with no compression at all. 1,248 trials. 26% of attacks
fire, 17% do something the user can't undo. The four wrappers came out in this
order."

```
tool_knowledge           46%
important_instructions   40%
ignore_previous          11%
direct                    9%
```

"That's AgentDojo's own published ordering and we fed it in nowhere. Best
evidence we have that the planting and the scorer work rather than producing
plausible noise."

"Run 4 is the same attacks under all six compressors. 1,746 trials, paired on
the same Agent 1 draws, so a difference between arms is the compressor."

```
compressor         attack  benign   trade
none                  89%     68%
unconditional         84%     56%     0.5
destination           67%     66%     9.0
instruction_strip     65%     43%     1.0
allowlist             15%     16%     1.4
sliding_window        14%      7%     1.2
```

"Trade is points of attack removed per point of benign task lost. Four of the
five cost about as much task as they buy safety. destination is the exception,
and that was Sourayan's direction."

Stop there. Caveats only if asked.

---

## 3. Why the original title doesn't work, four minutes

Two reasons, and both are facts rather than calls you made, which is what makes
this part easy to say.

**One, we got scooped on half of it.** Governance Decay, June 2026, 1,323
episodes across seven models. Context compaction erases in-context safety
constraints. Constraint survives, violations 0%. Constraint dropped, 38%. That
is most of "compression drops safety rules," already published. It's
single-agent and doesn't study a handoff, so we're not dead, but we can't claim
it as ours.

Three more landed near us. Relinking attacks the compression boundary directly
and uses AgentDojo. CompressionAttack manipulates which tokens survive.
PlanFlip injects into a planner and measures the cascade. None do what our
table does, but together they make "compression and injection" a crowded room.

**Two, and this is the one that actually killed it.** We ran two follow-ups
looking for the defence.

Spotlighting at the planner. We told Agent 1 explicitly to treat document text
as data and not as instructions. Plan adoption went 64% to 65%. Fisher p =
1.000. Nothing at all.

Delabelling. We took run 4's saved Agent 1 outputs and stripped the PLAN and
EVIDENCE headings, so the same sentences in the same order became one block.
Zero Agent 1 calls, so the upstream is byte-identical. Attack rate didn't move
under any compressor except allowlist.

Put those beside table one and the conclusion is forced. The attack becomes a
plan step before the compressor ever runs, so no compressor can be the fix, and
the one intervention aimed straight at the promotion step does nothing. "Can
compression defend the handoff" has an answer and the answer is no. That's a
result, not a paper.

The paper is the thing that's actually happening, which is table one.

---

## 4. What the paper is now, three minutes

**Instruction promotion at the document-to-plan boundary.** Three parts, one
for each thing we already have in hand.

1. **We isolate where it happens.** The attacker's identifiers land inside the
   PLAN section 65% of the time.
2. **We show it's selective, with matched controls nobody else has run.**
   Harmless instruction 14%, inert fact 14%, safety rule 7%. Same length, same
   documents, same slot. p = 1.3e-12 against N2.
3. **We show it predicts the harm.** In the plan, 99% execution. Not in the
   plan, 66%. p = 6.7e-14.

Then everything we built becomes an attempt to move that promotion rate.

- Spotlighting aims directly at it and fails, p = 1.000.
- Compression sits downstream of it so it can't undo it. That's why destination
  strips 90% of the planted harmless text and the attack still fires 67%.
- Section labelling doesn't change promotion but does change how filterable the
  result is later. Under allowlist, labelled 12%, delabelled 5%, p = 0.001,
  with task completion unchanged. The structure that makes a handoff readable
  is the structure that makes injected content harder to catch downstream.

So the shift is not "throw out the compression work." It's "compression stops
being the question and becomes the strongest evidence for the answer."

**Why this is a better paper to submit.** The old claim was partly an absence
claim and partly already published. This one is a positive measurement with a
registered contrast and matched controls, and nobody else has the controls. A
reviewer who dislikes our conclusion still has to deal with the table.

---

## 5. Tasks, five minutes

Say clearly what moved and what didn't.

**Yibo.** Unchanged and now more central. Paired confidence intervals on 65% vs
14% and on 99% vs 66%. These are the headline numbers so they need proper
intervals, not bare Fisher p-values. Head-keeping sliding_window still owed.

**Tanmay.** Unchanged. Failure taxonomy over run 4, focused on allowlist. 21%
of its trials make zero tool calls, and of the 230 that do act only 20%
complete the task.

**Sourayan.** Changed, and tell him before the meeting if you can. The
compression reading still goes in, it just moves later. New job is positioning
against the four papers, which is now the hardest unowned section in the paper.

**You.** Rewrite methods around the control design, since matched controls go
from a methodological footnote to the core of the method. Plus the related-work
table.

**Settle today, not on the 19th.** Author list. 63 tasks or the 51 that never
fail at baseline. Whether we also target STALA at NDSS on 11 December, which is
archival and buys nine extra days.

---

## If someone asks

**"So is all the compression work wasted?"**
No. Run 4 is a full section, all six arms reported, destination at 9.0 still
stands. It moved from being the answer to being the evidence, and the evidence
slot is the harder one to argue with.

**"Why are we hearing this the day before?"**
Because the two runs that settled it finished this week. The spotlighting
result is what closed it and that came back p = 1.000 a few days ago.

**"Can we still make the 20th?"**
Yes. No new data needed. Every number in the new framing is already collected
and sitting in the CSVs. What changes is the writing and the related work.

**"Isn't prompt injection already known?"**
That it happens, yes. What nobody has is a measurement of how often untrusted
document text becomes a trusted plan step, measured against matched content of
the same length in the same position. That's a measurement gap, not a
phenomenon gap, and measurement gaps are what workshops are for.

**"Won't a reviewer say Governance Decay already did this?"**
It's single-agent, so there is no handoff. And it compares a policy being
present, absent, or pinned. It never compares the policy against matched
non-policy content, so it cannot tell you whether safety rules are specifically
disfavoured or whether everything gets dropped equally. Ours can, and the
answer is that they are specifically disfavoured. 7% against 14%.

**"We have no defence to recommend. Isn't that weak?"**
We have a located mechanism, three interventions with measured effects, and one
of those is free and task-neutral. We also say plainly where an intervention
has to go, which is plan construction, upstream of the compressor. A workshop
paper that locates a problem and rules out three plausible fixes across 7,000
trials is a normal workshop paper. What reviewers punish is a defence that
works for unexamined reasons, which is what we nearly submitted.

---

## One last thing, for you not them

You're the only person who has looked at all of this, which is why you're the
only person who could have noticed. That isn't a communication failure, it's
what happens when one person runs the analysis. Catching it nine days out is
the good version of this. Catching it in a reviewer's comments is the bad one.
