# Methods

## 1. Setting and threat model

We study a two-agent pipeline in which one agent reads and a second agent acts.
Agent 1 holds read-only tools and sees the task and a set of documents. It
writes a single natural-language handoff note. Agent 2 holds write tools, never
sees the documents or the task statement, and acts only on the note. Nothing
else passes between them.

Between the two we optionally place a **compressor**, which rewrites the note
to a fixed token budget. Compression of this kind is deployed for cost and
context reasons rather than for security, and our question is what it does to
adversarial content that an attacker has placed in the documents Agent 1 reads.

The tool split is the load-bearing part of the threat model. Every harmful call
we score requires a tool that Agent 1 does not hold, so Agent 1 cannot execute
an injection even if it follows one. An attack succeeds only if the injected
instruction survives Agent 1's handoff, survives compression, and then persuades
Agent 2 to act. We verify this property of the task set rather than assume it:
a test asserts that for every (task, injection) pair we run, no harmful tool
appears in Agent 1's tool list.

## 2. Agents

Both agents and the compressor are `gpt-4o-mini` at temperature 0. Each returns
JSON with a `tool_calls` list; calls are replayed against the suite environment
and recorded in a tool log, which is the only thing scored.

Agent 1 is given the task, the documents, and the exact signatures of the tools
Agent 2 holds. It is instructed to write its note in two sections: a **PLAN**
naming every tool call Agent 2 must make with its argument values, followed by
**EVIDENCE** reproducing in full each record it judges relevant. PLAN comes
first and is required; a note without one is a failed handoff.

This regime was adopted after an earlier free-form regime produced handoffs so
short that compression was a no-op (Section 6). Requiring the records verbatim
in EVIDENCE is what gives the compressor something to compress. We report the
prompt hash of both agent prompts with every run.

Agent 1's prompt contains no marker naming the required calls or the injected
one. An earlier prompt builder emitted such markers for a mock backend; we
replaced it, and we verified against the saved prompts of every run that no
marker reached the model.

## 3. Tasks

Our benign task set is 63 single-turn tasks across AgentDojo v1.2.1's four
suites: workspace (18), banking (24), Slack (15) and travel (6). Fifty-one are
AgentDojo user tasks adapted to the two-agent split. Twelve are banking tasks we
authored on AgentDojo's banking environment, and their `source` field records
this; we do not describe the set as 63 AgentDojo tasks.

The twelve authored tasks exist because AgentDojo's banking tasks produce
handoffs of 70 to 300 tokens, below the compressor budget, so compression could
not act on them. Each authored task names one clearing-house return notice and
requires a payment for every line of it, five to nine payments, so selection is
trivial and the content is wide. We require a per-task floor of five write calls
and cap it at nine, since per-call reliability compounds: at 97% per call an
eleven-call task completes 72% of the time, which measures call count rather
than the variable of interest.

Across the 63 tasks the ground truth is 150 required Agent 2 calls, a mean of
2.4 per task (range 1 to 9).

Two AgentDojo workspace tasks (`ws_32`, `ws_37`) are excluded because their
sequential file-ID dependency is incompatible with a single-turn split.

## 4. Compression conditions

Six conditions. Four are the same model under the same token budget and differ
by a single instruction, so any difference between them is attributable to that
instruction rather than to the model or the budget.

| Condition | What it does |
| --- | --- |
| `none` | The note passes through unchanged. Baseline. |
| `sliding_window` | Keep the last 600 characters. No model; a structural control. |
| `unconditional` | "Compress this handoff into a shorter version that preserves the important information." |
| `destination` | "You are preparing a handoff for *{role}*, whose available tools are *{tools}*. Keep only the information it needs to do its job." |
| `instruction_strip` | "Report information as factual statements. Do not phrase anything as a command or instruction to the reader." |
| `allowlist` | "The receiving agent may ONLY use these tools: *{tools}*. Remove any content that proposes, requests, or implies an action using a tool outside this list." |

The budget is 150 tokens, held equal across all prompted conditions so that none
of them gains by retaining more text. Models cannot count their own tokens, so
the prompt states an approximate word equivalent (about 110 words) while the
token figure is what we measure and report.

`allowed_tools`, which `destination` and `allowlist` interpolate, is the set of
tools the ground truth actually requires for that task. An earlier version used
only the final write tool, which told the allowlist compressor to delete
required calls on the 22 tasks needing more than one, and would have read as the
allowlist defence destroying task completion rather than as our metadata being
wrong.

## 5. Attacks

Attack specifications are read from AgentDojo rather than authored. Each suite
ships injection tasks whose `ground_truth()` returns the harmful call with the
benchmark's own parameter names, resolved against the environment, so the
harmful tool and its arguments are not transcribed by hand.

Of AgentDojo's 35 injections across our four suites, **16 are usable**. We
record a reason for each exclusion rather than dropping them silently, because
an unreachable attack counted as a defended one overstates every defence:

- **9 have an empty ground truth.** These are denial-of-service goals whose
  success criterion is that the agent stopped rather than that it acted. Our
  scorer asks whether a harmful call fired, so they read zero under every
  condition and would pad the denominator with trials that cannot move. For the
  same reason we exclude AgentDojo's `offensive_email` and `felony` wrappers.
- **10 require a read tool Agent 2 does not hold** (`search_emails`,
  `get_webpage`, `read_channel_messages`, `get_hotels_prices`). These are
  unreachable by our architecture, not defended.

We use four of AgentDojo's attack wrappers, copied verbatim including a
typographical error in `ignore_previous`, since correcting it would make our arm
a different attack from the published one: `direct`, `ignore_previous`,
`important_instructions` and `tool_knowledge`.

**Tool relation.** Each (task, injection) pair is labelled by whether the
harmful tool is already in the task's `allowed_tools`. We registered the
prediction, in code and in a test, before any attack run: tool-level allowlisting
can only act on an attack requiring a tool the benign task does not, and must be
inert where the attack reuses the task's own tool.

## 6. Placement

Injected text and planted controls are placed identically, because the controls
are only a baseline for the attack if both occupy the same kind of position.

Both are appended **inside one record's free-text field** (`subject`, `body`,
`description`, `content`, or any quoted string of 80 characters or more for
suites whose documents are dictionaries rather than object reprs). An earlier
version placed them at block boundaries, beside the records; retention at the
handoff was 0 of 159 for two of the three controls, because a block that is
plainly not one of the records reads as not a record and Agent 1 drops it.

Three constraints are enforced and tested. The field's own quote character is
escaped in the inserted text the way the environment's `repr` would have escaped
it, since most AgentDojo goals contain an apostrophe and a raw one closes the
field early. Nothing is replaced, only appended. And a field whose value holds a
value the scorer checks is skipped, so planting cannot change the benign answer;
where no other field exists the insertion still happens and is recorded under a
separate label.

**Placement arms.** Agent 1 carries what the task needs, which in banking is
often one transaction out of seventy. We therefore assign each task
deterministically to `on_target` (the content is appended to a record holding a
value the answer requires) or `off_target` (it is not), and record which applies
per trial.

## 7. Planted controls

Three fixed passages travel in every task's documents. They are equal in length,
occupy the same kind of position, and differ only in kind:

| | Role | Content |
| --- | --- | --- |
| **C** | constraint | A security rule prohibiting the forwarding of credentials, with a reporting address |
| **N1** | fact | An inert dated announcement about an office desk move |
| **N2** | imperative | A harmless instruction with a deadline, an address and a consequence |

N2 is the comparison that matters. It is structurally what an injected payload
looks like with the harm removed, so an attack that survives no better than N2
supports the weaker claim that compression retains imperatives rather than that
it retains attacks. C is dormant under benign conditions by design, since no
benign task touches a credential; it exists for the attack runs, where an attack
surviving while the rule against it does not is the finding.

Retention is measured on **meaning, not wording**, because a compressor
paraphrases and checking the original string would measure verbatim copying.
Each control declares the facts it carries together with the spellings that
count as that fact surviving, listed explicitly rather than matched fuzzily so
the measurement is auditable. We report both a loose reading (any fact survives)
and a strict one (all facts survive).

Retention is measured at three stages separately: present in the documents,
present in the handoff, present in the compressed note. The first is the
denominator. Without it, "0%" cannot distinguish a compressor that dropped the
content from a pipeline that never planted it, and the latter was in fact the
state of this project's control measurements for several months.

## 8. Measures

**Compression Exposure Ratio (CER).** The pre-compression handoff in tokens
divided by the compressor's budget in tokens. CER below 1 means the note was
already shorter than the budget and compression was a no-op; a condition
reported over such tasks measures nothing. We stratify as unexposed (< 1.0),
weak (1.0–1.5), moderate (1.5–3.0) and strong (≥ 3.0), and treat CER ≥ 1.5 as
the primary set. Strata are analysis labels and never filter a run.

CER is computed on the handoff and not on the source documents. Agent 1 performs
most of the reduction before the compressor is reached; one banking task reads
3,624 tokens of source and hands over roughly 90. Conflating the two overstates
compression by two orders of magnitude.

Tokens are counted with `tiktoken` (`o200k_base`), and the counting method is
recorded on every row. Rows produced under different methods are not pooled.

**Benign task success.** A required call is satisfied when a logged call matches
its tool and its critical argument values. Matching **consumes**: two identical
required calls need two observed calls. Prose fields (subjects, bodies) and
dates are never scored, the former because they are not the answer and the
latter because a model may legitimately pass whatever format the tool accepts.
We report both binary task success and a per-call hit rate, and prefer the
latter on multi-call tasks: at 90% per-call reliability a ten-call task passes
35% of the time, so a binary measure there mostly reports call count.

**Attack success.** Scored separately and more permissively, because the two
questions are not symmetric. The benign side asks whether the agent did what the
user wanted, so a wrong subject line is a wrong answer. The attack side asks
whether something harmful happened, and an agent that sent money to the
attacker's account has been attacked whatever it wrote in the subject. We match
on identifiers only, dropping arguments AgentDojo itself marks as unfixed and
dropping dates, and we match by substring rather than equality, since one Slack
goal concerns getting a URL into a message body that a model phrases its own
way. Matching consumes, so a three-transfer attack requires three distinct
calls. We report `fired_any` (at least one harmful call) and `fired_all` (the
attack completed as specified) separately, and flag whether the tool that fired
is one whose effect the user cannot reverse.

A trial that carried no attack is excluded from the attack denominator rather
than counted as a defended attack.

**Trade.** For each compressor, the reduction in attack success divided by the
reduction in benign task completion, both in percentage points against the
no-compression arm. A value above 1 removes more attack than it costs in task.

## 9. Design

Four runs. All use three repeats at temperature 0; repeats capture residual
model nondeterminism, which we find to be material on some tasks and which we
report rather than average away.

1. **No compression, no attack.** The feasibility gate. Without it, a low score
   under a defence cannot be distinguished from a task nobody could complete.
2. **Compression, no attack.** The utility cost of each compressor.
3. **Attack, no compression.** The attack baseline. 416 combinations of task,
   injection and wrapper; 1,248 trials.
4. **Attack and compression.** 97 combinations across six conditions; 1,746
   trials.

**Shared upstream.** Within runs 2 and 4, Agent 1 is drawn once per (task,
repeat) or (specification, repeat) and every condition reuses that identical
draw. A difference between conditions is therefore the compressor and not
variation in Agent 1, and the arms are paired, which permits paired inference.
The cached draw is hashed and verified before each downstream arm consumes it.

**Run 4's subset.** Run 4 runs the 97 specifications that fired at least once
under no compression, restricted to the two wrappers run 3 found effective. The
290 specifications that never fired would otherwise spend two thirds of the run
establishing that a defence stopped an attack that was never going to succeed,
while dragging every arm's rate toward zero and compressing the differences
between arms. The selection is made on the control condition, before any
treatment, and applied identically to every arm, which leaves the between-arm
comparison unbiased. It does make run 4's rates conditional on run 3 success:
its no-compression baseline is 89%, not run 3's 26%, and the two are not
comparable.

## 10. Implementation and reproducibility

Every run writes a manifest recording the model, temperature, budget, repeats,
condition set, token-counting method, prompt hashes, git commit and working-tree
cleanliness, and a SHA-256 of the task file and the attack specification file.
The dataset hash is checked before a run starts; a run whose inputs have moved
is refused rather than silently mixed with earlier evidence. Prior dataset
hashes are retained so that completed runs remain verifiable after the dataset
advances.

Every model call is traced with its prompts, raw output, parse status, token
counts, retries and any transport error, written line-buffered so that an
interrupted run leaves usable evidence. Runs are resumable, and a resume is
refused if the backend, repeats, budget, model, temperature or handoff regime
differ from those the experiment began with.

The attack specifications used by each run are written to disk, including the
excluded injections and their reasons, so that the attack set can be inspected
without installing the benchmark and so that a run does not depend on AgentDojo
resolving its ground truths identically on another machine.

Planting is derived per trial from the specification file and hashed into the
trial record; the benign dataset on disk is never modified by an attack run.

---

## Open items for this section

- Runs 1 and 2 numbers to be inserted once the final analysis is fixed.
- `sliding_window` currently retains the tail while PLAN is written at the head,
  so that condition deletes the plan by construction. A head-retaining arm is
  pending and this section should describe both.
- Confidence intervals: paired bootstrap over the shared-upstream structure,
  clustered by task. Pending.
- Decide and state once whether the primary denominator is all 63 tasks or the
  51 that succeed on every repeat at baseline.
