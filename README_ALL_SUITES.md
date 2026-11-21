# All four suites: 18 tasks to 53

The benign set used one AgentDojo suite out of four. This extends it to all of
them. No adversarial content anywhere; attacks go in AgentDojo's own injection
slots and are not part of this.

```
python3 explore_suites.py     what each suite holds, tool split sanity check
python3 measure_all.py        context per task BEFORE expansion
python3 benign_all.py --dump  build all four, write benign_tasks_all.json
python3 verify_all.py         ground truth re-run, must print PASS
python3 to_harness_all.py     write sample_tasks_all.json for the harness
```

## What came out

| suite | released | context | squeeze |
|---|---|---|---|
| workspace | 20 | 1,013 to 6,824 tok | 4.2:1 to 28.4:1 |
| slack | 15 | 1,323 to 8,756 | 5.5:1 to 36.5:1 |
| banking | 12 | 1,340 to 3,184 | 5.6:1 to 13.3:1 |
| travel | 6 | 1,213 to 6,636 | 5.1:1 to 27.6:1 |
| **total** | **53** | | |

**Verification: 57 of 58 ground truths pass, 0 broken by expansion.** The one
exception is `slack:user_task_11`, where AgentDojo's own `utility()` raises
`NotImplementedError`. That is upstream, not us, and the task is still usable
because our scorer matches on `handoff_critical` rather than their utility.

18 final-write tools are exercised instead of 10, spread across calendars,
email, chat, payments, files and bookings.

## Why more than one suite matters more than more tasks

More tasks in workspace buys n. A second suite buys something a reviewer will
otherwise ask for: evidence the finding is not an artefact of one environment's
documents. Twelve banking tasks are worth more to the paper than eighteen more
workspace tasks would be.

## The funnel

AgentDojo ships 97 user tasks. 60 of them end in a write action, which is the
only kind a two-agent split can use: a read-only task ("how many meetings
today?") leaves Agent 2 with nothing to do.

| | workspace | slack | banking | travel |
|---|---|---|---|---|
| user tasks | 40 | 21 | 20 | 16 |
| end in a write | 22 | 20 | 6 | 12 |
| released | 20 | 15 | 6 | 12 |

Travel loses the most: 14 of its 20 tasks are lookups. Slack loses almost
nothing.

## Everything starts too small

This is the finding that drove the work. Measured by running each task's read
steps for real, **56 of the 60 candidates sit under 1,000 tokens before
expansion**. Slack's median is 49 tokens. Banking's is 184. Workspace, the one
suite already in use, is the largest at 453 and still mostly thin.

At a 180-word handoff cap there is nothing for a compressor to bite on at those
sizes. Expansion is not padding here, it is the only thing that creates a
compression ratio at all.

## What each suite needed

`filler.py` covers workspace: prose documents, CSV files, email threads,
calendar descriptions. The other three hold different things.

| suite | expanded | unit |
|---|---|---|
| slack | channels, DM inboxes, web pages | a message, a todo line, an article paragraph |
| banking | the ledger, the filesystem | a transaction row |
| travel | hotels, restaurants, car rental | a review |

Same rule as before. Every unit is the same kind of thing the container already
holds, built from the environment's own entities, a plausible candidate answer,
and a share carry a short qualifier that rules them out. *"Moved to #private,
kept here for the record." "Duplicate, refunded." "This was before the
refurbishment."*

## Three bugs this round, all of them the same bug

**Fixed pools repeat at volume.** A slack channel ships three messages and
needs about ninety. A hotel ships four reviews and needs fifty. A fifteen-line
pool at that volume prints every line six times, which is exactly the
content-free boilerplate the scoped-distractor design was built to replace, and
a summariser correctly drops all of it. So every generator here composes from
independent slots instead: topic, move, tail, scope. Twenty topics by fifteen
moves by ten tails is three thousand lines from forty-five fragments.

**The same trap one level up, in `filler.py`.** `unit_agenda` drew its topic
from a pool of eight but ended every single unit with the identical sentence,
"Decision not required this session." Thirty agenda items, thirty copies. Tails
are pooled now.

**Rejected units left holes in the numbering.** `expand_prose` numbered each
unit by loop counter, not by how many had been accepted, so a document whose
candidate collided with an answer read "7. ... 9. ..." with a visible gap where
the filter fired. Nothing breaks, but it advertises that the content was
generated, which is the one thing expanded content must never do.

**And a register failure.** Web pages went through `expand_prose`, which picks
a document family from the filename. A news article about an unemployment
figure came back with a numbered meeting agenda bolted into it, room bookings
and all. Pages now get their own continuation by register: article, todo list,
or personal page.

## What the forbidden set cannot catch

`collect_forbidden` takes the arguments of each task's final write call and
stops the expander emitting any of them, so it never creates a second valid
answer. That catches string collisions.

It cannot catch a **count** changing. Several slack tasks ask for the channel
with the fewest messages, or for every user not yet in a channel. Adding a
channel or a user would silently change the right answer and no string would
ever match.

So the slack inflater adds messages and never channels or users. The cost is
five slack tasks that read only `get_channels` and `get_users_in_channel`,
which cannot be expanded without touching exactly those counts. They stay in
the thin tier and out of the working set. `verify_all.py` exists because the
forbidden set alone is not enough to trust.

## Still open

**The controls are still planted as inbox email.** C, N1 and N2 travel as
fields on every task, but nothing puts them where a task will actually read
them. Nearly every task reads by targeted search, so Agent 1 probably never
opens them, and a retention of 0% that means "never read" is indistinguishable
in the logs from one that means "the compressor dropped it". Only the second is
about compression. They need moving into the documents each task reads, and
retention needs measuring at three points, not one: present in the documents
given to Agent 1, present in Agent 1's handoff, present after compression.

**`harmful_critical` is a field with nothing in it.** The benign scorer matches
the final write on `handoff_critical`, a reduced dict of identifiers. The
attack scorer matches the full args dict. If an attack spec carries prose, the
attack only registers when the model reproduces it word for word, which is the
same bug that scored three perfect benign runs at zero. The attack side needs
the identifier-level subset, and whoever writes the attacks needs telling.

**Two workspace tasks are still dropped as oversized.** `user_task_35` and
`user_task_38` call `list_files`, which returns the whole drive. Capping or
paginating that one tool brings them back, and real file APIs paginate anyway.

**The expansion is a large share of each task's context.** Say it before anyone
asks. The defence is not that it is small, it is that it is load-bearing:
unlike boilerplate, this content competes for space in the summary, so it is
part of the instrument rather than inflation of a number.

## Files

| | |
|---|---|
| `suites.py` | per-suite tool split and inflater |
| `filler_suites.py` | slack, banking and travel generators |
| `benign_all.py` | the build, all four suites |
| `verify_all.py` | ground-truth re-run per suite |
| `to_harness_all.py` | writes `sample_tasks_all.json` |
| `explore_suites.py` | what each suite holds |
| `measure_all.py` | context per task before expansion |

`benign.py`, `verify.py` and `to_harness.py` still work and still do workspace
only. `filler.py` gained two bug fixes and is shared.
