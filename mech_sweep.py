"""
What happens to the required values under MECHANICAL compression?

No model is called. An oracle Agent 1 builds the handoff the `plan` regime asks
for, PLAN then EVIDENCE, and deterministic compressors cut it to a token
budget. Then we count how many of the values Agent 2 cannot obtain any other
way are still present.

Why bother when the real compressor is an LLM. Because this is the floor. A
summariser that cannot beat "keep the first N tokens" is not doing anything
worth measuring, and until now nobody knew what that floor was. It also gives a
dose-response curve today, for free, against which the model runs can be read.

Four compressors, chosen to separate position from structure:

  head        first N tokens. The PLAN is at the top, so this is the best case
              a position-blind method can do on this layout.
  tail        last N tokens, which is what sliding_window does now. The PLAN is
              gone immediately, so this is the worst case, and the gap between
              head and tail is pure layout.
  uniform     drop every k-th sentence to hit budget. Position-neutral.
  plan_first  keep the PLAN whole, fill the rest with evidence. What a
              structure-aware compressor would do, and the ceiling for a
              mechanical method.

Retention is scored the same way the pipeline scores a run: a required value
counts as present if it survives in any rendering a reader would accept, so a
date written back differently is not a false failure.

    python3 mech_sweep.py
    python3 mech_sweep.py --budgets 100,150,200,300,400,600
"""

import argparse
import json
import os
import re
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import obtainable

CHARS_PER_TOKEN = 4          # the estimate used throughout the build


def toklen(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def plain(doc: str) -> str:
    """Render a repr-shaped document the way a transcriber would write it.

    The documents arrive as CalendarEvent(id_='1', title=...). The `plan`
    prompt explicitly lets Agent 1 drop the surrounding syntax and write the
    field values plainly, so the oracle does the same. Counting the
    constructor punctuation would overstate the handoff by about a fifth.
    """
    s = re.sub(r"\b\w+\(", " ", doc)
    s = re.sub(r"[=\[\]{}()']", " ", s)
    return re.sub(r"[ \t]+", " ", s).strip()


def oracle_handoff(task, wide=False):
    """What a perfect Agent 1 writes under the `plan` regime.

    The PLAN carries the action, plus only the values that appear nowhere in
    the records and therefore cannot be recovered by anyone downstream. Six of
    53 tasks have one; for the rest it is a tool name.

    `wide=True` restores the earlier version, where PLAN restated every
    required value. Kept so the two can be run as arms: the gap between them
    is how much a planning step decides on its own, which is the same question
    Yibo raised about delegation, answered with a number.
    """
    final = task["required_calls"][-1]
    crit = task.get("handoff_critical") or {}
    # every write Agent 2 must make, falling back to the final call alone for
    # task files built before agent2_calls existed
    calls = task.get("agent2_calls") or [{"tool": final["tool"],
                                          "critical": crit}]
    evidence = "EVIDENCE: " + " ".join(plain(d) for d in task["documents"])

    if wide:
        plan = "PLAN: " + " THEN ".join(
            "call " + c["tool"] + " with "
            + "; ".join(f"{k} = {v}" for k, v in c["critical"].items())
            for c in calls)
        return plan, evidence

    # What the PLAN must carry is what the EVIDENCE cannot supply, and the
    # evidence is the documents alone. Agent 2 never sees the task prompt, by
    # design, because that is what makes the handoff load-bearing.
    #
    # Measured: of 117 required values, 45 are in the documents only, 27 in
    # both, 40 in the PROMPT only, and 5 in neither. So 45 of 117 reach Agent 2
    # only if Agent 1 states them. An earlier version of this function carried
    # just the 5 computed ones and the uncompressed control scored 40%, not
    # 100%: the 40 prompt-only values were gone before compression ran, and
    # the run looked like a compression effect when it was a plumbing hole.
    #
    # The other 72 stay in the evidence and have to survive the cut on their
    # own, alongside any injection, which is the comparison the paper needs.
    # Test against the evidence AS AGENT 2 WILL SEE IT, not against the raw
    # documents. plain() strips the repr punctuation, and a value that was
    # findable with its quotes and brackets intact is not always findable
    # afterwards. Checking the wrong text left 12 tasks failing with no
    # compression at all.
    hay_docs = obtainable.norm(" ".join(plain(d) for d in task["documents"]))

    def unrecoverable(c):
        return {k: v for k, v in c["critical"].items()
                if not all(obtainable.findable(k, p, hay_docs)[0]
                           for p in (v if isinstance(v, (list, tuple)) else [v]))}

    bits = []
    for c in calls:
        carry = unrecoverable(c)
        bits.append("call " + c["tool"]
                    + (" with " + "; ".join(f"{k} = {v}"
                                            for k, v in carry.items())
                       if carry else ""))
    return "PLAN: " + " THEN ".join(bits), evidence


# ------------------------------------------------------------- compressors

def c_head(plan, evidence, budget):
    return (plan + " " + evidence)[:budget * CHARS_PER_TOKEN]


def c_tail(plan, evidence, budget):
    return (plan + " " + evidence)[-budget * CHARS_PER_TOKEN:]


def c_uniform(plan, evidence, budget):
    """Keep every k-th sentence, k chosen so the result lands on budget.

    Position-neutral by construction: it samples across the whole handoff
    rather than from one end, so comparing it against head and tail separates
    "the compressor sampled badly" from "the value was at the wrong end".
    """
    sents = [s for s in re.split(r"(?<=[.!?])\s+|\n+", plan + " " + evidence)
             if s.strip()]
    if not sents:
        return ""
    limit = budget * CHARS_PER_TOKEN
    total = sum(len(s) + 1 for s in sents)
    step = max(1, -(-total // max(1, limit)))      # ceil(total / limit)
    out, size = [], 0
    for s in sents[::step]:
        if size + len(s) + 1 > limit:
            break
        out.append(s)
        size += len(s) + 1
    return " ".join(out)


def c_plan_first(plan, evidence, budget):
    room = budget * CHARS_PER_TOKEN - len(plan) - 1
    if room <= 0:
        return plan[:budget * CHARS_PER_TOKEN]
    return plan + " " + evidence[:room]


COMPRESSORS = {"head": c_head, "tail": c_tail, "uniform": c_uniform,
               "plan_first": c_plan_first}


# ----------------------------------------------------------------- scoring

def retention(task, text):
    """Fraction of required values still present, and which are gone."""
    hay = obtainable.norm(text)
    vals = obtainable.flatten(task.get("handoff_critical") or {})
    if not vals:
        return 1.0, []
    lost = [(k, v) for k, v in vals if not obtainable.findable(k, v, hay)[0]]
    return (len(vals) - len(lost)) / len(vals), lost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/sample_tasks_all.json")
    ap.add_argument("--budgets", default="100,150,200,300,400,600")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           args.tasks)) as fh:
        tasks = json.load(fh)

    handoffs = {t["id"]: oracle_handoff(t) for t in tasks}
    sizes = sorted(toklen(p + " " + e) for p, e in handoffs.values())
    plans = sorted(toklen(p) for p, _ in handoffs.values())

    print("=" * 78)
    print(f"MECHANICAL COMPRESSION SWEEP  |  {len(tasks)} tasks  |  no model calls")
    print("=" * 78)
    print(f"\n  oracle handoff : {sizes[0]} to {sizes[-1]} tokens, "
          f"median {int(median(sizes))}")
    print(f"  of which PLAN  : {plans[0]} to {plans[-1]} tokens, "
          f"median {int(median(plans))}")
    print(f"  so the PLAN alone fits inside every budget tested; what the")
    print(f"  compressor loses is evidence, and whether that matters depends")
    print(f"  on whether the PLAN itself survives the cut")

    print(f"\n  retention of handoff_critical, mean over {len(tasks)} tasks")
    print(f"\n  {'budget':<9}{'ratio':>8}" +
          "".join(f"{c:>13}" for c in COMPRESSORS))
    print("  " + "-" * (17 + 13 * len(COMPRESSORS)))

    grid = {}
    for b in budgets:
        ratio = median(sizes) / b
        row = f"  {b:<9}{ratio:>7.1f}:1"
        for cname, fn in COMPRESSORS.items():
            rs = []
            for t in tasks:
                p, e = handoffs[t["id"]]
                r, _ = retention(t, fn(p, e, b))
                rs.append(r)
            grid[(b, cname)] = rs
            row += f"{100 * sum(rs) / len(rs):>12.0f}%"
        print(row)

    print(f"\n  tasks keeping EVERY required value")
    print(f"\n  {'budget':<9}" + "".join(f"{c:>13}" for c in COMPRESSORS))
    print("  " + "-" * (9 + 13 * len(COMPRESSORS)))
    for b in budgets:
        row = f"  {b:<9}"
        for cname in COMPRESSORS:
            rs = grid[(b, cname)]
            row += f"{sum(1 for r in rs if r >= 1.0):>9}/{len(rs):<3}"
        print(row)

    # where position matters
    b = budgets[len(budgets) // 2]
    print(f"\n  At {b} tokens, by suite, plan_first")
    print(f"  {'suite':<12}{'n':>4}{'retention':>12}{'all values kept':>18}")
    print("  " + "-" * 46)
    for dom in dict.fromkeys(t["domain"] for t in tasks):
        idx = [i for i, t in enumerate(tasks) if t["domain"] == dom]
        rs = [grid[(b, "plan_first")][i] for i in idx]
        print(f"  {dom:<12}{len(rs):>4}{100 * sum(rs) / len(rs):>11.0f}%"
              f"{sum(1 for r in rs if r >= 1.0):>14}/{len(rs)}")

    if args.verbose:
        b = budgets[len(budgets) // 2]
        print(f"\n  what gets lost at {b} tokens, head")
        for t in tasks:
            p, e = handoffs[t["id"]]
            r, lost = retention(t, c_head(p, e, b))
            if lost:
                print(f"    {t['id']:<10}{100*r:>4.0f}%  lost {lost}")

    print("\n" + "=" * 78)
    print("  Read this as the floor. An LLM compressor that scores below")
    print("  `head` is actively destroying information, not summarising.")
    print("  `plan_first` is what structure buys, and the gap between it and")
    print("  `tail` is the cost of a position-blind method on this layout.")
    print("=" * 78)


if __name__ == "__main__":
    sys.exit(main())
