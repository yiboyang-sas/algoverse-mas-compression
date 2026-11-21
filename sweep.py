"""
What the compressor budget costs you, computed from runs you already paid for.

THE BUDGET IS A FREE PARAMETER AND WE PICKED IT BLIND. 150 tokens was chosen
before anything counted a token, and it decides how many tasks the
experiment can say anything about: a handoff smaller than the budget is one
the compressor is allowed to pass through untouched, so compression on that
task is a no-op and the task carries no information about compression.

That makes the budget a reporting decision, not a detail. A single budget
chosen after seeing the results is tuning. The whole sweep, declared
together, is not. This prints the sweep.

NO API CALLS. Run 1 has no compressor, so the budget does not change what
the model did; it only changes what CER those same handoffs imply. Every
number here comes out of runs/*.jsonl.

    python sweep.py results/redesign/<run_id> [more run dirs...]
    python sweep.py --csv exposure.csv results/redesign/*/

TWO THINGS ARE COMBINED and they are different:

  EXPOSED     the handoff is at least PRIMARY_CER_MIN times the budget, so
              compression has to remove something.
  RELIABLE    the task completed in EVERY repeat with no compression. A task
              that already fails cannot show compression damage, so it is
              not in the denominator — that is what the denominator means,
              not a filter applied for convenience.

The analysis set is the intersection. It is the n the paper can report.
"""

from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys

import config as runconfig
import metrics

DEFAULT_BUDGETS = [25, 50, 75, 100, 150, 200, 300]

# Below this a correct handoff cannot fit, so a failure is mechanical rather
# than semantic and the task says nothing about compression. Measured, not
# guessed: see `floor_for`.
MIN_SENSIBLE_BUDGET = 25


def load(run_dirs):
    """(handoff medians, suite, reliability) over every run directory given."""
    hand = collections.defaultdict(list)
    suite, passed, total = {}, collections.Counter(), collections.Counter()
    method = set()
    for d in run_dirs:
        path = os.path.join(d, "runs.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            lines = fh.readlines()
        for line in lines:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = r.get("task_id")
            if not tid or r.get("api_failure"):
                continue
            total[tid] += 1
            passed[tid] += int(bool(r.get("task_success")))
            v = r.get("precompression_handoff_tokens")
            if isinstance(v, (int, float)):
                hand[tid].append(v)
            if r.get("suite"):
                suite[tid] = r["suite"]
            if r.get("token_count_method"):
                method.add(r["token_count_method"])
    median = {t: statistics.median(v) for t, v in hand.items() if v}
    reliable = {t for t in total if total[t] and total[t] == passed[t]}
    return median, suite, reliable, total, method


def answer_floors(tasks_file) -> dict:
    """Per task, the smallest compressed handoff that could still be correct.

    PER TASK, NOT GLOBAL, and that distinction is the whole point. The budget
    is one number for the whole run, but the answer is not: bk_h10 needs nine
    payments written out, 224 tokens of pure account-and-amount, while bk_5
    needs one account and one figure. A 150-token budget leaves bk_5 room to
    spare and makes bk_h10 impossible, and "impossible" does not mean
    compression destroyed the answer, it means the answer never fit.

    A task whose floor exceeds the budget says nothing about compression. It
    has to come out of the analysis set for the same reason an unexposed task
    does, at the opposite end: one cannot be squeezed, the other cannot be
    satisfied.

    This is a FLOOR and a generous one. It counts only the values Agent 2
    cannot obtain any other way, with no prose, no tool syntax and no
    connecting words, so a real correct handoff is always larger.
    """
    if not os.path.exists(tasks_file):
        return {}
    with open(tasks_file) as fh:
        tasks = json.load(fh)
    out = {}
    for t in tasks:
        text = " ".join(
            f"{k} = {v}"
            for c in (t.get("agent2_calls") or [])
            for k, v in (c.get("critical") or {}).items())
        out[t.get("id")], _ = metrics.count_tokens(text)
    return out


def table(median, suite, reliable, budgets, strata_min, floors=None):
    """One row per budget.

    Three ways a task drops out, and they are different:
      unexposed   the handoff already fits, so compression is a no-op
      infeasible  the ANSWER does not fit, so compression cannot succeed
      unreliable  the task fails without compression anyway
    """
    floors = floors or {}
    rows = []
    suites = sorted({s for s in suite.values()})
    for b in budgets:
        exposed = {t for t, h in median.items() if h / b >= strata_min}
        feasible = {t for t in median
                    if floors.get(t) is None or floors[t] <= b}
        keep = exposed & feasible & reliable
        per = collections.Counter()
        for t in keep:
            per[suite.get(t, "?")] += 1
        rows.append({"budget": b,
                     "exposed": len(exposed),
                     "infeasible": len(set(median) - feasible),
                     "analysis_set": len(keep),
                     **{s: per[s] for s in suites}})
    return rows, suites


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+",
                    help="results/redesign/<run_id> directories")
    ap.add_argument("--budgets", default=",".join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument("--tasks", default="tasks/sample_tasks_63.json",
                    help="used only to compute the budget floor")
    ap.add_argument("--csv", default="")
    args = ap.parse_args(argv)

    dirs = [d for pat in args.run_dirs for d in sorted(glob.glob(pat))]
    median, suite, reliable, total, method = load(dirs)
    if not median:
        print("no runs found in: " + ", ".join(dirs))
        return 1

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    floors = answer_floors(args.tasks)
    rows, suites = table(median, suite, reliable, budgets,
                         runconfig.PRIMARY_CER_MIN, floors)

    print(f"{len(median)} tasks across {len(dirs)} run(s); "
          f"{len(reliable)} pass every repeat")
    if len(method) > 1:
        print(f"  WARNING: mixed token counting methods {sorted(method)}. "
              f"Do not pool these.")
    elif method:
        print(f"  tokens: {method.pop()}")

    if floors:
        heavy = max(floors, key=lambda t: floors[t])
        print(f"  the answer itself needs {min(floors.values())} to "
              f"{floors[heavy]} tokens depending on the task "
              f"({heavy} is the heaviest), and a budget under a task's own "
              f"floor makes that task impossible rather than compressed")

    head = (f"  {'budget':>7}{'exposed':>9}{'infeas':>8}{'analysis':>10}  "
            + "".join(f"{s[:9]:>10}" for s in suites))
    print()
    print(head)
    print("  " + "-" * (len(head) - 2))
    for r in rows:
        line = (f"  {r['budget']:>7}{r['exposed']:>9}{r['infeasible']:>8}"
                f"{r['analysis_set']:>10}  ")
        line += "".join(f"{r[s]:>10}" for s in suites)
        print(line)

    best = max(rows, key=lambda r: r["analysis_set"])
    print(f"\n  largest analysis set: {best['analysis_set']} tasks at a "
          f"{best['budget']}-token budget")
    print(f"  `infeas` is tasks whose ANSWER does not fit the budget. Those "
          f"are not compressed, they are impossible,")
    print(f"  and a budget that makes them impossible cannot be used to say "
          f"anything about compression.")
    print(f"  REPORT THE WHOLE SWEEP. One budget picked after seeing this "
          f"table is tuning; the table is not.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
