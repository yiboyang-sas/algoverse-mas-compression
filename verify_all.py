"""
Does the expanded environment still have correct tasks, in every suite?

The expander inserts plausible candidate answers into the same containers the
tasks read. The failure mode is that one becomes a SECOND valid answer, or
displaces the real one, and the task quietly turns unanswerable or ambiguous.
`collect_forbidden` is meant to prevent that; this is the check.

Method: run each task's own ground truth against the expanded environment and
ask AgentDojo's `utility()` whether it was solved. Ground truth is the correct
answer by definition, so a task that now fails utility() was broken by us.

Each suite is compared against the same check on its untouched environment, so
a task AgentDojo itself cannot verify is not counted against us.

Structural collisions are the thing to watch for outside workspace. The
forbidden set catches string collisions. It cannot catch a COUNT changing:
several slack tasks ask for the channel with the fewest messages or the most
users, and adding either would silently change the right answer without any
string ever matching. That is why the slack inflater adds messages but never
channels or users, and why this script exists rather than trusting the
forbidden set.

  python3 verify_all.py
  python3 verify_all.py --suite banking
"""

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

import benign_all
import suites


def run_ground_truth(suite, env, task):
    """Execute a task's ground truth; return (pre_env, post_env, output)."""
    runtime = FunctionsRuntime(suite.tools)
    pre = copy.deepcopy(env)
    post = copy.deepcopy(env)
    outputs = []
    for call in task.ground_truth(post):
        try:
            res, err = runtime.run_function(post, call.function, call.args,
                                            raise_on_error=False)
            outputs.append(str(res) if err is None else f"ERROR: {err}")
        except Exception as e:
            outputs.append(f"EXCEPTION: {e}")
    return pre, post, "\n".join(outputs)


def check(suite, env, tids):
    out = {}
    for tid in tids:
        task = suite.user_tasks[tid]
        try:
            pre, post, res = run_ground_truth(suite, env, task)
            out[tid] = bool(task.utility(res, pre, post, strict=False))
        except Exception as e:
            out[tid] = f"error: {type(e).__name__}"
    return out


def verify_suite(name):
    suite = get_suite(benign_all.SUITE_VERSION, name)
    plain = suite.load_and_inject_default_environment({})

    _, expanded, tasks, stats, dropped = benign_all.build_suite(name)
    tids = [t["raw_task_id"] for t in tasks]

    before = check(suite, plain, tids)
    after = check(suite, expanded, tids)

    broken, unverifiable, fine = [], [], []
    rows = []
    for tid in tids:
        b, a = before[tid], after[tid]
        if b is True and a is True:
            verdict, bucket = "ok", fine
        elif b is True and a is not True:
            verdict, bucket = "BROKEN BY EXPANSION", broken
        else:
            verdict, bucket = "not verifiable upstream", unverifiable
        bucket.append(tid)
        rows.append((tid, b, a, verdict))

    return {"name": name, "rows": rows, "broken": broken,
            "unverifiable": unverifiable, "fine": fine,
            "stats": stats, "dropped": dropped, "tasks": tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, choices=suites.ALL)
    ap.add_argument("--quiet", action="store_true",
                    help="suite totals only, no per-task lines")
    args = ap.parse_args()

    names = [args.suite] if args.suite else suites.ALL
    reports = []

    print("=" * 72)
    print("GROUND TRUTH VERIFICATION  |  does expansion break any task?")
    print("=" * 72)

    for name in names:
        r = verify_suite(name)
        reports.append(r)
        print(f"\n{name.upper()}  {len(r['tasks'])} tasks")
        if not args.quiet:
            print(f"  {'task':<18}{'plain':>8}{'expanded':>11}   verdict")
            print("  " + "-" * 52)
            for tid, b, a, verdict in r["rows"]:
                mark = "  " if verdict == "ok" else "<-"
                print(f"  {tid:<18}{str(b):>8}{str(a):>11} {mark} {verdict}")
        s = r["stats"]
        print(f"  pass before and after : {len(r['fine'])}")
        print(f"  broken by expansion   : {len(r['broken'])}  {r['broken'] or ''}")
        print(f"  unverifiable upstream : {len(r['unverifiable'])}  "
              f"{r['unverifiable'] or ''}")
        print(f"  units: {s['accepted']} accepted, {s['rejected']} rejected")
        print(f"  starved: {s['starved'] or 'none'}")

    print("\n" + "=" * 72)
    tb = sum(len(r["broken"]) for r in reports)
    tf = sum(len(r["fine"]) for r in reports)
    tu = sum(len(r["unverifiable"]) for r in reports)
    ncore = sum(1 for r in reports for t in r["tasks"] if t["tier"] == "core")
    print(f"  {tf} pass, {tb} broken, {tu} unverifiable upstream")
    print(f"  core working set across all suites: {ncore}")
    print("=" * 72)

    if tb:
        print("\n  FAIL: expansion changed the answer to a task.")
        print("  String collision -> widen the forbidden set.")
        print("  Count or ordering change -> the inflater is adding the wrong "
              "kind of thing for that suite.")
        return 1
    print("\n  PASS: every task AgentDojo can verify still solves correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
