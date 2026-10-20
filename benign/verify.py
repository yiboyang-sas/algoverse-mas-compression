"""
Does the expanded environment still have correct tasks?

The expander inserts plausible candidate answers into the same documents the
tasks read. The obvious failure mode is that one of them becomes a SECOND valid
answer, or displaces the real one, and the task quietly becomes unanswerable or
ambiguous. `collect_forbidden` is meant to prevent that; this checks it.

Method: run each task's own ground truth against the expanded environment and
ask AgentDojo's `utility()` whether the task was solved. Ground truth is the
correct answer by definition, so any task that now fails utility() has been
broken by the expansion.

This compares against the same check on the untouched environment, so a task
that AgentDojo itself cannot verify is not counted against us.

It also reports accepted and rejected unit counts, and names any document that
got no expansion at all. That is how a too-wide forbidden set surfaces: two
tasks were silently unexpanded for a whole build before this was added.

  python3 verify.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benign
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite


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
    results = {}
    for tid in tids:
        task = suite.user_tasks[tid]
        try:
            pre, post, out = run_ground_truth(suite, env, task)
            results[tid] = bool(task.utility(out, pre, post, strict=False))
        except Exception as e:
            results[tid] = f"error: {type(e).__name__}"
    return results


def main():
    suite = get_suite(benign.SUITE_VERSION, benign.SUITE_NAME)

    # untouched baseline
    plain = suite.load_and_inject_default_environment({})
    benign.plant_controls(plain)

    # expanded
    _, expanded, tasks, sizes, _, dropped = benign.build()
    tids = [t["task_id"] for t in tasks]

    before = check(suite, plain, tids)
    after = check(suite, expanded, tids)

    print("=" * 66)
    print("GROUND TRUTH VERIFICATION  |  does expansion break any task?")
    print("=" * 66)
    print(f"\n  {'task':<16}{'plain':>10}{'expanded':>12}   verdict")
    print("  " + "-" * 52)

    broken, already, fine = [], [], []
    for tid in tids:
        b, a = before[tid], after[tid]
        if b is True and a is True:
            verdict, bucket = "ok", fine
        elif b is True and a is not True:
            verdict, bucket = "BROKEN BY EXPANSION", broken
        else:
            verdict, bucket = "not verifiable upstream", already
        bucket.append(tid)
        print(f"  {tid:<16}{str(b):>10}{str(a):>12}   {verdict}")

    print(f"\n  pass before and after : {len(fine)}")
    print(f"  broken by expansion   : {len(broken)}  {broken or ''}")
    print(f"  unverifiable upstream : {len(already)}  {already or ''}")

    print(f"\n  Context after expansion")
    print(f"    drive   : {sizes['drive_chars_before']//4} -> {sizes['drive_tokens_after']} tokens")
    print(f"    inbox   : {sizes['inbox_chars_before']//4} -> {sizes['inbox_tokens_after']} tokens")
    print(f"    calendar: {sizes['cal_chars_before']//4} -> {sizes['cal_tokens_after']} tokens")

    print(f"\n  Candidate units: {sizes['accepted']} accepted, "
          f"{sizes['rejected']} rejected for colliding with an answer")
    print(f"    {'STARVED: ' + str(sizes['starved']) if sizes['starved'] else 'no starved documents'}")
    print(f"  Dropped as oversized: {len(dropped)} "
          f"{[t for t, _ in dropped] or ''}")

    if broken:
        print("\n  FAIL: expansion changed the answer to a task. Fix forbidden set.")
        return 1
    print("\n  PASS: every task AgentDojo can verify still solves correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
