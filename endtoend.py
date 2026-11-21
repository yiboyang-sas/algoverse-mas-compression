"""
End-to-end rehearsal through the real pipeline, with no model calls.

Everything measured so far was retention: is the value still present in the
text. That is not the number the paper reports. The paper reports task
success, which asks a harder question: can Agent 2 BUILD the correct tool call
out of what reached it. A value can survive compression and still be unusable
if Agent 2 cannot tell which argument it belongs to.

So this runs pipeline.run_pipeline() for real, with three stand-ins:

  Agent 1   writes the handoff the `plan` regime asks for: a narrow PLAN
            naming the action plus any figure that exists nowhere, then the
            records as EVIDENCE.
  compressor  deterministic. head, uniform or tail at a token budget.
  Agent 2   maximally competent extraction. It takes the tool from the PLAN
            and, for each argument, searches the compressed text for the
            value. It never invents. If the value did not survive, the call
            goes out missing that argument and the scorer fails it.

Agent 2 being maximally competent is the point. A real model will do worse,
never better, so what comes out is an UPPER BOUND on task success under
mechanical compression. If the ceiling is already too low, no model run will
rescue it and the budget is wrong.

It also exercises the wiring: TOOL_REGISTRY, the per-agent tool scoping, the
replay, the scorer. A missing registry entry silently drops every call and
scores a perfect run at zero, which has happened three times on this project.

    python3 endtoend.py
    python3 endtoend.py --budgets 150,300,600,1200 --compressor uniform
"""

import argparse
import json
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import baseline            # applies the registry, scorer and prompt patches
import pipeline
import mech_sweep as mech
import obtainable


class _Agent1:
    """Writes the `plan` handoff. Also emits the read calls it holds."""

    name = "oracle-agent1"

    def __init__(self):
        self.task = None

    def respond(self):
        t = self.task
        held = set(t.get("agent1_tools", []))
        mine = [c for c in t["required_calls"] if c["tool"] in held]
        plan, evidence = mech.oracle_handoff(t)
        return json.dumps({"tool_calls": mine,
                           "handoff_note": plan + "\n" + evidence})


class _Agent2:
    """Rebuilds the final call from the compressed handoff, inventing nothing.

    For each argument the task requires, it looks for the value in what it
    received. Present means it can use it. Absent means the argument is left
    out, and the scorer then fails the call, which is exactly what should
    happen: Agent 2 holds no read tools, so a value that did not survive is
    not recoverable by any means.
    """

    name = "oracle-agent2"

    def __init__(self):
        self.task = None
        self.lost = []

    def respond(self, compressed):
        t = self.task
        final = t["required_calls"][-1]
        calls = t.get("agent2_calls") or [
            {"tool": final["tool"], "critical": t.get("handoff_critical") or {}}]
        hay = obtainable.norm(compressed)

        out, missing = [], []
        for c in calls:
            args = {}
            for k, v in c["critical"].items():
                parts = v if isinstance(v, (list, tuple)) else [v]
                if all(obtainable.findable(k, p, hay)[0] for p in parts):
                    args[k] = v
                else:
                    missing.append(f"{c['tool']}.{k}")
            out.append({"tool": c["tool"], "args": args})
        if missing:
            self.lost.append((t["id"], missing))
        return json.dumps({"tool_calls": out, "handoff_note": "done"})


class _Backend:
    """One object pretending to be both agents, because run_pipeline takes one.

    Branch on the system prompt. Note the test: Agent 2's own system prompt
    contains the string "Agent 1" (it says it receives a handoff from it), so
    a substring check sends both agents down the same path and Agent 2 never
    acts. That bug cost a full self-test cycle.
    """

    name = "oracle"

    def __init__(self):
        self.a1, self.a2 = _Agent1(), _Agent2()
        self.compress = None

    def set_task(self, t):
        self.a1.task = self.a2.task = t

    def chat(self, system: str, user: str) -> str:
        if system.startswith("You are Agent 1"):
            return self.a1.respond()
        handoff = user.split("Handoff from Agent 1:", 1)[-1].strip()
        return self.a2.respond(handoff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/sample_tasks_all.json")
    ap.add_argument("--budgets", default="150,300,600,1200,2400")
    ap.add_argument("--compressor", default="head",
                    choices=list(mech.COMPRESSORS))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, args.tasks)) as fh:
        tasks = json.load(fh)
    budgets = [int(b) for b in args.budgets.split(",")]
    cfn = mech.COMPRESSORS[args.compressor]

    print("=" * 76)
    print(f"END TO END  |  {len(tasks)} tasks  |  real pipeline, no model calls")
    print(f"compressor: {args.compressor}   Agent 2: perfect extraction, "
          f"invents nothing")
    print("=" * 76)

    # uncompressed control: run_pipeline with condition "none"
    be = _Backend()
    ok = 0
    for t in tasks:
        be.set_task(t)
        r = pipeline.run_pipeline(t, be, be, "none")
        ok += bool(r["task_success"])
    print(f"\n  no compression        {ok}/{len(tasks)} = {100*ok/len(tasks):.0f}%"
          f"   <- the ceiling. Anything below this is compression.")
    if ok != len(tasks):
        print("    NOT 53/53. The wiring or the handoff is broken, not the")
        print("    compressor. Fix before reading anything else.")

    print(f"\n  {'budget':<9}{'ratio':>8}{'success':>10}{'rate':>8}"
          f"{'values lost':>14}")
    print("  " + "-" * 50)

    sizes = [mech.toklen(" ".join(mech.oracle_handoff(t))) for t in tasks]
    rows = []
    for b in budgets:
        # patch compression for this budget
        orig = pipeline.compress
        pipeline.compress = (lambda cond, handoff, backend, task, _b=b:
                             handoff if cond == "none"
                             else cfn(*handoff.split("\n", 1), _b)
                             if "\n" in handoff else cfn("", handoff, _b))
        be2 = _Backend()
        ok = 0
        for t in tasks:
            be2.set_task(t)
            r = pipeline.run_pipeline(t, be2, be2, "unconditional")
            ok += bool(r["task_success"])
        pipeline.compress = orig
        nlost = sum(len(m) for _, m in be2.a2.lost)
        rows.append((b, ok, be2.a2.lost))
        print(f"  {b:<9}{median(sizes)/b:>7.1f}:1{ok:>8}/{len(tasks)}"
              f"{100*ok/len(tasks):>7.0f}%{nlost:>14}")

    print("\n  by suite")
    doms = list(dict.fromkeys(t["domain"] for t in tasks))
    print(f"  {'budget':<9}" + "".join(f"{d:>12}" for d in doms))
    print("  " + "-" * (9 + 12 * len(doms)))
    for b, ok, lost in rows:
        failed = {tid for tid, _ in lost}
        row = f"  {b:<9}"
        for d in doms:
            ids = [t["id"] for t in tasks if t["domain"] == d]
            good = sum(1 for i in ids if i not in failed)
            row += f"{good:>8}/{len(ids):<3}"
        print(row)

    if args.verbose and rows:
        b, ok, lost = rows[len(rows) // 2]
        print(f"\n  which arguments go missing at {b} tokens")
        from collections import Counter
        c = Counter(k for _, ks in lost for k in ks)
        for k, n in c.most_common(12):
            print(f"    {k:<20}{n:>3}")

    print("\n" + "=" * 76)
    print("  This is an UPPER BOUND. Agent 2 here never misreads and never")
    print("  invents; a real model does both. If a budget already fails here,")
    print("  no model run recovers it.")
    print("=" * 76)


if __name__ == "__main__":
    sys.exit(main())
