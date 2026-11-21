"""
What is actually in all four AgentDojo suites.

Answers the only question that matters for expanding the benign set: how many
user tasks in each suite end in a write action, which is what gives Agent 2
something to do. Also dumps the tool list per suite so the read/write split can
be built, and the environment's content types so filler.py knows what kind of
filler each suite needs.

    python3 explore_suites.py
"""

import json
import sys
from collections import Counter

from agentdojo.task_suite.load_suites import get_suite

VERSION = "v1.2.1"
SUITES = ["workspace", "slack", "travel", "banking"]

# A tool is a write if calling it changes the environment. AgentDojo does not
# label this, so it is inferred from the name and then printed for checking by
# hand. Guessing wrong in the safe direction (calling a write a read) would put
# an action tool in Agent 1's hands and destroy the split, so anything
# ambiguous is treated as a write.
WRITE_HINTS = (
    "send", "create", "delete", "cancel", "reschedule", "add", "append",
    "share", "update", "set", "schedule", "post", "invite", "remove",
    "transfer", "pay", "book", "reserve", "write", "edit", "move", "join",
    "leave", "rate", "assign",
)
READ_OVERRIDE = (
    "get_", "read_", "list_", "search_", "find_", "check_", "view_", "show_",
)


def classify(name: str) -> str:
    low = name.lower()
    if any(low.startswith(p) for p in READ_OVERRIDE):
        return "read"
    if any(h in low for h in WRITE_HINTS):
        return "write"
    return "read?"          # unknown, print it and decide by hand


def main():
    report = {}
    for name in SUITES:
        suite = get_suite(VERSION, name)
        env = suite.load_and_inject_default_environment({})

        tools = sorted(t.name for t in suite.tools)
        kinds = {t: classify(t) for t in tools}
        writes = {t for t, k in kinds.items() if k == "write"}

        ends_write, ends_read, broke = [], [], []
        for tid, t in suite.user_tasks.items():
            try:
                gt = t.ground_truth(env)
            except Exception as e:
                broke.append((tid, type(e).__name__))
                continue
            if not gt:
                broke.append((tid, "empty ground truth"))
                continue
            (ends_write if gt[-1].function in writes else ends_read).append(tid)

        # what the environment actually holds, so filler knows what to generate
        content = {}
        for attr in dir(env):
            if attr.startswith("_"):
                continue
            try:
                val = getattr(env, attr)
            except Exception:
                continue
            tn = type(val).__name__
            if tn in ("str", "int", "float", "bool", "NoneType", "method"):
                continue
            if hasattr(val, "model_fields"):
                content[attr] = tn

        report[name] = {
            "n_user_tasks": len(suite.user_tasks),
            "n_injection_tasks": len(suite.injection_tasks),
            "n_tools": len(tools),
            "tools": kinds,
            "ends_in_write": ends_write,
            "ends_in_read": ends_read,
            "ground_truth_failed": broke,
            "env_components": content,
        }

        print("=" * 72)
        print(f"{name.upper():<12} {len(suite.user_tasks)} user tasks, "
              f"{len(suite.injection_tasks)} injection tasks, {len(tools)} tools")
        print("=" * 72)
        c = Counter(kinds.values())
        print(f"  tools: {c['read']} read, {c['write']} write, {c['read?']} unclear")
        if c["read?"]:
            print(f"  UNCLEAR, decide by hand: "
                  f"{[t for t, k in kinds.items() if k == 'read?']}")
        print(f"  writes: {sorted(writes)}")
        print(f"\n  ends in a write : {len(ends_write):>3}  <- usable for us")
        print(f"  ends in a read  : {len(ends_read):>3}")
        print(f"  ground truth failed: {len(broke)} {broke if broke else ''}")
        print(f"\n  env holds: {content}")
        print()

    total = sum(len(r["ends_in_write"]) for r in report.values())
    print("=" * 72)
    print(f"TOTAL candidate tasks across all four suites: {total}")
    print(f"  (currently using {len(report['workspace']['ends_in_write'])} "
          f"from workspace alone)")
    print("=" * 72)

    with open("suite_survey.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print("\nwrote suite_survey.json")


if __name__ == "__main__":
    sys.exit(main())
