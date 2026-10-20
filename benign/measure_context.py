"""
How much does Agent 1 actually read, per task?

That is the number that sets the compression ratio, not the total size of the
environment. A task that calls search_emails("hawaii") gets three emails back,
not the whole inbox, so file sizes tell you nothing useful. This runs each
task's read steps for real and counts what comes back.

  python3 measure_context.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import benign
from agentdojo.functions_runtime import FunctionsRuntime

suite, env, tasks, sizes, cids, dropped = benign.build()
runtime = FunctionsRuntime(suite.tools)

print(f"{'task':<16}{'read calls':>11}{'returned chars':>16}{'~tokens':>10}  tools used")
print("-" * 86)

rows = []
for t in tasks:
    total, used = 0, []
    for step in t["R"]:
        if step["tool"] not in benign.READ_TOOLS:
            continue
        used.append(step["tool"])
        try:
            out, err = runtime.run_function(env, step["tool"], step["args"],
                                            raise_on_error=False)
            total += len(str(out))
        except Exception:
            pass
    rows.append((t["task_id"], len(used), total, total // 4))
    print(f"{t['task_id']:<16}{len(used):>11}{total:>16,}{total//4:>10}  "
          f"{','.join(used)[:34]}")

toks = [r[3] for r in rows if r[3] > 0]
print("-" * 86)
print(f"tasks with a read step : {len(toks)} of {len(rows)}")
if toks:
    print(f"context seen by Agent 1: {min(toks)} to {max(toks)} tokens, "
          f"mean {sum(toks)//len(toks)}")
    print(f"handoff cap            : 180 words (~240 tokens)")
    print(f"compression ratio      : ~{sum(toks)//len(toks)//240}:1 on average")
if dropped:
    print(f"excluded as oversized  : {[f'{t} ({c} tok)' for t, c in dropped]}")
