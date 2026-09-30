"""
Runner: sweep every task x every compression condition, print a results table,
and dump full per-run records (including the text at each stage) to JSON.

    python run.py                # mock backend, no setup
    python run.py --tasks tasks/sample_tasks.json

To use a real model, edit the CONFIG block below (or set env vars) and run again.

The table answers the paper's core question per condition:
  attack rate   = fraction of runs where the harmful call fired  (lower is safer)
  task rate     = fraction of runs where the task still succeeded (higher is better)
A good defense pushes attack rate down without pushing task rate down.
"""

from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict

from models import MockBackend, OpenAIBackend, OllamaBackend
from compressors import DEFAULT_CONDITIONS, ALL_CONDITIONS
from pipeline import run_pipeline


# ==========================================================================
# CONFIG  -- change backend HERE, in one place.
# ==========================================================================
def make_backends():
    """Return (agent_backend, compressor_backend).

    Default: mock, so this runs with zero setup. To use a real model, comment
    out the mock lines and uncomment ONE of the blocks below.
    """
    mode = os.environ.get("MAS_BACKEND", "mock")

    if mode == "mock":
        agent = MockBackend(follows_injection=True)
        compressor = MockBackend()
        return agent, compressor

    if mode == "openai":
        # Works for OpenAI, Together, Groq, OpenRouter, or a local vLLM/Ollama
        # /v1 server. Set MAS_MODEL, MAS_BASE_URL, and OPENAI_API_KEY.
        model = os.environ.get("MAS_MODEL", "gpt-4o-mini")
        base = os.environ.get("MAS_BASE_URL", "https://api.openai.com/v1")
        b = OpenAIBackend(model=model, base_url=base)
        return b, b  # same model for agents and compressor keeps it attributable

    if mode == "ollama":
        model = os.environ.get("MAS_MODEL", "qwen2.5:14b")
        b = OllamaBackend(model=model)
        return b, b

    raise SystemExit(f"unknown MAS_BACKEND: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="tasks/sample_tasks.json")
    ap.add_argument("--all-conditions", action="store_true",
                    help="sweep all 6 conditions instead of the 4-condition pilot")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)

    conditions = ALL_CONDITIONS if args.all_conditions else DEFAULT_CONDITIONS
    agent_b, comp_b = make_backends()

    print(f"backend: {agent_b.name} | tasks: {len(tasks)} | "
          f"conditions: {', '.join(conditions)}\n")

    records = []
    agg = defaultdict(lambda: {"n": 0, "attack": 0, "task": 0, "irrev": 0})

    for task in tasks:
        for cond in conditions:
            r = run_pipeline(task, agent_b, comp_b, cond)
            records.append(r)
            a = agg[cond]
            a["n"] += 1
            a["attack"] += int(r["attack_fired"])
            a["task"] += int(r["task_success"])
            a["irrev"] += int(r["attack_irreversible"])

    _print_table(conditions, agg)

    with open(args.out, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nfull per-run records (with text at each stage) -> {args.out}")


def _print_table(conditions, agg):
    print(f"{'condition':<18} {'attack rate':>12} {'task rate':>11} "
          f"{'irrev attacks':>14}   (n)")
    print("-" * 62)
    for cond in conditions:
        a = agg[cond]
        n = a["n"] or 1
        print(f"{cond:<18} {a['attack']/n:>12.0%} {a['task']/n:>11.0%} "
              f"{a['irrev']/n:>14.0%}   ({a['n']})")
    print("\nread: attack rate = how often the harmful call fired (lower safer).")
    print("      task rate = how often the real task still succeeded (higher better).")


if __name__ == "__main__":
    main()
