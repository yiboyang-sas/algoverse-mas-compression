"""
Benign task set across ALL FOUR AgentDojo suites.

benign.py did workspace only and released 18 tasks. Workspace has 22 that end
in a write action; the other three suites hold 38 more. This builds the whole
set under one tool split per suite, one expansion policy, and one definition of
what has to survive the handoff.

    python3 benign_all.py                  build and report
    python3 benign_all.py --dump           also write benign_tasks_all.json
    python3 benign_all.py --suite slack    one suite only

The pipeline per suite is unchanged from workspace:

  1. keep the user tasks whose ground truth ends in a write action, because a
     read-only task leaves Agent 2 nothing to do
  2. collect the strings that are an answer, so the expander never emits a
     second valid one
  3. inflate the environment with scoped distractors
  4. measure what the read steps actually return, by running them
  5. drop anything still outside the usable band
  6. record handoff_critical: the values Agent 2 cannot obtain any other way
"""

import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

import obtainable
import suites

SUITE_VERSION = "v1.2.1"

# Bands. Below thin there is nothing for a handoff cap to bite on; above
# oversized a small model cannot hold the uncompressed baseline, so the
# compression ratio has no reference point and alpha is undefined.
THIN_BELOW = 1000
OVERSIZED_ABOVE = 16000
EXCLUDE_TIERS = ("oversized",)


def tier_of(tokens):
    if tokens < THIN_BELOW:
        return "thin"
    if tokens <= OVERSIZED_ABOVE:
        return "core"
    return "oversized"


def collect_forbidden(suite, env, writes, wide: bool = False) -> set[str]:
    """Strings that would create a SECOND valid answer if the expander emitted
    them.

    `wide` decides whether free-text arguments count.

    They must, for the FILLER. ws_36 asks the agent to read the Hawaii
    vacation file and then create a packing list. The answer lives in a
    `content` argument, which is prose and therefore excluded here, so the
    packing items were never protected, and filler.unit_item cheerfully
    inserted its own packing items into the very document the task reads.
    The itinerary ended up buried under "Spare charger", "Travel adapter",
    "First aid kit": distractors that are candidate answers for the thing the
    agent is being asked to produce.

    For SCORING the opposite holds: prose cannot be matched by dictionary
    comparison and demanding it fails correct runs. One set cannot serve both,
    so the filler gets the wide set and handoff_critical gets the narrow one.

    Otherwise, only the arguments of each task's FINAL write call qualify. Read-step
    arguments do not: a search query is a lookup term, not an answer, and
    forbidding it silently blocks every expansion of the documents that search
    returns. That bug cost two workspace tasks before it was caught.
    """
    bad = set()

    def walk(v):
        if isinstance(v, str) and 3 <= len(v.strip()) <= 60:
            bad.add(v.strip())
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    for t in suite.user_tasks.values():
        try:
            gt = t.ground_truth(env)
        except Exception:
            continue
        if not gt or gt[-1].function not in writes:
            continue
        for k, v in gt[-1].args.items():
            if k in obtainable.PROSE_KEYS and not wide:
                continue
            walk(v)
    return bad


def read_context(runtime, env, gt_steps, reads):
    """What the read steps actually hand Agent 1: the text and its token count.

    Run, not estimated. A task that searches one channel gets one channel back,
    not the whole workspace, and the spread across tasks is wide enough that
    guessing from store size is useless.

    The text is kept, not just measured, because handoff_critical has to be
    checked against it. A required value Agent 1 never sees is unrecoverable:
    Agent 2 holds no read tools, so nothing can put it back.
    """
    parts = []
    for step in gt_steps:
        if step.function not in reads:
            continue
        try:
            out, _ = runtime.run_function(env, step.function, step.args,
                                          raise_on_error=False)
            parts.append(str(out))
        except Exception:
            pass
    return parts, sum(len(p) for p in parts) // 4


def _named_resources(env, prompt: str, already: str) -> list[str]:
    """Anything the prompt names by URL or filename that is not already in
    the documents. Added verbatim, exactly as the tool would have returned
    it."""
    out = []
    web = getattr(env, "web", None)
    if web is not None:
        for url in re.findall(r"www\.[\w./-]+", prompt or ""):
            url = url.rstrip(".,")
            body = getattr(web, "web_content", {}).get(url)
            if body and body[:60] not in already:
                out.append(f"{url}: {body}")
    drive = getattr(env, "cloud_drive", None)
    if drive is not None:
        for fn in re.findall(r"[\w-]+\.(?:docx|txt|xlsx|pdf|csv|md)",
                             prompt or ""):
            for f in getattr(drive, "files", {}).values():
                if f.filename == fn and f.content[:60] not in already:
                    out.append(str(f))
    fs = getattr(env, "filesystem", None)
    if fs is not None:
        for fn in re.findall(r"[\w-]+\.(?:txt|csv|md)", prompt or ""):
            body = getattr(fs, "files", {}).get(fn)
            if body and body[:60] not in already:
                out.append(f"{fn}: {body}")
    return out



def _records(text: str):
    """Each individual record in the shipped environment, one string each.

    Walks the parsed JSON instead of splitting it with a regex. The regex
    version merged the last ledger row into the scheduled_transactions block
    that followed it, so "Spotify" appeared to sit beside three accounts when
    it sits beside one, and the alias was refused for the wrong reason.
    """
    try:
        data = json.loads(text)
    except Exception:
        return re.split(r"\},\s*\{|\n", text)

    out = []

    def walk(node):
        if isinstance(node, dict):
            if not any(isinstance(v, (dict, list)) for v in node.values()):
                out.append(json.dumps(node))
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return out


# Collections a task acts on that its ground truth never reads.
#
# Same hole as _named_resources, different shape. AgentDojo's ground truth for
# bk_9 is get_most_recent_transactions then
# update_scheduled_transaction(id=7, amount=1100). A real agent loop would
# have fetched the schedule itself; our documents are built from the ground
# truth's read steps, so Agent 1 was handed seventy LEDGER rows, no schedule
# at all, and asked to update scheduled id 7. The ledger has its own id 7.
#
# Not a hard task and not a model failure. A record the task acts on has to be
# in front of the agent that has to find it.
IMPLIED_READS = {
    "update_scheduled_transaction": ("get_scheduled_transactions", {}),
}


def implied_reads(gt_steps, reads, already_run):
    """Read calls the documents need but the ground truth never makes."""
    out = []
    for step in gt_steps:
        extra = IMPLIED_READS.get(step.function)
        if extra and extra[0] in reads and extra[0] not in already_run:
            out.append(extra)
            already_run.add(extra[0])
    return out


def build_suite(name, handoff_cap=180):
    reads, writes = suites.READ[name], set(suites.WRITE[name])

    suite = get_suite(SUITE_VERSION, name)
    env = suite.load_and_inject_default_environment({})

    forbidden = collect_forbidden(suite, env, writes, wide=True)
    # Which scheduled transactions are actually an answer somewhere. Only
    # those subjects need protecting from the filler.
    kw = {}
    if name == "banking":
        ids = set()
        for ut in suite.user_tasks.values():
            try:
                g = ut.ground_truth(env)
            except Exception:
                continue
            for c in g:
                if c.function == "update_scheduled_transaction":
                    v = c.args.get("id")
                    if v is not None:
                        ids.add(int(v))
        kw["update_targets"] = ids

    # The records as AgentDojo ships them, before the filler adds anything.
    # critical_aliases is resolved against THIS, not against the inflated
    # documents. bk_5 requires recipient "Spotify" and the model sends
    # SE3550000000054910000003; both name the same payee, and AgentDojo cannot
    # decide which it wants, so the alias has to come from somewhere. In the
    # inflated ledger "Spotify" sits beside SIX different accounts, because
    # the filler built six competing Spotify rows on purpose, and the alias
    # builder correctly refused to guess among them. In the shipped ledger
    # there is exactly one.
    try:
        pre_text = env.model_dump_json()
    except Exception:
        pre_text = str(getattr(env, "__dict__", env))

    stats = suites.INFLATE[name](env, forbidden, **kw)
    runtime = FunctionsRuntime(suite.tools)

    tasks, dropped = [], []
    for tid, t in suite.user_tasks.items():
        try:
            gt = t.ground_truth(env)
        except Exception:
            continue
        if not gt or gt[-1].function not in writes:
            continue

        parts, ctx = read_context(runtime, env, gt, reads)

        # A collection the task WRITES to but never READ. See implied_reads.
        ran = {step.function for step in gt if step.function in reads}
        for fn, fargs in implied_reads(gt, reads, ran):
            try:
                out, _ = runtime.run_function(env, fn, fargs,
                                              raise_on_error=False)
                parts.append(str(out))
            except Exception:
                pass

        # A resource the TASK points at, that the ground truth never fetches.
        #
        # sl_19 says "do all the tasks on my TODO list at
        # www.company-todo-list.com/bob". That page holds the only statement
        # of what the task actually is. AgentDojo's ground truth does not call
        # get_webpage for it, because a real agent loop would fetch it itself;
        # our documents come from those read steps, so Agent 1 was handed
        # channel listings and no instructions and could not possibly answer.
        # Not a model failure and not a hard task, a hole in this builder.
        parts += _named_resources(env, t.PROMPT, " ".join(parts))
        ctx = sum(len(x) for x in parts) // 4
        tier = tier_of(ctx)
        if tier in EXCLUDE_TIERS:
            dropped.append((tid, ctx, tier))
            continue

        final = gt[-1]
        raw = {k: v for k, v in final.args.items() if v not in (None, [], "")}

        # Keep only what Agent 1 can actually see. Requiring more than that
        # makes our scorer stricter than AgentDojo's own utility(), which is
        # how three perfect runs scored zero the first time round.
        hay = obtainable.norm(" ".join(parts) + " " + t.PROMPT)
        critical, pruned = obtainable.prune(raw, hay, t.PROMPT)

        # EVERY write Agent 2 has to make, not just the last one.
        #
        # handoff_critical describes the final call alone, which is wrong for
        # the 12 tasks that need more than one write: create a file then share
        # it, send three emails, invite a user then add them to a channel.
        # For those, Agent 2 was never told how to make the first call, so the
        # task failed even with no compression at all and the loss looked like
        # a compression effect. Only an end-to-end replay surfaced it.
        # Another spelling of the same payee.
        #
        # bk_5 requires recipient "Spotify" and the model sent
        # SE3550000000054910000003. Both name the same payee, and AgentDojo
        # cannot decide which it wants: its ground_truth says "Spotify" while
        # its utility() checks recipient == "SE3550000000054910000003". We
        # scored against the stricter one and failed a correct answer whose
        # arithmetic was right.
        #
        # So where a required value appears in a document beside an
        # account-number-shaped token, that token is recorded as an
        # acceptable alternative. Built from the documents, not hardcoded.
        critical_aliases = {}
        doc_text = pre_text
        IBAN = r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"
        for ck, cv in critical.items():
            # Only a NAME gets an alias. A required value that is already an
            # account number is exact by nature, and aliasing it would accept
            # money sent to the wrong account, which is worse than the bug
            # this fixes. A first pass using a character window did exactly
            # that: bk_3 ended up accepting all nine IBANs in the environment.
            if (not isinstance(cv, str) or len(cv) > 40
                    or re.fullmatch(IBAN, cv)):
                continue
            # THE SAME FIELD, not any account number lying nearby.
            #
            # Taking every IBAN in the record found two for Spotify, because
            # the scheduled row carries a `sender` as well as a `recipient`
            # and the sender is the user's own account. Two candidates meant
            # the alias was refused, for a reason that had nothing to do with
            # ambiguity about Spotify. The question is only ever "which
            # account does THIS field hold in the record that names this
            # payee", so ask that.
            alts = set()
            for rec in _records(doc_text):
                try:
                    obj = json.loads(rec)
                except Exception:
                    continue
                if cv.lower() not in json.dumps(obj).lower():
                    continue
                v = obj.get(ck)
                if isinstance(v, str) and re.fullmatch(IBAN, v):
                    alts.add(v)

            # Unambiguous, or not at all. An alias that accepted any of six
            # accounts would accept money sent to the wrong one, which is
            # worse than the bug it fixes. And the account has to be one
            # Agent 1 can actually see, or Agent 2 could never produce it.
            if len(alts) == 1 and alts <= set(re.findall(IBAN, " ".join(parts))):
                critical_aliases[ck] = sorted({cv} | alts)

        agent2_calls = []
        for c in gt:
            if c.function not in writes:
                continue
            a = {k: v for k, v in c.args.items() if v not in (None, [], "")}
            kept, _ = obtainable.prune(a, hay, t.PROMPT)
            agent2_calls.append({"tool": c.function, "critical": kept})

        tasks.append({
            "suite": name,
            "task_id": f"{name}:{tid}",
            "raw_task_id": tid,
            "prompt": t.PROMPT,
            "agent1": {"role": "Analyst. Reads records, changes nothing.",
                       "tools": reads, "handoff_cap_words": handoff_cap},
            "agent2": {"role": "Executor. Acts on the handoff. No read tools, "
                               "cannot look anything up.",
                       "tools": sorted(writes),
                       "output_format": '{"calls": [{"tool": "...", "args": {...}}]}'},
            "R": [{"tool": c.function, "args": c.args} for c in gt],
            "final_write": {"tool": final.function, "args": final.args},
            "handoff_critical": critical,
            "agent2_calls": agent2_calls,
            "critical_aliases": critical_aliases,
            "pruned_critical": pruned,
            "documents": parts,
            "n_steps": len(gt),
            "context_tokens": ctx,
            "compression_ratio": round(ctx / 240, 1) if ctx else 0,
            "tier": tier,
            "scoring": "suite.user_tasks[raw_task_id].utility(output, pre_env, post_env)",
        })

    return suite, env, tasks, stats, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, choices=suites.ALL)
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--out", default="benign_tasks_all.json")
    args = ap.parse_args()

    names = [args.suite] if args.suite else suites.ALL
    everything, grand = [], []

    for name in names:
        suite, env, tasks, stats, dropped = build_suite(name)
        everything += tasks

        core = [t for t in tasks if t["tier"] == "core"]
        thin = [t for t in tasks if t["tier"] == "thin"]
        toks = [t["context_tokens"] for t in tasks] or [0]

        print("=" * 76)
        print(f"{name.upper():<12} {len(tasks)} released of {len(suite.user_tasks)} "
              f"in the suite")
        print("=" * 76)
        print(f"  tool split : {len(suites.READ[name])} read / "
              f"{len(suites.WRITE[name])} write, overlap "
              f"{set(suites.READ[name]) & set(suites.WRITE[name]) or 'none'}")
        print(f"  expansion  : {stats['accepted']} units accepted, "
              f"{stats['rejected']} rejected for colliding with an answer")
        print(f"  starved    : {stats['starved'] or 'none'}")
        print(f"  dropped    : {[f'{t} ({c} tok, {k})' for t, c, k in dropped] or 'none'}")
        print(f"  context    : {min(toks)} to {max(toks)} tokens")
        print(f"  core {len(core)}, thin {len(thin)}")
        print()
        print(f"  {'task':<22}{'tok':>7}{'ratio':>8}  {'tier':<7}{'final write':<28}")
        print("  " + "-" * 72)
        for t in sorted(tasks, key=lambda x: x["context_tokens"]):
            print(f"  {t['task_id']:<22}{t['context_tokens']:>7}"
                  f"{t['compression_ratio']:>7}:1  {t['tier']:<7}"
                  f"{t['final_write']['tool']:<28}")
        print()
        grand.append((name, len(core), len(thin), len(dropped)))

    print("=" * 76)
    print(f"{'suite':<14}{'core':>6}{'thin':>6}{'dropped':>9}")
    print("-" * 76)
    for n, c, th, d in grand:
        print(f"{n:<14}{c:>6}{th:>6}{d:>9}")
    tc = sum(g[1] for g in grand)
    tt = sum(g[2] for g in grand)
    print("-" * 76)
    print(f"{'TOTAL':<14}{tc:>6}{tt:>6}{sum(g[3] for g in grand):>9}")
    print(f"\ncore is the working set: {tc} tasks "
          f"(workspace alone released 18)")
    print("=" * 76)

    empty = [t["task_id"] for t in everything if not t["handoff_critical"]]
    print(f"\nTasks with no extractable handoff_critical: {empty or 'none'}")

    if args.dump:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, args.out), "w") as fh:
            json.dump(everything, fh, indent=2, default=str)
        print(f"wrote {args.out} ({len(everything)} tasks)")


if __name__ == "__main__":
    sys.exit(main())
