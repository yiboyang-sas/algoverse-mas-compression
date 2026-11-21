"""
Where a required value died, decided mechanically.

ONE MECHANISM, NOT A TAXONOMY. This module answers a single question that
needs no judgement: given the tool Agent 1 named in its PLAN, could the
required value have been written down at all?

Why that question earns its own module. Five of the twelve baseline failures
in the main 51 are Agent 1 naming a tool whose signature has no slot for a
value it had already retrieved:

    sl_14  needs send_direct_message(recipient, body)
           PLAN said send_channel_message(channel, body)
           All four user names sit in 11,632 characters of EVIDENCE.
           send_channel_message has no `recipient`, so none of them can
           appear in the PLAN. The tool choice deleted them.

    bk_12  needs update_scheduled_transaction(id, ...)
           PLAN said schedule_transaction, which creates a new payment and
           has no `id`. EVIDENCE held scheduled id=7.

In run 2 and run 4 this is indistinguishable from compression loss by
outcome: the value is in the documents, absent from the compressed handoff,
and the action is wrong. It is not compression. It happened before the
compressor ran, and on a run where no compressor runs at all it still
happens at the same rate.

So it has to be subtracted first, and it can be: a field is either a
parameter of the named tool or it is not. No model, no heuristic, no
threshold.

WHAT THIS DELIBERATELY DOES NOT DO. It does not decide that compression
caused anything. It does not rank causes. It answers "was this value
expressible", and everything else is left to analysis with the traces in
hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

# A PLAN line names its tool as `call <tool> with ...`. Agent 1's prompt
# specifies that shape and every regime produces it, but the parser is
# tolerant: `PLAN: send_money ...` and `PLAN: call send_money(...)` also
# resolve, because a missed tool name would silently read as "no mismatch"
# and this check exists to stop a confound, not to create one.
_PLAN = re.compile(r"^\s*PLAN\b", re.IGNORECASE)
# Two patterns tried in order, NOT one alternation. As a single regex with
# `|`, the engine matched `PLAN\s*:\s*(\w+)` at position 0 of
# "PLAN: call send_money ..." and captured the literal word "call" as the
# tool name, which then looked like an unknown tool on every well-formed
# line and reported every value as unexpressible. Exactly the false positive
# this module must not produce.
_CALL = re.compile(r"\bcall\s+([a-z_][a-z0-9_]*)", re.IGNORECASE)
_BARE = re.compile(r"^\s*PLAN\s*:\s*([a-z_][a-z0-9_]*)", re.IGNORECASE)

EXPRESSIBLE = "expressible"
UNEXPRESSIBLE = "tool_cannot_carry_value"
TOOL_NOT_NAMED = "no_tool_named"
UNKNOWN = "unknown_tool"


def plan_tools(handoff: str) -> list[str]:
    """Every tool named on a PLAN line, in order, duplicates kept.

    Duplicates matter: a task needing four direct messages and a PLAN naming
    one is a different failure from a PLAN naming four of the wrong tool.
    """
    out = []
    for line in (handoff or "").splitlines():
        if not _PLAN.match(line):
            continue
        m = _CALL.search(line)
        if m:
            out.append(m.group(1).lower())
            continue
        m = _BARE.match(line)
        if m and m.group(1).lower() != "call":
            out.append(m.group(1).lower())
    return out


@dataclass
class Mismatch:
    """One required call whose values the named tools could not hold."""

    task_id: str
    required_tool: str
    named_tools: list
    unexpressible: list              # required fields with nowhere to go
    status: str

    def as_dict(self) -> dict:
        return asdict(self)


def check(task: dict, handoff: str, tool_params: dict) -> list:
    """One Mismatch per required write call. Empty list means nothing to see.

    `tool_params` maps a tool name to its parameter names, which is
    baseline.TOOL_PARAMS, taken from AgentDojo's own signatures.
    """
    named = plan_tools(handoff)
    named_set = set(named)
    out = []

    for call in task.get("agent2_calls") or []:
        tool = call.get("tool")
        want = dict(call.get("critical") or {})
        if not tool:
            continue

        if tool in named_set:
            out.append(Mismatch(task.get("id"), tool, named, [], EXPRESSIBLE))
            continue

        if not named:
            out.append(Mismatch(task.get("id"), tool, named, sorted(want),
                                TOOL_NOT_NAMED))
            continue

        # Which of the required fields no named tool has a slot for. A field
        # that IS a parameter of something the PLAN named was expressible,
        # whether or not the model filled it in; that is a different failure
        # and not this one.
        slots = set()
        unknown = False
        for t in named_set:
            params = tool_params.get(t)
            if params is None:
                unknown = True
                continue
            slots |= set(params)

        missing = sorted(k for k in want if k not in slots)
        if unknown and missing:
            # A tool we have no signature for might well have had the slot.
            # Saying UNEXPRESSIBLE here would subtract a run we have not
            # understood from the results, which is the one direction this
            # check must never get wrong.
            status = UNKNOWN
        elif missing:
            status = UNEXPRESSIBLE
        else:
            status = EXPRESSIBLE
        out.append(Mismatch(task.get("id"), tool, named, missing, status))

    return out


def summarise(all_mismatches) -> dict:
    """Counts across a run, for the line that goes under the results table."""
    flat = [m for ms in all_mismatches for m in ms]
    by_status = {}
    for m in flat:
        by_status[m.status] = by_status.get(m.status, 0) + 1
    blocked = [m for m in flat if m.status == UNEXPRESSIBLE]
    return {
        "required_calls": len(flat),
        "by_status": by_status,
        "unexpressible_calls": len(blocked),
        "tasks_affected": sorted({m.task_id for m in blocked}),
        "fields_lost": sorted({f for m in blocked for f in m.unexpressible}),
    }
