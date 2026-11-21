"""One observable trial profile, without rescoring or inferring value loss.

Precedence: native pipeline/API error; saved success; unusable/empty upstream
handoff; empty compressor output; unparseable downstream reply; downstream
schema/access problem; downstream no action; missing required action counts;
saved argument misses despite enough actions; unclassified/missing evidence.
Only native error flags change status to ``error``. Every scored failure has
exactly one category and stays failed even when its evidence is incomplete.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import baseline
import instrument


CATEGORIES = (
    "success", "pipeline_or_transport_error",
    "upstream_unusable_reply_or_empty_handoff", "compressor_empty_output",
    "downstream_unparseable_reply", "downstream_no_action",
    "downstream_schema_or_access_error", "required_actions_not_issued",
    "required_arguments_mismatch", "unclassified_or_missing_evidence",
)
LIMITATIONS = [
    "Categories are mutually exclusive observable profiles, not proven root causes; precedence can hide co-occurring problems.",
    "Saved primary scores are authoritative; no score, argument match, control retention, or upstream tool mismatch is recomputed.",
    "Native controls and tool_mismatch are annotations only; neither establishes compression loss or changes the cohort.",
    "Raw attempted calls are not executed calls; only the saved accepted log supplies issued-action counts.",
    "Required argument mismatch means saved scorer misses with enough same-named actions; pairing, order, and normalized values are not re-evaluated.",
    "Absent traces are unavailable evidence, whereas an explicitly empty reply or accepted log is an observed empty result.",
    "The tolerant team parser can recover malformed JSON; calls.parse_status describes transport capture, not the final parser outcome.",
]
_PREFIX = "Handoff from Agent 1:\n"
_STAGES = ("agent1", "compressor", "agent2")


def _parse(raw):
    if raw is None:
        return None, "missing"
    if not isinstance(raw, str):
        return None, "non_text"
    try:
        # _parse_tolerant normally appends to the live experiment context.
        # Offline inspection must not mutate those native parse measurements.
        with instrument.run_context(None):
            parsed = baseline._parse_tolerant(raw)
        if not isinstance(parsed, dict):
            return None, "invalid_envelope"
        fallback = bool(raw.strip()) and not parsed.get("tool_calls") and parsed.get("handoff_note") == raw.strip()
        return parsed, "text_fallback" if fallback else "decoded"
    except (TypeError, ValueError, AttributeError, KeyError):
        return None, "unusable"


def _integer(value):
    return type(value) is int and value >= 0


def classify(task, record, calls: list):
    """Classify native saved evidence. Never call a backend or modify inputs.

    ``calls`` contains model-call traces from calls.jsonl, not tool actions.
    ``record`` is the saved run, optionally enriched with its native results
    text/controls/tool_mismatch. Missing annotations remain None.
    """
    annotations = {name: deepcopy(record.get(name)) for name in
                   ("controls", "tool_mismatch", "parse_events", "failure_stage")}
    annotations.update(flags=[], attempted_agent1_writes=[],
                       agent2_schema_or_access_issues=[], extra_accepted_actions_by_tool={},
                       extra_attempted_agent2_actions_by_tool={})
    evidence = {name: deepcopy(record.get(name)) for name in
                ("task_success", "required_hit", "required_total", "missed", "api_failure")}
    evidence.update(observations=[], stage_traces={})

    def note(code, source, **details):
        annotations["flags"].append(code)
        evidence["observations"].append({"code": code, "source": source, **details})

    def finish(category, status):
        return {"category": category, "status": status, "evidence": evidence,
                "annotations": annotations, "limitations": list(LIMITATIONS)}

    grouped = {stage: [] for stage in _STAGES}
    errors = []
    for index, call in enumerate(calls if isinstance(calls, list) else []):
        if not isinstance(call, dict):
            note("malformed_model_call_trace", "calls.jsonl", call_index=index)
            continue
        stage = call.get("stage")
        if stage in grouped:
            grouped[stage].append((index, call))
        if call.get("api_error"):
            errors.append({"call_index": index, "stage": stage, "api_error": call["api_error"]})
    by_stage = {}
    for stage, values in grouped.items():
        evidence["stage_traces"][stage] = [index for index, _ in values]
        if len(values) == 1:
            by_stage[stage] = values[0][1]
        elif len(values) > 1:
            note("ambiguous_duplicate_stage_trace", "calls.jsonl", stage=stage)
    evidence["failed_model_calls"] = errors
    native_error = bool(record.get("api_failure") or record.get("api_error"))
    if native_error:
        evidence["error_scope"] = "transport_call_observed" if errors else "pipeline_or_unknown"
        evidence["native_api_error"] = record.get("api_error")
    elif errors:
        note("failed_call_without_native_error_flag", "calls.jsonl")

    parsed, parse_kind = {}, {}
    for stage in ("agent1", "agent2"):
        parsed[stage], parse_kind[stage] = _parse(by_stage.get(stage, {}).get("raw_output"))
    evidence["inspection_parse_kind"] = parse_kind
    events = record.get("parse_events")
    downstream_event = events[1] if isinstance(events, list) and len(events) == 2 else None
    if isinstance(events, list) and len(events) not in (0, 2):
        note("parse_events_not_a_complete_two_agent_pair", "record.parse_events")
    if record.get("parse_failure") and downstream_event is None:
        note("native_parse_failure_stage_unresolved", "record.parse_failure")

    text = record.get("text") if isinstance(record.get("text"), dict) else {}
    handoff, handoff_source = text.get("handoff"), "record.text.handoff"
    if "handoff" not in text:
        a1 = parsed["agent1"]
        if a1 is not None:
            handoff = a1.get("handoff_note", "") or by_stage.get("agent1", {}).get("raw_output")
            handoff_source = "calls.agent1.raw_output:team_parser"
    upstream_empty = isinstance(handoff, str) and not handoff.strip()
    upstream_unusable = handoff is not None and not isinstance(handoff, str)
    upstream_unusable |= handoff is None and parse_kind["agent1"] in ("non_text", "invalid_envelope", "unusable")
    if upstream_empty or upstream_unusable:
        note("upstream_empty_handoff" if upstream_empty else "upstream_unusable_reply", handoff_source)

    transmitted, transmitted_source = None, None
    a2_input = by_stage.get("agent2", {}).get("user_input")
    if isinstance(a2_input, str) and a2_input.startswith(_PREFIX):
        transmitted, transmitted_source = a2_input[len(_PREFIX):], "calls.agent2.user_input"
    elif isinstance(text.get("compressed"), str):
        transmitted, transmitted_source = text["compressed"], "record.text.compressed"
    elif isinstance(by_stage.get("compressor", {}).get("raw_output"), str):
        transmitted = by_stage["compressor"]["raw_output"].strip()
        transmitted_source = "calls.compressor.raw_output"
    compressor_empty = record.get("condition") not in (None, "none") and isinstance(transmitted, str) and not transmitted.strip()
    if compressor_empty:
        note("compressor_empty_output", transmitted_source)

    held_value = task.get("agent2_held_tools", task.get("allowed_tools"))
    held = set(held_value) if isinstance(held_value, (list, tuple)) else None
    a1_calls = (parsed["agent1"] or {}).get("tool_calls")
    if isinstance(a1_calls, list):
        annotations["attempted_agent1_writes"] = [c["tool"] for c in a1_calls
            if isinstance(c, dict) and c.get("tool") in baseline.TOOL_PARAMS]
        if annotations["attempted_agent1_writes"]:
            note("agent1_attempted_write_actions", "calls.agent1.raw_output:team_parser")

    a2 = parsed["agent2"]
    attempts = a2.get("tool_calls", []) if a2 is not None else None
    issues = annotations["agent2_schema_or_access_issues"]
    if a2 is not None and not isinstance(attempts, list):
        issues.append({"category": "tool_calls_not_list"})
    elif isinstance(attempts, list):
        for index, attempt in enumerate(attempts):
            if not isinstance(attempt, dict) or not isinstance(attempt.get("tool"), str) or not attempt["tool"]:
                issues.append({"category": "malformed_tool_call", "attempt_index": index})
                continue
            tool, args = attempt["tool"], attempt.get("args", {})
            if held is not None and tool not in held:
                issues.append({"category": "tool_not_held", "attempt_index": index, "tool": tool})
            if tool not in baseline.AGENTDOJO_TOOLS:
                issues.append({"category": "unknown_tool", "attempt_index": index, "tool": tool})
            if not isinstance(args, dict):
                issues.append({"category": "arguments_not_object", "attempt_index": index, "tool": tool})
            elif tool in baseline.TOOL_PARAMS and set(args) - set(baseline.TOOL_PARAMS[tool]):
                issues.append({"category": "unknown_argument_keys", "attempt_index": index, "tool": tool,
                               "keys": sorted(set(args) - set(baseline.TOOL_PARAMS[tool]))})
    if issues:
        note("downstream_schema_or_access_issues", "calls.agent2.raw_output:team_parser", issue_count=len(issues))
    downstream_unparseable = parse_kind["agent2"] in ("invalid_envelope", "non_text", "unusable", "text_fallback")
    a2_raw = by_stage.get("agent2", {}).get("raw_output")
    if downstream_event == "unparsed" and not attempts and a2 is not None and isinstance(a2_raw, str) and a2_raw.strip():
        downstream_unparseable = True
    if downstream_unparseable:
        note("downstream_unparseable_reply", "calls.agent2.raw_output:team_parser")
    downstream_no_action = isinstance(attempts, list) and not attempts and not downstream_unparseable
    if downstream_no_action:
        note("downstream_no_action", "calls.agent2.raw_output:team_parser")

    accepted = record.get("tool_calls", record.get("calls"))
    accepted_valid = isinstance(accepted, list) and all(
        isinstance(c, dict) and isinstance(c.get("tool"), str) and isinstance(c.get("args"), dict) for c in accepted)
    required = None
    # This is the team's existing write denominator, not agent2_calls or a new
    # interpretation of prose/critical fields. Only names/counts are compared.
    if isinstance(task.get("required_calls"), list) and isinstance(task.get("agent2_held_tools"), (list, tuple)):
        required = Counter(c["tool"] for c in task["required_calls"]
                           if isinstance(c, dict) and c.get("tool") in task["agent2_held_tools"])
    counts_missing = None
    if required is not None:
        evidence["required_tool_counts"] = dict(required)
        if accepted_valid:
            issued = Counter(c["tool"] for c in accepted)
            evidence["accepted_tool_counts"] = dict(issued)
            counts_missing = {tool: n - issued[tool] for tool, n in required.items() if issued[tool] < n}
            evidence["missing_required_tool_counts"] = counts_missing
            annotations["extra_accepted_actions_by_tool"] = {tool: n - required[tool] for tool, n in issued.items()
                if held is not None and tool in held and n > required[tool]}
        if isinstance(attempts, list):
            attempted = Counter(c["tool"] for c in attempts if isinstance(c, dict) and isinstance(c.get("tool"), str))
            annotations["extra_attempted_agent2_actions_by_tool"] = {tool: n - required[tool]
                for tool, n in attempted.items() if n > required[tool]}
    if accepted is not None and not accepted_valid:
        note("malformed_accepted_tool_log", "record.tool_calls_or_calls")

    if native_error:
        if record.get("task_success") is True:
            note("native_error_conflicts_with_saved_success", "record.api_failure_or_api_error")
        return finish("pipeline_or_transport_error", "error")
    if record.get("task_success") is True:
        return finish("success", "success")
    if record.get("task_success") is not False:
        return finish("unclassified_or_missing_evidence", "unverified")
    if errors:
        return finish("unclassified_or_missing_evidence", "failed")
    if upstream_empty or upstream_unusable:
        return finish("upstream_unusable_reply_or_empty_handoff", "failed")
    if compressor_empty:
        return finish("compressor_empty_output", "failed")
    if downstream_unparseable:
        return finish("downstream_unparseable_reply", "failed")
    if issues:
        return finish("downstream_schema_or_access_error", "failed")
    if downstream_no_action:
        return finish("downstream_no_action", "failed")
    hit, total = record.get("required_hit"), record.get("required_total")
    saved_miss = _integer(hit) and _integer(total) and hit < total
    denominator_matches = required is not None and _integer(total) and total == sum(required.values())
    if saved_miss and denominator_matches and counts_missing is not None:
        return finish("required_actions_not_issued" if counts_missing else "required_arguments_mismatch", "failed")
    return finish("unclassified_or_missing_evidence", "failed")
