"""Scorer-aligned call checks and conservative, reviewable stage hypotheses.

Literal retention is evidence, not a semantic judge or a causal experiment.
Every failed-trial attribution below is a candidate requiring human review.
"""
from __future__ import annotations

import json
import re

from environment import _arg_ok, _norm
from pipeline import _norm_call, _parse


def required_writes(task):
    reads = set(task.get("agent1_tools", []))
    return [c for c in task.get("required_calls", []) if c["tool"] not in reads]


def call_matches(actual, required):
    tool, args = _norm_call(actual)
    return tool == required["tool"] and all(
        _arg_ok(k, args.get(k), v, required.get("match", {}).get(k))
        for k, v in required.get("args", {}).items()
    )


def _structured_calls(text):
    """Recover explicit call JSON and the pipeline's delegated call notation."""
    calls = []
    parsed = _parse(text)
    value = parsed.get("tool_calls", [])
    if isinstance(value, dict):
        value = [value]
    if isinstance(value, list):
        calls.extend(c for c in value if isinstance(c, dict))
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\b([a-z_]\w*)\(\s*(\{)", text):
        try:
            args, _ = decoder.raw_decode(text[match.start(2):])
        except ValueError:
            continue
        calls.append({"tool": match.group(1), "args": args})
    return calls


def _atoms(value):
    if isinstance(value, dict):
        # The scorer compares attachment ids, not type labels.
        if "file_id" in value:
            return _atoms(value["file_id"])
        return [a for v in value.values() for a in _atoms(v)]
    if isinstance(value, list):
        return [a for v in value for a in _atoms(v)]
    if value is None or value == "":
        return []
    return [_norm(re.sub(r"^[\s\-*•]+", "", x))
            for x in str(value).splitlines() if x.strip()]


def _contains_atom(text, atom):
    text = _norm(text)
    if len(atom) <= 2:
        return bool(re.search(r"(?<!\w)" + re.escape(atom) + r"(?!\w)", text))
    return atom in text


def retention(required, before, after):
    evidence = []
    for key, value in required.get("args", {}).items():
        mode = required.get("match", {}).get(key)
        if mode == "nonempty":
            evidence.append({"argument": key, "mode": mode,
                             "probe": "unassessable: any nonempty value scores"})
            continue
        if mode == "any_of":
            # Alternatives are not conjunctive facts. Avoid false loss claims.
            evidence.append({"argument": key, "mode": mode,
                             "probe": "unassessable: alternative accepted values"})
            continue
        for atom in _atoms(value):
            evidence.append({"argument": key, "literal": atom,
                             "in_handoff": _contains_atom(before, atom),
                             "in_compressed": _contains_atom(after, atom),
                             "low_specificity": len(atom) <= 2})
    return evidence


def diagnose(record, task):
    writes = required_writes(task)
    calls = record.get("calls", [])
    text = record.get("text", {})
    handoff = text.get("handoff", "")
    compressed = text.get("compressed", "")
    before_calls, after_calls = _structured_calls(handoff), _structured_calls(compressed)
    checks, candidates, categories, evaluator_limits = [], set(), set(), set()
    for index, required in enumerate(writes):
        related = []
        for ci, actual in enumerate(calls):
            tool, args = _norm_call(actual)
            if tool == required["tool"]:
                related.append({"call_index": ci, "argument_correctness": {
                    key: _arg_ok(key, args.get(key), want,
                                 required.get("match", {}).get(key))
                    for key, want in required.get("args", {}).items()
                }})
        matched = any(call_matches(c, required) for c in calls)
        facts = retention(required, handoff, compressed)
        if "nonempty" in required.get("match", {}).values():
            evaluator_limits.add("reference_accepts_any_nonempty_text")
        explicit_before = any(call_matches(c, required) for c in before_calls)
        explicit_after = any(call_matches(c, required) for c in after_calls)
        lost = [f for f in facts if f.get("in_handoff") and not f.get("in_compressed")]
        absent = [f for f in facts if "literal" in f and not f["in_handoff"]]
        kept = [f for f in facts if f.get("in_compressed")]
        stage = []
        if not matched:
            categories.add("required_tool_wrong_arguments" if related else "required_tool_missing")
            if explicit_before and not explicit_after:
                stage.append("compression_structured_call_loss_candidate")
            elif lost:
                stage.append("compression_literal_information_loss_candidate")
            if absent and not explicit_before:
                stage.append("agent1_missing_or_rephrased_information_candidate")
            literal_facts = [f for f in facts if "literal" in f]
            if explicit_after or (literal_facts and len(kept) == len(literal_facts)):
                stage.append("agent2_instruction_or_argument_handling_candidate")
            if not stage:
                stage.append("unresolved_requires_semantic_review")
            candidates.update(stage)
            if any(not c["argument_correctness"].get(key, True)
                   for c in related for key in ("body", "content", "title", "description")):
                evaluator_limits.add("text_match_failure_requires_semantic_review_for_possible_evaluator_mismatch")
        checks.append({"required_write_index": index, "tool": required["tool"],
                       "expected_args": required.get("args", {}),
                       "match_modes": required.get("match", {}), "matched": matched,
                       "same_tool_calls": related, "literal_retention": facts,
                       "structured_call_in_handoff": explicit_before,
                       "structured_call_in_compressed": explicit_after,
                       "stage_candidates": stage})

    reads = set(task.get("agent1_tools", []))
    a2_calls = record.get("stage_calls", {}).get("agent2")
    if a2_calls is None:
        a2_calls = [c for c in calls if _norm_call(c)[0] not in reads]
    quality = []
    for index, actual in enumerate(a2_calls):
        tool, args = _norm_call(actual)
        matches = [i for i, required in enumerate(writes) if call_matches(actual, required)]
        quality.append({"agent2_call_index": index, "tool": tool, "args": args,
                        "replayed": True,
                        "matches_required_write_indices": matches,
                        "scorer_correct": bool(matches),
                        "category": "required_write_match" if matches else (
                            "unrequested_tool" if tool not in {r["tool"] for r in writes}
                            else "required_tool_wrong_arguments_or_extra_action")})
    dropped_a2 = record.get("dropped_calls", {}).get("agent2", [])
    for index, actual in enumerate(dropped_a2):
        tool, args = _norm_call(actual)
        quality.append({"agent2_dropped_call_index": index, "tool": tool, "args": args,
                        "replayed": False, "matches_required_write_indices": [],
                        "scorer_correct": False,
                        "category": "dropped_unknown_or_unavailable_tool",
                        "drop_reason": actual.get("why")})
    extra_recipients = []
    for q in quality:
        if len(q["matches_required_write_indices"]) > 1:
            evaluator_limits.add("one_actual_call_matches_multiple_required_references")
        for wi in q["matches_required_write_indices"]:
            wanted = writes[wi].get("args", {}).get("recipients")
            got = q["args"].get("recipients")
            if isinstance(wanted, list) and isinstance(got, list):
                extra = sorted(set(map(_norm, got)) - set(map(_norm, wanted)))
                if extra:
                    extra_recipients.append({"agent2_call_index": q["agent2_call_index"],
                                             "extra_recipients": extra})
    total_correct = sum(q["scorer_correct"] for q in quality)
    if len(quality) > total_correct:
        categories.add("extra_or_wrong_agent2_action")
    if extra_recipients:
        categories.add("extra_recipient_not_penalized_by_proxy")
        evaluator_limits.add("extra_recipient_accepted_by_subset_match")
    for agent, dropped in record.get("dropped_calls", {}).items():
        if dropped:
            categories.add(f"{agent}_unavailable_or_unknown_tool_attempt")
    for agent, status in record.get("parse_status", {}).items():
        if status in {"empty_response", "text_fallback", "malformed_tool_calls", "structured_handoff"}:
            categories.add(f"{agent}_{status}")
    lexical = [fact for check in checks for fact in check["literal_retention"] if "literal" in fact]
    before = sum(f["in_handoff"] for f in lexical)
    after = sum(f["in_compressed"] for f in lexical)
    both = sum(f["in_handoff"] and f["in_compressed"] for f in lexical)
    explicit_before = sum(check["structured_call_in_handoff"] for check in checks)
    explicit_after = sum(check["structured_call_in_compressed"] for check in checks)
    critical = {
        "critical_literal_units": len(lexical), "handoff_literal_units": before,
        "compressed_literal_units": after, "retained_handoff_literal_units": both,
        "lost_handoff_literal_units": before - both,
        "low_specificity_literal_units": sum(f["low_specificity"] for f in lexical),
        "unassessable_reference_arguments": sum("literal" not in fact for check in checks for fact in check["literal_retention"]),
        "handoff_literal_recall": before / len(lexical) if lexical else None,
        "compressed_literal_recall": after / len(lexical) if lexical else None,
        "conditional_literal_retention": both / before if before else None,
        "structured_required_calls_handoff": explicit_before,
        "structured_required_calls_compressed": explicit_after,
        "required_write_count": len(checks),
    }
    return {
        "required_write_checks": checks, "agent2_call_checks": quality,
        "correct_agent2_calls": total_correct, "agent2_call_count": len(quality),
        "replayed_agent2_call_count": len(a2_calls), "dropped_agent2_call_count": len(dropped_a2),
        "agent2_call_correctness": total_correct / len(quality) if quality else None,
        "extra_or_wrong_agent2_calls": len(quality) - total_correct,
        "extra_recipient_warnings": extra_recipients,
        "dropped_calls": record.get("dropped_calls", {}),
        "parse_status": record.get("parse_status", {}),
        "stage_candidates": sorted(candidates),
        "failure_categories": sorted(categories),
        "evaluator_limitations": sorted(evaluator_limits),
        "critical_information_proxy": critical,
        "requires_manual_review": bool(candidates or categories),
        "interpretation": (
            "Candidates use scorer matches and literal/structured-call retention. "
            "Missing wording may be a valid paraphrase; retained values may be incidental, "
            "negated, or ambiguous. Agent 2 handling candidates do not establish model "
            "misunderstanding. Information loss and misunderstanding require manual "
            "review or a controlled replay. Task success is a required-write proxy; "
            "extra calls, extra recipients, and action order are not fully penalized. "
            "Agent 2 call correctness counts all normalized attempts, including dropped "
            "unknown/unavailable tools as incorrect; matching a required call does not "
            "prove that duplicate actions or extra argument values are appropriate. "
            "Critical-information metrics probe atomic literal values from every required "
            "write, not neutral controls; low-specificity ids and semantic paraphrases "
            "limit interpretation. Nonempty and alternative references are unassessable "
            "by this lexical metric. Evaluator limitations are audit flags, not proven "
            "evaluator errors or grounds to change saved scores."
        ),
    }
