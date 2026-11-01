#!/usr/bin/env python3
"""
Run 2 readout: the two things Asra asked for, from a shared_run.py output dir.

    python3 run2_report.py --shared-run runs/run2_primary

1. Controls, read against the HANDOFF column, not against 100.
   Agent 1 decides what reaches the handoff. The compressor can only lose
   what got that far, so "survived compression" is computed over the slots
   where the control was in the handoff.

2. Tool mismatch, subtracted. Some required calls have no slot in the tool
   Agent 1 named, so the value is gone before any compressor runs. Handoffs
   are shared across all six arms, so the affected slots are identical in
   every arm. Success is reported on all slots and on the mismatch-free
   slots, which is the clean paired comparison.

Reads saved records only. Never rescores, never calls a model.
analyze_redesign.py still owns the paired bootstrap and failure taxonomy.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import failure_stage  # noqa: E402

CONDITIONS = ("none", "sliding_window", "unconditional", "destination",
              "instruction_strip", "allowlist")
KEYS = ("C", "N1", "N2")


def _rows(run_dir, condition):
    path = Path(run_dir) / "arms" / condition / "runs.jsonl"
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


class _M:
    """Same shape baseline.py hands to failure_stage.summarise."""
    def __init__(self, d):
        self.task_id = d.get("task_id")
        self.required_tool = d.get("required_tool")
        self.named_tools = d.get("named_tools") or []
        self.unexpressible = d.get("unexpressible") or []
        self.status = d.get("status")


def _blocked(r):
    return sum(1 for d in (r.get("tool_mismatch") or [])
               if d.get("status") == failure_stage.UNEXPRESSIBLE)


def _blocked_hits(r):
    """Blocked required calls that Agent 2 completed anyway (from EVIDENCE).

    `missed` names each missed required call as "tool" or "tool:detail", and
    each mismatch entry is one required call, so per tool the blocked calls
    that were hit are the blocked ones beyond the number missed.
    """
    missed = {}
    for m in r.get("missed") or []:
        tool = str(m).split(":", 1)[0]
        missed[tool] = missed.get(tool, 0) + 1
    blocked = {}
    for d in r.get("tool_mismatch") or []:
        if d.get("status") == failure_stage.UNEXPRESSIBLE:
            blocked[d["required_tool"]] = blocked.get(d["required_tool"], 0) + 1
    return sum(max(0, n - missed.get(tool, 0)) for tool, n in blocked.items())


def _pct(num, den):
    return None if not den else num / den


def summarise_condition(rows, clean_slots):
    scored = [r for r in rows if not r.get("api_failure")
              and r.get("task_success") is not None]
    out = {"planned": len(rows), "scored": len(scored),
           "errors": sum(1 for r in rows if r.get("api_failure"))}
    out["success"] = _pct(sum(bool(r["task_success"]) for r in scored), len(scored))
    hit = sum(r.get("required_hit") or 0 for r in scored)
    tot = sum(r.get("required_total") or 0 for r in scored)
    blocked = sum(_blocked(r) for r in scored)
    out["call_completion"] = _pct(hit, tot)
    out["required_calls"] = tot
    out["mismatch_calls"] = blocked
    # Blocked calls were lost before any compressor ran, so they come out of
    # both sides: the denominator, and the hits on the few that Agent 2
    # recovered from EVIDENCE anyway.
    blocked_hit = sum(_blocked_hits(r) for r in scored)
    out["mismatch_calls_recovered"] = blocked_hit
    out["call_completion_ex_mismatch"] = _pct(hit - blocked_hit, tot - blocked)
    clean = [r for r in scored if (r["task_id"], r["repeat"]) in clean_slots]
    out["clean_scored"] = len(clean)
    out["success_clean"] = _pct(sum(bool(r["task_success"]) for r in clean), len(clean))

    tm = failure_stage.summarise([[_M(d) for d in (r.get("tool_mismatch") or [])]
                                  for r in scored])
    out["mismatch_tasks"] = tm["tasks_affected"]
    out["mismatch_fields"] = tm["fields_lost"]

    ctrl = {}
    for key in KEYS:
        planted = handoff = compressed = survived = measured = 0
        for r in scored:
            for c in r.get("controls") or []:
                if c.get("key") != key or not c.get("in_documents"):
                    continue
                planted += 1
                in_h = (c.get("in_handoff") or 0) > 0
                handoff += in_h
                if c.get("in_compressed") is not None:
                    measured += 1
                    in_c = c["in_compressed"] > 0
                    compressed += in_c
                    survived += in_h and in_c
        ctrl[key] = {"planted": planted,
                     "in_handoff": _pct(handoff, planted),
                     "in_compressed": _pct(compressed, measured),
                     "handoff_slots": handoff,
                     "survived_compression": _pct(survived, handoff)}
    out["controls"] = ctrl

    pre = [r["precompression_handoff_tokens"] for r in scored
           if isinstance(r.get("precompression_handoff_tokens"), (int, float))]
    post = [r["compressed_tokens"] for r in scored
            if isinstance(r.get("compressed_tokens"), (int, float))]
    out["handoff_tokens_median"] = statistics.median(pre) if pre else None
    out["compressed_tokens_median"] = statistics.median(post) if post else None
    return out


def build(run_dir):
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    by_cond = {c: _rows(run_dir, c) for c in CONDITIONS}
    base = [r for r in by_cond["none"] if not r.get("api_failure")]
    # Shared handoffs: a slot's mismatches come from Agent 1 and are the same
    # in every arm. Take them from the reference arm.
    clean_slots = {(r["task_id"], r["repeat"]) for r in base if _blocked(r) == 0}
    result = {"experiment_id": manifest["experiment_id"],
              "backend": manifest["backend"],
              "settings": {k: manifest["settings"][k] for k in
                           ("model", "temperature", "compressor_budget_tokens", "repeats")},
              "tasks": len(manifest["settings"]["task_ids"]),
              "clean_slots": len(clean_slots),
              "conditions": {c: summarise_condition(rows, clean_slots)
                             for c, rows in by_cond.items()}}
    return result


def _f(x):
    return "n/a" if x is None else f"{x:.0%}"


def markdown(res):
    c = res["conditions"]
    lines = [f"# Run 2 readout ({res['experiment_id']}, backend: {res['backend']})", "",
             f"{res['tasks']} tasks x {res['settings']['repeats']} repeats, "
             f"{res['settings']['model']}, temp {res['settings']['temperature']}, "
             f"budget {res['settings']['compressor_budget_tokens']}. "
             f"Shared Agent 1 handoff reused across all six arms.", "",
             "## Task success", "",
             "| Condition | Scored | Errors | Success | Success, mismatch-free slots | "
             "Call completion | Call completion ex mismatch | Median handoff -> compressed tokens |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond in CONDITIONS:
        s = c[cond]
        tok = (f"{s['handoff_tokens_median']:.0f} -> {s['compressed_tokens_median']:.0f}"
               if s["handoff_tokens_median"] is not None
               and s["compressed_tokens_median"] is not None else "n/a")
        lines.append(f"| {cond} | {s['scored']}/{s['planned']} | {s['errors']} | "
                     f"{_f(s['success'])} | {_f(s['success_clean'])} (n={s['clean_scored']}) | "
                     f"{_f(s['call_completion'])} | {_f(s['call_completion_ex_mismatch'])} | {tok} |")
    ref = c["none"]
    lines += ["", f"Tool mismatch: {ref['mismatch_calls']}/{ref['required_calls']} required "
              f"calls had no slot in the tool Agent 1 named (same slots in every arm). "
              f"Tasks: {', '.join(ref['mismatch_tasks']) or 'none'}. "
              f"Mismatch-free slots: {res['clean_slots']}.", "",
              "## Planted controls, read against the handoff", "",
              "'Survived compression' = of the slots where the control reached the "
              "handoff, how often it was still there after the compressor.", "",
              "| Condition | Control | Planted | In handoff | In compressed | Survived compression |",
              "|---|---|---:|---:|---:|---:|"]
    names = {"C": "C safety rule", "N1": "N1 inert fact", "N2": "N2 harmless instruction"}
    for cond in CONDITIONS:
        for k in KEYS:
            x = c[cond]["controls"][k]
            lines.append(f"| {cond} | {names[k]} | {x['planted']} | {_f(x['in_handoff'])} | "
                         f"{_f(x['in_compressed'])} | {_f(x['survived_compression'])} "
                         f"(of {x['handoff_slots']}) |")
    lines += ["", "Paired bootstrap intervals and the failure taxonomy are in "
              "analyze_redesign.py output (report.md, analysis.json)."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shared-run", required=True)
    ap.add_argument("--out", help="directory for run2_readout.md/.json (default: the run dir)")
    a = ap.parse_args(argv)
    res = build(a.shared_run)
    out = Path(a.out or a.shared_run)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run2_readout.json").write_text(json.dumps(res, indent=2))
    md = markdown(res)
    (out / "run2_readout.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
