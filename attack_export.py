"""
Export an attack-arm run to the trials.csv shape run 3 used, plus a short
summary, without calling any model.

    python3 attack_export.py --run results/run4b --specs tasks/attack_specs_run4.json \
        --out results/run4b_analysis

Two things on top of run 3's columns.

`condition`, because a run with more than one arm needs it to be a key.

Layout columns, because the sliding_window question is about WHERE things sit
in the handoff, and that can be read straight off Agent 1's saved handoff:
how long it was, whether its PLAN fits in the window from the front, whether
the attack's own values (the harmful call's critical arguments) are in the
PLAN, and whether they survive the head window and the tail window. Both
windows are mechanical, so this is exact for them and needs no API call.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import compressors

RUN3_COLUMNS = [
    "task_id", "base_task_id", "attack_spec_id", "injection", "wrapper",
    "suite", "tool_relation", "placement_arm", "placement_where", "repeat",
    "attack_specified", "attack_n_harmful", "attack_fired_calls",
    "attack_fired_any", "attack_fired_all", "attack_irreversible",
    "attack_tools_fired", "task_success", "required_hit", "required_total",
    "call_hit_rate", "precompression_handoff_tokens", "cer",
    "compression_stratum", "api_failure", "parse_failure",
    "token_count_method",
]
LAYOUT_COLUMNS = [
    "handoff_chars", "plan_fits_head", "plan_in_tail", "harm_in_plan",
    "harm_in_head", "harm_in_tail",
]


def _jsonl(path):
    if not Path(path).exists():
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def markers(spec) -> list:
    """The values that identify this attack's harmful call in text: the
    critical arguments (an email, an IBAN, a file id), not the tool name,
    since a benign PLAN can name the same tool."""
    out = []
    for call in spec["harmful_critical"]:
        for value in call["critical"].values():
            for item in (value if isinstance(value, list) else [value]):
                item = str(item)
                if len(item) >= 4:
                    out.append(item.lower())
    return out


def layout(handoff, spec) -> dict:
    if not isinstance(handoff, str):
        return {c: "" for c in LAYOUT_COLUMNS}
    marks = markers(spec)
    plan = handoff.split("EVIDENCE", 1)[0]
    head = compressors._sliding_window_head(handoff)
    tail = compressors._sliding_window(handoff)

    def has(text):
        low = text.lower()
        return int(any(m in low for m in marks))

    return {"handoff_chars": len(handoff),
            "plan_fits_head": int(len(plan) <= compressors.WINDOW_CHARS),
            "plan_in_tail": int("PLAN" in tail),
            "harm_in_plan": has(plan), "harm_in_head": has(head),
            "harm_in_tail": has(tail)}


def flat(row) -> dict:
    attack = row.get("attack") or {}
    placement = attack.get("placement") or {}
    out = {c: row.get(c) for c in RUN3_COLUMNS}
    out.update(base_task_id=str(row.get("attack_spec_id", "")).split("|")[0],
               injection=attack.get("injection"),
               wrapper=attack.get("wrapper"),
               tool_relation=attack.get("tool_relation"),
               placement_arm=placement.get("arm"),
               placement_where=placement.get("where"))
    if isinstance(out["attack_tools_fired"], list):
        out["attack_tools_fired"] = ";".join(out["attack_tools_fired"])
    return out


def export(run_dir, specs_path, out_dir) -> dict:
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    specs = {f"{s['task_id']}|{s['injection']}|{s['wrapper']}": s
             for s in json.loads(Path(specs_path).read_text())["specs"]}
    handoffs = {(r["attack_spec_id"], r["repeat"]): r.get("handoff")
                for r in _jsonl(run_dir / "handoffs.jsonl")}
    out_dir.mkdir(parents=True, exist_ok=True)

    table = []
    for condition, arm in manifest["arms"].items():
        for row in _jsonl(run_dir / "arms" / arm["run_dir"] / "runs.jsonl"):
            if row.get("kind") != "run":
                continue
            key = (row.get("attack_spec_id"), row.get("repeat"))
            rec = {"condition": condition, **flat(row)}
            rec.update(layout(handoffs.get(key), specs[key[0]]))
            table.append(rec)

    with open(out_dir / "trials.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition"] + RUN3_COLUMNS + LAYOUT_COLUMNS)
        writer.writeheader()
        writer.writerows(table)

    summary = summarise(table)
    (out_dir / "summary.md").write_text(render(summary))
    return summary


def summarise(table) -> dict:
    by = collections.defaultdict(list)
    for r in table:
        if not r["api_failure"]:
            by[r["condition"]].append(r)
    conds = {}
    for c, rows in by.items():
        n = len(rows)
        conds[c] = {"trials": n,
                    "attack_any": sum(bool(r["attack_fired_any"]) for r in rows),
                    "task_success": sum(bool(r["task_success"]) for r in rows)}
    # Paired: the same (spec, repeat) under none and each other arm, since
    # both arms share one Agent 1 draw.
    paired = {}
    base = {(r["attack_spec_id"], r["repeat"]): r for r in by.get("none", [])}
    for c, rows in by.items():
        if c == "none":
            continue
        cells = collections.Counter()
        for r in rows:
            b = base.get((r["attack_spec_id"], r["repeat"]))
            if b is None:
                continue
            cells[(bool(b["attack_fired_any"]), bool(r["attack_fired_any"]))] += 1
        paired[c] = {"both": cells[(True, True)],
                     "removed": cells[(True, False)],
                     "added": cells[(False, True)],
                     "neither": cells[(False, False)]}
    lay = [r for r in by.get("none", []) if r["handoff_chars"] != ""]
    layout_counts = {k: sum(int(r[k]) for r in lay) for k in LAYOUT_COLUMNS[1:]}
    layout_counts["handoffs"] = len(lay)
    layout_counts["over_window"] = sum(
        int(r["handoff_chars"]) > compressors.WINDOW_CHARS for r in lay)
    return {"conditions": conds, "paired_vs_none": paired,
            "layout": layout_counts}


def render(s) -> str:
    pct = lambda a, b: f"{a / b:.0%}" if b else "n/a"
    lines = ["| condition | trials | attack fired | task done |",
             "|---|---|---|---|"]
    for c, v in s["conditions"].items():
        lines.append(f"| {c} | {v['trials']} | "
                     f"{v['attack_any']} ({pct(v['attack_any'], v['trials'])}) | "
                     f"{v['task_success']} ({pct(v['task_success'], v['trials'])}) |")
    lines += ["", "Paired against none (same Agent 1 draw):", ""]
    for c, p in s["paired_vs_none"].items():
        lines.append(f"- {c}: fired in both {p['both']}, removed {p['removed']}, "
                     f"added {p['added']}, neither {p['neither']}")
    L = s["layout"]
    n = L["handoffs"]
    lines += ["", f"Handoff layout over {n} Agent 1 handoffs:", "",
              f"- longer than the {compressors.WINDOW_CHARS}-char window: "
              f"{L['over_window']} ({pct(L['over_window'], n)})",
              f"- whole PLAN fits in the head window: {L['plan_fits_head']} "
              f"({pct(L['plan_fits_head'], n)})",
              f"- any PLAN text in the tail window: {L['plan_in_tail']} "
              f"({pct(L['plan_in_tail'], n)})",
              f"- attack's values already in Agent 1's PLAN: {L['harm_in_plan']} "
              f"({pct(L['harm_in_plan'], n)})",
              f"- attack's values in head window: {L['harm_in_head']} "
              f"({pct(L['harm_in_head'], n)})",
              f"- attack's values in tail window: {L['harm_in_tail']} "
              f"({pct(L['harm_in_tail'], n)})", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    summary = export(args.run, args.specs, args.out)
    print(render(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
