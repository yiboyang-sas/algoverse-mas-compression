#!/usr/bin/env python3
"""Analyze provenance-validated Run 2 records without calling any model.

Usage: python scripts/analyze_run2.py --run-dir runs/full --out-dir analysis/full
The primary utility is the existing scorer's required-write success proxy.
Repeated tasks are clusters: the bootstrap resamples task ids with every repeat.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressors import ALL_CONDITIONS
from environment import ToolLog, score_run
from scripts.run2_diagnostics import diagnose


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    i = (len(values) - 1) * fraction
    lo = int(i)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (i - lo)


def cluster_interval(task_means, samples=5000, seed=20261021):
    """Percentile interval; each input is one task's mean over all repeats."""
    if not task_means:
        return None, None
    rng, n = random.Random(seed), len(task_means)
    boot = [sum(task_means[rng.randrange(n)] for _ in range(n)) / n
            for _ in range(samples)]
    return percentile(boot, .025), percentile(boot, .975)


def load_run(run_dir):
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    journal = next((run_dir / n for n in ("raw.jsonl", "records.jsonl")
                    if (run_dir / n).exists()), None)
    if journal:
        records = []
        for number, line in enumerate(journal.read_text().splitlines(), 1):
            if not line.strip():
                raise ValueError(f"empty journal row {number}; recover with the runner first")
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"malformed journal row {number}; recover with the runner first") from exc
    elif (run_dir / "results.json").exists():
        records = json.loads((run_dir / "results.json").read_text())
    else:
        records = []
    status = json.loads((run_dir / "status.json").read_text()) if (run_dir / "status.json").exists() else {}
    return manifest, records, status


def validate(manifest, records, tasks, tasks_bytes=None):
    identity = manifest.get("identity", manifest)
    if manifest.get("manifest_sha256"):
        unhashed = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if canonical_sha(unhashed) != manifest["manifest_sha256"]:
            raise ValueError("manifest checksum mismatch")
    if manifest.get("config_fingerprint") and canonical_sha(identity) != manifest["config_fingerprint"]:
        raise ValueError("identity fingerprint mismatch")
    config = identity.get("configuration", {})
    ids = identity.get("task_ids", [t["id"] for t in tasks])
    conditions = identity.get("conditions", ALL_CONDITIONS)
    repeats = identity.get("repeats", 3)
    by_task = {t["id"]: t for t in tasks}
    if len(by_task) != len(tasks) or len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    if set(ids) - set(by_task):
        raise ValueError("manifest tasks absent from supplied dataset")
    for tid in ids:
        t = by_task[tid]
        if t.get("harmful_call") or t.get("payload_scope") or "INJECT_CALL:" in json.dumps(t):
            raise ValueError("attack-bearing task in Run 2 analysis")
        task_hash = identity.get("task_sha256", {}).get(tid)
        if task_hash and canonical_sha(t) != task_hash:
            raise ValueError(f"task content differs from manifest: {tid}")
    if len(conditions) != len(set(conditions)) or set(conditions) - set(ALL_CONDITIONS):
        raise ValueError("invalid conditions")
    if repeats < 1:
        raise ValueError("invalid repeat count")
    scorer_hash = identity.get("code", {}).get("files", {}).get("environment.py")
    if scorer_hash and hashlib.sha256((ROOT / "environment.py").read_bytes()).hexdigest() != scorer_hash:
        raise ValueError("current scorer source differs from run manifest; use original committed scorer")
    if tasks_bytes is not None and identity.get("dataset_sha256"):
        if hashlib.sha256(tasks_bytes).hexdigest() != identity["dataset_sha256"]:
            raise ValueError("dataset bytes differ from manifest; use frozen run dataset")
    observed = set()
    fields = ("manifest_sha256", "config_fingerprint", "model", "temperature", "delegate", "backend", "code_commit")
    for field in fields:
        values = {json.dumps(r.get(field), sort_keys=True) for r in records}
        if len(values) > 1:
            raise ValueError(f"mixed provenance: {field}")
    for record in records:
        key = (record.get("task_id"), record.get("condition"), record.get("repeat"))
        if key in observed:
            raise ValueError(f"duplicate trial: {key}")
        observed.add(key)
        if key[0] not in ids or key[1] not in conditions or key[2] not in range(repeats):
            raise ValueError(f"unexpected trial: {key}")
        if record.get("run_id") and record["run_id"] != manifest.get("run_id"):
            raise ValueError("record run id differs from manifest")
        code_commit = identity.get("code", {}).get("commit")
        if code_commit and record.get("code_commit") != code_commit:
            raise ValueError("record code commit differs from manifest")
        for field in ("manifest_sha256", "config_fingerprint"):
            if manifest.get(field) and record.get(field) != manifest[field]:
                raise ValueError(f"record {field} differs from manifest")
        for field in ("backend", "model", "temperature", "delegate"):
            if field in config and record.get(field) != config[field]:
                raise ValueError(f"record {field} differs from configuration")
        if record.get("configuration") is not None and record["configuration"] != config:
            raise ValueError("record configuration differs from manifest")
        if identity.get("prompt_version") is not None and record.get("prompt_version") != identity["prompt_version"]:
            raise ValueError("record prompt version differs from manifest")
        if "mock" in record and bool(record["mock"]) != (config.get("backend") == "mock"):
            raise ValueError("record mock marker differs from configured backend")
        expected_task_hash = identity.get("task_sha256", {}).get(key[0])
        if expected_task_hash and record.get("task_sha256") != expected_task_hash:
            raise ValueError(f"record task hash differs: {key[0]}")
        task = by_task[key[0]]
        if task.get("harmful_call") or task.get("payload_scope"):
            raise ValueError("attack-bearing task in Run 2 analysis")
        if record.get("attack_fired") or record.get("payload_scope"):
            raise ValueError("attack-bearing record in Run 2 analysis")
        if type(record.get("task_success")) is not bool:
            raise ValueError("task_success must be a JSON boolean")
        for field in ("writes_hit", "writes_total"):
            if type(record.get(field)) is not int:
                raise ValueError(f"{field} must be an integer")
        rescored = score_run(ToolLog(calls=record.get("calls", [])), task)
        for field in ("task_success", "writes_hit", "writes_total"):
            if field not in record or record[field] != rescored[field]:
                raise ValueError(f"saved score differs from current scorer: {key}, {field}")
    return identity, by_task, ids, conditions, repeats


def analyze(manifest, records, tasks, samples=5000, tasks_bytes=None):
    identity, by_task, ids, conditions, repeats = validate(manifest, records, tasks, tasks_bytes)
    slots = {(r["task_id"], r["condition"], r["repeat"]): r for r in records}
    all_conditions = list(ALL_CONDITIONS)
    summaries, per_task, pairs, failures = [], [], [], []
    for record in records:
        diagnostic = diagnose(record, by_task[record["task_id"]])
        per_task.append({"run_id": record.get("run_id", manifest.get("run_id")),
                         "task_id": record["task_id"], "condition": record["condition"],
                         "repeat": record["repeat"], "task_success": record["task_success"],
                         "writes_hit": record["writes_hit"], "writes_total": record["writes_total"],
                         "correct_agent2_calls": diagnostic["correct_agent2_calls"],
                         "agent2_call_count": diagnostic["agent2_call_count"],
                         "dropped_agent2_call_count": diagnostic["dropped_agent2_call_count"],
                         "agent2_call_correctness": diagnostic["agent2_call_correctness"],
                         "extra_or_wrong_agent2_calls": diagnostic["extra_or_wrong_agent2_calls"],
                         "stage_candidates": ";".join(diagnostic["stage_candidates"]),
                         "failure_categories": ";".join(diagnostic["failure_categories"]),
                         "evaluator_limitations": ";".join(diagnostic["evaluator_limitations"]),
                         **diagnostic["critical_information_proxy"],
                         "token_counts": json.dumps(record.get("token_counts", {}), sort_keys=True)})
        if diagnostic["requires_manual_review"] or not record["task_success"]:
            failures.append({"task_id": record["task_id"], "condition": record["condition"],
                             "repeat": record["repeat"], "task_success": record["task_success"],
                             **diagnostic})

    for ci, condition in enumerate(all_conditions):
        rs = [r for r in records if r["condition"] == condition]
        expected = len(ids) * repeats if condition in conditions else 0
        complete = expected > 0 and len(rs) == expected
        status = "complete" if complete else "partial" if rs else "not run"
        rate = statistics.mean(r["task_success"] for r in rs) if rs else None
        by_repeat = []
        for repeat in range(repeats):
            rr = [r for r in rs if r["repeat"] == repeat]
            by_repeat.append({"repeat": repeat, "n": len(rr), "successes": sum(r["task_success"] for r in rr),
                              "rate": statistics.mean(r["task_success"] for r in rr) if rr else None})
        repeat_rates = [r["rate"] for r in by_repeat if r["rate"] is not None]
        task_means = [statistics.mean(slots[(tid, condition, rep)]["task_success"] for rep in range(repeats))
                      for tid in ids] if complete else []
        low, high = cluster_interval(task_means, samples, 20261021 + ci)
        def mean_text_tokens(field):
            values = [r.get("token_counts", {}).get(field, {}).get("tokens") for r in rs]
            return statistics.mean(values) if values and all(isinstance(v, int) for v in values) else None
        handoff_tokens = mean_text_tokens("handoff")
        compressed_tokens = mean_text_tokens("compressed")
        provider_totals = {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [r.get("provider_usage", {}).get(field) for r in rs]
            provider_totals["provider_" + field] = sum(values) if values and all(isinstance(v, int) for v in values) else None
        call_rows = [r for r in per_task if r["condition"] == condition]
        correct_calls = sum(r["correct_agent2_calls"] for r in call_rows)
        attempted_calls = sum(r["agent2_call_count"] for r in call_rows)
        literal_totals = {key: sum(r[key] for r in call_rows) for key in (
            "critical_literal_units", "handoff_literal_units", "compressed_literal_units",
            "retained_handoff_literal_units", "lost_handoff_literal_units", "low_specificity_literal_units",
            "unassessable_reference_arguments", "structured_required_calls_handoff",
            "structured_required_calls_compressed", "required_write_count")}
        denominator = literal_totals["critical_literal_units"]
        lexical_before = literal_totals["handoff_literal_units"]
        lexical_summary = {
            **literal_totals,
            "handoff_literal_recall": literal_totals["handoff_literal_units"] / denominator if denominator else None,
            "compressed_literal_recall": literal_totals["compressed_literal_units"] / denominator if denominator else None,
            "conditional_literal_retention": literal_totals["retained_handoff_literal_units"] / lexical_before if lexical_before else None,
        }
        paired_ids = [tid for tid in ids if all((tid, c, rep) in slots
                      for c in ("none", condition) for rep in range(repeats))]
        differences, wins, losses, ties = [], 0, 0, 0
        if condition != "none":
            for tid in paired_ids:
                ds = []
                for rep in range(repeats):
                    a = int(slots[(tid, condition, rep)]["task_success"])
                    b = int(slots[(tid, "none", rep)]["task_success"])
                    ds.append(a - b)
                    wins += a > b
                    losses += a < b
                    ties += a == b
                difference = statistics.mean(ds)
                differences.append(difference)
                pairs.append({"task_id": tid, "condition": condition, "paired_repeats": repeats,
                              "baseline_successes": sum(slots[(tid, "none", r)]["task_success"] for r in range(repeats)),
                              "condition_successes": sum(slots[(tid, condition, r)]["task_success"] for r in range(repeats)),
                              "delta_pp": 100 * difference,
                              "trial_differences": json.dumps(ds)})
        delta = statistics.mean(differences) if differences else (0 if condition == "none" and complete else None)
        dlo, dhi = cluster_interval(differences, samples, 20262021 + ci)
        summaries.append({"condition": condition, "status": status, "n": len(rs), "expected_n": expected,
                          "successes": sum(r["task_success"] for r in rs),
                          "success_pct": rate * 100 if rate is not None else None,
                          "success_ci95_low_pct": low * 100 if low is not None else None,
                          "success_ci95_high_pct": high * 100 if high is not None else None,
                          "success_ci95_degenerate": low == high if low is not None else None,
                          "complete_task_clusters": len(task_means),
                          "repeat_rates": json.dumps(by_repeat, sort_keys=True),
                          "repeat_sd_pp": statistics.stdev(repeat_rates) * 100 if complete and len(repeat_rates) > 1 else None,
                          "repeat_min_pct": min(repeat_rates) * 100 if repeat_rates else None,
                          "repeat_max_pct": max(repeat_rates) * 100 if repeat_rates else None,
                          "paired_task_clusters": len(paired_ids) if condition != "none" else 0,
                          "paired_delta_pp": delta * 100 if delta is not None else None,
                          "paired_delta_ci95_low_pp": dlo * 100 if dlo is not None else None,
                          "paired_delta_ci95_high_pp": dhi * 100 if dhi is not None else None,
                          "paired_delta_ci95_degenerate": dlo == dhi if dlo is not None else None,
                          "paired_trial_wins": wins, "paired_trial_losses": losses, "paired_trial_ties": ties,
                          "mean_handoff_tokens": handoff_tokens, "mean_compressed_tokens": compressed_tokens,
                          "compression_retained_token_ratio": compressed_tokens / handoff_tokens if handoff_tokens else None,
                          "agent2_correct_call_attempts": correct_calls, "agent2_call_attempts": attempted_calls,
                          "agent2_call_correctness_pct": 100 * correct_calls / attempted_calls if attempted_calls else None,
                          "evaluator_limitation_flags": json.dumps(dict(Counter(
                              flag for row in call_rows for flag in row["evaluator_limitations"].split(";") if flag)), sort_keys=True),
                          **lexical_summary,
                          **provider_totals})
    return {"summary": summaries, "per_task": per_task, "paired_tasks": pairs, "failures": failures,
            "run_id": manifest.get("run_id"), "configuration": identity.get("configuration", {}),
            "config_fingerprint": manifest.get("config_fingerprint"),
            "complete": len(slots) == len(ids) * len(conditions) * repeats,
            "n_records": len(records), "expected_records": len(ids) * len(conditions) * repeats,
            "task_ids": ids, "repeats": repeats, "conditions": conditions,
            "mock": identity.get("configuration", {}).get("backend") == "mock" or any(r.get("mock") for r in records),
            "bootstrap_samples": samples,
            "stage_candidates": dict(Counter(c for f in failures for c in f["stage_candidates"])),
            "evaluator_limitations": dict(Counter(flag for row in per_task
                for flag in row["evaluator_limitations"].split(";") if flag)),
            "failure_categories": dict(Counter(c for f in failures for c in f["failure_categories"]))}


def _csv(path, rows, empty_fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else empty_fields)
        writer.writeheader()
        writer.writerows(rows)


def table_markdown(result, manifest, status):
    def fmt(value):
        return "—" if value is None else f"{value:.1f}"
    lines = ["# Run 2 results", "", f"Run: `{manifest.get('run_id', 'unknown')}`. "
             f"Completed records: {result['n_records']}/{result['expected_records']}."]
    if result["mock"]:
        lines += ["", "**MOCK INFRASTRUCTURE CHECK — no scientific API results.**"]
    if not result["n_records"]:
        lines += ["", "**No measured results. All success rates and changes remain unverified.**"]
    if not result["complete"]:
        lines += ["", "Incomplete run: observed rates are partial; condition intervals require all planned task/repeat slots. "
                  "Paired changes use only tasks with all repeats in both arms."]
    lines += ["", "| Condition | Status | Successes / trials | Required-write success (%) | 95% task-cluster interval (%) | Δ vs baseline (pp) | Paired 95% interval (pp) | Repeat SD (pp) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        ci = "—" if row["success_ci95_low_pct"] is None else f"[{fmt(row['success_ci95_low_pct'])}, {fmt(row['success_ci95_high_pct'])}]"
        dci = "—" if row["paired_delta_ci95_low_pp"] is None else f"[{fmt(row['paired_delta_ci95_low_pp'])}, {fmt(row['paired_delta_ci95_high_pp'])}]"
        count = f"{row['successes']} / {row['n']}" if row["n"] else "—"
        lines.append(f"| {row['condition']} | {row['status']} | {count} | {fmt(row['success_pct'])} | {ci} | {fmt(row['paired_delta_pp'])} | {dci} | {fmt(row['repeat_sd_pp'])} |")
    lines += ["", "The current scorer counts required writes, with its existing argument matching; it is a legitimate-task "
              "success proxy, not full AgentDojo execution utility. Reads, action order, extra calls, extra recipients, "
              "and content omitted from the ground truth may not affect success. Per-call checks and failure evidence "
              "are in per_task.csv and failures.jsonl.", "",
              f"Intervals are deterministic percentile bootstraps ({result['bootstrap_samples']} resamples) over task "
              "clusters, preserving every repeat and both paired arms. They describe variation over this small task "
              "set, not 54 independent tasks or population-wide proof. Repeats are independent stochastic requests "
              "matched by task and repeat index; they do not share an Agent 1 handoff or a common random seed. "
              "When all observed task means (or paired differences) coincide, the empirical bootstrap interval "
              "collapses to a point; this is boundary degeneracy, not evidence of zero uncertainty or a guaranteed "
              "true success rate. Such intervals are flagged as degenerate in summary.csv. "
              "Do not interpret intervals as strong significance evidence with 18 tasks or generalize beyond this suite.", "",
              "Stage categories are conservative candidates based on literal/structured-call retention. Manual semantic "
              "review is required to distinguish information loss from paraphrase or instruction misunderstanding. "
              "No causal failure attribution is claimed without controlled replay.", "",
              "Critical-information retention is a lexical proxy over atomic argument values from every required "
              "write (including intermediate writes). Handoff/compressed recall divides matched literal units by "
              "reference units; conditional retention divides units present at both stages by units present in the "
              "handoff. Counts are pooled across task/repeat records. Short ids can match incidentally; paraphrases "
              "can score as absent. Nonempty/alternative references are marked unassessable. These metrics do not "
              "measure neutral-control retention or prove semantic information loss. Evaluator limitation flags "
              "identify weak references and possible semantic mismatch for review, without altering scores."]
    if status:
        lines += ["", "Runner status:", "", "```json", json.dumps(status, indent=2, sort_keys=True), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks/sample_tasks.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be >= 100")
    manifest, records, status = load_run(args.run_dir)
    task_bytes = args.tasks.read_bytes()
    result = analyze(manifest, records, json.loads(task_bytes), args.bootstrap_samples, task_bytes)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _csv(args.out_dir / "summary.csv", result["summary"], ["condition", "status", "n"])
    _csv(args.out_dir / "per_task.csv", result["per_task"], ["task_id", "condition", "repeat", "task_success"])
    _csv(args.out_dir / "paired_tasks.csv", result["paired_tasks"], ["task_id", "condition", "delta_pp"])
    (args.out_dir / "failures.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in result["failures"]))
    (args.out_dir / "analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "results_table.md").write_text(table_markdown(result, manifest, status))
    print(f"Analyzed {len(records)} records; complete={result['complete']}; mock={result['mock']}")
    print(args.out_dir / "results_table.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
