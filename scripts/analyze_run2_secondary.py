#!/usr/bin/env python3
"""Design-aware secondary analysis; never changes primary scores or artifacts.

python scripts/analyze_run2_secondary.py --run-dir runs/run2 --out-dir analysis/secondary
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compressors import ALL_CONDITIONS
from scripts.analyze_run2 import analyze, cluster_interval, load_run, validate
from secondary_evaluation import canonical_sha, evaluate_secondary, load_references
from telemetry import usage_totals


def _load_cache(run_dir):
    path = Path(run_dir) / "handoffs.jsonl"
    if not path.exists():
        return []
    values = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            value = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"malformed shared handoff cache row {number}; recover with runner first") from exc
        values.append(value)
    return values


def validate_design(manifest, records, cache):
    """Bind each shared arm to exactly one task/repeat upstream draw."""
    identity = manifest.get("identity", manifest)
    design = identity.get("design", "independent")
    if design not in {"independent", "shared-handoff"}:
        raise ValueError("unknown experiment design")
    if any(record.get("design", "independent") != design for record in records):
        raise ValueError("mixed experiment designs")
    if design == "independent":
        if cache or any(any(record.get(k) for k in ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256")) for record in records):
            raise ValueError("shared upstream metadata in independent design")
        return design
    by_key, ids, digests, prepared_digests = {}, {}, {}, {}
    from scripts.run2 import validate_shared_handoff, validate_shared_links
    for envelope in cache:
        key = (envelope.get("task_id"), envelope.get("repeat"))
        if key in by_key:
            raise ValueError("duplicate shared handoff task/repeat")
        if key[0] not in identity.get("task_ids", []) or type(key[1]) is not int or key[1] not in range(identity.get("repeats", 3)):
            raise ValueError("unexpected shared handoff task/repeat")
        digest = canonical_sha({k: v for k, v in envelope.items() if k != "shared_handoff_sha256"})
        if envelope.get("shared_handoff_sha256") != digest:
            raise ValueError("shared handoff envelope checksum mismatch")
        prepared = envelope.get("prepared_agent1", {})
        if envelope.get("prepared_agent1_sha256") != canonical_sha(prepared):
            raise ValueError("shared prepared Agent 1 checksum mismatch")
        for field in ("run_id", "manifest_sha256", "config_fingerprint"):
            if envelope.get(field) != manifest.get(field):
                raise ValueError(f"shared cache {field} mismatch")
        task_hash = identity.get("task_sha256", {}).get(key[0])
        if envelope.get("task_sha256") != task_hash or prepared.get("task_sha256") != task_hash or prepared.get("task_id") != key[0]:
            raise ValueError("shared upstream task identity mismatch")
        for field, index in (("shared_handoff_id", ids), ("shared_handoff_sha256", digests), ("prepared_agent1_sha256", prepared_digests)):
            if not envelope.get(field) or envelope[field] in index:
                raise ValueError("shared id/hash collision across task/repeat draws")
            index[envelope[field]] = key
        upstream = prepared.get("model_requests", [])
        if len(upstream) != 1 or upstream[0].get("stage") != "agent1" or upstream[0].get("reused"):
            raise ValueError("invalid cached Agent 1 request trace")
        if prepared.get("provider_usage") != usage_totals(upstream):
            raise ValueError("cached Agent 1 usage differs from trace")
        # The runner additionally reconstructs canonical prompts, parsing,
        # reads and delegation from raw output; rehashed edits must not pass.
        validate_shared_handoff(envelope, manifest)
        by_key[key] = envelope
    for record in records:
        key = (record["task_id"], record["repeat"])
        if key not in by_key:
            raise ValueError("shared arm has no matching upstream cache")
        envelope = by_key[key]
        prepared = envelope["prepared_agent1"]
        for field in ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256"):
            if record.get(field) != envelope[field]:
                raise ValueError("shared arm upstream id/hash mismatch")
        for field in ("handoff", "agent1_input", "agent1_raw"):
            if record.get("text", {}).get(field) != prepared.get(field):
                raise ValueError(f"shared arm {field} differs from cache")
        if record.get("stage_calls", {}).get("agent1") != prepared.get("calls") or record.get("dropped_calls", {}).get("agent1") != prepared.get("dropped_calls"):
            raise ValueError("shared arm Agent 1 tool records differ from cache")
        upstream = [r for r in record.get("model_requests", []) if r.get("stage") == "agent1"]
        if len(upstream) != 1 or not upstream[0].get("reused"):
            raise ValueError("shared arm must mark its sole Agent 1 trace reused")
        if {k: v for k, v in upstream[0].items() if k != "reused"} != prepared["model_requests"][0]:
            raise ValueError("shared arm Agent 1 trace differs from cache")
        if record.get("usage_scope") != "condition_requests_excluding_shared_agent1":
            raise ValueError("ambiguous shared arm usage accounting")
        new_requests = [r for r in record["model_requests"] if not r.get("reused")]
        if record.get("provider_usage") != usage_totals(new_requests):
            raise ValueError("shared arm usage double-counts Agent 1 or differs from trace")
    validate_shared_links({(r["task_id"], r["condition"], r["repeat"]): r for r in records}, by_key, manifest)
    return design


def _sum_usage(rows):
    result = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [row.get(key) for row in rows]
        result[key] = sum(values) if values and all(type(v) is int for v in values) else None
    return result


def usage_report(design, records, cache):
    arms = _sum_usage([r.get("provider_usage", {}) for r in records])
    upstream = _sum_usage([r["prepared_agent1"].get("provider_usage", {}) for r in cache]) if design == "shared-handoff" else None
    combined = {key: (arms[key] + upstream[key] if arms[key] is not None and upstream[key] is not None else None)
                for key in arms} if upstream is not None else arms
    return {"design": design, "completed_arm_usage": arms,
            "unique_cached_agent1_draws": len(cache) if design == "shared-handoff" else 0,
            "unique_cached_agent1_usage_counted_once": upstream,
            "completed_arms_plus_unique_cached_agent1_usage": combined,
            "scope": "Completed records and successful upstream cache draws only. Reused Agent 1 traces are excluded from arm totals and added once from handoffs.jsonl. Failed/in-flight requests and error journals are excluded; consult the budget ledger for spending."}


def analyze_secondary(manifest, records, tasks, policy, cache=None, samples=5000, tasks_bytes=None):
    # These are the frozen, unchanged primary validation and mathematics.
    identity, by_task, ids, conditions, repeats = validate(manifest, records, tasks, tasks_bytes)
    design = validate_design(manifest, records, cache or [])
    primary = analyze(manifest, records, tasks, samples=samples, tasks_bytes=tasks_bytes)
    covered = [tid for tid in ids if not policy["tasks"][tid]["manual_review_requirements"]]
    details, per_task = [], []
    for record in records:
        evaluated = evaluate_secondary(record, by_task[record["task_id"]], policy)
        evaluated.update(condition=record["condition"], repeat=record["repeat"], run_id=record.get("run_id"), design=design)
        details.append(evaluated)
        per_task.append({key: evaluated[key] for key in (
            "run_id", "design", "task_id", "condition", "repeat", "primary_task_success",
            "deterministic_checks_pass", "fully_automatically_referenced", "strict_reference_conformity")})
        per_task[-1].update(problems=";".join(evaluated["problems"]),
                            manual_review_requirements=";".join(evaluated["manual_review_requirements"]))
    slots = {(r["task_id"], r["condition"], r["repeat"]): r for r in details}
    primary_rows = {r["condition"]: r for r in primary["summary"]}
    summaries = []
    for ci, condition in enumerate(ALL_CONDITIONS):
        rows = [r for r in details if r["condition"] == condition]
        strict_rows = [r for r in rows if r["fully_automatically_referenced"]]
        expected = len(covered) * repeats if condition in conditions else 0
        complete = expected > 0 and len(strict_rows) == expected
        rate = statistics.mean(r["strict_reference_conformity"] for r in strict_rows) if strict_rows else None
        primary_covered = statistics.mean(r["primary_task_success"] for r in strict_rows) if strict_rows else None
        task_means = [statistics.mean(slots[(tid, condition, rep)]["strict_reference_conformity"] for rep in range(repeats)) for tid in covered] if complete else []
        low, high = cluster_interval(task_means, samples, 20261101 + ci)
        paired = [tid for tid in covered if all((tid, arm, rep) in slots for arm in ("none", condition) for rep in range(repeats))]
        differences = [statistics.mean(int(slots[(tid, condition, rep)]["strict_reference_conformity"]) - int(slots[(tid, "none", rep)]["strict_reference_conformity"]) for rep in range(repeats)) for tid in paired] if condition != "none" else []
        delta = statistics.mean(differences) if differences else (0 if condition == "none" and complete else None)
        dlo, dhi = cluster_interval(differences, samples, 20261201 + ci)
        def pct(value):
            return 100 * value if value is not None else None
        summaries.append({"condition": condition, "design": design,
            "status": "complete reference subset" if complete else "partial" if rows else "not run",
            "primary_all_task_success_pct": primary_rows[condition]["success_pct"],
            "primary_all_delta_vs_baseline_pp": primary_rows[condition]["paired_delta_pp"],
            "primary_all_ci95_low_pct": primary_rows[condition]["success_ci95_low_pct"],
            "primary_all_ci95_high_pct": primary_rows[condition]["success_ci95_high_pct"],
            "primary_all_delta_ci95_low_pp": primary_rows[condition]["paired_delta_ci95_low_pp"],
            "primary_all_delta_ci95_high_pp": primary_rows[condition]["paired_delta_ci95_high_pp"],
            "all_observed_trials": len(rows), "reference_covered_task_ids": json.dumps(covered),
            "reference_covered_tasks": len(covered), "selected_tasks": len(ids),
            "covered_observed_trials": len(strict_rows), "covered_expected_trials": expected,
            "primary_covered_success_pct": pct(primary_covered),
            "strict_reference_conformity_pct": pct(rate),
            "strict_minus_primary_on_same_subset_pp": pct(rate-primary_covered) if rate is not None else None,
            "strict_ci95_low_pct": pct(low), "strict_ci95_high_pct": pct(high),
            "strict_ci95_degenerate": low == high if low is not None else None,
            "strict_paired_task_clusters": len(paired) if condition != "none" else 0,
            "strict_delta_vs_baseline_pp": pct(delta),
            "strict_delta_ci95_low_pp": pct(dlo), "strict_delta_ci95_high_pp": pct(dhi),
            "manual_review_trials": len(rows)-len(strict_rows),
            "deterministic_partial_checks_passed": sum(r["deterministic_checks_pass"] for r in rows)})
    return {"run_id": manifest.get("run_id"), "design": design, "mock": primary["mock"],
            "policy_id": policy["policy_id"], "references_sha256": policy.get("references_sha256"),
            "selected_task_ids": ids, "fully_referenced_task_ids": covered,
            "manual_review_task_ids": [tid for tid in ids if tid not in covered],
            "n_records": len(records), "repeats": repeats, "summary": summaries,
            "per_task": per_task, "details": details, "usage": usage_report(design, records, cache or []),
            "metric": "Strict reference conformity on a declared subset, not complete legitimate-task utility.",
            "primary_artifacts_unchanged": True, "bootstrap_samples": samples}


def _csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else fields)
        writer.writeheader()
        writer.writerows(rows)


def markdown(result, provenance):
    def number(value):
        return "—" if value is None else f"{value:.1f}"
    design_text = ("Agent 1 is sampled independently for each condition/task/repeat. Pairing is by task and repeat, so differences include upstream variation." if result["design"] == "independent" else
                   "The same cached Agent 1 draw is reused within every condition for a task/repeat. Each repeat has its own upstream draw. Compression and Agent 2 are sampled independently. This is a separately labeled design, not pooled with independent-handoff runs.")
    lines = ["# Run 2 secondary evaluation", "", f"Run `{result['run_id']}` · design `{result['design']}` · policy `{result['policy_id']}`.", "", design_text, ""]
    if result["mock"]:
        lines += ["**MOCK INFRASTRUCTURE CHECK — no scientific model results.**", ""]
    if not result["n_records"]:
        lines += ["**No measured results. Every rate and difference below is unverified.**", ""]
    lines += [f"Fully referenced subset: {len(result['fully_referenced_task_ids'])}/{len(result['selected_task_ids'])} selected tasks. "
              "Strict reference conformity is evaluated only on this predeclared subset. The all-task primary rate and covered-subset primary rate have different denominators.", "",
              "| Condition | Status | Primary all tasks (%) | Primary same subset (%) | Strict reference conformity (%) | Strict 95% interval (%) | Strict Δ baseline (pp) | Covered trials |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["summary"]:
        ci = "—" if row["strict_ci95_low_pct"] is None else f"[{number(row['strict_ci95_low_pct'])}, {number(row['strict_ci95_high_pct'])}]"
        lines.append(f"| {row['condition']} | {row['status']} | {number(row['primary_all_task_success_pct'])} | {number(row['primary_covered_success_pct'])} | {number(row['strict_reference_conformity_pct'])} | {ci} | {number(row['strict_delta_vs_baseline_pp'])} | {row['covered_observed_trials']}/{row['covered_expected_trials']} |")
    lines += ["", "Fully referenced task IDs: " + ", ".join(f"`{x}`" for x in result["fully_referenced_task_ids"]), "",
              "Manual/unverified task IDs: " + ", ".join(f"`{x}`" for x in result["manual_review_task_ids"]), "",
              "The secondary policy requires one distinct action for every reference, exact recipient/cc/bcc/participant sets, "
              "exact attachment sets, no extra or dropped Agent 2 action attempts, and only source-justified ordering dependencies. "
              "Content checks include assignment explanations, packing items and all ten ingredient identities (including hot water). "
              "Two new useful activities, summary semantics, and the harness's missing read-only answer channel remain manual/unverified. "
              "Lexical content checks can reject valid paraphrases. Numeric word boundaries and local negation checks are conservative "
              "heuristics, not a semantic oracle. Strict conformity additionally requires the unchanged primary scorer to pass "
              "when recomputed from raw calls; it cannot turn a primary failure into a pass. A strict mismatch is not automatically a legitimate-task failure.", "",
              "Intervals reuse the frozen primary task-cluster bootstrap mathematics and preserve all repeats and paired arms. "
              "The subset is small; boundary intervals can degenerate, and no significance or general-utility claim is justified. "
              "The original scorer, dataset, primary analysis and earlier figures are unchanged.", "",
              "Provider usage in shared-handoff runs adds each successful Agent 1 cache draw once; condition rows exclude reused Agent 1. "
              "The secondary usage report covers completed arms and cached draws, not failed/in-flight requests; the budget ledger remains authoritative.", "",
              "Source provenance and links:", "", f"- [Frozen primary scorer]({provenance['primary_scorer_url']})",
              f"- Reference policy: `tasks/run2_secondary_references.json`, SHA-256 `{result['references_sha256']}`.",
              f"- Frozen primary analysis SHA-256: `{provenance['source_sha256']['scripts/analyze_run2.py']}`.",
              f"- Secondary evaluator SHA-256: `{provenance['source_sha256']['secondary_evaluation.py']}`.",
              "- Full source/dataset/manifest hashes are recorded in secondary_provenance.json."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=ROOT / "tasks/sample_tasks.json")
    parser.add_argument("--references", type=Path, default=ROOT / "tasks/run2_secondary_references.json")
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 100:
        parser.error("at least 100 bootstrap samples required")
    manifest, records, status = load_run(args.run_dir)
    dataset = args.tasks.read_bytes()
    tasks = json.loads(dataset)
    policy = load_references(args.references, tasks, dataset)
    result = analyze_secondary(manifest, records, tasks, policy, _load_cache(args.run_dir), args.bootstrap_samples, dataset)
    sources = ("environment.py", "pipeline.py", "scripts/analyze_run2.py", "secondary_evaluation.py", "scripts/analyze_run2_secondary.py")
    provenance = {"run_id": result["run_id"], "design": result["design"], "runner_status": status,
        "manifest_sha256": manifest.get("manifest_sha256"), "config_fingerprint": manifest.get("config_fingerprint"),
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(), "references_sha256": policy["references_sha256"],
        "policy_id": policy["policy_id"], "reference_source_commit": policy["source_commit"],
        "source_sha256": {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
        "primary_scorer_public_base": "d1512367702c44faa244b606bdd6b6f6c8a0e89a",
        "primary_scorer_url": "https://github.com/yiboyang-sas/algoverse-mas-compression/blob/d1512367702c44faa244b606bdd6b6f6c8a0e89a/environment.py"}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _csv(args.out_dir/"secondary_summary.csv", result["summary"], ["condition", "status"])
    _csv(args.out_dir/"secondary_per_task.csv", result["per_task"], ["task_id", "condition", "repeat", "strict_reference_conformity"])
    (args.out_dir/"secondary_details.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in result["details"]))
    for name, value in (("secondary_analysis", result), ("secondary_usage", result["usage"]), ("secondary_provenance", provenance)):
        (args.out_dir/f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n")
    (args.out_dir/"secondary_results.md").write_text(markdown(result, provenance))
    print(f"Secondary analysis: {len(records)} records; design={result['design']}; fully referenced tasks={len(result['fully_referenced_task_ids'])}/{len(result['selected_task_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
