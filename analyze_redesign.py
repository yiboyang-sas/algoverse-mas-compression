#!/usr/bin/env python3
"""Offline paired analysis of shared-handoff primary experiments and legacy streams.

No backend is constructed, no token vocabulary is loaded, and primary scores
are read, never replaced. PR #4's task-cluster uncertainty is reused through
analysis_stats. Native retention, tool mismatch, logging and scoring are reused.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys

from analysis_stats import cluster_interval, paired_interval
import baseline
from failure_taxonomy import classify

VERSION = "shared-handoff-primary-analysis-v2"
PREFIX = "Handoff from Agent 1:\n"
PROMPTED = {"unconditional", "destination", "instruction_strip", "allowlist"}
PROTOCOL_FIELDS = (
    "model", "compressor_model", "temperature", "seed", "handoff_regime",
    "experiment_mode", "compressor_budget_tokens", "primary_cer_min",
    "prompt_hashes", "token_count_method", "git_commit", "suites",
    "attack_enabled", "controls_enabled", "synthetic_expansion", "cer_strata",
)


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_config(value):
    """The upstream strong CER stratum has an infinite upper bound."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_config(item) for item in value]
    return value


def _jsonl(path):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as error:
            raise ValueError(f"{path.name}:{number}: invalid JSON; no rows discarded") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{number}: expected an object")
        rows.append(row)
    return rows


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _key(row, cfg, ids):
    if row.get("run_id") != cfg["run_id"]:
        raise ValueError("row run_id disagrees with config")
    tid = row.get("task_id")
    if tid not in ids:
        raise ValueError(f"unexpected selected task: {tid}")
    rep = _integer(row.get("repeat"), "repeat")
    if rep >= cfg["repeats"]:
        raise ValueError("repeat outside configured range")
    return tid, rep


def _dataset_provenance(cfg, tasks_path):
    """Check a clean historical task blob when available; never invent a hash."""
    name = cfg.get("tasks_file")
    if not isinstance(name, str) or Path(name).name != tasks_path.name:
        raise ValueError("--tasks disagrees with config.tasks_file")
    commit = cfg.get("git_commit", "")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit):
        return "unverified: recorded commit is not a full object ID"
    reference = f"{commit}:{name}"
    result = subprocess.run(
        ["git", "show", reference], cwd=Path(__file__).resolve().parent,
        capture_output=True, check=False)
    if result.returncode:
        return "unverified: historical dataset blob unavailable"
    expected = hashlib.sha256(result.stdout).hexdigest()
    if expected != _digest(tasks_path):
        raise ValueError("analysis dataset differs from the recorded commit's task file")
    if cfg.get("git_dirty"):
        return "unverified: matches commit, but executed tree was dirty"
    return "matches clean recorded commit"


def _stage_text(calls, condition):
    by = {c["stage"]: c for c in calls}
    expected = {"agent1", "agent2"} | ({"compressor"} if condition in PROMPTED else set())
    warnings = ["missing_" + stage + "_trace" for stage in sorted(expected - set(by))]
    a1, a2 = by.get("agent1", {}), by.get("agent2", {})
    raw = a1.get("raw_output")
    handoff = None
    if isinstance(raw, str) and not a1.get("api_error"):
        try:
            parsed = baseline._parse_tolerant(raw)
            handoff = parsed.get("handoff_note", "") or raw
            if not isinstance(handoff, str):
                handoff = None
                warnings.append("agent1_handoff_not_text")
        except (TypeError, ValueError, AttributeError):
            warnings.append("agent1_output_unrecoverable")
    transmitted = None
    user = a2.get("user_input")
    if isinstance(user, str):
        if not user.startswith(PREFIX):
            raise ValueError("Agent 2 input has an unsupported boundary; cannot infer transmitted text")
        transmitted = user[len(PREFIX):]
    comp = by.get("compressor")
    if condition == "none":
        if comp:
            raise ValueError("condition=none unexpectedly contains a compressor call")
        if handoff is not None and transmitted is not None and handoff != transmitted:
            raise ValueError("uncompressed handoff differs from Agent 2 input")
    elif condition in PROMPTED:
        if not comp:
            pass  # already exposed in the expected-stage coverage above
        elif handoff is not None and comp.get("user_input") != handoff:
            raise ValueError("compressor input differs from reconstructed Agent 1 handoff")
        if comp and isinstance(comp.get("raw_output"), str) and transmitted is not None:
            if comp["raw_output"].strip() != transmitted:
                raise ValueError("compressor output differs from Agent 2 input")
    elif condition != "sliding_window":
        raise ValueError(f"unsupported condition: {condition}")
    elif comp:
        raise ValueError("mechanical sliding_window unexpectedly contains a model compressor call")
    elif handoff is not None and transmitted is not None:
        if baseline.compressors._sliding_window(handoff) != transmitted:
            raise ValueError("mechanical sliding_window output differs from Agent 2 input")
    return raw, handoff, transmitted, a2.get("raw_output"), warnings


def load_run(run_dir, tasks_path, *, shared=None):
    """Load one run without guessing repeats from results.json array order."""
    root, tasks_path = Path(run_dir), Path(tasks_path)
    cfg = json.loads((root / "config.json").read_text())
    if not isinstance(cfg, dict) or not cfg.get("run_id") or not cfg.get("condition"):
        raise ValueError("config needs run_id and condition")
    if cfg["condition"] not in baseline.compressors.ALL_CONDITIONS:
        raise ValueError("unknown configured compression condition")
    for name in ("model", "compressor_model", "temperature", "handoff_regime", "experiment_mode",
                 "token_count_method", "git_commit", "prompt_hashes"):
        if name not in cfg or cfg[name] is None:
            raise ValueError(f"configuration missing comparison identity: {name}")
    if not isinstance(cfg["token_count_method"], str) or not cfg["token_count_method"]:
        raise ValueError("configuration must identify its token counting method")
    _integer(cfg.get("repeats"), "repeats", 1)
    _integer(cfg.get("compressor_budget_tokens"), "compressor_budget_tokens", 1)
    for name in ("temperature", "primary_cer_min"):
        value = cfg.get(name)
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError(f"nonfinite or invalid configuration value: {name}")
    selected = cfg.get("task_ids")
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("config.task_ids must explicitly list unique selected tasks")
    tasks = json.loads(tasks_path.read_text())
    if not isinstance(tasks, list) or any(not isinstance(t, dict) or not t.get("id") for t in tasks):
        raise ValueError("--tasks must contain task objects with IDs")
    task_map = {t["id"]: t for t in tasks}
    if len(task_map) != len(tasks):
        raise ValueError("duplicate task IDs in dataset")
    if set(selected) - set(task_map):
        raise ValueError("configured task IDs are absent from analysis dataset")
    dataset_status = _dataset_provenance(cfg, tasks_path)
    if shared is not None:
        if cfg.get("design") != "shared-handoff-primary" or cfg.get("experiment_id") != shared["manifest"]["experiment_id"]:
            raise ValueError("arm configuration disagrees with shared design/experiment")
    if not (root / "runs.jsonl").exists():
        raise ValueError("runs.jsonl is required; results.json cannot identify repeats safely")
    by_key = {}
    for row in _jsonl(root / "runs.jsonl"):
        key = _key(row, cfg, set(selected))
        if key in by_key:
            raise ValueError(f"duplicate task/repeat in runs.jsonl: {key}")
        for field in ("condition", "model", "temperature", "git_commit", "handoff_regime",
                      "experiment_mode", "compressor_budget_tokens", "token_count_method"):
            if field in row and field in cfg and row[field] != cfg[field]:
                raise ValueError(f"run row/config mismatch: {field}")
        if type(row.get("api_failure")) is not bool:
            raise ValueError("native api_failure must be boolean; never infer error coverage from truthiness")
        if row["api_failure"] and row.get("task_success") is not None:
            raise ValueError("native API/pipeline errors must have nullable task_success")
        if not row["api_failure"]:
            if type(row.get("task_success")) is not bool:
                raise ValueError("non-error primary task_success must be boolean")
            hit = _integer(row.get("required_hit"), "required_hit")
            total = _integer(row.get("required_total"), "required_total")
            task = task_map[key[0]]
            expected_total = sum(c["tool"] in set(task.get("agent2_held_tools", []))
                                 for c in task.get("required_calls", []))
            if total != expected_total:
                raise ValueError("saved required_total differs from the team's task write denominator")
            if hit > total or row["task_success"] != (hit == total):
                raise ValueError("inconsistent saved primary scoring fields")
            rate = row.get("call_hit_rate")
            if total and rate is not None and not math.isclose(rate, hit / total):
                raise ValueError("inconsistent saved call_hit_rate")
        for name in ("source_context_tokens", "agent1_input_tokens", "precompression_handoff_tokens", "compressed_tokens"):
            if row.get(name) is not None:
                _integer(row[name], name)
        for name in ("cer", "realized_ratio", "fraction_removed"):
            if row.get(name) is not None and (type(row[name]) not in (int, float) or not math.isfinite(row[name])):
                raise ValueError(f"nonfinite or invalid metric: {name}")
        if "suite" in row and row["suite"] != task_map[key[0]].get("domain"):
            raise ValueError("run suite disagrees with task snapshot")
        by_key[key] = row
    calls = collections.defaultdict(list)
    stages = set()
    all_call_usage = {name: 0 for name in ("prompt_tokens", "output_tokens")}
    missing_usage = collections.Counter()
    for call in _jsonl(root / "calls.jsonl"):
        key = _key(call, cfg, set(selected))
        stage = call.get("stage")
        if shared is not None and stage == "agent1":
            raise ValueError("shared arm must not repeat Agent 1 calls")
        if stage not in {"agent1", "agent2", "compressor"}:
            raise ValueError("unknown model stage")
        if (key, stage) in stages:
            raise ValueError("ambiguous duplicate model stage for task/repeat")
        stages.add((key, stage))
        if shared is not None and stage == "agent2":
            task = task_map[key[0]]
            held = task.get("agent2_held_tools") or task.get("allowed_tools") or []
            expected_prompt = baseline._AGENT2_TEMPLATE.replace("{sigs}", baseline._signatures(held))
            if call.get("system_prompt") != expected_prompt:
                raise ValueError("Agent 2 trace has wrong task-specific tool-schema prompt")
        expected_model = cfg["compressor_model"] if stage == "compressor" else cfg["model"]
        if call.get("model") != expected_model or call.get("temperature") != cfg["temperature"]:
            raise ValueError("call model/temperature disagrees with configured stage")
        if call.get("requested") is not True or type(call.get("succeeded")) is not bool:
            raise ValueError("call transport flags must be explicit booleans")
        if call["succeeded"] != (call.get("api_error") is None):
            raise ValueError("inconsistent call transport success/api_error flags")
        method = call.get("token_count_method")
        if method and cfg.get("token_count_method") and method != cfg["token_count_method"]:
            raise ValueError("mixed token counting methods in calls/config")
        for name in all_call_usage:
            value = call.get(name)
            if value is None:
                missing_usage[name] += 1
            else:
                all_call_usage[name] += _integer(value, name)
        calls[key].append(call)
    rows = []
    for key, row in sorted(by_key.items()):
        task = task_map[key[0]]
        effective_calls = list(calls[key])
        shared_id = None
        if shared is not None:
            cache = shared["cache"].get(key)
            if cache is None:
                raise ValueError("shared arm outcome has no durable Agent 1 cache")
            for name in ("shared_handoff_id", "agent1_output_sha256", "handoff_sha256"):
                if row.get(name) != cache.get(name):
                    raise ValueError("arm outcome disagrees with cached " + name)
            if (row.get("design") != "shared-handoff-primary"
                    or row.get("agent1_reused") is not (cache.get("status") == "ready")
                    or row.get("experiment_id") != shared["manifest"]["experiment_id"]):
                raise ValueError("arm outcome experiment_id mismatch")
            shared_id = cache["shared_handoff_id"]
            upstream = shared["calls"].get(key)
            if upstream:
                effective_calls.insert(0, upstream)
        raw, handoff, transmitted, a2_raw, warnings = _stage_text(effective_calls, cfg["condition"])
        if shared is not None and cache.get("status") == "ready":
            if raw != cache["agent1_raw"] or handoff != cache["handoff"]:
                raise ValueError("cached Agent 1 trace differs from reused handoff")
        info = classify(task, row, effective_calls)
        if any(c.get("api_error") for c in effective_calls) and not row.get("api_failure"):
            raise ValueError("failed model call conflicts with successful pipeline record")
        usage = {name: sum(c.get(name) or 0 for c in calls[key])
                 for name in ("prompt_tokens", "output_tokens")}
        rows.append({"record": row, "diagnostics": info, "boundary_warnings": warnings,
                     "usage": usage, "handoff": handoff, "transmitted": transmitted,
                     "shared_handoff_id": shared_id})
    planned = {(tid, rep) for tid in selected for rep in range(cfg["repeats"])}
    return {"config": cfg, "rows": rows,
            "missing": sorted(planned - set(by_key)),
            "orphan_calls": sorted(set(calls) - set(by_key)),
            "dataset_status": dataset_status,
            "all_call_usage": all_call_usage,
            "missing_usage_counts": dict(missing_usage),
            "input_hashes": {name: _digest(root / name) for name in
                             ("config.json", "runs.jsonl", "calls.jsonl", "results.json")
                             if (root / name).exists()},
            "path": str(root.resolve()), "tasks": task_map}


def _confined(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("shared manifest paths must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("shared manifest path escapes experiment root")
    return path


def load_shared_run(root, tasks_path):
    """Validate durable upstream identity before reading any paired outcomes."""
    from shared_run import validate_cache_record, validate_manifest
    root, tasks_path = Path(root).resolve(), Path(tasks_path)
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(manifest)
    settings = manifest.get("settings", {})
    if manifest.get("design") != "shared-handoff-primary":
        raise ValueError("not a shared-handoff-primary manifest")
    if not manifest.get("experiment_id") or manifest.get("backend") not in {"mock", "openai"}:
        raise ValueError("shared manifest requires experiment and backend identity")
    if (settings.get("model") != "gpt-4o-mini" or settings.get("temperature") != 0.0
            or settings.get("compressor_budget_tokens") != 150):
        raise ValueError("shared primary settings differ from finalized mini/T0/150 protocol")
    tasks = json.loads(tasks_path.read_text())
    if len(tasks) != 63 or {"ws_32", "ws_37"} & {task["id"] for task in tasks}:
        raise ValueError("shared primary requires canonical 63-task benchmark without ws_32/ws_37")
    task_map = {task["id"]: task for task in tasks}
    if len(task_map) != 63 or settings.get("dataset_sha256") != _digest(tasks_path):
        raise ValueError("shared dataset fingerprint differs from explicit analysis snapshot")
    ids = settings.get("task_ids")
    repeats = _integer(settings.get("repeats"), "shared repeats", 1)
    if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
            or set(ids) - set(task_map)):
        raise ValueError("invalid shared selected task IDs")
    conditions = list(baseline.compressors.ALL_CONDITIONS)
    if manifest.get("conditions") != conditions or set(manifest.get("arms", {})) != set(conditions):
        raise ValueError("shared primary manifest must contain all six conditions in native order")
    cache_path = _confined(root, manifest["handoffs_file"])
    cache = {}
    for row in _jsonl(cache_path):
        key = row.get("task_id"), row.get("repeat")
        if key[0] not in ids or type(key[1]) is not int or not 0 <= key[1] < repeats:
            raise ValueError("shared cache task/repeat outside frozen selection")
        if key in cache:
            raise ValueError("duplicate shared Agent 1 cache record")
        validate_cache_record(row, task_map[key[0]], manifest)
        cache[key] = row
    if len({row["shared_handoff_id"] for row in cache.values()}) != len(cache):
        raise ValueError("duplicate shared handoff identifier across task/repeat draws")
    upstream_dir = _confined(root, manifest["agent1_trace_dir"])
    upstream_calls, upstream_usage = {}, {"prompt_tokens": 0, "output_tokens": 0}
    missing_usage = collections.Counter()
    for call in _jsonl(upstream_dir / "calls.jsonl"):
        key = call.get("task_id"), call.get("repeat")
        if key not in cache or key in upstream_calls:
            raise ValueError("Agent 1 trace is orphaned or duplicated; cannot certify a single upstream draw")
        if call.get("run_id") != manifest["experiment_id"] + "-agent1":
            raise ValueError("shared Agent 1 run_id mismatch")
        if (call.get("stage") != "agent1" or call.get("model") != settings["model"]
                or call.get("temperature") != settings["temperature"]):
            raise ValueError("shared Agent 1 trace stage/model/temperature mismatch")
        if call.get("requested") is not True or type(call.get("succeeded")) is not bool:
            raise ValueError("shared Agent 1 transport flags must be explicit")
        if call["succeeded"] != (call.get("api_error") is None):
            raise ValueError("shared Agent 1 transport flags contradict each other")
        row = cache[key]
        if call.get("token_count_method") != settings["token_count_method"]:
            raise ValueError("shared upstream token counting method differs")
        if (call.get("raw_output") != row["agent1_raw"]
                or call.get("user_input") != row["agent1_input"]
                or call.get("system_prompt") != row["agent1_system"]):
            raise ValueError("shared Agent 1 cache differs from verbatim upstream call")
        if row.get("status") == "ready" and call.get("api_error"):
            raise ValueError("ready shared handoff conflicts with upstream transport error")
        upstream_calls[key] = call
        for name in upstream_usage:
            if call.get(name) is None:
                missing_usage[name] += 1
            else:
                upstream_usage[name] += _integer(call[name], name)
    if set(cache) != set(upstream_calls):
        raise ValueError("shared cache must have exactly one native Agent 1 call per persisted draw")
    run_dirs = []
    for condition in conditions:
        arm = manifest["arms"][condition]
        path = _confined(root, arm["run_dir"])
        cfg = json.loads((path / "config.json").read_text())
        if (cfg.get("condition") != condition or cfg.get("run_id") != arm["run_id"]
                or cfg.get("model") != settings["model"] or cfg.get("compressor_model") != settings["model"]
                or cfg.get("temperature") != settings["temperature"]
                or cfg.get("compressor_budget_tokens") != 150
                or cfg.get("task_ids") != ids or cfg.get("repeats") != repeats
                or cfg.get("manifest_sha256") != manifest["manifest_sha256"]
                or cfg.get("dataset_sha256") != settings["dataset_sha256"]
                or cfg.get("token_count_method") != settings["token_count_method"]):
            raise ValueError("shared arm config disagrees with frozen manifest")
        run_dirs.append(path)
    shared = {"manifest": manifest, "cache": cache, "calls": upstream_calls,
              "provenance": {"root": str(root), "manifest_sha256": _digest(root / "manifest.json"),
                  "handoffs_sha256": _digest(cache_path),
                  "upstream_calls_sha256": _digest(upstream_dir / "calls.jsonl") if (upstream_dir / "calls.jsonl").exists() else None,
                  "cached_draws": len(cache), "ready_draws": sum(r.get("status") == "ready" for r in cache.values()),
                  "expected_draws": len(ids)*repeats,
                  "local_text_token_counts_counted_once": upstream_usage,
                  "missing_local_usage_counts": dict(missing_usage),
                  "manifest": _json_config(manifest)}}
    return shared, run_dirs


def _values(rows, field):
    values = collections.defaultdict(list)
    for item in rows:
        row = item["record"]
        if not row.get("api_failure"):
            value = row.get(field)
            if value is not None:
                values[row["task_id"]].append(float(value))
    return dict(values)


def _summarise(run, draws, seed):
    rows, cfg = run["rows"], run["config"]
    scored = [i for i in rows if not i["record"].get("api_failure")]
    hits = sum(i["record"]["required_hit"] for i in scored)
    total = sum(i["record"]["required_total"] for i in scored)
    counters = collections.Counter()
    for item in rows:
        counters[item["diagnostics"]["category"]] += 1
    groups = {}
    for field in ("suite", "compression_stratum"):
        groups[field] = {}
        for value in sorted({i["record"].get(field, "unknown") for i in rows}):
            group = [i for i in rows if i["record"].get(field, "unknown") == value]
            groups[field][value] = cluster_interval(_values(group, "task_success"), draws=draws, seed=seed)
    repeat_rates = {}
    for rep in range(cfg["repeats"]):
        valid = [i["record"] for i in scored if i["record"]["repeat"] == rep]
        repeat_rates[str(rep)] = sum(r["task_success"] for r in valid) / len(valid) if valid else None
    compressed = [i["record"].get("compressed_tokens") for i in scored
                  if i["record"].get("compressed_tokens") is not None]
    cer = [i["record"].get("cer") for i in scored if i["record"].get("cer") is not None]
    return {"condition": cfg["condition"], "run_id": cfg["run_id"],
            "selected_tasks": len(cfg["task_ids"]),
            "expected_trials": len(cfg["task_ids"]) * cfg["repeats"],
            "observed_trials": len(rows), "scored_trials": len(scored),
            "missing_trials": len(run["missing"]),
            "api_or_pipeline_failures": len(rows) - len(scored),
            "orphan_call_groups": len(run["orphan_calls"]),
            "scored_trials_with_incomplete_stage_traces": sum(bool(i["boundary_warnings"]) for i in scored),
            "task_success": cluster_interval(_values(rows, "task_success"), draws=draws, seed=seed),
            "observed_task_success_rate": sum(i["record"]["task_success"] for i in scored) / len(scored) if scored else None,
            "macro_task_call_completion": cluster_interval(_values(rows, "call_hit_rate"), draws=draws, seed=seed),
            "required_call_hit_rate": hits / total if total else None,
            "required_hits": hits, "required_calls": total,
            "repeat_task_success": repeat_rates,
            "group_task_success": groups,
            "failure_taxonomy_counts": dict(counters),
            "compressed_length_observations": len(compressed),
            "compressed_tokens_median": statistics.median(compressed) if compressed else None,
            "over_nominal_budget": sum(n > cfg["compressor_budget_tokens"] for n in compressed),
            "cer_median": statistics.median(cer) if cer else None,
            "local_text_token_counts": run["all_call_usage"],
            "missing_local_usage_counts": run["missing_usage_counts"],
            "token_count_method": cfg.get("token_count_method"),
            "usage_is_billed_provider_usage": False}


def analyze(run_dirs, tasks_path, *, draws=2000, seed=1729, fixture=False, shared=None):
    if type(draws) is not int or draws < 1:
        raise ValueError("bootstrap draws must be positive")
    runs = [load_run(path, tasks_path, shared=shared) for path in run_dirs]
    seen, run_ids, protocol = set(), set(), None
    for run in runs:
        cfg = run["config"]
        if cfg["run_id"] in run_ids:
            raise ValueError("duplicate run_id across input directories")
        run_ids.add(cfg["run_id"])
        if cfg["condition"] in seen:
            raise ValueError("one run directory per condition is required for unambiguous repeat pairing")
        seen.add(cfg["condition"])
        signature = {key: cfg.get(key) for key in PROTOCOL_FIELDS}
        signature["task_ids"] = cfg["task_ids"]
        signature["tasks_file"] = cfg["tasks_file"]
        if protocol is not None and signature != protocol:
            differences = [k for k in signature if signature[k] != protocol[k]]
            raise ValueError("incompatible run configurations: " + ", ".join(differences))
        protocol = signature
    summaries = [_summarise(run, draws, seed) for run in runs]
    base = next((run for run in runs if run["config"]["condition"] == "none"), None)
    comparisons = []
    def slots(run):
        return {(i["record"]["task_id"], i["record"]["repeat"]): float(i["record"]["task_success"])
                for i in run["rows"] if not i["record"].get("api_failure")}
    # All 15 contrasts share the exact upstream sample. The five none-reference
    # contrasts are primary; the remaining contrasts are descriptive.
    contrasts = []
    if shared is not None:
        from itertools import combinations
        contrasts = list(combinations(runs, 2))
    elif base:
        contrasts = [(base, run) for run in runs if run is not base]
    for reference, run in contrasts:
        base_slots, treatment_slots = slots(reference), slots(run)
        common = set(base_slots) & set(treatment_slots)
        reference_rows = {(i["record"]["task_id"], i["record"]["repeat"]): i for i in reference["rows"]}
        treatment_rows = {(i["record"]["task_id"], i["record"]["repeat"]): i for i in run["rows"]}
        if shared is not None:
            for key in set(reference_rows) & set(treatment_rows):
                a, b = reference_rows[key]["record"], treatment_rows[key]["record"]
                for field in ("shared_handoff_id", "agent1_output_sha256", "handoff_sha256"):
                    if field not in a or field not in b or a[field] != b[field]:
                        raise ValueError("condition comparison does not share identical upstream sample: " + field)
        comparison = paired_interval(base_slots, treatment_slots, draws=draws, seed=seed)
        for pair in comparison["per_pair"]:
            key = pair["task_id"], pair["repeat"]
            pair.update(shared_handoff_id=reference_rows[key]["shared_handoff_id"],
                        baseline_category=reference_rows[key]["diagnostics"]["category"],
                        treatment_category=treatment_rows[key]["diagnostics"]["category"])
        comparison.update(condition=run["config"]["condition"],
                          baseline_condition=reference["config"]["condition"],
                          primary_contrast=reference["config"]["condition"] == "none" and shared is not None,
                          interpretation=("same task/repeat and byte-identical upstream handoff; task-cluster uncertainty"
                              if shared is not None else
                              "legacy independent upstream draws; repeat labels alone do not control Agent 1 variability"))
        comparisons.append(comparison)
    return {"analysis_version": VERSION,
            "design": "shared-handoff-primary" if shared is not None else "legacy-independent",
            "data_kind": ("deterministic_fixture_non_empirical" if fixture or (shared is not None and shared["manifest"].get("backend") == "mock")
                          else "backend_unverified_from_upstream_metadata"),
            "shared_upstream": shared["provenance"] if shared is not None else None,
            "conditions": summaries, "comparisons": comparisons,
            "execution_local_text_token_counts": {
                name: sum(run["all_call_usage"][name] for run in runs)
                      + (shared["provenance"]["local_text_token_counts_counted_once"][name] if shared is not None else 0)
                for name in ("prompt_tokens", "output_tokens")},
            "baseline_available": base is not None,
            "provenance": {"analysis_tasks_path": str(Path(tasks_path).resolve()),
                           "analysis_tasks_sha256": _digest(tasks_path),
                           "analysis_dataset_task_count": len(runs[0]["tasks"]) if runs else 0,
                           "analysis_code_sha256": {name: _digest(Path(__file__).resolve().parent / name)
                                                    for name in ("analyze_redesign.py", "analysis_stats.py", "failure_taxonomy.py", "shared_run.py",
                                                                 "baseline.py", "failure_stage.py", "obtainable.py", "pipeline.py",
                                                                 "compressors.py", "suite_tools.py", "config.py")},
                           "python_version": sys.version.split()[0],
                           "bootstrap_draws": draws, "bootstrap_seed": seed,
                           "interval": "95% percentile bootstrap; whole tasks resampled with their repeats",
                           "config_serialization": "nonfinite upstream CER bounds are represented as strings; input hashes preserve exact original bytes",
                           "runs": [{k: _json_config(run[k]) for k in ("path", "config", "input_hashes", "dataset_status", "missing", "orphan_calls")} for run in runs]},
            "limitations": [
                "Scores are saved team primary scores, never rescored by this analysis.",
                "All selected tasks remain in coverage; API/pipeline failures and missing rows are not model failures.",
                "Failure taxonomy describes exclusive observable failure profiles, not proven semantic causes. Existing retention and schema checks are passed through unchanged.",
                "PLAN schema warnings and configuration coverage warnings never remove primary rows.",
                "calls.parse_status is not a parser outcome; use runs.parse_events and raw evidence.",
                "Token counts are local text lengths, not billed usage; nominal budget is not a hard token limit.",
                "Legacy config does not identify mock execution. Shared manifests record backend identity; mock outputs are non-empirical.",
                "Shared handoffs control upstream sampling, but do not prove information loss or instruction misunderstanding. Review native controls and verbatim traces.",
                "All 15 pairwise contrasts are exploratory; the five none-reference contrasts are marked primary. No multiplicity-adjusted significance claim is made.",
                "Task-cluster intervals are exploratory; repeated calls are not independent tasks, and small samples can yield degenerate intervals.",
            ]}, runs


def write_outputs(analysis, runs, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("analysis output directory must be empty")
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
    columns = ["run_id", "task_id", "repeat", "condition", "suite", "task_success",
               "required_hit", "required_total", "call_hit_rate", "api_failure",
               "shared_handoff_id", "agent1_output_sha256", "handoff_sha256",
               "parse_events", "precompression_handoff_tokens", "compressed_tokens",
               "compressor_budget_tokens", "cer", "compression_stratum", "token_count_method"]
    with (out / "trials.csv").open("w", newline="") as handle, (out / "diagnostics.jsonl").open("w") as detail:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for run in runs:
            for item in run["rows"]:
                row = item["record"]
                writer.writerow({key: json.dumps(row[key]) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in columns})
                detail.write(json.dumps({"run_id": row["run_id"], "task_id": row["task_id"],
                                         "repeat": row["repeat"], "condition": row["condition"],
                                         "shared_handoff_id": item["shared_handoff_id"],
                                         "boundary_warnings": item["boundary_warnings"],
                                         **item["diagnostics"]}, allow_nan=False) + "\n")
    def percent(value):
        return "unavailable" if value is None else f"{100 * value:.1f}%"
    lines = ["# Compression redesign offline analysis", "", analysis["data_kind"], "",
             "Design: " + analysis["design"], "",
             "The team primary scores are unchanged. Rates below use available scored observations; coverage is reported separately.", "",
             "| Condition | Scored / planned | Missing | API or pipeline errors | Observed success | Equal-task success and 95% interval | Required-call completion |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in analysis["conditions"]:
        ci = row["task_success"]
        interval = f"{percent(ci['estimate'])} [{percent(ci['low'])}, {percent(ci['high'])}]"
        lines.append(f"| {row['condition']} | {row['scored_trials']} / {row['expected_trials']} | {row['missing_trials']} | {row['api_or_pipeline_failures']} | {percent(row['observed_task_success_rate'])} | {interval} | {percent(row['required_call_hit_rate'])} |")
    lines += ["", "Observed success weights available trials equally. Equal-task success averages repeats within each task before averaging tasks; these can differ when coverage is incomplete.", "",
              "## Matched comparisons", ""]
    if not analysis["baseline_available"]:
        lines += ["No no-compression baseline was supplied; no baseline differences are available.", ""]
    for comparison in analysis["comparisons"]:
        lines += [f"- {comparison['condition']} minus {comparison['baseline_condition']}: {percent(comparison['estimate'])} (absolute percentage-point difference); 95% interval [{percent(comparison['low'])}, {percent(comparison['high'])}], {comparison['n_pairs']} matched slots across {comparison['n_tasks']} tasks; discordant improvements={comparison['n01']}, regressions={comparison['n10']}."]
    lines += ["", "## Exclusive observable profiles", "",
              "Counts include success and infrastructure errors; annotations are not additional failures.", "",
              "| Condition | Profile | Trials |", "|---|---|---:|"]
    for condition in analysis["conditions"]:
        for category, count in sorted(condition["failure_taxonomy_counts"].items()):
            lines.append(f"| {condition['condition']} | {category} | {count} |")
    if analysis["shared_upstream"] is not None:
        upstream = analysis["shared_upstream"]
        lines += ["", f"Durable upstream draws: {upstream['cached_draws']} / {upstream['expected_draws']}; ready: {upstream['ready_draws']}. Agent 1 call tokens are counted once, separately from arm call tokens."]
    lines += ["", "Per-task paired differences and coverage are in analysis.json. Shared-primary pairs validate handoff IDs and both output hashes. Legacy independent pairs only match task/repeat labels. No primary cohort is selected by observed success or schema warnings.", "", "## Interpretation limits", ""]
    lines += ["- " + text for text in analysis["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="*")
    parser.add_argument("--shared-run", help="primary shared-handoff experiment root")
    parser.add_argument("--legacy-independent", action="store_true", help="explicit compatibility analysis of historical independent draws; not the primary design")
    parser.add_argument("--tasks", required=True, help="explicit benchmark snapshot; never default to a legacy task set")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--fixture", action="store_true", help="label deterministic or mock inputs as non-empirical")
    args = parser.parse_args(argv)
    try:
        shared = None
        run_dirs = args.run_dirs
        if args.shared_run:
            if run_dirs or args.legacy_independent:
                raise ValueError("use --shared-run alone, or --legacy-independent run directories")
            shared, run_dirs = load_shared_run(args.shared_run, args.tasks)
        elif not args.legacy_independent or not run_dirs:
            raise ValueError("primary analysis requires --shared-run; historical directories require --legacy-independent")
        analysis, runs = analyze(run_dirs, args.tasks, draws=args.bootstrap_draws,
                                 seed=args.seed, fixture=args.fixture, shared=shared)
        write_outputs(analysis, runs, args.out_dir)
    except (ValueError, OSError, TypeError, KeyError) as error:
        print(f"analysis rejected: {error}", file=sys.stderr)
        return 2
    print(f"Wrote validated offline analysis to {args.out_dir}; no model calls made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
