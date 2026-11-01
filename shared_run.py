"""Primary benign experiment: one durable Agent 1 draw, six compression arms.

The default backend is the repository's offline mock. Paid execution is an
explicit future action; this module never obtains credentials or changes the
native prompts, compressors, control measurements, parser or scorer.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

# These two native import-time defaults must not inherit an unrelated sweep.
_saved_env = {k: os.environ.get(k) for k in ("MAS_AGENT1", "MAS_BUDGET_TOKENS")}
os.environ["MAS_AGENT1"] = "plan"
os.environ["MAS_BUDGET_TOKENS"] = "150"
try:
    import baseline
finally:
    for _key, _value in _saved_env.items():
        if _value is None:
            os.environ.pop(_key, None)
        else:
            os.environ[_key] = _value

import agent1_prompts
import compressors
import config
import environment
import instrument
import metrics
import pipeline
import tracing
from models import MockBackend, OpenAIBackend

ROOT = Path(__file__).resolve().parent
TASKS_FILE = "tasks/sample_tasks_63.json"
DATASET_SHA256 = "eb78f48f100b05c3277bb748a5a8e78e81981bfe00532063e570ee4d99090eed"
DESIGN = "shared-handoff-primary"
CONDITIONS = tuple(compressors.ALL_CONDITIONS)
SOURCE_FILES = (
    "shared_run.py", "pipeline.py", "baseline.py", "models.py", "instrument.py",
    "tracing.py", "metrics.py", "config.py", "compressors.py", "controls.py",
    "failure_stage.py", "agent1_prompts.py", "environment.py", "obtainable.py",
    "suite_tools.py",
)


class IntegrityError(ValueError):
    """Evidence is inconsistent; resuming could silently redraw a trial."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _digest_record(record, field):
    return sha256(canonical({k: v for k, v in record.items() if k != field}))


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_identity():
    return {"git_commit": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "files": {name: sha256((ROOT / name).read_bytes()) for name in SOURCE_FILES}}


def load_tasks():
    raw = (ROOT / TASKS_FILE).read_bytes()
    if sha256(raw) != DATASET_SHA256:
        raise IntegrityError("canonical 63-task dataset bytes changed")
    tasks = json.loads(raw)
    ids = [t.get("id") for t in tasks]
    if len(tasks) != 63 or len(set(ids)) != 63 or {"ws_32", "ws_37"} & set(ids):
        raise IntegrityError("expected 63 unique canonical tasks without ws_32/ws_37")
    for task in tasks:
        if (task.get("harmful_call") or task.get("payload_scope")
                or "INJECT_CALL" in canonical(task).decode()):
            raise IntegrityError("attack-bearing task is forbidden")
    return tasks


def validate_manifest(manifest):
    if manifest.get("manifest_sha256") != _digest_record(manifest, "manifest_sha256"):
        raise IntegrityError("manifest checksum mismatch")
    s = manifest.get("settings", {})
    if (manifest.get("design") != DESIGN or tuple(manifest.get("conditions", [])) != CONDITIONS
            or s.get("model") != "gpt-4o-mini" or s.get("compressor_model") != "gpt-4o-mini"
            or s.get("temperature") != 0 or s.get("compressor_budget_tokens") != 150
            or s.get("handoff_regime") != "plan" or s.get("attack_enabled") is not False
            or s.get("dataset_sha256") != DATASET_SHA256
            or manifest.get("backend") != s.get("backend")
            or s.get("backend") not in ("mock", "openai")):
        raise IntegrityError("manifest differs from fixed primary design")
    if type(s.get("repeats")) is not int or s["repeats"] < 1:
        raise IntegrityError("invalid repeat count")
    if not s.get("task_ids") or len(s["task_ids"]) != len(set(s["task_ids"])):
        raise IntegrityError("invalid selected task identifiers")
    if not set(s["task_ids"]) <= set(s.get("dataset_task_ids", [])):
        raise IntegrityError("selected task outside dataset")
    if {"ws_32", "ws_37"} & set(s["task_ids"]):
        raise IntegrityError("excluded tasks selected")
    eid = manifest.get("experiment_id")
    if (not isinstance(eid, str) or not eid
            or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in eid)):
        raise IntegrityError("invalid experiment identifier")
    if (manifest.get("handoffs_file") != "handoffs.jsonl"
            or manifest.get("agent1_trace_dir") != "shared_agent1"
            or manifest.get("arms") != {c: {"run_dir": "arms/" + c, "run_id": eid + "-" + c}
                                       for c in CONDITIONS}):
        raise IntegrityError("manifest arm/cache paths differ from fixed layout")
    if (type(s.get("workers")) is not int or s["workers"] < 1
            or len(s.get("dataset_task_ids", [])) != 63
            or len(set(s["dataset_task_ids"])) != 63
            or s.get("tasks_file") != TASKS_FILE):
        raise IntegrityError("invalid canonical dataset/execution identity")
    return manifest


def validate_cache_record(record, task, manifest):
    """Validate immutable cache provenance and native raw-derived preparation.

    Empty outputs are valid draws. An upstream exception is a terminal error
    cache, rather than permission to draw a different Agent 1 response.
    """
    if record.get("record_sha256") != _digest_record(record, "record_sha256"):
        raise IntegrityError("handoff cache checksum mismatch")
    checks = {"task_id": task["id"], "task_sha256": sha256(canonical(task)),
              "dataset_sha256": manifest["settings"]["dataset_sha256"],
              "experiment_id": manifest["experiment_id"],
              "manifest_sha256": manifest["manifest_sha256"],
              "agent1_input": pipeline._build_agent1_input(task),
              "agent1_system": agent1_prompts.REGIMES["plan"]}
    if any(record.get(k) != v for k, v in checks.items()):
        raise IntegrityError("handoff cache task/input/identity mismatch")
    if (task["id"] not in manifest["settings"]["task_ids"]
            or type(record.get("repeat")) is not int
            or not 0 <= record["repeat"] < manifest["settings"]["repeats"]
            or not isinstance(record.get("shared_handoff_id"), str)
            or not record["shared_handoff_id"]):
        raise IntegrityError("handoff cache key mismatch")
    raw, handoff = record.get("agent1_raw"), record.get("handoff")
    if raw is not None and not isinstance(raw, str):
        raise IntegrityError("Agent 1 raw output must be text or null")
    if record.get("agent1_output_sha256") != (sha256(raw.encode()) if raw is not None else None):
        raise IntegrityError("Agent 1 output checksum mismatch")
    if record.get("handoff_sha256") != (sha256(handoff.encode()) if isinstance(handoff, str) else None):
        raise IntegrityError("handoff checksum mismatch")
    if record.get("status") == "ready":
        if raw is None or not isinstance(handoff, str):
            raise IntegrityError("ready cache lacks raw output/handoff")
        parsed = pipeline._parse(raw)
        reads = environment.ToolLog()
        pipeline._replay(parsed.get("tool_calls", []), reads,
                         allowed=task.get("agent1_tools", ["read_file", "read_db"]))
        if (record.get("parsed_agent1") != parsed or record.get("read_calls") != reads.calls
                or handoff != (parsed.get("handoff_note", "") or raw)):
            raise IntegrityError("cache preparation differs from native parsing/replay")
    elif record.get("status") == "error":
        if not record.get("error_type") or handoff is not None:
            raise IntegrityError("invalid upstream error cache")
    else:
        raise IntegrityError("unknown cache status")
    return record


def _read_jsonl(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        raise IntegrityError(f"incomplete journal (manual review required): {path}")
    rows = []
    for line in data.splitlines():
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise IntegrityError(f"malformed journal: {path}") from exc
        if not isinstance(row, dict):
            raise IntegrityError(f"non-object journal row: {path}")
        rows.append(row)
    return rows


def _durable_json(path, data, *, exclusive=False):
    # Manifests/configs are created once. Status is a replaceable summary.
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=True).encode() + b"\n"
    if exclusive:
        with path.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        temp = path.with_suffix(path.suffix + ".tmp")
        with temp.open("wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync(path):
    if path.exists():
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        # Persist the journal's directory entry as well as its contents.
        fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _key(row, manifest):
    key = (row.get("task_id"), row.get("repeat"))
    if (key[0] not in manifest["settings"]["task_ids"]
            or type(key[1]) is not int
            or not 0 <= key[1] < manifest["settings"]["repeats"]):
        raise IntegrityError("journal contains an unplanned task/repeat")
    return key


def _unique(rows, manifest, label):
    out = {}
    for row in rows:
        key = _key(row, manifest)
        if key in out:
            raise IntegrityError(f"duplicate {label} for {key}")
        out[key] = row
    return out


def _group_calls(rows, manifest, run_id, allowed_stages):
    out = {}
    for row in rows:
        key = _key(row, manifest)
        if (row.get("kind") != "call" or row.get("run_id") != run_id
                or row.get("stage") not in allowed_stages
                or row.get("model") != "gpt-4o-mini" or row.get("temperature") != 0
                or row.get("requested") is not True):
            raise IntegrityError("native call provenance/stage mismatch")
        succeeded = row.get("succeeded")
        if (type(succeeded) is not bool
                or (succeeded and row.get("api_error") is not None)
                or (not succeeded and (not isinstance(row.get("api_error"), str)
                                       or not row["api_error"] or row.get("raw_output") is not None))):
            raise IntegrityError("native call transport flag/error mismatch")
        out.setdefault(key, []).append(row)
    return out


def _shared_fields(cache):
    return {"design": DESIGN, "experiment_id": cache["experiment_id"],
            "shared_handoff_id": cache["shared_handoff_id"],
            "agent1_output_sha256": cache["agent1_output_sha256"],
            "handoff_sha256": cache["handoff_sha256"],
            "agent1_reused": cache["status"] == "ready",
            "agent1_trace_dir": "../../shared_agent1"}


def _config(manifest, condition, out_dir):
    s = manifest["settings"]
    cfg = config.RunConfig(
        experiment_mode="stress_test", handoff_regime="plan", condition=condition,
        tasks_file=TASKS_FILE, task_ids=tuple(s["task_ids"]),
        compressor_budget_tokens=150, model="gpt-4o-mini", compressor_model="gpt-4o-mini",
        temperature=0.0, seed=None, workers=s["workers"], repeats=s["repeats"],
        require_measured_tokens=s["backend"] == "openai", synthetic_expansion=True,
        controls_enabled=True, attack_enabled=False, output_dir=str(out_dir / "arms"),
        run_id=manifest["arms"][condition]["run_id"], started_at=manifest["created_at"],
        git_commit=manifest["source"]["git_commit"], git_branch=manifest["source"]["git_branch"],
        git_dirty=manifest["source"]["git_dirty"], prompt_hashes=s["prompt_hashes"],
        token_count_method=s["token_count_method"])
    return cfg


def _config_json(cfg, manifest):
    value = cfg.as_dict()
    value.update(design=DESIGN, experiment_id=manifest["experiment_id"],
                 backend=manifest["backend"], dataset_sha256=DATASET_SHA256,
                 manifest_sha256=manifest["manifest_sha256"],
                 shared_handoffs_file="../../handoffs.jsonl",
                 shared_agent1_trace_dir="../../shared_agent1",
                 seed_policy="unset_native_default_not_bit_reproducible")
    return value


def _prepare_manifest(args, all_tasks, method):
    selected = all_tasks[:args.limit] if args.limit else all_tasks
    settings = {"model": "gpt-4o-mini", "compressor_model": "gpt-4o-mini",
                "temperature": 0.0, "compressor_budget_tokens": 150,
                "handoff_regime": "plan", "repeats": args.repeats, "workers": args.workers,
                "tasks_file": TASKS_FILE, "task_ids": [t["id"] for t in selected],
                "dataset_task_ids": [t["id"] for t in all_tasks], "dataset_sha256": DATASET_SHA256,
                "token_count_method": method, "backend": "openai" if args.execute_paid else "mock",
                "attack_enabled": False, "controls_enabled": True, "synthetic_expansion": True,
                "seed": None, "seed_policy": "unset_native_default_not_bit_reproducible",
                "prompt_hashes": {"agent1": config.prompt_hash(agent1_prompts.REGIMES["plan"]),
                                  "agent2_template": config.prompt_hash(baseline._AGENT2_TEMPLATE)}}
    return selected, settings


def _open_or_create(args, tasks, method):
    out = Path(args.out_dir).resolve()
    selected, settings = _prepare_manifest(args, tasks, method)
    path = out / "manifest.json"
    source = source_identity()
    if path.exists():
        manifest = validate_manifest(json.loads(path.read_text()))
        # Dirty/branch metadata describes creation. Generated run artifacts or
        # unrelated report edits do not change the frozen generating sources.
        source_changed = any(manifest["source"].get(k) != source[k]
                             for k in ("git_commit", "files"))
        if (manifest["settings"] != settings or source_changed
                or (args.experiment_id and args.experiment_id != manifest["experiment_id"])):
            raise IntegrityError("resume configuration/source differs from frozen manifest")
    else:
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise IntegrityError("new experiment directory must be empty")
        eid = args.experiment_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:8]
        if not eid or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in eid):
            raise IntegrityError("experiment ID must be alphanumeric, dash or underscore")
        manifest = {"schema_version": 1, "design": DESIGN, "experiment_id": eid,
                    "backend": settings["backend"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "experimental_design": "One immutable Agent 1 draw per task/repeat reused across all six arms; none completed first.",
                    "settings": settings, "conditions": list(CONDITIONS), "source": source,
                    "arms": {c: {"run_dir": "arms/" + c, "run_id": eid + "-" + c} for c in CONDITIONS},
                    "handoffs_file": "handoffs.jsonl", "agent1_trace_dir": "shared_agent1",
                    "planned": {"handoffs": len(selected) * args.repeats,
                                "arm_trials": len(selected) * args.repeats * len(CONDITIONS),
                                "model_calls_without_transport_retries": len(selected) * args.repeats * 11},
                    "full_primary": len(selected) == 63 and args.repeats == 3,
                    "cache_atomicity_limit": "A call intent without a durable terminal cache requires manual review; it is never redrawn automatically."}
        manifest["manifest_sha256"] = _digest_record(manifest, "manifest_sha256")
        validate_manifest(manifest)
        _durable_json(path, manifest, exclusive=True)
    return out, selected, manifest


def _validate_existing(out, tasks, manifest, configs):
    by_id = {t["id"]: t for t in tasks}
    cache = _unique(_read_jsonl(out / "handoffs.jsonl"), manifest, "cache")
    if len({r["shared_handoff_id"] for r in cache.values()}) != len(cache):
        raise IntegrityError("duplicate shared handoff identifier")
    upstream_dir = out / "shared_agent1"
    calls = _group_calls(_read_jsonl(upstream_dir / "calls.jsonl"), manifest,
                         manifest["experiment_id"] + "-agent1", {"agent1"})
    intents = _unique(_read_jsonl(upstream_dir / "attempts.jsonl"), manifest, "upstream intent")
    if set(calls) - set(cache) or set(intents) - set(cache):
        raise IntegrityError("upstream attempt without durable cache; automatic redraw forbidden")
    for key, rec in cache.items():
        validate_cache_record(rec, by_id[key[0]], manifest)
        traces = calls.get(key, [])
        if key not in intents or len(traces) != 1:
            raise IntegrityError("cache requires exactly one native upstream call and intent")
        call = traces[0]
        if (intents[key].get("shared_handoff_id") != rec["shared_handoff_id"]
                or call.get("raw_output") != rec["agent1_raw"]
                or call.get("user_input") != rec["agent1_input"]
                or call.get("system_prompt") != rec["agent1_system"]
                or (rec["status"] == "ready" and call.get("succeeded") is not True)):
            raise IntegrityError("cache/native upstream trace mismatch")
    completed = {}
    for condition in CONDITIONS:
        arm_dir = out / manifest["arms"][condition]["run_dir"]
        arm_dir.mkdir(parents=True, exist_ok=True)
        config_path = arm_dir / "config.json"
        expected = _config_json(configs[condition], manifest)
        if config_path.exists():
            if json.loads(config_path.read_text()) != expected:
                raise IntegrityError("arm configuration changed")
        else:
            if any(arm_dir.iterdir()):
                raise IntegrityError("arm data without immutable configuration")
            _durable_json(config_path, expected, exclusive=True)
        rows = _unique(_read_jsonl(arm_dir / "runs.jsonl"), manifest, "arm result")
        native = _group_calls(_read_jsonl(arm_dir / "calls.jsonl"), manifest,
                             configs[condition].run_id, {"compressor", "agent2"})
        attempted = _unique(_read_jsonl(arm_dir / "attempts.jsonl"), manifest, "arm intent")
        if set(native) - set(rows) or set(attempted) - set(rows):
            raise IntegrityError("orphan downstream attempt without result; manual review required")
        for key, row in rows.items():
            if key not in cache:
                raise IntegrityError("completed arm has no durable shared cache")
            if key in attempted and attempted[key].get("shared_handoff_id") != cache[key]["shared_handoff_id"]:
                raise IntegrityError("downstream intent shared identity mismatch")
            if (row.get("kind") != "run" or row.get("run_id") != configs[condition].run_id
                    or row.get("condition") != condition or row.get("model") != "gpt-4o-mini"
                    or row.get("temperature") != 0
                    or any(row.get(k) != v for k, v in _shared_fields(cache[key]).items())):
                raise IntegrityError("arm result shared identity mismatch")
            traces = native.get(key, [])
            stages = [c["stage"] for c in traces]
            expected_stages = ["agent2"] if condition in ("none", "sliding_window") else ["compressor", "agent2"]
            # Each native wrapper emits at most one terminal trace per stage.
            # A local/transport exception may stop at any prefix, including
            # zero calls, but cannot skip, repeat or proceed past a failed call.
            if (stages != expected_stages[:len(stages)]
                    or any(c["succeeded"] is False for c in traces[:-1])):
                raise IntegrityError("downstream trace stages are not a valid native prefix")
            if row.get("api_failure"):
                if row.get("task_success") is not None:
                    raise IntegrityError("infrastructure failure cannot be scored")
                if cache[key]["status"] == "error" and (traces or key in attempted or row.get("error_stage") != "agent1"):
                    raise IntegrityError("upstream error must not execute downstream")
                if cache[key]["status"] == "ready" and key not in attempted:
                    raise IntegrityError("downstream error without intent")
            else:
                if key not in attempted or stages != expected_stages or any(c.get("succeeded") is not True for c in traces):
                    raise IntegrityError("successful arm lacks exact native downstream traces")
                transmitted = cache[key]["handoff"]
                if condition == "sliding_window":
                    transmitted = compressors._sliding_window(transmitted)
                elif condition not in ("none", "sliding_window"):
                    if traces[0]["user_input"] != transmitted:
                        raise IntegrityError("compressor did not receive exact shared handoff")
                    transmitted = traces[0]["raw_output"].strip()
                if traces[-1]["user_input"] != "Handoff from Agent 1:\n" + transmitted:
                    raise IntegrityError("Agent 2 handoff differs from recorded treatment")
        completed[condition] = rows
        for name in ("runs", "calls"):
            (arm_dir / (name + ".jsonl")).touch(exist_ok=True)
    return cache, completed


def _error_row(task, repeat, cfg, cache, error_type, stage):
    return tracing.run_record(
        cfg=cfg, task=task, repeat=repeat, metrics={}, scoring={"task_success": None},
        api_failure=True, failure_stage="API", extra={**_shared_fields(cache),
            "error_type": error_type, "error_stage": stage, "controls": None,
            "tool_mismatch": None, "n_model_calls": 0 if stage == "agent1" else None})


def _run_locked(args, *, backend=None):
    """Execute/resume using native instrumentation; backend injection is offline-test only."""
    if args.repeats < 1 or args.workers < 1 or (args.limit is not None and not 1 <= args.limit <= 63):
        raise IntegrityError("positive repeats/workers and limit in 1..63 required")
    pipeline.AGENT1_SYSTEM = agent1_prompts.REGIMES["plan"]
    tasks = load_tasks()  # validate before constructing any network backend
    method = metrics.tokenizer_name("gpt-4o-mini")
    if args.execute_paid:
        if not os.environ.get("OPENAI_API_KEY"):
            raise IntegrityError("paid execution requires explicitly supplied OPENAI_API_KEY")
        if not metrics.is_measured(method):
            raise IntegrityError("paid primary requires measured native tokenizer")
    elif backend is not None and getattr(backend, "name", "") == "openai":
        raise IntegrityError("paid backend forbidden without --execute-paid")
    out, selected, manifest = _open_or_create(args, tasks, method)
    configs = {c: _config(manifest, c, out) for c in CONDITIONS}
    cache, completed = _validate_existing(out, selected, manifest, configs)
    backend = backend or (baseline._Retrying(OpenAIBackend("gpt-4o-mini", temperature=0.0))
                          if args.execute_paid else MockBackend(follows_injection=False))
    instrument.traced(backend)  # install once before concurrent workers
    upstream_dir = out / "shared_agent1"
    upstream = tracing.TraceWriter(str(upstream_dir))
    cache_writer = tracing.TraceWriter(str(out))
    writers = {c: tracing.TraceWriter(str(out / manifest["arms"][c]["run_dir"])) for c in CONDITIONS}

    def prepare(task, repeat):
        key = (task["id"], repeat)
        if key in cache:
            return cache[key]
        rec = {"task_id": task["id"], "repeat": repeat, "shared_handoff_id": uuid.uuid4().hex,
               "task_sha256": sha256(canonical(task)), "dataset_sha256": DATASET_SHA256,
               "experiment_id": manifest["experiment_id"], "manifest_sha256": manifest["manifest_sha256"],
               "agent1_input": pipeline._build_agent1_input(task), "agent1_system": pipeline.AGENT1_SYSTEM}
        upstream.write("attempts", {"task_id": task["id"], "repeat": repeat,
                                    "shared_handoff_id": rec["shared_handoff_id"]})
        _fsync(upstream_dir / "attempts.jsonl")
        ctx = instrument.TaskRunContext(run_id=manifest["experiment_id"] + "-agent1",
              task_id=task["id"], repeat=repeat, budget_tokens=150, condition="shared_agent1",
              model="gpt-4o-mini", temperature=0.0, trace=upstream)
        try:
            with instrument.run_context(ctx):
                rec.update(pipeline.prepare_agent1(task, backend))
            rec["status"] = "ready"
        except Exception as exc:
            # Preserve any response even if preparation failed after transport.
            raw = ctx.calls[-1].get("raw_output") if ctx.calls else None
            rec.update(status="error", agent1_raw=raw, handoff=None,
                       parsed_agent1=None, read_calls=[], error_type=type(exc).__name__)
        rec["agent1_output_sha256"] = sha256(rec["agent1_raw"].encode()) if rec["agent1_raw"] is not None else None
        rec["handoff_sha256"] = sha256(rec["handoff"].encode()) if rec["handoff"] is not None else None
        rec["record_sha256"] = _digest_record(rec, "record_sha256")
        validate_cache_record(rec, task, manifest)
        _fsync(upstream_dir / "calls.jsonl")
        cache_writer.write("handoffs", rec)
        _fsync(out / "handoffs.jsonl")
        cache[key] = rec  # only visible after durable cache is committed
        return rec

    def arm(task, repeat, condition):
        key = (task["id"], repeat)
        if key in completed[condition]:
            return
        shared = prepare(task, repeat)
        writer, cfg = writers[condition], configs[condition]
        arm_dir = out / manifest["arms"][condition]["run_dir"]
        if shared["status"] == "error":
            writer.write("runs", _error_row(task, repeat, cfg, shared, shared["error_type"], "agent1"))
        else:
            writer.write("attempts", {"task_id": task["id"], "repeat": repeat,
                                      "shared_handoff_id": shared["shared_handoff_id"]})
            _fsync(arm_dir / "attempts.jsonl")
            try:
                pipeline.run_pipeline(task, backend, backend, condition, repeat=repeat,
                                      prepared_agent1=shared, run_config=cfg, trace_writer=writer)
            except Exception as exc:
                writer.write("runs", _error_row(task, repeat, cfg, shared, type(exc).__name__, "downstream"))
        _fsync(arm_dir / "calls.jsonl")
        _fsync(arm_dir / "runs.jsonl")
        completed[condition][key] = True

    try:
        planned = [(task, repeat) for repeat in range(args.repeats) for task in selected]
        # All baseline trials finish before any treatment; every arm reuses cache.
        for condition in CONDITIONS:
            with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(lambda item: arm(*item, condition), planned))
            print(f"{condition}: {len(completed[condition])}/{len(planned)} terminal trials", flush=True)
    finally:
        upstream.close()
        cache_writer.close()
        for writer in writers.values():
            writer.close()
    cache, completed = _validate_existing(out, selected, manifest, configs)
    errors = sum(bool(r.get("api_failure")) for rows in completed.values() for r in rows.values())
    status = {"experiment_id": manifest["experiment_id"], "manifest_sha256": manifest["manifest_sha256"],
              "status": "complete_with_errors" if errors else "complete", "backend": manifest["backend"],
              "shared_handoffs": len(cache), "terminal_trials": sum(map(len, completed.values())),
              "api_or_pipeline_failures": errors, "scientific_model_results": manifest["backend"] != "mock"}
    _durable_json(out / "status.json", status)
    return status


def run(args, *, backend=None):
    """Hold one experiment-wide writer lock, including all resume checks."""
    out = Path(args.out_dir).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # A sibling survives a failed creation without making the new root nonempty.
    lock_path = out.parent / ("." + out.name + ".lock")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise IntegrityError("another writer owns this experiment directory") from exc
        try:
            return _run_locked(args, backend=backend)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--experiment-id")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="prefix subset for an explicitly labelled smoke test")
    ap.add_argument("--execute-paid", action="store_true", help="explicit opt-in to future real API execution")
    return ap


def main(argv=None):
    try:
        status = run(parser().parse_args(argv))
    except (IntegrityError, OSError, ValueError) as exc:
        print(f"shared_run blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(status, sort_keys=True))
    return 1 if status["api_or_pipeline_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
