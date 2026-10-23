#!/usr/bin/env python3
"""Reproducible benign Run 2; plan and mock are explicitly non-API modes.

Examples (run from the repository root):
    python scripts/run2.py --phase plan --out-dir experiments/run2-plan
    python scripts/run2.py --phase smoke --backend mock --out-dir experiments/mock-smoke
    python scripts/run2.py --phase preflight --out-dir experiments/preflight

Real smoke/full additionally require the user-approved OPENAI_API_KEY source,
explicit input/output prices, --max-cost-usd and --max-output-tokens. No paid
request is permitted without those settings; preflight never sends requests.

The default full sweep is 18 tasks x 6 conditions x 3 independent pipeline draws.
Optional --design shared-handoff reuses one durable Agent 1 output per task/repeat
across all six conditions; compression and Agent 2 still receive fresh draws.
The no-compression baseline is completed before any compressed condition.
No attack tasks, injected payloads, alternate models, or silent mock fallbacks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import compressors
from environment import TOOL_REGISTRY, ToolLog, score_run
import pipeline
from run import make_backends
from telemetry import text_size

SCHEMA_VERSION = 1
DESIGNS = ("independent", "shared-handoff")
SHARED_USAGE_SCOPE = "condition_requests_excluding_shared_agent1"
# Every original task field is frozen; only allowed_tools is corrected.
PRESERVED_TASKS_SHA256 = "b8c8cdff7d3086fbdacade3d22bf1b4894b9b06e7369490f070c75adc7b6504d"
TASK_IDS = (
    "ws_6", "ws_7", "ws_12", "ws_21", "ws_29", "ws_31", "ws_32", "ws_15",
    "ws_18", "ws_34", "ws_20", "ws_33", "ws_25", "ws_13", "ws_4", "ws_19",
    "ws_36", "ws_37",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_tasks(path):
    data = Path(path).read_bytes()
    tasks = json.loads(data)
    if not isinstance(tasks, list) or len(tasks) != 18:
        raise ValueError("Run 2 requires the same 18-task evaluation set")
    ids = [t.get("id") for t in tasks]
    if len(set(ids)) != 18 or set(ids) != set(TASK_IDS):
        raise ValueError("Run 2 task IDs must be the 18 committed workspace task IDs, without duplicates")
    for task in tasks:
        tid = task["id"]
        if task.get("harmful_call") is not None or task.get("payload_scope") is not None:
            raise ValueError(f"{tid}: attack metadata is prohibited")
        if "INJECT_CALL:" in json.dumps(task) or "attack" in str(task.get("source", "")).lower():
            raise ValueError(f"{tid}: attack marker/source is prohibited")
        if not isinstance(task.get("documents"), list) or not all(isinstance(d, str) for d in task["documents"]):
            raise ValueError(f"{tid}: documents must be strings")
        if not isinstance(task.get("agent1_task"), str) or not task["agent1_task"].strip():
            raise ValueError(f"{tid}: missing Agent 1 task")
        reads = set(task.get("agent1_tools", []))
        held = set(task.get("agent2_held_tools", task.get("allowed_tools", [])))
        writes = [c for c in task.get("required_calls", []) if c.get("tool") not in reads]
        if not writes:
            raise ValueError(f"{tid}: no required write actions to evaluate")
        if reads & held:
            raise ValueError(f"{tid}: Agent 1/2 tool split overlaps")
        for call in task.get("required_calls", []):
            if call.get("tool") not in TOOL_REGISTRY or not isinstance(call.get("args"), dict):
                raise ValueError(f"{tid}: unregistered required tool or malformed args")
        if not {c["tool"] for c in writes} <= held:
            raise ValueError(f"{tid}: a required write is unavailable to Agent 2")
        if {c["tool"] for c in writes} != set(task.get("allowed_tools", [])):
            raise ValueError(f"{tid}: compression allowlist must contain exactly the required write tools")
    preserved = [{k: v for k, v in task.items() if k != "allowed_tools"} for task in tasks]
    if sha256(canonical(preserved)) != PRESERVED_TASKS_SHA256:
        raise ValueError("task content differs from the frozen scorer-fix evaluation set")
    return tasks, sha256(data), {t["id"]: sha256(canonical(t)) for t in tasks}


def configuration(backend, environ=None, max_output_tokens=None):
    if max_output_tokens is not None and (type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 16384):
        raise ValueError("max_output_tokens must be a positive integer no larger than GPT-4o's 16384-token output limit")
    env = os.environ if environ is None else environ
    expected = {"MAS_BACKEND": backend, "MAS_MODEL": "gpt-4o", "MAS_DELEGATE": "1", "MAS_TEMPERATURE": "0.7"}
    for key, value in expected.items():
        if key in env:
            if key == "MAS_TEMPERATURE":
                agrees = float(env[key]) == float(value)
            else:
                agrees = env[key] == value
            if not agrees:
                raise ValueError(f"{key} conflicts with Run 2 primary configuration ({value})")
    base = env.get("MAS_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    parsed = urlsplit(base)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("MAS_BASE_URL must not include credentials, query, or fragment")
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("MAS_BASE_URL must be an HTTPS API endpoint")
    base = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
    # Avoid calling an unknown OpenAI-compatible service while labelling it GPT-4o.
    if backend == "openai" and base != "https://api.openai.com/v1":
        raise ValueError("Run 2 primary GPT-4o configuration requires https://api.openai.com/v1")
    return {"backend": backend, "model": "gpt-4o" if backend == "openai" else "mock",
            "temperature": 0.7, "delegate": True, "base_url": base if backend == "openai" else None,
            "mock_markers": backend == "mock", "max_output_tokens": max_output_tokens,
            "seed": None, "seed_policy": "unset, as in the original harness; independent API draws, not bit-for-bit reproducible"}


@contextmanager
def configured_environment(config):
    values = {"MAS_BACKEND": config["backend"], "MAS_MODEL": "gpt-4o", "MAS_TEMPERATURE": "0.7", "MAS_DELEGATE": "1"}
    old = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def code_identity(root=ROOT):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    # Include all pipeline implementation modules, including newly added telemetry.
    paths = sorted(root.glob("*.py")) + [root / "scripts" / "run2.py"]
    files = {str(path.relative_to(root)): sha256(path.read_bytes()) for path in paths}
    return {"commit": git("rev-parse", "HEAD"), "tree": git("rev-parse", "HEAD^{tree}"),
            "dirty": bool(git("status", "--porcelain", "--", *files)), "files": files}


def runtime_identity(config):
    try:
        version = importlib.metadata.version("tiktoken")
    except importlib.metadata.PackageNotFoundError:
        version = None
    # Tiktoken downloads missing BPE tables on first use. Preflight must not
    # perform even that network request: only load an already cached encoding.
    cache = os.environ.get("TIKTOKEN_CACHE_DIR", os.environ.get("DATA_GYM_CACHE_DIR", str(Path(tempfile.gettempdir()) / "data-gym-cache")))
    cache_key = hashlib.sha1(b"https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken").hexdigest()
    cache_path = Path(cache) / cache_key if cache else None
    expected_cache_sha256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
    cached = bool(cache_path and cache_path.is_file() and sha256(cache_path.read_bytes()) == expected_cache_sha256)
    tokenizer = text_size("", config["model"]) if version and cached else {"tokenizer": "unavailable", "tokens": None}
    return {"python": sys.version, "implementation": platform.python_implementation(),
            "tiktoken_version": version, "tokenizer": {
                "name": tokenizer["tokenizer"], "available": tokenizer["tokens"] is not None}}


def make_identity(tasks, dataset_sha, task_sha, config, phase, selected_ids, code=None, budget=None, design="independent"):
    if design not in DESIGNS:
        raise ValueError("unknown experimental design")
    repeats = 1 if phase == "smoke" else 3
    return {"design": design, "dataset_sha256": dataset_sha, "task_sha256": task_sha,
            "all_task_ids": [t["id"] for t in tasks], "task_ids": selected_ids,
            "configuration": config, "conditions": list(compressors.ALL_CONDITIONS),
            "repeats": repeats, "prompt_version": pipeline.PROMPT_VERSION,
            "compression": {"budget_words": compressors.BUDGET_WORDS, "sliding_window_chars": 600},
            "code": code_identity() if code is None else code, "runtime": runtime_identity(config),
            "budget": budget or {"cap_usd": None, "input_usd_per_million": None, "output_usd_per_million": None, "credential_source": None}}


def bind_manifest(directory, identity, phase, run_id=None):
    path = directory / "manifest.json"
    fingerprint = sha256(canonical(identity))
    if path.exists():
        manifest = json.loads(path.read_text())
        recorded_hash = manifest.get("manifest_sha256")
        unhashed = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
        if recorded_hash != sha256(canonical(unhashed)):
            raise ValueError("existing manifest checksum mismatch")
        if manifest.get("identity") != identity or manifest.get("phase") != phase or manifest.get("config_fingerprint") != fingerprint:
            raise ValueError("resume rejected: dataset, configuration, code, phase, or task selection changed; use a new output directory")
        if run_id is not None and run_id != manifest.get("run_id"):
            raise ValueError("resume rejected: run ID changed")
        return manifest
    if any(directory.iterdir()):
        raise ValueError("new output directory must be empty (no manifest found)")
    manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id or f"run2-{phase}-{uuid.uuid4().hex[:12]}",
                "phase": phase, "created_at": utc_now(), "config_fingerprint": fingerprint, "identity": identity,
                "experimental_design": ("independent full-pipeline draws per condition; pair by task and repeat; all baseline trials first"
                    if identity.get("design", "independent") == "independent" else
                    "one durable Agent1 output per task/repeat shared across all six conditions; fresh compression and Agent2 draws; all baseline trials first; unique upstream usage in handoffs.jsonl")}
    manifest["manifest_sha256"] = sha256(canonical(manifest))
    # Exclusive creation prevents accidentally replacing another run's identity.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def schedule(identity):
    return [(tid, condition, repeat) for condition in identity["conditions"]
            for repeat in range(identity["repeats"]) for tid in identity["task_ids"]]


def validate_record(record, manifest):
    identity = manifest["identity"]
    key = (record.get("task_id"), record.get("condition"), record.get("repeat"))
    if key not in set(schedule(identity)):
        raise ValueError(f"journal record is outside this run's schedule: {key}")
    if record.get("run_id") != manifest["run_id"] or record.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("journal record belongs to a different run/manifest")
    if record.get("config_fingerprint") != manifest["config_fingerprint"] or record.get("configuration") != identity["configuration"]:
        raise ValueError("journal configuration mismatch")
    if record.get("task_sha256") != identity["task_sha256"][key[0]]:
        raise ValueError("journal task content mismatch")
    if record.get("code_commit") != identity["code"]["commit"]:
        raise ValueError("journal code commit mismatch")
    if not isinstance(record.get("task_success"), bool) or record.get("attack_fired") is not False:
        raise ValueError("journal record lacks valid no-attack task outcome")
    if not isinstance(record.get("calls"), list) or not isinstance(record.get("text"), dict):
        raise ValueError("journal record lacks raw calls/text")
    design = identity.get("design", "independent")
    if record.get("design", "independent") != design:
        raise ValueError("journal mixes experimental designs")
    if design == "independent" and any(record.get(field) is not None for field in ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256")):
        raise ValueError("independent trial unexpectedly links a shared handoff")
    if design == "shared-handoff":
        if record.get("usage_scope") != SHARED_USAGE_SCOPE or record.get("shared_agent1_usage_scope") != "recorded_once_in_handoffs.jsonl":
            raise ValueError("shared trial usage accounting scope is missing")
        for field in ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError("shared trial lacks its immutable handoff link")
    config = identity["configuration"]
    for name in ("model", "temperature", "delegate", "backend"):
        if record.get(name) != config[name]:
            raise ValueError(f"journal top-level {name} differs from configuration")
    if record.get("mock") is not (config["backend"] == "mock"):
        raise ValueError("journal mock label mismatch")
    if record.get("prompt_version") != identity["prompt_version"]:
        raise ValueError("journal prompt version mismatch")
    try:
        uuid.UUID(record.get("trial_attempt_id", ""))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("journal record lacks a valid trial attempt UUID") from None
    expected_stages = ["agent1", "agent2"] if key[1] in ("none", "sliding_window") else ["agent1", "compression", "agent2"]
    requests = record.get("model_requests")
    if not isinstance(requests, list) or [r.get("stage") for r in requests] != expected_stages:
        raise ValueError("journal has missing or unexpected model request stages")
    for request in requests:
        if request.get("status") != "completed" or not isinstance(request.get("response"), str):
            raise ValueError("journal has incomplete model request stages")
        if request.get("model") != config["model"] or request.get("backend") != config["backend"]:
            raise ValueError("journal request model/backend differs from configuration")
        if not isinstance(request.get("system"), str) or not isinstance(request.get("user"), str):
            raise ValueError("journal is missing model prompts")
        if config["backend"] == "openai" and any(marker in request["user"] for marker in ("REQUIRED_CALL:", "NEXT_CALL:", "INJECT_CALL:")):
            raise ValueError("real model input contains mock answer/attack markers")
    if design == "shared-handoff":
        if requests[0].get("reused") is not True:
            raise ValueError("shared trial Agent1 trace must be marked reused and excluded from arm usage")
        if any(request.get("reused") is True for request in requests[1:]):
            raise ValueError("shared trial downstream requests must be fresh")
    elif any(request.get("reused") is True for request in requests):
        raise ValueError("independent trial unexpectedly reuses an upstream request")
    return key


def load_journal(path, manifest, repair=True):
    if not path.exists():
        return {}
    raw = path.read_bytes()
    records = {}
    offset = 0
    for index, line in enumerate(raw.splitlines(keepends=True)):
        end = offset + len(line)
        # A crash can leave an incomplete tail. Never discard corruption in a
        # newline-terminated record or any record before the final tail.
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not line.endswith(b"\n") and end == len(raw) and repair:
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f"malformed completed journal line {index + 1}")
        key = validate_record(record, manifest)
        if key in records:
            raise ValueError(f"duplicate trial in journal: {key}")
        records[key] = record
        if end == len(raw) and not line.endswith(b"\n") and repair:
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        offset = end
    return records


def make_shared_handoff(prepared, task_id, repeat, manifest):
    """Wrap one completed upstream call before any downstream work starts."""
    artifact = {"schema_version": SCHEMA_VERSION, "shared_handoff_id": str(uuid.uuid4()),
        "run_id": manifest["run_id"], "manifest_sha256": manifest["manifest_sha256"],
        "config_fingerprint": manifest["config_fingerprint"], "task_id": task_id,
        "task_sha256": manifest["identity"]["task_sha256"][task_id], "repeat": repeat,
        "created_at": utc_now(), "prepared_agent1": prepared,
        "prepared_agent1_sha256": sha256(canonical(prepared)),
        "usage_scope": "unique_agent1_request_recorded_once"}
    artifact["shared_handoff_sha256"] = sha256(canonical(artifact))
    return artifact


def validate_shared_handoff(artifact, manifest):
    identity = manifest["identity"]
    if identity.get("design", "independent") != "shared-handoff":
        raise ValueError("independent run unexpectedly contains a shared handoff cache")
    if not isinstance(artifact, dict):
        raise ValueError("shared handoff artifact must be an object")
    key = (artifact.get("task_id"), artifact.get("repeat"))
    if type(key[1]) is not int or key not in {(tid, repeat) for tid, _, repeat in schedule(identity)}:
        raise ValueError("shared handoff task/repeat is outside the run schedule")
    try:
        uuid.UUID(artifact.get("shared_handoff_id", ""))
    except (ValueError, TypeError, AttributeError):
        raise ValueError("invalid shared handoff UUID") from None
    for name in ("run_id", "manifest_sha256", "config_fingerprint"):
        if artifact.get(name) != manifest[name]:
            raise ValueError("shared handoff belongs to a different manifest/run")
    if artifact.get("task_sha256") != identity["task_sha256"][key[0]]:
        raise ValueError("shared handoff task hash mismatch")
    unhashed = {k: v for k, v in artifact.items() if k != "shared_handoff_sha256"}
    if artifact.get("shared_handoff_sha256") != sha256(canonical(unhashed)):
        raise ValueError("shared handoff cache checksum mismatch")
    prepared = artifact.get("prepared_agent1")
    if not isinstance(prepared, dict) or artifact.get("prepared_agent1_sha256") != sha256(canonical(prepared)):
        raise ValueError("prepared Agent1 checksum mismatch")
    config = identity["configuration"]
    expected = {"task_id": key[0], "task_sha256": identity["task_sha256"][key[0]],
                "prompt_version": identity["prompt_version"], "delegate": config["delegate"],
                "mock_markers": config["mock_markers"]}
    if any(prepared.get(name) != value for name, value in expected.items()):
        raise ValueError("prepared Agent1 task/prompt/configuration mismatch")
    requests = prepared.get("model_requests")
    if not isinstance(requests, list) or len(requests) != 1 or not isinstance(requests[0], dict):
        raise ValueError("shared handoff must contain exactly one upstream model request")
    from telemetry import usage_totals
    if prepared.get("provider_usage") != usage_totals(requests):
        raise ValueError("cached upstream provider usage differs from its unique request")
    request = requests[0]
    if request.get("stage") != "agent1" or request.get("status") != "completed" or request.get("reused") is True:
        raise ValueError("cached upstream model request is incomplete or already reused")
    if request.get("backend") != config["backend"] or request.get("model") != config["model"]:
        raise ValueError("cached upstream backend/model mismatch")
    if not isinstance(request.get("response"), str) or request["response"] != prepared.get("agent1_raw"):
        raise ValueError("cached upstream raw response differs from its request trace")
    if not isinstance(request.get("user"), str) or not isinstance(request.get("system"), str):
        raise ValueError("cached upstream prompts must be strings")
    if request.get("user") != prepared.get("agent1_input"):
        raise ValueError("cached upstream input differs from its request trace")
    if config["backend"] == "openai" and any(marker in request["user"] for marker in ("REQUIRED_CALL:", "NEXT_CALL:", "INJECT_CALL:")):
        raise ValueError("real upstream cache contains answer/attack markers")
    # The pipeline's pure validator reconstructs reads, delegation and the
    # handoff from raw output. Checksums cannot legitimize edited content.
    task = next(t for t in json.loads((ROOT / "tasks" / "sample_tasks.json").read_text()) if t["id"] == key[0])
    old_flags = pipeline.DELEGATE, pipeline.EMIT_MOCK_MARKERS
    pipeline.DELEGATE, pipeline.EMIT_MOCK_MARKERS = config["delegate"], config["mock_markers"]
    try:
        pipeline.validate_prepared_agent1(prepared, task)
    finally:
        pipeline.DELEGATE, pipeline.EMIT_MOCK_MARKERS = old_flags
    return key


def load_shared_handoffs(path, manifest, repair=True):
    """Read the durable upstream journal; repair only an interrupted tail."""
    path = Path(path)
    if not path.exists():
        return {}
    if manifest["identity"].get("design", "independent") != "shared-handoff":
        raise ValueError("independent run unexpectedly contains handoffs.jsonl")
    raw, artifacts, ids, offset = path.read_bytes(), {}, set(), 0
    for number, line in enumerate(raw.splitlines(keepends=True), 1):
        end = offset + len(line)
        try:
            artifact = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            if repair and end == len(raw) and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            raise ValueError(f"malformed completed handoff cache line {number}") from None
        key = validate_shared_handoff(artifact, manifest)
        if key in artifacts or artifact["shared_handoff_id"] in ids:
            raise ValueError("duplicate shared handoff task/repeat or UUID")
        artifacts[key] = artifact
        ids.add(artifact["shared_handoff_id"])
        if repair and end == len(raw) and not line.endswith(b"\n"):
            with path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        offset = end
    return artifacts


def validate_shared_links(records, handoffs, manifest):
    """Fail closed if an observed arm lost or changed its upstream artifact."""
    design = manifest["identity"].get("design", "independent")
    if design != "shared-handoff":
        if handoffs:
            raise ValueError("independent run has unexpected shared upstream artifacts")
        return
    from telemetry import usage_totals
    for record in records.values():
        group = (record["task_id"], record["repeat"])
        if group not in handoffs:
            raise ValueError("completed shared trial is missing its durable upstream cache")
        artifact, prepared = handoffs[group], handoffs[group]["prepared_agent1"]
        if any(record.get(name) != artifact[name] for name in
               ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256")):
            raise ValueError("shared trial/cache handoff link mismatch")
        if any(record["text"].get(name) != prepared[name] for name in
               ("agent1_input", "agent1_raw", "handoff")):
            raise ValueError("shared trial text differs from the cached upstream output")
        if (record.get("stage_calls", {}).get("agent1") != prepared["calls"]
                or record.get("dropped_calls", {}).get("agent1") != prepared["dropped_calls"]
                or record.get("parsed_replies", {}).get("agent1") != prepared["parsed_reply"]
                or record.get("parse_status", {}).get("agent1") != prepared["parse_status"]):
            raise ValueError("shared trial Agent1 calls/parsing differ from its upstream cache")
        trace = {k: v for k, v in record["model_requests"][0].items() if k != "reused"}
        if trace != prepared["model_requests"][0]:
            raise ValueError("shared trial upstream trace differs from its cache")
        if record.get("provider_usage") != usage_totals(record["model_requests"][1:]):
            raise ValueError("shared arm provider usage includes cached upstream usage or is incomplete")


def verify_smoke(smoke_dir, identity):
    path = Path(smoke_dir)
    smoke = json.loads((path / "manifest.json").read_text())
    unhashed = {k: v for k, v in smoke.items() if k != "manifest_sha256"}
    if smoke.get("manifest_sha256") != sha256(canonical(unhashed)):
        raise ValueError("smoke manifest checksum mismatch")
    if smoke.get("phase") != "smoke" or smoke["identity"]["configuration"]["backend"] != "openai":
        raise ValueError("full API sweep requires a real OpenAI smoke test")
    for key in ("design", "configuration", "dataset_sha256", "task_sha256", "all_task_ids", "conditions", "code", "compression", "prompt_version", "runtime"):
        if smoke["identity"].get(key) != identity.get(key):
            raise ValueError(f"smoke/full {key} mismatch")
    # Smoke and full use separate spending caps, but identical prices/source
    # and the same max-output inference setting in configuration.
    for key in ("input_usd_per_million", "output_usd_per_million", "credential_source"):
        if smoke["identity"].get("budget", {}).get(key) != identity.get("budget", {}).get(key):
            raise ValueError(f"smoke/full budget {key} mismatch")
    if not set(smoke["identity"]["task_ids"]) <= set(identity["task_ids"]) or len(smoke["identity"]["task_ids"]) < 2:
        raise ValueError("smoke test must cover at least two evaluation tasks")
    done = load_journal(path / "raw.jsonl", smoke, repair=False)
    shared = load_shared_handoffs(path / "handoffs.jsonl", smoke, repair=False)
    validate_shared_links(done, shared, smoke)
    tasks_by_id = {t["id"]: t for t in json.loads((ROOT / "tasks" / "sample_tasks.json").read_text())}
    for record in done.values():
        rescored = score_run(ToolLog(calls=record["calls"]), tasks_by_id[record["task_id"]])
        if rescored["task_success"] != record["task_success"]:
            raise ValueError("smoke task_success differs from current scorer applied to raw calls")
        for request in record["model_requests"]:
            if not request["response"].strip():
                raise ValueError("smoke contains an empty model response; inspect before full sweep")
            response_model = request.get("provider", {}).get("response_model", "")
            if response_model != "gpt-4o" and not response_model.startswith("gpt-4o-20"):
                raise ValueError("smoke provider response model is missing or is not GPT-4o")
    if len(done) != len(schedule(smoke["identity"])):
        raise ValueError("real smoke test is incomplete")
    status = json.loads((path / "status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("real smoke test status is not complete")
    # A zero-success baseline or missing replies is a harness/quality red flag;
    # inspect rather than spending on a full sweep after a meaningless smoke.
    baseline = [r for r in done.values() if r["condition"] == "none"]
    if not any(r["task_success"] for r in baseline):
        raise ValueError("smoke baseline has zero successes; inspect it before full API sweep")
    for record in done.values():
        text = record.get("text", {})
        if not text.get("agent1_raw") or not text.get("agent2_raw") or not text.get("compressed", "").strip():
            raise ValueError("smoke contains missing model replies or empty compressed handoff")
    return smoke["run_id"]


def error_category(exc):
    """Classify known API blockers without persisting potentially secret text."""
    cause = exc.__cause__ or exc
    detail = str(cause).lower()
    if "insufficient_quota" in detail or "no credits" in detail:
        return "insufficient_quota"
    if "401" in detail or "authentication" in detail or "invalid_api_key" in detail:
        return "authentication"
    if "daily" in detail or "per day" in detail:
        return "daily_request_limit"
    if "429" in detail or "rate limit" in detail:
        return "rate_limit"
    if "budget" in detail or type(cause).__name__ in ("BudgetError", "BudgetExceeded"):
        return "budget_limit"
    if "timeout" in detail or "timed out" in detail:
        return "timeout"
    if isinstance(cause, KeyboardInterrupt):
        return "user_interruption"
    return "infrastructure_or_pipeline_error"


def write_status(directory, manifest, state, completed, **extra):
    atomic_json(directory / "status.json", {"run_id": manifest["run_id"], "status": state,
                "phase": manifest["phase"], "mock": manifest["identity"]["configuration"]["backend"] == "mock",
                "completed_trials": completed, "planned_trials": len(schedule(manifest["identity"])),
                "updated_at": utc_now(), **extra})


def budget_configuration(args):
    def normalized(value, name):
        if value is None:
            return None
        try:
            number = Decimal(value)
        except InvalidOperation:
            raise ValueError(f"{name} must be a positive decimal") from None
        if not number.is_finite() or number <= 0:
            raise ValueError(f"{name} must be a positive decimal")
        return format(number.normalize(), "f")
    return {"cap_usd": normalized(args.max_cost_usd, "--max-cost-usd"),
            "input_usd_per_million": normalized(args.input_usd_per_million, "--input-usd-per-million"),
            "output_usd_per_million": normalized(args.output_usd_per_million, "--output-usd-per-million"),
            "credential_source": args.credential_source}


def preflight_checks(identity):
    config, budget = identity["configuration"], identity["budget"]
    checks = {"frozen_18_task_dataset": True,
              "scorer_sha256": identity["code"]["files"].get("environment.py"),
              "credential_available": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
              "approved_credential_source": budget.get("credential_source") == "OPENAI_API_KEY",
              "positive_cost_cap": budget.get("cap_usd") is not None,
              "explicit_input_price": budget.get("input_usd_per_million") is not None,
              "explicit_output_price": budget.get("output_usd_per_million") is not None,
              "positive_max_output_tokens": config.get("max_output_tokens") is not None,
              "tokenizer_available": identity["runtime"]["tokenizer"]["available"]}
    blockers = []
    if config["backend"] == "openai":
        messages = {"credential_available": "OPENAI_API_KEY is not available in the process environment",
                    "approved_credential_source": "user-approved credential source is missing (--credential-source OPENAI_API_KEY)",
                    "positive_cost_cap": "user-approved spending cap is missing (--max-cost-usd)",
                    "explicit_input_price": "explicit API input price is missing (--input-usd-per-million)",
                    "explicit_output_price": "explicit API output price is missing (--output-usd-per-million)",
                    "positive_max_output_tokens": "completion bound is missing (--max-output-tokens)",
                    "tokenizer_available": "GPT-4o tokenizer unavailable; install tiktoken and warm its o200k_base cache"}
        blockers = [message for name, message in messages.items() if not checks[name]]
    return {"checks": checks, "blockers": blockers, "api_calls_made": 0,
            "credential_value_recorded": False, "budget": budget}


def planning_estimate(identity, tasks):
    """Planning assumptions only; the durable per-request ledger enforces cap.

    Assumes handoff length <= twice the Agent 1 completion bound because
    delegation can repeat actions. This is an estimate, not a spend guarantee.
    Actual tokenized prompt bounds and every retry are reserved by the backend.
    """
    order = schedule(identity)
    shared = identity.get("design", "independent") == "shared-handoff"
    groups = {(tid, repeat) for tid, _, repeat in order}
    requests = (len(groups) + sum(1 if cond in ("none", "sliding_window") else 2 for _, cond, _ in order)
                if shared else sum(2 if cond in ("none", "sliding_window") else 3 for _, cond, _ in order))
    result = {"planned_trials": len(order), "planned_model_requests": requests,
              "includes_retries": False, "estimated_usd": None,
              "assumptions": "One completion bound per request; handoff <= twice Agent 1 completion bound; 16 chat framing tokens per request; actual spend guard is budget.jsonl."}
    config, budget = identity["configuration"], identity["budget"]
    bound = config.get("max_output_tokens")
    if not identity["runtime"]["tokenizer"]["available"] or bound is None or not budget.get("input_usd_per_million") or not budget.get("output_usd_per_million"):
        result["unavailable_reason"] = "tokenizer, output bound, or explicit prices unavailable"
        return result
    from environment import describe_tools
    by_id = {task["id"]: task for task in tasks}
    prompt_tokens = 0
    counted_agent1 = set()
    for tid, condition, repeat in order:
        task = by_id[tid]
        held = task.get("agent2_held_tools", task.get("allowed_tools", []))
        a1_system = pipeline.AGENT1_SYSTEM.format(a1_tools=describe_tools(task["agent1_tools"]), a2_tools=describe_tools(held))
        # Build only the real task/documents, never the mock answer markers.
        a1_user = "TASK: " + task["agent1_task"] + "\nDOCUMENTS:\n" + "\n".join(f"--- document {i} ---\n{doc}" for i, doc in enumerate(task["documents"]))
        if not shared or (tid, repeat) not in counted_agent1:
            prompt_tokens += sum(text_size(text, config["model"])["tokens"] for text in (a1_system, a1_user)) + 16
            counted_agent1.add((tid, repeat))
        a2_system = pipeline.AGENT2_SYSTEM.format(tools=describe_tools(held))
        handoff = min(600, 2 * bound) if condition == "sliding_window" else 2 * bound if condition == "none" else bound
        prompt_tokens += text_size(a2_system, config["model"])["tokens"] + handoff + 24
        if condition not in ("none", "sliding_window"):
            if condition == "destination":
                comp_system = compressors._DESTINATION.format(role=task.get("agent2_role", "the next agent"), tools=", ".join(task.get("allowed_tools", [])))
            elif condition == "allowlist":
                comp_system = compressors._ALLOWLIST.format(tools=", ".join(task.get("allowed_tools", [])))
            else:
                comp_system = compressors._UNCONDITIONAL if condition == "unconditional" else compressors._INSTRUCTION_STRIP
            prompt_tokens += text_size(comp_system, config["model"])["tokens"] + 2 * bound + 16
    completion_tokens = requests * bound
    cost = (Decimal(prompt_tokens) * Decimal(budget["input_usd_per_million"]) + Decimal(completion_tokens) * Decimal(budget["output_usd_per_million"])) / Decimal(1000000)
    result.update(estimated_prompt_tokens=prompt_tokens, assumed_max_completion_tokens=completion_tokens,
                  estimated_usd=format(cost, "f"), estimate_is_hard_cap=False)
    return result


def run(args):
    tasks, dataset_sha, task_sha = load_tasks(args.tasks)
    config = configuration(args.backend, max_output_tokens=args.max_output_tokens)
    budget_config = budget_configuration(args)
    if args.task_ids and args.phase != "smoke":
        raise ValueError("--task-ids is only permitted for a small smoke test")
    selected = args.task_ids.split(",") if args.task_ids else (["ws_6", "ws_32"] if args.phase == "smoke" else [t["id"] for t in tasks])
    if len(set(selected)) != len(selected) or not set(selected) <= set(TASK_IDS):
        raise ValueError("smoke selection contains duplicate or unknown task IDs")
    identity = make_identity(tasks, dataset_sha, task_sha, config, args.phase, selected, budget=budget_config, design=args.design)
    directory = Path(args.out_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = bind_manifest(directory, identity, args.phase, args.run_id)
    done = load_journal(directory / "raw.jsonl", manifest)
    shared_handoffs = load_shared_handoffs(directory / "handoffs.jsonl", manifest)
    validate_shared_links(done, shared_handoffs, manifest)
    ordered = schedule(identity)
    preflight = preflight_checks(identity)
    estimate = planning_estimate(identity, tasks)
    if args.phase in ("plan", "preflight"):
        atomic_json(directory / "results.json", [])
        atomic_json(directory / "preflight.json", {"run_id": manifest["run_id"], **preflight, "planning_estimate": estimate})
        state = "planned" if args.phase == "plan" else "blocked" if preflight["blockers"] else "ready"
        write_status(directory, manifest, state, 0, planned_model_requests=estimate["planned_model_requests"],
                     api_calls_made=0, blockers=preflight["blockers"], planning_estimate=estimate,
                     note="No API calls. Counts and cost estimate exclude retries; the ledger separately reserves each actual retry.")
        print(f"{args.phase}: {len(ordered)} trials, {estimate['planned_model_requests']} model requests before retries; no API calls.")
        return 2 if args.phase == "preflight" and preflight["blockers"] else 0
    if len(done) == len(ordered):
        atomic_json(directory / "results.json", [done[key] for key in ordered])
        previous_status = json.loads((directory / "status.json").read_text()) if (directory / "status.json").exists() else {}
        write_status(directory, manifest, "complete", len(done), budget=previous_status.get("budget"))
        print(f"Already complete: {len(done)} trials in {directory}")
        return 0
    if preflight["blockers"]:
        atomic_json(directory / "results.json", [done[key] for key in ordered if key in done])
        atomic_json(directory / "preflight.json", {"run_id": manifest["run_id"], **preflight, "planning_estimate": estimate})
        write_status(directory, manifest, "blocked", len(done), blocker="; ".join(preflight["blockers"]),
                     blockers=preflight["blockers"], api_calls_made=0)
        print("Blocked: " + "; ".join(preflight["blockers"]) + ". Zero API calls; no mock fallback.")
        return 2
    if args.phase == "full" and config["backend"] == "openai":
        if not args.smoke_dir:
            write_status(directory, manifest, "blocked", len(done), blocker="matching real smoke test required via --smoke-dir")
            atomic_json(directory / "results.json", [done[key] for key in ordered if key in done])
            return 2
        try:
            verify_smoke(args.smoke_dir, identity)
        except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else "matching smoke artifacts are missing or unreadable"
            atomic_json(directory / "results.json", [done[key] for key in ordered if key in done])
            write_status(directory, manifest, "blocked", len(done), blocker="smoke gate: " + reason, api_calls_made=0)
            print("Blocked by smoke gate: " + reason)
            return 2
    ledger = None
    if config["backend"] == "openai":
        from budget import BudgetLedger
        ledger = BudgetLedger(directory / "budget.jsonl", cap_usd=budget_config["cap_usd"],
            input_per_million_usd=budget_config["input_usd_per_million"],
            output_per_million_usd=budget_config["output_usd_per_million"])
    with configured_environment(config):
        agent, compressor = make_backends()
    if ledger is not None:
        for backend in (agent, compressor):
            backend.budget = ledger
            backend.max_output_tokens = config["max_output_tokens"]
    pipeline.DELEGATE = True
    pipeline.EMIT_MOCK_MARKERS = config["mock_markers"]
    tasks_by_id = {t["id"]: t for t in tasks}
    write_status(directory, manifest, "running", len(done), budget=ledger.snapshot() if ledger else None)
    for key in ordered:
        if key in done:
            continue
        tid, condition, repeat = key
        started = utc_now()
        attempt_id = str(uuid.uuid4())
        record = None
        prepared = None
        shared_artifact = None
        shared_artifact_persisted = False
        try:
            if args.design == "shared-handoff":
                group = (tid, repeat)
                if group not in shared_handoffs:
                    prepared = pipeline.prepare_agent1(tasks_by_id[tid], agent)
                    artifact = make_shared_handoff(prepared, tid, repeat, manifest)
                    shared_artifact = artifact
                    validate_shared_handoff(artifact, manifest)
                    # fsync before any compression/Agent2 call. A downstream
                    # interruption reuses this exact upstream output on resume.
                    append_jsonl(directory / "handoffs.jsonl", artifact)
                    shared_handoffs[group] = artifact
                shared_artifact = shared_handoffs[group]
                shared_artifact_persisted = True
                record = pipeline.run_pipeline(tasks_by_id[tid], agent, compressor, condition,
                    prepared_agent1=shared_artifact["prepared_agent1"])
            else:
                record = pipeline.run_pipeline(tasks_by_id[tid], agent, compressor, condition)
            record.update({"design": args.design, "trial_attempt_id": attempt_id, "run_id": manifest["run_id"], "manifest_sha256": manifest["manifest_sha256"],
                "config_fingerprint": manifest["config_fingerprint"], "task_sha256": task_sha[tid],
                "repeat": repeat, "backend": config["backend"], "mock": config["backend"] == "mock",
                "model": config["model"], "temperature": config["temperature"], "delegate": True,
                "prompt_version": pipeline.PROMPT_VERSION, "configuration": config,
                "code_commit": identity["code"]["commit"], "started_at": started, "finished_at": utc_now()})
            if shared_artifact is not None:
                record.update({field: shared_artifact[field] for field in
                    ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256")})
                record["usage_scope"] = SHARED_USAGE_SCOPE
                record["shared_agent1_usage_scope"] = "recorded_once_in_handoffs.jsonl"
            validate_record(record, manifest)
            if shared_artifact is not None:
                validate_shared_links({key: record}, shared_handoffs, manifest)
            append_jsonl(directory / "raw.jsonl", record)
        except (Exception, SystemExit, KeyboardInterrupt) as exc:
            # Do not count a network/quota/parse infrastructure exception as an
            # observed legitimate-task failure. Raw exceptions can contain API
            # credentials (e.g. URLs); persist type only, with safe guidance.
            error = {"design": args.design, "trial_attempt_id": attempt_id, "run_id": manifest["run_id"], "task_id": tid, "condition": condition,
                     "repeat": repeat, "started_at": started, "finished_at": utc_now(),
                     "error_type": type(exc).__name__, "error_category": error_category(exc),
                     "stage": getattr(exc, "stage", "unknown"), "counts_as_task_failure": False,
                     "manifest_sha256": manifest["manifest_sha256"],
                     "config_fingerprint": manifest["config_fingerprint"], "task_sha256": task_sha[tid],
                     "partial_trial": getattr(exc, "partial_trial", record if record is not None else prepared)}
            if shared_artifact is not None:
                error.update({field: shared_artifact[field] for field in
                    ("shared_handoff_id", "shared_handoff_sha256", "prepared_agent1_sha256")})
                if not shared_artifact_persisted:
                    error["shared_handoff_durability_unconfirmed"] = shared_artifact
            append_jsonl(directory / "errors.jsonl", error)
            atomic_json(directory / "results.json", [done[k] for k in ordered if k in done])
            write_status(directory, manifest, "interrupted", len(done), blocker="Trial raised " + type(exc).__name__ + "; failed trial is unscored and will be retried on resume", last_error={k: v for k, v in error.items() if k != "partial_trial"}, budget=ledger.snapshot() if ledger else None)
            print(f"Stopped at {tid}/{condition}/repeat {repeat}: {type(exc).__name__}. Finished trials are saved; inspect errors.jsonl and rerun.")
            return 1
        done[key] = record
        print(f"[{len(done)}/{len(ordered)}] {tid} {condition} repeat={repeat}: {'ok' if record['task_success'] else 'fail'}", flush=True)
        write_status(directory, manifest, "running", len(done), budget=ledger.snapshot() if ledger else None)
    atomic_json(directory / "results.json", [done[key] for key in ordered])
    write_status(directory, manifest, "complete", len(done), budget=ledger.snapshot() if ledger else None)
    print(f"Completed {len(done)} trials ({'MOCK: not API research evidence' if config['backend'] == 'mock' else 'real model'}): {directory}")
    return 0


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=("plan", "preflight", "smoke", "full"), required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--design", choices=DESIGNS, default="independent", help="independent pipeline draws (default) or one durable Agent1 handoff shared across conditions per task/repeat")
    ap.add_argument("--backend", choices=("openai", "mock"), default="openai")
    ap.add_argument("--tasks", default=str(ROOT / "tasks" / "sample_tasks.json"))
    ap.add_argument("--run-id")
    ap.add_argument("--task-ids", help="comma-separated IDs for smoke phase only")
    ap.add_argument("--smoke-dir", help="completed matching real smoke directory, required for full API sweep")
    ap.add_argument("--credential-source", choices=("OPENAI_API_KEY",), help="environment source explicitly approved by the user; never pass the credential value here")
    ap.add_argument("--max-cost-usd", help="positive user-approved cap for this run, enforced across retries/resume")
    ap.add_argument("--input-usd-per-million", help="explicit model input token price in USD per million")
    ap.add_argument("--output-usd-per-million", help="explicit model output token price in USD per million")
    ap.add_argument("--max-output-tokens", type=int, help="positive completion bound, frozen identically across real smoke/full")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError, KeyError, json.JSONDecodeError) as exc:
        print(f"Run 2 stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
