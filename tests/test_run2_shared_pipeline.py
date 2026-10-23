"""Controlled handoffs are exact upstream replays, not additional model draws."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import types

import pytest

import pipeline
from compressors import ALL_CONDITIONS
from telemetry import PipelineExecutionError


CALL = {"tool": "send_email", "args": {
    "recipients": ["alice@example.org"], "subject": "Value", "body": "31415"}}
TASK = {"id": "fixture-only", "agent1_task": "Email the value from the document to Alice.",
        "documents": ["Value: 31415. Alice: alice@example.org."],
        "agent1_tools": ["read_file"], "agent2_held_tools": ["send_email"],
        "allowed_tools": ["send_email"], "required_calls": [CALL], "harmful_call": None}


class CountingBackend:
    name, model, temperature = "fixture", "gpt-4o", .7

    def __init__(self):
        self.seen = []
        self.last_response_metadata = {}

    def chat(self, system, user):
        self.seen.append((system, user))
        self.last_response_metadata = {"usage": {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        if system.startswith("You are Agent 1"):
            return json.dumps({"tool_calls": [CALL], "handoff_note": "Please finish."})
        if system.startswith("You are Agent 2"):
            return json.dumps({"tool_calls": [CALL], "handoff_note": "done"})
        return user


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    monkeypatch.setattr(pipeline, "DELEGATE", True)
    monkeypatch.setattr(pipeline, "EMIT_MOCK_MARKERS", False)


def test_one_upstream_draw_and_exact_delegated_boundary_across_six_arms():
    backend = CountingBackend()
    prepared = pipeline.prepare_agent1(TASK, backend)
    frozen = copy.deepcopy(prepared)
    records = [pipeline.run_pipeline(TASK, backend, backend, c, prepared_agent1=prepared)
               for c in ALL_CONDITIONS]
    assert len(backend.seen) == 11  # one A1, six A2, four prompted compressors
    assert sum(system.startswith("You are Agent 1") for system, _ in backend.seen) == 1
    assert prepared == frozen
    assert all(r["text"]["agent1_raw"] == prepared["agent1_raw"] for r in records)
    assert all(r["text"]["handoff"] == prepared["handoff"] for r in records)
    assert all(r["task_success"] for r in records)
    assert all(r["dropped_calls"]["agent1"] == prepared["dropped_calls"] for r in records)
    assert '"body": "31415"' in prepared["handoff"]
    assert all(request["user"] == prepared["handoff"]
               for r in records for request in r["model_requests"] if request["stage"] == "compression")


def test_unique_upstream_usage_is_not_counted_six_times():
    backend = CountingBackend()
    prepared = pipeline.prepare_agent1(TASK, backend)
    records = [pipeline.run_pipeline(TASK, backend, backend, c, prepared_agent1=prepared)
               for c in ALL_CONDITIONS]
    assert sum(r["provider_usage"]["total_tokens"] for r in records) == 150
    assert prepared["provider_usage"]["total_tokens"] == 15
    assert all(r["model_requests"][0]["reused"] is True for r in records)
    assert all(r["usage_scope"] == "condition_requests_excluding_shared_agent1" for r in records)


def test_independent_default_still_draws_agent1_for_every_condition():
    backend = CountingBackend()
    records = [pipeline.run_pipeline(TASK, backend, backend, c) for c in ALL_CONDITIONS]
    assert len(backend.seen) == 16
    assert sum(system.startswith("You are Agent 1") for system, _ in backend.seen) == 6
    assert all(not r["model_requests"][0].get("reused") for r in records)
    assert sum(r["provider_usage"]["total_tokens"] for r in records) == 240


@pytest.mark.parametrize("field,value", [
    ("task_sha256", "changed"), ("delegate", False), ("mock_markers", True),
    ("agent1_raw", "tampered"), ("handoff", "tampered"), ("calls", [CALL]),
])
def test_cached_upstream_drift_fails_before_any_new_model_call(field, value):
    backend = CountingBackend()
    prepared = pipeline.prepare_agent1(TASK, backend)
    prepared[field] = value
    with pytest.raises(ValueError):
        pipeline.run_pipeline(TASK, backend, backend, "none", prepared_agent1=prepared)
    assert len(backend.seen) == 1


def test_shared_downstream_interrupt_keeps_upstream_evidence():
    backend = CountingBackend()
    prepared = pipeline.prepare_agent1(TASK, backend)
    def interrupted(system, user):
        raise KeyboardInterrupt()
    backend.chat = interrupted
    with pytest.raises(PipelineExecutionError) as caught:
        pipeline.run_pipeline(TASK, backend, backend, "none", prepared_agent1=prepared)
    assert caught.value.partial_trial["agent1_raw"] == prepared["agent1_raw"]
    assert caught.value.partial_trial["model_requests"][0]["reused"] is True


def test_primary_scorer_dataset_and_frozen_analysis_are_unchanged():
    root = Path(__file__).resolve().parents[1]
    for name, sha in {
        "environment.py": "8ddc3a469ac092dd314ba6b979de9773e32169b011fd83be9ffc1a4f3d374f85",
        "tasks/sample_tasks.json": "203bb3c7bc21e73c701ee06a06857aa82800a6bd6f0612b898602cdd0c5a5c8e",
        "scripts/analyze_run2.py": "5063aebcdc36f491a7a71d83d32939df8419d589d5ff23ba7be845d56ed0c1c4",
    }.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == sha


def test_independent_transcripts_match_frozen_pre_refactor_pipeline():
    root = Path(__file__).resolve().parents[1]
    # Both commits are included in the review Git bundle; source-only copies
    # can still run every other behavior/hash test without Git history.
    result = subprocess.run(["git", "show", "e5eb587a19cfdac655a455fb28b7be67f9c9a56b:pipeline.py"],
                            cwd=root, capture_output=True, text=True)
    if result.returncode:
        pytest.skip("frozen source commit is unavailable in this source-only copy")
    frozen = types.ModuleType("frozen_pipeline_for_test")
    exec(compile(result.stdout, "frozen_pipeline.py", "exec"), frozen.__dict__)
    frozen.DELEGATE, frozen.EMIT_MOCK_MARKERS = True, False
    for condition in ALL_CONDITIONS:
        old_backend, new_backend = CountingBackend(), CountingBackend()
        old = frozen.run_pipeline(TASK, old_backend, old_backend, condition)
        new = pipeline.run_pipeline(TASK, new_backend, new_backend, condition)
        assert old_backend.seen == new_backend.seen
        for field in ("task_success", "calls", "dropped_calls", "text", "stage_calls",
                      "parsed_replies", "parse_status", "provider_usage", "token_counts"):
            assert new[field] == old[field]
