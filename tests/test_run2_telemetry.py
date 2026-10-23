"""Scripted model responses verify stage boundaries; these are not API results."""
import io
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

import models
import pipeline
from telemetry import PipelineExecutionError, text_size, usage_totals


class ScriptedBackend:
    name, model, temperature = "scripted", "gpt-4o", .7

    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def chat(self, system, user):
        self.seen.append((system, user))
        result = next(self.replies)
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def task():
    return {"id": "test-only", "agent1_task": "Send the value from the document.",
            "agent1_tools": ["read_file"], "agent2_held_tools": ["send_email"],
            "allowed_tools": ["send_email"], "documents": ["Value: 31415"],
            "required_calls": [{"tool": "send_email", "args": {"body": "31415"}}],
            "harmful_call": None}


def reply(calls=None, note="done"):
    return json.dumps({"tool_calls": calls or [], "handoff_note": note})


def test_delegated_actions_cross_compressor_and_trace_preserves_prompts(task, monkeypatch):
    monkeypatch.setattr(pipeline, "DELEGATE", True)
    monkeypatch.setattr(pipeline, "EMIT_MOCK_MARKERS", False)
    call = {"tool": "send_email", "args": {"body": "31415"}}
    backend = ScriptedBackend([reply([call], "Value ready"), "Send 31415", reply([call])])
    result = pipeline.run_pipeline(task, backend, backend, "allowlist")
    assert result["task_success"] is True
    assert [r["stage"] for r in result["model_requests"]] == ["agent1", "compression", "agent2"]
    assert '"body": "31415"' in backend.seen[1][1]
    assert backend.seen[2][1] == "Handoff from Agent 1:\nSend 31415"
    assert "REQUIRED_CALL" not in backend.seen[0][1]
    assert "NEXT_CALL" not in backend.seen[0][1]
    assert result["stage_calls"] == {"agent1": [], "agent2": [call]}
    assert result["dropped_calls"]["agent1"][0]["why"] == "not held"
    assert result["provider_usage"]["total_tokens"] is None
    assert result["text"]["compressed"] != result["text"]["handoff"]
    assert result["reference_calls"] == task["required_calls"]


def test_none_is_exact_passthrough_and_uses_two_requests(task, monkeypatch):
    monkeypatch.setattr(pipeline, "EMIT_MOCK_MARKERS", False)
    backend = ScriptedBackend([reply(note="Exact note \n next"), reply()])
    result = pipeline.run_pipeline(task, backend, backend, "none")
    assert result["text"]["handoff"] == result["text"]["compressed"]
    assert len(result["model_requests"]) == 2


@pytest.mark.parametrize("calls", [123, "send_email", True])
def test_malformed_tool_calls_are_recorded_as_model_failure_without_crash(task, monkeypatch, calls):
    monkeypatch.setattr(pipeline, "DELEGATE", True)
    backend = ScriptedBackend([reply(note={"body": "31415"}),
                               json.dumps({"tool_calls": calls})])
    result = pipeline.run_pipeline(task, backend, backend, "none")
    assert result["task_success"] is False
    assert result["parse_status"]["agent1"] == "structured_handoff"
    assert result["parse_status"]["agent2"] == "malformed_tool_calls"
    assert result["dropped_calls"]["agent2"][0]["why"] == "malformed tool_calls"


def test_failed_stage_keeps_completed_trace_for_resume_audit(task):
    backend = ScriptedBackend([reply(note="31415"), RuntimeError("test network error")])
    with pytest.raises(PipelineExecutionError) as caught:
        pipeline.run_pipeline(task, backend, backend, "unconditional")
    error = caught.value
    assert error.stage == "compression"
    assert error.partial_trial["handoff"] == "31415"
    assert [r["status"] for r in error.partial_trial["model_requests"]] == ["completed", "infrastructure_error"]


def test_keyboard_interrupt_keeps_previous_paid_stage_evidence(task):
    backend = ScriptedBackend([reply(note="31415"), "31415", KeyboardInterrupt()])
    with pytest.raises(PipelineExecutionError) as caught:
        pipeline.run_pipeline(task, backend, backend, "unconditional")
    assert caught.value.stage == "agent2"
    partial = caught.value.partial_trial
    assert partial["agent1_raw"] and partial["compressed"] == "31415"
    assert [r["status"] for r in partial["model_requests"]] == ["completed", "completed", "interrupted"]


def test_openai_request_config_and_returned_usage_are_captured_without_sdk():
    data = {"id": "test-response", "model": "gpt-4o-test", "system_fingerprint": "test-fingerprint",
            "choices": [{"finish_reason": "stop", "message": {"content": "test reply"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}}
    with patch.object(models, "_request_with_retry", return_value=data) as send:
        backend = models.OpenAIBackend("gpt-4o", api_key="test-only-not-a-key", temperature=.7)
        assert backend.chat("system", "user") == "test reply"
    sent = json.loads(send.call_args.args[0].data)
    assert sent["model"] == "gpt-4o" and sent["temperature"] == .7
    assert sent["messages"] == [{"role": "system", "content": "system"}, {"role": "user", "content": "user"}]
    assert backend.last_response_metadata["usage"] == data["usage"]
    assert backend.last_response_metadata["response_model"] == "gpt-4o-test"
    assert "api_key" not in backend.last_response_metadata


def test_http_retry_attempts_counted_and_nontransient_errors_not_retried():
    error = HTTPError("https://api.openai.com/v1/chat/completions", 401, "Unauthorized", {}, io.BytesIO(b'{}'))
    attempts = []
    with patch.object(models.urllib.request, "urlopen", side_effect=error) as send:
        with pytest.raises(SystemExit):
            models._request_with_retry(object(), audit=attempts)
    assert send.call_count == 1
    assert attempts == [{"attempt": 1, "status": "http_error", "http_status": 401}]


def test_provider_refusal_is_a_scored_empty_reply_not_infrastructure_failure(task):
    data = {"choices": [{"finish_reason": "stop", "message": {"content": None, "refusal": "test refusal"}}]}
    with patch.object(models, "_request_with_retry", return_value=data):
        backend = models.OpenAIBackend("gpt-4o", api_key="test-only", temperature=.7)
        result = pipeline.run_pipeline(task, backend, backend, "none")
    assert result["task_success"] is False
    assert result["parse_status"]["agent2"] == "empty_response"
    assert result["model_requests"][1]["provider"]["refusal"] == "test refusal"


def test_budget_cap_stops_http_before_network(tmp_path):
    from budget import BudgetExceeded, BudgetLedger
    backend = models.OpenAIBackend("gpt-4o", api_key="test-only", temperature=.7)
    backend.max_output_tokens = 10
    backend.budget = BudgetLedger(tmp_path / "budget.jsonl", cap_usd=0,
                                  input_per_million_usd="2.5", output_per_million_usd="10")
    with patch.object(models.urllib.request, "urlopen") as send:
        with pytest.raises(BudgetExceeded):
            backend.chat("test system", "test user")
    send.assert_not_called()
    assert backend.last_response_metadata["http_attempts"][0]["status"] == "budget_blocked"


def test_retry_requires_fresh_budget_and_unknown_first_charge_is_reserved(tmp_path):
    from budget import BudgetExceeded, BudgetLedger, prompt_token_bound
    messages = [{"role": "system", "content": "test system"}, {"role": "user", "content": "test user"}]
    cap = str((prompt_token_bound(messages) * 2.5 + 10 * 10) / 1000000 * 1.5)
    backend = models.OpenAIBackend("gpt-4o", api_key="test-only", temperature=.7)
    backend.max_output_tokens = 10
    backend.budget = BudgetLedger(tmp_path / "budget.jsonl", cap_usd=cap,
                                  input_per_million_usd="2.5", output_per_million_usd="10")
    error = HTTPError("https://api.openai.com/v1/chat/completions", 429, "Rate limited", {}, io.BytesIO(b'{}'))
    with patch.object(models.urllib.request, "urlopen", side_effect=error) as send, patch.object(models.time, "sleep"):
        with pytest.raises(BudgetExceeded):
            backend.chat("test system", "test user")
    assert send.call_count == 1
    assert backend.budget.snapshot()["requests_with_unknown_usage"] == 1
    assert [a["status"] for a in backend.last_response_metadata["http_attempts"]] == ["http_error", "budget_blocked"]


def test_returned_usage_settles_ledger_and_completion_limit_is_sent(tmp_path):
    from budget import BudgetLedger
    from decimal import Decimal
    backend = models.OpenAIBackend("gpt-4o", api_key="test-only", temperature=.7)
    backend.max_output_tokens = 10
    backend.budget = BudgetLedger(tmp_path / "budget.jsonl", cap_usd="1",
                                  input_per_million_usd="2.5", output_per_million_usd="10")
    data = {"model": "gpt-4o-test", "choices": [{"message": {"content": "test reply"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}}
    class Response(io.BytesIO):
        status = 200
    with patch.object(models.urllib.request, "urlopen", return_value=Response(json.dumps(data).encode())) as send:
        assert backend.chat("system", "user") == "test reply"
    sent = json.loads(send.call_args.args[0].data)
    assert sent["max_completion_tokens"] == 10
    snapshot = backend.budget.snapshot()
    assert Decimal(snapshot["committed_usd"]) == Decimal("0.000115")
    assert Decimal(snapshot["reserved_usd"]) == 0
    assert snapshot["total_tokens"] == 34


def test_missing_token_counts_never_become_fake_zero(monkeypatch):
    with patch("telemetry._encoding", return_value=None):
        assert text_size("hello", "unknown") == {
            "characters": 5, "utf8_bytes": 5, "tokens": None, "tokenizer": "unavailable"}
    assert usage_totals([{"provider": {"usage": {"total_tokens": 5}}}, {"provider": {}}])["total_tokens"] is None


def test_absent_tokenizer_cache_does_not_download(tmp_path, monkeypatch):
    from telemetry import _encoding
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    _encoding.cache_clear()
    try:
        with patch("urllib.request.urlopen", side_effect=AssertionError("network prohibited")):
            assert _encoding("gpt-4o") is None
    finally:
        _encoding.cache_clear()


def test_current_scorer_module_is_unchanged_from_scorer_fix():
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root / "environment.py").read_bytes()).hexdigest() == (
        "8ddc3a469ac092dd314ba6b979de9773e32169b011fd83be9ffc1a4f3d374f85")


@pytest.mark.parametrize("task_id", ["ws_32", "ws_37"])
@pytest.mark.parametrize("file_ref,expected", [("26", True), ("hawaii-packing-list.docx", True), ("wrong-file", False)])
def test_create_then_share_accepts_filename_or_gold_id_only(task_id, file_ref, expected):
    from copy import deepcopy
    from pathlib import Path
    from environment import ToolLog, score_run
    tasks = json.loads((Path(__file__).resolve().parents[1] / "tasks/sample_tasks.json").read_text())
    task = next(t for t in tasks if t["id"] == task_id)
    writes = deepcopy([c for c in task["required_calls"] if c["tool"] not in task["agent1_tools"]])
    next(c for c in writes if c["tool"] == "share_file")["args"]["file_id"] = file_ref
    assert score_run(ToolLog(calls=writes), task)["task_success"] is expected
