"""Stricter secondary checks use synthetic outputs only; no model API calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from environment import ToolLog, score_run
from models import MockBackend
from scripts import run2
from scripts.analyze_run2_secondary import analyze_secondary, main, usage_report, validate_design
from secondary_evaluation import _assignment, canonical_sha, evaluate_secondary, load_references


ROOT = Path(__file__).resolve().parents[1]
DATASET = (ROOT / "tasks/sample_tasks.json").read_bytes()
TASKS = json.loads(DATASET)
TASK_MAP = {t["id"]: t for t in TASKS}
POLICY = load_references(ROOT / "tasks/run2_secondary_references.json", TASKS, DATASET)


def gold(tid):
    """Conservative literal reference fixture, not a recorded agent result."""
    calls = []
    for reference in POLICY["tasks"][tid]["calls"]:
        args = deepcopy(reference["args"])
        for key, mode in reference.get("match", {}).items():
            if mode == "any_of":
                args[key] = args[key][-1]
        for key, requirement in reference["text_requirements"].items():
            if requirement.get("all_of"):
                args[key] = "\n".join(group[0] for group in requirement["all_of"])
            else:
                args[key] = "- New useful activity one\n- New useful activity two"
        calls.append({"tool": reference["tool"], "args": args})
    return {**score_run(ToolLog(calls=calls), TASK_MAP[tid]), "task_id": tid,
            "calls": calls, "stage_calls": {"agent2": calls}, "dropped_calls": {"agent2": []}}


def evaluate(tid, record):
    return evaluate_secondary(record, TASK_MAP[tid], POLICY)


@pytest.mark.parametrize("tid", [tid for tid, value in POLICY["tasks"].items() if not value["manual_review_requirements"]])
def test_gold_passes_all_fully_referenced_tasks(tid):
    outcome = evaluate(tid, gold(tid))
    assert outcome["primary_task_success"] is True
    assert outcome["strict_reference_conformity"] is True


@pytest.mark.parametrize("field", ["recipients", "cc", "bcc"])
def test_wrong_or_extra_recipients_fail(field):
    record = gold("ws_25")
    record["calls"][0]["args"].setdefault(field, []).append("unexpected@example.org")
    assert evaluate("ws_25", record)["strict_reference_conformity"] is False


def test_extra_calendar_participant_fails():
    record = gold("ws_6")
    record["calls"][0]["args"]["participants"].append("unexpected@example.org")
    assert score_run(ToolLog(calls=record["calls"]), TASK_MAP["ws_6"])["task_success"]
    assert evaluate("ws_6", record)["strict_reference_conformity"] is False


def test_one_bulk_email_cannot_satisfy_three_ws25_actions():
    record = gold("ws_25")
    body = "\n".join(c["args"]["body"] for c in record["calls"])
    recipients = [c["args"]["recipients"][0] for c in record["calls"]]
    record["calls"] = [{"tool": "send_email", "args": {"recipients": recipients, "subject": "Task", "body": body}}]
    record["stage_calls"]["agent2"] = record["calls"]
    assert score_run(ToolLog(calls=record["calls"]), TASK_MAP["ws_25"])["task_success"]
    assert evaluate("ws_25", record)["strict_reference_conformity"] is False
    assert _assignment([[0], [0]], []) is None


def test_empty_and_vague_activity_content_do_not_claim_success():
    record = gold("ws_29")
    for content in ("", "Two more activities", "- Activity\n- Activity"):
        record["calls"][0]["args"]["content"] = content
        outcome = evaluate("ws_29", record)
        assert outcome["deterministic_checks_pass"] is False
        assert outcome["strict_reference_conformity"] is None
        assert outcome["manual_review_requirements"]


def test_extra_actions_duplicates_and_dropped_attempts_fail():
    for action in ({"tool": "delete_file", "args": {"file_id": "42"}}, gold("ws_6")["calls"][0]):
        record = gold("ws_6")
        record["calls"].append(action)
        assert evaluate("ws_6", record)["strict_reference_conformity"] is False
    record = gold("ws_6")
    record["dropped_calls"]["agent2"].append({"tool": "invented_tool", "args": {}})
    assert evaluate("ws_6", record)["strict_reference_conformity"] is False


def test_reads_are_not_extra_actions():
    record = gold("ws_6")
    record["calls"] = [{"tool": TASK_MAP["ws_6"]["agent1_tools"][0], "args": {}}] + record["calls"]
    assert evaluate("ws_6", record)["strict_reference_conformity"] is True


def test_stage_trace_cannot_hide_an_extra_accepted_write():
    record = gold("ws_6")
    record["calls"] = record["calls"] + [{"tool": "delete_file", "args": {"file_id": "42"}}]
    assert score_run(ToolLog(calls=record["calls"]), TASK_MAP["ws_6"])["task_success"]
    with pytest.raises(ValueError, match="inconsistent_action_log"):
        evaluate("ws_6", record)


def test_reversed_create_share_dependency_fails_without_changing_primary():
    record = gold("ws_32")
    record["calls"].reverse()
    assert score_run(ToolLog(calls=record["calls"]), TASK_MAP["ws_32"])["task_success"]
    outcome = evaluate("ws_32", record)
    assert outcome["strict_reference_conformity"] is False
    assert "required_dependency_order_violation" in outcome["problems"]


def test_independent_ws25_emails_have_no_arbitrary_order_constraint():
    record = gold("ws_25")
    record["calls"].reverse()
    assert evaluate("ws_25", record)["strict_reference_conformity"] is True
    assert POLICY["tasks"]["ws_13"]["order_dependencies"] == []
    assert POLICY["tasks"]["ws_19"]["order_dependencies"] == []


def test_content_omission_negation_and_numeric_boundaries_fail():
    record = gold("ws_31")
    record["calls"][0]["args"]["content"] = record["calls"][0]["args"]["content"].replace("Sunscreen", "No Sunscreen")
    assert evaluate("ws_31", record)["strict_reference_conformity"] is False
    record = gold("ws_25")
    record["calls"][0]["args"]["body"] = "2024-05-20"
    assert evaluate("ws_25", record)["strict_reference_conformity"] is False
    record = gold("ws_25")
    record["calls"][0]["args"]["body"] = record["calls"][0]["args"]["body"].replace("2024-05-20", "2024-05-200")
    assert evaluate("ws_25", record)["strict_reference_conformity"] is False


def test_missing_hot_water_is_new_secondary_failure():
    record = gold("ws_34")
    record["calls"][0]["args"]["content"] = record["calls"][0]["args"]["content"].replace("hot water\n", "")
    assert score_run(ToolLog(calls=record["calls"]), TASK_MAP["ws_34"])["task_success"]
    assert evaluate("ws_34", record)["strict_reference_conformity"] is False


def test_strict_cannot_pass_primary_failure_even_if_synonym_accepted():
    record = gold("ws_34")
    record["calls"][0]["args"]["content"] = record["calls"][0]["args"]["content"].replace("eggs", "egg")
    record["task_success"] = True  # The evaluator must recompute, not trust this.
    outcome = evaluate("ws_34", record)
    assert outcome["primary_task_success"] is False
    assert outcome["strict_reference_conformity"] is False
    assert "primary_requirements_not_satisfied" in outcome["problems"]


def test_reference_provenance_and_coverage_are_explicit():
    assert len([tid for tid, value in POLICY["tasks"].items() if not value["manual_review_requirements"]]) == 11
    for tid in ("ws_4", "ws_13", "ws_19", "ws_29", "ws_33", "ws_36", "ws_37"):
        assert evaluate(tid, gold(tid))["strict_reference_conformity"] is None
    changed = deepcopy(TASKS)
    changed[0]["agent1_task"] += " changed"
    with pytest.raises(ValueError, match="task hash"):
        load_references(ROOT / "tasks/run2_secondary_references.json", changed)


class UsageFixture(MockBackend):
    def chat(self, system, user):
        result = super().chat(system, user)
        self.last_response_metadata = {"usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}
        return result


@pytest.fixture
def shared_run(tmp_path, monkeypatch):
    for name in ("MAS_BACKEND", "MAS_MODEL", "MAS_TEMPERATURE", "MAS_DELEGATE", "MAS_BASE_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    fixture = UsageFixture()
    monkeypatch.setattr(run2, "make_backends", lambda: (fixture, fixture))
    monkeypatch.setattr(run2, "code_identity", lambda: {"commit": "fixture", "tree": "fixture", "dirty": False, "files": {}})
    out = tmp_path / "shared"
    assert run2.main(["--phase", "smoke", "--backend", "mock", "--design", "shared-handoff", "--task-ids", "ws_6", "--out-dir", str(out)]) == 0
    manifest = json.loads((out/"manifest.json").read_text())
    records = [json.loads(line) for line in (out/"raw.jsonl").read_text().splitlines()]
    cache = [json.loads(line) for line in (out/"handoffs.jsonl").read_text().splitlines()]
    return out, manifest, records, cache


def test_shared_design_validates_links_and_counts_agent1_once(shared_run):
    _, manifest, records, cache = shared_run
    assert validate_design(manifest, records, cache) == "shared-handoff"
    usage = usage_report("shared-handoff", records, cache)
    assert usage["unique_cached_agent1_draws"] == 1
    assert usage["completed_arm_usage"]["total_tokens"] == 120  # six A2 + four compressors
    assert usage["unique_cached_agent1_usage_counted_once"]["total_tokens"] == 12
    assert usage["completed_arms_plus_unique_cached_agent1_usage"]["total_tokens"] == 132


def test_mixed_design_or_changed_shared_hash_is_rejected(shared_run):
    _, manifest, records, cache = shared_run
    bad = deepcopy(records)
    bad[0]["design"] = "independent"
    with pytest.raises(ValueError, match="mixed"):
        validate_design(manifest, bad, cache)
    bad = deepcopy(records)
    bad[0]["prepared_agent1_sha256"] = "changed"
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_design(manifest, bad, cache)


def test_rehashed_edited_cached_handoff_is_rejected_by_pure_pipeline_validator(shared_run):
    _, manifest, records, cache = shared_run
    cache = deepcopy(cache)
    cache[0]["prepared_agent1"]["handoff"] += " forged"
    cache[0]["prepared_agent1_sha256"] = canonical_sha(cache[0]["prepared_agent1"])
    cache[0]["shared_handoff_sha256"] = canonical_sha({k:v for k,v in cache[0].items() if k != "shared_handoff_sha256"})
    with pytest.raises(ValueError):
        validate_design(manifest, records, cache)


def test_secondary_cli_outputs_design_aware_files_without_primary_overwrite(shared_run, tmp_path):
    out, _, _, _ = shared_run
    destination = tmp_path/"secondary"
    assert main(["--run-dir", str(out), "--out-dir", str(destination), "--bootstrap-samples", "100"]) == 0
    result = json.loads((destination/"secondary_analysis.json").read_text())
    assert result["design"] == "shared-handoff"
    assert result["mock"] is True
    assert result["fully_referenced_task_ids"] == ["ws_6"]
    text = (destination/"secondary_results.md").read_text()
    assert "same cached Agent 1" in text
    assert "MOCK INFRASTRUCTURE" in text
    assert not (destination/"results_table.md").exists()
    assert not (destination/"analysis.json").exists()
