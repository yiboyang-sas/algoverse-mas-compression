"""Keep Run 2 on the team's frozen 18-task benign evaluation set.

Source: scorer-fix d1512367702c44faa244b606bdd6b6f6c8a0e89a.
Original tasks/sample_tasks.json SHA256:
348b8ab77d10bb96ecc6cb4ca1b03e035edd679167d7f9b13ac028bbfb91f469.
Only allowed_tools is corrected: every required action must be permitted.
"""

import hashlib
import json
from pathlib import Path

import pytest


TASKS_PATH = Path(__file__).resolve().parents[1] / "tasks" / "sample_tasks.json"
TASKS = json.loads(TASKS_PATH.read_text())
TASK_IDS = [
    "ws_6", "ws_7", "ws_12", "ws_21", "ws_29", "ws_31", "ws_32", "ws_15",
    "ws_18", "ws_34", "ws_20", "ws_33", "ws_25", "ws_13", "ws_4", "ws_19",
    "ws_36", "ws_37",
]


def test_exact_frozen_task_set_and_no_attack():
    assert [task["id"] for task in TASKS] == TASK_IDS
    assert len(set(TASK_IDS)) == 18
    for task in TASKS:
        assert task["harmful_call"] is None
        assert task["payload_scope"] is None
        assert not task.get("injection")
        assert all("INJECT_CALL:" not in doc for doc in task["documents"])


def test_original_documents_and_scoring_requirements_are_frozen():
    # This digest includes EVERY original field except allowed_tools, including
    # complete document strings, prompts, required_calls, match modes, controls,
    # and handoff_critical. It detects accidental task redesign or regeneration.
    preserved = [{k: v for k, v in task.items() if k != "allowed_tools"}
                 for task in TASKS]
    encoded = json.dumps(preserved, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "b8c8cdff7d3086fbdacade3d22bf1b4894b9b06e7369490f070c75adc7b6504d"
    )


@pytest.mark.parametrize("task", TASKS, ids=TASK_IDS)
def test_all_required_actions_are_allowed_and_held(task):
    required_writes = {call["tool"] for call in task["required_calls"]
                       if call["tool"] not in task["agent1_tools"]}
    assert required_writes
    assert set(task["allowed_tools"]) == required_writes
    assert required_writes <= set(task["agent2_held_tools"])
    assert not set(task["agent1_tools"]) & set(task["agent2_held_tools"])


@pytest.mark.parametrize("task_id,expected", [
    ("ws_13", {"append_to_file", "send_email"}),
    ("ws_19", {"append_to_file", "send_email"}),
    ("ws_32", {"create_file", "share_file"}),
    ("ws_37", {"create_file", "share_file"}),
])
def test_intermediate_actions_survive_allowlist_configuration(task_id, expected):
    task = next(task for task in TASKS if task["id"] == task_id)
    assert set(task["allowed_tools"]) == expected
