"""Observable categories use native evidence; fixtures are not API results."""
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import instrument
from failure_taxonomy import CATEGORIES, classify


def action(tool="send_money", amount=10, **args):
    return {"tool": tool, "args": {"recipient": "Alice", "amount": amount, **args}}


def task(n=1):
    return {"id": "fixture", "agent2_held_tools": ["send_money", "schedule_transaction"],
            "allowed_tools": ["send_money"], "agent1_tools": ["get_balance"],
            "required_calls": [action(amount=10 + index) for index in range(n)]}


def record(**values):
    row = {"task_id": "fixture", "condition": "unconditional", "task_success": False,
           "api_failure": False, "required_hit": 0, "required_total": 1,
           "missed": ["send_money:10"], "tool_calls": [action(amount=0)],
           "parse_events": ["ok", "ok"], "parse_failure": False,
           "text": {"handoff": "PLAN: call send_money.\nEVIDENCE: Alice 10",
                    "compressed": "Alice 10"},
           "controls": [{"key": "C", "in_documents": 3, "in_handoff": 2, "in_compressed": 0}],
           "tool_mismatch": [{"required_tool": "send_money", "status": "UNEXPRESSIBLE"}]}
    row.update(values)
    return row


def trace(stage, raw, **values):
    row = {"stage": stage, "raw_output": raw, "api_error": None,
           "requested": True, "succeeded": True, "parse_status": "ok"}
    row.update(values)
    return row


def traces(a2=None):
    return [trace("agent1", json.dumps({"tool_calls": [], "handoff_note": "Alice 10"})),
            trace("compressor", "Alice 10"),
            trace("agent2", a2 if a2 is not None else json.dumps({"tool_calls": [action(amount=0)]}),
                  user_input="Handoff from Agent 1:\nAlice 10")]


class FailureTaxonomy(unittest.TestCase):
    def setUp(self):
        # These tests would fail if taxonomy started buying calls, rebuilding
        # retention/tool mismatch, or rescoring primary argument correctness.
        for name in ("urllib.request.urlopen", "baseline.score_checkable", "baseline._args_match",
                     "controls.measure", "failure_stage.check"):
            patcher = patch(name, side_effect=AssertionError("classification must inspect saved evidence only"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_case(self, row=None, calls=None, benchmark=None):
        return classify(benchmark or task(), row if row is not None else record(),
                        traces() if calls is None else calls)

    def test_enough_actions_and_saved_misses_are_argument_profile_without_rescoring(self):
        out = self.run_case()
        self.assertEqual((out["category"], out["status"]), ("required_arguments_mismatch", "failed"))
        self.assertEqual(out["evidence"]["missing_required_tool_counts"], {})

    def test_success_survives_schema_control_and_upstream_mismatch_warnings(self):
        raw = json.dumps({"tool_calls": [action(to="wrong_key"), action("schedule_transaction")]})
        calls = traces(raw)
        calls[0]["raw_output"] = json.dumps({"tool_calls": [action()], "handoff_note": "Alice 10"})
        row = record(task_success=True, required_hit=1, missed=[], tool_calls=[action(), action("schedule_transaction")])
        out = self.run_case(row, calls)
        self.assertEqual((out["category"], out["status"]), ("success", "success"))
        self.assertTrue(out["annotations"]["agent2_schema_or_access_issues"])
        self.assertEqual(out["annotations"]["attempted_agent1_writes"], ["send_money"])
        self.assertEqual(out["annotations"]["extra_accepted_actions_by_tool"], {"schedule_transaction": 1})

    def test_native_error_wins_over_multiple_observable_failure_flags(self):
        calls = traces("broken JSON")
        calls[0].update(raw_output=None, api_error="TimeoutError: fixture", succeeded=False)
        calls[2]["user_input"] = "Handoff from Agent 1:\n"
        row = record(api_failure=True, api_error="TimeoutError: fixture",
                     text={"handoff": "", "compressed": ""}, tool_calls=[])
        out = self.run_case(row, calls)
        self.assertEqual((out["category"], out["status"]), ("pipeline_or_transport_error", "error"))
        self.assertEqual(out["evidence"]["failed_model_calls"][0]["stage"], "agent1")
        self.assertEqual(out["evidence"]["error_scope"], "transport_call_observed")
        self.assertIn("compressor_empty_output", out["annotations"]["flags"])

    def test_native_generic_pipeline_error_does_not_invent_api_stage(self):
        out = self.run_case(record(api_failure=True, api_error="AttributeError: pipeline fixture"), [])
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["evidence"]["error_scope"], "pipeline_or_unknown")
        self.assertEqual(out["evidence"]["failed_model_calls"], [])

    def test_conflicting_failed_call_does_not_silently_exclude_scored_failure(self):
        calls = traces()
        calls[2].update(api_error="TimeoutError: conflicting fixture", succeeded=False)
        out = self.run_case(calls=calls)
        self.assertEqual((out["category"], out["status"]), ("unclassified_or_missing_evidence", "failed"))
        self.assertIn("failed_call_without_native_error_flag", out["annotations"]["flags"])

    def test_native_error_and_success_conflict_is_explicit(self):
        out = self.run_case(record(api_failure=True, task_success=True))
        self.assertEqual(out["status"], "error")
        self.assertIn("native_error_conflicts_with_saved_success", out["annotations"]["flags"])

    def test_upstream_empty_wins_over_empty_compressor_and_no_downstream_action(self):
        calls = traces('{"tool_calls": []}')
        calls[2]["user_input"] = "Handoff from Agent 1:\n"
        row = record(text={"handoff": " ", "compressed": ""}, tool_calls=[])
        out = self.run_case(row, calls)
        self.assertEqual(out["category"], "upstream_unusable_reply_or_empty_handoff")
        self.assertIn("compressor_empty_output", out["annotations"]["flags"])
        self.assertIn("downstream_no_action", out["annotations"]["flags"])

    def test_non_text_upstream_note_is_unusable_without_semantic_judgment(self):
        calls = traces()
        calls[0]["raw_output"] = '{"handoff_note": 123, "tool_calls": []}'
        out = self.run_case(record(text={}), calls)
        self.assertEqual(out["category"], "upstream_unusable_reply_or_empty_handoff")

    def test_plain_upstream_text_is_not_automatically_an_unusable_handoff(self):
        calls = traces()
        calls[0]["raw_output"] = "Alice 10"
        out = self.run_case(record(text={}, parse_events=["unparsed", "ok"]), calls)
        self.assertEqual(out["category"], "required_arguments_mismatch")

    def test_empty_compressed_boundary_wins_over_downstream_no_action(self):
        calls = traces('{"tool_calls": []}')
        calls[2]["user_input"] = "Handoff from Agent 1:\n"
        out = self.run_case(record(tool_calls=[]), calls)
        self.assertEqual(out["category"], "compressor_empty_output")

    def test_none_condition_never_claims_an_empty_compressor(self):
        calls = traces('{"tool_calls": []}')
        calls[2]["user_input"] = "Handoff from Agent 1:\n"
        out = self.run_case(record(condition="none", tool_calls=[]), calls)
        self.assertEqual(out["category"], "downstream_no_action")

    def test_plain_downstream_prose_is_unparseable_not_a_transport_error(self):
        out = self.run_case(record(tool_calls=[], parse_events=["ok", "unparsed"]), traces("I cannot do that."))
        self.assertEqual((out["category"], out["status"]), ("downstream_unparseable_reply", "failed"))

    def test_repaired_downstream_call_is_not_reported_as_unparseable(self):
        raw = '{"tool_calls":[{"tool":"send_money","args":{"recipient":"Alice","amount":0,"subject":"line1\nline2"}}]}'
        out = self.run_case(record(parse_events=["ok", "repaired"]), traces(raw))
        self.assertEqual(out["category"], "required_arguments_mismatch")

    def test_transport_capture_parse_status_is_not_used_as_content_parse_status(self):
        calls = traces("not JSON")
        self.assertEqual(calls[2]["parse_status"], "ok")
        out = self.run_case(record(parse_events=[]), calls)
        self.assertEqual(out["category"], "downstream_unparseable_reply")

    def test_observed_empty_reply_is_no_action(self):
        for raw in ("", " \n\t"):
            for events in ([], ["ok", "unparsed"]):
                with self.subTest(raw=raw, events=events):
                    out = self.run_case(record(tool_calls=[], parse_events=events), traces(raw))
                    self.assertEqual(out["category"], "downstream_no_action")

    def test_missing_reply_and_missing_accepted_log_remain_unclassified(self):
        calls = traces()
        calls[2]["raw_output"] = None
        out = self.run_case(record(tool_calls=None, parse_events=None, text={}), calls)
        self.assertEqual((out["category"], out["status"]), ("unclassified_or_missing_evidence", "failed"))

    def test_known_empty_accepted_log_supports_only_an_observable_count_profile(self):
        calls = traces()
        calls[2]["raw_output"] = None
        out = self.run_case(record(tool_calls=[], parse_events=None), calls)
        self.assertEqual(out["category"], "required_actions_not_issued")

    def test_missing_primary_score_is_unverified_even_with_failure_signs(self):
        out = self.run_case(record(task_success=None, tool_calls=[]), traces(""))
        self.assertEqual((out["category"], out["status"]), ("unclassified_or_missing_evidence", "unverified"))

    def test_unknown_argument_key_is_schema_profile_before_missing_actions(self):
        out = self.run_case(record(tool_calls=[]), traces(json.dumps({"tool_calls": [action(to="Bob")]})))
        self.assertEqual(out["category"], "downstream_schema_or_access_error")
        self.assertEqual(out["annotations"]["agent2_schema_or_access_issues"][0]["keys"], ["to"])

    def test_unheld_and_unknown_tool_attempts_are_access_or_schema_issues(self):
        for name, category in (("append_to_file", "tool_not_held"), ("invented_tool", "unknown_tool")):
            with self.subTest(tool=name):
                out = self.run_case(record(tool_calls=[]), traces(json.dumps({"tool_calls": [{"tool": name, "args": {}}]})))
                self.assertEqual(out["category"], "downstream_schema_or_access_error")
                self.assertIn(category, {i["category"] for i in out["annotations"]["agent2_schema_or_access_issues"]})

    def test_bad_tool_call_container_and_entries_are_schema_issues(self):
        for attempts in (None, {}, "send_money", ["send_money"], [{"tool": "send_money", "args": []}]):
            with self.subTest(attempts=attempts):
                out = self.run_case(record(tool_calls=[]), traces(json.dumps({"tool_calls": attempts})))
                self.assertEqual(out["category"], "downstream_schema_or_access_error")

    def test_duplicate_required_calls_consume_accepted_counts_once(self):
        row = record(required_hit=1, required_total=2, tool_calls=[action()])
        out = self.run_case(row, benchmark=task(2))
        self.assertEqual(out["category"], "required_actions_not_issued")
        self.assertEqual(out["evidence"]["missing_required_tool_counts"], {"send_money": 1})

    def test_multiple_tools_require_each_tool_count_not_only_total_count(self):
        benchmark = task(2)
        benchmark["required_calls"][1]["tool"] = "schedule_transaction"
        row = record(required_hit=1, required_total=2, tool_calls=[action(), action()])
        out = self.run_case(row, benchmark=benchmark)
        self.assertEqual(out["category"], "required_actions_not_issued")
        self.assertEqual(out["evidence"]["missing_required_tool_counts"], {"schedule_transaction": 1})

    def test_required_denominator_uses_native_held_write_filter(self):
        benchmark = task()
        benchmark["required_calls"].insert(0, {"tool": "get_balance", "args": {}})
        out = self.run_case(benchmark=benchmark)
        self.assertEqual(out["evidence"]["required_tool_counts"], {"send_money": 1})
        self.assertEqual(out["category"], "required_arguments_mismatch")

    def test_malformed_log_or_task_score_denominator_conflict_is_not_argument_blame(self):
        for row in (record(tool_calls=["bad entry"]), record(required_total=2), record(required_hit=None)):
            with self.subTest(row=row):
                out = self.run_case(row)
                self.assertEqual(out["category"], "unclassified_or_missing_evidence")

    def test_partial_parse_events_do_not_locate_failure_in_agent2(self):
        out = self.run_case(record(parse_events=["unparsed"], parse_failure=True))
        self.assertEqual(out["category"], "required_arguments_mismatch")
        self.assertIn("native_parse_failure_stage_unresolved", out["annotations"]["flags"])

    def test_native_measurements_are_carried_unchanged_and_never_infer_loss(self):
        row = record()
        out = self.run_case(row)
        self.assertEqual(out["annotations"]["controls"], row["controls"])
        self.assertEqual(out["annotations"]["tool_mismatch"], row["tool_mismatch"])
        out["annotations"]["controls"][0]["in_handoff"] = -1
        self.assertEqual(row["controls"][0]["in_handoff"], 2)
        self.assertNotIn("compression_loss", out["category"])

    def test_inputs_live_parse_context_and_single_category_are_preserved(self):
        benchmark, row, calls = task(), record(), traces()
        before = deepcopy((benchmark, row, calls))
        ctx = instrument.TaskRunContext(run_id="fixture", task_id="fixture")
        ctx.parse_events.append("native_event")
        with instrument.run_context(ctx):
            out = classify(benchmark, row, calls)
            self.assertIs(instrument.current(), ctx)
            self.assertEqual(ctx.parse_events, ["native_event"])
        self.assertEqual((benchmark, row, calls), before)
        self.assertIn(out["category"], CATEGORIES)
        self.assertIsInstance(out["category"], str)
        self.assertEqual(out, classify(benchmark, row, calls))
        json.dumps(out)


if __name__ == "__main__":
    unittest.main()
