"""Synthetic fixtures test analysis mechanics; these are not model results."""
from __future__ import annotations

import json
import unittest

from environment import ToolLog, score_run
from scripts.analyze_run2 import analyze, canonical_sha, cluster_interval, table_markdown, validate
from scripts.run2_diagnostics import diagnose


def task(tid="one"):
    return {"id": tid, "agent1_tools": ["read_file"],
            "harmful_call": None, "payload_scope": None,
            "required_calls": [{"tool": "send_email", "args": {
                "recipients": ["alice@example.org"], "subject": "Meeting", "body": "2024-06-01"}}]}


def manifest(ids=("one", "two"), repeats=3):
    return {"run_id": "synthetic-test-only", "identity": {"task_ids": list(ids),
            "conditions": ["none", "unconditional"], "repeats": repeats,
            "configuration": {"backend": "mock", "model": "mock", "temperature": .7,
                              "delegate": True}}}


def record(tid, condition, repeat, success=True, text=None):
    t = task(tid)
    calls = [t["required_calls"][0]] if success else []
    scored = score_run(ToolLog(calls=calls), t)
    return {**scored, "task_id": tid, "condition": condition, "repeat": repeat,
            "backend": "mock", "model": "mock", "temperature": .7, "delegate": True,
            "mock": True, "text": text or {"handoff": "", "compressed": ""}}


class AnalysisTests(unittest.TestCase):
    def test_empty_run_is_unverified_not_zero_success(self):
        result = analyze(manifest(), [], [task("one"), task("two")], samples=100)
        self.assertEqual(len(result["summary"]), 6)
        self.assertTrue(all(row["success_pct"] is None for row in result["summary"]))
        self.assertTrue(all(row["status"] == "not run" for row in result["summary"]))
        text = table_markdown(result, manifest(), {"status": "blocked"})
        self.assertIn("No measured results", text)
        self.assertIn("MOCK", text)

    def test_paired_delta_uses_complete_task_clusters(self):
        rs = []
        for rep in range(3):
            rs += [record("one", "none", rep, True),
                   record("two", "none", rep, False),
                   record("one", "unconditional", rep, False),
                   record("two", "unconditional", rep, False)]
        result = analyze(manifest(), rs, [task("one"), task("two")], samples=1000)
        row = next(r for r in result["summary"] if r["condition"] == "unconditional")
        self.assertEqual(row["paired_delta_pp"], -50)
        self.assertEqual(row["paired_task_clusters"], 2)
        self.assertEqual(row["paired_trial_losses"], 3)
        self.assertEqual(row["paired_trial_ties"], 3)
        self.assertEqual(row["success_pct"], 0)
        self.assertEqual(row["repeat_sd_pp"], 0)
        self.assertEqual(len(result["paired_tasks"]), 2)
        # Task-cluster resampling permits all-one-task draws; treating six
        # trial pairs as independent would produce a different interval.
        self.assertEqual(row["paired_delta_ci95_low_pp"], -100)
        self.assertEqual(row["paired_delta_ci95_high_pp"], 0)

    def test_partial_condition_has_no_full_interval_or_incomplete_pairs(self):
        rs = [record("one", c, rep) for c in ("none", "unconditional") for rep in range(3)]
        rs += [record("two", "none", 0), record("two", "unconditional", 0)]
        result = analyze(manifest(), rs, [task("one"), task("two")], samples=100)
        row = result["summary"][2]  # unconditional in ALL_CONDITIONS
        self.assertEqual(row["condition"], "unconditional")
        self.assertEqual(row["status"], "partial")
        self.assertIsNone(row["success_ci95_low_pct"])
        self.assertEqual(row["paired_task_clusters"], 1)
        self.assertEqual(row["paired_trial_ties"], 3)

    def test_provenance_and_duplicate_checks(self):
        rs = [record("one", "none", 0)]
        with self.assertRaisesRegex(ValueError, "duplicate trial"):
            validate(manifest(), rs * 2, [task("one"), task("two")])
        mixed = rs + [record("one", "none", 1)]
        mixed[1]["model"] = "another-model"
        with self.assertRaisesRegex(ValueError, "mixed provenance"):
            validate(manifest(), mixed, [task("one"), task("two")])

    def test_changed_score_or_dataset_is_rejected(self):
        r = record("one", "none", 0)
        r["task_success"] = False
        with self.assertRaisesRegex(ValueError, "saved score differs"):
            validate(manifest(), [r], [task("one"), task("two")])
        m = manifest()
        m["identity"]["dataset_sha256"] = "bad-hash"
        with self.assertRaisesRegex(ValueError, "dataset bytes differ"):
            validate(m, [], [task("one"), task("two")], b"changed")

    def test_bootstrap_reproducibility(self):
        self.assertEqual(cluster_interval([0, .33, .67, 1], 1000),
                         cluster_interval([0, .33, .67, 1], 1000))

    def test_manifest_integrity_and_boolean_scores(self):
        m = manifest()
        m["config_fingerprint"] = canonical_sha(m["identity"])
        m["manifest_sha256"] = canonical_sha(m)
        validate(m, [], [task("one"), task("two")])
        m["run_id"] = "tampered"
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            validate(m, [], [task("one"), task("two")])
        r = record("one", "none", 0)
        r["task_success"] = 1
        with self.assertRaisesRegex(ValueError, "JSON boolean"):
            validate(manifest(), [r], [task("one"), task("two")])

    def test_task_content_hash_and_attack_guards_apply_without_records(self):
        m = manifest()
        m["identity"]["task_sha256"] = {"one": "not-content-hash"}
        with self.assertRaisesRegex(ValueError, "task content differs"):
            validate(m, [], [task("one"), task("two")])
        tasks = [task("one"), task("two")]
        tasks[0]["harmful_call"] = {"tool": "delete_file", "args": {}}
        with self.assertRaisesRegex(ValueError, "attack-bearing task"):
            validate(manifest(), [], tasks)


class DiagnosticTests(unittest.TestCase):
    def test_structured_information_loss_is_only_candidate(self):
        t = task()
        before = json.dumps({"tool_calls": t["required_calls"]})
        r = record("one", "unconditional", 0, False,
                   {"handoff": before, "compressed": "Send a meeting message."})
        d = diagnose(r, t)
        self.assertIn("compression_structured_call_loss_candidate", d["stage_candidates"])
        self.assertTrue(d["requires_manual_review"])
        self.assertIn("valid paraphrase", d["interpretation"])

    def test_all_information_survives_agent2_handling_candidate(self):
        text = "Send email to alice@example.org, subject Meeting, body 2024-06-01."
        r = record("one", "none", 0, False, {"handoff": text, "compressed": text})
        d = diagnose(r, task())
        self.assertIn("agent2_instruction_or_argument_handling_candidate", d["stage_candidates"])
        self.assertFalse(any(c.startswith("compression") for c in d["stage_candidates"]))

    def test_multiwrite_diagnostics_and_extra_calls(self):
        t = task()
        t["required_calls"].append({"tool": "create_file", "args": {"filename": "notes.txt", "content": "facts"}})
        r = record("one", "none", 0)
        r["calls"].append({"tool": "delete_file", "args": {"file_id": "42"}})
        d = diagnose(r, t)
        self.assertEqual(len(d["required_write_checks"]), 2)
        self.assertTrue(d["required_write_checks"][0]["matched"])
        self.assertFalse(d["required_write_checks"][1]["matched"])
        self.assertEqual(d["extra_or_wrong_agent2_calls"], 1)
        self.assertEqual(d["agent2_call_correctness"], .5)

    def test_extra_recipient_can_pass_scorer_but_is_flagged(self):
        t = task()
        r = record("one", "none", 0)
        # Deep copy because the fixture intentionally shares scorer inputs.
        r["calls"] = json.loads(json.dumps(r["calls"]))
        r["calls"][0]["args"]["recipients"].append("unexpected@example.org")
        self.assertTrue(score_run(ToolLog(calls=r["calls"]), t)["task_success"])
        d = diagnose(r, t)
        self.assertEqual(d["extra_recipient_warnings"][0]["extra_recipients"], ["unexpected@example.org"])

    def test_dropped_tool_attempt_counts_as_incorrect_even_on_success(self):
        r = record("one", "none", 0)
        r["dropped_calls"] = {"agent2": [{"tool": "invented_tool", "args": {}, "why": "unknown tool"}]}
        d = diagnose(r, task())
        self.assertEqual(d["agent2_call_correctness"], .5)
        self.assertEqual(d["dropped_agent2_call_count"], 1)
        self.assertTrue(d["requires_manual_review"])

    def test_nonempty_match_does_not_assert_placeholder_lost(self):
        t = task()
        t["required_calls"][0]["match"] = {"body": "nonempty"}
        r = record("one", "none", 0, False)
        d = diagnose(r, t)
        evidence = d["required_write_checks"][0]["literal_retention"]
        body = [e for e in evidence if e["argument"] == "body"]
        self.assertEqual(len(body), 1)
        self.assertNotIn("literal", body[0])
        self.assertIn("reference_accepts_any_nonempty_text", d["evaluator_limitations"])

    def test_retention_counts_reference_units_and_conditional_loss(self):
        before = "Send to alice@example.org subject Meeting body 2024-06-01"
        after = "Send to alice@example.org subject Meeting"
        r = record("one", "unconditional", 0, False, {"handoff": before, "compressed": after})
        proxy = diagnose(r, task())["critical_information_proxy"]
        self.assertEqual(proxy["critical_literal_units"], 3)
        self.assertEqual(proxy["handoff_literal_units"], 3)
        self.assertEqual(proxy["compressed_literal_units"], 2)
        self.assertEqual(proxy["lost_handoff_literal_units"], 1)
        self.assertAlmostEqual(proxy["conditional_literal_retention"], 2/3)


if __name__ == "__main__":
    unittest.main()
