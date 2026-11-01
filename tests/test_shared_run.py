"""Offline evidence/integrity tests; authored fixtures are not model results."""
from __future__ import annotations

import concurrent.futures
import contextlib
import fcntl
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import shared_run as shared


class FixtureBackend:
    name = "authored_fixture"

    def __init__(self, *, empty=False, fail_stage=None, fail_condition=None):
        self.tasks = {t["id"]: t for t in shared.load_tasks()}
        self.calls = []
        self.empty = empty
        self.fail_stage = fail_stage
        self.fail_condition = fail_condition

    def chat(self, system, user):
        ctx = shared.instrument.current()
        stage = ctx.stage_override or shared.instrument.stage_from_system(system)
        self.calls.append((ctx.task_id, ctx.repeat, ctx.condition, stage))
        if stage == self.fail_stage and (self.fail_condition is None or ctx.condition == self.fail_condition):
            raise RuntimeError("authored transport failure")
        task = self.tasks[ctx.task_id]
        if stage == "agent1":
            if self.empty:
                return ""
            reads = [c for c in task["required_calls"] if c["tool"] in task["agent1_tools"]]
            return json.dumps({"tool_calls": reads, "handoff_note":
                "PLAN: " + task["required_calls"][-1]["tool"] + "\n" +
                json.dumps(task["required_calls"]) + "\n" + "\n".join(task["documents"])})
        if stage == "compressor":
            return user
        writes = [c for c in task["required_calls"] if c["tool"] in task["agent2_held_tools"]]
        return json.dumps({"tool_calls": writes, "handoff_note": "done"})


class SharedRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="shared-run-test-")
        self.out = Path(self.temp.name) / "experiment"
        # Exercise native counting's documented offline estimate; never resolve
        # tiktoken/download its vocabulary in an infrastructure fixture test.
        self.encoder = mock.patch.multiple(shared.metrics, _ENCODER=None,
                                          _ENCODER_NAME=None, _ENCODER_TRIED=True)
        self.encoder.start()

    def tearDown(self):
        self.encoder.stop()
        self.temp.cleanup()

    def args(self, *more):
        return shared.parser().parse_args(["--out-dir", str(self.out), "--limit", "2",
                                           "--repeats", "2", "--workers", "4", *more])

    def execute(self, backend=None, args=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return shared.run(args or self.args(), backend=backend or FixtureBackend())

    def manifest(self):
        return json.loads((self.out / "manifest.json").read_text())

    def rows(self, condition):
        return shared._read_jsonl(self.out / "arms" / condition / "runs.jsonl")

    def calls(self, condition=None):
        path = self.out / ("arms/" + condition if condition else "shared_agent1") / "calls.jsonl"
        return shared._read_jsonl(path)

    def rewrite(self, path, rows):
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_fixed_primary_defaults_and_canonical_population(self):
        args = shared.parser().parse_args(["--out-dir", str(self.out)])
        self.assertEqual((args.repeats, args.workers, args.limit, args.execute_paid), (3, 8, None, False))
        tasks = shared.load_tasks()
        self.assertEqual(len(tasks), 63)
        self.assertFalse({"ws_32", "ws_37"} & {t["id"] for t in tasks})
        with mock.patch.dict(shared.os.environ, {"MAS_MODEL": "wrong", "MAS_TEMPERATURE": "1.0", "MAS_BACKEND": "openai"}):
            self.execute()
        s = self.manifest()["settings"]
        self.assertEqual((s["model"], s["compressor_model"], s["temperature"], s["handoff_regime"], s["compressor_budget_tokens"]),
                         ("gpt-4o-mini", "gpt-4o-mini", 0, "plan", 150))
        self.assertFalse(s["attack_enabled"])
        self.assertEqual(self.manifest()["backend"], "mock")

    def test_agent1_once_six_arms_native_traces_and_shared_content(self):
        backend = FixtureBackend()
        status = self.execute(backend)
        self.assertEqual(status["terminal_trials"], 24)
        self.assertEqual(status["shared_handoffs"], 4)
        self.assertEqual(len(self.calls()), 4)
        self.assertEqual(sum(c[-1] == "agent1" for c in backend.calls), 4)
        cache = shared._read_jsonl(self.out / "handoffs.jsonl")
        cache_by_key = {(r["task_id"], r["repeat"]): r for r in cache}
        for condition in shared.CONDITIONS:
            rows = self.rows(condition)
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(c["stage"] != "agent1" for c in self.calls(condition)))
            for row in rows:
                handoff = cache_by_key[(row["task_id"], row["repeat"])]
                for key in ("shared_handoff_id", "agent1_output_sha256", "handoff_sha256"):
                    self.assertEqual(row[key], handoff[key])
                self.assertEqual(row["parse_events"], ["ok", "ok"])
                self.assertTrue(row["task_success"])
                self.assertIsInstance(row["controls"], list)
                self.assertIsInstance(row["tool_mismatch"], list)
                task = backend.tasks[row["task_id"]]
                log = shared.environment.ToolLog()
                log.calls = row["tool_calls"]
                self.assertEqual(row["task_success"], shared.baseline.score_checkable(log, task)["task_success"])
        self.assertEqual(len(backend.calls), 44)  # 4 groups × (A1 + 6 A2 + 4 compressor)

    def test_cache_committed_before_any_downstream_call(self):
        backend = FixtureBackend()
        original = backend.chat

        def checked(system, user):
            if shared.instrument.stage_from_system(system) != "agent1":
                ctx = shared.instrument.current()
                cached = shared._read_jsonl(self.out / "handoffs.jsonl")
                self.assertTrue(any(r["task_id"] == ctx.task_id and r["repeat"] == ctx.repeat for r in cached))
            return original(system, user)

        backend.chat = checked
        self.execute(backend)

    def test_resume_makes_zero_additional_requests(self):
        backend = FixtureBackend()
        self.execute(backend)
        count = len(backend.calls)
        self.execute(backend)
        self.assertEqual(len(backend.calls), count)
        self.assertEqual(len(self.calls()), 4)

    def test_partial_condition_resume_never_redraws_agent1(self):
        backend = FixtureBackend()
        real_pool = concurrent.futures.ThreadPoolExecutor
        pools = []

        def stop_between_arms(*args, **kwargs):
            if pools:
                raise KeyboardInterrupt()
            pools.append(True)
            return real_pool(*args, **kwargs)

        with mock.patch.object(shared.futures, "ThreadPoolExecutor", side_effect=stop_between_arms):
            with self.assertRaises(KeyboardInterrupt):
                self.execute(backend)
        self.assertEqual(len(self.rows("none")), 4)
        self.assertEqual(len(self.calls()), 4)
        self.execute(backend)
        self.assertEqual(sum(c[-1] == "agent1" for c in backend.calls), 4)
        self.assertEqual(len(self.rows("allowlist")), 4)

    def test_empty_agent1_draw_is_reused_without_redraw(self):
        backend = FixtureBackend(empty=True)
        self.execute(backend)
        self.assertTrue(all(r["agent1_raw"] == "" and r["handoff"] == "" and r["status"] == "ready"
                            for r in shared._read_jsonl(self.out / "handoffs.jsonl")))
        self.execute(backend)
        self.assertEqual(sum(c[-1] == "agent1" for c in backend.calls), 4)

    def test_upstream_failures_are_terminal_unscored_all_arms(self):
        backend = FixtureBackend(fail_stage="agent1")
        status = self.execute(backend)
        self.assertEqual(status["api_or_pipeline_failures"], 24)
        self.assertEqual(len(backend.calls), 4)
        for condition in shared.CONDITIONS:
            self.assertTrue(all(r["task_success"] is None and r["error_stage"] == "agent1"
                                for r in self.rows(condition)))
        self.execute(backend)
        self.assertEqual(len(backend.calls), 4)

    def test_downstream_failure_does_not_redraw_or_block_other_arms(self):
        backend = FixtureBackend(fail_stage="compressor", fail_condition="destination")
        status = self.execute(backend)
        self.assertEqual(status["api_or_pipeline_failures"], 4)
        self.assertTrue(all(r["task_success"] is None for r in self.rows("destination")))
        self.assertTrue(all(r["task_success"] for r in self.rows("allowlist")))
        count = len(backend.calls)
        self.execute(backend)
        self.assertEqual(len(backend.calls), count)

    def test_error_history_duplicate_or_skipped_stage_rejected_before_resume(self):
        backend = FixtureBackend(fail_stage="compressor", fail_condition="destination")
        self.execute(backend)
        path = self.out / "arms/destination/calls.jsonl"
        original = shared._read_jsonl(path)
        # A failed compressor followed by another compressor cannot be a native
        # terminal stage sequence, even when the arm itself is marked error.
        self.rewrite(path, original + [original[0]])
        count = len(backend.calls)
        with self.assertRaisesRegex(shared.IntegrityError, "valid native prefix"):
            self.execute(backend)
        self.assertEqual(len(backend.calls), count)
        altered = [dict(c) for c in original]
        altered[0]["stage"] = "agent2"
        self.rewrite(path, altered)
        with self.assertRaisesRegex(shared.IntegrityError, "valid native prefix"):
            self.execute(backend)

    def test_call_transport_flags_must_match_native_error_semantics(self):
        backend = FixtureBackend(fail_stage="compressor", fail_condition="destination")
        self.execute(backend)
        path = self.out / "arms/destination/calls.jsonl"
        original = shared._read_jsonl(path)
        for changes in ({"succeeded": "false"}, {"succeeded": True},
                        {"api_error": None}, {"raw_output": "invented"}):
            altered = [dict(c) for c in original]
            altered[0].update(changes)
            self.rewrite(path, altered)
            with self.assertRaisesRegex(shared.IntegrityError, "transport flag/error"):
                self.execute(backend)

    def test_manifest_and_cache_tampering_rejected(self):
        self.execute()
        cache_path = self.out / "handoffs.jsonl"
        rows = shared._read_jsonl(cache_path)
        rows[0]["handoff"] += "changed"
        self.rewrite(cache_path, rows)
        with self.assertRaisesRegex(shared.IntegrityError, "checksum"):
            self.execute()

    def test_rehashed_cache_tamper_still_rejected_by_native_raw_evidence(self):
        self.execute()
        cache_path = self.out / "handoffs.jsonl"
        rows = shared._read_jsonl(cache_path)
        rows[0]["agent1_input"] += "different input"
        rows[0]["record_sha256"] = shared._digest_record(rows[0], "record_sha256")
        self.rewrite(cache_path, rows)
        with self.assertRaisesRegex(shared.IntegrityError, "input/identity"):
            self.execute()

    def test_cache_missing_with_completed_arms_fails_closed(self):
        self.execute()
        (self.out / "handoffs.jsonl").unlink()
        backend = FixtureBackend()
        with self.assertRaisesRegex(shared.IntegrityError, "without durable cache"):
            self.execute(backend)
        self.assertEqual(backend.calls, [])

    def test_orphan_downstream_call_fails_closed(self):
        self.execute()
        (self.out / "arms/none/runs.jsonl").write_text("")
        with self.assertRaisesRegex(shared.IntegrityError, "orphan downstream"):
            self.execute()

    def test_upstream_intent_without_cache_fails_closed(self):
        self.execute()
        path = self.out / "handoffs.jsonl"
        self.rewrite(path, shared._read_jsonl(path)[1:])
        with self.assertRaisesRegex(shared.IntegrityError, "automatic redraw forbidden"):
            self.execute()

    def test_duplicate_and_malformed_cache_fail_closed(self):
        self.execute()
        path = self.out / "handoffs.jsonl"
        original = path.read_text()
        path.write_text(original + original.splitlines()[0] + "\n")
        with self.assertRaisesRegex(shared.IntegrityError, "duplicate cache"):
            self.execute()
        path.write_text(original + "{\n")
        with self.assertRaisesRegex(shared.IntegrityError, "malformed"):
            self.execute()
        path.write_text(original + "{")
        with self.assertRaisesRegex(shared.IntegrityError, "incomplete journal"):
            self.execute()

    def test_config_source_and_model_settings_frozen_on_resume(self):
        self.execute()
        with self.assertRaisesRegex(shared.IntegrityError, "configuration/source"):
            self.execute(args=self.args("--repeats", "3"))
        original = shared.source_identity()
        changed = dict(original, git_commit="different")
        with mock.patch.object(shared, "source_identity", return_value=changed):
            with self.assertRaisesRegex(shared.IntegrityError, "configuration/source"):
                self.execute()

    def test_second_writer_is_rejected_before_backend_or_state(self):
        lock_path = self.out.parent / ("." + self.out.name + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            backend = FixtureBackend()
            with self.assertRaisesRegex(shared.IntegrityError, "another writer"):
                self.execute(backend)
            self.assertEqual(backend.calls, [])
            self.assertFalse(self.out.exists())

    def test_paid_gate_rejects_missing_credentials_and_estimated_tokens(self):
        with mock.patch.dict(shared.os.environ, {}, clear=True), mock.patch.object(shared, "OpenAIBackend") as network:
            with self.assertRaisesRegex(shared.IntegrityError, "OPENAI_API_KEY"):
                self.execute(args=self.args("--execute-paid"))
            network.assert_not_called()
        with mock.patch.dict(shared.os.environ, {"OPENAI_API_KEY": "authored-not-real"}), mock.patch.object(shared, "OpenAIBackend") as network:
            with self.assertRaisesRegex(shared.IntegrityError, "measured"):
                self.execute(args=self.args("--execute-paid"))
            network.assert_not_called()

    def test_independent_pipeline_retains_two_model_calls_and_parser_events(self):
        backend = FixtureBackend()
        task = shared.load_tasks()[0]
        # The context used by native _run_instrumented supplies fixture task id.
        result = shared.pipeline.run_pipeline(task, backend, backend, "none")
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual([c[-1] for c in backend.calls], ["agent1", "agent2"])
        self.assertEqual(result["parse_events"], ["ok", "ok"])
        self.assertTrue(result["task_success"])


if __name__ == "__main__":
    unittest.main()
