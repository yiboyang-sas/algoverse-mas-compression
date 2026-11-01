"""Shared-primary integration fixtures are offline bookkeeping, never API data."""
from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import analyze_redesign
import instrument
import metrics
import pipeline
import shared_run


ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / shared_run.TASKS_FILE
ENCODER_PATCH = {"_ENCODER": None, "_ENCODER_NAME": None, "_ENCODER_TRIED": True}
PAID_KEYS = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "AZURE_OPENAI_API_KEY")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows))


def seal_cache(row):
    row["record_sha256"] = shared_run._digest_record(row, "record_sha256")


class _AuthoredBackend:
    """Native tasks, authored exact writes, and one condition-specific failure."""
    name = "authored_offline_fixture"

    def __init__(self, tasks, *, upstream_error=False):
        self.tasks = {task["id"]: task for task in tasks}
        self.failure_task = tasks[0]["id"]
        self.upstream_error = upstream_error

    def chat(self, system, user):
        ctx = instrument.current()
        task = self.tasks[ctx.task_id]
        stage = ctx.stage_override or instrument.stage_from_system(system)
        writes = [deepcopy(call) for call in task["required_calls"]
                  if call["tool"] in task["agent2_held_tools"]]
        if stage == "agent1":
            if self.upstream_error and ctx.task_id == self.failure_task and ctx.repeat == 0:
                raise TimeoutError("authored offline upstream transport failure; no API request")
            reads = [deepcopy(call) for call in task["required_calls"]
                     if call["tool"] in task["agent1_tools"]]
            return json.dumps({"tool_calls": reads, "handoff_note":
                "PLAN: " + json.dumps(writes) + "\nEVIDENCE: authored offline fixture; no model result."})
        if stage == "compressor":
            return user
        if stage != "agent2":
            raise AssertionError("unexpected fixture stage")
        if ctx.condition == "unconditional" and ctx.task_id == self.failure_task and ctx.repeat == 0:
            writes = []
        return json.dumps({"tool_calls": writes, "handoff_note": "authored fixture"})


class SharedAnalysisIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.fixture_root = Path(cls.temporary.name)
        cls.native_fixture = cls.fixture_root / "native"
        cls.authored_fixture = cls.fixture_root / "authored"
        cls.error_fixture = cls.fixture_root / "upstream-error"
        tasks = shared_run.load_tasks()
        previous_system = pipeline.AGENT1_SYSTEM
        try:
            safe_env = {key: value for key, value in os.environ.items() if key not in PAID_KEYS}
            with patch.dict(os.environ, safe_env, clear=True), patch.multiple(metrics, **ENCODER_PATCH), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden in shared fixture")), \
                    contextlib.redirect_stdout(io.StringIO()):
                for destination, backend, errors in ((cls.native_fixture, None, 0),
                        (cls.authored_fixture, _AuthoredBackend(tasks), 0),
                        (cls.error_fixture, _AuthoredBackend(tasks, upstream_error=True), 6)):
                    args = shared_run.parser().parse_args([
                        "--out-dir", str(destination), "--experiment-id", destination.name + "-fixture",
                        "--limit", "2", "--repeats", "3", "--workers", "2"])
                    status = shared_run.run(args, backend=backend)
                    if status["api_or_pipeline_failures"] != errors:
                        raise AssertionError("offline fixture unexpectedly failed: " + str(status))
        finally:
            pipeline.AGENT1_SYSTEM = previous_system

    def setUp(self):
        self.working = tempfile.TemporaryDirectory()
        self.addCleanup(self.working.cleanup)
        self.root = Path(self.working.name)
        self.run_dir = self.root / "run"
        shutil.copytree(self.native_fixture, self.run_dir)

    def use_authored(self):
        shutil.rmtree(self.run_dir)
        shutil.copytree(self.authored_fixture, self.run_dir)

    def analyze(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("analysis must remain offline")):
            shared, run_dirs = analyze_redesign.load_shared_run(self.run_dir, TASKS_PATH)
            data, _ = analyze_redesign.analyze(run_dirs, TASKS_PATH, draws=100, seed=1729, shared=shared)
            return data

    def cache(self):
        return read_jsonl(self.run_dir / "handoffs.jsonl")

    def edit_cache(self, edit, *, seal=False):
        rows = self.cache()
        edit(rows)
        if seal:
            for row in rows:
                seal_cache(row)
        write_jsonl(self.run_dir / "handoffs.jsonl", rows)

    def edit_upstream(self, **changes):
        path = self.run_dir / "shared_agent1" / "calls.jsonl"
        rows = read_jsonl(path)
        rows[0].update(changes)
        write_jsonl(path, rows)

    def test_native_mock_bookkeeping_uses_six_draws_and_sixty_six_calls(self):
        manifest = json.loads((self.run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["backend"], "mock")
        self.assertEqual(manifest["planned"]["handoffs"], 6)
        self.assertEqual(manifest["planned"]["arm_trials"], 36)
        self.assertEqual(manifest["planned"]["model_calls_without_transport_retries"], 66)
        self.assertEqual(len(manifest["settings"]["dataset_task_ids"]), 63)
        self.assertFalse({"ws_32", "ws_37"} & set(manifest["settings"]["dataset_task_ids"]))
        self.assertEqual(len(self.cache()), 6)
        upstream = read_jsonl(self.run_dir / "shared_agent1" / "calls.jsonl")
        self.assertEqual(len(upstream), 6)
        self.assertEqual({row["stage"] for row in upstream}, {"agent1"})
        calls, records = [], []
        for condition in shared_run.CONDITIONS:
            calls.extend(read_jsonl(self.run_dir / "arms" / condition / "calls.jsonl"))
            records.extend(read_jsonl(self.run_dir / "arms" / condition / "runs.jsonl"))
        self.assertEqual(len(records), 36)
        self.assertEqual(len(upstream) + len(calls), 66)
        self.assertNotIn("agent1", {row["stage"] for row in calls})
        self.assertTrue(all(row["design"] == shared_run.DESIGN for row in records))
        status = json.loads((self.run_dir / "status.json").read_text())
        self.assertIs(status["scientific_model_results"], False)

    def test_all_fifteen_comparisons_reuse_exact_ids_with_five_primary_contrasts(self):
        data = self.analyze()
        self.assertEqual(data["design"], "shared-handoff-primary")
        self.assertEqual(data["data_kind"], "deterministic_fixture_non_empirical")
        self.assertEqual(len(data["conditions"]), 6)
        self.assertTrue(all(row["observed_trials"] == 6 for row in data["conditions"]))
        self.assertEqual(len(data["comparisons"]), 15)
        self.assertEqual(sum(row["primary_contrast"] for row in data["comparisons"]), 5)
        cache = {(row["task_id"], row["repeat"]): row for row in self.cache()}
        for contrast in data["comparisons"]:
            self.assertEqual(contrast["n_pairs"], 6)
            self.assertEqual(contrast["n_tasks"], 2)
            self.assertEqual(contrast["baseline_only"], 0)
            self.assertEqual(contrast["treatment_only"], 0)
            for pair in contrast["per_pair"]:
                self.assertEqual(pair["shared_handoff_id"], cache[pair["task_id"], pair["repeat"]]["shared_handoff_id"])
                self.assertIn("baseline_category", pair)
                self.assertIn("treatment_category", pair)

    def test_shared_upstream_usage_is_counted_once_and_arms_exclude_it(self):
        data = self.analyze()
        upstream = read_jsonl(self.run_dir / "shared_agent1" / "calls.jsonl")
        summary = data["shared_upstream"]
        self.assertEqual(summary["cached_draws"], 6)
        self.assertEqual(summary["expected_draws"], 6)
        for field in ("prompt_tokens", "output_tokens"):
            expected = sum(row[field] for row in upstream)
            self.assertEqual(summary["local_text_token_counts_counted_once"][field], expected)
            arm_total = 0
            for condition in data["conditions"]:
                calls = read_jsonl(self.run_dir / "arms" / condition["condition"] / "calls.jsonl")
                actual = sum(row[field] for row in calls)
                self.assertEqual(condition["local_text_token_counts"][field], actual)
                self.assertIs(condition["usage_is_billed_provider_usage"], False)
                arm_total += actual
            self.assertEqual(arm_total + expected, sum(
                row[field] for path in self.run_dir.rglob("calls.jsonl") for row in read_jsonl(path)))

    def test_authored_discordance_keeps_pair_category_and_task_repeat_pattern(self):
        self.use_authored()
        data = self.analyze()
        contrast = next(row for row in data["comparisons"]
                        if row["baseline_condition"] == "none" and row["condition"] == "unconditional")
        self.assertEqual(contrast["n10"], 1)
        self.assertEqual(contrast["n01"], 0)
        self.assertEqual(contrast["n11"], 5)
        self.assertEqual(contrast["n_discordant"], 1)
        self.assertAlmostEqual(contrast["estimate"], -1 / 6)
        pair = next(row for row in contrast["per_pair"] if row["delta"] == -1)
        self.assertEqual(pair["repeat"], 0)
        self.assertEqual(pair["baseline_category"], "success")
        self.assertEqual(pair["treatment_category"], "downstream_no_action")
        task = next(row for row in contrast["per_task"] if row["task_id"] == pair["task_id"])
        self.assertEqual(task["treatment_failed_repeats"], [0])
        self.assertEqual(task["pattern"], "regressed_only")
        self.assertEqual(data["data_kind"], "deterministic_fixture_non_empirical")

    def test_missing_complete_arm_tuple_is_coverage_gap_and_not_zero(self):
        self.use_authored()
        directory = self.run_dir / "arms" / "unconditional"
        target = read_jsonl(directory / "runs.jsonl")[0]
        key = target["task_id"], target["repeat"]
        for name in ("runs", "calls", "attempts"):
            path = directory / (name + ".jsonl")
            write_jsonl(path, [row for row in read_jsonl(path)
                               if (row["task_id"], row["repeat"]) != key])
        data = self.analyze()
        condition = next(row for row in data["conditions"] if row["condition"] == "unconditional")
        self.assertEqual(condition["expected_trials"], 6)
        self.assertEqual(condition["observed_trials"], 5)
        self.assertEqual(condition["missing_trials"], 1)
        self.assertEqual(condition["api_or_pipeline_failures"], 0)
        contrast = next(row for row in data["comparisons"]
                        if row["baseline_condition"] == "none" and row["condition"] == "unconditional")
        self.assertEqual(contrast["n_pairs"], 5)
        self.assertEqual(contrast["baseline_only"], 1)
        self.assertEqual(contrast["treatment_only"], 0)
        self.assertEqual(contrast["baseline_only_slots"][0]["task_id"], key[0])
        self.assertEqual(contrast["baseline_only_slots"][0]["repeat"], key[1])

    def test_upstream_transport_failure_is_nullable_and_never_a_scored_zero(self):
        shutil.rmtree(self.run_dir)
        shutil.copytree(self.error_fixture, self.run_dir)
        data = self.analyze()
        self.assertEqual(data["shared_upstream"]["cached_draws"], 6)
        self.assertEqual(data["shared_upstream"]["ready_draws"], 5)
        failed = [row for row in self.cache() if row["status"] == "error"]
        self.assertEqual(len(failed), 1)
        for condition in data["conditions"]:
            self.assertEqual(condition["observed_trials"], 6)
            self.assertEqual(condition["scored_trials"], 5)
            self.assertEqual(condition["api_or_pipeline_failures"], 1)
            records = read_jsonl(self.run_dir / "arms" / condition["condition"] / "runs.jsonl")
            errors = [row for row in records if row["api_failure"]]
            self.assertEqual(len(errors), 1)
            self.assertIsNone(errors[0]["task_success"])
        self.assertTrue(all(row["n_pairs"] == 5 for row in data["comparisons"]))

    def test_tampered_manifest_checksum_is_rejected(self):
        path = self.run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["backend"] = "openai"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            self.analyze()

    def test_rehashed_manifest_cannot_change_fixed_model(self):
        path = self.run_dir / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["settings"]["model"] = "some-other-model"
        manifest["manifest_sha256"] = shared_run._digest_record(manifest, "manifest_sha256")
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "fixed primary design"):
            self.analyze()

    def test_tampered_cache_record_checksum_is_rejected(self):
        self.edit_cache(lambda rows: rows[0].update(handoff="altered"))
        with self.assertRaisesRegex(ValueError, "cache checksum"):
            self.analyze()

    def test_rehashed_cache_cannot_hide_raw_output_hash_mismatch(self):
        self.edit_cache(lambda rows: rows[0].update(agent1_raw="altered"), seal=True)
        with self.assertRaisesRegex(ValueError, "output checksum"):
            self.analyze()

    def test_rehashed_cache_cannot_hide_handoff_hash_mismatch(self):
        self.edit_cache(lambda rows: rows[0].update(handoff_sha256="0" * 64), seal=True)
        with self.assertRaisesRegex(ValueError, "handoff checksum"):
            self.analyze()

    def test_rehashed_cache_must_still_derive_handoff_from_raw(self):
        def edit(rows):
            rows[0]["handoff"] = "substituted handoff"
            rows[0]["handoff_sha256"] = shared_run.sha256(rows[0]["handoff"].encode())
        self.edit_cache(edit, seal=True)
        with self.assertRaisesRegex(ValueError, "native parsing/replay"):
            self.analyze()

    def test_arm_cannot_substitute_shared_handoff_identifier(self):
        path = self.run_dir / "arms" / "unconditional" / "runs.jsonl"
        rows = read_jsonl(path)
        rows[0]["shared_handoff_id"] = "unrelated-draw"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "shared_handoff_id"):
            self.analyze()

    def test_duplicate_handoff_identifiers_across_task_repeat_keys_are_rejected(self):
        self.edit_cache(lambda rows: rows[1].update(shared_handoff_id=rows[0]["shared_handoff_id"]), seal=True)
        with self.assertRaisesRegex(ValueError, "duplicate.*handoff|handoff.*duplicate"):
            self.analyze()

    def test_upstream_input_must_match_cached_native_task_input(self):
        self.edit_upstream(user_input="different task")
        with self.assertRaisesRegex(ValueError, "upstream call|input"):
            self.analyze()

    def test_upstream_system_prompt_must_match_cached_native_prompt(self):
        self.edit_upstream(system_prompt="a different Agent 1 prompt")
        with self.assertRaisesRegex(ValueError, "prompt|upstream call"):
            self.analyze()

    def test_upstream_run_id_must_identify_the_shared_experiment(self):
        self.edit_upstream(run_id="unrelated-experiment-agent1")
        with self.assertRaisesRegex(ValueError, "run_id"):
            self.analyze()

    def test_upstream_raw_output_must_match_durable_cache(self):
        self.edit_upstream(raw_output="different raw response")
        with self.assertRaisesRegex(ValueError, "upstream call"):
            self.analyze()

    def test_agent2_system_prompt_must_match_native_task_held_tools(self):
        path = self.run_dir / "arms" / "none" / "calls.jsonl"
        rows = read_jsonl(path)
        next(row for row in rows if row["stage"] == "agent2")["system_prompt"] = "different tools"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "system prompt|held|schema"):
            self.analyze()


if __name__ == "__main__":
    unittest.main()
