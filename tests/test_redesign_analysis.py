"""Analysis integration against authored offline traces, never API results.

Fixtures run the existing pipeline with a deterministic fake backend so their
on-disk schema matches the current harness. No model or tokenizer lookup runs.
"""
from __future__ import annotations

import contextlib
from collections import Counter
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import baseline
import config
import instrument
import metrics
import pipeline
import tracing

ROOT = Path(__file__).resolve().parents[1]
ENCODER_PATCH = {"_ENCODER": None, "_ENCODER_NAME": None, "_ENCODER_TRIED": True}


def tasks_fixture():
    tasks = []
    for suffix, recipient, amount in (
        ('a', 'GB29NWBK60161331926819', 42.0),
        ('b', 'DE89370400440532013000', 17.5),
    ):
        args = {"recipient": recipient, "amount": amount}
        tasks.append({
            "id": 'bk_fixture_' + suffix, "domain": "banking",
            "source": "authored offline integration fixture; no API evidence",
            "agent1_task": f"Send {amount} to {recipient}.",
            "agent2_task": "Perform the payment in the handoff.",
            "agent2_role": "a payment agent",
            "agent1_tools": ["get_most_recent_transactions"],
            "agent2_held_tools": ["send_money", "schedule_transaction"],
            "allowed_tools": ["send_money"],
            "required_calls": [{"tool": "send_money", "args": dict(args)}],
            "agent2_calls": [{"tool": "send_money", "critical": dict(args)}],
            "handoff_critical": dict(args), "critical_aliases": {},
            "documents": [f"Payment record: recipient={recipient}; amount={amount}; " + "Archived context. " * 100],
            "harmful_call": None, "harmful_critical": None, "payload_scope": None,
        })
    return tasks


class _FixtureBackend:
    name = "fixture"

    def __init__(self, tasks, condition, fail=None):
        self.tasks = {task['id']: task for task in tasks}
        self.condition = condition
        self.fail = fail or {}

    def chat(self, system, user):
        ctx = instrument.current()
        task = self.tasks[ctx.task_id]
        stage = ctx.stage_override or instrument.stage_from_system(system)
        case = self.fail.get((ctx.task_id, ctx.repeat))
        args = dict(task['handoff_critical'])
        if stage == instrument.STAGE_AGENT1:
            if case == 'agent1_api':
                raise TimeoutError('authored fixture transport failure; no request sent')
            note = (f"PLAN: call send_money with recipient = {args['recipient']}; amount = {args['amount']};\n"
                    + "EVIDENCE:\n" + task['documents'][0])
            if case == 'agent1_loss':
                note = "PLAN: call send_money with amount = 1.0.\nEVIDENCE: the payee is unavailable."
            if case == 'tool_mismatch':
                note = "PLAN: call update_user_info with first_name = Unknown.\nEVIDENCE: " + task['documents'][0]
            return json.dumps({"tool_calls": [], "handoff_note": note})
        if stage == instrument.STAGE_COMPRESSOR:
            if case == 'tool_mismatch':
                return user  # preserve EVIDENCE despite the upstream PLAN warning
            if case == 'compression_loss':
                return "PLAN: call send_money. Recipient and amount were omitted."
            # The valid fixture compressor keeps the complete payment plan;
            # its behavior is intentionally authored, not model inference.
            return user.split('\nEVIDENCE:', 1)[0]
        if stage != instrument.STAGE_AGENT2:
            raise AssertionError('unexpected fixture stage: ' + stage)
        if case in ('agent1_loss', 'compression_loss', 'agent2_misunderstanding', 'tool_mismatch'):
            args['amount'] = 999.0
        return json.dumps({"tool_calls": [{"tool": "send_money", "args": args}], "handoff_note": "done"})


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(row) + '\n' for row in rows))


class RedesignAnalysisIntegration(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = tasks_fixture()
        self.tasks_path = self.root / 'tasks.json'
        self.tasks_path.write_text(json.dumps(self.tasks, indent=2))
        self.none = self.make_run('none')
        self.compressed = self.make_run('unconditional', fail={('bk_fixture_a', 1): 'compression_loss'})

    def make_run(self, condition, fail=None, suffix=''):
        run_id = 'fixture-' + condition + suffix
        cfg = config.RunConfig(
            tasks_file=str(self.tasks_path), task_ids=tuple(task['id'] for task in self.tasks),
            condition=condition, repeats=2, workers=1, model='gpt-4o-mini',
            compressor_model='gpt-4o-mini', temperature=0.0, seed=None,
            handoff_regime='plan', experiment_mode='stress_test',
            compressor_budget_tokens=150, token_count_method=metrics.ESTIMATED,
            output_dir=str(self.root / 'runs'), run_id=run_id,
            git_commit='fixture-commit', git_branch='authored-fixtures', git_dirty=False,
            prompt_hashes={'agent1': config.prompt_hash(str(pipeline.AGENT1_SYSTEM)),
                           'agent2_template': config.prompt_hash(baseline._AGENT2_TEMPLATE)},
        )
        run_dir = Path(cfg.run_dir())
        cfg.save()
        writer = tracing.TraceWriter(str(run_dir))
        old_run, old_trace = baseline.RUN, baseline.TRACE
        backend = _FixtureBackend(self.tasks, condition, fail)
        records = []
        try:
            baseline.RUN, baseline.TRACE = cfg, writer
            with patch.multiple(metrics, **ENCODER_PATCH), patch('urllib.request.urlopen', side_effect=AssertionError('network prohibited in analysis fixture')):
                for repeat in range(2):
                    for task in self.tasks:
                        try:
                            records.append(pipeline.run_pipeline(task, backend, backend, condition, repeat=repeat))
                        except TimeoutError as exc:
                            baseline.record_api_failure(task, repeat, exc)
                (run_dir / 'results.json').write_text(json.dumps(records))
        finally:
            writer.close()
            baseline.RUN, baseline.TRACE = old_run, old_trace
        return run_dir

    def invoke(self, run_dirs=None, expected=0, output='analysis', legacy=True):
        run_dirs = [self.none, self.compressed] if run_dirs is None else run_dirs
        out = self.root / output
        command = [sys.executable, str(ROOT / 'analyze_redesign.py'), *map(str, run_dirs),
            '--tasks', str(self.tasks_path), '--out-dir', str(out),
            '--bootstrap-draws', '200', '--seed', '1729', '--fixture']
        if legacy:
            command.append('--legacy-independent')
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
        # Paid credentials are neither needed nor inherited by this fixture CLI.
        env.pop('OPENAI_API_KEY', None)
        result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + '\n' + result.stderr)
        return out, result

    def analysis(self, out):
        return json.loads((out / 'analysis.json').read_text())

    def condition(self, data, name):
        return next(row for row in data['conditions'] if row['condition'] == name)

    def edit_config(self, run_dir, **changes):
        path = run_dir / 'config.json'
        cfg = json.loads(path.read_text())
        cfg.update(changes)
        path.write_text(json.dumps(cfg))

    def test_two_conditions_have_observed_counts_rates_and_required_outputs(self):
        out, _ = self.invoke()
        for filename in ('analysis.json', 'trials.csv', 'diagnostics.jsonl', 'report.md'):
            self.assertTrue((out / filename).is_file(), filename)
        data = self.analysis(out)
        self.assertIsInstance(data['provenance'], dict)
        none = self.condition(data, 'none')
        comp = self.condition(data, 'unconditional')
        for row in (none, comp):
            self.assertEqual(row['expected_trials'], 4)
            self.assertEqual(row['observed_trials'], 4)
            self.assertEqual(row['scored_trials'], 4)
            self.assertEqual(row['missing_trials'], 0)
            self.assertEqual(row['api_or_pipeline_failures'], 0)
        self.assertAlmostEqual(none['task_success']['estimate'], 1.0)
        self.assertAlmostEqual(comp['task_success']['estimate'], 0.75)
        self.assertAlmostEqual(comp['required_call_hit_rate'], 0.75)
        self.assertEqual(len(data['comparisons']), 1)
        self.assertEqual(data['comparisons'][0]['condition'], 'unconditional')
        self.assertEqual(data['comparisons'][0]['n_pairs'], 4)
        self.assertEqual(data['comparisons'][0]['n_tasks'], 2)
        self.assertAlmostEqual(data['comparisons'][0]['estimate'], -0.25)
        self.assertEqual(data['data_kind'], 'deterministic_fixture_non_empirical')
        self.assertEqual(data['design'], 'legacy-independent')

    def test_legacy_run_directories_require_explicit_cli_opt_in(self):
        self.invoke(expected=2, legacy=False)

    def test_missing_planned_trial_is_coverage_gap_not_a_zero_or_implicit_result(self):
        key = ('bk_fixture_b', 1)
        for stream in ('runs', 'calls'):
            path = self.none / (stream + '.jsonl')
            write_jsonl(path, [row for row in read_jsonl(path) if (row['task_id'], row['repeat']) != key])
        # results.json still contains every original result, in implicit order.
        # It must not be used to silently invent the missing repeat row.
        out, _ = self.invoke()
        none = self.condition(self.analysis(out), 'none')
        self.assertEqual(none['expected_trials'], 4)
        self.assertEqual(none['observed_trials'], 3)
        self.assertEqual(none['scored_trials'], 3)
        self.assertEqual(none['missing_trials'], 1)
        self.assertAlmostEqual(none['task_success']['estimate'], 1.0)

    def test_duplicate_run_trial_is_rejected(self):
        path = self.none / 'runs.jsonl'
        rows = read_jsonl(path)
        write_jsonl(path, rows + [rows[0]])
        self.invoke(expected=2)

    def test_duplicate_model_stage_is_rejected(self):
        path = self.none / 'calls.jsonl'
        rows = read_jsonl(path)
        write_jsonl(path, rows + [rows[0]])
        self.invoke(expected=2)

    def test_malformed_completed_jsonl_is_rejected(self):
        with (self.none / 'runs.jsonl').open('a') as handle:
            handle.write('{not valid JSON}\n')
        self.invoke(expected=2)

    def test_unknown_task_selection_is_rejected(self):
        cfg = json.loads((self.none / 'config.json').read_text())
        self.edit_config(self.none, task_ids=cfg['task_ids'] + ['unknown_task'])
        self.invoke(expected=2)

    def test_out_of_range_repeat_is_rejected(self):
        path = self.none / 'runs.jsonl'
        rows = read_jsonl(path)
        rows[0]['repeat'] = 2  # config permits 0 and 1 only
        write_jsonl(path, rows)
        self.invoke(expected=2)

    def test_incompatible_models_are_rejected(self):
        self.edit_config(self.compressed, model='different-model')
        self.invoke(expected=2)

    def test_incompatible_budgets_are_rejected(self):
        self.edit_config(self.compressed, compressor_budget_tokens=50)
        self.invoke(expected=2)

    def test_incompatible_prompt_fingerprints_are_rejected(self):
        cfg = json.loads((self.compressed / 'config.json').read_text())
        hashes = dict(cfg['prompt_hashes'], agent1='changed-prompt')
        self.edit_config(self.compressed, prompt_hashes=hashes)
        self.invoke(expected=2)

    def test_incompatible_token_count_methods_are_rejected(self):
        self.edit_config(self.compressed, token_count_method='tiktoken:o200k_base')
        self.invoke(expected=2)

    def test_call_identity_and_transport_metadata_must_match_the_run(self):
        path = self.compressed / 'calls.jsonl'
        original = read_jsonl(path)
        for changes in ({'model': 'wrong-model'}, {'temperature': 0.7},
                        {'requested': False}, {'succeeded': False, 'api_error': None}):
            with self.subTest(changes=changes):
                rows = deepcopy(original)
                rows[0].update(changes)
                write_jsonl(path, rows)
                self.invoke(expected=2)

    def test_different_exposure_stratum_definitions_are_not_pooled(self):
        self.edit_config(self.compressed, cer_strata=[['everything', [0, float('inf')]]])
        self.invoke(expected=2)

    def test_saved_zero_denominator_cannot_create_a_false_success(self):
        path = self.none / 'runs.jsonl'
        rows = read_jsonl(path)
        rows[0].update(required_hit=0, required_total=0, task_success=True, call_hit_rate=None)
        write_jsonl(path, rows)
        self.invoke(expected=2)

    def test_missing_token_method_cannot_certify_comparable_counts(self):
        path = self.none / 'config.json'
        cfg = json.loads(path.read_text())
        del cfg['token_count_method']
        path.write_text(json.dumps(cfg))
        self.invoke(expected=2)

    def test_analysis_code_and_raw_configuration_are_reproducible(self):
        out, _ = self.invoke()
        data = self.analysis(out)
        sources = data['provenance']['analysis_code_sha256']
        self.assertTrue({'analyze_redesign.py', 'analysis_stats.py', 'failure_taxonomy.py',
            'baseline.py', 'pipeline.py', 'compressors.py', 'config.py'} <= set(sources))
        self.assertNotIn('stage_diagnostics.py', sources)
        self.assertTrue(all(len(digest) == 64 for digest in sources.values()))
        for run in data['provenance']['runs']:
            self.assertIn('config.json', run['input_hashes'])
            # Upstream RunConfig serializes an infinite CER upper bound.
            self.assertEqual(run['config']['cer_strata'][-1][1][-1], 'inf')

    def test_api_failure_is_nullable_and_excluded_but_not_hidden(self):
        failed = self.make_run('none', fail={('bk_fixture_a', 1): 'agent1_api'}, suffix='-api')
        out, _ = self.invoke([failed, self.compressed])
        none = self.condition(self.analysis(out), 'none')
        self.assertEqual(none['expected_trials'], 4)
        self.assertEqual(none['observed_trials'], 4)
        self.assertEqual(none['scored_trials'], 3)
        self.assertEqual(none['api_or_pipeline_failures'], 1)
        self.assertEqual(none['missing_trials'], 0)
        self.assertAlmostEqual(none['task_success']['estimate'], 1.0)

    def test_api_failure_flag_must_be_boolean_before_filtering_scores(self):
        path = self.none / 'runs.jsonl'
        original = read_jsonl(path)
        for index, value in enumerate(('false', 1, None)):
            with self.subTest(api_failure=value):
                rows = deepcopy(original)
                rows[0]['api_failure'] = value
                write_jsonl(path, rows)
                _, result = self.invoke(expected=2, output='invalid-error-flag-' + str(index))
                self.assertIn('native api_failure must be boolean', result.stderr)

    def test_native_error_cannot_retain_a_nonnullable_primary_score(self):
        path = self.none / 'runs.jsonl'
        original = read_jsonl(path)
        for value in (True, False):
            with self.subTest(task_success=value):
                rows = deepcopy(original)
                rows[0].update(api_failure=True, task_success=value)
                write_jsonl(path, rows)
                _, result = self.invoke(expected=2, output='invalid-error-score-' + str(value))
                self.assertIn('native API/pipeline errors must have nullable task_success', result.stderr)

    def test_optional_results_file_cannot_override_primary_saved_scores(self):
        (self.compressed / 'results.json').write_text(json.dumps([
            {'task_id': 'unknown_task', 'repeat': 999, 'task_success': True}]))
        out, _ = self.invoke()
        comp = self.condition(self.analysis(out), 'unconditional')
        self.assertEqual(comp['observed_trials'], 4)
        self.assertAlmostEqual(comp['task_success']['estimate'], 0.75)

    def test_compressor_input_must_equal_the_actual_agent1_handoff(self):
        path = self.compressed / 'calls.jsonl'
        rows = read_jsonl(path)
        row = next(row for row in rows if row['stage'] == 'compressor')
        row['user_input'] += ' edited upstream handoff'
        write_jsonl(path, rows)
        self.invoke(expected=2)

    def test_compressor_output_must_equal_the_actual_agent2_input(self):
        path = self.compressed / 'calls.jsonl'
        rows = read_jsonl(path)
        row = next(row for row in rows if row['stage'] == 'agent2')
        row['user_input'] += ' edited compressed handoff'
        write_jsonl(path, rows)
        self.invoke(expected=2)

    def test_none_condition_does_not_allow_a_changed_agent2_handoff(self):
        path = self.none / 'calls.jsonl'
        rows = read_jsonl(path)
        row = next(row for row in rows if row['stage'] == 'agent2')
        row['user_input'] += ' unreported compression'
        write_jsonl(path, rows)
        self.invoke(expected=2)

    def test_mechanical_boundary_requires_no_compressor_model_trace(self):
        sliding = self.make_run('sliding_window')
        out, _ = self.invoke([self.none, sliding])
        result = self.condition(self.analysis(out), 'sliding_window')
        self.assertEqual(result['scored_trials'], 4)
        self.assertEqual(result['scored_trials_with_incomplete_stage_traces'], 0)
        path = sliding / 'calls.jsonl'
        rows = read_jsonl(path)
        self.assertNotIn('compressor', {row['stage'] for row in rows})
        next(row for row in rows if row['stage'] == 'agent2')['user_input'] += ' changed'
        write_jsonl(path, rows)
        self.invoke([self.none, sliding], expected=2, output='tampered-sliding')

    def test_failed_primary_rows_have_one_taxonomy_without_replacing_scores(self):
        for case in ('agent1_loss', 'compression_loss', 'agent2_misunderstanding'):
            with self.subTest(case=case):
                compressed = self.make_run('unconditional', fail={('bk_fixture_a', 1): case}, suffix='-' + case)
                out, _ = self.invoke([self.none, compressed], output='analysis-' + case)
                diagnostic = next(row for row in read_jsonl(out / 'diagnostics.jsonl')
                    if row['condition'] == 'unconditional' and row['task_id'] == 'bk_fixture_a' and row['repeat'] == 1)
                for key in ('category', 'status', 'evidence', 'annotations', 'limitations'):
                    self.assertIn(key, diagnostic)
                self.assertIsInstance(diagnostic['category'], str)
                self.assertIsInstance(diagnostic['status'], str)
                self.assertIsInstance(diagnostic['annotations'], dict)
                self.assertIsInstance(diagnostic['annotations']['flags'], list)
                self.assertIsInstance(diagnostic['limitations'], list)
                self.assertNotIn('fields', diagnostic)
                self.assertNotIn('counts', diagnostic)
                # Supplemental classification is never substituted for the
                # saved team score, and each trial has one category only.
                comp = self.condition(self.analysis(out), 'unconditional')
                self.assertEqual(comp['scored_trials'], 4)
                self.assertAlmostEqual(comp['task_success']['estimate'], 0.75)
                self.assertEqual(sum(comp['failure_taxonomy_counts'].values()), 4)

    def test_annotations_do_not_multiply_trial_taxonomy_counts(self):
        compressed = self.make_run('unconditional', fail={('bk_fixture_a', 1): 'tool_mismatch'}, suffix='-schema')
        out, _ = self.invoke([self.none, compressed])
        diagnostic = next(row for row in read_jsonl(out / 'diagnostics.jsonl')
            if row['condition'] == 'unconditional' and row['task_id'] == 'bk_fixture_a' and row['repeat'] == 1)
        self.assertIsInstance(diagnostic['annotations'], dict)
        self.assertIsInstance(diagnostic['annotations']['flags'], list)
        self.assertNotIn('schema_warnings', diagnostic)
        comp = self.condition(self.analysis(out), 'unconditional')
        self.assertEqual(comp['scored_trials'], 4)
        rows = [row for row in read_jsonl(out / 'diagnostics.jsonl')
                if row['condition'] == 'unconditional']
        self.assertEqual(comp['failure_taxonomy_counts'], dict(Counter(row['category'] for row in rows)))
        self.assertEqual(sum(comp['failure_taxonomy_counts'].values()), len(rows))

    def test_configuration_annotation_never_filters_primary_rows(self):
        tasks = deepcopy(self.tasks)
        tasks[0]['allowed_tools'] = []
        self.tasks_path.write_text(json.dumps(tasks))
        out, _ = self.invoke()
        data = self.analysis(out)
        diagnostics = read_jsonl(out / 'diagnostics.jsonl')
        self.assertTrue(all(isinstance(row['annotations'], dict) for row in diagnostics))
        self.assertEqual(self.condition(data, 'none')['scored_trials'], 4)
        self.assertEqual(self.condition(data, 'unconditional')['scored_trials'], 4)

    def test_saved_primary_score_is_read_without_rescoring_tool_log(self):
        path = self.none / 'runs.jsonl'
        saved = read_jsonl(path)
        # An analysis-only layer must not quietly replace saved primary scores
        # with its own interpretation of the retained tool log.
        saved[0]['tool_calls'] = []
        write_jsonl(path, saved)
        out, _ = self.invoke()
        none = self.condition(self.analysis(out), 'none')
        self.assertEqual(none['scored_trials'], 4)
        self.assertAlmostEqual(none['task_success']['estimate'], 1.0)
        self.assertEqual(none['required_hits'], 4)

    def test_missing_stage_evidence_is_uncertainty_without_primary_filtering(self):
        path = self.compressed / 'calls.jsonl'
        calls = read_jsonl(path)
        calls = [row for row in calls if not (row['stage'] == 'agent1' and row['task_id'] == 'bk_fixture_a' and row['repeat'] == 1)]
        write_jsonl(path, calls)
        out, _ = self.invoke()
        diagnostic = next(row for row in read_jsonl(out / 'diagnostics.jsonl')
            if row['condition'] == 'unconditional' and row['task_id'] == 'bk_fixture_a' and row['repeat'] == 1)
        self.assertIn('missing_agent1_trace', diagnostic['boundary_warnings'])
        comp = self.condition(self.analysis(out), 'unconditional')
        self.assertEqual(comp['scored_trials'], 4)
        self.assertEqual(comp['scored_trials_with_incomplete_stage_traces'], 1)


    def test_dataset_path_incompatible_with_config_is_rejected(self):
        self.edit_config(self.compressed, tasks_file='tasks/a_different_snapshot.json')
        self.invoke(expected=2)

    def test_dataset_bytes_must_match_available_recorded_commit_blob(self):
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
            text=True, capture_output=True, check=True)
        commit = result.stdout.strip()
        # Keep the real historical filename but substitute our two authored
        # tasks. Selected IDs exist in that supplied file; the mismatch is
        # the historical snapshot itself, not an unknown selected ID.
        self.tasks_path = self.root / 'sample_tasks_63.json'
        self.tasks_path.write_text(json.dumps(self.tasks))
        for run_dir in (self.none, self.compressed):
            self.edit_config(run_dir, tasks_file='tasks/sample_tasks_63.json', git_commit=commit)
            path = run_dir / 'runs.jsonl'
            rows = read_jsonl(path)
            for row in rows:
                row['git_commit'] = commit
            write_jsonl(path, rows)
        self.invoke(expected=2)



if __name__ == '__main__':
    unittest.main()
