"""No API calls: test no-attack boundaries, journals, and smoke/full identity."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run2


def task_data():
    return json.loads((ROOT / 'tasks/sample_tasks.json').read_text())


def identity(phase='full', backend='mock'):
    tasks = task_data()
    hashes = {t['id']: run2.sha256(run2.canonical(t)) for t in tasks}
    config = run2.configuration(backend, {})
    return run2.make_identity(tasks, 'dataset', hashes, config, phase,
        ['ws_6', 'ws_7'] if phase == 'smoke' else list(run2.TASK_IDS),
        code={'commit': 'commit', 'tree': 'tree', 'dirty': False, 'files': {'pipeline.py': 'hash'}})


def manifest(tmp_path, phase='full', backend='mock'):
    return run2.bind_manifest(tmp_path, identity(phase, backend), phase, 'test-run')


def record(man, task='ws_6', cond='none', repeat=0):
    config = man['identity']['configuration']
    task_spec = next(t for t in task_data() if t['id'] == task)
    calls = []
    for call in task_spec['required_calls']:
        if call['tool'] in task_spec['agent1_tools']:
            continue
        args = deepcopy(call['args'])
        for key, mode in call.get('match', {}).items():
            if mode == 'any_of':
                args[key] = args[key][0]
        calls.append({'tool': call['tool'], 'args': args})
    a1 = json.dumps({'tool_calls': [], 'handoff_note': 'action details'})
    a2 = json.dumps({'tool_calls': calls, 'handoff_note': 'done'})
    stages = ['agent1', 'agent2'] if cond in ('none', 'sliding_window') else ['agent1', 'compression', 'agent2']
    requests = [{'stage': stage, 'status': 'completed', 'model': config['model'],
        'backend': config['backend'], 'system': 'system', 'user': 'user',
        'response': a1 if stage == 'agent1' else a2 if stage == 'agent2' else 'action details',
        'provider': {'response_model': 'gpt-4o-2024-08-06'}} for stage in stages]
    return {'task_id': task, 'condition': cond, 'repeat': repeat,
        'run_id': man['run_id'], 'manifest_sha256': man['manifest_sha256'],
        'config_fingerprint': man['config_fingerprint'],
        'configuration': config, 'model': config['model'],
        'temperature': config['temperature'], 'delegate': config['delegate'],
        'backend': config['backend'], 'mock': config['backend'] == 'mock',
        'prompt_version': man['identity']['prompt_version'],
        'trial_attempt_id': str(uuid.uuid4()), 'model_requests': requests,
        'task_sha256': man['identity']['task_sha256'][task],
        'code_commit': man['identity']['code']['commit'], 'task_success': True,
        'attack_fired': False, 'calls': calls,
        'text': {'agent1_raw': a1, 'agent2_raw': a2, 'compressed': 'action details'}}


def write_tasks(tmp_path, tasks):
    path = tmp_path / 'tasks.json'
    path.write_text(json.dumps(tasks))
    return path


@pytest.mark.parametrize('change', ['harm', 'scope', 'marker', 'source', 'duplicate', 'count'])
def test_attack_or_wrong_task_set_rejected(tmp_path, change):
    tasks = task_data()
    if change == 'harm':
        tasks[0]['harmful_call'] = {'tool': 'send_email'}
    elif change == 'scope':
        tasks[0]['payload_scope'] = 'in_scope'
    elif change == 'marker':
        tasks[0]['documents'].append('INJECT_CALL: send_email()')
    elif change == 'source':
        tasks[0]['source'] = 'attack task'
    elif change == 'duplicate':
        tasks[1]['id'] = tasks[0]['id']
    elif change == 'count':
        tasks.pop()
    with pytest.raises(ValueError):
        run2.load_tasks(write_tasks(tmp_path, tasks))


def test_allowlist_must_cover_every_required_write(tmp_path):
    tasks = task_data()
    tasks[0]['allowed_tools'] = []
    with pytest.raises(ValueError, match='allowlist'):
        run2.load_tasks(write_tasks(tmp_path, tasks))


def test_fixed_primary_configuration():
    config = run2.configuration('openai', {})
    assert config['model'] == 'gpt-4o'
    assert config['temperature'] == 0.7 and config['delegate'] is True
    with pytest.raises(ValueError, match='MAS_MODEL'):
        run2.configuration('openai', {'MAS_MODEL': 'gpt-4o-mini'})
    with pytest.raises(ValueError, match='MAS_DELEGATE'):
        run2.configuration('openai', {'MAS_DELEGATE': '0'})
    with pytest.raises(ValueError, match='credentials'):
        run2.configuration('openai', {'MAS_BASE_URL': 'https://secret:token@example.com/v1'})
    with pytest.raises(ValueError, match='requires'):
        run2.configuration('openai', {'MAS_BASE_URL': 'https://example.com/v1'})


def test_full_schedule_baseline_first_and_three_repeats():
    order = run2.schedule(identity())
    assert len(order) == 324
    assert len(set(order)) == 324
    assert all(cond == 'none' for _, cond, _ in order[:54])
    assert sum(2 if cond in ('none', 'sliding_window') else 3 for _, cond, _ in order) == 864
    assert {rep for _, _, rep in order} == {0, 1, 2}


def test_manifest_immutable_resume_and_mismatch_rejected(tmp_path):
    man = manifest(tmp_path)
    before = (tmp_path / 'manifest.json').read_bytes()
    assert run2.bind_manifest(tmp_path, identity(), 'full')['run_id'] == man['run_id']
    assert (tmp_path / 'manifest.json').read_bytes() == before
    modified = identity()
    modified['dataset_sha256'] = 'different'
    with pytest.raises(ValueError, match='resume rejected'):
        run2.bind_manifest(tmp_path, modified, 'full')
    modified = identity()
    modified['code']['files']['pipeline.py'] = 'new code'
    with pytest.raises(ValueError, match='resume rejected'):
        run2.bind_manifest(tmp_path, modified, 'full')


def test_corrupt_manifest_rejected(tmp_path):
    manifest(tmp_path)
    path = tmp_path / 'manifest.json'
    data = json.loads(path.read_text())
    data['run_id'] = 'tampered'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='checksum'):
        run2.bind_manifest(tmp_path, identity(), 'full')


def test_journal_partial_tail_repaired_before_append(tmp_path):
    man = manifest(tmp_path)
    path = tmp_path / 'raw.jsonl'
    first = record(man)
    run2.append_jsonl(path, first)
    with path.open('ab') as handle:
        handle.write(b'{"task_id":"ws_')
    loaded = run2.load_journal(path, man)
    assert len(loaded) == 1
    run2.append_jsonl(path, record(man, task='ws_7'))
    assert len(run2.load_journal(path, man)) == 2
    assert len(path.read_text().splitlines()) == 2


def test_valid_last_record_without_newline_separated(tmp_path):
    man = manifest(tmp_path)
    path = tmp_path / 'raw.jsonl'
    path.write_text(json.dumps(record(man)))
    assert len(run2.load_journal(path, man)) == 1
    run2.append_jsonl(path, record(man, task='ws_7'))
    assert len(run2.load_journal(path, man)) == 2


def test_completed_corrupt_journal_line_is_not_ignored(tmp_path):
    man = manifest(tmp_path)
    path = tmp_path / 'raw.jsonl'
    path.write_bytes(b'{broken}\n')
    with pytest.raises(ValueError, match='malformed completed'):
        run2.load_journal(path, man)
    assert path.read_bytes() == b'{broken}\n'


def test_duplicate_and_foreign_trial_rejected(tmp_path):
    man = manifest(tmp_path)
    path = tmp_path / 'raw.jsonl'
    run2.append_jsonl(path, record(man))
    run2.append_jsonl(path, record(man))
    with pytest.raises(ValueError, match='duplicate'):
        run2.load_journal(path, man)
    wrong = record(man)
    wrong['task_sha256'] = 'changed task'
    with pytest.raises(ValueError, match='task content'):
        run2.validate_record(wrong, man)


def complete_smoke(tmp_path, backend='openai'):
    man = manifest(tmp_path, 'smoke', backend)
    for tid, cond, rep in run2.schedule(man['identity']):
        run2.append_jsonl(tmp_path / 'raw.jsonl', record(man, tid, cond, rep))
    run2.atomic_json(tmp_path / 'status.json', {'status': 'complete'})
    return man


def test_full_gate_requires_real_complete_matching_smoke(tmp_path):
    smoke = complete_smoke(tmp_path)
    full = identity('full', 'openai')
    assert run2.verify_smoke(tmp_path, full) == smoke['run_id']
    full['configuration']['temperature'] = 0
    with pytest.raises(ValueError, match='configuration mismatch'):
        run2.verify_smoke(tmp_path, full)


def test_mock_cannot_unlock_real_sweep(tmp_path):
    complete_smoke(tmp_path, 'mock')
    with pytest.raises(ValueError, match='real OpenAI'):
        run2.verify_smoke(tmp_path, identity('full', 'openai'))


def test_smoke_zero_success_baseline_stops_spend(tmp_path):
    man = manifest(tmp_path, 'smoke', 'openai')
    for tid, cond, rep in run2.schedule(man['identity']):
        item = record(man, tid, cond, rep)
        if cond == 'none':
            item['task_success'] = False
        run2.append_jsonl(tmp_path / 'raw.jsonl', item)
    run2.atomic_json(tmp_path / 'status.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='zero successes|differs from current scorer'):
        run2.verify_smoke(tmp_path, identity('full', 'openai'))


def test_missing_key_writes_blocked_and_never_calls_backend(tmp_path, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    for key in ('MAS_BACKEND', 'MAS_MODEL', 'MAS_TEMPERATURE', 'MAS_DELEGATE', 'MAS_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    # Keep this test independent of the separate allowed_tools patch.
    tasks = task_data()
    for task in tasks:
        task['allowed_tools'] = sorted({c['tool'] for c in task['required_calls'] if c['tool'] not in task['agent1_tools']})
    tasks_path = write_tasks(tmp_path, tasks)
    def forbidden():
        pytest.fail('missing key must never construct a network backend')
    monkeypatch.setattr(run2, 'make_backends', forbidden)
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {}})
    result = run2.main(['--phase', 'smoke', '--out-dir', str(tmp_path / 'blocked'), '--tasks', str(tasks_path)])
    assert result == 2
    status = json.loads((tmp_path / 'blocked/status.json').read_text())
    assert status['status'] == 'blocked' and status['api_calls_made'] == 0
    assert json.loads((tmp_path / 'blocked/results.json').read_text()) == []
    assert not (tmp_path / 'blocked/raw.jsonl').exists()


def test_interrupted_trial_keeps_stage_trace_but_is_not_scored(tmp_path, monkeypatch):
    from telemetry import PipelineExecutionError
    for key in ('MAS_BACKEND', 'MAS_MODEL', 'MAS_TEMPERATURE', 'MAS_DELEGATE', 'MAS_BASE_URL'):
        monkeypatch.delenv(key, raising=False)
    tasks = task_data()
    for task in tasks:
        task['allowed_tools'] = sorted({c['tool'] for c in task['required_calls'] if c['tool'] not in task['agent1_tools']})
    tasks_path = write_tasks(tmp_path, tasks)
    monkeypatch.setattr(run2, 'make_backends', lambda: (object(), object()))
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {}})
    def fail(*args):
        raise PipelineExecutionError('agent2', RuntimeError('secret not persisted'),
            {'handoff': 'completed Agent 1 note', 'model_requests': [{'stage': 'agent1', 'response': 'completed'}]})
    monkeypatch.setattr(run2.pipeline, 'run_pipeline', fail)
    result = run2.main(['--phase', 'smoke', '--backend', 'mock', '--out-dir', str(tmp_path / 'interrupted'), '--tasks', str(tasks_path)])
    assert result == 1
    error_text = (tmp_path / 'interrupted/errors.jsonl').read_text()
    error = json.loads(error_text)
    assert error['stage'] == 'agent2'
    assert error['counts_as_task_failure'] is False
    assert error['partial_trial']['handoff'] == 'completed Agent 1 note'
    assert 'secret not persisted' not in error_text
    assert json.loads((tmp_path / 'interrupted/results.json').read_text()) == []
    assert not (tmp_path / 'interrupted/raw.jsonl').exists()


def test_substituted_task_documents_rejected_even_with_same_ids(tmp_path):
    tasks = task_data()
    tasks[0]['documents'][0] = 'replacement evaluation context'
    with pytest.raises(ValueError, match='frozen scorer-fix'):
        run2.load_tasks(write_tasks(tmp_path, tasks))


def test_observed_empty_model_response_remains_a_scored_trial(tmp_path):
    man = manifest(tmp_path)
    item = record(man)
    item['model_requests'][1]['response'] = ''
    item['text']['agent2_raw'] = ''
    item['task_success'] = False
    item['calls'] = []
    assert run2.validate_record(item, man) == ('ws_6', 'none', 0)
    run2.append_jsonl(tmp_path / 'raw.jsonl', item)
    assert len(run2.load_journal(tmp_path / 'raw.jsonl', man)) == 1


@pytest.mark.parametrize('field,value', [('model', 'gpt-4o-mini'), ('temperature', 0), ('delegate', False)])
def test_top_level_config_cannot_disagree_with_manifest(tmp_path, field, value):
    man = manifest(tmp_path)
    item = record(man)
    item[field] = value
    with pytest.raises(ValueError, match='top-level'):
        run2.validate_record(item, man)


def test_smoke_success_must_rescore_from_observed_calls(tmp_path):
    man = manifest(tmp_path, 'smoke', 'openai')
    for tid, cond, rep in run2.schedule(man['identity']):
        item = record(man, tid, cond, rep)
        item['calls'] = []
        run2.append_jsonl(tmp_path / 'raw.jsonl', item)
    run2.atomic_json(tmp_path / 'status.json', {'status': 'complete'})
    with pytest.raises(ValueError, match='differs from current scorer'):
        run2.verify_smoke(tmp_path, identity('full', 'openai'))


@pytest.mark.parametrize('text,expected', [
    ('OpenAI insufficient_quota', 'insufficient_quota'),
    ('API error 401', 'authentication'),
    ('Hit the daily request limit', 'daily_request_limit'),
    ('API error 429', 'rate_limit'),
    ('connection timed out', 'timeout'),
])
def test_error_category_preserves_blocker_without_secret_text(text, expected):
    assert run2.error_category(RuntimeError(text)) == expected


def without_mas_env(monkeypatch):
    for key in ('MAS_BACKEND', 'MAS_MODEL', 'MAS_TEMPERATURE', 'MAS_DELEGATE', 'MAS_BASE_URL'):
        monkeypatch.delenv(key, raising=False)


def test_present_key_without_approved_source_and_budget_cannot_construct_backend(tmp_path, monkeypatch):
    without_mas_env(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-secret-do-not-persist')
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {}})
    monkeypatch.setattr(run2, 'runtime_identity', lambda config: {'python': 'test', 'implementation': 'test', 'tiktoken_version': 'test', 'tokenizer': {'name': 'o200k_base', 'available': True}})
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('No approved source/budget: backend must not be constructed'))
    out = tmp_path / 'blocked-budget'
    assert run2.main(['--phase', 'smoke', '--out-dir', str(out)]) == 2
    status_text = (out / 'status.json').read_text()
    status = json.loads(status_text)
    assert status['status'] == 'blocked' and status['api_calls_made'] == 0
    assert any('--max-cost-usd' in blocker for blocker in status['blockers'])
    assert any('--credential-source' in blocker for blocker in status['blockers'])
    assert 'fixture-secret-do-not-persist' not in status_text
    assert 'fixture-secret-do-not-persist' not in (out / 'manifest.json').read_text()


@pytest.mark.parametrize('cap', ['0', '-1', 'NaN', 'Infinity'])
def test_nonpositive_or_nonfinite_cost_cap_never_constructs_backend(tmp_path, monkeypatch, cap):
    without_mas_env(monkeypatch)
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('invalid budget must not construct backend'))
    assert run2.main(['--phase', 'smoke', '--out-dir', str(tmp_path / 'budget'), '--max-cost-usd', cap]) == 2


def test_no_network_preflight_records_exact_blockers_and_secret_presence_only(tmp_path, monkeypatch):
    without_mas_env(monkeypatch)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {'environment.py': 'scorer-hash'}})
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('preflight is strictly read-only'))
    out = tmp_path / 'preflight'
    assert run2.main(['--phase', 'preflight', '--out-dir', str(out)]) == 2
    preflight = json.loads((out / 'preflight.json').read_text())
    assert preflight['api_calls_made'] == 0
    assert preflight['checks']['credential_available'] is False
    assert preflight['checks']['approved_credential_source'] is False
    assert preflight['checks']['positive_cost_cap'] is False
    assert preflight['checks']['scorer_sha256'] == 'scorer-hash'
    assert preflight['planning_estimate']['planned_trials'] == 324
    assert preflight['planning_estimate']['planned_model_requests'] == 864
    assert preflight['planning_estimate']['estimated_usd'] is None
    assert preflight['credential_value_recorded'] is False


def test_plan_budget_and_output_bound_are_frozen_in_manifest(tmp_path, monkeypatch):
    without_mas_env(monkeypatch)
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {}})
    monkeypatch.setattr(run2, 'runtime_identity', lambda config: {'python': 'test', 'implementation': 'test', 'tiktoken_version': None, 'tokenizer': {'name': 'unavailable', 'available': False}})
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('plan must not construct backend'))
    out = tmp_path / 'plan-budget'
    argv = ['--phase', 'plan', '--out-dir', str(out), '--credential-source', 'OPENAI_API_KEY',
        '--max-cost-usd', '0.50', '--input-usd-per-million', '2.50', '--output-usd-per-million', '10.00', '--max-output-tokens', '512']
    assert run2.main(argv) == 0
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['identity']['configuration']['max_output_tokens'] == 512
    assert manifest['identity']['budget'] == {'cap_usd': '0.5', 'input_usd_per_million': '2.5', 'output_usd_per_million': '10', 'credential_source': 'OPENAI_API_KEY'}
    changed = list(argv)
    changed[-1] = '1024'
    assert run2.main(changed) == 2


def test_smoke_full_allows_separate_caps_but_never_changed_inference_bound(tmp_path):
    man = complete_smoke(tmp_path)
    full = identity('full', 'openai')
    full['budget']['cap_usd'] = '5'
    assert run2.verify_smoke(tmp_path, full) == man['run_id']
    full['configuration']['max_output_tokens'] = 512
    with pytest.raises(ValueError, match='configuration mismatch'):
        run2.verify_smoke(tmp_path, full)


def test_preflight_cannot_download_missing_tokenizer_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('TIKTOKEN_CACHE_DIR', str(tmp_path / 'missing-cache'))
    monkeypatch.setattr(run2.importlib.metadata, 'version', lambda name: 'test-version')
    monkeypatch.setattr(run2, 'text_size', lambda *a: pytest.fail('No cache: tokenizer must not be called/downloaded'))
    info = run2.runtime_identity(run2.configuration('openai', {}))
    assert info['tokenizer']['available'] is False


def test_corrupt_tokenizer_cache_is_not_loaded_or_downloaded(tmp_path, monkeypatch):
    cache_key = run2.hashlib.sha1(b'https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken').hexdigest()
    (tmp_path / cache_key).write_bytes(b'corrupt cached encoding')
    monkeypatch.setenv('TIKTOKEN_CACHE_DIR', str(tmp_path))
    monkeypatch.setattr(run2.importlib.metadata, 'version', lambda name: 'test-version')
    monkeypatch.setattr(run2, 'text_size', lambda *a: pytest.fail('corrupt cache must not trigger tokenizer download'))
    assert run2.runtime_identity(run2.configuration('openai', {}))['tokenizer']['available'] is False


def test_runner_attaches_budget_and_stops_before_unaffordable_request(tmp_path, monkeypatch):
    without_mas_env(monkeypatch)
    monkeypatch.setenv('OPENAI_API_KEY', 'fixture-no-network-secret')
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'x', 'tree': 'x', 'dirty': False, 'files': {}})
    monkeypatch.setattr(run2, 'runtime_identity', lambda config: {'python': 'test', 'implementation': 'test', 'tiktoken_version': 'test', 'tokenizer': {'name': 'o200k_base', 'available': True}})
    monkeypatch.setattr(run2, 'planning_estimate', lambda *a: {'planned_trials': 12, 'planned_model_requests': 32})
    class FixtureBackend:
        pass
    backend = FixtureBackend()
    monkeypatch.setattr(run2, 'make_backends', lambda: (backend, backend))
    def fake_pipeline(task, agent, compressor, condition):
        assert agent is compressor
        assert agent.max_output_tokens == 1
        agent.budget.before_request(1_000_000, agent.max_output_tokens)
        pytest.fail('unaffordable reservation must stop the fake transport')
    monkeypatch.setattr(run2.pipeline, 'run_pipeline', fake_pipeline)
    out = tmp_path / 'budget-stop'
    argv = ['--phase', 'smoke', '--out-dir', str(out), '--credential-source', 'OPENAI_API_KEY',
        '--max-cost-usd', '0.001', '--input-usd-per-million', '2.5', '--output-usd-per-million', '10', '--max-output-tokens', '1']
    assert run2.main(argv) == 1
    status = json.loads((out / 'status.json').read_text())
    assert status['completed_trials'] == 0
    assert status['last_error']['error_category'] == 'budget_limit'
    assert status['budget']['accounted_usd'] == '0'
    assert (out / 'budget.jsonl').exists()
    assert 'fixture-no-network-secret' not in (out / 'errors.jsonl').read_text()
