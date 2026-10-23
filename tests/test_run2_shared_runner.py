"""Shared-handoff experiments use fixture models exclusively; no HTTP calls."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import MockBackend
from scripts import run2


class CountingFixture(MockBackend):
    def __init__(self, out_dir, fail_agent2_at=None):
        super().__init__()
        self.out_dir = out_dir
        self.counts = {'agent1': 0, 'agent2': 0, 'compression': 0}
        self.fail_agent2_at = fail_agent2_at
        self.require_cache = True

    def chat(self, system, user):
        stage = 'agent1' if system.startswith('You are Agent 1') else 'agent2' if system.startswith('You are Agent 2') else 'compression'
        self.counts[stage] += 1
        self.last_response_metadata = {}
        if stage in ('agent2', 'compression') and self.require_cache:
            # Cache must be on disk and newline-terminated BEFORE controlled
            # work begins; an in-memory cache does not satisfy this protocol.
            saved = (self.out_dir / 'handoffs.jsonl').read_bytes()
            assert saved.endswith(b'\n')
            assert all(json.loads(row)['prepared_agent1']['model_requests'][0]['stage'] == 'agent1' for row in saved.splitlines())
        if stage == 'agent2' and self.counts[stage] == self.fail_agent2_at:
            raise RuntimeError('fixture downstream transport interruption')
        response = super().chat(system, user)
        self.last_response_metadata = {'usage': {'prompt_tokens': 5, 'completion_tokens': 7, 'total_tokens': 12}}
        return response


@pytest.fixture
def setup_runner(tmp_path, monkeypatch):
    for name in ('MAS_BACKEND', 'MAS_MODEL', 'MAS_TEMPERATURE', 'MAS_DELEGATE', 'MAS_BASE_URL', 'OPENAI_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(run2, 'code_identity', lambda: {'commit': 'fixture', 'tree': 'fixture', 'dirty': False, 'files': {}})
    monkeypatch.setattr(run2, 'runtime_identity', lambda config: {'python': 'fixture', 'implementation': 'fixture', 'tiktoken_version': None, 'tokenizer': {'name': 'unavailable', 'available': False}})
    def setup(fail_at=None, task_ids='ws_6', design='shared-handoff'):
        out = tmp_path / 'run'
        fixture = CountingFixture(out, fail_agent2_at=fail_at)
        fixture.require_cache = design == 'shared-handoff'
        monkeypatch.setattr(run2, 'make_backends', lambda: (fixture, fixture))
        argv = ['--phase', 'smoke', '--backend', 'mock', '--out-dir', str(out), '--task-ids', task_ids, '--design', design]
        return out, fixture, argv
    return setup


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_six_conditions_reuse_exact_agent1_evidence(setup_runner):
    out, fixture, argv = setup_runner()
    assert run2.main(argv) == 0
    assert fixture.counts == {'agent1': 1, 'agent2': 6, 'compression': 4}
    cached = rows(out / 'handoffs.jsonl')
    records = rows(out / 'raw.jsonl')
    assert len(cached) == 1 and len(records) == 6
    artifact = cached[0]
    for field in ('shared_handoff_id', 'shared_handoff_sha256', 'prepared_agent1_sha256'):
        assert {record[field] for record in records} == {artifact[field]}
    assert {record['text']['handoff'] for record in records} == {artifact['prepared_agent1']['handoff']}
    assert {record['text']['agent1_raw'] for record in records} == {artifact['prepared_agent1']['agent1_raw']}
    assert all(record['model_requests'][0]['reused'] is True for record in records)
    assert all(not request.get('reused') for record in records for request in record['model_requests'][1:])
    assert all(record['usage_scope'] == 'condition_requests_excluding_shared_agent1' for record in records)
    assert all(record['shared_agent1_usage_scope'] == 'recorded_once_in_handoffs.jsonl' for record in records)
    assert records[0]['condition'] == 'none'
    assert artifact['prepared_agent1']['model_requests'][0].get('reused') is not True


def test_resume_reuses_durable_upstream_even_without_any_completed_arm(setup_runner):
    out, fixture, argv = setup_runner(fail_at=1)
    assert run2.main(argv) == 1
    cache_bytes = (out / 'handoffs.jsonl').read_bytes()
    assert fixture.counts['agent1'] == 1
    assert json.loads((out / 'results.json').read_text()) == []
    assert run2.main(argv) == 0
    assert fixture.counts['agent1'] == 1
    assert (out / 'handoffs.jsonl').read_bytes() == cache_bytes
    assert len(rows(out / 'raw.jsonl')) == 6
    error = rows(out / 'errors.jsonl')[0]
    assert error['stage'] == 'agent2'
    assert error['shared_handoff_id'] == rows(out / 'handoffs.jsonl')[0]['shared_handoff_id']


def test_resume_with_completed_baseline_never_redraws_agent1(setup_runner):
    out, fixture, argv = setup_runner(fail_at=2, task_ids='ws_6,ws_32')
    assert run2.main(argv) == 1
    assert fixture.counts['agent1'] == 2
    assert len(rows(out / 'raw.jsonl')) == 1
    cache_bytes = (out / 'handoffs.jsonl').read_bytes()
    assert run2.main(argv) == 0
    assert fixture.counts['agent1'] == 2
    assert (out / 'handoffs.jsonl').read_bytes() == cache_bytes
    records = rows(out / 'raw.jsonl')
    assert len(records) == 12
    assert all(record['condition'] == 'none' for record in records[:2])
    before = deepcopy(fixture.counts)
    assert run2.main(argv) == 0
    assert fixture.counts == before


def test_cache_corruption_fails_before_backend_construction(setup_runner, monkeypatch):
    out, _, argv = setup_runner(fail_at=1)
    assert run2.main(argv) == 1
    artifact = rows(out / 'handoffs.jsonl')[0]
    artifact['prepared_agent1']['handoff'] += 'tampered'
    (out / 'handoffs.jsonl').write_text(json.dumps(artifact) + '\n')
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('corrupt cache cannot construct backend'))
    assert run2.main(argv) == 2


def test_recomputed_checksums_cannot_hide_edited_handoff(setup_runner, monkeypatch):
    out, _, argv = setup_runner(fail_at=1)
    assert run2.main(argv) == 1
    artifact = rows(out / 'handoffs.jsonl')[0]
    artifact['prepared_agent1']['handoff'] += 'edited instruction'
    artifact['prepared_agent1_sha256'] = run2.sha256(run2.canonical(artifact['prepared_agent1']))
    artifact['shared_handoff_sha256'] = run2.sha256(run2.canonical({k: v for k, v in artifact.items() if k != 'shared_handoff_sha256'}))
    (out / 'handoffs.jsonl').write_text(json.dumps(artifact) + '\n')
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('edited cache cannot construct backend'))
    assert run2.main(argv) == 2


def test_missing_cache_with_completed_trial_fails_closed(setup_runner, monkeypatch):
    out, _, argv = setup_runner()
    assert run2.main(argv) == 0
    (out / 'handoffs.jsonl').unlink()
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('missing completed cache cannot redraw upstream'))
    assert run2.main(argv) == 2


def test_saved_arm_cannot_switch_shared_handoff_link(setup_runner, monkeypatch):
    out, _, argv = setup_runner()
    assert run2.main(argv) == 0
    records = rows(out / 'raw.jsonl')
    records[-1]['shared_handoff_sha256'] = 'different'
    (out / 'raw.jsonl').write_text(''.join(json.dumps(record) + '\n' for record in records))
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('mismatched arm cannot construct backend'))
    assert run2.main(argv) == 2


def test_design_switch_rejected_on_resume(setup_runner, monkeypatch):
    out, _, argv = setup_runner()
    assert run2.main(argv) == 0
    switched = list(argv)
    switched[-1] = 'independent'
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('mixed design cannot construct backend'))
    assert run2.main(switched) == 2


def test_independent_remains_default_and_draws_agent1_each_arm(setup_runner):
    out, fixture, argv = setup_runner(design='independent')
    argv = argv[:-2]  # omit --design entirely: default must remain unchanged
    assert run2.main(argv) == 0
    assert fixture.counts == {'agent1': 6, 'agent2': 6, 'compression': 4}
    assert not (out / 'handoffs.jsonl').exists()
    manifest = json.loads((out / 'manifest.json').read_text())
    assert manifest['identity']['design'] == 'independent'
    records = rows(out / 'raw.jsonl')
    assert all(record['design'] == 'independent' for record in records)
    assert all('shared_handoff_id' not in record for record in records)
    assert all(not request.get('reused') for record in records for request in record['model_requests'])


@pytest.mark.parametrize('phase,design,expected', [('smoke', 'independent', 32), ('smoke', 'shared-handoff', 22), ('full', 'independent', 864), ('full', 'shared-handoff', 594)])
def test_design_specific_planning_counts(phase, design, expected):
    tasks, digest, hashes = run2.load_tasks(ROOT / 'tasks/sample_tasks.json')
    selected = ['ws_6', 'ws_32'] if phase == 'smoke' else [task['id'] for task in tasks]
    identity = run2.make_identity(tasks, digest, hashes, run2.configuration('mock', {}), phase, selected,
        code={'commit': 'fixture', 'tree': 'fixture', 'dirty': False, 'files': {}}, design=design)
    assert run2.planning_estimate(identity, tasks)['planned_model_requests'] == expected
    assert identity['configuration'].get('design') is None


def test_shared_cache_tail_repair_keeps_committed_handoff(setup_runner):
    out, fixture, argv = setup_runner(fail_at=1)
    assert run2.main(argv) == 1
    cache = out / 'handoffs.jsonl'
    committed = cache.read_bytes()
    with cache.open('ab') as handle:
        handle.write(b'{"interrupted_cache_entry":')
    assert run2.main(argv) == 0
    assert fixture.counts['agent1'] == 1
    assert cache.read_bytes() == committed


def test_shared_usage_counts_each_upstream_request_exactly_once(setup_runner):
    out, fixture, argv = setup_runner()
    assert run2.main(argv) == 0
    arms, caches = rows(out / 'raw.jsonl'), rows(out / 'handoffs.jsonl')
    # Each fixture request reports 12 provider tokens. Ten fresh downstream
    # requests plus one cached upstream request must total eleven, not sixteen.
    upstream = sum(row['prepared_agent1']['provider_usage']['total_tokens'] for row in caches)
    downstream = sum(row['provider_usage']['total_tokens'] for row in arms)
    assert upstream == 12
    assert downstream == 120
    assert upstream + downstream == sum(fixture.counts.values()) * 12 == 132


def test_cache_persistence_failure_retains_paid_upstream_evidence(setup_runner, monkeypatch):
    out, fixture, argv = setup_runner()
    original = run2.append_jsonl
    def fail_cache_fsync(path, value):
        if path.name == 'handoffs.jsonl':
            raise OSError('fixture cache persistence failure')
        return original(path, value)
    monkeypatch.setattr(run2, 'append_jsonl', fail_cache_fsync)
    assert run2.main(argv) == 1
    assert fixture.counts == {'agent1': 1, 'agent2': 0, 'compression': 0}
    error = rows(out / 'errors.jsonl')[0]
    assert error['partial_trial']['agent1_raw']
    assert error['partial_trial']['provider_usage']['total_tokens'] == 12
    assert error['shared_handoff_durability_unconfirmed']['prepared_agent1']['agent1_raw'] == error['partial_trial']['agent1_raw']
    assert error['shared_handoff_durability_unconfirmed']['shared_handoff_id'] == error['shared_handoff_id']
    assert error['counts_as_task_failure'] is False
    assert not (out / 'raw.jsonl').exists()


def test_rehashed_upstream_prompt_edit_is_rejected_on_completed_run(setup_runner, monkeypatch):
    out, _, argv = setup_runner()
    assert run2.main(argv) == 0
    artifact = rows(out / 'handoffs.jsonl')[0]
    artifact['prepared_agent1']['agent1_input'] += 'edited task'
    artifact['prepared_agent1']['model_requests'][0]['user'] += 'edited task'
    artifact['prepared_agent1_sha256'] = run2.sha256(run2.canonical(artifact['prepared_agent1']))
    artifact['shared_handoff_sha256'] = run2.sha256(run2.canonical({k: v for k, v in artifact.items() if k != 'shared_handoff_sha256'}))
    (out / 'handoffs.jsonl').write_text(json.dumps(artifact) + '\n')
    monkeypatch.setattr(run2, 'make_backends', lambda: pytest.fail('edited upstream prompt must fail before backend'))
    assert run2.main(argv) == 2


@pytest.mark.parametrize('smoke_design,full_design', [('independent', 'shared-handoff'), ('shared-handoff', 'independent')])
def test_real_smoke_gate_cannot_mix_designs(tmp_path, smoke_design, full_design):
    tasks, digest, hashes = run2.load_tasks(ROOT / 'tasks/sample_tasks.json')
    config = run2.configuration('openai', {})
    code = {'commit': 'fixture', 'tree': 'fixture', 'dirty': False, 'files': {}}
    smoke = run2.make_identity(tasks, digest, hashes, config, 'smoke', ['ws_6', 'ws_32'], code=code, design=smoke_design)
    run2.bind_manifest(tmp_path, smoke, 'smoke', 'design-fixture')
    full = run2.make_identity(tasks, digest, hashes, config, 'full', [task['id'] for task in tasks], code=code, design=full_design)
    with pytest.raises(ValueError, match='design mismatch'):
        run2.verify_smoke(tmp_path, full)
