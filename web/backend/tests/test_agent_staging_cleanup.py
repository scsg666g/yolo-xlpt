import fcntl
import importlib.util
import json
import os
import threading
import time
from pathlib import Path
from urllib.error import URLError

import pytest


@pytest.fixture
def agent():
    path = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('staging_agent_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path, **overrides):
    return {'training_state_dir': str(tmp_path / '.training-state'),
            'node_id': 'e' * 32, 'center_url': 'http://center.example', **overrides}


def staging(tmp_path, token, *, owner=None, identity='a' * 32):
    state = tmp_path / '.dataset-state' / identity / token
    state.mkdir(parents=True)
    (state / 'bundle.tar').write_bytes(b'pending archive')
    if owner is not None:
        (state / 'owner.json').write_text(json.dumps(owner), encoding='utf-8')
    stamp = time.time() - 7200
    os.utime(state, (stamp, stamp))
    return state


def test_staging_reconciliation_removes_only_approved_unlocked_tokens(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    owner = agent.dataset_staging_owner(cfg)
    approved = staging(tmp_path, 'b' * 32, owner=owner)
    protected = staging(tmp_path, 'c' * 32, owner=owner)
    busy = staging(tmp_path, 'd' * 32, owner=owner)
    calls = []
    def api(config, route, method, payload, **kwargs):
        calls.append(payload)
        return {'reclaimable': [{'id': 'a' * 32, 'token': token} for token in ('b' * 32, 'd' * 32)]}
    monkeypatch.setattr(agent, 'dataset_api', api)
    with (busy / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = time.time() - 7200
        os.utime(busy, (stamp, stamp))
        assert agent.reclaim_dataset_staging(cfg) == 1
    assert not approved.exists()
    assert protected.is_dir() and busy.is_dir()
    assert len(calls[0]['receipts']) == 3
    assert agent.reclaim_dataset_staging(cfg) == 0
    assert len(calls) == 1


def test_staging_reconciliation_preserves_offline_and_recent_work(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    owner = agent.dataset_staging_owner(cfg)
    old = staging(tmp_path, 'b' * 32, owner=owner)
    recent = staging(tmp_path, 'c' * 32, owner=owner)
    recent.touch()
    def offline(*args, **kwargs):
        raise URLError('offline')
    monkeypatch.setattr(agent, 'dataset_api', offline)
    assert agent.reclaim_dataset_staging(cfg) == 0
    assert old.is_dir() and recent.is_dir()


def test_staging_reconciliation_preserves_other_center_node_and_legacy_receipts(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    own = staging(tmp_path, 'b' * 32, owner=agent.dataset_staging_owner(cfg))
    other_center = staging(tmp_path, 'c' * 32, owner=agent.dataset_staging_owner(
        config(tmp_path, center_url='http://other-center.example')))
    other_node = staging(tmp_path, 'd' * 32, owner=agent.dataset_staging_owner(
        config(tmp_path, node_id='f' * 32)))
    legacy = staging(tmp_path, 'f' * 32)
    for state in (other_center, other_node, legacy):
        (state / 'result.json').write_text('{"published":true}', encoding='utf-8')
        stamp = time.time() - 7200
        os.utime(state, (stamp, stamp))
    before = {str(path): path.read_bytes() for state in (other_center, other_node, legacy)
              for path in state.iterdir()}
    calls = []
    def api(config, route, method, payload, **kwargs):
        calls.extend(payload['receipts'])
        # A center approves unknown operations, so filtering must happen locally.
        return {'reclaimable': payload['receipts']}
    monkeypatch.setattr(agent, 'dataset_api', api)
    assert agent.reclaim_dataset_staging(cfg) == 1
    assert not own.exists()
    assert calls == [{'id': 'a' * 32, 'token': 'b' * 32}]
    assert before == {str(path): path.read_bytes() for state in (other_center, other_node, legacy)
                      for path in state.iterdir()}


def test_staging_reconciliation_rotates_past_large_protected_backlog(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    owner = agent.dataset_staging_owner(cfg)
    states = [staging(tmp_path, 'b' * 32, owner=owner, identity=f'{n:032x}') for n in range(1025)]
    eligible = states[-1]
    calls = []
    def api(config, route, method, payload, **kwargs):
        calls.extend(payload['receipts'])
        return {'reclaimable': [item for item in payload['receipts']
                                if item['id'] == eligible.parent.name]}
    monkeypatch.setattr(agent, 'dataset_api', api)
    now = time.time()
    assert agent.reclaim_dataset_staging(cfg, now=now) == 0
    assert len(calls) == 1024 and eligible.is_dir()
    assert agent.reclaim_dataset_staging(cfg, now=now + 301) == 1
    assert not eligible.exists()
    assert all(state.is_dir() for state in states[:-1])


def test_dataset_phase_records_owner_after_authenticated_lease_check(agent, tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    cfg = config(tmp_path, dataset_roots=[str(source)], training_cwd=str(tmp_path), token='secret-token')
    work = {'id': 'a' * 32, 'token': 'b' * 32, 'phase': 'check', 'path': str(source)}
    calls = []
    def api(config, route, method='GET', payload=None, **kwargs):
        calls.append(route)
        return {}
    monkeypatch.setattr(agent, 'dataset_api', api)
    agent.run_dataset_operation(cfg, work, threading.Event())
    state = tmp_path / '.dataset-state' / work['id'] / work['token']
    assert calls[0].endswith('/heartbeat') and calls[-1].endswith('/finish')
    assert json.loads((state / 'owner.json').read_text()) == agent.dataset_staging_owner(cfg)
    assert 'secret-token' not in (state / 'owner.json').read_text()
    assert (state / 'result.json').is_file() and (state / 'acked').is_file()


def test_failed_lease_does_not_claim_unknown_legacy_state(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    state = staging(tmp_path, 'b' * 32)
    def offline(*args, **kwargs):
        raise URLError('offline')
    monkeypatch.setattr(agent, 'dataset_api', offline)
    agent.run_dataset_operation(cfg, {'id': 'a' * 32, 'token': 'b' * 32, 'phase': 'export'}, threading.Event())
    assert (state / 'bundle.tar').read_bytes() == b'pending archive'
    assert not (state / 'owner.json').exists()


def test_legacy_completed_receipt_replays_without_becoming_reclaimable(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    state = staging(tmp_path, 'b' * 32)
    receipt = {'resolved_path': '/legacy-dataset', 'status': 'ready'}
    (state / 'result.json').write_text(json.dumps(receipt), encoding='utf-8')
    calls = []
    def api(config, route, method='GET', payload=None, **kwargs):
        calls.append((route, payload))
        return {}
    monkeypatch.setattr(agent, 'dataset_api', api)
    agent.run_dataset_operation(cfg, {'id': 'a' * 32, 'token': 'b' * 32, 'phase': 'check'}, threading.Event())
    assert len(calls) == 1 and calls[0][0].endswith('/finish')
    assert calls[0][1]['result'] == receipt
    assert json.loads((state / 'result.json').read_text()) == receipt
    assert not (state / 'owner.json').exists()
    stamp = time.time() - 7200
    os.utime(state, (stamp, stamp))
    assert agent.reclaim_dataset_staging(cfg) == 0
    assert len(calls) == 1 and state.is_dir()


def test_agent_provenance_extension_is_only_sent_after_negotiation(agent):
    payload = {'status': 'running'}
    task = {'dataset_snapshot': {'version': 1}}
    assert agent.training_report_payload(task, payload) is payload
    task['dataset_provenance_version'] = 1
    assert agent.training_report_payload(task, payload) == {**payload, 'dataset_snapshot': {'version': 1}}
