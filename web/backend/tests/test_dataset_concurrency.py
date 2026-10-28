"""Training and light directory work coexist without relaxing file mutation locks."""
import threading
from datetime import timedelta

import pytest

from app.models import ComputeNode, TrainingRun, TrainingTask, utcnow
from test_dataset_storage import (env, location, node, operation, pump, start_scan,
                                  make_existing, candidates, store, finish_local)


def running_node(factory, root, *, concurrent=True):
    identity = node(factory, root)
    with factory() as db:
        n = db.get(ComputeNode, identity)
        n.agent_version = '2.4.1' if concurrent else '2.4.0'
        n.last_seen_at = utcnow()
        n.snapshot = {'training': {'dataset_roots': [str(root)], 'dataset_discovery': True,
                                  'dataset_concurrent_reads': concurrent, 'lock_occupied': True}}
        task = TrainingTask(id='busy-task', name='training', execution_node_id=identity,
                            model_path='model.pt', data_path='data.yaml', status='running')
        db.add(task); db.flush()
        db.add(TrainingRun(id='busy-run', task_id=task.id, attempt_number=1,
                          status='running', execution_node_id=identity,
                          model_path=task.model_path, data_path=task.data_path))
        db.commit()
    return identity


def config(root, identity):
    return {'node_id': identity, 'dataset_roots': [str(root)], 'training_cwd': str(root.parent),
            'training_state_dir': str(root.parent / 'state'), 'training_output_roots': [],
            'allowed_training_roots': [str(root)], 'training_lock_file': str(root.parent / 'workspace.lock'),
            'enable_training': False}


@pytest.mark.parametrize('phase', ['export', 'import', 'remove'])
def test_check_bypasses_blocked_heavy_work_and_runs_with_training_lock(env, phase):
    client, factory, root, agent = env
    remote_root = root / 'remote'; remote_root.mkdir()
    remote = running_node(factory, remote_root)
    with factory() as db:
        heavy = store.new_operation(db, 'register', remote, {'phase': phase, 'source_path': str(remote_root)})
        db.commit(); heavy_id = heavy.id
    location(client, remote_root, remote)
    cfg = config(remote_root, remote)
    lock = agent.acquire_training_lock(cfg)
    assert lock is not None
    try:
        work = agent.poll_dataset_operation(cfg, busy=True)
        assert work['phase'] == 'check'
        view = operation(client, work['id'])
        assert not view['waiting_training'] and not view['waiting_lock']
        agent.run_dataset_operation(cfg, work, threading.Event())
        assert operation(client, work['id'])['status'] == 'completed'
        assert operation(client, heavy_id)['status'] == 'pending'
        assert operation(client, heavy_id)['waiting_training']
        assert agent.poll_dataset_operation(cfg, busy=True) is None
        assert agent.acquire_training_lock(cfg) is None
    finally:
        agent.release_training_lock(lock)


def test_discovery_runs_with_training_and_preserves_source_files(env):
    client, factory, root, agent = env
    remote_root = root / 'remote'
    remote = node(factory, remote_root)
    loc = location(client, remote_root, remote)
    pump(client, factory, remote_root, agent, remote)
    make_existing(remote_root / 'sample')
    (remote_root / 'link').symlink_to(remote_root / 'sample', target_is_directory=True)
    before = {p.relative_to(remote_root): p.read_bytes() for p in remote_root.rglob('*') if p.is_file()}
    with factory() as db:
        n = db.get(ComputeNode, remote)
        n.agent_version = '2.4.1'
        n.snapshot = {'training': {'dataset_roots': [str(remote_root)], 'dataset_discovery': True,
                                  'dataset_concurrent_reads': True, 'lock_occupied': True}}
        db.commit()
    scan, _ = start_scan(client, [loc])
    with factory() as db:
        assert not store.node_dataset_file_work(db, remote)
    cfg = config(remote_root, remote)
    lock = agent.acquire_training_lock(cfg)
    try:
        work = agent.poll_dataset_operation(cfg, busy=True)
        assert work['phase'] == 'discover'
        agent.run_dataset_operation(cfg, work, threading.Event())
        assert operation(client, work['id'])['status'] == 'completed'
        assert candidates(client, scan['id'])['counts']['ready'] == 1
        assert {p.relative_to(remote_root): p.read_bytes() for p in remote_root.rglob('*') if p.is_file()} == before
        assert agent.acquire_training_lock(cfg) is None
    finally:
        agent.release_training_lock(lock)


def test_light_queue_counts_eligible_work_but_never_bypasses_live_lease(env):
    client, factory, root, _ = env
    remote = running_node(factory, root)
    with factory() as db:
        heavy = store.new_operation(db, 'register', remote, {'phase': 'export', 'source_path': str(root)})
        db.flush()
        first = store.new_operation(db, 'check', remote, {'phase': 'check', 'path': str(root)})
        db.flush()
        second = store.new_operation(db, 'check', remote, {'phase': 'check', 'path': str(root)})
        db.commit()
        heavy_id, first_id, second_id = heavy.id, first.id, second.id
    assert operation(client, first_id)['queue_ahead'] == 0
    assert operation(client, second_id)['queue_ahead'] == 1
    with factory() as db:
        heavy = db.get(store.DatasetOperation, heavy_id)
        heavy.status, heavy.lease_until = 'running', utcnow() + timedelta(seconds=120)
        db.commit()
    assert operation(client, first_id)['queue_ahead'] == 1
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll?lightweight_only=true').json()['operation'] is None
    with factory() as db:
        db.get(store.DatasetOperation, heavy_id).lease_until = utcnow() - timedelta(seconds=1)
        db.commit()
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation']['id'] == first_id
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation'] is None


def test_old_agent_keeps_training_wait_reason_and_discovery_barrier(env):
    client, factory, root, _ = env
    remote = running_node(factory, root, concurrent=False)
    with factory() as db:
        op = store.new_operation(db, 'check', remote, {'phase': 'check', 'path': str(root)})
        db.commit(); identity = op.id
    assert operation(client, identity)['waiting_training']
    with factory() as db:
        op = db.get(store.DatasetOperation, identity)
        op.payload = {**op.payload, 'phase': 'discover'}
        db.commit()
        assert store.node_dataset_file_work(db, remote)
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation'] is None


def test_agent_rejects_heavy_work_from_old_center_ignoring_filter(env, monkeypatch):
    _, _, root, agent = env
    calls = []
    def old_center(_config, path, *_args):
        calls.append(path)
        return {'operation': {'phase': 'export'}}
    monkeypatch.setattr(agent, 'dataset_api', old_center)
    cfg = config(root, 'node')
    assert agent.poll_dataset_operation(cfg, busy=True) is None
    assert calls[-1].endswith('?lightweight_only=true')
    assert agent.poll_dataset_operation(cfg, busy=False)['phase'] == 'export'
    assert '?' not in calls[-1]


def test_local_directory_scan_does_not_block_training_claim_gate(env):
    client, factory, root, _ = env
    loc = location(client, root / 'data'); finish_local()
    make_existing(root / 'data/sample')
    scan, _ = start_scan(client, [loc])
    with factory() as db:
        assert not store.node_dataset_file_work(db, 'local')
    finish_local()
    assert candidates(client, scan['id'])['counts']['ready'] == 1


@pytest.mark.parametrize('phase', ['export', 'import', 'remove'])
def test_heavy_executor_still_requires_workspace_lock(env, monkeypatch, phase):
    _, _, root, agent = env
    calls = []
    def api(_config, path, method='GET', payload=None, **kwargs):
        calls.append((path, payload))
        return {'ok': True}
    monkeypatch.setattr(agent, 'dataset_api', api)
    cfg = config(root, 'node')
    lock = agent.acquire_training_lock(cfg)
    try:
        agent.run_dataset_operation(cfg, {'id': 'a' * 32, 'token': 'b' * 32,
                                         'phase': phase, 'progress_run_id': 'c' * 32}, threading.Event())
        assert any((body.get('progress') or {}).get('stage') == 'waiting_lock' for _, body in calls)
        assert not any(path.endswith('/finish') for path, _ in calls)
        assert not list(root.glob('bundle*'))
    finally:
        agent.release_training_lock(lock)
