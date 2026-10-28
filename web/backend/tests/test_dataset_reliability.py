"""Regression coverage for catalogue freshness, idempotency and cache retention."""
from datetime import timedelta
import io
import threading

import pytest
from PIL import Image

from app.models import DatasetEntry, DatasetOperation, TrainingRun, TrainingTask, utcnow
from test_dataset_storage import env, finish_local, store
from test_dataset_refresh import setup_entry


def revision(client):
    return client.get('/api/dataset-storage/operations/summary').json()['revision']


def test_summary_tracks_reference_fields_without_metric_churn(env):
    client, factory, root, _ = env
    first = revision(client)
    with factory() as db:
        task = TrainingTask(id='task', name='task', model_path='model.pt', data_path='data.yaml')
        db.add(task)
        db.flush()
        db.add(TrainingRun(id='run', task_id=task.id, attempt_number=1, model_path='model.pt', data_path='data.yaml'))
        db.commit()
    added = revision(client)
    assert added != first
    with factory() as db:
        task = db.get(TrainingTask, 'task')
        task.progress = 50
        task.latest_metrics = {'loss': 1.5}
        db.commit()
    assert revision(client) == added
    with factory() as db:
        db.get(TrainingTask, 'task').status = 'running'
        db.commit()
    running = revision(client)
    assert running != added
    with factory() as db:
        db.delete(db.get(TrainingRun, 'run'))
        db.commit()
    assert revision(client) != running


@pytest.mark.parametrize('remote', [False, True])
def test_refresh_request_retry_returns_same_operation(env, remote):
    client, factory, root, agent = env
    entry, _, _ = setup_entry(env, remote)
    url = f"/api/dataset-storage/datasets/{entry['id']}/refresh"
    first = client.post(url, json={'request_id': 'retry-safe-1'})
    again = client.post(url, json={'request_id': 'retry-safe-1'})
    assert first.status_code == again.status_code == 202
    assert first.json()['id'] == again.json()['id']
    assert client.post('/api/dataset-storage/datasets/another/refresh', json={'request_id': 'retry-safe-1'}).status_code == 409
    assert client.post(url, json={'request_id': '../../bad'}).status_code == 422


@pytest.mark.parametrize('remote', [False, True])
def test_registration_rejects_duplicate_image_stems(env, remote):
    from test_dataset_storage import make_existing, node, location, pump
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    make_existing(folder / 'duplicate')
    image = io.BytesIO()
    Image.new('RGB', (8, 8), 'blue').save(image, 'JPEG')
    (folder / 'duplicate/images/train/a.jpg').write_bytes(image.getvalue())
    response = client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'name': 'duplicate'})
    assert response.status_code == 202
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    result = client.get('/api/dataset-storage/operations/' + response.json()['id']).json()
    assert result['status'] == 'failed'
    assert '同名但不同扩展名' in result['error']
    with factory() as db:
        assert db.query(DatasetEntry).count() == 0


def test_pack_reuses_planning_scan_but_verifies_source_again(env, monkeypatch):
    from test_dataset_storage import make_existing
    client, _, root, _ = env
    source = root / 'source'
    make_existing(source)
    calls = []
    original = store.files.scan
    def scan(*args, **kwargs):
        calls.append(kwargs.get('stage', args[2] if len(args) > 2 else 'scan'))
        return original(*args, **kwargs)
    monkeypatch.setattr(store.files, 'scan', scan)
    manifest = store.files.scan(source)
    result = store.files.pack(source, root / 'bundle.tar', manifest=manifest)
    assert calls == ['scan', 'source_verify']
    assert result['fingerprint'] == manifest['fingerprint']


def test_cleanup_retains_active_and_committed_but_restarts_expired_snapshots(env):
    client, factory, root, _ = env
    old = utcnow() - timedelta(days=9)
    rows = [
        ('expired', {'phase': 'validate', 'source_node': 'remote', 'source_path': str(root), 'request_id': 'stable'}, None),
        ('committed', {'phase': 'remove', 'published': True, 'commit_started': True}, None),
        ('leased', {'phase': 'export', 'source_node': 'remote'}, utcnow() + timedelta(hours=1)),
    ]
    identities = {}
    with factory() as db:
        for index, (name, payload, lease_until) in enumerate(rows, 1):
            identity = f'{index:032x}'
            identities[name] = identity
            db.add(DatasetOperation(id=identity, node_id='local', kind='register', status='failed',
                                    lease='a' * 32, lease_until=lease_until, payload=payload, result={}, updated_at=old))
            directory = store.job_dir(identity)
            directory.mkdir(parents=True)
            (directory / 'bundle.tar').write_bytes(b'large cache')
        db.commit()
        store.DatasetWorker().cleanup(db)
    assert not store.job_dir(identities['expired']).exists()
    assert store.job_dir(identities['committed']).exists()
    assert store.job_dir(identities['leased']).exists()
    retry = client.post('/api/dataset-storage/operations/' + identities['expired'] + '/retry')
    assert retry.status_code == 200
    assert retry.json()['phase'] == 'export' and retry.json()['node_id'] == 'remote'
    with factory() as db:
        op = db.get(DatasetOperation, identities['expired'])
        assert op.lease != 'a' * 32
        assert op.payload['request_id'] == 'stable'
        assert not op.payload.get('snapshot_reclaimed')


def test_agent_reconcile_is_node_scoped_and_protects_live_or_committed_receipts(env):
    client, factory, root, _ = env
    old = utcnow() - timedelta(days=9)
    ids = [f'{index:032x}' for index in range(1, 7)]
    with factory() as db:
        for identity, status, payload, token in [
            (ids[0], 'running', {'phase': 'export', 'source_node': 'node'}, 'a' * 32),
            (ids[1], 'failed', {'phase': 'export', 'source_node': 'node'}, 'a' * 32),
            (ids[2], 'failed', {'phase': 'import', 'source_node': 'node', 'commit_started': True}, 'a' * 32),
            (ids[3], 'completed', {'source_node': 'node'}, 'a' * 32),
            (ids[4], 'completed', {'source_node': 'other'}, 'a' * 32),
        ]:
            db.add(DatasetOperation(id=identity, node_id=payload['source_node'], kind='register', status=status,
                                    lease=token, payload=payload, result={}, updated_at=old))
        db.commit()
    receipts = [{'id': identity, 'token': 'a' * 32} for identity in ids]
    response = client.post('/api/dataset-storage/agent/node/reconcile', json={'receipts': receipts})
    assert response.status_code == 200
    assert {item['id'] for item in response.json()['reclaimable']} == {ids[1], ids[3], ids[5]}
    obsolete = client.post('/api/dataset-storage/agent/node/reconcile', json={'receipts': [{'id': ids[0], 'token': 'b' * 32}]})
    assert obsolete.json()['reclaimable'] == [{'id': ids[0], 'token': 'b' * 32}]
    assert client.post('/api/dataset-storage/agent/node/reconcile', json={'receipts': [{'id': '../bad', 'token': 'a'*32}]}).status_code == 422


def test_worker_health_reports_failure_and_recovery(env, monkeypatch):
    worker = store.DatasetWorker()
    assert worker.health_state()['status'] == 'stopped'
    done = threading.Event()
    calls = []
    def tick():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError('test worker failure')
        worker.last_tick_at = utcnow()
        worker.stop_event.set()
        done.set()
    monkeypatch.setattr(worker, 'tick', tick)
    worker.start()
    assert done.wait(4)
    worker.stop()
    state = worker.health_state()
    assert state['last_tick_at'] and state['last_error_at']
    assert state['last_error'] == 'test worker failure' and state['active_error'] is None
