"""Regression coverage for catalogue freshness, idempotency and cache retention."""
import base64
from datetime import timedelta
import hashlib
import io
import threading
from pathlib import Path

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


@pytest.mark.parametrize('remote', [False, True])
def test_completed_registration_reclaims_centre_snapshot_without_touching_source(env, remote):
    from test_dataset_storage import make_existing, node, location, pump, operation
    client, factory, root, agent = env
    parent = root / 'datasets'
    node_id = node(factory, parent) if remote else 'local'
    loc = location(client, parent, node_id)
    pump(client, factory, parent, agent, node_id) if remote else finish_local()
    source = parent / 'sample'
    make_existing(source)
    response = client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'name': 'sample'})
    assert response.status_code == 202
    identity = response.json()['id']
    pump(client, factory, parent, agent, node_id) if remote else finish_local()
    assert operation(client, identity)['status'] == 'completed'
    assert not store.job_dir(identity).exists()
    assert (source / 'data.yaml').is_file() and (source / 'images/train/a.png').is_file()


def test_cancelled_local_export_waits_for_writer_before_reclaim(env, monkeypatch):
    from test_dataset_storage import make_existing, location, operation
    client, factory, root, _ = env
    parent = root / 'datasets'
    loc = location(client, parent)
    finish_local()
    make_existing(parent / 'sample')
    response = client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'name': 'sample'})
    assert response.status_code == 202
    identity = response.json()['id']
    packed, release = threading.Event(), threading.Event()
    original = store.files.pack
    def hold_completed_pack(*args, **kwargs):
        result = original(*args, **kwargs)
        packed.set()
        if not release.wait(10):
            raise TimeoutError('test writer did not resume')
        return result
    monkeypatch.setattr(store.files, 'pack', hold_completed_pack)
    executor = threading.Thread(target=store.worker.tick)
    executor.start()
    try:
        assert packed.wait(5)
        assert (store.job_dir(identity) / 'bundle.tar').is_file()
        cancelled = client.post(f'/api/dataset-storage/operations/{identity}/cancel')
        assert cancelled.status_code == 200
        assert operation(client, identity)['status'] == 'cancelled'
        with factory() as db:
            store.DatasetWorker().cleanup(db)
        assert (store.job_dir(identity) / 'bundle.tar').is_file()
    finally:
        release.set()
        executor.join(timeout=10)
    assert not executor.is_alive()
    assert not store.job_dir(identity).exists()
    assert (parent / 'sample/data.yaml').is_file()


def test_cancelled_refresh_lane_does_not_race_main_worker_cleanup(env, monkeypatch):
    from test_dataset_storage import operation
    client, factory, root, _ = env
    entry, source, _node_id = setup_entry(env, False)
    response = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh")
    assert response.status_code == 202
    identity = response.json()['id']
    packed, release = threading.Event(), threading.Event()
    original = store.files.pack
    def hold_completed_pack(*args, **kwargs):
        result = original(*args, **kwargs)
        packed.set()
        if not release.wait(10):
            raise TimeoutError('test refresh did not resume')
        return result
    monkeypatch.setattr(store.files, 'pack', hold_completed_pack)
    executor = threading.Thread(target=store.worker.tick_refresh)
    executor.start()
    try:
        assert packed.wait(5)
        assert client.post(f'/api/dataset-storage/operations/{identity}/cancel').status_code == 200
        store.worker.tick()  # Main lane may clean while refresh is still reading.
        assert (store.job_dir(identity) / 'bundle.tar').is_file()
    finally:
        release.set()
        executor.join(timeout=10)
    assert not executor.is_alive()
    assert operation(client, identity)['status'] == 'cancelled'
    assert not store.job_dir(identity).exists()
    assert (source / 'data.yaml').is_file()


def test_cleanup_reclaims_historical_terminal_snapshots_without_schema_migration(env):
    client, factory, root, _ = env
    identities = [f'{index:032x}' for index in range(1, 5)]
    with factory() as db:
        for identity, status in zip(identities, ('completed', 'cancelled', 'failed', 'cancelled')):
            db.add(DatasetOperation(id=identity, node_id='local', kind='register', status=status,
                                    lease_until=utcnow() + timedelta(minutes=5) if identity == identities[3] else None,
                                    payload={'phase': 'validate'}, result={}))
            directory = store.job_dir(identity)
            directory.mkdir(parents=True)
            (directory / 'bundle.tar').write_bytes(b'historical snapshot')
        db.commit()
        store.DatasetWorker().cleanup(db)
        assert [db.get(DatasetOperation, identity).status for identity in identities] == [
            'completed', 'cancelled', 'failed', 'cancelled']
    assert not store.job_dir(identities[0]).exists()
    assert not store.job_dir(identities[1]).exists()
    assert store.job_dir(identities[2]).exists()
    assert store.job_dir(identities[3]).exists()  # Old executor may still run.
    with factory() as db:
        db.get(DatasetOperation, identities[3]).lease_until = utcnow() - timedelta(seconds=1)
        db.commit()
        store.DatasetWorker().cleanup(db)
    assert not store.job_dir(identities[3]).exists()


def test_terminal_snapshot_cleanup_retries_detached_directory(env, monkeypatch):
    client, factory, root, _ = env
    identity = 'f' * 32
    with factory() as db:
        db.add(DatasetOperation(id=identity, node_id='local', kind='register',
                                status='completed', payload={'phase': 'validate'}, result={}))
        db.commit()
    directory = store.job_dir(identity)
    directory.mkdir(parents=True)
    (directory / 'bundle.tar').write_bytes(b'old snapshot')
    original = store.shutil.rmtree
    failed = []
    def transient_failure(path, *args, **kwargs):
        if Path(path).name.startswith('.reclaimed-') and not failed:
            failed.append(True)
            raise PermissionError('temporary cleanup lock')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(store.shutil, 'rmtree', transient_failure)
    with factory() as db:
        store.DatasetWorker().cleanup(db)
        assert db.get(DatasetOperation, identity).status == 'completed'
    assert failed and not directory.exists()
    assert list(store.STORE_ROOT.glob('.reclaimed-*'))
    with factory() as db:
        store.DatasetWorker().cleanup(db)
    assert not list(store.STORE_ROOT.glob('.reclaimed-*'))


def test_cancelled_remote_export_reclaims_partial_bundle_and_rejects_late_chunks(env):
    from test_dataset_storage import location, node, pump, operation
    client, factory, root, agent = env
    parent = root / 'remote'
    node_id = node(factory, parent)
    loc = location(client, parent, node_id)
    pump(client, factory, parent, agent, node_id)
    identity = 'd' * 32
    token = 'e' * 32
    with factory() as db:
        db.add(DatasetOperation(id=identity, node_id=node_id, kind='register', status='running',
                                lease=token, payload={'phase': 'export', 'snapshot_lock_v1': True, 'source_node': node_id,
                                                      'source_path': str(parent / 'sample'), 'location_id': loc,
                                                      'name': 'sample', 'ownership': 'external'}, result={}))
        db.commit()
    url = f'/api/dataset-storage/agent/{node_id}/{identity}/chunk'
    chunk = {'token': token, 'offset': 0, 'data': base64.b64encode(b'partial archive').decode()}
    assert client.post(url, json=chunk).status_code == 200
    assert (store.job_dir(identity) / 'bundle.part').read_bytes() == b'partial archive'
    assert client.post(f'/api/dataset-storage/operations/{identity}/cancel').status_code == 200
    with factory() as db:
        store.DatasetWorker().cleanup(db)
    assert operation(client, identity)['status'] == 'cancelled'
    assert not store.job_dir(identity).exists()
    assert client.post(url, json={**chunk, 'offset': 0}).status_code == 409
    assert not store.job_dir(identity).exists()


def test_cancelled_remote_finish_keeps_archive_until_hash_reader_exits(env, monkeypatch):
    from test_dataset_storage import location, node, pump
    client, factory, root, agent = env
    parent = root / 'remote'
    node_id = node(factory, parent)
    loc = location(client, parent, node_id)
    pump(client, factory, parent, agent, node_id)
    identity = 'c' * 32
    token = 'b' * 32
    archive_bytes = b'remote snapshot being checked'
    with factory() as db:
        db.add(DatasetOperation(id=identity, node_id=node_id, kind='register', status='running',
                                lease=token, payload={'phase': 'export', 'snapshot_lock_v1': True, 'source_node': node_id,
                                                      'source_path': str(parent / 'sample'), 'location_id': loc,
                                                      'name': 'sample', 'ownership': 'external'}, result={}))
        db.commit()
    directory = store.job_dir(identity)
    directory.mkdir()
    (directory / 'bundle.part').write_bytes(archive_bytes)
    hashing, release = threading.Event(), threading.Event()
    digest = store.files.digest
    def hold_hash(path, *args, **kwargs):
        if path == directory / 'bundle.part':
            hashing.set()
            if not release.wait(10):
                raise TimeoutError('test hash reader did not resume')
        return digest(path, *args, **kwargs)
    monkeypatch.setattr(store.files, 'digest', hold_hash)
    responses = []
    def finish():
        responses.append(client.post(f'/api/dataset-storage/agent/{node_id}/{identity}/finish', json={
            'token': token, 'phase': 'export', 'result': {'archive_size': len(archive_bytes),
            'archive_sha256': hashlib.sha256(archive_bytes).hexdigest()}}))
    reader = threading.Thread(target=finish)
    reader.start()
    try:
        assert hashing.wait(5)
        assert client.post(f'/api/dataset-storage/operations/{identity}/cancel').status_code == 200
        with factory() as db:
            store.DatasetWorker().cleanup(db)
        assert (directory / 'bundle.part').is_file()
    finally:
        release.set()
        reader.join(timeout=10)
    assert not reader.is_alive()
    assert responses and responses[0].status_code == 409
    assert not directory.exists()


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
