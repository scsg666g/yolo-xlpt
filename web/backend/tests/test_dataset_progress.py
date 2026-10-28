"""Real-file simulations of measured progress and old/new Agent interoperability."""
import threading
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from app import dataset_progress as progress
from app.models import DatasetOperation
from test_dataset_storage import (env, entries, finish_local, images, location,
                                  node, operation, pump, remote_tick, store, upload)


def test_file_progress_preserves_snapshot_and_roundtrip(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'large.bin').write_bytes(b'x' * (2 * 1024 * 1024 + 7))
    (source / 'empty.txt').touch()
    reports = []
    def check():
        pass
    check.progress = reports.append
    archive = tmp_path / 'bundle.tar'
    manifest = store.files.pack(source, archive, check)
    store.files.unpack(archive, tmp_path / 'target', manifest, check)
    assert (tmp_path / 'target/large.bin').read_bytes() == (source / 'large.bin').read_bytes()
    assert manifest['fingerprint'] == store.files.scan(source)['fingerprint']
    for stage in ('pack', 'archive_hash', 'unpack'):
        samples = [r for r in reports if r['stage'] == stage]
        assert samples[0]['processed'] == 0
        assert samples[-1]['processed'] == samples[-1]['total']
        assert all(a['processed'] <= b['processed'] for a, b in zip(samples, samples[1:]))


def test_local_running_and_validation_reported_before_completion(env, monkeypatch):
    client, factory, root, _ = env
    loc = location(client, root / 'local')
    finish_local()
    identity = upload(client, loc)
    captured = []
    original = store.record_work_progress
    def record(op, value, run_id, executor):
        result = original(op, value, run_id, executor)
        captured.append((op.status, value['stage'], executor))
        return result
    monkeypatch.setattr(store, 'record_work_progress', record)
    finish_local()
    op = operation(client, identity)
    assert op['status'] == 'completed', op
    assert op['progress']['executor'] == 'local'
    assert op['progress']['processed'] == op['progress']['total']
    assert op['heartbeat_at'] and op['started_at']
    assert any(s == 'running' and stage == 'content_validate' for s, stage, _ in captured)
    assert 'run_id' not in op['progress']
    assert entries(client)


def test_old_history_and_queue_are_readable(env):
    client, factory, root, _ = env
    with factory() as db:
        first = store.new_operation(db, 'check', 'local', {'phase': 'check', 'path': str(root)})
        db.flush()
        second = store.new_operation(db, 'check', 'local', {'phase': 'check', 'path': str(root)})
        db.commit()
        one, two = first.id, second.id
    assert operation(client, one)['progress'] is None
    assert operation(client, two)['queue_ahead'] == 1


def test_progress_rejects_stale_retry_and_out_of_order_reports():
    run = uuid.uuid4().hex
    op = DatasetOperation(payload={'phase': 'export', 'progress_run_id': run})
    assert progress.record_progress(op, {'stage': 'pack', 'processed': 5, 'total': 10, 'sequence': 2}, run, 'local')
    assert not progress.record_progress(op, {'stage': 'scan', 'sequence': 1}, run, 'local')
    assert not progress.record_progress(op, {'stage': 'scan', 'sequence': 3}, uuid.uuid4().hex, 'local')
    assert op.payload['progress']['processed'] == 5
    assert progress.record_progress(op, {'stage': 'archive_hash', 'sequence': 3}, run, 'local')
    assert op.payload['progress_steps'][0]['stage'] == 'pack'
    op.payload = {**op.payload, 'progress_run_id': uuid.uuid4().hex}
    assert not progress.record_progress(op, {'stage': 'pack'}, run, 'local')


def test_resource_pause_does_not_mark_reading_or_pause_as_completed():
    run = uuid.uuid4().hex
    op = DatasetOperation(payload={'phase': 'export', 'progress_run_id': run})
    assert progress.record_progress(op, {'stage': 'pack', 'processed': 5, 'total': 10}, run, 'local')
    assert progress.record_progress(op, {'stage': 'paused_resource',
                                         'resource_reason': 'disk_pressure'}, run, 'local')
    assert op.payload['progress_steps'] == []
    assert progress.record_progress(op, {'stage': 'pack', 'processed': 5, 'total': 10}, run, 'local')
    assert op.payload['progress_steps'] == []


@pytest.mark.parametrize('value', [
    {'stage': 'invalid'}, {'stage': 'pack', 'processed': -1},
    {'stage': 'pack', 'processed': float('nan')},
    {'stage': 'pack', 'processed': 4, 'total': 3},
    {'stage': 'pack', 'unit': 'unbounded'},
])
def test_invalid_progress_cannot_pollute_history(value):
    with pytest.raises(ValueError):
        progress.clean_progress(value)


def test_progress_reports_are_throttled_and_finish_immediately(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(progress.time, 'monotonic', lambda: clock[0])
    reports = []
    report = progress.progress_reporter(reports.append)
    for count in range(100):
        report({'stage': 'pack', 'processed': count, 'total': 100})
    assert len(reports) == 1
    report({'stage': 'pack', 'processed': 100, 'total': 100})
    assert len(reports) == 2
    assert reports[-1]['sequence'] == 2
    report({'stage': 'archive_hash'})
    assert len(reports) == 3


def test_speed_and_last_progress_are_independent_of_heartbeat(monkeypatch):
    moment = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment[0]
    monkeypatch.setattr(progress, 'datetime', Clock)
    run = uuid.uuid4().hex
    op = DatasetOperation(payload={'phase': 'export', 'progress_run_id': run})
    progress.record_progress(op, {'stage': 'pack', 'processed': 0, 'total': 100}, run, 'local')
    moment[0] += timedelta(seconds=10)
    progress.record_progress(op, {'stage': 'pack', 'processed': 50, 'total': 100}, run, 'local')
    first = op.payload['progress'].copy()
    assert first['rate'] == 5 and first['eta_seconds'] == 10
    moment[0] += timedelta(seconds=5)
    progress.record_progress(op, {'stage': 'pack', 'processed': 50, 'total': 100}, run, 'local')
    assert op.payload['progress']['changed_at'] == first['changed_at']
    assert op.payload['progress']['sampled_at'] != first['sampled_at']
    progress.record_progress(op, {'stage': 'scan'}, run, 'local')
    assert op.payload['progress']['rate'] is None
    assert op.payload['progress']['eta_seconds'] is None


def test_validation_can_be_cancelled_without_publishing(env, monkeypatch):
    client, factory, root, _ = env
    loc = location(client, root / 'local')
    finish_local()
    identity = upload(client, loc)
    from app import dataset_uploads
    validate = dataset_uploads.validate_contents
    def cancel_then_validate(*args, **kwargs):
        result = client.post(f'/api/dataset-storage/operations/{identity}/cancel')
        assert result.status_code == 200, result.text
        return validate(*args, **kwargs)
    monkeypatch.setattr(dataset_uploads, 'validate_contents', cancel_then_validate)
    finish_local()
    assert operation(client, identity)['status'] == 'cancelled'
    assert not entries(client)
    assert not (store.job_dir(identity) / 'validate').exists()


def test_corrupt_file_never_reports_verified_completion(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'file.txt').write_bytes(b'example')
    manifest = store.files.pack(source, tmp_path / 'bundle.tar')
    manifest['files'][0]['sha256'] = '0' * 64
    reports = []
    def check():
        pass
    check.progress = reports.append
    with pytest.raises(ValueError, match='SHA-256'):
        store.files.unpack(tmp_path / 'bundle.tar', tmp_path / 'target', manifest, check)
    samples = [r for r in reports if r['stage'] == 'unpack']
    assert all(r['processed'] < r['total'] for r in samples)


def test_remote_progress_uses_current_run_and_old_agent_can_finish(env, monkeypatch):
    client, factory, root, agent = env
    remote = root / 'remote'
    identity = node(factory, remote)
    loc = location(client, remote, identity)
    pump(client, factory, remote, agent, identity)
    upload_id = upload(client, loc)
    reports = []
    api = agent.dataset_api
    def capture(config, route, method='GET', payload=None, **kwargs):
        if route.endswith('/heartbeat') and payload and payload.get('progress'):
            reports.append(payload)
        return api(config, route, method, payload, **kwargs)
    monkeypatch.setattr(agent, 'dataset_api', capture)
    pump(client, factory, remote, agent, identity)
    assert operation(client, upload_id)['status'] == 'completed'
    assert any(r['progress']['stage'] == 'transfer' for r in reports)
    assert any(r['progress']['stage'] == 'unpack' for r in reports)
    assert all(r['progress_run_id'] for r in reports)

    # A previous centre does not send progress_run_id. New Agent keeps the old protocol.
    old_reports = len(reports)
    work = {'id': uuid.uuid4().hex, 'token': uuid.uuid4().hex, 'phase': 'check', 'path': str(remote)}
    cfg = {'node_id': identity, 'dataset_roots': [str(remote)], 'training_cwd': str(root),
           'training_state_dir': str(root / 'legacy-state')}
    monkeypatch.setattr(agent, 'dataset_api', lambda config, route, *args, **kwargs: reports.append(route) or {})
    agent.run_dataset_operation(cfg, work, threading.Event())
    assert all(isinstance(route, str) for route in reports[old_reports:])
