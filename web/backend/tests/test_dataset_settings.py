"""Settings persist safely and account for disposable files without reading data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import os
import threading

import pytest

from app import dataset_policy as policy, dataset_settings as settings
from app.models import ComputeNode, DatasetEntry, DatasetOperation, utcnow
from test_dataset_storage import env, store, location, finish_local, upload, operation


def test_policy_updates_persist_and_reject_lost_updates(env):
    client, _, root, _ = env
    before = client.get('/api/dataset-storage/settings').json()
    assert before['revision'] == 0 and before['policy_source'] == 'environment'
    response = client.patch('/api/dataset-storage/settings', json={
        'revision': 0, 'policy': {'storage_quota_bytes': 123456, 'failed_retention_seconds': 0}})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['revision'] == 1 and result['policy_source'] == 'instance'
    assert result['policy']['storage_quota_bytes'] == 123456
    assert store.dataset_policy()['failed_retention_seconds'] == 0
    stored = json.loads((root / 'transfer/.settings.json').read_text())
    assert stored['version'] == 1 and stored['revision'] == 1
    assert client.patch('/api/dataset-storage/settings', json={
        'revision': 0, 'policy': {'storage_quota_bytes': 1}}).status_code == 409
    with policy._cache_lock:
        policy._cache.clear()
    assert client.get('/api/dataset-storage/settings').json()['policy']['storage_quota_bytes'] == 123456


@pytest.mark.parametrize('values', [
    {'storage_quota_bytes': -1}, {'refresh_io_pressure_pct': 101},
    {'refresh_read_bytes_per_second': 1024**4 + 1}, {'failed_retention_seconds': 3650 * 86400 + 1},
    {'refresh_min_free_bytes': 2**53}, {'storage_quota_bytes': True},
    {'storage_quota_bytes': '1'}, {'storage_quota_bytes': 1.5}, {'unknown': 1},
])
def test_policy_api_validates_strict_values(env, values):
    client, _, _, _ = env
    response = client.patch('/api/dataset-storage/settings', json={'revision': 0, 'policy': values})
    assert response.status_code == 422
    assert not (store.STORE_ROOT / '.settings.json').exists()


def test_file_lock_serializes_competing_writers(tmp_path):
    barrier = threading.Barrier(2)
    defaults = store.dataset_policy_defaults()
    def update(value):
        barrier.wait()
        try:
            return policy.update(tmp_path, defaults, 0, {'storage_quota_bytes': value})['revision']
        except FileExistsError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(update, value) for value in (10, 20)]
        assert sorted(str(future.result()) for future in futures) == ['1', 'conflict']


def test_malformed_config_and_special_files_fail_without_rewriting(tmp_path):
    defaults = store.dataset_policy_defaults()
    path = tmp_path / '.settings.json'
    path.write_text('{broken')
    assert policy.read(tmp_path, defaults)['policy_error']
    with pytest.raises(ValueError):
        policy.update(tmp_path, defaults, 0, {'storage_quota_bytes': 1})
    assert path.read_text() == '{broken'
    path.unlink()
    external = tmp_path / 'external.json'
    external.write_text('{}')
    path.symlink_to(external)
    assert policy.read(tmp_path, defaults)['policy_error']
    path.unlink()
    os.mkfifo(path)
    assert policy.read(tmp_path, defaults)['policy_error']


def test_sampler_counts_bundles_unpack_uploads_and_ignores_symlinks(tmp_path):
    transfer, uploads = tmp_path / 'transfer', tmp_path / 'uploads'
    identity = 'a' * 32
    job = transfer / identity
    (job / 'verify/sub').mkdir(parents=True)
    (job / 'bundle.tar').write_bytes(b'archive')
    (job / 'verify/sub/image.png').write_bytes(b'unpacked')
    staged = uploads / identity
    staged.mkdir(parents=True)
    (staged / 'file').write_bytes(b'upload')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'huge').write_bytes(b'x' * 200)
    (job / 'linked').symlink_to(outside, target_is_directory=True)
    (transfer / ('b' * 32)).symlink_to(outside, target_is_directory=True)
    result = settings.sample_roots(transfer, uploads)
    assert result['status'] == 'partial'  # Link targets are deliberately not followed.
    assert result['transfer_bytes'] == len(b'archiveunpacked')
    assert result['upload_staging_bytes'] == 6
    assert result['quota_used_bytes'] == 7
    assert result['operations'][0]['bundle_bytes'] == 7
    assert result['operations'][0]['transfer_bytes'] == 15


def test_quota_meter_matches_executor_glob_including_quarantine(tmp_path):
    transfer, uploads = tmp_path / 'transfer', tmp_path / 'uploads'
    for name, amount in [('a' * 32, 10), ('.reclaimed-' + 'b' * 32, 20), ('legacy-unexpected', 30)]:
        job = transfer / name
        job.mkdir(parents=True)
        (job / 'bundle.part').write_bytes(b'x' * amount)
    result = settings.sample_roots(transfer, uploads)
    expected = sum(path.stat().st_size for path in transfer.glob('*/bundle*'))
    assert result['status'] == 'complete'
    assert result['quota_used_bytes'] == expected == 60


def test_sampler_is_bounded_and_marks_partial(tmp_path, monkeypatch):
    transfer, uploads = tmp_path / 'transfer', tmp_path / 'uploads'
    job = transfer / ('a' * 32)
    job.mkdir(parents=True)
    for index in range(10):
        (job / str(index)).write_bytes(b'xx')
    monkeypatch.setattr(settings, 'SAMPLE_ENTRIES', 3)
    result = settings.sample_roots(transfer, uploads)
    assert result['status'] == 'partial' and result['limit_reached']
    assert result['entries_scanned'] == 3
    assert result['total_bytes'] < 20


def test_sampler_only_runs_one_thread_and_first_get_does_not_wait(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def sample(*_args):
        entered.set()
        release.wait(3)
        return {'status': 'complete', 'sampled_at': 'now'}
    monkeypatch.setattr(settings, 'sample_roots', sample)
    sampler = settings.StorageSampler()
    first = sampler.get(tmp_path / 'a', tmp_path / 'b')
    assert entered.wait(1)
    assert first['status'] == 'pending' and first['total_bytes'] is None
    thread = sampler.thread
    assert sampler.get(tmp_path / 'a', tmp_path / 'b', refresh=True)['sampling']
    assert sampler.thread is thread
    release.set()
    thread.join(2)
    assert sampler.get(tmp_path / 'a', tmp_path / 'b')['status'] == 'complete'


def test_get_uses_catalogue_not_dataset_paths_and_excludes_local_node(env, monkeypatch):
    client, factory, root, _ = env
    with factory() as db:
        db.add_all([
            ComputeNode(id='local', name='本机', kind='local'),
            ComputeNode(id='remote', name='远程', kind='agent', last_seen_at=utcnow(),
                        snapshot={'training': {'dataset_roots': ['relative-data']}}),
            DatasetEntry(id='known', location_id='old', node_id='local', name='historical',
                         path='/unavailable/dataset', details={'total_bytes': 42}),
            DatasetEntry(id='unknown', location_id='old', node_id='remote', name='historical remote',
                         path='/unavailable/remote', details={}),
            DatasetOperation(id='f' * 32, node_id='local', kind='check', status='pending',
                             payload={'phase': 'check', 'path': '/unavailable/dataset'})])
        db.commit()
    monkeypatch.setattr(store, 'local_mutation_blocked_by_refresh', lambda *_args: pytest.fail('GET probed original paths'))
    data = client.get('/api/dataset-storage/settings').json()
    assert [node['id'] for node in data['remote_nodes']] == ['remote']
    assert data['remote_nodes'][0]['online']
    assert data['storage']['registered_bytes']['local'] == {'bytes': 42, 'count': 1, 'unknown_count': 0}
    assert data['storage']['registered_bytes']['remote'] == {'bytes': 0, 'count': 1, 'unknown_count': 1}
    assert data['operations']['active_count'] == 1


def test_policy_applies_to_real_upload_quota_and_failed_retention(env):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets')
    finish_local()
    response = client.patch('/api/dataset-storage/settings', json={'revision': 0, 'policy': {'storage_quota_bytes': 1}})
    assert response.status_code == 200
    identity = upload(client, loc)
    finish_local()
    assert operation(client, identity)['status'] == 'failed'
    assert '中转配额不足' in operation(client, identity)['error']
    with factory() as db:
        op = db.get(DatasetOperation, identity)
        op.updated_at = utcnow() - timedelta(days=10)
        op.lease_until = utcnow() - timedelta(days=1)
        db.commit()
        assert store.failed_snapshot_expired(op)
        client.patch('/api/dataset-storage/settings', json={'revision': 1, 'policy': {'failed_retention_seconds': 0}})
        assert not store.failed_snapshot_expired(op)


def test_runtime_budget_observes_disabled_limit_while_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(store, 'STORE_ROOT', tmp_path)
    policy.update(tmp_path, store.dataset_policy_defaults(), 0,
                  {'refresh_read_bytes_per_second': 1, 'refresh_min_free_bytes': 0, 'refresh_io_pressure_pct': 0})
    checks = []
    def check():
        checks.append(True)
    check.progress = lambda _value: None
    budget = store.LocalRefreshBudget(check)
    monkeypatch.setattr(budget, 'training_now', lambda: True)
    def change_policy(_seconds):
        policy.update(tmp_path, store.dataset_policy_defaults(), 1, {'refresh_read_bytes_per_second': 0})
    monkeypatch.setattr(store.time, 'sleep', change_policy)
    budget.throttle(1024)
    assert checks and budget.next_read_at == 0


def test_cleanup_reasons_distinguish_retention_and_pending(env):
    _, factory, _, _ = env
    with factory() as db:
        op = DatasetOperation(id='c' * 32, node_id='local', kind='upload', status='completed',
                              payload={'snapshot_lock_v1': True})
        db.add(op)
        db.commit()
        state, _ = settings.cleanup_state(op, {'upload_staging_bytes': 10}, store)
        assert state == 'reserved'
        state, reason = settings.cleanup_state(op, {'upload_staging_bytes': 0}, store)
        assert state == 'pending' and '等待' in reason
