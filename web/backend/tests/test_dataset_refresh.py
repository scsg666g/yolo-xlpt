"""Reload real dataset snapshots through local and legacy-compatible remote paths."""
import threading
from pathlib import Path

import pytest
import yaml

from app.models import DatasetEntry, DatasetOperation, TrainingTask
from test_dataset_storage import (env, entries, finish_local, images, location, make_existing,
                                  node, operation, pump, remote_tick, store)


def setup_entry(env, remote):
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    make_existing(folder / 'sample')
    result = client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'name': 'sample'})
    assert result.status_code == 202, result.text
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    entry = entries(client)[0]
    return entry, folder / 'sample', identity


def files_at(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


@pytest.mark.parametrize('remote', [False, True])
def test_refresh_updates_content_statistics_without_replacing_identity_or_references(env, remote):
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, remote)
    with factory() as db:
        db.add(TrainingTask(id='referenced', name='历史引用', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='completed',
                            execution_node_id=identity if remote else None))
        old = db.get(DatasetEntry, entry['id'])
        old.details = {**old.details, 'history': [{'node_id': identity, 'path': str(folder), 'operation_id': 'old'}]}
        db.commit()
    (folder / 'images/train/new.png').write_bytes(images()['images/train/a.png'])
    (folder / 'labels/train/a.txt').write_text('')
    cfg = yaml.safe_load((folder / 'data.yaml').read_text())
    cfg['names'] = ['updated', 'new class']
    (folder / 'data.yaml').write_text(yaml.safe_dump(cfg))
    before = files_at(folder)
    response = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh")
    assert response.status_code == 202, response.text
    op_id = response.json()['id']
    pending = entries(client)[0]
    assert pending['stats'] == entry['stats']
    assert pending['blocking_operation_kind'] == 'refresh'
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").status_code == 409
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)}).status_code == 409
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    result = operation(client, op_id)
    assert result['status'] == 'completed', result
    current = entries(client)[0]
    assert len(entries(client)) == 1
    assert current['id'] == entry['id'] and current['created_at'] == entry['created_at']
    assert current['ownership'] == 'external' and current['history'][0]['operation_id'] == 'old'
    assert current['stats']['train_images'] == 2 and current['stats']['val_images'] == 1
    assert current['stats']['background_images'] == 2
    assert current['classes'] == ['updated', 'new class'] and len(current['warnings']) == 2
    assert current['total_bytes'] != entry['total_bytes'] and current['checked_at']
    assert current['references'][0]['id'] == 'referenced' and not current['can_delete']
    assert current['blocking_operation_id'] is None
    assert files_at(folder) == before
    assert client.get('/api/dataset-storage/operations/page', params={'kind': 'refresh'}).status_code == 200
    assert client.post('/api/dataset-storage/operations/history-removal-preview', json={
        'kind': 'refresh', 'before': '2099-01-01T00:00:00Z'}).status_code == 200


@pytest.mark.parametrize('remote', [False, True])
def test_failed_refresh_keeps_good_statistics_and_retry_exports_corrected_files(env, remote):
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, remote)
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)}).status_code == 200
    (folder / 'labels/train/a.txt').write_text('invalid label')
    result = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, result['id'])['status'] == 'failed'
    current = entries(client)[0]
    assert current['stats'] == entry['stats'] and current['checked_at'] == entry['checked_at']
    assert current['refresh_error'] and current['refresh_status'] == 'failed'
    assert not current['can_delete']
    old_token = None
    with factory() as db:
        old_token = db.get(DatasetOperation, result['id']).lease
    (folder / 'labels/train/a.txt').write_text('')
    fixed = files_at(folder)
    response = client.post(f"/api/dataset-storage/operations/{result['id']}/retry")
    assert response.status_code == 200
    assert response.json()['phase'] == 'export' and response.json()['node_id'] == identity
    with factory() as db:
        assert db.get(DatasetOperation, result['id']).lease != old_token
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, result['id'])['status'] == 'completed'
    current = entries(client)[0]
    assert current['stats']['background_images'] == 1 and current['ownership'] == 'managed'
    assert current['refresh_status'] == 'completed' and not current['refresh_error']
    assert files_at(folder) == fixed
    # Updated fingerprint must still allow the normal protected delete flow.
    deletion = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert deletion.status_code == 202, deletion.text
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, deletion.json()['id'])['status'] == 'completed'
    assert not folder.exists()


@pytest.mark.parametrize('remote', [False, True])
def test_cancel_and_old_entry_without_new_metadata_remain_supported(env, remote):
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, remote)
    with factory() as db:
        e = db.get(DatasetEntry, entry['id'])
        e.details = {k: v for k, v in e.details.items() if k != 'checked_at'}
        db.commit()
    assert entries(client)[0]['checked_at'] is None
    response = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh")
    op_id = response.json()['id']
    assert client.post(f'/api/dataset-storage/operations/{op_id}/cancel').status_code == 200
    assert entries(client)[0]['refresh_status'] == 'cancelled'
    assert entries(client)[0]['stats'] == entry['stats']
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").status_code == 202
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert entries(client)[0]['checked_at']
    assert client.post('/api/dataset-storage/datasets/missing/refresh').status_code == 404


@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('bad', ['missing', 'yaml_path', 'symlink'])
def test_invalid_or_missing_source_preserves_previous_catalogue(env, remote, bad):
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, remote)
    if bad == 'missing':
        folder.rename(folder.with_name('renamed'))
    elif bad == 'yaml_path':
        cfg = yaml.safe_load((folder / 'data.yaml').read_text())
        cfg['train'] = 'other/train'
        (folder / 'data.yaml').write_text(yaml.safe_dump(cfg))
    else:
        (folder / 'linked').symlink_to(root, target_is_directory=True)
    result = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, result['id'])['status'] == 'failed'
    current = entries(client)[0]
    assert current['stats'] == entry['stats'] and current['classes'] == entry['classes']
    assert current['checked_at'] == entry['checked_at']


@pytest.mark.parametrize('remote', [False, True])
def test_reload_is_read_only_during_training_without_changing_training(env, remote):
    from app.models import TrainingRun
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, remote)
    with factory() as db:
        task = TrainingTask(id='busy', name='training', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='running',
                            execution_node_id=identity if remote else None)
        db.add(task); db.flush()
        db.add(TrainingRun(id='busy-run', task_id=task.id, attempt_number=1, status='running',
                          execution_node_id=identity if remote else None,
                          model_path='model.pt', data_path=str(folder / 'data.yaml')))
        db.commit()
    reload = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    assert operation(client, reload['id']).get('waiting_training', False) == remote
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, reload['id'])['status'] == ('pending' if remote else 'completed')
    with factory() as db:
        assert db.get(TrainingRun, 'busy-run').status == 'running'
        assert db.get(TrainingTask, 'busy').status == 'running'
        db.get(TrainingRun, 'busy-run').status = 'completed'
        db.get(TrainingTask, 'busy').status = 'completed'
        db.commit()
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, reload['id'])['status'] == 'completed'
    assert entries(client)[0]['references'][0]['id'] == 'busy'


def test_local_reload_does_not_block_later_directory_checks(env):
    from app.models import TrainingRun
    client, factory, root, _ = env
    entry, folder, identity = setup_entry(env, False)
    with factory() as db:
        task = TrainingTask(id='busy-check', name='training', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='running')
        db.add(task); db.flush()
        db.add(TrainingRun(id='busy-check-run', task_id=task.id, attempt_number=1,
                          status='running', model_path='model.pt', data_path=task.data_path))
        db.commit()
    refresh = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    check = client.post(f"/api/dataset-storage/locations/{entry['location_id']}/check").json()
    finish_local()
    assert operation(client, refresh['id'])['status'] == 'completed'
    assert operation(client, check['id'])['status'] == 'completed'


def test_local_reload_lane_keeps_other_checks_running_and_guards_same_path(env, monkeypatch):
    from app.models import TrainingRun
    client, factory, root, _ = env
    entry, folder, _ = setup_entry(env, False)
    with factory() as db:
        task = TrainingTask(id='lane-training', name='training', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='running')
        db.add(task); db.flush()
        db.add(TrainingRun(id='lane-run', task_id=task.id, attempt_number=1,
                           status='running', model_path='model.pt', data_path=task.data_path))
        db.commit()
    entered, release = threading.Event(), threading.Event()
    original_scan = store.files.scan

    def slow_scan(source, check=None, stage='scan'):
        if Path(source) == folder and stage == 'scan' and not entered.is_set():
            entered.set()
            assert release.wait(5)
        return original_scan(source, check, stage)

    monkeypatch.setattr(store.files, 'scan', slow_scan)
    refresh = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    background = threading.Thread(target=store.worker.tick_refresh)
    background.start()
    try:
        assert entered.wait(5)
        with factory() as db:
            assert not store.node_dataset_file_work(db, 'local')
        parent_check = client.post(f"/api/dataset-storage/locations/{entry['location_id']}/check").json()
        exact_location = location(client, folder)
        store.worker.tick()
        assert operation(client, parent_check['id'])['status'] == 'completed'
        with factory() as db:
            exact_check = next(op for op in db.query(DatasetOperation).all()
                               if op.kind == 'check' and op.payload.get('location_id') == exact_location)
            exact_id = exact_check.id
        assert operation(client, exact_id)['waiting_path_lock']
        store.worker.tick()
        assert operation(client, exact_id)['status'] == 'pending'
    finally:
        release.set()
        background.join(timeout=8)
    assert not background.is_alive()
    finish_local()
    assert operation(client, refresh['id'])['status'] == 'completed'
    assert operation(client, exact_id)['status'] == 'completed'
    with factory() as db:
        assert db.get(TrainingRun, 'lane-run').status == 'running'


def test_local_reload_detects_changes_after_snapshot_validation(env, monkeypatch):
    client, _, _, _ = env
    entry, folder, _ = setup_entry(env, False)
    original_verify = store.verify_snapshot

    def change_source_during_validation(op, check=None):
        result = original_verify(op, check)
        (folder / 'labels/train/a.txt').write_text('0 0.4 0.4 0.2 0.2\n')
        return result

    monkeypatch.setattr(store, 'verify_snapshot', change_source_during_validation)
    refresh = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    finish_local()
    assert operation(client, refresh['id'])['status'] == 'failed'
    current = entries(client)[0]
    assert current['stats'] == entry['stats'] and current['checked_at'] == entry['checked_at']


def test_local_refresh_budget_pauses_on_io_pressure_only_during_training(monkeypatch):
    progress, heartbeats = [], []
    def check():
        heartbeats.append(True)
    check.progress = progress.append
    budget = store.LocalRefreshBudget(check)
    monkeypatch.setattr(budget, 'training_now', lambda: True)
    monkeypatch.setattr(store, 'DATASET_REFRESH_READ_BYTES_PER_SECOND', 0)
    monkeypatch.setattr(store, 'local_refresh_free_bytes', lambda: 1024**4)
    monkeypatch.setattr(store, 'DATASET_REFRESH_IO_PRESSURE_PCT', 25)
    values = iter([80.0, 0.0])
    monkeypatch.setattr(store, 'local_io_pressure_pct', lambda: next(values))
    monkeypatch.setattr(store.time, 'sleep', lambda _: setattr(budget, 'last_resource_probe', 0))
    budget.throttle(1024)
    assert progress == [{'stage': 'paused_resource', 'resource_reason': 'disk_pressure'}]
    assert heartbeats
    monkeypatch.setattr(budget, 'training_now', lambda: False)
    budget.throttle(1024)
    assert len(progress) == 1


def test_local_refresh_budget_pauses_when_disk_reserve_is_lost(monkeypatch):
    progress = []
    def check():
        pass
    check.progress = progress.append
    budget = store.LocalRefreshBudget(check)
    monkeypatch.setattr(budget, 'training_now', lambda: False)
    monkeypatch.setattr(store, 'DATASET_REFRESH_MIN_FREE_BYTES', 100)
    free = iter([50, 200])
    monkeypatch.setattr(store, 'local_refresh_free_bytes', lambda: next(free))
    monkeypatch.setattr(store.time, 'sleep', lambda _: setattr(budget, 'last_resource_probe', 0))
    budget.throttle(1024)
    assert progress == [{'stage': 'paused_resource', 'resource_reason': 'low_disk_space'}]


def test_older_same_path_mutation_finishes_before_queued_reload(env):
    client, factory, _, _ = env
    entry, folder, _ = setup_entry(env, False)
    exact_location = location(client, folder)
    with factory() as db:
        check_id = next(op.id for op in db.query(DatasetOperation).all()
                        if op.kind == 'check' and op.payload.get('location_id') == exact_location)
    refresh_id = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    with factory() as db:
        assert store.local_refresh_blocked_by_older_mutation(db, db.get(DatasetOperation, refresh_id))
        assert not store.local_mutation_blocked_by_refresh(db, db.get(DatasetOperation, check_id))
    finish_local()
    assert operation(client, check_id)['status'] == 'completed'
    assert operation(client, refresh_id)['status'] == 'completed'


def test_preparing_upload_can_run_while_reload_waits_only_for_import(env):
    client, factory, _, _ = env
    entry, folder, _ = setup_entry(env, False)
    with factory() as db:
        upload = store.new_operation(db, 'upload', 'local', {
            'phase': 'prepare_upload', 'target_node': 'local',
            'destination': str(folder / 'nested')})
        db.commit()
        upload_id = upload.id
    refresh_id = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    with factory() as db:
        upload, refresh = db.get(DatasetOperation, upload_id), db.get(DatasetOperation, refresh_id)
        assert not store.local_mutation_blocked_by_refresh(db, upload)
        assert not store.local_refresh_blocked_by_older_mutation(db, refresh)
        upload.payload = {**upload.payload, 'phase': 'import'}
        db.commit()
    with factory() as db:
        upload, refresh = db.get(DatasetOperation, upload_id), db.get(DatasetOperation, refresh_id)
        assert store.local_refresh_blocked_by_older_mutation(db, refresh)
        assert not store.local_mutation_blocked_by_refresh(db, upload)


def test_remote_validation_shares_read_lane_and_queue_with_local_reload(env, monkeypatch):
    client, factory, root, agent = env
    local, _, _ = setup_entry(env, False)
    remote_root = root / 'remote'
    remote = node(factory, remote_root)
    loc = location(client, remote_root, remote)
    pump(client, factory, remote_root, agent, remote)
    make_existing(remote_root / 'sample')
    registered = client.post('/api/dataset-storage/datasets/register', json={
        'location_id': loc, 'name': 'sample'})
    assert registered.status_code == 202
    pump(client, factory, remote_root, agent, remote)
    remote_entry = next(entry for entry in entries(client) if entry['node_id'] == remote)
    remote_reload = client.post(f"/api/dataset-storage/datasets/{remote_entry['id']}/refresh").json()
    remote_tick(client, factory, remote_root, agent, remote)
    assert operation(client, remote_reload['id'])['phase'] == 'validate'
    local_reload = client.post(f"/api/dataset-storage/datasets/{local['id']}/refresh").json()
    assert operation(client, local_reload['id'])['queue_ahead'] == 1
    store.worker.tick()
    assert operation(client, remote_reload['id'])['status'] == 'pending'
    original_verify = store.verify_snapshot
    budgeted = []

    def verify_with_budget(op, check=None):
        if op.id == remote_reload['id']:
            budgeted.append(callable(getattr(check, 'throttle_bytes', None)))
        return original_verify(op, check)

    monkeypatch.setattr(store, 'verify_snapshot', verify_with_budget)
    store.worker.tick_refresh()
    assert operation(client, remote_reload['id'])['status'] == 'completed'
    assert budgeted == [True]
    assert operation(client, local_reload['id'])['queue_ahead'] == 0
    finish_local()
    assert operation(client, local_reload['id'])['status'] == 'completed'


def test_blocked_local_reload_does_not_count_ahead_of_runnable_remote_validation(env, monkeypatch):
    client, factory, root, _ = env
    entry, folder, _ = setup_entry(env, False)
    location(client, folder)  # The older exact-path check must finish first.
    local_id = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    assert operation(client, local_id)['waiting_path_lock']
    with factory() as db:
        remote = store.new_operation(db, 'refresh', 'local', {
            'phase': 'validate', 'source_node': 'remote', 'source_path': str(root / 'remote/sample')})
        regular = store.new_operation(db, 'check', 'local', {
            'phase': 'check', 'path': str(root / 'unrelated'), 'create': False})
        db.commit()
        remote_id, regular_id = remote.id, regular.id
    assert operation(client, remote_id)['queue_ahead'] == 0
    assert operation(client, regular_id)['queue_ahead'] == 1  # The older exact-path check only.
    dispatched = []
    monkeypatch.setattr(store, 'execute_local', lambda db, op: dispatched.append(op.id))
    store.worker.tick_refresh()
    assert dispatched == [remote_id]


def test_exact_path_recheck_without_create_waits_for_source_reload(env):
    client, _, _, _ = env
    entry, folder, _ = setup_entry(env, False)
    exact_location = location(client, folder)
    finish_local()
    refresh_id = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    recheck = client.post(f'/api/dataset-storage/locations/{exact_location}/check').json()
    assert operation(client, recheck['id'])['waiting_path_lock']
    store.worker.tick()
    assert operation(client, recheck['id'])['status'] == 'pending'
    finish_local()
    assert operation(client, refresh_id)['status'] == 'completed'
    assert operation(client, recheck['id'])['status'] == 'completed'


def test_refresh_disk_budget_accounts_for_long_unicode_paths():
    item = '分类/' + '烟火' * 120 + '/图片.png'
    manifest = {'total_bytes': 10, 'files': [{'path': item, 'size': 10}]}
    archive, unpacked = store.refresh_snapshot_disk_budget(manifest)
    assert archive == unpacked == 10 + 4096 + 2 * len(item.encode('utf-8'))
    assert archive > 10 + 4096 + 2 * len(item)


def test_local_budget_restores_stage_after_pressure_clears(monkeypatch):
    progress = []
    def check():
        pass
    check.progress = progress.append
    budget = store.LocalRefreshBudget(check)
    budget.remember_progress({'stage': 'pack', 'processed': 10, 'total': 100, 'unit': 'bytes'})
    monkeypatch.setattr(budget, 'training_now', lambda: True)
    monkeypatch.setattr(store, 'local_refresh_free_bytes', lambda: 1024**4)
    monkeypatch.setattr(store, 'DATASET_REFRESH_IO_PRESSURE_PCT', 25)
    values = iter([80.0, 0.0])
    monkeypatch.setattr(store, 'local_io_pressure_pct', lambda: next(values))
    monkeypatch.setattr(store.time, 'sleep', lambda _: setattr(budget, 'last_resource_probe', 0))
    budget.throttle(1024)
    assert progress == [
        {'stage': 'paused_resource', 'resource_reason': 'disk_pressure'},
        {'stage': 'pack', 'processed': 10, 'total': 100, 'unit': 'bytes'},
    ]
