"""Dataset lifecycle failures use isolated files and outbound-only Agent simulation."""
from datetime import timedelta
import json
from pathlib import Path
import shutil

import pytest

from app.models import DatasetEntry, DatasetLocation, DatasetOperation, TrainingRun, TrainingTask, utcnow
from test_dataset_storage import (env, entries, finish_local, location, make_existing,
                                  node, operation, pump, remote_tick, store, upload, uploads)
from test_dataset_refresh import setup_entry


@pytest.mark.parametrize('remote', [False, True])
@pytest.mark.parametrize('split', ['train', 'val', 'both'])
def test_empty_registration_is_rejected_on_local_and_remote(env, remote, split):
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    make_existing(folder / 'empty')
    for selected in ('train', 'val') if split == 'both' else (split,):
        shutil.rmtree(folder / 'empty/images' / selected)
    queued = client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'name': 'empty'})
    assert queued.status_code == 202
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    result = operation(client, queued.json()['id'])
    assert result['status'] == 'failed' and '至少需要一张有效图片' in result['error']
    assert entries(client) == []


@pytest.mark.parametrize('remote', [False, True])
def test_refresh_to_empty_split_preserves_last_valid_snapshot(env, remote):
    client, factory, _, agent = env
    entry, folder, identity = setup_entry(env, remote)
    shutil.rmtree(folder / 'images/val')
    queued = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, queued['id'])['status'] == 'failed'
    current = entries(client)[0]
    assert current['stats'] == entry['stats']
    assert current['checked_at'] == entry['checked_at']
    assert '至少需要一张有效图片' in current['refresh_error']


@pytest.mark.parametrize('remote', [False, True])
def test_location_root_cannot_gain_deletion_rights_via_adoption(env, remote):
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    make_existing(folder)
    queued = client.post('/api/dataset-storage/datasets/register', json={
        'location_id': loc, 'name': 'root-data', 'relative_path': '.'})
    assert queued.status_code == 202, queued.text
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    entry = entries(client)[0]
    adopted = client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)})
    assert adopted.status_code == 409 and '根目录' in adopted.text
    # Historical buggy adoption receives the same block in preview and execution.
    with factory() as db:
        db.get(DatasetEntry, entry['id']).ownership = 'managed'
        db.commit()
    preview = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert not preview['can_delete'] and '根目录' in preview['delete_block_reason']
    deletion = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert deletion.status_code == 409 and '根目录' in deletion.text
    assert folder.is_dir()


@pytest.mark.parametrize('remote', [False, True])
def test_adoption_cannot_overlap_another_external_entry(env, remote):
    client, factory, _, _ = env
    entry, folder, identity = setup_entry(env, remote)
    with factory() as db:
        db.add(DatasetEntry(id='nested', location_id=entry['location_id'], node_id=identity,
                            name='nested', path=str(folder / 'nested'), ownership='external',
                            status='ready', details={}))
        db.commit()
    adopted = client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)})
    assert adopted.status_code == 409 and '互相包含' in adopted.text


@pytest.mark.parametrize('remote', [False, True])
def test_deleted_history_does_not_pin_nondefault_location(env, remote):
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    uploaded = upload(client, loc)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    entry = entries(client)[0]
    deleted = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert deleted.status_code == 202, deleted.text
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    assert operation(client, deleted.json()['id'])['status'] == 'completed'
    assert client.patch(f'/api/dataset-storage/locations/{loc}', json={'enabled': True, 'is_default': False}).status_code == 200
    assert client.delete(f'/api/dataset-storage/locations/{loc}').status_code == 204
    with factory() as db:
        assert db.get(DatasetLocation, loc) is None
        history = db.get(DatasetEntry, entry['id'])
        assert history.status == 'deleted' and history.location_id == loc
        assert history.details['location_snapshot']['path'] == str(folder)
        receipt = db.get(DatasetOperation, uploaded)
        assert receipt.status == 'completed' and receipt.payload['location_name'] == folder.name
    history_page = client.get('/api/dataset-storage/operations/page', params={'q': folder.name}).json()
    assert history_page['total'] >= 2  # The location check and upload keep their location name.


def test_terminal_staging_cleanup_retries_without_transfer_cache(env, monkeypatch):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets')
    finish_local()
    identity = upload(client, loc)
    finish_local()
    with factory() as db:
        db.get(DatasetOperation, identity).updated_at = utcnow() - timedelta(days=9)
        db.commit()
    original = store.shutil.rmtree
    failed = []
    def transient_failure(path, *args, **kwargs):
        if Path(path).name == '.reclaimed-upload-' + identity and not failed:
            failed.append(True)
            raise PermissionError('temporary test lock')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(store.shutil, 'rmtree', transient_failure)
    with factory() as db:
        store.DatasetWorker().cleanup(db)
    assert failed and not store.job_dir(identity).exists()
    quarantine = uploads.UPLOAD_ROOT / ('.reclaimed-upload-' + identity)
    assert quarantine.is_dir()
    with factory() as db:
        store.DatasetWorker().cleanup(db)
    assert not quarantine.exists()
    assert (root / 'datasets/测试集/data.yaml').is_file()


def test_staging_cleanup_preserves_active_unknown_and_locked_sessions(env):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets')
    finish_local()
    identity = upload(client, loc)
    unknown = 'e' * 32
    unknown_dir = uploads.UPLOAD_ROOT / unknown
    unknown_dir.mkdir(parents=True)
    (unknown_dir / 'manifest.json').write_text(json.dumps({'operation_id': unknown}))
    (unknown_dir / '.lock').touch()
    with factory() as db:
        store.DatasetWorker().cleanup(db)
    assert (uploads.UPLOAD_ROOT / identity).is_dir() and unknown_dir.is_dir()
    finish_local()
    with factory() as db:
        db.get(DatasetOperation, identity).updated_at = utcnow() - timedelta(days=9)
        db.commit()
    with uploads.upload_session(identity, touch=False):
        with factory() as db:
            store.DatasetWorker().cleanup(db)
    assert (uploads.UPLOAD_ROOT / identity).is_dir() and unknown_dir.is_dir()


def test_remote_archive_hash_receipt_avoids_duplicate_read_and_invalidates_on_change(env, monkeypatch):
    client, factory, root, agent = env
    entry, folder, identity = setup_entry(env, True)
    calls = []
    original = store.files.digest
    def counted(path, check=None, stage=None):
        if Path(path).name in ('bundle.part', 'bundle.tar'):
            calls.append(str(path))
        return original(path, check, stage)
    monkeypatch.setattr(store.files, 'digest', counted)
    first = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    pump(client, factory, folder.parent, agent, identity)
    assert operation(client, first)['status'] == 'completed'
    assert len(calls) == 1
    calls.clear()
    second = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    remote_tick(client, factory, folder.parent, agent, identity)
    assert operation(client, second)['phase'] == 'validate'
    archive = store.job_dir(second) / 'bundle.tar'
    archive.write_bytes(archive.read_bytes() + b'changed')
    finish_local()
    assert operation(client, second)['status'] == 'failed'
    assert len(calls) == 2
    assert entries(client)[0]['stats'] == entry['stats']


def test_catalogue_reference_queries_are_batched_without_losing_historical_refs(env):
    from sqlalchemy import event
    client, factory, root, _ = env
    identity = node(factory, root / 'remote')
    with factory() as db:
        for index in range(30):
            path = str(root / 'remote' / str(index))
            db.add(DatasetEntry(id=f'entry-{index}', location_id='history', node_id=identity,
                                name=str(index), path=path, ownership='managed', status='ready',
                                details={'config_sha256': 'verified', 'fingerprint': 'known'}))
            db.add(TrainingTask(id=f'task-{index}', name=str(index), model_path='model.pt',
                                data_path=path + '/data.yaml', execution_node_id=identity, status='completed'))
        db.flush()
        db.add(TrainingRun(id='historical-run', task_id='task-0', attempt_number=1,
                          model_path='model.pt', data_path=str(root / 'remote/1/data.yaml'),
                          execution_node_id=identity, status='completed'))
        db.commit()
    queries = []
    engine = factory.kw['bind']
    def count(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith('SELECT'):
            queries.append(statement)
    event.listen(engine, 'before_cursor_execute', count)
    try:
        catalogue = {entry['id']: entry for entry in entries(client)}
    finally:
        event.remove(engine, 'before_cursor_execute', count)
    assert len(queries) <= 12, f'Catalogue emitted {len(queries)} SELECTs for 30 entries'
    assert {ref['id'] for ref in catalogue['entry-1']['references']} == {'task-0', 'task-1'}
    assert {ref['id'] for ref in catalogue['entry-0']['references']} == {'task-0'}
    assert all(not entry['can_delete'] for entry in catalogue.values())


def test_explicit_local_node_reference_blocks_dataset_removal(env):
    client, factory, _, _ = env
    entry, folder, _ = setup_entry(env, False)
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)}).status_code == 200
    with factory() as db:
        db.add(TrainingTask(id='explicit-local', name='local', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), execution_node_id='local', status='completed'))
        db.commit()
    current = entries(client)[0]
    assert current['references'][0]['id'] == 'explicit-local' and not current['can_delete']
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 409


@pytest.mark.parametrize('remote', [False, True])
def test_historical_reference_survives_task_moving_to_another_node(env, remote):
    client, factory, root, _ = env
    entry, folder, identity = setup_entry(env, remote)
    assert client.post(f"/api/dataset-storage/datasets/{entry['id']}/adopt", json={'confirm_path': str(folder)}).status_code == 200
    new_node = None if remote else node(factory, root / 'other-server')
    with factory() as db:
        db.add(TrainingTask(id='moved-task', name='moved', model_path='model.pt',
                            data_path=str(root / 'new-dataset/data.yaml'), execution_node_id=new_node,
                            status='completed'))
        db.flush()
        db.add(TrainingRun(id='old-node-run', task_id='moved-task', attempt_number=1,
                          model_path='model.pt', data_path=str(folder / 'data.yaml'),
                          execution_node_id=identity if remote else None, status='completed'))
        db.commit()
    current = entries(client)[0]
    assert current['references'][0]['id'] == 'moved-task' and not current['can_delete']
    preview = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert preview['references'][0]['id'] == 'moved-task' and not preview['can_delete']
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 409
