"""Removing a catalogue row preserves data, training and deliberate re-registration."""
import json
import uuid

import pytest
from sqlalchemy import select

from app.models import ComputeNode, DatasetEntry, DatasetLocation, DatasetOperation, DatasetOperationHistoryRemoval, TrainingRun, TrainingTask
from test_dataset_storage import env, entries, finish_local, location, make_existing, node, operation, services, store, uploads


def register(client, loc, name, ownership='external'):
    response = client.post('/api/dataset-storage/datasets/register', json={
        'location_id': loc, 'name': name, 'ownership': ownership, 'acknowledge': ownership == 'managed'})
    assert response.status_code == 202, response.text
    finish_local()
    assert operation(client, response.json()['id'])['status'] == 'completed'
    return entries(client)[0]


@pytest.mark.parametrize('status', ['running', 'queued', 'stopped', 'completed'])
@pytest.mark.parametrize('ownership', ['external', 'managed'])
def test_remove_real_row_during_referenced_training_preserves_files_and_path_access(env, monkeypatch, status, ownership):
    client, factory, root, _ = env
    loc = location(client, root / 'custom'); finish_local()
    folder = root / 'custom' / '原数据'; make_existing(folder)
    entry = register(client, loc, folder.name, ownership)
    before = {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    with factory() as db:
        task = TrainingTask(id='task', name='训练中', model_path='base.pt', data_path=entry['config_path'],
                            status=status, current_run_id='run')
        db.add(task); db.flush()
        db.add(TrainingRun(id='run', task_id='task', attempt_number=1, status=status,
                          model_path='base.pt', data_path=entry['config_path'],
                          parameter_snapshot={'dataset_snapshot': {'dataset_id': entry['id']}}))
        db.commit()
    def no_file_work(*args, **kwargs):
        raise AssertionError('metadata removal must never scan, move or remove dataset files')
    monkeypatch.setattr(store.files, 'remove_dataset', no_file_work)
    monkeypatch.setattr(store.files, 'scan', no_file_work)
    monkeypatch.setattr(store, 'node_or_error', no_file_work)
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204
    assert entries(client) == []
    assert {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()} == before
    with factory() as db:
        assert db.get(DatasetEntry, entry['id']) is None
        task, run = db.get(TrainingTask, 'task'), db.get(TrainingRun, 'run')
        assert task.status == run.status == status
        assert task.data_path == run.data_path == entry['config_path']
        assert run.parameter_snapshot['dataset_snapshot']['dataset_id'] == entry['id']
        receipt = db.scalar(select(DatasetOperation).where(DatasetOperation.kind == 'unregister'))
        assert receipt.status == 'completed' and receipt.result['files_preserved']
        # Removing visible journal history must not revoke existing task access.
        db.add(DatasetOperationHistoryRemoval(operation_id=receipt.id)); db.commit()
    assert services.validate_dataset_config(entry['config_path']).is_file()
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204


def test_legacy_import_does_not_resurrect_removed_row_and_explicit_registration_works(env):
    client, factory, root, _ = env
    folder = root / 'legacy' / '旧数据'; make_existing(folder)
    (folder / uploads.MANAGED_MARKER).write_text(json.dumps({
        'version': 1, 'name': folder.name, 'upload_id': uuid.uuid4().hex, 'classes': ['人']}))
    entry = entries(client)[0]
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204
    assert entries(client) == entries(client) == []
    with factory() as db:
        loc = db.get(DatasetLocation, entry['location_id'])
        # Even recreating a legacy location cannot resurrect an explicit removal.
        db.delete(loc); db.commit()
    assert entries(client) == []
    from test_dataset_storage import start_scan, candidates
    scan, _ = start_scan(client, [entry['location_id']]); finish_local()
    found = candidates(client, scan['id'])
    assert found['counts']['ready'] == 1
    assert entries(client) == []  # Discovery itself does not re-register files.
    replacement = register(client, entry['location_id'], folder.name)
    assert replacement['id'] != entry['id']
    assert replacement['ownership'] == 'external' and folder.is_dir()
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204
    assert entries(client)[0]['id'] == replacement['id']


def test_operation_page_filters_unregister_receipts_and_rejects_unknown_kind(env):
    client, _, root, _ = env
    loc = location(client, root / 'datasets'); finish_local()
    make_existing(root / 'datasets' / 'A')
    entry = register(client, loc, 'A')
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204
    response = client.get('/api/dataset-storage/operations/page', params={'kind': 'unregister'})
    assert response.status_code == 200, response.text
    page = response.json()
    assert page['total'] == len(page['items']) == 1
    receipt = page['items'][0]
    assert receipt['kind'] == 'unregister' and receipt['status'] == 'completed'
    assert receipt['name'] == 'A'
    all_items = client.get('/api/dataset-storage/operations/page').json()['items']
    assert any(item['kind'] == 'register' for item in all_items)
    assert client.get('/api/dataset-storage/operations/page', params={'kind': 'unknown'}).status_code == 422


@pytest.mark.parametrize('kind,status,by_id', [
    ('refresh', 'running', True), ('delete', 'pending', True), ('migrate', 'failed', False),
    ('register', 'pending', False), ('upload', 'committing', False)])
def test_related_operations_block_registration_removal_without_deleting_anything(env, kind, status, by_id):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets'); finish_local()
    folder = root / 'datasets' / 'A'; make_existing(folder)
    entry = register(client, loc, 'A')
    with factory() as db:
        payload = {'phase': 'export', 'source_node': 'local', 'source_path': str(folder), 'name': 'A'}
        if by_id:
            payload['entry_id'] = entry['id']
        if kind == 'upload':
            payload = {'phase': 'import', 'target_node': 'local', 'destination': str(folder), 'name': 'A'}
        db.add(DatasetOperation(id=uuid.uuid4().hex, kind=kind, status=status, node_id='local', payload=payload))
        db.commit()
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 409
    assert entries(client)[0]['can_unregister'] is False
    assert folder.is_dir()
    with factory() as db:
        assert db.get(DatasetEntry, entry['id']) is not None


def test_remote_metadata_operations_work_during_training_without_agent_connection(env):
    client, factory, root, _ = env
    remote = node(factory, root / 'remote')
    with factory() as db:
        db.add(DatasetLocation(id='remote-loc', node_id=remote, path=str(root / 'remote'), name='远程'))
        entry = DatasetEntry(id='remote-dataset', node_id=remote, location_id='remote-loc', name='远程数据',
                             path=str(root / 'remote' / 'A'), ownership='external', status='ready')
        db.add(entry)
        db.add(TrainingTask(id='remote-task', name='训练', execution_node_id=remote, status='running',
                            model_path='base.pt', data_path=str(root / 'remote' / 'A' / 'data.yaml')))
        db.commit()
    assert client.post('/api/dataset-storage/datasets/remote-dataset/adopt', json={
        'confirm_path': str(root / 'remote' / 'A')}).status_code == 200
    with factory() as db:
        assert db.get(DatasetEntry, 'remote-dataset').ownership == 'managed'
        db.get(ComputeNode, remote).enabled = False; db.commit()
    assert client.delete('/api/dataset-storage/datasets/remote-dataset/registration').status_code == 204
    with factory() as db:
        assert db.get(DatasetEntry, 'remote-dataset') is None
        assert db.get(TrainingTask, 'remote-task').status == 'running'


def test_adopt_checks_path_keeps_row_on_error_and_allows_running_local_task(env):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets'); finish_local()
    folder = root / 'datasets' / 'A'; make_existing(folder)
    entry = register(client, loc, 'A')
    with factory() as db:
        db.add(TrainingTask(id='running', name='训练', model_path='base.pt', data_path=entry['config_path'], status='running'))
        db.commit()
    url = f"/api/dataset-storage/datasets/{entry['id']}/adopt"
    assert client.post(url, json={'confirm_path': str(root)}).status_code == 422
    assert client.post(url, json={'confirm_path': str(folder)}).status_code == 200
    assert entries(client)[0]['ownership'] == 'managed'
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 409
    assert folder.is_dir()


def test_old_client_upload_after_bootstrap_immediately_registers_and_retry_does_not_resurrect(env):
    from test_dataset_storage import images
    client, factory, root, _ = env
    legacy = root / 'legacy'; legacy.mkdir()
    assert entries(client) == []  # Finish the one-time legacy import before uploading.
    files = images()
    response = client.post('/api/datasets/uploads', json={'name': '旧客户端上传',
        'files': [{'path': name, 'size': len(content)} for name, content in files.items()]})
    identity = response.json()['upload_id']
    for index, content in enumerate(files.values()):
        assert client.put(f'/api/datasets/uploads/{identity}/files/{index}', content=content).status_code == 204
    url = f'/api/datasets/uploads/{identity}/complete'
    assert client.post(url, json={'names': ['人']}).status_code == 200
    entry = entries(client)[0]
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204
    assert client.post(url, json={'names': ['人']}).status_code == 200
    assert entries(client) == []
    with factory() as db:
        store.record_legacy_publication(db, legacy / '旧客户端上传', recovering=True)
    assert entries(client) == []
    # A later same-path publication interrupted before its receipt is written
    # is a new dataset, rather than a replay of the explicitly removed card.
    from datetime import datetime, timedelta, timezone
    marker = legacy / '旧客户端上传' / uploads.MANAGED_MARKER
    metadata = json.loads(marker.read_text())
    metadata.update(upload_id=uuid.uuid4().hex, created_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat())
    marker.write_text(json.dumps(metadata))
    with factory() as db:
        store.record_legacy_publication(db, marker.parent, recovering=True)
    assert len(entries(client)) == 1


def test_unrelated_dataset_work_does_not_block_unregister(env):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets'); finish_local()
    make_existing(root / 'datasets' / 'A')
    entry = register(client, loc, 'A')
    with factory() as db:
        db.add(DatasetOperation(id=uuid.uuid4().hex, kind='refresh', status='running', node_id='local',
                               payload={'phase': 'export', 'source_path': str(root / 'datasets' / 'B')}))
        db.commit()
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}/registration").status_code == 204


def test_location_recheck_preserves_import_marker_but_clears_obsolete_error(env):
    client, factory, root, _ = env
    legacy = root / 'legacy'; legacy.mkdir()
    assert entries(client) == []
    with factory() as db:
        loc = db.scalar(select(DatasetLocation).where(DatasetLocation.path == str(legacy)))
        loc.details = {**loc.details, 'message': '旧错误'}
        loc_id = loc.id; db.commit()
    response = client.post(f'/api/dataset-storage/locations/{loc_id}/check', json={})
    assert response.status_code == 202
    finish_local()
    with factory() as db:
        loc = db.get(DatasetLocation, loc_id)
        assert loc.details['legacy_catalogue_imported'] is True
        assert loc.details.get('message') != '旧错误'
