"""Reload real dataset snapshots through local and legacy-compatible remote paths."""
import pytest
import yaml

from app.models import DatasetEntry, DatasetOperation, TrainingTask
from test_dataset_storage import (env, entries, finish_local, images, location, make_existing,
                                  node, operation, pump, store)


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
def test_reload_waits_for_training_then_finishes_without_changing_training(env, remote):
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
    assert operation(client, reload['id'])['waiting_training']
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, reload['id'])['status'] == 'pending'
    with factory() as db:
        assert db.get(TrainingRun, 'busy-run').status == 'running'
        assert db.get(TrainingTask, 'busy').status == 'running'
        db.get(TrainingRun, 'busy-run').status = 'completed'
        db.get(TrainingTask, 'busy').status = 'completed'
        db.commit()
    pump(client, factory, folder.parent, agent, identity) if remote else finish_local()
    assert operation(client, reload['id'])['status'] == 'completed'
    assert entries(client)[0]['references'][0]['id'] == 'busy'


def test_waiting_local_reload_does_not_block_later_directory_checks(env):
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
    assert operation(client, refresh['id'])['status'] == 'pending'
    assert operation(client, check['id'])['status'] == 'completed'
