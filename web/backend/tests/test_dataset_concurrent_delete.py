"""Path scoped deletion proves unrelated training is safe without weakening references."""
from pathlib import Path
import uuid

import pytest
import yaml

from app import dataset_conflicts as conflicts
from app.models import ComputeNode, DatasetEntry, DatasetOperation, TrainingRun, TrainingTask, utcnow
from test_dataset_storage import env, entries, finish_local, location, make_existing as make_storage_existing, node, pump, store
from test_compute_nodes import monitoring_client, create as create_node, headers, sample, _queue_remote_run


def make_existing(path, *, config=None, bad_label=False):
    # The shared storage fixture uses path='.', whose trainer CWD is unknown.
    # These safety tests need a proven YAML-parent root unless explicitly set.
    if config is None:
        config = {'train': 'images/train', 'val': 'images/val', 'names': ['人员']}
    make_storage_existing(path, config=config, bad_label=bad_label)


def managed_entry(env, *, remote=False, concurrent=False):
    client, factory, root, agent = env
    folder = root / 'datasets'
    identity = node(factory, folder) if remote else 'local'
    loc = location(client, folder, identity)
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    make_existing(folder / 'B')
    result = client.post('/api/dataset-storage/datasets/register', json={
        'location_id': loc, 'name': 'B', 'ownership': 'managed', 'acknowledge': True})
    assert result.status_code == 202, result.text
    pump(client, factory, folder, agent, identity) if remote else finish_local()
    if remote:
        with factory() as db:
            n = db.get(ComputeNode, identity)
            n.snapshot = {**n.snapshot, 'training': {**n.snapshot.get('training', {}),
                          'dataset_concurrent_delete': concurrent}}
            db.commit()
    return entries(client)[0], folder, identity


def add_training(factory, data_path, *, node_id='local', status='running', launch_mode='fresh', run_path=None, run_dir=None):
    with factory() as db:
        task = TrainingTask(id=uuid.uuid4().hex, name='A training', model_path='model.pt',
                            data_path=str(data_path), status=status, execution_node_id=None if node_id == 'local' else node_id)
        db.add(task); db.flush()
        run = TrainingRun(id=uuid.uuid4().hex, task_id=task.id, attempt_number=1, status=status,
                          model_path='model.pt', data_path=str(run_path or data_path),
                          launch_mode=launch_mode, execution_node_id=task.execution_node_id,
                          run_dir=str(run_dir) if run_dir else None)
        db.add(run); task.current_run_id = run.id; db.commit()
        return task.id, run.id


def test_unrelated_local_training_allows_delete_and_target_scoped_queue(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    add_training(factory, folder / 'A/data.yaml')
    view = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert view['can_delete'], view
    deletion = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert deletion.status_code == 202, deletion.text
    with factory() as db:
        op = db.get(DatasetOperation, deletion.json()['id'])
        assert op.payload['concurrent_delete'] is True
        assert store.mutation_pending(db, 'local', str(folder / 'B/data.yaml'))
        assert not store.mutation_pending(db, 'local', str(folder / 'A/data.yaml'))
        assert not store.mutation_pending(db, 'local', node_only=True)
    finish_local()
    assert not (folder / 'B').exists()
    assert (folder / 'A').is_dir()


@pytest.mark.parametrize('mode', ['same', 'yaml_alias', 'manifest', 'manifest_label_link', 'symlink', 'nested_link', 'label_link', 'unknown', 'invalid', 'run_only', 'resume_unknown'])
def test_active_training_conflicts_and_unknown_evidence_block_delete(env, mode):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    data_path = folder / 'A/data.yaml'
    run_path = None
    launch = 'fresh'
    if mode == 'same':
        data_path = folder / 'B/data.yaml'
    elif mode == 'yaml_alias':
        config = yaml.safe_load(data_path.read_text()); config['path'] = str(folder / 'B')
        data_path.write_text(yaml.safe_dump(config))
    elif mode == 'manifest':
        manifest = folder / 'list.txt'; manifest.write_text(str(folder / 'B/images/train/a.png') + chr(10))
        config = yaml.safe_load(data_path.read_text()); config['train'] = str(manifest)
        data_path.write_text(yaml.safe_dump(config))
    elif mode == 'manifest_label_link':
        import shutil
        shutil.rmtree(folder / 'A/labels')
        (folder / 'A/labels').symlink_to(folder / 'B/labels', target_is_directory=True)
        manifest = folder / 'list.txt'; manifest.write_text(str(folder / 'A/images/train/a.png') + chr(10))
        config = yaml.safe_load(data_path.read_text()); config['train'] = str(manifest)
        data_path.write_text(yaml.safe_dump(config))
    elif mode == 'symlink':
        link = folder / 'alias'; link.symlink_to(folder / 'B', target_is_directory=True)
        config = yaml.safe_load(data_path.read_text()); config['train'] = str(link / 'images/train')
        data_path.write_text(yaml.safe_dump(config))
    elif mode == 'unknown':
        data_path = folder / 'missing.yaml'
    elif mode == 'nested_link':
        (folder / 'A/images/train/nested').symlink_to(folder / 'B/images/train', target_is_directory=True)
    elif mode == 'label_link':
        (folder / 'A/labels/train/nested').symlink_to(folder / 'B/labels/train', target_is_directory=True)
    elif mode == 'invalid':
        data_path.write_text('train: [broken')
    elif mode == 'run_only':
        run_path = folder / 'B/data.yaml'
    elif mode == 'resume_unknown':
        launch = 'resume'
    add_training(factory, data_path, run_path=run_path, launch_mode=launch)
    view = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert not view['can_delete'], mode
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 409
    assert (folder / 'B').is_dir()


def test_resume_runtime_alias_and_historical_recycle_references_remain_protected(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    runtime = root / 'run'; runtime.mkdir()
    (runtime / 'args.yaml').write_text(yaml.safe_dump({'data': str(folder / 'B/data.yaml')}))
    task_id, _ = add_training(factory, folder / 'A/data.yaml', status='completed', launch_mode='resume', run_dir=runtime)
    with factory() as db:
        db.get(TrainingTask, task_id).deleted_at = utcnow(); db.commit()
    view = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert not view['can_delete'] and view['references'][0]['deleted']


def test_pending_and_committed_failed_deletes_protect_only_target(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    deletion = client.delete(f"/api/dataset-storage/datasets/{entry['id']}").json()
    with factory() as db:
        op = db.get(DatasetOperation, deletion['id'])
        op.status = 'failed'; db.commit()
        assert store.mutation_pending(db, 'local', str(folder / 'B/data.yaml'))
        assert not store.mutation_pending(db, 'local', str(folder / 'A/data.yaml'))
        op.payload = {**op.payload, 'commit_started': True}; db.commit()
        assert store.mutation_pending(db, 'local', str(folder / 'B/data.yaml'))
        assert not store.mutation_pending(db, 'local', str(folder / 'A/data.yaml'))
        op.payload = {key: value for key, value in op.payload.items() if key != 'concurrent_delete'}; db.commit()
        assert store.mutation_pending(db, 'local', str(folder / 'A/data.yaml'))


@pytest.mark.parametrize('capable', [False, True])
def test_remote_old_agent_falls_back_and_capable_agent_receives_scoped_delete(env, capable):
    client, factory, root, _ = env
    entry, folder, identity = managed_entry(env, remote=True, concurrent=capable)
    make_existing(folder / 'A')
    with factory() as db:
        db.add(DatasetEntry(id=uuid.uuid4().hex, node_id=identity, location_id=entry['location_id'],
                            name='A', path=str(folder / 'A'), ownership='external', status='ready',
                            details={'config_sha256': 'a' * 64}))
        db.commit()
    add_training(factory, folder / 'A/data.yaml', node_id=identity)
    result = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert result.status_code == (202 if capable else 409), result.text
    if capable:
        work = client.post(f'/api/dataset-storage/agent/{identity}/poll', params={'lightweight_only': True}).json()['operation']
        assert work['kind'] == 'delete' and work['concurrent_delete'] is True
        with factory() as db:
            assert not store.node_dataset_file_work(db, identity)
            assert store.mutation_pending(db, identity, str(folder / 'B/data.yaml'))
            assert not store.mutation_pending(db, identity, str(folder / 'A/data.yaml'))


@pytest.mark.parametrize('capable', [False, True])
@pytest.mark.parametrize('task_status', ['running', 'completed'])
def test_catalogue_batches_active_training_checks_without_weakening_run_protection(env, capable, task_status):
    from sqlalchemy import event
    client, factory, _, _ = env
    entry, folder, identity = managed_entry(env, remote=True, concurrent=capable)
    with factory() as db:
        db.add(DatasetEntry(id='active-source', node_id=identity, location_id=entry['location_id'],
                            name='A', path=str(folder / 'A'), ownership='external', status='ready',
                            details={'config_sha256': 'verified'}))
        for index in range(29):
            db.add(DatasetEntry(id=f'independent-{index}', node_id=identity, location_id=entry['location_id'],
                                name=f'independent-{index}', path=str(folder / f'independent-{index}'),
                                ownership='managed', status='ready',
                                details={'config_sha256': 'verified', 'fingerprint': 'known'}))
        db.commit()
    task_id, _ = add_training(factory, folder / 'A/data.yaml', node_id=identity)
    with factory() as db:
        db.get(TrainingTask, task_id).status = task_status
        db.commit()
    queries = []
    engine = factory.kw['bind']
    def count(_connection, _cursor, statement, _parameters, _context, _many):
        if statement.lstrip().upper().startswith('SELECT'):
            queries.append(statement)
    event.listen(engine, 'before_cursor_execute', count)
    try:
        catalogue = {item['id']: item for item in entries(client)}
    finally:
        event.remove(engine, 'before_cursor_execute', count)
    assert len(queries) <= 12, f'Catalogue emitted {len(queries)} SELECTs for 31 entries'
    targets = [item for key, item in catalogue.items() if key != 'active-source']
    assert len(targets) == 30
    assert all(item['can_delete'] is capable and item['references'] == [] for item in targets)
    assert {ref['id'] for ref in catalogue['active-source']['references']} == {task_id}


def test_path_parser_rejects_unknown_and_follows_manifest(env):
    _, _, root, _ = env
    make_existing(root / 'A'); make_existing(root / 'B')
    assert conflicts.data_path_conflict(root / 'A/data.yaml', root / 'B') is False
    assert conflicts.data_path_conflict(root / 'missing.yaml', root / 'B') is None
    listing = root / 'A/list.txt'
    listing.write_text('images/train/a.png' + chr(10))
    assert conflicts.manifest_conflict(listing, root / 'B') is None
    listing.write_text('./images/train/a.png' + chr(10))
    assert conflicts.manifest_conflict(listing, root / 'B') is False
    config = yaml.safe_load((root / 'A/data.yaml').read_text())
    config['minival'] = str(root / 'B/images/val')
    (root / 'A/data.yaml').write_text(yaml.safe_dump(config))
    assert conflicts.data_path_conflict(root / 'A/data.yaml', root / 'B') is True
    (root / 'A/class-link').symlink_to(root / 'B', target_is_directory=True)
    assert conflicts.data_path_conflict(root / 'A', root / 'B', task_type='classify') is True


def test_relative_config_root_cannot_prove_yaml_sibling_is_independent(env, monkeypatch):
    client, factory, _, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    # A YAML-relative interpretation would see an independent, existing tree.
    # Trainer CWD resolves this same root to the deletion target instead.
    make_existing(folder / 'A' / 'B')
    config_path = folder / 'A' / 'data.yaml'
    config = yaml.safe_load(config_path.read_text())
    config['path'] = 'B'
    config_path.write_text(yaml.safe_dump(config))
    monkeypatch.chdir(folder)
    assert Path(config['path']).resolve() == folder / 'B'
    assert conflicts.data_path_conflict(config_path, folder / 'B') is None
    add_training(factory, config_path)
    preview = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert not preview['can_delete']
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 409
    assert (folder / 'B').is_dir()
    # Absolute split evidence remains sufficient to identify a real conflict.
    config['train'] = str(folder / 'B' / 'images' / 'train')
    config_path.write_text(yaml.safe_dump(config))
    assert conflicts.data_path_conflict(config_path, folder / 'B') is True


@pytest.mark.parametrize('explicit_root', [False, True])
def test_proven_default_or_absolute_config_root_allows_unrelated_delete(env, explicit_root):
    client, factory, _, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    config_path = folder / 'A' / 'data.yaml'
    config = yaml.safe_load(config_path.read_text())
    if explicit_root:
        config['path'] = str(folder / 'A')
    else:
        config.pop('path', None)
    config_path.write_text(yaml.safe_dump(config))
    assert conflicts.data_path_conflict(config_path, folder / 'B') is False
    add_training(factory, config_path)
    preview = client.get(f"/api/dataset-storage/datasets/{entry['id']}/delete-preview").json()
    assert preview['can_delete'], preview
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 202
    finish_local()
    assert not (folder / 'B').exists() and (folder / 'A').is_dir()


def test_pending_delete_rechecks_new_reference_before_unlink(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    response = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    assert response.status_code == 202
    add_training(factory, folder / 'B/data.yaml', status='queued')
    finish_local()
    result = client.get('/api/dataset-storage/operations/' + response.json()['id']).json()
    assert result['status'] == 'failed'
    assert (folder / 'B').is_dir()


def test_completed_delete_cannot_retry_against_same_named_new_dataset(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    response = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    finish_local()
    make_existing(folder / 'B')
    assert client.post('/api/dataset-storage/operations/' + response.json()['id'] + '/retry').status_code == 409
    assert (folder / 'B').is_dir()
    with factory() as db:
        old = db.get(DatasetOperation, response.json()['id'])
        old.status = 'failed'; db.commit()
    assert client.post('/api/dataset-storage/operations/' + response.json()['id'] + '/retry').status_code == 200
    finish_local()
    assert (folder / 'B').is_dir()


def test_scoped_delete_does_not_relax_move_or_unknown_resume_start(env):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    add_training(factory, folder / 'A/data.yaml')
    with factory() as db:
        e = db.get(DatasetEntry, entry['id'])
        assert store.removal_problem(db, e, allow_concurrent=False)
    response = client.delete(f"/api/dataset-storage/datasets/{entry['id']}")
    _, run_id = add_training(factory, folder / 'A/data.yaml', status='queued', launch_mode='resume')
    with factory() as db:
        run = db.get(TrainingRun, run_id)
        assert store.mutation_pending(db, 'local', run.data_path, run=run)


def test_catalogue_reuses_directory_evidence_per_response(env, monkeypatch):
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    add_training(factory, folder / 'A/data.yaml')
    for index in range(5):
        with factory() as db:
            db.add(DatasetEntry(id=uuid.uuid4().hex, node_id='local', location_id=entry['location_id'],
                                name=f'empty-{index}', path=str(folder / f'empty-{index}'), ownership='managed', status='ready'))
            db.commit()
    calls = []
    original = conflicts.os.scandir
    def scan(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(conflicts.os, 'scandir', scan)
    result = client.get('/api/dataset-storage/datasets')
    assert result.status_code == 200
    assert len([path for path in calls if path == str(folder / 'A/images/train')]) == 1


@pytest.mark.parametrize('mode', ['separate', 'same', 'unknown', 'old_operation', 'old_agent'])
def test_remote_training_poll_applies_candidate_path_guard(monitoring_client, mode):
    client, factory = monitoring_client
    credential = create_node(client)
    node_id = credential['node']['id']
    report = sample()
    report['agent_version'] = '2.6.3'
    report['training'].update(dataset_roots=['/remote/datasets'], dataset_concurrent_delete=mode != 'old_agent')
    response = client.post(f'/api/compute/nodes/{node_id}/report', headers=headers(credential), json=report)
    assert response.status_code == 200, response.text
    with factory() as db:
        run = _queue_remote_run(db, node_id, 'candidate', 'cpu')
        path = '/remote/datasets/B/data.yaml' if mode == 'same' else '/remote/datasets/A/data.yaml'
        run.data_path = path
        db.get(TrainingTask, run.task_id).data_path = path
        db.add(DatasetEntry(id='target', node_id=node_id, location_id='loc', name='B', path='/remote/datasets/B',
                            ownership='managed', status='ready', details={'config_sha256': 'b' * 64}))
        if mode != 'unknown':
            db.add(DatasetEntry(id='unrelated', node_id=node_id, location_id='loc', name='A', path='/remote/datasets/A',
                                ownership='external', status='ready', details={'config_sha256': 'a' * 64}))
        payload = {'phase': 'remove', 'entry_id': 'target', 'source_node': node_id, 'source_path': '/remote/datasets/B'}
        if mode != 'old_operation':
            payload['concurrent_delete'] = True
        store.new_operation(db, 'delete', node_id, payload)
        db.commit()
    response = client.post(f'/api/compute/nodes/{node_id}/agent/poll', headers=headers(credential))
    assert response.status_code == 200, response.text
    assert (response.json()['task'] is not None) == (mode == 'separate'), response.json()


@pytest.mark.parametrize('separate', [False, True])
def test_local_scheduler_candidate_path_is_guarded(env, separate):
    from app.services import TrainingScheduler
    client, factory, root, _ = env
    entry, folder, _ = managed_entry(env)
    make_existing(folder / 'A')
    assert client.delete(f"/api/dataset-storage/datasets/{entry['id']}").status_code == 202
    _, run_id = add_training(factory, folder / ('A' if separate else 'B') / 'data.yaml', status='queued')
    with factory() as db:
        claimed = TrainingScheduler()._claim_queued_run(db, run_id, assignments=[])
        assert claimed == separate
