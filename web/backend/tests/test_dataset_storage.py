"""Dataset storage transactions use real files and a simulated outbound-only Agent."""
import base64
import importlib.util
import io
import json
import shutil
import uuid
from pathlib import Path
from urllib.parse import urlencode

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app import dataset_files as core, dataset_store as store, dataset_uploads as uploads, services, dataset_discovery as discovery
from app.database import Base, get_db
from app.models import ComputeNode, DatasetEntry, DatasetOperation, TrainingTask


@pytest.fixture
def env(tmp_path, monkeypatch):
    engine = create_engine(f'sqlite:///{tmp_path}/test.db', connect_args={'check_same_thread': False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(store, 'SessionLocal', factory)
    monkeypatch.setattr(services, 'SessionLocal', factory)
    monkeypatch.setattr(store, 'PROJECT_ROOT', tmp_path)
    monkeypatch.setattr(store, 'STORE_ROOT', tmp_path / 'transfer')
    monkeypatch.setattr(store, 'DATASET_STORAGE_ALLOWED_ROOTS', [tmp_path])
    import app.config as config
    monkeypatch.setattr(config, 'DATASET_STORAGE_ALLOWED_ROOTS', [tmp_path])
    monkeypatch.setattr(uploads, 'UPLOAD_ROOT', tmp_path / 'staging')
    monkeypatch.setattr(uploads, 'DATASET_ROOTS', [tmp_path / 'legacy'])
    monkeypatch.setattr(services, 'DATASET_ROOTS', [tmp_path / 'legacy'])
    app = FastAPI()
    app.include_router(store.router)
    app.include_router(discovery.router)
    app.include_router(uploads.router)
    app.include_router(uploads.management_router)
    def override():
        with factory() as db:
            yield db
    app.dependency_overrides[get_db] = override
    spec = importlib.util.spec_from_file_location('dataset_test_agent', Path(__file__).resolve().parents[2] / 'agent' / 'monitor_agent.py')
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    def api(_config, path, method='GET', payload=None, timeout=120):
        response = client.request(method, path, json=payload)
        if response.status_code >= 400:
            raise ValueError(response.text)
        return response.json()
    monkeypatch.setattr(agent, 'dataset_api', api)
    monkeypatch.setattr(store, 'auth', lambda node_id, request, db: db.get(ComputeNode, node_id))
    with TestClient(app) as client:
        yield client, factory, tmp_path, agent
    engine.dispose()


def finish_local():
    for _ in range(8):
        store.worker.tick()


def location(client, path, node='local'):
    result = client.post('/api/dataset-storage/locations', json={'node_id': node, 'name': path.name, 'path': str(path), 'create': True, 'is_default': True})
    assert result.status_code == 201, result.text
    return result.json()['id']


def images():
    stream = io.BytesIO()
    Image.new('RGB', (8, 8), 'red').save(stream, 'PNG')
    return {'images/train/a.png': stream.getvalue(), 'images/val/b.png': stream.getvalue(),
            'labels/train/a.txt': b'0 0.5 0.5 0.2 0.2\n', 'labels/val/b.txt': b'0 0.5 0.5 0.2 0.2\n'}


def upload(client, loc, name='测试集'):
    content = images()
    manifest = [{'path': p, 'size': len(v), 'sha256': __import__('hashlib').sha256(v).hexdigest()} for p, v in content.items()]
    response = client.post('/api/datasets/uploads', json={'name': name, 'files': manifest, 'location_id': loc})
    assert response.status_code == 201, response.text
    identity = response.json()['upload_id']
    for index, value in enumerate(content.values()):
        assert client.put(f'/api/datasets/uploads/{identity}/files/{index}', content=value).status_code == 204
    response = client.post(f'/api/datasets/uploads/{identity}/complete', json={'names': ['人员']})
    assert response.status_code == 200, response.text
    return identity


def entries(client):
    response = client.get('/api/dataset-storage/datasets')
    assert response.status_code == 200, response.text
    return response.json()


def operation(client, identity):
    return client.get('/api/dataset-storage/operations/' + identity).json()


def node(factory, path):
    identity = uuid.uuid4().hex
    with factory() as db:
        db.add(ComputeNode(id=identity, name='远程', kind='agent', enabled=True, agent_version='2.1.0',
                           snapshot={'training': {'dataset_roots': [str(path)]}}))
        db.commit()
    return identity


def remote_tick(client, factory, root, agent, identity):
    cfg = {'node_id': identity, 'dataset_roots': [str(root)], 'training_cwd': str(root.parent),
           'training_state_dir': str(root.parent / ('state-' + identity)),
           'training_output_roots': [], 'allowed_training_roots': [str(root)],
           'training_lock_file': str(root.parent / 'workspace.lock'), 'enable_training': False}
    import threading
    result = client.post(f'/api/dataset-storage/agent/{identity}/poll').json()
    if result['operation']:
        agent.run_dataset_operation(cfg, result['operation'], threading.Event())


def pump(client, factory, root, agent, identity):
    for _ in range(10):
        store.worker.tick()
        remote_tick(client, factory, root, agent, identity)


def test_local_upload_change_default_move_stable_id_delete(env):
    client, factory, root, _ = env
    a = location(client, root / 'a'); finish_local()
    identity = upload(client, a); finish_local()
    assert operation(client, identity)['status'] == 'completed'
    entry = entries(client)[0]
    original_id = entry['id']
    assert entry['directory'] == str(root / 'a' / '测试集')
    assert services.validate_dataset_config(entry['config_path']).is_file()
    b = location(client, root / 'b'); finish_local()
    assert entries(client)[0]['id'] == original_id
    response = client.post(f'/api/dataset-storage/datasets/{original_id}/migrate', json={'location_id': b, 'name': '新集', 'remove_source': True})
    assert response.status_code == 202, response.text
    finish_local()
    op = operation(client, response.json()['id'])
    assert op['status'] == 'completed', op
    e = entries(client)[0]
    assert e['id'] == original_id
    assert e['directory'] == str(root / 'b' / '新集')
    assert e['history'][0]['path'] == entry['directory']
    assert not Path(entry['directory']).exists()
    response = client.delete(f'/api/dataset-storage/datasets/{e["id"]}')
    assert response.status_code == 202, response.text
    finish_local()
    assert operation(client, response.json()['id'])['status'] == 'completed'
    assert not Path(e['directory']).exists()
    assert entries(client) == []
    # A new publication at a deleted path gets a new identity, without unique conflicts.
    new_upload = upload(client, b, '新集'); finish_local()
    assert operation(client, new_upload)['status'] == 'completed'
    assert entries(client)[0]['id'] != original_id


def test_local_to_remote_and_remote_to_local(env):
    client, factory, root, agent = env
    a = location(client, root / 'local'); finish_local()
    remote_root = root / 'remote'
    remote_id = node(factory, remote_root)
    b = location(client, remote_root, remote_id)
    pump(client, factory, remote_root, agent, remote_id)
    identity = upload(client, b)
    pump(client, factory, remote_root, agent, remote_id)
    op = operation(client, identity)
    assert op['status'] == 'completed', op
    e = entries(client)[0]
    assert e['node_id'] == remote_id
    assert (remote_root / '测试集' / 'images/train/a.png').is_file()
    response = client.post(f'/api/dataset-storage/datasets/{e["id"]}/migrate', json={'location_id': a, 'remove_source': True})
    assert response.status_code == 202, response.text
    pump(client, factory, remote_root, agent, remote_id)
    op = operation(client, response.json()['id'])
    assert op['status'] == 'completed', op
    e2 = entries(client)[0]
    assert e2['id'] == e['id'] and e2['node_id'] == 'local'
    assert not (remote_root / '测试集').exists()
    assert (root / 'local' / '测试集' / 'data.yaml').is_file()


def test_reference_protection_and_external_registration(env):
    client, factory, root, _ = env
    a = location(client, root / 'a'); finish_local()
    folder = root / 'a' / 'outside'
    for p, data in images().items():
        target = folder / p
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (folder / 'data.yaml').write_text(yaml.safe_dump({'path': str(folder), 'train': 'images/train', 'val': 'images/val', 'names': ['人']}))
    response = client.post('/api/dataset-storage/datasets/register', json={'location_id': a, 'name': 'outside'})
    assert response.status_code == 202
    finish_local()
    assert operation(client, response.json()['id'])['status'] == 'completed'
    e = entries(client)[0]
    assert e['ownership'] == 'external'
    assert client.delete('/api/dataset-storage/datasets/' + e['id']).status_code == 409
    assert client.post('/api/dataset-storage/datasets/' + e['id'] + '/adopt', json={'confirm_path': str(folder)}).status_code == 200
    with factory() as db:
        db.add(TrainingTask(id='task', name='引用', status='draft', model_path='model.pt', data_path=str(folder / 'data.yaml'),
                            run_dir=str(root / 'output')))
        db.commit()
    assert client.get('/api/dataset-storage/datasets/' + e['id'] + '/delete-preview').json()['can_delete'] is False
    assert client.delete('/api/dataset-storage/datasets/' + e['id']).status_code == 409


def test_cancel_queued_and_changed_snapshot_rejected(env):
    client, factory, root, _ = env
    a = location(client, root / 'a'); finish_local()
    identity = upload(client, a)
    response = client.post('/api/dataset-storage/operations/' + identity + '/cancel')
    assert response.status_code == 200
    finish_local()
    assert not (root / 'a' / '测试集').exists()
    identity2 = upload(client, a, '数据'); finish_local()
    e = entries(client)[0]
    (Path(e['directory']) / 'labels/train/a.txt').write_text('0 0.4 0.4 0.1 0.1\n')
    response = client.delete('/api/dataset-storage/datasets/' + e['id'])
    assert response.status_code == 202
    finish_local()
    op = operation(client, response.json()['id'])
    assert op['status'] == 'failed' and '变化' in op['error'], op
    assert Path(e['directory']).exists()


def test_root_symlink_tar_and_hash_guards(tmp_path):
    root = tmp_path / 'root'; root.mkdir()
    link = tmp_path / 'link'; link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        core.check_root(str(link), [tmp_path])
    with pytest.raises(ValueError):
        core.remove_dataset(root, [root], 'fake')
    import tarfile
    tar = tmp_path / 'bad.tar'
    with tarfile.open(tar, 'w') as archive:
        item = tarfile.TarInfo('../escape'); item.size = 1
        archive.addfile(item, io.BytesIO(b'x'))
    with pytest.raises(ValueError):
        core.unpack(tar, tmp_path / 'out', {'files': [{'path': '../escape', 'size': 1, 'sha256': 'bad'}]})
    assert not (tmp_path / 'escape').exists()


def test_quarantine_crash_recovery_never_deletes_new_source(tmp_path, monkeypatch):
    root = tmp_path / 'root'; root.mkdir()
    source = root / 'data'; source.mkdir()
    (source / 'file.txt').write_text('old')
    sha = core.scan(source)['fingerprint']
    original = core.shutil.rmtree
    monkeypatch.setattr(core.shutil, 'rmtree', lambda p: (_ for _ in ()).throw(OSError('disk interrupted')))
    with pytest.raises(OSError):
        core.remove_dataset(source, [root], sha)
    source.mkdir(); (source / 'file.txt').write_text('new')
    monkeypatch.setattr(core.shutil, 'rmtree', original)
    assert core.remove_dataset(source, [root], sha)['recovered']
    assert (source / 'file.txt').read_text() == 'new'


def test_agent_embedded_core_matches_backend(env):
    *_, agent = env
    assert agent.DATASET_FILES_SOURCE == Path(core.__file__).read_text()


def test_upload_rejects_wrong_client_checksum(env):
    client, _, root, _ = env
    a = location(client, root / 'a'); finish_local()
    data = images()
    specs = [{'path': p, 'size': len(v), 'sha256': '0' * 64} for p, v in data.items()]
    identity = client.post('/api/datasets/uploads', json={'name': 'wrong', 'files': specs, 'location_id': a}).json()['upload_id']
    assert client.put(f'/api/datasets/uploads/{identity}/files/0', content=next(iter(data.values()))).status_code == 422


def test_agent_lost_finish_ack_replays_receipt_after_restart(env, monkeypatch):
    client, factory, root, agent = env
    remote_root = root / 'remote'
    remote_id = node(factory, remote_root)
    loc = location(client, remote_root, remote_id)
    pump(client, factory, remote_root, agent, remote_id)
    identity = upload(client, loc)
    finish_local()
    api = agent.dataset_api
    def drop_finish(config, path, method='GET', payload=None, timeout=120):
        if path.endswith('/finish') and payload.get('phase') == 'import':
            raise __import__('urllib.error', fromlist=['URLError']).URLError('lost before ACK')
        return api(config, path, method, payload, timeout)
    monkeypatch.setattr(agent, 'dataset_api', drop_finish)
    remote_tick(client, factory, remote_root, agent, remote_id)
    assert (remote_root / '测试集').is_dir()
    assert operation(client, identity)['status'] == 'committing'
    assert client.post('/api/dataset-storage/operations/' + identity + '/cancel').status_code == 409
    with factory() as db:
        op = db.get(DatasetOperation, identity)
        op.lease_until = None
        db.commit()
    monkeypatch.setattr(agent, 'dataset_api', api)
    remote_tick(client, factory, remote_root, agent, remote_id)
    assert operation(client, identity)['status'] == 'completed'
    assert len(entries(client)) == 1


def test_remote_to_remote_move_and_remote_deletion(env):
    client, factory, root, agent = env
    aroot, broot = root / 'remote-a', root / 'remote-b'
    aid, bid = node(factory, aroot), node(factory, broot)
    a, b = location(client, aroot, aid), location(client, broot, bid)
    for _ in range(8):
        store.worker.tick(); remote_tick(client, factory, aroot, agent, aid); remote_tick(client, factory, broot, agent, bid)
    identity = upload(client, a)
    pump(client, factory, aroot, agent, aid)
    e = entries(client)[0]
    result = client.post('/api/dataset-storage/datasets/' + e['id'] + '/migrate', json={'location_id': b, 'remove_source': True})
    assert result.status_code == 202
    for _ in range(12):
        store.worker.tick(); remote_tick(client, factory, aroot, agent, aid); remote_tick(client, factory, broot, agent, bid)
    assert operation(client, result.json()['id'])['status'] == 'completed'
    e2 = entries(client)[0]
    assert e2['id'] == e['id'] and e2['node_id'] == bid
    assert not (aroot / '测试集').exists()
    deletion = client.delete('/api/dataset-storage/datasets/' + e2['id'])
    assert deletion.status_code == 202
    pump(client, factory, broot, agent, bid)
    assert operation(client, deletion.json()['id'])['status'] == 'completed'
    assert not (broot / '测试集').exists()


def test_location_freeze_offline_retry_and_legacy_bootstrap(env):
    client, factory, root, agent = env
    legacy = root / 'legacy' / '旧数据'
    for p, data in images().items():
        f = legacy / p; f.parent.mkdir(parents=True, exist_ok=True); f.write_bytes(data)
    (legacy / 'data.yaml').write_text(yaml.safe_dump({'path': str(legacy), 'train': 'images/train', 'val': 'images/val', 'names': ['人']}))
    (legacy / uploads.MANAGED_MARKER).write_text(json.dumps({'version': 1, 'name': legacy.name, 'upload_id': uuid.uuid4().hex, 'classes': ['人']}))
    old = entries(client)[0]
    assert old['ownership'] == 'managed'
    assert client.delete('/api/dataset-storage/datasets/' + old['id']).status_code == 202
    finish_local()
    assert not legacy.exists()
    # Missing directory errors are visible and rechecking can explicitly create it.
    missing = root / 'missing'
    r = client.post('/api/dataset-storage/locations', json={'name': '空目录', 'path': str(missing)})
    finish_local()
    assert operation(client, r.json()['operation_id'])['status'] == 'failed'
    client.post('/api/dataset-storage/locations/' + r.json()['id'] + '/check', json={'create': True})
    finish_local()
    assert missing.is_dir()
    remote_root = root / 'remote'
    remote_id = node(factory, remote_root)
    loc = location(client, remote_root, remote_id)
    pump(client, factory, remote_root, agent, remote_id)
    identity = upload(client, loc)
    finish_local()  # No Agent polling: remains pending offline with original target.
    op = operation(client, identity)
    assert op['waiting_offline'] and op['destination'] == str(remote_root / '测试集')
    assert op['status'] == 'pending'
    pump(client, factory, remote_root, agent, remote_id)
    assert operation(client, identity)['status'] == 'completed'


def test_copy_preserves_referenced_source_and_starts_blocked_during_delete(env):
    client, factory, root, _ = env
    a, b = location(client, root / 'a'), location(client, root / 'b')
    finish_local()
    upload(client, a); finish_local()
    e = entries(client)[0]
    with factory() as db:
        task = TrainingTask(id='old', name='旧任务', status='stopped', model_path='base.pt', data_path=e['config_path'])
        db.add(task); db.commit()
    result = client.post('/api/dataset-storage/datasets/' + e['id'] + '/migrate', json={'location_id': b, 'remove_source': False})
    assert result.status_code == 202
    finish_local()
    assert operation(client, result.json()['id'])['status'] == 'completed'
    assert len(entries(client)) == 2
    assert Path(e['directory']).is_dir()
    with factory() as db:
        db.delete(db.get(TrainingTask, 'old')); db.commit()
    deletion = client.delete('/api/dataset-storage/datasets/' + e['id'])
    with factory() as db:
        assert store.mutation_pending(db, 'local')
        with pytest.raises(Exception) as exc:
            store.reject_training_mutation(db, 'local')
        assert exc.value.status_code == 409
    finish_local()
    assert operation(client, deletion.json()['id'])['status'] == 'completed'


def test_remote_file_queue_defers_new_training_claims(env):
    client, factory, root, agent = env
    remote_root = root / 'remote'
    remote_id = node(factory, remote_root)
    loc = location(client, remote_root, remote_id)
    pump(client, factory, remote_root, agent, remote_id)
    identity = upload(client, loc)
    finish_local()
    with factory() as db:
        assert store.node_dataset_file_work(db, remote_id)
        assert not store.mutation_pending(db, remote_id)  # Upload doesn't alter old references.
    pump(client, factory, remote_root, agent, remote_id)
    with factory() as db:
        assert not store.node_dataset_file_work(db, remote_id)
    assert operation(client, identity)['status'] == 'completed'


def test_remote_config_alias_change_cannot_bypass_reference_protection(env):
    client, factory, root, agent = env
    remote_root = root / 'remote'
    remote_id = node(factory, remote_root)
    loc = location(client, remote_root, remote_id)
    pump(client, factory, remote_root, agent, remote_id)
    upload(client, loc, 'A'); pump(client, factory, remote_root, agent, remote_id)
    upload(client, loc, 'B'); pump(client, factory, remote_root, agent, remote_id)
    a = next(e for e in entries(client) if e['name'] == 'A')
    b = next(e for e in entries(client) if e['name'] == 'B')
    with factory() as db:
        db.add(TrainingTask(id='ref-b', name='引用B', status='stopped', execution_node_id=remote_id,
                            model_path='base.pt', data_path=b['config_path']))
        db.commit()
    bconfig = Path(b['config_path'])
    original = bconfig.read_text()
    config = yaml.safe_load(original); config['path'] = a['directory']
    bconfig.write_text(yaml.safe_dump(config))
    response = client.delete('/api/dataset-storage/datasets/' + a['id'])
    assert response.status_code == 202
    pump(client, factory, remote_root, agent, remote_id)
    op = operation(client, response.json()['id'])
    assert op['status'] == 'failed' and '配置发生变化' in op['error'], op
    assert Path(a['directory']).is_dir()
    # Restoring the independently registered config makes its no-alias proof
    # valid again; retry doesn't need the source config after a deletion crash.
    bconfig.write_text(original)
    with factory() as db:
        op = db.get(DatasetOperation, response.json()['id'])
        assert [g['path'] for g in op.payload['config_guards']] == [b['config_path']]
    assert client.post('/api/dataset-storage/operations/' + response.json()['id'] + '/retry').status_code == 200
    pump(client, factory, remote_root, agent, remote_id)
    assert operation(client, response.json()['id'])['status'] == 'completed'
    assert not Path(a['directory']).exists() and Path(b['directory']).exists()


def test_operation_history_all_counts_pagination_snapshot_and_filters(env):
    from datetime import timedelta, timezone
    from app.models import utcnow
    client, factory, _, _ = env
    timestamp = utcnow() - timedelta(hours=1)
    with factory() as db:
        db.add_all([DatasetOperation(id=f'{i:032x}', node_id='local', kind='upload',
                    status='completed', payload={'name': f'数据集{i}'}, result={}, created_at=timestamp)
                    for i in range(205)])
        db.add(DatasetOperation(id='failed', node_id='local', kind='migrate', status='failed',
               payload={'name': '安全帽_100%', 'source_node': 'old-remote', 'target_node': 'new-remote'},
               result={}, created_at=timestamp - timedelta(hours=1)))
        db.add(DatasetOperation(id='active', node_id='old-remote', kind='delete', status='running',
               payload={'name': '旧数据集'}, result={}, created_at=timestamp - timedelta(hours=2)))
        db.commit()
    summary = client.get('/api/dataset-storage/operations/summary').json()
    assert summary['total'] == 207 and summary['active'] == 1 and summary['failed'] == 1
    assert {'local', 'old-remote', 'new-remote'} == {n['id'] for n in summary['nodes']}
    assert summary['latest_created_at'].endswith('Z') or summary['latest_created_at'].endswith('+00:00')
    first = client.get('/api/dataset-storage/operations/page').json()
    assert len(first['items']) == 20 and first['total'] == 207
    assert first['items'][0]['id'] == f'{204:032x}'
    assert first['items'][0]['created_at'].endswith(('Z', '+00:00'))
    with factory() as db:
        db.add(DatasetOperation(id='new', node_id='local', kind='check', status='pending', payload={}, result={}))
        db.commit()
    second = client.get('/api/dataset-storage/operations/page', params={'page': 2, 'before': first['snapshot']}).json()
    assert second['total'] == 207 and second['page'] == 2
    assert not {j['id'] for j in first['items']} & {j['id'] for j in second['items']}
    assert second['items'][0]['id'] == f'{184:032x}'
    for node_id in ('old-remote', 'new-remote'):
        response = client.get('/api/dataset-storage/operations/page', params={'node_id': node_id, 'kind': 'migrate', 'q': '_100%'}).json()
        assert response['total'] == 1 and response['items'][0]['id'] == 'failed'
    assert client.get('/api/dataset-storage/operations/page', params={'status': 'attention', 'before': first['snapshot']}).json()['total'] == 2
    assert client.get('/api/dataset-storage/operations/page', params={'q': '安全帽X100'}).json()['total'] == 0
    assert len(client.get('/api/dataset-storage/operations').json()) == 200
    assert client.get('/api/dataset-storage/operations/page', params={'page': 999}).json()['page'] == 11
    for params in ({'page': 0}, {'page_size': 101}, {'kind': 'bad'}, {'status': 'bad'}, {'before': 'bad'}):
        assert client.get('/api/dataset-storage/operations/page', params=params).status_code == 422
    offset = (timestamp + timedelta(minutes=1)).astimezone(timezone(timedelta(hours=8)))
    assert client.get('/api/dataset-storage/operations/page', params={'before': offset.isoformat()}).json()['total'] == 207


def test_dataset_deletable_state_blocks_external_and_unfinished_operations(env):
    client, factory, root, _ = env
    loc = location(client, root / 'datasets')
    finish_local()
    upload(client, loc)
    finish_local()
    item = entries(client)[0]
    assert item['can_delete'] and item['blocking_operation_id'] is None
    with factory() as db:
        entry = db.get(DatasetEntry, item['id'])
        entry.ownership = 'external'
        db.commit()
    assert not entries(client)[0]['can_delete']
    with factory() as db:
        entry = db.get(DatasetEntry, item['id'])
        entry.ownership = 'managed'
        db.add(DatasetOperation(id='blocking', node_id='local', kind='migrate', status='failed',
                               payload={'entry_id': entry.id}, result={}))
        db.commit()
    blocked = entries(client)[0]
    assert not blocked['can_delete'] and blocked['blocking_operation_id'] == 'blocking'
    assert not client.get(f"/api/dataset-storage/datasets/{item['id']}/delete-preview").json()['can_delete']
    assert client.post('/api/dataset-storage/operations/blocking/cancel').status_code == 200
    assert entries(client)[0]['can_delete']
    result = client.get('/api/dataset-storage/operations/page', params={'kind': 'check', 'q': 'datasets'}).json()
    assert result['total'] == 1 and result['items'][0]['name'] == 'datasets'


def test_history_summary_revision_ignores_progress_but_tracks_catalogue_and_completion(env, monkeypatch):
    from datetime import timedelta
    from app.models import utcnow
    client, factory, _, _ = env
    with factory() as db:
        db.add(DatasetOperation(id='work', node_id='local', kind='upload', status='running', payload={}, result={}))
        db.commit()
    def summary():
        return client.get('/api/dataset-storage/operations/summary').json()['revision']
    initial = summary()
    with factory() as db:
        op = db.get(DatasetOperation, 'work')
        op.transferred = 12345
        op.updated_at = utcnow() + timedelta(seconds=1)
        db.commit()
    assert summary() == initial
    with factory() as db:
        db.add(DatasetEntry(id='data', node_id='local', location_id='loc', name='snapshot', path='/test'))
        db.commit()
    catalogued = summary()
    assert catalogued != initial
    # Aggregate history does not invoke expensive dataset file/reference reads.
    monkeypatch.setattr(store, 'reference_rows', lambda *_: pytest.fail('summary must not scan references'))
    with factory() as db:
        op = db.get(DatasetOperation, 'work')
        op.status = 'completed'
        db.commit()
    assert summary() != catalogued


def test_hide_upload_history_keeps_result_files_and_reclamation(env):
    from datetime import timedelta
    from app.models import DatasetOperationHistoryRemoval, utcnow
    client, factory, root, _ = env
    loc = location(client, root / 'datasets')
    finish_local()
    identity = upload(client, loc)
    finish_local()
    dataset = entries(client)[0]
    actual = Path(dataset['directory'])
    before = operation(client, identity)
    request = {'ids': [identity], 'before': utcnow().isoformat()}
    preview = client.post('/api/dataset-storage/operations/history-removal-preview', json=request).json()
    assert preview['count'] == 1
    response = client.post('/api/dataset-storage/operations/history-removal', json={**request, 'fingerprint': preview['fingerprint']})
    assert response.status_code == 200, response.text
    assert response.json()['removed'] == 1
    assert operation(client, identity) == before
    assert (actual / 'data.yaml').is_file() and (actual / 'images/train/a.png').is_file()
    assert entries(client)[0]['id'] == dataset['id']
    assert identity not in {j['id'] for j in client.get('/api/dataset-storage/operations').json()}
    page = client.get('/api/dataset-storage/operations/page', params={'kind': 'upload'}).json()
    assert page['total'] == 0
    summary = client.get('/api/dataset-storage/operations/summary').json()
    assert summary['total'] == 1 and summary['counts'] == {'completed': 1}  # location check remains
    replay = client.post(f'/api/datasets/uploads/{identity}/complete', json={'names': ['人员']})
    assert replay.status_code == 200, replay.text
    with factory() as db:
        assert db.get(DatasetOperationHistoryRemoval, identity)
        op = db.get(DatasetOperation, identity)
        op.updated_at = utcnow() - timedelta(hours=25)
        db.commit()
    assert store.job_dir(identity).is_dir()
    finish_local()
    assert not store.job_dir(identity).exists()
    assert not (uploads.UPLOAD_ROOT / identity).exists()
    assert (actual / 'data.yaml').is_file()
    assert client.delete(f'/api/dataset-storage/operations/{identity}/history').status_code == 200


@pytest.mark.parametrize('status', ['pending', 'running', 'committing', 'failed'])
def test_history_removal_rejects_unfinished_and_never_partially_hides_selection(env, status):
    from app.models import DatasetOperationHistoryRemoval, utcnow
    client, factory, _, _ = env
    with factory() as db:
        db.add_all([DatasetOperation(id='protected', node_id='offline', kind='migrate', status=status,
                    payload={'phase': 'remove', 'published': True, 'commit_started': True}, result={}),
                    DatasetOperation(id='finished', node_id='local', kind='check', status='completed', payload={}, result={})])
        db.commit()
    body = {'ids': ['finished', 'protected'], 'before': utcnow().isoformat()}
    assert client.post('/api/dataset-storage/operations/history-removal-preview', json=body).status_code == 409
    assert client.post('/api/dataset-storage/operations/history-removal', json={**body, 'fingerprint': 'invalid'}).status_code == 409
    assert client.delete('/api/dataset-storage/operations/protected/history').status_code == 409
    with factory() as db:
        assert db.get(DatasetOperationHistoryRemoval, 'finished') is None
        assert db.get(DatasetOperationHistoryRemoval, 'protected') is None
        assert db.get(DatasetOperation, 'protected').status == status


def test_bulk_history_removal_across_pages_nodes_dates_and_offline_agent_replay(env):
    from datetime import timedelta
    from app.models import utcnow
    client, factory, _, _ = env
    cutoff = utcnow()
    with factory() as db:
        db.add_all([DatasetOperation(id=f'{i:032x}', node_id='local', kind='migrate', status='completed',
                    payload={'name': '安全帽_100%', 'source_node': 'offline-source', 'target_node': 'offline-target', 'phase': 'import'},
                    result={}, created_at=cutoff - timedelta(hours=1)) for i in range(205)])
        db.add_all([DatasetOperation(id='old', node_id='local', kind='migrate', status='completed', payload={'name': '安全帽_100%', 'target_node': 'offline-target'}, result={}, created_at=cutoff - timedelta(days=2)),
                    DatasetOperation(id='active', node_id='offline-target', kind='migrate', status='running', payload={'name': '安全帽_100%'}, result={}, created_at=cutoff - timedelta(minutes=30)),
                    DatasetOperation(id='failed', node_id='offline-target', kind='migrate', status='failed', payload={'name': '安全帽_100%'}, result={}, created_at=cutoff - timedelta(minutes=30)),
                    DatasetOperation(id='future', node_id='offline-target', kind='migrate', status='completed', payload={'name': '安全帽_100%'}, result={}, created_at=cutoff + timedelta(hours=1))])
        db.commit()
    body = {'node_id': 'offline-target', 'kind': 'migrate', 'q': '_100%', 'before': cutoff.isoformat(), 'after': (cutoff - timedelta(days=1)).isoformat()}
    preview = client.post('/api/dataset-storage/operations/history-removal-preview', json=body).json()
    assert preview['count'] == 205  # more than both one page and legacy limit
    assert client.post('/api/dataset-storage/operations/history-removal', json={**body, 'fingerprint': preview['fingerprint']}).json()['removed'] == 205
    assert client.get('/api/dataset-storage/operations/page', params={'node_id': 'offline-source'}).json()['total'] == 0
    page = client.get('/api/dataset-storage/operations/page', params={'page': 11, 'before': cutoff.isoformat()}).json()
    assert page['page'] == 1 and page['total'] == 3
    completed = f'{0:032x}'
    assert client.get(f'/api/dataset-storage/operations/{completed}').status_code == 200
    # A completed journal still acknowledges late remote completion after it is hidden.
    ack = client.post(f'/api/dataset-storage/agent/offline-target/{completed}/finish', json={'token': 'old-token', 'phase': 'import', 'result': {}})
    assert ack.status_code == 200, ack.text
    assert client.get('/api/dataset-storage/operations/summary').json()['active'] == 1
    assert client.post('/api/dataset-storage/operations/history-removal-preview', json=body).json()['count'] == 0
    assert client.get('/api/dataset-storage/operations/page', params={'after': cutoff.isoformat(), 'through': (cutoff - timedelta(days=1)).isoformat()}).status_code == 422


def test_history_removal_preview_detects_changed_candidates_and_requires_confirmation(env):
    from app.models import utcnow
    client, factory, _, _ = env
    with factory() as db:
        db.add_all([DatasetOperation(id='done', node_id='local', kind='check', status='completed', payload={}, result={}),
                    DatasetOperation(id='finishing', node_id='local', kind='check', status='running', payload={}, result={})])
        db.commit()
    body = {'before': utcnow().isoformat()}
    first = client.post('/api/dataset-storage/operations/history-removal-preview', json=body).json()
    assert first['count'] == 1
    assert client.post('/api/dataset-storage/operations/history-removal', json=body).status_code == 409
    with factory() as db:
        db.get(DatasetOperation, 'finishing').status = 'completed'
        db.commit()
    assert client.post('/api/dataset-storage/operations/history-removal', json={**body, 'fingerprint': first['fingerprint']}).status_code == 409
    assert client.get('/api/dataset-storage/operations/page').json()['total'] == 2
    second = client.post('/api/dataset-storage/operations/history-removal-preview', json=body).json()
    assert second['count'] == 2
    assert client.post('/api/dataset-storage/operations/history-removal', json={**body, 'fingerprint': second['fingerprint']}).json()['removed'] == 2
    for request in ({'ids': [], **body}, {'ids': ['missing'], **body}, {'after': 'bad', **body}):
        assert client.post('/api/dataset-storage/operations/history-removal-preview', json=request).status_code in (404, 422)


def test_old_database_gets_history_marker_table_without_changing_existing_jobs(env):
    from sqlalchemy import inspect
    from app.models import DatasetOperationHistoryRemoval
    client, factory, _, _ = env
    engine = factory.kw['bind']
    with factory() as db:
        db.add(DatasetOperation(id='old-cancelled', node_id='offline', kind='upload', status='cancelled',
                               payload={'phase': 'import', 'target_node': 'offline'}, result={}))
        db.commit()
    # Simulate a database from before history cleanup was introduced.
    DatasetOperationHistoryRemoval.__table__.drop(engine)
    Base.metadata.create_all(engine)
    assert inspect(engine).has_table('dataset_operation_history_removals')
    assert client.get('/api/dataset-storage/operations/page').json()['total'] == 1
    assert client.delete('/api/dataset-storage/operations/old-cancelled/history').status_code == 200
    assert client.get('/api/dataset-storage/operations/page').json()['total'] == 0
    # New sessions keep the visibility marker; cancellation is still durable.
    with factory() as db:
        assert db.get(DatasetOperationHistoryRemoval, 'old-cancelled')
        assert db.get(DatasetOperation, 'old-cancelled').status == 'cancelled'
    assert client.get('/api/dataset-storage/operations/old-cancelled').json()['status'] == 'cancelled'


def make_existing(path, *, config=None, bad_label=False):
    path.mkdir(parents=True, exist_ok=True)
    for relative, content in images().items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (path / 'data.yaml').write_text(yaml.safe_dump(config or {'path': '.', 'train': 'images/train', 'val': 'images/val', 'names': ['人员']}, allow_unicode=True))
    if bad_label:
        (path / 'labels/train/a.txt').write_text('9 0.5 0.5 0.2 0.2\n')


def start_scan(client, locations, **options):
    payload = {'request_id': uuid.uuid4().hex, 'location_ids': locations, **options}
    response = client.post('/api/dataset-storage/discovery/scans', json=payload)
    assert response.status_code == 202, response.text
    return response.json(), payload


def candidates(client, identity, **params):
    response = client.get(f'/api/dataset-storage/discovery/scans/{identity}/candidates', params=params)
    assert response.status_code == 200, response.text
    return response.json()


def test_discovery_nested_batch_local_replay_and_partial_failure(env):
    client, _, root, _ = env
    loc = location(client, root / 'data'); finish_local()
    make_existing(root / 'data/yan/v2')
    make_existing(root / 'data/bad-labels', bad_label=True)
    (root / 'data/incomplete/images').mkdir(parents=True)
    scan, spec = start_scan(client, [loc]); finish_local()
    assert client.post('/api/dataset-storage/discovery/scans', json=spec).json()['id'] == scan['id']
    result = candidates(client, scan['id'])
    assert result['counts'] == {'needs_attention': 1, 'ready': 2}
    chosen = [c['id'] for c in result['items'] if c['state'] == 'ready']
    payload = {'request_id': uuid.uuid4().hex, 'candidate_ids': chosen, 'candidate_tokens': {c['id']: c['selection_token'] for c in result['items'] if c['state'] == 'ready'}}
    url = f"/api/dataset-storage/discovery/scans/{scan['id']}/batches"
    response = client.post(url, json=payload)
    assert response.status_code == 202, response.text
    batch = response.json()
    assert batch['counts']['pending'] == 2
    assert client.post(url, json=payload).json()['id'] == batch['id']
    assert client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'relative_path': 'yan/v2'}).status_code == 409
    finish_local()
    batch = client.get(url).json()[0]
    assert batch['counts']['completed'] == 1 and batch['counts']['failed'] == 1
    assert batch['status'] == 'completed_with_errors'
    registered = entries(client)
    assert len(registered) == 1 and registered[0]['directory'] == str(root / 'data/yan/v2')
    assert registered[0]['ownership'] == 'external'
    assert not (root / 'data/yan/v2/.yolo-platform-dataset.json').exists()
    counts = candidates(client, scan['id'])['counts']
    assert counts['registered'] == 1 and counts['registering'] == 1
    failed_op = next(i['operation_id'] for i in batch['items'] if i['status'] == 'failed')
    assert client.post('/api/dataset-storage/operations/' + failed_op + '/cancel').status_code == 200
    assert candidates(client, scan['id'])['counts']['ready'] == 1
    assert client.post(url, json={**payload, 'ownership': 'managed', 'acknowledge': True}).status_code == 409


def test_discovery_overlap_limits_and_no_source_mutation(env):
    client, _, root, agent = env
    from app import dataset_discovery_files as portable
    assert agent.DATASET_DISCOVERY_SOURCE == Path(portable.__file__).read_text()
    outer = location(client, root / 'data'); finish_local()
    inner = location(client, root / 'data/nested'); finish_local()
    make_existing(root / 'data/nested/a'); make_existing(root / 'data/z'); make_existing(root / 'outside')
    (root / 'data/linked').symlink_to(root / 'outside', target_is_directory=True)
    before = {str(p): p.read_bytes() for p in (root / 'data').rglob('*') if p.is_file() and not p.is_symlink()}
    scan, _ = start_scan(client, [outer, inner]); finish_local()
    result = candidates(client, scan['id'])
    assert result['total'] == 2
    assert not any('/linked' in c['path'] for c in result['items'])
    assert candidates(client, scan['id'], page=50, page_size=1)['page'] == 2
    assert candidates(client, scan['id'], q='nested')['total'] == 1
    assert before == {str(p): p.read_bytes() for p in (root / 'data').rglob('*') if p.is_file() and not p.is_symlink()}
    limited, _ = start_scan(client, [outer], max_candidates=1); finish_local()
    summary = client.get('/api/dataset-storage/discovery/scans/' + limited['id']).json()
    assert summary['operations'][0]['progress']['limited'] is True
    assert candidates(client, limited['id'])['total'] == 1
    shallow, _ = start_scan(client, [outer], depth=0); finish_local()
    assert candidates(client, shallow['id'])['total'] == 0
    assert client.get('/api/dataset-storage/discovery/scans/' + shallow['id']).json()['operations'][0]['progress']['skipped_depth'] > 0


def test_discovery_remote_capability_chunks_and_batch_registration(env):
    client, factory, root, agent = env
    remote = node(factory, root / 'remote')
    loc = location(client, root / 'remote', remote)
    pump(client, factory, root / 'remote', agent, remote)
    assert client.post('/api/dataset-storage/discovery/scans', json={'request_id': uuid.uuid4().hex, 'location_ids': [loc]}).status_code == 409
    with factory() as db:
        n = db.get(ComputeNode, remote)
        n.agent_version, n.snapshot = '2.2.0', {'training': {'dataset_roots': [str(root / 'remote')], 'dataset_discovery': True}}
        db.commit()
    for i in range(23):
        make_existing(root / f'remote/版本{i}')
    scan, _ = start_scan(client, [loc]); pump(client, factory, root / 'remote', agent, remote)
    result = candidates(client, scan['id'], page_size=100)
    assert result['total'] == 23 and result['counts']['ready'] == 23
    op = client.get('/api/dataset-storage/discovery/scans/' + scan['id']).json()['operations'][0]
    assert op['progress']['found'] == 23 and not store.job_dir(op['id']).exists()
    payload = {'request_id': uuid.uuid4().hex, 'candidate_ids': [result['items'][0]['id']], 'candidate_tokens': {result['items'][0]['id']: result['items'][0]['selection_token']}}
    url = f"/api/dataset-storage/discovery/scans/{scan['id']}/batches"
    batch = client.post(url, json=payload).json()
    pump(client, factory, root / 'remote', agent, remote)
    assert client.get(url).json()[0]['counts']['completed'] == 1
    assert entries(client)[0]['node_id'] == remote
    with factory() as db:
        row = db.get(DatasetOperation, op['id'])
        receipt = {'token': row.lease, 'phase': 'discover', 'result': row.result}
    assert client.post(f"/api/dataset-storage/agent/{remote}/{op['id']}/finish", json=receipt).status_code == 200
    assert candidates(client, scan['id'])['total'] == 23
    assert len(entries(client)) == 1 and batch['id'] == payload['request_id']


def test_discovery_configuration_change_requires_rescan_and_batch_cancel(env):
    client, _, root, _ = env
    loc = location(client, root / 'data'); finish_local()
    make_existing(root / 'data/a'); make_existing(root / 'data/b')
    scan, _ = start_scan(client, [loc]); finish_local()
    chosen = candidates(client, scan['id'])['items']
    (root / 'data/a/data.yaml').write_text('train: images/train\nval: images/val\nnames: [changed]\n')
    url = f"/api/dataset-storage/discovery/scans/{scan['id']}/batches"
    first = client.post(url, json={'request_id': uuid.uuid4().hex, 'candidate_ids': [chosen[0]['id']], 'candidate_tokens': {chosen[0]['id']: chosen[0]['selection_token']}}).json()
    finish_local()
    failed = client.get(url).json()[0]['items'][0]
    assert failed['status'] == 'failed' and '扫描后发生变化' in failed['error']
    assert not entries(client)
    second = client.post(url, json={'request_id': uuid.uuid4().hex, 'candidate_ids': [chosen[1]['id']], 'candidate_tokens': {chosen[1]['id']: chosen[1]['selection_token']}}).json()
    stopped = client.post('/api/dataset-storage/discovery/batches/' + second['id'] + '/cancel').json()
    assert stopped['counts']['cancelled'] == 1
    finish_local(); assert not entries(client)
    assert client.get('/api/dataset-storage/operations/' + first['items'][0]['operation_id']).json()['can_cancel']


def test_discovery_cancel_generations_and_untrusted_remote_paths(env):
    client, factory, root, _ = env
    remote = node(factory, root / 'remote')
    from app.models import DatasetLocation
    with factory() as db:
        n = db.get(ComputeNode, remote)
        n.agent_version = '2.2.0'
        n.snapshot = {'training': {'dataset_roots': [str(root / 'remote')], 'dataset_discovery': True}}
        loc = DatasetLocation(id=uuid.uuid4().hex, node_id=remote, name='远程盘', path=str(root / 'remote'), resolved_path=str(root / 'remote'), enabled=True, status='ready')
        db.add(loc); db.commit(); loc_id = loc.id
    scan, _ = start_scan(client, [loc_id])
    op = client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation']
    url = f"/api/dataset-storage/agent/{remote}/{op['id']}/heartbeat"
    progress = {'generation': uuid.uuid4().hex, 'visited': 1, 'found': 1, 'candidates': [{'relative_path': '../escape', 'state': 'needs_attention'}]}
    payload = {'token': op['token'], 'phase': 'discover', 'result': progress}
    assert client.post(url, json=payload).status_code == 422
    progress['candidates'][0]['relative_path'] = 'a'
    assert client.post(url, json=payload).status_code == 200
    assert candidates(client, scan['id'])['counts']['scan_incomplete'] == 1
    progress.update(generation=uuid.uuid4().hex, found=0, candidates=[])
    assert client.post(url, json=payload).status_code == 200
    assert candidates(client, scan['id'])['total'] == 0
    assert client.post('/api/dataset-storage/discovery/scans/' + scan['id'] + '/cancel').status_code == 200
    assert client.post(url, json=payload).status_code == 409
    assert client.get('/api/dataset-storage/operations/page', params={'kind': 'discover'}).json()['total'] == 1
    assert client.delete('/api/dataset-storage/operations/' + op['id'] + '/history').status_code == 200
    assert client.get('/api/dataset-storage/operations/page', params={'kind': 'discover'}).json()['total'] == 0
    assert client.get('/api/dataset-storage/discovery/scans/' + scan['id']).status_code == 200


@pytest.mark.parametrize('relative', ['../outside', 'a/../../outside', '/absolute', 'a//b', 'a/./b', '.hidden', 'a\\b'])
def test_manual_nested_registration_rejects_unsafe_paths(env, relative):
    client, _, root, _ = env
    loc = location(client, root / 'data'); finish_local()
    assert client.post('/api/dataset-storage/datasets/register', json={'location_id': loc, 'relative_path': relative}).status_code == 422


def test_discovery_rescan_invalidates_selection_and_old_database_adds_tables(env):
    client, factory, root, _ = env
    from app.models import DatasetDiscoveryCandidate
    loc = location(client, root / 'data'); finish_local()
    make_existing(root / 'data/a')
    scan, _ = start_scan(client, [loc]); finish_local()
    chosen = candidates(client, scan['id'])['items'][0]
    # A restarted scan can reuse candidate IDs; the preview generation must still match.
    with factory() as db:
        candidate = db.get(DatasetDiscoveryCandidate, chosen['id'])
        candidate.generation = uuid.uuid4().hex
        op = db.get(DatasetOperation, candidate.operation_id)
        op.result = {**op.result, 'generation': candidate.generation}
        db.commit()
    result = client.post(f"/api/dataset-storage/discovery/scans/{scan['id']}/batches", json={
        'request_id': uuid.uuid4().hex, 'candidate_ids': [chosen['id']], 'candidate_tokens': {chosen['id']: chosen['selection_token']}})
    assert result.status_code == 202, result.text
    assert result.json()['counts']['skipped'] == 1
    assert '预览后发生变化' in result.json()['items'][0]['error']
    assert not entries(client)
    # Empty old schemas gain only the new tables; pre-existing legacy rows remain intact.
    from app.models import DatasetDiscoverySession, DatasetRegistrationBatch
    engine = create_engine(f'sqlite:///{root}/old-schema.db')
    new_tables = {m.__table__ for m in (DatasetDiscoveryCandidate, DatasetDiscoverySession, DatasetRegistrationBatch)}
    Base.metadata.create_all(engine, tables=[t for t in Base.metadata.sorted_tables if t not in new_tables])
    with engine.begin() as conn:
        conn.execute(DatasetOperation.__table__.insert().values(id='old-op', kind='check', node_id='local', status='completed', payload={}, result={}, transferred=0))
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        assert conn.scalar(select(DatasetOperation.__table__.c.id)) == 'old-op'
        assert conn.scalar(select(__import__('sqlalchemy').func.count()).select_from(DatasetDiscoveryCandidate)) == 0
    engine.dispose()


def test_discovery_marks_non_detection_and_requires_explicit_takeover(env):
    client, _, root, _ = env
    loc = location(client, root / 'data'); finish_local()
    make_existing(root / 'data/pose', config={'task': 'pose', 'train': 'images/train', 'val': 'images/val', 'names': ['person'], 'kpt_shape': [17, 3]})
    make_existing(root / 'data/detect')
    scan, _ = start_scan(client, [loc]); finish_local()
    result = candidates(client, scan['id'])
    assert result['counts']['unsupported'] == 1 and result['counts']['ready'] == 1
    chosen = next(c for c in result['items'] if c['state'] == 'ready')
    payload = {'request_id': uuid.uuid4().hex, 'candidate_ids': [chosen['id']], 'candidate_tokens': {chosen['id']: chosen['selection_token']}, 'ownership': 'managed'}
    url = f"/api/dataset-storage/discovery/scans/{scan['id']}/batches"
    assert client.post(url, json=payload).status_code == 422
    assert client.post(url, json={**payload, 'acknowledge': True}).status_code == 202
    finish_local()
    assert entries(client)[0]['ownership'] == 'managed'


def test_remote_discovery_waits_for_training_and_blocks_new_file_claims(env):
    client, factory, root, agent = env
    from app.models import TrainingRun
    remote = node(factory, root / 'remote')
    loc = location(client, root / 'remote', remote)
    pump(client, factory, root / 'remote', agent, remote)
    with factory() as db:
        n = db.get(ComputeNode, remote)
        n.agent_version = '2.2.0'
        n.snapshot = {'training': {'dataset_roots': [str(root / 'remote')], 'dataset_discovery': True}}
        task = TrainingTask(id='training-task', name='仍在训练', execution_node_id=remote, model_path='model.pt', data_path='/data.yaml', status='running')
        db.add(task); db.flush()
        db.add(TrainingRun(id='training-run', task_id=task.id, attempt_number=1, status='running', execution_node_id=remote, model_path=task.model_path, data_path=task.data_path))
        db.commit()
    scan, _ = start_scan(client, [loc])
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation'] is None
    with factory() as db:
        assert store.node_dataset_file_work(db, remote)
        db.get(TrainingRun, 'training-run').status = 'stopped'; db.commit()
    assert client.post(f'/api/dataset-storage/agent/{remote}/poll').json()['operation']['phase'] == 'discover'
    assert client.post('/api/dataset-storage/discovery/scans/' + scan['id'] + '/cancel').status_code == 200
    with factory() as db:
        assert not store.node_dataset_file_work(db, remote)
