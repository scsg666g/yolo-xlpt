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

from app import dataset_files as core, dataset_store as store, dataset_uploads as uploads, services
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
