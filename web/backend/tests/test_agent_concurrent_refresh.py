"""The negotiated read-only remote refresh may run beside training safely."""
import base64
import fcntl
import importlib.util
import json
import sys
import threading
from pathlib import Path

import pytest


@pytest.fixture
def agent():
    path = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('concurrent_refresh_agent_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config(tmp_path):
    root = tmp_path / 'datasets'
    root.mkdir()
    return {'center_url': 'http://center.example', 'node_id': 'a' * 32,
            'training_cwd': str(tmp_path), 'training_state_dir': str(tmp_path / '.training-state'),
            'training_lock_file': str(tmp_path / '.training.lock'),
            'allowed_training_roots': [str(root)], 'training_output_roots': [str(root)],
            'dataset_roots': [str(root)], 'dataset_refresh_io_pressure_pct': 0,
            'dataset_refresh_read_rate_bytes_per_second': 1024**3,
            'dataset_refresh_min_free_bytes': 0}


def test_old_agent_config_uses_refresh_budget_defaults(agent, tmp_path):
    path = tmp_path / 'agent-config.json'
    path.write_text(json.dumps({'center_url': 'http://center.example', 'node_id': 'a' * 32,
                                'token': 'ylp_' + 'x' * 32, 'enable_training': False}))
    path.chmod(0o600)
    loaded = agent.load_config(path)
    assert loaded['dataset_refresh_read_rate_bytes_per_second'] == 32 * 1024**2
    assert loaded['dataset_refresh_min_free_bytes'] == 512 * 1024**2
    assert loaded['dataset_refresh_io_pressure_pct'] == 25


def test_only_negotiated_refresh_export_bypasses_busy_filter(agent, monkeypatch):
    offered = {'phase': 'export', 'kind': 'refresh', 'read_only_during_training': True}
    paths = []

    def api(_config, path, *_args):
        paths.append(path)
        return {'operation': offered}

    monkeypatch.setattr(agent, 'dataset_api', api)
    assert agent.poll_dataset_operation({'node_id': 'a' * 32}, busy=True) is offered
    assert paths[-1].endswith('?lightweight_only=true')
    active = type('RunningWorker', (), {'is_alive': lambda self: True})()
    assert not agent.dataset_blocks_training(active, 'export', True)
    assert agent.dataset_blocks_training(active, 'export', False)
    for invalid in ({'phase': 'export', 'kind': 'register', 'read_only_during_training': True},
                    {'phase': 'export', 'kind': 'refresh'},
                    {'phase': 'remove', 'kind': 'refresh', 'read_only_during_training': True}):
        offered = invalid
        assert agent.poll_dataset_operation({'node_id': 'a' * 32}, busy=True) is None


def test_main_keeps_training_claims_open_during_read_only_refresh(agent, tmp_path, monkeypatch):
    cfg = {**config(tmp_path), 'enable_training': True, 'nvidia_smi_bin': 'nvidia-smi',
           'report_interval_seconds': 5}
    work = {'id': 'b' * 32, 'token': 'c' * 32, 'phase': 'export',
            'kind': 'refresh', 'read_only_during_training': True}
    calls = []

    class Event:
        def __init__(self):
            self.stopped = False
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
        def wait(self, _seconds):
            self.stopped = True
            return True

    class Worker:
        def __init__(self, **_kwargs):
            self.running = False
        def start(self):
            self.running = True
        def is_alive(self):
            return self.running
        def join(self):
            self.running = False

    monkeypatch.setattr(agent, 'load_config', lambda _path: cfg)
    monkeypatch.setattr(agent, 'acquire_agent_instance_lock', lambda _cfg: object())
    monkeypatch.setattr(agent, 'release_training_lock', lambda _lock: None)
    monkeypatch.setattr(agent, 'register_workspace', lambda _cfg: None)
    monkeypatch.setattr(agent, 'recover_active_workers', lambda *_args: None)
    monkeypatch.setattr(agent, 'collect', lambda *_args: {
        'training': {'ready': True, 'lock_occupied': True}, 'compute': {'gpus': []}})
    monkeypatch.setattr(agent, 'send', lambda *_args: (5, None))
    monkeypatch.setattr(agent, 'retry_pending_finals', lambda _cfg: True)
    monkeypatch.setattr(agent, 'load_active_runs', lambda _cfg: [{'task': {'gpu_assignment': ['gpu']}}])
    monkeypatch.setattr(agent, 'poll_dataset_operation', lambda *_args: work)
    monkeypatch.setattr(agent, 'api_json', lambda _cfg, path, *_args: (
        calls.append(path) or ({'operation': None} if path.endswith('/operations/poll') else {'task': None})))
    monkeypatch.setattr(agent.threading, 'Event', Event)
    monkeypatch.setattr(agent.threading, 'Thread', Worker)
    monkeypatch.setattr(sys, 'argv', ['monitor_agent.py', '--config', 'unused.json'])
    assert agent.main() == 0
    assert any(path.endswith('/agent/poll') for path in calls)


def test_refresh_export_completes_with_training_workspace_lock_held(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    source = Path(cfg['dataset_roots'][0]) / 'sample'
    source.mkdir()
    (source / 'data.yaml').write_text('train: images/train\nval: images/val\n')
    (source / 'image.bin').write_bytes(b'unchanged source' * 1024)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    chunks = bytearray()
    receipts = []

    def api(_config, path, method='GET', payload=None, **_kwargs):
        if path.endswith('/chunk'):
            assert method == 'POST'
            data = base64.b64decode(payload['data'])
            if data:
                assert payload['offset'] == len(chunks)
                chunks.extend(data)
            return {'offset': len(chunks)}
        if path.endswith('/finish'):
            receipts.append(payload)
        return {}

    monkeypatch.setattr(agent, 'dataset_api', api)
    lock = agent.acquire_training_lock(cfg)
    assert lock is not None
    dataset_lock = (Path(cfg['dataset_roots'][0]) / '.dataset-storage.lock').open('a')
    fcntl.flock(dataset_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    try:
        work = {'id': 'b' * 32, 'token': 'c' * 32, 'phase': 'export',
                'kind': 'refresh', 'read_only_during_training': True,
                'path': str(source), 'progress_run_id': 'd' * 32}
        agent.run_dataset_operation(cfg, work, threading.Event())
        assert len(receipts) == 1 and 'error' not in receipts[0]
        assert receipts[0]['result']['archive_size'] == len(chunks)
        assert {p.name: p.read_bytes() for p in source.iterdir()} == before
        assert agent.acquire_training_lock(cfg) is None
        # A lost finish acknowledgement can replay a saved receipt. It must
        # not claim to be a new refresh after the source has since changed.
        receipts.clear()
        (source / 'image.bin').write_bytes(b'changed source')
        agent.run_dataset_operation(cfg, work, threading.Event())
        assert len(receipts) == 1 and '源数据' in receipts[0]['error']
    finally:
        fcntl.flock(dataset_lock, fcntl.LOCK_UN)
        dataset_lock.close()
        agent.release_training_lock(lock)


def test_refresh_reader_respects_legacy_exclusive_dataset_lock(agent, tmp_path, monkeypatch):
    cfg = config(tmp_path)
    seen = []

    def api(_cfg, path, _method='GET', payload=None, **_kwargs):
        seen.append((path, payload))
        return {}

    monkeypatch.setattr(agent, 'dataset_api', api)
    lock_path = Path(cfg['dataset_roots'][0]) / '.dataset-storage.lock'
    with lock_path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        agent.run_dataset_operation(cfg, {
            'id': 'b' * 32, 'token': 'c' * 32, 'phase': 'export', 'kind': 'refresh',
            'read_only_during_training': True, 'progress_run_id': 'd' * 32,
            'path': str(Path(cfg['dataset_roots'][0]) / 'sample')}, threading.Event())
    assert any((body.get('progress') or {}).get('stage') == 'waiting_lock' for _, body in seen)
    assert not any(path.endswith('/finish') for path, _ in seen)


def test_pack_rejects_changed_source_and_removes_temporary(agent, tmp_path):
    source = tmp_path / 'dataset'
    source.mkdir()
    file = source / 'image.bin'
    file.write_bytes(b'A' * (1024 * 1024 + 10))
    manifest = agent.DATASET_FILES['scan'](source)
    archive = tmp_path / 'bundle.tar'

    def check():
        pass

    def progress(value):
        if value['stage'] == 'pack' and value['processed'] > 0:
            file.write_bytes(b'B' * (1024 * 1024 + 10))

    check.progress = progress
    with pytest.raises(ValueError, match='修改'):
        agent.DATASET_FILES['pack'](source, archive, check, manifest=manifest)
    assert not archive.exists() and not archive.with_suffix('.tmp').exists()


def test_allow_list_root_snapshot_ignores_its_agent_lock(agent, tmp_path):
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'data.yaml').write_text('train: images/train\nval: images/val\n')
    (root / '.dataset-storage.lock').touch()
    manifest = agent.DATASET_FILES['scan'](root)
    assert [item['path'] for item in manifest['files']] == ['data.yaml']
