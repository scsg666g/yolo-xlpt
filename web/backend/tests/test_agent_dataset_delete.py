"""Dataset removal safety using real temporary files and CPU-only child processes."""
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def agent(monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('delete_agent_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '_live_agent_configs', lambda: iter(()))
    return module


@pytest.fixture
def setup(tmp_path):
    roots = tmp_path / 'datasets'; roots.mkdir()
    source = roots / 'active'; source.mkdir()
    for name in ('train', 'val'):
        (source / 'images' / name).mkdir(parents=True)
        (source / 'images' / name / 'sample.jpg').write_bytes(b'not-decoded-by-evidence-reader')
    data = source / 'data.yaml'
    data.write_text(yaml.safe_dump({'path': str(source), 'train': 'images/train', 'val': 'images/val'}))
    run = tmp_path / 'runs' / 'attempt-001'; run.mkdir(parents=True)
    (run / 'args.yaml').write_text(yaml.safe_dump({'data': str(data), 'resume': False}))
    model = tmp_path / 'model.pt'; model.write_bytes(b'weight')
    target = roots / 'wrong-upload'; target.mkdir(); (target / 'file.txt').write_text('delete me')
    cfg = {'node_id': '1' * 32, 'dataset_roots': [str(roots)], 'training_cwd': str(tmp_path),
           'yolo_bin': sys.executable, 'training_state_dir': str(tmp_path / '.training-state'),
           'training_output_roots': [], 'allowed_training_roots': [str(tmp_path)],
           'training_lock_file': str(tmp_path / 'workspace.lock'),
           '_workspace_state_dir': str(tmp_path / '.workspace-state')}
    task = {'run_id': '2' * 32, 'task_id': 'test', 'task_type': 'detect',
            'data_path': str(data), 'model_path': str(model), 'run_dir': str(run),
            'log_path': str(tmp_path / 'run.log'), 'parameters': {}}
    return cfg, task, target


def work(agent, target, **changes):
    return {'id': 'a' * 32, 'token': 'b' * 32, 'kind': 'delete', 'phase': 'remove',
            'concurrent_delete': True, 'path': str(target),
            'fingerprint': agent.DATASET_FILES['scan'](target)['fingerprint'], **changes}


@pytest.fixture
def live(agent, setup):
    cfg, task, target = setup
    lock = Path(cfg['training_lock_file']).open('a+')
    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
    command = [sys.executable, '-c', 'import time; time.sleep(60)', 'data=' + task['data_path']]
    child = subprocess.Popen(command, pass_fds=(lock.fileno(),))
    agent.save_active_run(cfg, task, 'running', command=command, identity=agent.process_identity(child.pid))
    try:
        yield cfg, task, target, child
    finally:
        child.terminate()
        child.wait(timeout=5)
        lock.close()


def capture(agent, monkeypatch):
    calls = []
    def api(_cfg, path, method='GET', payload=None, **kwargs):
        calls.append((path, payload))
        return {'ok': True}
    monkeypatch.setattr(agent, 'dataset_api', api)
    return calls


def test_delete_unrelated_dataset_preserves_running_process_and_replay(agent, live, monkeypatch):
    cfg, task, target, child = live
    calls = capture(agent, monkeypatch)
    job = work(agent, target)
    before = Path(task['data_path']).read_bytes()
    agent.run_dataset_operation(cfg, job, threading.Event())
    assert not target.exists()
    assert child.poll() is None
    assert Path(task['data_path']).read_bytes() == before
    assert calls[-1][1]['result']['deleted'] is True
    target.mkdir(); (target / 'new.txt').write_text('new upload')
    monkeypatch.setattr(agent, 'verify_dataset_delete', lambda *_a, **_k: pytest.fail('receipt must only ACK'))
    agent.run_dataset_operation(cfg, job, threading.Event())
    assert (target / 'new.txt').read_text() == 'new upload'
    assert child.poll() is None


@pytest.mark.parametrize('key', ['train', 'val', 'test', 'minival'])
def test_yaml_split_cross_reference_is_protected(agent, live, key):
    cfg, task, target, _ = live
    p = Path(task['data_path']); data = yaml.safe_load(p.read_text()); data[key] = str(target)
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='重叠'):
        agent.verify_dataset_delete(cfg, work(agent, target))


def test_manifest_image_cross_reference_is_protected(agent, live):
    cfg, task, target, _ = live
    image = target / 'wrong.jpg'; image.write_bytes(b'image')
    source = Path(task['data_path']).parent
    manifest = source / 'train.txt'; manifest.write_text(str(image) + '\n')
    p = Path(task['data_path']); data = yaml.safe_load(p.read_text()); data['train'] = 'train.txt'
    p.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='重叠'):
        agent.verify_dataset_delete(cfg, work(agent, target))


@pytest.mark.parametrize('kind', ['nested_image', 'label', 'manifest_label', 'classify'])
def test_symlink_inputs_fail_closed(agent, live, kind):
    cfg, task, target, _ = live
    source = Path(task['data_path']).parent
    if kind == 'nested_image':
        (source / 'images/train/link').symlink_to(target, target_is_directory=True)
    elif kind in ('label', 'manifest_label'):
        (source / 'labels').symlink_to(target, target_is_directory=True)
        if kind == 'manifest_label':
            (source / 'train.txt').write_text(str(source / 'images/train/sample.jpg') + '\n')
            p = Path(task['data_path']); data = yaml.safe_load(p.read_text()); data['train'] = 'train.txt'
            p.write_text(yaml.safe_dump(data))
    else:
        (source / 'train').mkdir(); (source / 'train/class').symlink_to(target, target_is_directory=True)
        task['task_type'] = 'classify'; task['data_path'] = str(source)
        (Path(task['run_dir']) / 'args.yaml').write_text(yaml.safe_dump({'data': str(source)}))
        agent.save_active_run(cfg, task, 'running', command=[], identity=agent.process_identity(os.getpid()))
    with pytest.raises(ValueError):
        agent.verify_dataset_delete(cfg, work(agent, target))


def test_resume_uses_source_args_and_refuses_missing_evidence(agent, live):
    cfg, task, target, _ = live
    previous = Path(cfg['training_cwd']) / 'runs/previous'
    (previous / 'weights').mkdir(parents=True)
    checkpoint = previous / 'weights/last.pt'; checkpoint.write_bytes(b'checkpoint')
    (previous / 'args.yaml').write_text(yaml.safe_dump({'data': str(target / 'data.yaml')}))
    task['checkpoint_path'] = str(checkpoint)
    states = agent.load_active_runs(cfg); states[0]['task'] = task
    with pytest.raises(ValueError):
        agent.dataset_delete_evidence(cfg, states)
    (target / 'images/train').mkdir(parents=True); (target / 'images/val').mkdir()
    (target / 'data.yaml').write_text(yaml.safe_dump({'path': str(target), 'train': 'images/train', 'val': 'images/val'}))
    agent.save_active_run(cfg, task, 'running', command=states[0]['command'], identity=states[0]['process'])
    with pytest.raises(ValueError, match='重叠'):
        agent.verify_dataset_delete(cfg, work(agent, target))


def test_command_resume_source_is_checked(agent, live):
    cfg, task, target, _ = live
    source = target / 'old/weights'; source.mkdir(parents=True)
    checkpoint = source / 'last.pt'; checkpoint.write_bytes(b'checkpoint')
    states = agent.load_active_runs(cfg)
    states[0]['command'] += ['resume=' + str(checkpoint)]
    with pytest.raises(ValueError):
        agent.dataset_delete_evidence(cfg, states)


def test_commit_rechecks_mutated_running_args(agent, live, monkeypatch):
    cfg, task, target, child = live
    calls = capture(agent, monkeypatch)
    original = agent.DATASET_FILES['remove_dataset']
    def changed(*args):
        p = Path(task['run_dir']) / 'args.yaml'
        p.write_text(p.read_text() + '\n# changed during deletion verification\n')
        return original(*args)
    monkeypatch.setitem(agent.DATASET_FILES, 'remove_dataset', changed)
    agent.run_dataset_operation(cfg, work(agent, target), threading.Event())
    assert target.is_dir() and child.poll() is None
    assert '发生变化' in calls[-1][1]['error']


@pytest.mark.parametrize('mode', [fcntl.LOCK_SH, fcntl.LOCK_EX])
def test_unknown_external_workspace_lock_blocks_deletion(agent, setup, mode):
    cfg, _, target = setup
    command = [sys.executable, '-c', 'import fcntl,sys,time; f=open(sys.argv[1],"a+"); '
               'fcntl.flock(f,int(sys.argv[2])); print("ready",flush=True); time.sleep(60)',
               cfg['training_lock_file'], str(mode)]
    child = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'ready'
        with pytest.raises(ValueError, match='未知外部进程'):
            agent.verify_dataset_delete(cfg, work(agent, target))
        assert target.is_dir()
    finally:
        child.terminate(); child.wait(timeout=5); child.stdout.close()


def test_other_or_unreadable_agent_config_blocks_deletion(agent, setup, monkeypatch):
    cfg, _, target = setup
    for other in (None, {'enable_training': True}):
        monkeypatch.setattr(agent, '_live_agent_configs', lambda: iter([(123456, other)]))
        with pytest.raises(ValueError, match='其他训练 Agent'):
            agent.verify_dataset_delete(cfg, work(agent, target))


def test_launch_gate_blocks_start_and_close_releases_it(agent, setup):
    cfg, task, _ = setup
    gate = agent.dataset_launch_gate(cfg, exclusive=True)
    try:
        assert agent.acquire_task_training_lock(cfg, task) is None
    finally:
        gate.close()
    lock = agent.acquire_task_training_lock(cfg, task)
    assert lock is not None
    assert agent.dataset_launch_gate(cfg, exclusive=True) is None
    lock.close()
    gate = agent.dataset_launch_gate(cfg, exclusive=True)
    assert gate is not None
    gate.close()


@pytest.mark.parametrize('change', [{}, {'concurrent_delete': 'true'}, {'kind': 'migrate', 'concurrent_delete': True},
                                     {'phase': 'export', 'concurrent_delete': True}])
def test_old_or_unnegotiated_heavy_work_is_not_polled_busy(agent, setup, monkeypatch, change):
    cfg, _, target = setup
    job = work(agent, target); job.pop('concurrent_delete'); job.update(change)
    monkeypatch.setattr(agent, 'dataset_api', lambda *_a, **_k: {'operation': job})
    assert agent.poll_dataset_operation(cfg, busy=True) is None


def test_negotiated_delete_is_polled_busy_and_pauses_new_claim(agent, setup, monkeypatch):
    cfg, _, target = setup
    job = work(agent, target)
    monkeypatch.setattr(agent, 'dataset_api', lambda *_a, **_k: {'operation': job})
    assert agent.poll_dataset_operation(cfg, busy=True) == job
    class Worker:
        def is_alive(self): return True
    assert agent.dataset_blocks_training(Worker(), 'remove') is True


def test_training_python_dependency_failure_is_safe(agent, live, monkeypatch):
    cfg, _, _, _ = live
    monkeypatch.setattr(agent.subprocess, 'run', lambda *_a, **_k: subprocess.CompletedProcess([], 1, '', 'missing yaml'))
    with pytest.raises(ValueError, match='无法完整核验'):
        agent.dataset_delete_evidence(cfg, agent.load_active_runs(cfg))


def test_historical_running_state_without_args_is_preserved_and_refused(agent, live):
    cfg, task, target, child = live
    (Path(task['run_dir']) / 'args.yaml').unlink()
    with pytest.raises(ValueError, match='无法完整核验'):
        agent.verify_dataset_delete(cfg, work(agent, target))
    assert target.is_dir() and child.poll() is None
    assert agent.load_active_runs(cfg)[0]['run_id'] == task['run_id']


def test_nul_in_actual_dataset_path_is_rejected(agent, live):
    cfg, _, _, _ = live
    states = agent.load_active_runs(cfg)
    states[0]['task']['data_path'] += chr(0) + 'ignored.yaml'
    with pytest.raises(ValueError, match='无法完整核验'):
        agent.dataset_delete_evidence(cfg, states)


def test_changed_center_config_guard_blocks_deletion(agent, live):
    cfg, task, target, _ = live
    p = Path(task['data_path'])
    guard = {'path': str(p), 'sha256': agent.DATASET_FILES['digest'](p)}
    p.write_text(p.read_text() + '\n# updated after center check\n')
    with pytest.raises(ValueError, match='同节点数据配置发生变化'):
        agent.verify_dataset_delete(cfg, work(agent, target, config_guards=[guard]))
    assert target.is_dir()


def test_launched_identity_waits_for_readable_stable_matching_exec(agent, monkeypatch):
    identities = iter([None, {'pid': 123, 'cmdline_sha256': 'before-exec'},
                       {'pid': 123, 'cmdline_sha256': 'after-exec'},
                       {'pid': 123, 'cmdline_sha256': 'after-exec'}])
    monkeypatch.setattr(agent, 'process_identity', lambda _pid: next(identities))
    monkeypatch.setattr(agent, '_process_argv', lambda _pid: ['python', '-c', 'training'])
    monkeypatch.setattr(agent.time, 'sleep', lambda _seconds: None)
    class Process:
        pid = 123
        def poll(self): return None
    assert agent.launched_process_identity(Process(), ['python', '-c', 'training']) == {
        'pid': 123, 'cmdline_sha256': 'after-exec'}


def test_launched_identity_does_not_accept_unrelated_command(agent, monkeypatch):
    monkeypatch.setattr(agent, 'process_identity', lambda _pid: {'pid': 123})
    monkeypatch.setattr(agent, '_process_argv', lambda _pid: ['other-process'])
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(agent.time, 'monotonic', lambda: next(ticks))
    class Process:
        pid = 123
        def poll(self): return None
    assert agent.launched_process_identity(Process(), ['python', '-c', 'training']) is None


def test_launched_identity_returns_without_waiting_for_exited_child(agent, monkeypatch):
    monkeypatch.setattr(agent, 'process_identity', lambda _pid: None)
    monkeypatch.setattr(agent.time, 'sleep', lambda _seconds: pytest.fail('exited child must not wait'))
    class Process:
        pid = 123
        def poll(self): return 0
    assert agent.launched_process_identity(Process(), ['python', '-c', 'training']) is None


def test_explicit_relative_yaml_root_with_cwd_and_yaml_siblings_is_refused(agent, live):
    cfg, task, target, child = live
    source = Path(task['data_path']).parent
    # Both are valid trees, but YOLO selects cwd/shared before DATASETS_DIR.
    yaml_candidate = source / 'shared'
    cwd_candidate = Path(cfg['training_cwd']) / 'shared'
    for root in (yaml_candidate, cwd_candidate):
        for split in ('train', 'val'):
            (root / 'images' / split).mkdir(parents=True)
    Path(task['data_path']).write_text(yaml.safe_dump({
        'path': 'shared', 'train': 'images/train', 'val': 'images/val'}))
    # The wrongly inferred YAML-relative root must not permit deletion of cwd/shared.
    cfg['dataset_roots'].append(str(Path(cfg['training_cwd'])))
    with pytest.raises(ValueError, match='无法完整核验'):
        agent.verify_dataset_delete(cfg, work(agent, cwd_candidate))
    assert cwd_candidate.is_dir() and child.poll() is None


@pytest.mark.parametrize('root_kind', ['missing', 'none', 'empty', 'absolute'])
def test_implicit_and_absolute_yaml_roots_remain_verifiable(agent, live, root_kind):
    cfg, task, target, child = live
    data = {'train': 'images/train', 'val': 'images/val'}
    if root_kind != 'missing':
        data['path'] = {'none': None, 'empty': '', 'absolute': str(Path(task['data_path']).parent)}[root_kind]
    Path(task['data_path']).write_text(yaml.safe_dump(data))
    evidence = agent.verify_dataset_delete(cfg, work(agent, target))
    assert str(Path(task['data_path']).parent / 'images/train') in evidence['paths']
    assert child.poll() is None
