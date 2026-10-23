"""Remote execution regressions, using a small dummy process rather than GPU training."""
import base64
import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest


@pytest.fixture
def agent():
    script = Path(__file__).resolve().parents[2] / "agent/monitor_agent.py"
    spec = importlib.util.spec_from_file_location("training_agent_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def training(tmp_path):
    root = tmp_path / "training"
    root.mkdir()
    (root / "model.pt").write_bytes(b"model")
    (root / "data.yaml").write_text("train: images/train\nval: images/val\n")
    (root / "runs").mkdir()
    run_dir = root / "runs/detect/task-id/attempt-001"
    task = {"run_id": "a" * 32, "task_id": "task-id", "attempt_number": 1,
            "task_type": "detect", "model_path": str(root / "model.pt"),
            "data_path": str(root / "data.yaml"), "run_dir": str(run_dir),
            "log_path": str(root / "runs/.platform-logs/20261012/task-id/attempt-001.log"),
            "parameters": {"epochs": 3, "batch": 2, "imgsz": 640, "device": "0,1",
                           "workers": 0, "optimizer": "auto", "remote_output_root": str(root / "runs")}}
    config = {"node_id": "b" * 32, "allowed_training_roots": [str(root)],
              "training_output_roots": [str(root / "runs")], "yolo_bin": sys.executable,
              "training_cwd": str(root), "training_state_dir": str(tmp_path / ".training-state"),
              "_workspace_state_dir": str(tmp_path / ".workspace-state")}
    return config, task


def test_real_directory_cleanup_keeps_external_log(agent, training, monkeypatch):
    config, task = training
    build = agent.build_training_command

    def dummy_command(config, task):
        build(config, task)
        # The actual DDP startup deletes and recreates save_dir.
        return [sys.executable, "-c",
                "import pathlib,shutil,sys; p=pathlib.Path(sys.argv[1]); "
                "shutil.rmtree(p); p.mkdir(); "
                "(p/'results.csv').write_text('epoch,time,score\\n1,1,0.5\\n'); "
                "print('DDP output after directory cleanup', flush=True)", task["run_dir"]]

    monkeypatch.setattr(agent, "build_training_command", dummy_command)
    updates = []

    def api(_config, _path, method="GET", payload=None):
        if payload is not None:
            updates.append(payload)
            if len(updates) == 1:
                raise URLError("initial report disconnected")
        return {}

    monkeypatch.setattr(agent, "api_json", api)
    agent.run_remote_task(config, task, threading.Event())
    assert "DDP output after directory cleanup" in Path(task["log_path"]).read_text()
    assert updates[-1]["status"] == "completed" and updates[-1]["exit_code"] == 0
    assert updates[-1]["command"]
    assert not list(Path(config["training_state_dir"]).glob("*.json"))


def test_remote_detect_command_includes_training_trace_runner(agent, training, tmp_path):
    config, task = training
    config["enable_training_trace"] = True
    config["yolo_bin"] = str(tmp_path / "yolo")
    Path(config["yolo_bin"]).write_text("#!/bin/sh\n")
    Path(config["yolo_bin"]).chmod(0o700)
    command = [config["yolo_bin"], "detect", "train", "model=/tmp/model.pt", "data=/tmp/data.yaml"]
    wrapped = agent.prepare_trace_command(config, task, command)
    assert wrapped[1].endswith("training_trace_runner.py")
    assert "--trace-dir" in wrapped and str(Path(task["run_dir"]) / "training-trace") in wrapped
    assert wrapped[-len(command):] == command
    assert (Path(config["training_state_dir"]) / "training_trace_runner.py").is_file()


def test_remote_trace_runner_rewrite_is_atomic_and_keeps_complete_file(agent, training, monkeypatch):
    config, task = training
    config["enable_training_trace"] = True
    config["yolo_bin"] = str(Path(config["training_cwd"]) / "fake-yolo")
    Path(config["yolo_bin"]).write_text("#!/bin/sh\n")
    Path(config["yolo_bin"]).chmod(0o700)
    writes = []
    original_replace = Path.replace

    def track_replace(path, target):
        writes.append(path.read_text(encoding="utf-8"))
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", track_replace)
    for _ in range(2):
        agent.prepare_trace_command(config, task, [config["yolo_bin"], "detect", "train"])
    runner = Path(config["training_state_dir"]) / "training_trace_runner.py"
    assert runner.read_text(encoding="utf-8") == agent.REMOTE_TRACE_RUNNER
    assert writes == [agent.REMOTE_TRACE_RUNNER]
    assert not list(runner.parent.glob("*.tmp"))


def test_gpu_assignment_validates_inventory_and_maps_devices(agent, training, monkeypatch):
    config, task = training
    task["parameters"]["device"] = "0,2"
    task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-a"}, {"index": 2, "uuid": "GPU-c"}]
    monkeypatch.setattr(agent, "query", lambda *_args: "0, GPU-a\n1, GPU-b\n2, GPU-c\n")

    assignment = agent.validate_gpu_assignment(config, task)
    environment = agent.training_launch_environment(config, task)
    command = ["yolo", "detect", "train", "device=0,2"]
    mapped = agent.training_command_for_assignment(command, task)

    assert assignment == task["gpu_assignment"]
    assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-a,GPU-c"
    assert mapped[-1] == "device=0,1"
    task["gpu_assignment"][0]["uuid"] = "GPU-wrong"
    with pytest.raises(ValueError, match="UUID"):
        agent.validate_gpu_assignment(config, task)


def test_gpu_assignment_uses_per_uuid_locks_and_conflicts_with_legacy_lock(agent, training, monkeypatch):
    config, task = training
    task["parameters"]["device"] = "0"
    task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-a"}]
    monkeypatch.setattr(agent, "query", lambda *_args: "0, GPU-a\n")

    parallel = agent.acquire_task_training_lock(config, task)
    assert parallel is not None
    try:
        assert agent.acquire_task_training_lock(config, task) is None
        # Existing pre-2.0 Agents take the exclusive legacy gate and must not
        # overlap with a 2.0 GPU-scoped run.
        assert agent.acquire_training_lock(config) is None
    finally:
        agent.release_training_lock(parallel)

    legacy = agent.acquire_training_lock(config)
    assert legacy is not None
    try:
        assert agent.acquire_task_training_lock(config, task) is None
    finally:
        agent.release_training_lock(legacy)


def test_different_gpu_uuid_locks_can_be_held_at_the_same_time(agent, training, monkeypatch):
    config, first_task = training
    first_task["parameters"]["device"] = "0"
    first_task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-A"}]
    second_task = {**first_task, "run_id": "b" * 32, "parameters": {
        **first_task["parameters"], "device": "1"},
        "gpu_assignment": [{"index": 1, "uuid": "GPU-B"}]}
    monkeypatch.setattr(agent, "query", lambda *_args: "0, GPU-A\n1, GPU-B\n")

    first = agent.acquire_task_training_lock(config, first_task)
    second = agent.acquire_task_training_lock(config, second_task)
    try:
        assert first is not None and second is not None
        assert first.filenos() != second.filenos()
    finally:
        agent.release_training_lock(second)
        agent.release_training_lock(first)


def test_recovery_detects_child_holding_gpu_and_workspace_lock_bundle(agent, training, monkeypatch):
    config, task = training
    task["parameters"]["device"] = "0"
    task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-A"}]
    monkeypatch.setattr(agent, "query", lambda *_args: "0, GPU-A\n")
    lock = agent.acquire_task_training_lock(config, task)
    assert lock is not None
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               pass_fds=lock.filenos())
    lock.close()
    try:
        assert agent.process_holds_task_locks(config, task, process.pid)
    finally:
        process.terminate()
        process.wait(timeout=5)
    recovered = agent.acquire_task_training_lock(config, task)
    assert recovered is not None
    agent.release_training_lock(recovered)


def test_recovery_recognizes_legacy_process_holding_only_workspace_lock(agent, training):
    config, task = training
    lock = agent.acquire_training_lock(config)
    assert lock is not None
    workspace_fd = lock.handles[0].fileno()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               pass_fds=(workspace_fd,))
    lock.close()
    try:
        assert "gpu_assignment" not in task
        assert agent.process_holds_task_locks(config, task, process.pid)
    finally:
        process.terminate()
        process.wait(timeout=5)
    recovered = agent.acquire_training_lock(config)
    assert recovered is not None
    agent.release_training_lock(recovered)


def test_gpu_assignment_rejects_malformed_or_unverified_payload(agent, training, monkeypatch):
    config, task = training
    task["parameters"]["device"] = "0,1"
    task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-a"}]
    with pytest.raises(ValueError, match="数量"):
        agent.normalize_gpu_assignment(task)

    task["gpu_assignment"] = [{"index": 0, "uuid": "GPU-a"}, {"index": 1, "uuid": "GPU-b"}]
    monkeypatch.setattr(agent, "query", lambda *_args: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(ValueError, match="核对"):
        agent.validate_gpu_assignment(config, task)


def test_multiple_active_runs_load_independently(agent, training):
    config, first = training
    second = {**first, "run_id": "b" * 32, "task_id": "task-2"}
    agent.save_active_run(config, first, "running")
    agent.save_active_run(config, second, "running")

    states = agent.load_active_runs(config)

    assert {state["run_id"] for state in states} == {first["run_id"], second["run_id"]}
    with pytest.raises(ValueError, match="多个活动训练状态"):
        agent.load_active_run(config)


def test_training_environment_includes_only_valid_owner_markers(agent):
    config = {"node_id": "b" * 32}
    task = {"platform_instance_id": "c" * 32, "run_id": "a" * 32, "task_id": "d" * 32}

    environment = agent.training_process_environment(config, task)

    assert environment["YOLO_FORGE_INSTANCE_ID"] == "c" * 32
    assert environment["YOLO_FORGE_NODE_ID"] == "b" * 32
    assert environment["YOLO_FORGE_RUN_ID"] == "a" * 32
    assert environment["YOLO_FORGE_TASK_ID"] == "d" * 32
    assert "YOLO_FORGE_CLAIM_LEASE" not in environment


def test_gpu_process_owner_uses_run_marker_and_verified_ancestor(agent, monkeypatch):
    parents = {501: 500, 500: 400, 400: 1, 1: 0}
    owner = {
        "YOLO_FORGE_INSTANCE_ID": "c" * 32,
        "YOLO_FORGE_NODE_ID": "b" * 32,
        "YOLO_FORGE_RUN_ID": "a" * 32,
        "YOLO_FORGE_TASK_ID": "d" * 32,
    }
    monkeypatch.setattr(agent, "process_parent", lambda pid: parents.get(pid))
    monkeypatch.setattr(agent, "process_environment", lambda pid: owner if pid == 501 else {})

    marker_hint = agent.process_owner_hint(501, {})
    monkeypatch.setattr(agent, "process_environment", lambda _pid: {})
    ancestor_hint = agent.process_owner_hint(501, {400: {
        "instance_id": "c" * 32, "node_id": "b" * 32, "run_id": "a" * 32, "task_id": "d" * 32,
    }})

    assert marker_hint["match_method"] == "runtime_marker"
    assert marker_hint["scan_complete"] is True
    assert marker_hint["run_id"] == "a" * 32
    assert ancestor_hint["match_method"] == "ancestor"
    assert ancestor_hint["scan_complete"] is True
    assert ancestor_hint["run_id"] == "a" * 32


def test_gpu_process_owner_marks_complete_unmatched_process_as_unmanaged(agent, monkeypatch):
    parents = {501: 500, 500: 400, 400: 1, 1: 0}
    monkeypatch.setattr(agent, "process_parent", lambda pid: parents.get(pid))
    monkeypatch.setattr(agent, "process_environment", lambda _pid: {})

    hint = agent.process_owner_hint(501, {})

    assert hint == {"match_method": "none", "scan_complete": True}


def test_gpu_process_owner_keeps_unreadable_process_environment_unknown(agent, monkeypatch):
    parents = {501: 500, 500: 400, 400: 1, 1: 0}
    monkeypatch.setattr(agent, "process_parent", lambda pid: parents.get(pid))
    monkeypatch.setattr(agent, "process_environment", lambda pid: None if pid == 501 else {})

    hint = agent.process_owner_hint(501, {})

    assert hint == {"match_method": "none", "scan_complete": False}


@pytest.mark.parametrize("stop_mode", [None, "platform_stop", "agent_shutdown"])
def test_failed_updates_retry_metrics_and_explicit_stop(agent, training, monkeypatch, stop_mode):
    config, task = training
    killed = []

    class Process:
        pid = 123456
        returncode = None
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            self.returncode = self.returncode if self.returncode is not None else 0
            return self.returncode

    process = Process()

    def kill(pid, sig):
        killed.append((pid, sig))
        process.returncode = -15

    class Stop:
        def __init__(self):
            self.stopped = False
            self.ticks = 0
        def is_set(self):
            return self.stopped
        def wait(self, seconds):
            if seconds == 2:
                self.ticks += 1
                if stop_mode == "agent_shutdown":
                    self.stopped = True
                elif self.ticks == 3:
                    process.returncode = 0
            return self.stopped

    def popen(*args, **kwargs):
        inherited = kwargs.get("pass_fds", ())
        assert inherited and len(inherited) == len(set(inherited))
        run_dir = Path(task["run_dir"])
        shutil.rmtree(run_dir)
        run_dir.mkdir()
        (run_dir / "results.csv").write_text("epoch,time,score\n1,1,0.5\n")
        kwargs["stdout"].write("dummy output\n")
        return process

    monkeypatch.setattr(agent.subprocess, "Popen", popen)
    monkeypatch.setattr(agent.os, "killpg", kill)
    updates = []

    def api(_config, path, method="GET", payload=None):
        if payload is None:
            return {"stop": False}
        if path.endswith("/artifacts"):
            return {}
        updates.append(dict(payload))
        if stop_mode == "platform_stop":
            return {"stop_requested": True}
        if len(updates) <= 2:
            raise URLError("temporary outage")
        return {}

    monkeypatch.setattr(agent, "api_json", api)
    agent.run_remote_task(config, task, Stop())
    final = updates[-1]
    assert final["stop_reason"] == stop_mode
    if stop_mode is None:
        assert not killed
        assert updates[1]["metrics"] == updates[2]["metrics"]
        assert updates[2]["command"]
        assert final["status"] == "completed"
    else:
        assert len(killed) == 1
        assert final["status"] == "stopped" and final["exit_code"] == -15


def test_terminal_backlog_survives_outage_and_retries_all_epochs(agent, training, monkeypatch):
    config, task = training
    metrics = [{"epoch": i, "values": {"score": 0.5}} for i in range(1, 601)]
    payload = {"status": "completed", "completed_epochs": 600, "progress": 100,
               "exit_code": 0, "metrics": metrics, "log_tail": ["finished"]}
    active_path = agent.save_active_run(config, task, "running")
    path = agent.save_pending_final(config, task["run_id"], payload)
    assert os.stat(path).st_mode & 0o077 == 0
    assert "token" not in path.read_text()

    def offline(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(agent, "api_json", offline)
    assert agent.retry_pending_finals(config) is False
    assert json.loads(path.read_text()) == payload
    assert active_path.exists()
    received = []

    def online(_config, _path, method, payload):
        received.append(payload)
        assert len(payload["metrics"]) <= 256
        return {"status": payload["status"]}

    monkeypatch.setattr(agent, "api_json", online)
    assert agent.retry_pending_finals(config) is False
    assert agent.retry_pending_finals(config) is False
    assert agent.retry_pending_finals(config) is True
    assert [p["status"] for p in received] == ["running", "running", "completed"]
    assert [m["epoch"] for p in received for m in p["metrics"]] == list(range(1, 601))
    assert not path.exists()
    assert agent.load_active_run(config) is None


def test_external_log_symlink_cannot_escape_allowed_root(agent, training, tmp_path):
    config, task = training
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(task["parameters"]["remote_output_root"]) / ".platform-logs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="日志路径"):
        agent.build_training_command(config, task)
    assert not Path(task["run_dir"]).exists()


def test_missing_remote_training_paths_report_the_full_input(agent, training, tmp_path):
    config, task = training
    missing_model = tmp_path / "center-only/models/best.pt"
    task["model_path"] = str(missing_model)
    with pytest.raises(ValueError) as missing_model_error:
        agent.build_training_command(config, task)
    assert str(missing_model) in str(missing_model_error.value)
    assert "远程服务器" in str(missing_model_error.value)
    assert "不是中心平台路径" in str(missing_model_error.value)

    task["model_path"] = str(Path(config["training_cwd"]) / "model.pt")
    missing_output = tmp_path / "remote-runs-missing"
    task["parameters"]["remote_output_root"] = str(missing_output)
    task["run_dir"] = str(missing_output / "detect/task-id/attempt-001")
    task["log_path"] = str(missing_output / ".platform-logs/20261012/task-id/attempt-001.log")
    with pytest.raises(ValueError) as missing_output_error:
        agent.build_training_command(config, task)
    assert str(missing_output) in str(missing_output_error.value)
    assert "训练输出根目录" in str(missing_output_error.value)


def test_training_lock_is_shared_by_agents_for_same_workspace(agent, training):
    config, _ = training
    first = agent.acquire_training_lock(config)
    assert first is not None
    try:
        assert agent.acquire_training_lock(config) is None
    finally:
        agent.release_training_lock(first)
    second = agent.acquire_training_lock(config)
    assert second is not None
    agent.release_training_lock(second)


def test_training_process_inherits_workspace_lock(agent, training):
    config, _ = training
    lock = agent.acquire_training_lock(config)
    process = None
    try:
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                   pass_fds=(lock.fileno(),))
        assert agent.process_holds_training_lock(config, process.pid)
        # Simulate an Agent crash: close the parent's descriptor without an
        # explicit LOCK_UN while the training process keeps the inherited FD.
        lock.close()
        lock = None
        assert agent.acquire_training_lock(config) is None
    finally:
        if lock is not None:
            agent.release_training_lock(lock)
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
    second = agent.acquire_training_lock(config)
    assert second is not None
    agent.release_training_lock(second)


def test_workspace_registration_allows_recovery_while_child_holds_output_locks(agent, training):
    config, _ = training
    lock = agent.acquire_training_lock(config)
    assert lock is not None
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        pass_fds=lock.filenos(),
    )
    lock.close()
    try:
        assert agent.process_holds_training_lock(config, process.pid)
        registered = agent.register_workspace(config)
        payload = json.loads(registered.read_text())
        assert payload["node_id"] == config["node_id"]
        assert agent.acquire_output_root_locks(config) is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_output_root_locks_cover_different_configured_locks_and_nested_roots(agent, training):
    config, _ = training
    parent = Path(config["training_output_roots"][0])
    child = parent / "nested"
    sibling_a = parent.parent / "sibling-a"
    sibling_b = parent.parent / "sibling-b"
    for path in (child, sibling_a, sibling_b):
        path.mkdir(parents=True)

    parent_config = {**config, "training_lock_file": str(Path(config["training_cwd"]) / "parent.lock")}
    child_config = {**config, "node_id": "c" * 32, "training_output_roots": [str(child)],
                    "training_lock_file": str(Path(config["training_cwd"]) / "child.lock")}
    parent_lock = agent.acquire_training_lock(parent_config)
    assert parent_lock is not None
    try:
        assert agent.acquire_training_lock(child_config) is None
    finally:
        agent.release_training_lock(parent_lock)

    first_config = {**config, "training_output_roots": [str(sibling_a)],
                    "training_lock_file": str(Path(config["training_cwd"]) / "sibling-a.lock")}
    second_config = {**config, "node_id": "d" * 32, "training_output_roots": [str(sibling_b)],
                     "training_lock_file": str(Path(config["training_cwd"]) / "sibling-b.lock")}
    first = agent.acquire_training_lock(first_config)
    second = agent.acquire_training_lock(second_config)
    assert first is not None and second is not None
    agent.release_training_lock(second)
    agent.release_training_lock(first)


def test_workspace_registry_rejects_duplicate_node_and_reclaims_dead_process(agent, training):
    config, _ = training
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = agent.process_identity(process.pid)
        registry = agent.workspace_registry_dir(config)
        registry.mkdir(parents=True, exist_ok=True)
        path = registry / f"{agent.workspace_owner_id(config)}.json"
        agent._atomic_private_json(path, {
            "version": agent.WORKSPACE_REGISTRY_VERSION,
            "owner_id": agent.workspace_owner_id(config),
            "node_id": config["node_id"],
            "training_output_roots": config["training_output_roots"],
            "process": identity,
        })
        with pytest.raises(ValueError, match="node_id"):
            agent.register_workspace(config)
    finally:
        process.terminate()
        process.wait(timeout=5)

    registered = agent.register_workspace(config)
    payload = json.loads(registered.read_text())
    assert payload["process"]["pid"] == os.getpid()


def test_workspace_registry_is_stable_across_center_url_and_cleans_legacy_marker(agent, training):
    config, _ = training
    old = {**config, "center_url": "https://old-center.example"}
    moved = {**config, "center_url": "https://new-center.example:8443"}
    assert agent.workspace_owner_id(old) == agent.workspace_owner_id(moved)
    legacy_owner = hashlib.sha256(f"{old['center_url']}\0{old['node_id']}".encode()).hexdigest()
    legacy_dir = agent.legacy_workspace_registry_dir(old)
    legacy_dir.mkdir(parents=True)
    legacy = legacy_dir / f"{legacy_owner}.json"
    legacy.write_text(json.dumps({"version": 1, "owner_id": legacy_owner, "node_id": old["node_id"]}))
    agent.register_workspace(moved)
    assert not legacy.exists()


def test_workspace_registry_blocks_live_overlap_but_not_independent_or_stale_owner(agent, training):
    config, _ = training
    root = Path(config["training_output_roots"][0])
    sibling = root.parent / "independent"
    sibling.mkdir()
    agent.register_workspace(config)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    other_node = "e" * 32
    other_owner = hashlib.sha256(other_node.encode("ascii")).hexdigest()
    marker = agent.workspace_registry_dir(config) / f"{other_owner}.json"
    try:
        base = {"version": agent.WORKSPACE_REGISTRY_VERSION, "owner_id": other_owner,
                "node_id": other_node, "process": agent.process_identity(process.pid)}
        agent._atomic_private_json(marker, {**base, "training_output_roots": [str(root)]})
        assert agent.workspace_delete_conflicts(config, [root]) is True
        agent._atomic_private_json(marker, {**base, "training_output_roots": [str(sibling)]})
        assert agent.workspace_delete_conflicts(config, [root]) is False
    finally:
        process.terminate()
        process.wait(timeout=5)
    agent._atomic_private_json(marker, {**base, "training_output_roots": [str(root)]})
    assert agent.workspace_delete_conflicts(config, [root]) is False
    assert not marker.exists()


def test_delete_refuses_while_training_owns_output_root(agent, training):
    config, task = training
    agent.build_training_command(config, task)
    payload = {**task, "output_root": task["parameters"]["remote_output_root"]}
    lock = agent.acquire_training_lock(config)
    try:
        with pytest.raises(ValueError, match="正在使用"):
            agent.delete_remote_files(config, {"runs": [payload]})
    finally:
        agent.release_training_lock(lock)


def test_delete_legacy_flat_attempt_preserves_canonical_children(agent, training):
    config, task = training
    output_root = Path(task["parameters"]["remote_output_root"])
    task_root = output_root / task["task_type"] / task["task_id"]
    legacy_weight = task_root / "weights" / "last.pt"
    canonical_dir = task_root / "attempt-002"
    legacy_weight.parent.mkdir(parents=True)
    legacy_weight.write_bytes(b"legacy")
    canonical_dir.mkdir()
    marker = canonical_dir / "keep.txt"
    marker.write_text("canonical")
    log = task_root / "platform.log"
    log.write_text("legacy log")
    payload = {
        **task,
        "attempt_number": 1,
        "run_dir": str(task_root),
        "log_path": str(log),
        "output_root": str(output_root),
        "layout": "legacy_flat",
    }

    with pytest.raises(ValueError, match="仅允许在删除"):
        agent.operation_paths(config, payload)
    agent.delete_remote_files(config, {"runs": [payload]})

    assert not legacy_weight.exists() and not log.exists()
    assert marker.read_text() == "canonical"
    agent.delete_remote_files(config, {"runs": [payload]})  # idempotent after a lost ACK


def test_fetch_standalone_remote_model_uses_dedicated_root_and_stable_metadata(
    agent, training, monkeypatch,
):
    config, _ = training
    model_root = Path(config["training_cwd"]) / "pt"
    model_root.mkdir()
    model = model_root / "standalone.pt"
    content = (b"checkpoint-block" * 40_000) + b"end"
    model.write_bytes(content)
    config["model_test_roots"] = [str(model_root)]
    config["model_transfer_finalize_timeout_seconds"] = 321
    operation = {
        "id": "1" * 32,
        "kind": "fetch_model",
        "lease": "2" * 32,
        "payload": {"model_asset_id": "3" * 32, "path": str(model)},
    }
    chunks, timeouts = [], []

    def api(_config, path, method="GET", payload=None, extra_headers=None, *, timeout_seconds=20):
        assert method == "POST" and path.endswith("/chunk")
        chunks.append(dict(payload))
        timeouts.append(timeout_seconds)
        return {"status": "succeeded" if payload["final"] else "running"}

    monkeypatch.setattr(agent, "api_json", api)
    agent.execute_operation(config, operation, threading.Event())

    assert len(chunks) == 2
    assert b"".join(base64.b64decode(chunk["data"]) for chunk in chunks) == content
    assert [chunk["offset"] for chunk in chunks] == [0, 512 * 1024]
    assert all(chunk["size"] == len(content) for chunk in chunks)
    assert len({chunk["modified_at"] for chunk in chunks}) == 1
    assert len({chunk["mtime_ns"] for chunk in chunks}) == 1
    assert chunks[-1]["sha256"] == hashlib.sha256(content).hexdigest()
    assert timeouts == [20, 321]


def test_fetch_training_artifact_uses_long_timeout_only_for_final_chunk(
    agent, training, monkeypatch,
):
    config, task = training
    run_dir = Path(task["run_dir"])
    weight = run_dir / "weights/best.pt"
    weight.parent.mkdir(parents=True)
    content = (b"artifact-block" * 40_000) + b"end"
    weight.write_bytes(content)
    info = weight.stat()
    config["model_transfer_finalize_timeout_seconds"] = 432
    operation = {
        "id": "7" * 32,
        "kind": "fetch",
        "lease": "8" * 32,
        "payload": {
            **task,
            "output_root": task["parameters"]["remote_output_root"],
            "layout": "canonical",
            "name": "weights/best.pt",
            "size": len(content),
            "modified_at": datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
        },
    }
    chunks, timeouts = [], []

    def api(_config, path, method="GET", payload=None, extra_headers=None, *, timeout_seconds=20):
        assert method == "POST" and path.endswith("/chunk")
        chunks.append(dict(payload))
        timeouts.append(timeout_seconds)
        return {"status": "succeeded" if payload["final"] else "running"}

    monkeypatch.setattr(agent, "api_json", api)
    agent.execute_operation(config, operation, threading.Event())

    assert len(chunks) == 2
    assert b"".join(base64.b64decode(chunk["data"]) for chunk in chunks) == content
    assert [chunk["offset"] for chunk in chunks] == [0, 512 * 1024]
    assert chunks[-1]["sha256"] == hashlib.sha256(content).hexdigest()
    assert timeouts == [20, 432]


def test_standalone_remote_model_rejects_training_roots_links_and_non_pt(
    agent, training, tmp_path,
):
    config, _ = training
    model_root = Path(config["training_cwd"]) / "pt"
    model_root.mkdir()
    config["model_test_roots"] = [str(model_root)]
    payload = {"model_asset_id": "3" * 32}

    training_model = Path(config["training_cwd"]) / "model.pt"
    with pytest.raises(ValueError, match="model_test_roots"):
        agent.remote_model_path(config, {**payload, "path": str(training_model)})

    text_file = model_root / "model.onnx"
    text_file.write_bytes(b"model")
    with pytest.raises(ValueError, match=r"\.pt"):
        agent.remote_model_path(config, {**payload, "path": str(text_file)})

    real = model_root / "real.pt"
    real.write_bytes(b"model")
    linked = model_root / "linked.pt"
    linked.symlink_to(real)
    with pytest.raises(ValueError, match="符号链接"):
        agent.remote_model_path(config, {**payload, "path": str(linked)})

    linked_root = tmp_path / "linked-model-root"
    linked_root.symlink_to(model_root, target_is_directory=True)
    config["model_test_roots"] = [str(linked_root)]
    with pytest.raises(ValueError, match="符号链接"):
        agent.remote_model_path(config, {**payload, "path": str(linked_root / "real.pt")})


def test_fetch_standalone_remote_model_rejects_file_replacement(
    agent, training, monkeypatch,
):
    config, _ = training
    model_root = Path(config["training_cwd"]) / "pt"
    model_root.mkdir()
    model = model_root / "changing.pt"
    model.write_bytes(b"a" * (600 * 1024))
    config["model_test_roots"] = [str(model_root)]
    operation = {
        "id": "4" * 32,
        "kind": "fetch_model",
        "lease": "5" * 32,
        "payload": {"model_asset_id": "6" * 32, "path": str(model)},
    }
    chunks, results = [], []

    def api(_config, path, method="GET", payload=None):
        assert method == "POST"
        if path.endswith("/chunk"):
            chunks.append(dict(payload))
            replacement = model.with_suffix(".replacement")
            replacement.write_bytes(b"b" * (600 * 1024))
            replacement.replace(model)
            return {"status": "running"}
        assert path.endswith("/result")
        results.append(dict(payload))
        return {"status": "failed"}

    monkeypatch.setattr(agent, "api_json", api)
    agent.execute_operation(config, operation, threading.Event())

    assert len(chunks) == 1 and chunks[0]["final"] is False
    assert results and results[0]["success"] is False
    assert "发生变化" in results[0]["error"]


def test_claim_confirmation_retries_lost_response_and_honors_stop(agent, training, monkeypatch):
    config, task = training
    task["claim_lease"] = "c" * 32
    calls = []

    def api(_config, path, method="GET", payload=None, extra_headers=None):
        calls.append((path, payload, extra_headers))
        if path.endswith("/confirm") and sum(item[0].endswith("/confirm") for item in calls) == 1:
            raise URLError("response lost after commit")
        if path.endswith("/confirm"):
            return {"confirmed": True, "stop": True}
        assert path.endswith("/update")
        assert payload["status"] == "stopped" and payload["stop_reason"] == "platform_stop"
        assert extra_headers == {"X-Training-Lease": task["claim_lease"]}
        return {"status": "stopped"}

    monkeypatch.setattr(agent, "api_json", api)
    monkeypatch.setattr(agent, "CONFIRM_RETRY_DELAYS", (0,))
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("training must not start"))
    shutdown = threading.Event()
    shutdown.set()
    agent.run_remote_task(config, task, shutdown)
    assert sum(path.endswith("/confirm") for path, *_ in calls) == 2
    assert agent.load_active_run(config) is None
    assert not list(Path(config["training_state_dir"]).glob("*.json"))


def test_claim_rejection_never_starts_process_or_writes_terminal(agent, training, monkeypatch):
    config, task = training
    task["claim_lease"] = "d" * 32

    def reject(*_args, **_kwargs):
        raise HTTPError("confirm", 409, "expired", {}, None)

    monkeypatch.setattr(agent, "api_json", reject)
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("training must not start"))
    agent.run_remote_task(config, task, threading.Event())
    assert agent.load_active_run(config) is None
    assert not list(Path(config["training_state_dir"]).glob("*.json"))


def test_unknown_claim_confirmation_is_persisted_for_same_lease_retry(agent, training, monkeypatch):
    config, task = training
    task["claim_lease"] = "e" * 32
    monkeypatch.setattr(agent, "api_json", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("offline")))
    monkeypatch.setattr(agent, "CONFIRM_RETRY_DELAYS", ())
    monkeypatch.setattr(agent.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("training must not start"))
    agent.run_remote_task(config, task, threading.Event())
    state = agent.load_active_run(config)
    assert state["phase"] == "claimed"
    assert state["task"]["claim_lease"] == task["claim_lease"]
    assert not list(Path(config["training_state_dir"]).glob("*.json"))


def test_pending_final_acknowledgment_blocks_new_task_claims(agent, tmp_path, monkeypatch):
    calls = []
    config = {"enable_training": True, "node_id": "b" * 32,
              "nvidia_smi_bin": "nvidia-smi", "report_interval_seconds": 5,
              "training_state_dir": str(tmp_path / "state")}

    class StopEvent:
        def __init__(self):
            self.stopped = False
        def is_set(self):
            return self.stopped
        def set(self):
            self.stopped = True
        def wait(self, _seconds):
            self.stopped = True
            return True

    monkeypatch.setattr(agent, "load_config", lambda _path: config)
    monkeypatch.setattr(agent, "acquire_agent_instance_lock", lambda _config: object())
    monkeypatch.setattr(agent, "release_training_lock", lambda _lock: None)
    monkeypatch.setattr(agent, "register_workspace", lambda _config: None)
    monkeypatch.setattr(agent, "collect", lambda *_args: {
        "training": {"ready": True}, "compute": {"gpus": []}})
    monkeypatch.setattr(agent, "send", lambda *_args: 5)
    monkeypatch.setattr(agent, "retry_pending_finals", lambda _config: False)
    monkeypatch.setattr(agent, "load_active_runs", lambda _config: [])

    def api(_config, path, method="GET", payload=None):
        calls.append(path)
        return {"operation": None} if path.endswith("/operations/poll") else {"task": None}

    monkeypatch.setattr(agent, "api_json", api)
    monkeypatch.setattr(agent.threading, "Event", StopEvent)
    monkeypatch.setattr(sys, "argv", ["monitor_agent.py", "--config", "unused.json"])

    assert agent.main() == 0
    assert any(path.endswith("/operations/poll") for path in calls)
    assert not any(path.endswith("/agent/poll") for path in calls)


def test_active_state_is_private_and_rejects_reused_pid(agent, training):
    config, task = training
    config["token"] = "ylp_secret-that-must-not-be-persisted"
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        identity = agent.process_identity(process.pid)
        assert identity and agent.process_identity_matches(identity)
        path = agent.save_active_run(config, task, "running", command=[sys.executable], identity=identity)
        assert os.stat(path).st_mode & 0o077 == 0
        assert config["token"] not in path.read_text()
        assert agent.load_active_run(config)["process"] == identity
        reused = dict(identity, start_ticks=identity["start_ticks"] + 1)
        assert not agent.process_identity_matches(reused)
        changed_command = dict(identity, cmdline_sha256="0" * 64)
        with pytest.raises(ValueError, match="命令行身份"):
            agent._recovery_identity({"phase": "running", "process": changed_command,
                                      "command": [sys.executable]})
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_training_capability_uses_explicit_output_root_version_and_lock(agent, training):
    config, _ = training
    model_root = Path(config["training_cwd"]) / "models"
    output_root = Path(config["training_cwd"]) / "runs"
    model_root.mkdir()
    yolo = Path(config["training_cwd"]) / "fake-yolo"
    yolo.write_text("#!/bin/sh\necho 'Ultralytics YOLO 8.3.219'\n")
    yolo.chmod(0o700)
    config.update(enable_training=True, allowed_training_roots=[str(model_root), str(output_root)],
                  training_output_roots=[str(output_root)], yolo_bin=str(yolo),
                  training_lock_file=str(Path(config["training_cwd"]) / ".shared.lock"))
    capability = agent.training_capability(config)
    assert capability["ready"] is True and capability["output_writable"] is True
    assert capability["yolo_version"] == "8.3.219" and capability["lock_occupied"] is False
    lock = agent.acquire_training_lock(config)
    try:
        assert agent.training_capability(config)["lock_occupied"] is True
    finally:
        agent.release_training_lock(lock)


def test_current_catalog_parameters_reach_remote_command(agent, training):
    config, task = training
    values = {"channels_last": True, "cls_remap": False, "dis": 0.5,
              "distill_model": "/models/teacher.pt", "dnn": True, "nms": False,
              "quantize": "int8", "time": 2.5}
    task["parameters"]["config"] = values
    command = agent.build_training_command(config, task)
    expected = {f"{key}={agent.scalar_parameter(value)}" for key, value in values.items()}
    assert expected <= set(command)
    assert set(values) <= agent.TRAINING_KEYS


def test_missing_delete_root_is_always_rejected(agent, training, tmp_path):
    config, task = training
    allowed = Path(config["training_cwd"])
    payload = {**task, "output_root": str(allowed / "missing-output")}
    payload["run_dir"] = str(Path(payload["output_root"]) / "detect/task-id/attempt-001")
    payload["log_path"] = str(Path(payload["output_root"]) / ".platform-logs/20261012/task-id/attempt-001.log")
    with pytest.raises(ValueError, match="输出根目录不存在"):
        agent.operation_paths(config, payload, allow_missing_root=True)
    payload["output_root"] = str(tmp_path / "outside/missing-output")
    payload["run_dir"] = str(Path(payload["output_root"]) / "detect/task-id/attempt-001")
    payload["log_path"] = str(Path(payload["output_root"]) / ".platform-logs/20261012/task-id/attempt-001.log")
    with pytest.raises(ValueError, match="输出根目录不存在"):
        agent.operation_paths(config, payload, allow_missing_root=True)


def test_agent_instance_lock_blocks_duplicate_config(agent, training):
    config, _ = training
    first = agent.acquire_agent_instance_lock(config)
    assert first is not None
    try:
        assert agent.acquire_agent_instance_lock(config) is None
    finally:
        agent.release_training_lock(first)
    second = agent.acquire_agent_instance_lock(config)
    assert second is not None
    agent.release_training_lock(second)


def test_recovery_stops_only_identity_verified_process(agent, training, monkeypatch):
    config, task = training
    task["claim_lease"] = "f" * 32
    lock = agent.acquire_training_lock(config)
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True,
                               pass_fds=(lock.fileno(),))
    lock.close()
    updates = []
    try:
        identity = agent.process_identity(process.pid)
        state_path = agent.save_active_run(config, task, "running",
                                           command=[sys.executable, "-c", "import time; time.sleep(30)"],
                                           identity=identity)
        state = json.loads(state_path.read_text())
        assert agent.process_holds_training_lock(config, process.pid)

        def api(_config, path, method="GET", payload=None, extra_headers=None):
            if path.endswith("/control"):
                return {"status": "stopping", "stop": True}
            updates.append((payload, extra_headers))
            return {"status": payload["status"], "stop_requested": True}

        monkeypatch.setattr(agent, "api_json", api)
        monkeypatch.setattr(agent, "acquire_training_lock",
                            lambda *_args, **_kwargs: pytest.fail("process-held recovery must not acquire the lock"))
        monkeypatch.setattr(agent, "release_training_lock",
                            lambda *_args, **_kwargs: pytest.fail("process-held recovery must not release the lock"))
        agent.recover_remote_task(config, state, threading.Event(), process_holds_lock=True)
        process.wait(timeout=5)
        assert updates[-1][0]["status"] == "stopped"
        assert updates[-1][1] == {"X-Training-Lease": task["claim_lease"]}
        assert agent.load_active_run(config) is None
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


@pytest.mark.parametrize(("scenario", "expected"), [
    ("all_epochs", "completed"),
    ("early_stopping", "completed"),
    ("trace_oom", "completed"),
    ("epochs_only", "interrupted"),
    ("early_marker_only", "interrupted"),
    ("validation_error", "interrupted"),
    ("error_after_save", "interrupted"),
    ("wrong_path", "interrupted"),
    ("uncertain", "interrupted"),
])
def test_recovery_classifies_process_that_exited_before_agent_restart(
    agent, training, monkeypatch, scenario, expected,
):
    config, task = training
    run_dir = Path(task["run_dir"])
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    log = Path(task["log_path"])
    log.parent.mkdir(parents=True)
    (weights / "best.pt").write_bytes(b"best")
    (weights / "last.pt").write_bytes(b"last")
    full_epochs = scenario in {"all_epochs", "epochs_only"}
    if full_epochs:
        (run_dir / "results.csv").write_text("epoch,time,score\n1,1,0.1\n2,2,0.2\n3,3,0.3\n")
    else:
        (run_dir / "results.csv").write_text("epoch,time,score\n1,1,0.1\n")
    prefix = "EarlyStopping: Training stopped early as no improvement observed in last 2 epochs. Best results observed at epoch 1.\n"
    if scenario == "trace_oom":
        prefix = "Traceback (most recent call last):\nCUDA out of memory\nTrace collection disabled; training continues\n" + prefix
    text = prefix if not full_epochs else "training output\n"
    if scenario not in {"epochs_only", "early_marker_only", "uncertain"}:
        text += f"{3 if full_epochs else 1} epochs completed in 1.0 hours.\n"
        text += "".join(f"final validation line {index}\n" for index in range(200))
        if scenario == "validation_error":
            text += "Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n"
        else:
            text += f"Results saved to {run_dir if scenario != 'wrong_path' else run_dir.parent / 'attempt-000'}\n"
            if scenario == "error_after_save":
                text += "Traceback (most recent call last):\nRuntimeError: final model cleanup failed\n"
    log.write_text(text)
    updates = []

    def api(_config, path, method="GET", payload=None, extra_headers=None):
        if path.endswith("/control"):
            return {"status": "running", "stop": False}
        updates.append(payload)
        return {"status": payload["status"]}

    monkeypatch.setattr(agent, "api_json", api)
    state = {"phase": "running", "task": task, "run_id": task["run_id"],
             "process": {"pid": 99999999, "start_ticks": 1, "boot_id": "gone", "cmdline_sha256": "0" * 64},
             "command": [sys.executable, "missing-training-process"]}
    lock = agent.acquire_training_lock(config)
    agent.recover_remote_task(config, state, threading.Event(), training_lock=lock)
    assert updates[-1]["status"] == expected
    assert updates[-1]["completed_epochs"] == (3 if full_epochs else 1)
    assert updates[-1]["exit_code"] is None
    assert updates[-1]["progress"] == (100 if expected == "completed" else pytest.approx(min(99.99, (3 if full_epochs else 1) * 100 / 3)))
    if expected == "completed":
        assert updates[-1]["completion_info"]["source"] == "legacy_log"
    if expected == "interrupted":
        assert "无法取得原进程退出码" in updates[-1]["error_message"]


def test_remote_training_outcome_runner_is_used_when_trace_is_disabled(agent, training):
    config, task = training
    config["enable_training_trace"] = False
    command = ["yolo", "segment", "train", "model=model.pt", "epochs=3"]
    wrapped = agent.prepare_trace_command(config, task, command)
    assert wrapped[1].endswith("training_trace_runner.py")
    assert "--run-dir" in wrapped and task["run_dir"] in wrapped
    assert "--trace-enabled" not in wrapped


def test_remote_legacy_resume_summary_counts_only_new_epochs(agent, training):
    _config, task = training
    task["parameters"]["epochs"] = 100
    run_dir = Path(task["run_dir"])
    weights = run_dir / "weights"
    weights.mkdir(parents=True)
    for name in ("best.pt", "last.pt"):
        (weights / name).write_bytes(b"checkpoint")
    (run_dir / "results.csv").write_text("epoch,time,score\n55,1,0.8\n56,2,0.81\n")
    log = Path(task["log_path"])
    log.parent.mkdir(parents=True)
    log.write_text(f"2 epochs completed in 0.5 hours.\nValidating best.pt\nResults saved to {run_dir}\n")
    assert agent.recovered_natural_exit(task, 56, []) == ("completed", None, None)


@pytest.mark.parametrize("rank", ["-1", "1"])
@pytest.mark.parametrize("scenario", ["early_stopping", "all_epochs", "validation_failure", "no_patience", "resume_epochs", "keyboard_interrupt", "system_exit_zero", "system_exit_none", "system_exit_two", "system_exit_wraparound"])
def test_remote_runner_persists_actual_result_after_train_returns(agent, training, tmp_path, scenario, rank):
    _config, task = training
    if scenario == "early_stopping":
        task["parameters"]["epochs"] = 5
    run_dir = Path(task["run_dir"])
    fake = tmp_path / "fake-ultralytics"
    fake.mkdir()
    (fake / "ultralytics.py").write_text('''
import os
from pathlib import Path
from types import SimpleNamespace
class YOLO:
    def __init__(self, model, task):
        self.trainer = SimpleNamespace(epochs=5 if os.getenv("SCENARIO") == "early_stopping" else 3, stopper=SimpleNamespace(best_epoch=1, patience=float("inf") if os.getenv("SCENARIO") == "no_patience" else 2))
        if os.getenv("SCENARIO") == "resume_epochs":
            self.trainer.args = SimpleNamespace(epochs=8)
            self.trainer.stopper = SimpleNamespace(best_epoch=0, patience=2)
        self.metrics = SimpleNamespace(results_dict={"metrics/mAP50(B)": 0.818})
    def train(self, **values):
        root = Path(values["project"]) / values["name"]
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / "training-result.json").exists()
        (root / "results.csv").write_text("epoch,time,score\\n1,1,0.1\\n" + ("2,2,0.2\\n3,3,0.3\\n" if os.getenv("SCENARIO") in {"all_epochs", "early_stopping", "no_patience"} else ""))
        if os.getenv("SCENARIO") == "validation_failure":
            print("EarlyStopping: Training stopped early", flush=True)
            raise RuntimeError("final validation failed")
        if os.getenv("SCENARIO") == "keyboard_interrupt":
            raise KeyboardInterrupt()
        if os.getenv("SCENARIO").startswith("system_exit_"):
            raise SystemExit({"system_exit_zero": 0, "system_exit_none": None, "system_exit_two": 2, "system_exit_wraparound": 256}[os.getenv("SCENARIO")])
''')
    runner = tmp_path / "training-runner.py"
    runner.write_text(agent.REMOTE_TRACE_RUNNER)
    command = ["yolo", "detect", "train", "model=fake.pt", f"epochs={task['parameters']['epochs']}",
               f"project={run_dir.parent}", f"name={run_dir.name}"]
    result = subprocess.run([sys.executable, str(runner), "--trace-dir", str(run_dir / "training-trace"),
                             "--run-dir", str(run_dir), "--command-json", json.dumps(command)],
                            env={**os.environ, "PYTHONPATH": str(fake), "YOLO_FORGE_RUN_ID": task["run_id"],
                                 "YOLO_FORGE_TASK_ID": task["task_id"], "SCENARIO": scenario, "RANK": rank},
                            text=True, capture_output=True, timeout=10)
    failure = scenario not in {"early_stopping", "all_epochs", "no_patience", "resume_epochs"}
    expected_code = 130 if scenario == "keyboard_interrupt" else 2 if scenario == "system_exit_two" else 1 if failure else 0
    assert result.returncode == expected_code
    if rank == "1":
        assert not (run_dir / "training-result.json").exists()
        return
    record = json.loads((run_dir / "training-result.json").read_text())
    assert record["schema_version"] == 1
    assert record["run_id"] == task["run_id"] and record["task_id"] == task["task_id"]
    assert record["exit_code"] == result.returncode
    expected_status = "failed" if failure else "completed"
    assert record["status"] == expected_status
    assert record["reason"] == ("exception" if failure else
                                 "early_stopping" if scenario == "early_stopping" else
                                 "training_completed" if scenario == "resume_epochs" else "epochs_completed")
    assert record["expected_epochs"] == (8 if scenario == "resume_epochs" else task["parameters"]["epochs"])
    assert not list(run_dir.glob("*.tmp"))
    status, error, code = agent.recovered_natural_exit(task, record["completed_epochs"], [])
    assert status == expected_status and code == result.returncode
    assert bool(error) == failure
    if scenario == "validation_failure":
        assert "final validation failed" in error
    info = agent.training_completion_info(task, record["completed_epochs"], [])
    assert info["source"] == "training_result"
    assert info["final_metrics"]["metrics/mAP50(B)"] == 0.818


@pytest.mark.parametrize("invalid", ["run_id", "task_id", "expected_epochs", "finished_at", "status_code", "best_epoch", "patience", "schema_version", "future_finished_at"])
def test_remote_recovery_rejects_foreign_or_invalid_result_records(agent, training, invalid):
    _config, task = training
    run_dir = Path(task["run_dir"])
    run_dir.mkdir(parents=True)
    record = {"schema_version": 1, "run_id": task["run_id"], "task_id": task["task_id"],
              "status": "completed", "exit_code": 0, "reason": "epochs_completed", "completed_epochs": 3,
              "expected_epochs": 3, "finished_at": datetime.now(timezone.utc).isoformat()}
    if invalid == "status_code":
        record["exit_code"] = 1
    elif invalid == "future_finished_at":
        record["finished_at"] = "2999-01-01T00:00:00+00:00"
    else:
        record[invalid] = {"run_id": "f" * 32, "task_id": "other-task", "expected_epochs": 0,
                           "finished_at": "bad timestamp", "best_epoch": None, "patience": -1, "schema_version": True}[invalid]
    (run_dir / "training-result.json").write_text(json.dumps(record))
    assert agent.recovered_natural_exit(task, 3, [])[0] == "interrupted"


def test_recovered_weight_stat_race_degrades_to_interrupted(agent, training, monkeypatch):
    _config, task = training
    weights = Path(task["run_dir"]) / "weights"
    weights.mkdir(parents=True)
    best, last = weights / "best.pt", weights / "last.pt"
    best.write_bytes(b"best")
    last.write_bytes(b"last")
    log = Path(task["log_path"])
    log.parent.mkdir(parents=True)
    log.write_text("EarlyStopping: complete\n")
    original = Path.stat
    calls = 0

    def racing_stat(path, *args, **kwargs):
        nonlocal calls
        if path == best:
            calls += 1
            if calls > 1:
                raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", racing_stat)
    assert agent.recovered_natural_exit(task, 1, ["EarlyStopping: complete"])[0] == "interrupted"


def test_check_config_is_offline_and_never_prints_token(agent, training, tmp_path):
    root = Path(training[0]["training_cwd"])
    yolo = root / "check-yolo"
    yolo.write_text("#!/bin/sh\necho 'YOLO 9.0.1'\n")
    yolo.chmod(0o700)
    token = "ylp_" + "s" * 32
    config_path = tmp_path / "agent-config.json"
    config_path.write_text(json.dumps({
        "center_url": "http://127.0.0.1:9", "node_id": "1" * 32, "token": token,
        "report_interval_seconds": 5, "nvidia_smi_bin": "nvidia-smi", "enable_training": True,
        "allowed_training_roots": [str(root)], "training_output_roots": [str(root / "runs")],
        "yolo_bin": str(yolo), "training_cwd": str(root),
    }))
    (root / "runs").mkdir(exist_ok=True)
    config_path.chmod(0o600)
    script = Path(agent.__file__)
    result = subprocess.run([sys.executable, str(script), "--config", str(config_path), "--check-config"],
                            text=True, capture_output=True, timeout=10)
    output = result.stdout + result.stderr
    assert result.returncode == 0 and "配置检查通过" in output
    assert token not in output
    assert str(root / "runs") in output and "training_output_roots" in output


def write_relative_config(tmp_path, **overrides):
    project = tmp_path / "project"
    (project / "runs").mkdir(parents=True, exist_ok=True)
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir(exist_ok=True)
    path = agent_dir / "agent-config.json"
    payload = {
        "_comments": {"training_cwd": "相对于配置文件；其余训练目录相对于工作目录"},
        "center_url": "http://127.0.0.1:9", "node_id": "1" * 32,
        "token": "ylp_" + "s" * 32, "enable_training": True,
        "training_cwd": "../project", "allowed_training_roots": ["runs"],
        "training_output_roots": ["runs"], "yolo_bin": "./bin/yolo",
        "training_lock_file": "./workspace.lock", **overrides,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))
    path.chmod(0o600)
    return path, project


def test_commented_relative_config_is_independent_of_shell_directory(agent, tmp_path, monkeypatch):
    path, project = write_relative_config(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = agent.load_config(path)
    assert loaded["training_cwd"] == str(project)
    assert loaded["allowed_training_roots"] == [str(project / "runs")]
    assert loaded["training_output_roots"] == [str(project / "runs")]
    assert loaded["model_test_roots"] == [str(project / "pt")]
    assert loaded["model_transfer_finalize_timeout_seconds"] == 1800
    assert not Path(loaded["model_test_roots"][0]).exists()
    assert loaded["training_lock_file"] == str(project / "workspace.lock")
    assert loaded["yolo_bin"] == str(project / "bin/yolo")
    assert loaded["training_state_dir"] == str(path.parent / ".training-state")
    # A deployment moved as a unit resolves against the new config location.
    moved = tmp_path / "moved"
    shutil.copytree(path.parent, moved / "agent")
    shutil.copytree(project, moved / "project")
    relocated = agent.load_config(moved / "agent/agent-config.json")
    assert relocated["training_output_roots"] == [str(moved / "project/runs")]
    assert relocated["model_test_roots"] == [str(moved / "project/pt")]


@pytest.mark.parametrize("value", [59, 7201, True])
def test_model_transfer_finalize_timeout_is_bounded_integer(agent, tmp_path, value):
    path, _ = write_relative_config(
        tmp_path, model_transfer_finalize_timeout_seconds=value,
    )
    with pytest.raises(ValueError, match="model_transfer_finalize_timeout_seconds"):
        agent.load_config(path)


def test_relative_outputs_preserve_containment_and_symlink_protection(agent, tmp_path):
    path, project = write_relative_config(tmp_path, training_output_roots=["runs/../../outside"])
    with pytest.raises(ValueError, match="training_output_roots 必须位于"):
        agent.load_config(path)
    (tmp_path / "outside").mkdir()
    (project / "runs/link").symlink_to(tmp_path / "outside", target_is_directory=True)
    write_relative_config(tmp_path, training_output_roots=["runs/link"])
    with pytest.raises(ValueError, match="training_output_roots 必须位于"):
        agent.load_config(path)


def test_relative_live_agent_config_still_blocks_overlapping_deletion(agent, tmp_path, monkeypatch):
    path, project = write_relative_config(tmp_path)
    original = Path.iterdir
    monkeypatch.setattr(Path, "iterdir", lambda self: iter([Path("/proc/987654321")])
                        if self == Path("/proc") else original(self))
    monkeypatch.setattr(agent, "_process_argv", lambda pid: [])
    monkeypatch.setattr(agent, "_agent_config_path", lambda pid, argv: path)
    assert agent._other_live_agent_conflicts({}, [project / "runs"]) is True
    assert agent._other_live_agent_conflicts({}, [project / "runs-other"]) is False
    # Unreadable/invalid configuration must not silently remove ownership protection.
    path.write_text('{"training_output_roots": [null]}')
    assert agent._other_live_agent_conflicts({}, [project / "runs-other"]) is True


def test_output_mismatch_explains_both_paths_before_creating_run(agent, training):
    config, task = training
    allowed = Path(config["training_cwd"]) / "different-runs"
    allowed.mkdir()
    config["training_output_roots"] = [str(allowed)]
    with pytest.raises(ValueError) as failure:
        agent.build_training_command(config, task)
    assert "training_output_roots" in str(failure.value)
    assert str(allowed) in str(failure.value)
    assert task["parameters"]["remote_output_root"] in str(failure.value)
    assert not Path(task["run_dir"]).exists()


def test_capability_reports_actual_outputs_with_legacy_fallback(agent, training, monkeypatch):
    config, _ = training
    config["enable_training"] = True
    monkeypatch.setattr(agent, "_configured_yolo_version", lambda _: "test")
    capability = agent.training_capability(config)
    assert capability["ready"] is True
    assert capability["output_roots"] == config["training_output_roots"]
    assert capability["model_test_roots"] == []
    missing_model_root = Path(config["training_cwd"]) / "missing-model-tests"
    config["model_test_roots"] = [str(missing_model_root)]
    capability = agent.training_capability(config)
    assert capability["ready"] is True
    assert capability["model_test_roots"] == [str(missing_model_root)]
    config.pop("training_output_roots")
    assert agent.training_capability(config)["output_roots"] == config["allowed_training_roots"]


def test_broken_output_link_reports_unavailable_without_crashing_monitor(agent, tmp_path):
    path, project = write_relative_config(tmp_path)
    config = agent.load_config(path)
    output = project / "runs"
    output.rmdir()
    output.symlink_to("runs", target_is_directory=True)
    capability = agent.training_capability(config)
    assert capability["ready"] is False
    assert capability["output_roots"] == []
    assert capability["reason"]
    with pytest.raises(ValueError, match="路径无法解析"):
        agent.load_config(path)
