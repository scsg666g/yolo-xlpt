"""Remote execution regressions, using a small dummy process rather than GPU training."""
import importlib.util
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
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
    if scenario == "all_epochs":
        (run_dir / "results.csv").write_text("epoch,time,score\n1,1,0.1\n2,2,0.2\n3,3,0.3\n")
        log.write_text("training output\n")
    else:
        (run_dir / "results.csv").write_text("epoch,time,score\n1,1,0.1\n")
        (weights / "best.pt").write_bytes(b"best")
        (weights / "last.pt").write_bytes(b"last")
        marker = "EarlyStopping: Training stopped early\n" if scenario == "early_stopping" else "training vanished\n"
        log.write_text(marker + "".join(f"final validation line {index}\n" for index in range(200)))
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
    assert updates[-1]["completed_epochs"] == (3 if scenario == "all_epochs" else 1)
    assert updates[-1]["progress"] == (100 if expected == "completed" else pytest.approx(100 / 3))
    if expected == "interrupted":
        assert "无法取得原进程退出码" in updates[-1]["error_message"]


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
