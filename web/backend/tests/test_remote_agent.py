"""Remote execution regressions, using a small dummy process rather than GPU training."""
import importlib.util
import json
import os
import shutil
import sys
import threading
from pathlib import Path
from urllib.error import URLError

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
    config = {"node_id": "b" * 32, "allowed_training_roots": [str(root)], "yolo_bin": sys.executable,
              "training_cwd": str(root), "training_state_dir": str(tmp_path / ".training-state")}
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
    path = agent.save_pending_final(config, task["run_id"], payload)
    assert os.stat(path).st_mode & 0o077 == 0
    assert "token" not in path.read_text()

    def offline(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(agent, "api_json", offline)
    assert agent.retry_pending_finals(config) is False
    assert json.loads(path.read_text()) == payload
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


def test_external_log_symlink_cannot_escape_allowed_root(agent, training, tmp_path):
    config, task = training
    outside = tmp_path / "outside"
    outside.mkdir()
    (Path(task["parameters"]["remote_output_root"]) / ".platform-logs").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="日志路径"):
        agent.build_training_command(config, task)
    assert not Path(task["run_dir"]).exists()


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
