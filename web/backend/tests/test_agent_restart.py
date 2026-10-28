"""Real Agent processes and loopback HTTP; no GPU, live node or user data."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def wait_for(check, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(0.04)
    raise AssertionError("simulation timed out")


@pytest.mark.parametrize("policy", ["preserve", "stop"])
def test_real_agent_reload_and_restart(tmp_path, policy):
    source = Path(__file__).resolve().parents[2] / "agent/monitor_agent.py"
    spec = importlib.util.spec_from_file_location("restart_probe", source)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    training = tmp_path / "training"
    training.mkdir()
    (training / "model.pt").write_bytes(b"simulation")
    (training / "data.yaml").write_text("train: images\nval: images\n")
    outputs = training / "runs"
    outputs.mkdir()
    config = {"center_url": "", "node_id": "b" * 32, "token": "ylp_" + "x" * 32,
              "enable_training": True, "report_interval_seconds": 5, "training_cwd": "training",
              "allowed_training_roots": ["."], "training_output_roots": ["runs"],
              "yolo_bin": sys.executable, "enable_training_trace": False}
    task = {"run_id": "a" * 32, "task_id": "restart-simulation", "attempt_number": 1,
            "task_type": "detect", "model_path": str(training / "model.pt"),
            "data_path": str(training / "data.yaml"),
            "run_dir": str(outputs / "detect/restart-simulation/attempt-001"),
            "log_path": str(outputs / ".platform-logs/20261028/restart-simulation/attempt-001.log"),
            "parameters": {"epochs": 100000, "batch": 1, "imgsz": 32, "device": "cpu",
                           "workers": 0, "optimizer": "auto", "remote_output_root": str(outputs)}}
    updates, reports, stops, claimed = [], [], [], []
    refuse_reports = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            self.respond()

        def do_POST(self):
            self.respond()

        def respond(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            code, response = 200, {}
            if self.path.endswith("/report"):
                reports.append(payload)
                if refuse_reports.is_set():
                    code = 422
                response = {"report_interval_seconds": 5}
            elif self.path.endswith("/agent/poll"):
                response = {"task": None if claimed else task}
                claimed.append(True)
            elif self.path.endswith("/control"):
                response = {"status": "stopping" if stops else "running", "stop": bool(stops)}
            elif self.path.endswith("/update"):
                updates.append(payload)
                response = {"status": payload["status"], "stop_requested": bool(stops)}
            data = json.dumps(response).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config["center_url"] = f"http://127.0.0.1:{server.server_port}"
    config_path = tmp_path / "agent-config.json"
    config_path.write_text(json.dumps(config))
    config_path.chmod(0o600)
    trainer = tmp_path / "simulate_training.py"
    trainer.write_text(
        "import pathlib,sys,time\n"
        "root=pathlib.Path(sys.argv[1])\n"
        "with (root/'results.csv').open('w',buffering=1) as f:\n"
        " f.write('epoch,time,score\\n')\n"
        " for epoch in range(1,100000):\n"
        "  f.write(f'{epoch},{epoch},0.5\\n'); print(f'epoch {epoch}',flush=True); time.sleep(.1)\n")
    # This harness replaces only hardware/YOLO; the shipped main, HTTP, durable
    # state, locks, signal handlers, Popen and execv all execute unchanged.
    harness = tmp_path / "agent_harness.py"
    harness.write_text(
        "import importlib.util,pathlib,sys\n"
        f"spec=importlib.util.spec_from_file_location('real_agent',{str(source)!r})\n"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
        "build=m.build_training_command\n"
        "def command(config,task):\n"
        " build(config,task)\n"
        f" return [sys.executable,{str(trainer)!r},task['run_dir']]\n"
        "m.build_training_command=command\n"
        "m.collect=lambda *_: {'compute':{'gpus':[]},'training':{'ready':True}}\n"
        "m.__file__=__file__\n"
        "raise SystemExit(m.main())\n")
    processes, training_pid = [], None
    log = (tmp_path / "agent.log").open("w")
    command = [sys.executable, str(harness), "--config", str(config_path)]
    if policy == "stop":
        command.extend(["--shutdown-policy", policy])
    state_path = tmp_path / ".training-state/active" / f"{task['run_id']}.json"
    def state():
        try:
            return json.loads(state_path.read_text())
        except (OSError, ValueError):
            return {}
    def start():
        process = subprocess.Popen(command, cwd=tmp_path, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                                                               "XDG_STATE_HOME": str(tmp_path / "host-state")})
        processes.append(process)
        return process
    try:
        process = start()
        identity = wait_for(lambda: state().get("process"))
        training_pid = identity["pid"]
        wait_for(lambda: any(update.get("completed_epochs", 0) > 0 for update in updates))
        # HUP exec reload must retain MAINPID and the exact same training process,
        # even when telemetry is rejected after restart.
        refuse_reports.set()
        count = len(updates)
        process.send_signal(signal.SIGHUP)
        wait_for(lambda: len(reports) >= 2 and len(updates) > count)
        assert process.poll() is None
        assert state()["process"] == identity
        assert all(update["status"] == "running" for update in updates)
        assert updates[-1]["completed_epochs"] > 0
        if policy == "preserve":
            # Default TERM exits only the Agent. A new Agent adopts the old PID
            # before a failing monitoring report can block progress reporting.
            process.terminate()
            assert process.wait(timeout=10) == 0
            assert agent.process_identity_matches(identity)
            count = len(updates)
            process = start()
            wait_for(lambda: len(updates) > count)
            assert state()["process"] == identity
            assert len(claimed) == 1
            stops.append(True)
            wait_for(lambda: updates[-1]["status"] == "stopped")
            assert updates[-1]["stop_reason"] == "platform_stop"
        else:
            process.terminate()
            assert process.wait(timeout=10) == 0
            assert updates[-1]["status"] == "stopped"
            assert updates[-1]["stop_reason"] == "agent_shutdown"
        assert not agent.process_identity_matches(identity)
        assert not state_path.exists()
        assert len(claimed) == 1
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if training_pid and agent.process_identity(training_pid):
            try:
                os.killpg(training_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        log.close()
