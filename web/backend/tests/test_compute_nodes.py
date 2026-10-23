import importlib.util
import base64
import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.compute_nodes as monitoring
from app.database import Base, get_db
from app.main import app
from app.models import ComputeNode, RemoteOperation, TaskEvent, TrainingRun, TrainingTask, utcnow
from app.parameter_catalog import training_parameter_keys
from sqlalchemy import select
from app.services import create_training_run


def sample(gpus=None):
    return {"agent_version": "1.8.0", "compute": {
        "available": bool(gpus), "sampled_at": utcnow().isoformat(), "provider": "NVIDIA" if gpus else None,
        "message": None, "host": {"hostname": "remote-host", "platform": "Linux", "cpu_count": 8,
                                 "memory_total": 16000, "memory_used": 4000, "load_average": [1, 2, 3]},
        "gpus": gpus or [],
    }, "training": {"enabled": True, "ready": True, "supported_parameters": []}}


@pytest.fixture
def monitoring_client(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'monitor.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    def override_db():
        with factory() as db:
            yield db
    with factory() as db:
        monitoring.ensure_local_node(db)
    monkeypatch.setattr(monitoring, "local_snapshot", lambda: sample()["compute"])
    app.dependency_overrides[get_db] = override_db
    try:
        yield TestClient(app), factory
    finally:
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def create(client, name="Remote"):
    response = client.post("/api/compute/nodes", json={"name": name, "address": "10.180.0.123"})
    assert response.status_code == 201, response.text
    return response.json()


def headers(credential):
    return {"Authorization": "Bearer " + credential["token"]}


def parsed_time(value: str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_local_compute_sampling_does_not_block_api_caller(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    expected = sample()["compute"]
    expected["sampled_at"] = utcnow().isoformat()

    def slow_sample():
        started.set()
        release.wait(timeout=2)
        return expected

    monkeypatch.setattr(monitoring, "compute_inventory", slow_sample)
    monkeypatch.setattr(monitoring, "_local_cache", {"available": False, "gpus": [], "host": {},
                                                       "sampled_at": utcnow().isoformat()})
    monkeypatch.setattr(monitoring, "_local_sampled", 0.0)
    monkeypatch.setattr(monitoring, "_local_refreshing", False)
    try:
        started_at = time.monotonic()
        cached = monitoring.local_snapshot()
        assert time.monotonic() - started_at < 0.2
        assert cached["available"] is False
        assert started.wait(timeout=1)
        release.set()
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and monitoring.local_snapshot() is not expected:
            time.sleep(0.01)
        assert monitoring.local_snapshot() == expected
    finally:
        release.set()


def test_node_credentials_isolation_and_persistence(monitoring_client):
    client, factory = monitoring_client
    a, b = create(client), create(client, "Other")
    aid, bid = a["node"]["id"], b["node"]["id"]
    assert a["token"] != b["token"]
    assert a["node"]["status"] == "pending"
    with factory() as db:
        node = db.get(ComputeNode, aid)
        assert node.token_hash != a["token"]
        assert len(node.token_hash) == 64
    for node_id, auth in ((aid, {}), (aid, headers(b)), (bid, headers(a)), ("local", headers(a))):
        assert client.post(f"/api/compute/nodes/{node_id}/report", json=sample(), headers=auth).status_code == 401
    gpu = {"index": 0, "name": "GPU X", "memory_total": 24000, "memory_used": 6000, "utilization": 50,
           "processes": [{"pid": 100, "name": "python", "memory_used": 6000}]}
    assert client.post(f"/api/compute/nodes/{aid}/report", json=sample([gpu]), headers=headers(a)).status_code == 200
    detail = client.get(f"/api/compute/nodes/{aid}").json()
    assert detail["status"] == "online"
    assert detail["summary"] == {"gpu_count": 1, "utilization": 50, "memory_used": 6000, "memory_total": 24000, "process_count": 1}
    assert detail["compute"]["host"]["hostname"] == "remote-host"
    assert detail["compute"]["gpus"][0]["processes"][0]["association"]["state"] == "unknown"
    assert client.get(f"/api/compute/nodes/{bid}").json()["compute"] is None
    listing = client.get("/api/compute/nodes").json()
    assert listing[0]["id"] == "local"
    assert all("compute" not in item and "token_hash" not in item and "token" not in item for item in listing)
    assert a["token"] not in json.dumps(detail)


def test_remote_gpu_process_links_only_to_a_verified_active_run(monitoring_client, monkeypatch):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    instance_id = "c" * 32
    monkeypatch.setattr(monitoring, "platform_instance_id", lambda: instance_id)
    task = TrainingTask(
        id="gpu-task", name="Remote fire", task_type="detect", status="running",
        execution_node_id=node_id, model_path="/remote/model.pt", data_path="/remote/data.yaml", config={},
    )
    run = TrainingRun(
        id="a" * 32, task=task, attempt_number=2, launch_mode="fresh", status="running",
        execution_node_id=node_id, model_path=task.model_path, data_path=task.data_path, pid=1234,
    )
    with factory() as db:
        db.add_all([task, run])
        db.commit()

    payload = sample([{
        "index": 0, "uuid": "GPU-1", "name": "GPU X", "memory_total": 24000,
        "memory_used": 6000, "utilization": 50, "processes": [{
            "pid": 5678, "name": "python", "memory_used": 6000,
            "owner_hint": {
                "instance_id": instance_id, "node_id": node_id, "run_id": run.id,
                "task_id": task.id, "match_method": "runtime_marker", "scan_complete": True,
            },
        }],
    }])
    payload["agent_version"] = "1.9.0"
    response = client.post(
        f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=payload,
    )
    assert response.status_code == 200, response.text
    process = client.get(f"/api/compute/nodes/{node_id}").json()["compute"]["gpus"][0]["processes"][0]
    assert process["association"]["state"] == "managed"
    assert process["association"]["task_id"] == task.id
    assert process["association"]["run_id"] == run.id
    assert process["association"]["task_name"] == task.name
    assert process["association"]["attempt_number"] == 2

    payload["compute"]["gpus"][0]["processes"][0]["owner_hint"]["instance_id"] = "f" * 32
    response = client.post(
        f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=payload,
    )
    assert response.status_code == 200, response.text
    foreign = client.get(f"/api/compute/nodes/{node_id}").json()["compute"]["gpus"][0]["processes"][0]["association"]
    assert foreign["state"] == "other_platform"
    assert "task_name" not in foreign


def test_offline_disabled_rotation_and_removal(monitoring_client):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    url = f"/api/compute/nodes/{node_id}"
    report = sample()
    assert client.post(url + "/report", headers=headers(credential), json=report).status_code == 200
    with factory() as db:
        db.get(ComputeNode, node_id).last_seen_at = utcnow() - timedelta(seconds=31)
        db.commit()
    offline = client.get(url).json()
    assert offline["status"] == "offline" and offline["compute"] is not None
    settings = {"name": "Renamed", "enabled": False, "report_interval_seconds": 10}
    assert client.patch(url, json=settings).json()["status"] == "disabled"
    assert client.post(url + "/report", headers=headers(credential), json=sample()).status_code == 403
    settings["enabled"] = True
    assert client.patch(url, json=settings).status_code == 200
    replacement = client.post(url + "/token").json()
    assert replacement["node"]["status"] == "pending"
    assert client.post(url + "/report", headers=headers(credential), json=sample()).status_code == 401
    assert client.post(url + "/report", headers=headers(replacement), json=sample()).status_code == 200
    assert client.delete(url).status_code == 204
    assert client.post(url + "/report", headers=headers(replacement), json=sample()).status_code == 401
    assert client.get(url).status_code == 404
    assert client.delete("/api/compute/nodes/local").status_code == 409
    assert client.post("/api/compute/nodes/local/token").status_code == 409


@pytest.mark.parametrize(("operation_kind", "payload"), [
    ("fetch_model", {"model_asset_id": "a" * 32, "path": "/models/test.pt"}),
    ("fetch_log", {"run_id": "remote-run", "log_path": "/logs/train.log"}),
])
def test_active_file_transfer_protects_node_management(monitoring_client, operation_kind, payload):
    client, factory = monitoring_client
    credential = create(client, "Model transfer node")
    node_id = credential["node"]["id"]
    operation = RemoteOperation(
        id="f" * 32,
        node_id=node_id,
        task_id="model-asset:" + "a" * 32,
        kind=operation_kind,
        status="pending",
        payload=payload,
    )
    with factory() as db:
        db.add(operation)
        db.commit()

    settings = {"name": "Model transfer node", "enabled": False}
    assert client.patch(f"/api/compute/nodes/{node_id}", json=settings).status_code == 409
    assert client.delete(f"/api/compute/nodes/{node_id}").status_code == 409

    # A pending operation must not deadlock recovery when the old token is
    # unavailable; the replacement Agent can report and claim it later.
    assert client.post(f"/api/compute/nodes/{node_id}/token").status_code == 200
    with factory() as db:
        stored = db.get(RemoteOperation, operation.id)
        stored.status = "running"
        db.commit()

    assert client.post(f"/api/compute/nodes/{node_id}/token").status_code == 409
    assert client.patch(f"/api/compute/nodes/{node_id}", json=settings).status_code == 409
    assert client.delete(f"/api/compute/nodes/{node_id}").status_code == 409

    with factory() as db:
        stored = db.get(RemoteOperation, operation.id)
        stored.status = "succeeded"
        db.commit()
    assert client.patch(f"/api/compute/nodes/{node_id}", json=settings).status_code == 200
    assert client.delete(f"/api/compute/nodes/{node_id}").status_code == 204


def test_validation_stale_reports_and_cpu_only_nodes(monitoring_client):
    client, _ = monitoring_client
    for settings in ({"name": "  "}, {"name": "a", "password": "secret"}, {"name": "a", "report_interval_seconds": 1}):
        assert client.post("/api/compute/nodes", json=settings).status_code == 422
    credential = create(client)
    url = f"/api/compute/nodes/{credential['node']['id']}/report"
    payload = sample()
    assert client.post(url, headers=headers(credential), json=payload).status_code == 200
    older = sample()
    older["compute"]["sampled_at"] = (utcnow() - timedelta(hours=1)).isoformat()
    stale = client.post(url, headers=headers(credential), json=older)
    assert stale.status_code == 200 and stale.json()["status"] == "stale"
    detail = client.get(url.removesuffix("/report")).json()
    assert detail["status"] == "online" and detail["compute"]["available"] is False
    assert parsed_time(detail["compute"]["sampled_at"]) == parsed_time(payload["compute"]["sampled_at"])
    payload["compute"]["host"]["load_average"] = [-1]
    assert client.post(url, headers=headers(credential), json=payload).status_code == 422
    assert client.post(url, headers=headers(credential), content=b"x" * (monitoring.MAX_REPORT_BYTES + 1)).status_code == 413


def test_ordered_agent_reports_survive_clock_rollback_and_reject_replays(monitoring_client):
    client, _ = monitoring_client
    credential = create(client)
    url = f"/api/compute/nodes/{credential['node']['id']}/report"
    first = sample()
    first.update(agent_version="2.0.6", boot_id="a" * 32, report_seq=1)
    assert client.post(url, headers=headers(credential), json=first).status_code == 200

    rollback = sample()
    rollback.update(agent_version="2.0.6", boot_id="a" * 32, report_seq=2)
    rollback["compute"]["sampled_at"] = (utcnow() - timedelta(days=2)).isoformat()
    rollback["training"]["output_writable"] = True
    accepted = client.post(url, headers=headers(credential), json=rollback)
    assert accepted.status_code == 200 and accepted.json()["status"] == "ok"
    detail = client.get(url.removesuffix("/report")).json()
    assert parsed_time(detail["sampled_at"]) == parsed_time(rollback["compute"]["sampled_at"])
    assert detail["training"]["output_writable"] is True

    replay = sample()
    replay.update(agent_version="9.9.9", boot_id="a" * 32, report_seq=1)
    stale = client.post(url, headers=headers(credential), json=replay)
    assert stale.status_code == 200 and stale.json()["status"] == "stale"
    assert client.get(url.removesuffix("/report")).json()["agent_version"] == "2.0.6"

    restarted = sample()
    restarted.update(agent_version="2.0.6", boot_id="b" * 32, report_seq=1)
    restarted["compute"]["sampled_at"] = (utcnow() - timedelta(days=3)).isoformat()
    assert client.post(url, headers=headers(credential), json=restarted).json()["status"] == "ok"
    assert parsed_time(client.get(url.removesuffix("/report")).json()["sampled_at"]) == parsed_time(
        restarted["compute"]["sampled_at"],
    )


def test_remote_online_status_uses_center_receive_time_when_agent_clock_lags(monitoring_client):
    client, factory = monitoring_client
    credential = create(client)
    with factory() as db:
        node = db.get(ComputeNode, credential["node"]["id"])
        node.agent_version = "1.8.0"
        node.last_seen_at = utcnow()
        node.snapshot = {
            "sampled_at": (utcnow() - timedelta(days=2)).isoformat(),
            "host": {"hostname": "clock-behind"}, "gpus": [],
            "training": {"enabled": True, "ready": True, "supported_parameters": []},
        }
        db.commit()
    detail = client.get(f"/api/compute/nodes/{credential['node']['id']}").json()
    assert detail["status"] == "online"
    assert detail["required_agent_version"] == "1.8.0"
    assert detail["training_version_supported"] is True
    assert detail["training_unavailable_reason"] is None


def test_serialized_node_uses_platform_training_gate_over_agent_ready_flag(monitoring_client):
    client, _ = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    report = sample()
    report["agent_version"] = "1.7.0"
    assert report["training"]["ready"] is True
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=report).status_code == 200

    detail = client.get(f"/api/compute/nodes/{node_id}").json()
    assert detail["training"]["ready"] is True
    assert detail["training_version_supported"] is False
    assert detail["training_unavailable_reason"] == "请将远程 Agent 升级到 1.8.0 或以上版本"

    current = sample()
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=current).status_code == 200
    current_detail = client.get(f"/api/compute/nodes/{node_id}").json()
    assert current_detail["training_version_supported"] is True
    assert current_detail["training_unavailable_reason"] is None


def test_agent_download_and_cpu_report_schema(monitoring_client, tmp_path, monkeypatch):
    client, _ = monitoring_client
    response = client.get("/api/compute/agent/download")
    assert response.status_code == 200 and "monitor_agent.py" in response.headers["content-disposition"]
    script = Path(__file__).resolve().parents[2] / "agent" / "monitor_agent.py"
    spec = importlib.util.spec_from_file_location("monitor_agent", script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    monkeypatch.setattr(agent, "query", lambda *_: (_ for _ in ()).throw(FileNotFoundError()))
    payload = agent.collect()
    validated = monitoring.AgentReport.model_validate(payload)
    assert validated.compute.available is False and validated.compute.host.hostname
    credential = create(client)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"center_url": "https://example.com", "node_id": credential["node"]["id"], "token": credential["token"]}))
    os.chmod(config_path, 0o600)
    loaded = agent.load_config(config_path)
    assert loaded["report_interval_seconds"] == 5
    assert loaded["training_cwd"] == str(config_path.parent.resolve())
    os.chmod(config_path, 0o644)
    with pytest.raises(ValueError):
        agent.load_config(config_path)
    assert agent.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other.example") is None


def test_agent_gpu_collection_with_unavailable_fields(monitoring_client, monkeypatch):
    script = Path(__file__).resolve().parents[2] / "agent" / "monitor_agent.py"
    spec = importlib.util.spec_from_file_location("monitor_gpu_agent", script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    def query(_binary, arguments):
        if "compute-apps" in arguments[0]:
            return "GPU-1, 100, /bin/python, 2500\nGPU-1, 101, python, [N/A]"
        return "0, GPU-1, NVIDIA GPU, 570, 24000, 2500, 21500, 75, 20, 60, [N/A], 300, 1800, 7000"
    monkeypatch.setattr(agent, "query", query)
    payload = monitoring.AgentReport.model_validate(agent.collect())
    gpu = payload.compute.gpus[0]
    assert gpu.power_draw is None and gpu.utilization == 75
    assert gpu.processes[0].name == "python" and gpu.processes[1].memory_used is None


def test_remote_claim_update_stop_and_node_isolation(monitoring_client):
    client, factory = monitoring_client
    first, second = create(client), create(client, "Other")
    node_id = first["node"]["id"]
    task = TrainingTask(id="remote-test", name="remote", task_type="detect", execution_node_id=node_id,
                        model_path="/remote/model.pt", data_path="/remote/data.yaml",
                        config={"remote_output_root": "/remote/runs"})
    with factory() as db:
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run_id = run.id
        db.commit()
    poll_url = f"/api/compute/nodes/{node_id}/agent/poll"
    assert client.post(poll_url).status_code == 401
    assert client.post(poll_url, headers=headers(second)).status_code == 401
    assert client.post(f"/api/compute/nodes/{second['node']['id']}/agent/poll", headers=headers(second)).json()["task"] is None
    blocked = client.post(poll_url, headers=headers(first)).json()
    assert blocked["task"] is None and blocked["required_agent_version"] == "1.8.0"
    with factory() as db:
        assert db.get(TrainingRun, run_id).status == "queued"
    old_report = sample()
    old_report["agent_version"] = "1.7.0"
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(first), json=old_report).status_code == 200
    assert client.post(poll_url, headers=headers(first)).json()["required_agent_version"] == "1.8.0"
    unready_report = sample()
    unready_report["training"].update(ready=False, reason="output root is read-only")
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(first), json=unready_report).status_code == 200
    unready = client.post(poll_url, headers=headers(first)).json()
    assert unready["task"] is None and unready["message"] == "output root is read-only"
    with factory() as db:
        assert db.get(TrainingRun, run_id).status == "queued"
    report = sample()
    report["agent_version"] = "1.8.0"
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(first), json=report).status_code == 200
    claimed = client.post(poll_url, headers=headers(first)).json()["task"]
    assert claimed["run_id"] == run_id
    assert claimed["run_dir"] == "/remote/runs/detect/remote-test/attempt-001"
    log_path = Path(claimed["log_path"])
    assert log_path.parent.parent.parent == Path("/remote/runs/.platform-logs")
    assert log_path.name == "attempt-001.log" and log_path.parent.name == "remote-test"
    assert client.post(poll_url, headers=headers(first)).json()["task"] is None
    lease = claimed["claim_lease"]
    confirm_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm"
    assert client.post(confirm_url, headers=headers(first), json={"lease": lease}).status_code == 200
    update_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update"
    assert client.post(update_url, headers=headers(second), json={"status": "running"}).status_code == 401
    lease_headers = {**headers(first), "X-Training-Lease": lease}
    update = client.post(update_url, headers=lease_headers, json={"status": "running", "pid": 1234,
        "completed_epochs": 1, "progress": 10, "latest_metrics": {"score": 0.5},
        "metrics": [{"epoch": 1, "values": {"score": 0.5}}], "log_tail": ["epoch 1"]})
    assert update.status_code == 200, update.text
    assert client.get("/api/tasks/remote-test/logs", params={"tail": 20}).json()["lines"] == ["epoch 1"]
    assert len(client.get("/api/tasks/remote-test/metrics").json()) == 1
    assert client.post("/api/tasks/remote-test/stop").status_code == 200
    control = client.get(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/control", headers=headers(first))
    assert control.json()["stop"] is True
    reply = client.post(update_url, headers=lease_headers, json={"status": "running", "pid": 1234})
    assert reply.json()["status"] == "stopping" and reply.json()["stop_requested"] is True
    assert client.post(update_url, headers=lease_headers,
                       json={"status": "stopped", "exit_code": -15}).status_code == 200
    with factory() as db:
        assert db.get(TrainingRun, run_id).status == "stopped"
        from app.models import GpuReservation
        assert db.scalar(select(GpuReservation.resource_key).where(GpuReservation.run_id == run_id)) is None
        assert db.get(TrainingRun, run_id).error_message == "收到平台停止指令，训练进程已停止"
        event = db.scalar(select(TaskEvent).where(TaskEvent.run_id == run_id, TaskEvent.event_type == "stop_requested"))
        assert event.payload["source_ip"] == "testclient"
        assert event.payload["run_id"] == run_id
        assert "请求来源：testclient" in event.message


def _queue_remote_run(db, node_id, task_id, device):
    task = TrainingTask(
        id=task_id, name=task_id, task_type="detect", execution_node_id=node_id,
        model_path="/remote/model.pt", data_path="/remote/data.yaml",
        device=device, config={"remote_output_root": "/remote/runs"},
    )
    db.add(task)
    db.flush()
    run = create_training_run(db, task, "fresh")
    run.parameter_snapshot = {**(run.parameter_snapshot or {}), "device": device}
    return run


def test_parallel_agent_requires_real_gpu_uuid_and_skips_busy_head_of_line(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Parallel Agent")
    node_id = credential["node"]["id"]
    node_gpus = [
        {"index": 0, "uuid": "GPU-A", "name": "A"},
        {"index": 1, "uuid": "GPU-B", "name": "B"},
    ]
    report = sample(node_gpus)
    report["training"]["gpu_parallelism"] = True
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=report).status_code == 200
    with factory() as db:
        first = _queue_remote_run(db, node_id, "parallel-first", "0")
        second = _queue_remote_run(db, node_id, "parallel-second", "1")
        third = _queue_remote_run(db, node_id, "parallel-third", "2")
        third_id = third.id
        db.commit()

    endpoint = f"/api/compute/nodes/{node_id}/agent/poll"
    first_claim = client.post(endpoint, headers=headers(credential)).json()["task"]
    assert first_claim["gpu_assignment"] == [{"uuid": "GPU-A", "index": 0}]
    second_claim = client.post(endpoint, headers=headers(credential)).json()["task"]
    assert second_claim["gpu_assignment"] == [{"uuid": "GPU-B", "index": 1}]
    no_claim = client.post(endpoint, headers=headers(credential)).json()
    assert no_claim["task"] is None
    with factory() as db:
        assert db.get(TrainingRun, third_id).status == "queued"


def test_parallel_agent_fails_closed_for_missing_uuid_or_unknown_gpu_index(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "UUID Agent")
    node_id = credential["node"]["id"]
    report = sample([{"index": 0, "uuid": "", "name": "No UUID"}])
    report["training"]["gpu_parallelism"] = True
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=report).status_code == 200
    with factory() as db:
        missing = _queue_remote_run(db, node_id, "missing-uuid", "0")
        unknown = _queue_remote_run(db, node_id, "unknown-index", "4")
        ids = {missing.id, unknown.id}
        db.commit()
    response = client.post(f"/api/compute/nodes/{node_id}/agent/poll", headers=headers(credential)).json()
    assert response["task"] is None and "GPU" in response["message"]
    with factory() as db:
        assert all(db.get(TrainingRun, run_id).status == "queued" for run_id in ids)


def test_remote_cpu_claim_is_serialized_with_gpu_claim(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "CPU Serializer")
    node_id = credential["node"]["id"]
    report = sample([{"index": 0, "uuid": "GPU-A", "name": "A"}])
    report["training"]["gpu_parallelism"] = True
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=report).status_code == 200
    with factory() as db:
        _queue_remote_run(db, node_id, "gpu-first", "0")
        cpu = _queue_remote_run(db, node_id, "cpu-second", "cpu")
        cpu_id = cpu.id
        db.commit()
    endpoint = f"/api/compute/nodes/{node_id}/agent/poll"
    assert client.post(endpoint, headers=headers(credential)).json()["task"]["task_id"] == "gpu-first"
    assert client.post(endpoint, headers=headers(credential)).json()["task"] is None
    with factory() as db:
        assert db.get(TrainingRun, cpu_id).status == "queued"


def test_remote_legacy_agent_remains_node_serial(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Old Agent")
    node_id = credential["node"]["id"]
    report = sample([{"index": 0, "uuid": "GPU-A", "name": "A"},
                     {"index": 1, "uuid": "GPU-B", "name": "B"}])
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=report).status_code == 200
    with factory() as db:
        _queue_remote_run(db, node_id, "legacy-first", "0")
        second = _queue_remote_run(db, node_id, "legacy-second", "1")
        second_id = second.id
        db.commit()
    endpoint = f"/api/compute/nodes/{node_id}/agent/poll"
    first = client.post(endpoint, headers=headers(credential)).json()["task"]
    assert first["task_id"] == "legacy-first" and first["gpu_assignment"] == []
    assert client.post(endpoint, headers=headers(credential)).json()["task"] is None
    with factory() as db:
        assert db.get(TrainingRun, second_id).status == "queued"


def test_remote_claim_lease_requires_confirmation(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Leased Agent")
    node_id = credential["node"]["id"]
    task = TrainingTask(id="leased-remote", name="leased", task_type="detect", execution_node_id=node_id,
                        model_path="/remote/model.pt", data_path="/remote/data.yaml",
                        config={"remote_output_root": "/remote/runs"})
    with factory() as db:
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run_id = run.id
        db.commit()
    report = sample()
    report["agent_version"] = "1.8.0"
    report["training"] = {"enabled": True, "ready": True, "supported_parameters": []}
    assert client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=report).status_code == 200
    task_payload = client.post(f"/api/compute/nodes/{node_id}/agent/poll", headers=headers(credential)).json()["task"]
    lease = task_payload["claim_lease"]
    update_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update"
    assert client.post(update_url, headers=headers(credential), json={"status": "running"}).status_code == 409
    assert client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm", headers=headers(credential),
                       json={"lease": "0" * 32}).status_code == 409
    confirm_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm"
    first_confirmation = client.post(confirm_url, headers=headers(credential), json={"lease": lease})
    assert first_confirmation.status_code == 200
    # Treat the first HTTP response as lost. A delayed retry of the same
    # committed confirmation must remain successful after the lease timestamp.
    with factory() as db:
        db.get(TrainingRun, run_id).claim_expires_at = utcnow() - timedelta(seconds=1)
        db.commit()
    repeated_confirmation = client.post(confirm_url, headers=headers(credential), json={"lease": lease})
    assert repeated_confirmation.status_code == 200
    assert repeated_confirmation.json() == {"confirmed": True, "stop": False}
    update = client.post(update_url, headers={**headers(credential), "X-Training-Lease": lease},
                         json={"status": "running", "pid": 1234})
    assert update.status_code == 200
    assert client.post(confirm_url, headers=headers(credential), json={"lease": lease}).json() == {
        "confirmed": True, "stop": False,
    }


@pytest.mark.parametrize("terminal_status, stop_requested", [
    ("completed", False), ("failed", False), ("stopped", True), ("interrupted", True),
])
def test_terminal_remote_claim_confirmation_retry_forces_stop(
        monitoring_client, terminal_status, stop_requested):
    client, factory = monitoring_client
    credential = create(client, f"Terminal {terminal_status} Agent")
    node_id = credential["node"]["id"]
    task = TrainingTask(id=f"terminal-{terminal_status}", name="terminal", task_type="detect",
                        execution_node_id=node_id, model_path="/remote/model.pt",
                        data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
    with factory() as db:
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run_id = run.id
        db.commit()

    report = sample()
    report["agent_version"] = "1.8.0"
    report["training"] = {"enabled": True, "ready": True, "supported_parameters": []}
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=report).status_code == 200
    claimed = client.post(f"/api/compute/nodes/{node_id}/agent/poll",
                          headers=headers(credential)).json()["task"]
    lease = claimed["claim_lease"]
    confirm_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm"
    assert client.post(confirm_url, headers=headers(credential), json={"lease": lease}).status_code == 200

    # Simulate a stale Agent state file retrying a claim whose run has already
    # reached a terminal state. Completion and failure normally do not set a
    # stop request, but every terminal state must still prevent a new launch.
    with factory() as db:
        run = db.get(TrainingRun, run_id)
        run.status = terminal_status
        run.stop_requested = stop_requested
        run.claim_expires_at = utcnow() - timedelta(seconds=1)
        db.commit()

    retry = client.post(confirm_url, headers=headers(credential), json={"lease": lease})
    assert retry.status_code == 200
    assert retry.json() == {"confirmed": True, "stop": True, "status": terminal_status}

    update_url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update"
    late_update = client.post(update_url,
                              headers={**headers(credential), "X-Training-Lease": lease},
                              json={"status": "running", "pid": 4321})
    assert late_update.status_code == 200
    assert late_update.json() == {"status": terminal_status, "stop_requested": stop_requested}
    with factory() as db:
        run = db.get(TrainingRun, run_id)
        assert run.status == terminal_status
        assert run.pid is None


def test_remote_task_rejects_parameters_missing_from_agent_capability(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Parameter Agent")
    node_id = credential["node"]["id"]
    all_parameters = training_parameter_keys()

    def report(supported):
        payload = sample()
        payload["agent_version"] = "1.8.0"
        payload["training"] = {
            "enabled": True, "ready": True,
            "supported_parameters": sorted(supported),
        }
        response = client.post(f"/api/compute/nodes/{node_id}/report",
                               headers=headers(credential), json=payload)
        assert response.status_code == 200

    task_payload = {
        "name": "Remote time limit", "execution_node_id": node_id,
        "model_path": "/remote/model.pt", "data_path": "/remote/data.yaml",
        "remote_output_root": "/remote/runs", "advanced": {"time": 1.5},
    }
    report(all_parameters - {"time"})
    unsupported = client.post("/api/tasks", json=task_payload)
    assert unsupported.status_code == 422 and "time" in unsupported.json()["detail"]

    report(all_parameters)
    created = client.post("/api/tasks", json=task_payload)
    assert created.status_code == 201
    task_id = created.json()["id"]

    # A task created before an Agent downgrade must also be blocked at launch.
    report(all_parameters - {"time"})
    start = client.post(f"/api/tasks/{task_id}/start")
    assert start.status_code == 409 and "time" in start.json()["detail"]
    with factory() as db:
        task = db.get(TrainingTask, task_id)
        assert task.status == "draft" and not task.runs


def report_output_roots(client, credential, roots):
    payload = sample()
    payload["agent_version"] = "1.8.1"
    payload["training"].update(output_roots=roots, supported_parameters=sorted(training_parameter_keys()))
    response = client.post(f"/api/compute/nodes/{credential['node']['id']}/report",
                           headers=headers(credential), json=payload)
    assert response.status_code == 200, response.text


def test_output_roots_are_persisted_and_exposed_in_node_list(monitoring_client):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    report_output_roots(client, credential, ["/server/work/runs/", "/server/work/runs", "/outputs/team"])
    expected = ["/server/work/runs", "/outputs/team"]
    assert client.get(f"/api/compute/nodes/{node_id}").json()["training"]["output_roots"] == expected
    listing = client.get("/api/compute/nodes").json()
    assert next(node for node in listing if node["id"] == node_id)["training"]["output_roots"] == expected
    with factory() as db:
        assert db.get(ComputeNode, node_id).snapshot["training"]["output_roots"] == expected
    for roots in (["relative/runs"], ["/invalid\x00path"]):
        payload = sample()
        payload["training"]["output_roots"] = roots
        response = client.post(f"/api/compute/nodes/{node_id}/report", headers=headers(credential), json=payload)
        assert response.status_code == 422


@pytest.mark.parametrize(("output_root", "accepted"), [
    ("/remote/runs", True),
    ("/remote/runs/project", True),
    ("/remote/runs/sub/../project", True),
    ("/remote/runs-other", False),
    ("/remote/runs/../outside", False),
    ("/remote/runs/sub/../../outside", False),
    ("/remote", False),
])
def test_remote_task_create_validates_reported_output_roots(monitoring_client, output_root, accepted):
    client, factory = monitoring_client
    credential = create(client)
    report_output_roots(client, credential, ["/remote/runs"])
    response = client.post("/api/tasks", json={
        "name": "Remote output check", "execution_node_id": credential["node"]["id"],
        "model_path": "/remote/model.pt", "data_path": "/remote/data.yaml",
        "remote_output_root": output_root,
    })
    assert response.status_code == (201 if accepted else 422), response.text
    if not accepted:
        detail = response.json()["detail"]
        assert output_root in detail and "/remote/runs" in detail and "training_output_roots" in detail
        with factory() as db:
            assert db.scalar(select(TrainingTask)) is None


def test_remote_task_create_remains_compatible_with_old_agent_without_output_roots(monitoring_client):
    client, _ = monitoring_client
    credential = create(client)
    payload = sample()
    payload["training"]["supported_parameters"] = sorted(training_parameter_keys())
    assert "output_roots" not in payload["training"]
    assert client.post(f"/api/compute/nodes/{credential['node']['id']}/report",
                       headers=headers(credential), json=payload).status_code == 200
    response = client.post("/api/tasks", json={
        "name": "Legacy Agent", "execution_node_id": credential["node"]["id"],
        "model_path": "/remote/model.pt", "data_path": "/remote/data.yaml",
        "remote_output_root": "/remote/runs",
    })
    assert response.status_code == 201, response.text
    assert client.post(f"/api/tasks/{response.json()['id']}/start").status_code == 200
    assert client.post(f"/api/compute/nodes/{credential['node']['id']}/agent/poll",
                       headers=headers(credential)).json()["task"] is not None


@pytest.mark.parametrize(("action", "status"), [("start", "draft"), ("retry", "failed"), ("resume", "stopped")])
def test_remote_launch_rechecks_output_roots_without_creating_attempt(monitoring_client, action, status):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    report_output_roots(client, credential, ["/remote/new-runs"])
    with factory() as db:
        task = TrainingTask(id="changed-output-root", name="Changed output root", task_type="detect",
                            execution_node_id=node_id, model_path="/remote/model.pt",
                            data_path="/remote/data.yaml", config={"remote_output_root": "/remote/old-runs"})
        db.add(task)
        db.flush()
        if action != "start":
            previous = create_training_run(db, task, "fresh")
            previous.status = status
        task.status = status
        db.commit()
    response = client.post(f"/api/tasks/changed-output-root/{action}")
    assert response.status_code == 409
    assert "/remote/old-runs" in response.json()["detail"]
    assert "/remote/new-runs" in response.json()["detail"]
    with factory() as db:
        task = db.get(TrainingTask, "changed-output-root")
        assert task.status == status
        assert len(task.runs) == (0 if action == "start" else 1)
        assert not task.events


@pytest.mark.parametrize("expired_claim", [False, True])
def test_remote_poll_rejects_changed_output_roots_and_advances_queue(monitoring_client, expired_claim):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    report_output_roots(client, credential, ["/remote/new-runs"])
    with factory() as db:
        for task_id, root in [("changed-output", "/remote/old-runs"), ("valid-output", "/remote/new-runs")]:
            task = TrainingTask(id=task_id, name=task_id, task_type="detect", execution_node_id=node_id,
                                model_path="/remote/model.pt", data_path="/remote/data.yaml",
                                config={"remote_output_root": root})
            db.add(task)
            db.flush()
            run = create_training_run(db, task, "fresh")
            if task_id == "changed-output" and expired_claim:
                task.status = run.status = "starting"
                run.claim_lease = "a" * 32
                run.claim_expires_at = utcnow() - timedelta(seconds=1)
                run.claim_confirmed = False
        db.commit()
    poll_url = f"/api/compute/nodes/{node_id}/agent/poll"
    response = client.post(poll_url, headers=headers(credential))
    assert response.status_code == 200
    # A rejected queued claim must not block a later valid task on the same
    # Agent poll. The invalid run retains its own failure event for diagnosis.
    assert response.json()["task"]["task_id"] == "valid-output"
    with factory() as db:
        task = db.get(TrainingTask, "changed-output")
        assert task.status == task.runs[0].status == "failed"
        assert task.runs[0].finished_at is not None and task.runs[0].pid is None
        assert task.error_message == task.runs[0].error_message
        assert len(task.events) == 1 and task.events[0].event_type == "failed"
        valid = db.get(TrainingTask, "valid-output")
        assert valid.status == valid.runs[0].status == "starting"


def test_stop_between_remote_claim_and_confirmation_is_idempotent(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Stop Race Agent")
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="stop-claim-race", name="stop race", task_type="detect",
                            execution_node_id=node_id, model_path="/remote/model.pt",
                            data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run_id = run.id
        db.commit()

    report = sample()
    report["agent_version"] = "1.8.0"
    report["training"] = {"enabled": True, "ready": True, "supported_parameters": []}
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=report).status_code == 200
    claimed = client.post(f"/api/compute/nodes/{node_id}/agent/poll",
                          headers=headers(credential)).json()["task"]
    lease = claimed["claim_lease"]

    first_stop = client.post("/api/tasks/stop-claim-race/stop")
    repeated_stop = client.post("/api/tasks/stop-claim-race/stop")
    assert first_stop.status_code == 200 and first_stop.json()["status"] == "stopping"
    assert repeated_stop.status_code == 200 and repeated_stop.json()["status"] == "stopping"

    confirm = client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm",
                          headers=headers(credential), json={"lease": lease})
    assert confirm.status_code == 200
    assert confirm.json() == {"confirmed": True, "stop": True}
    assert client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm",
                       headers=headers(credential), json={"lease": lease}).json() == confirm.json()

    update = client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update",
                         headers={**headers(credential), "X-Training-Lease": lease},
                         json={"status": "stopped", "stop_reason": "platform_stop", "exit_code": -15})
    assert update.status_code == 200 and update.json()["status"] == "stopped"
    with factory() as db:
        run = db.get(TrainingRun, run_id)
        stop_events = db.scalars(select(TaskEvent).where(
            TaskEvent.run_id == run_id, TaskEvent.event_type == "stop_requested")).all()
        assert run.status == "stopped" and run.claim_confirmed is True
        assert len(stop_events) == 1


def test_expired_unconfirmed_stop_releases_remote_node(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Expired Stop Agent")
    node_id = credential["node"]["id"]
    with factory() as db:
        first_task = TrainingTask(id="expired-stop", name="expired stop", task_type="detect",
                                  execution_node_id=node_id, model_path="/remote/model.pt",
                                  data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
        second_task = TrainingTask(id="after-expired-stop", name="next", task_type="detect",
                                   execution_node_id=node_id, model_path="/remote/model.pt",
                                   data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
        db.add_all([first_task, second_task])
        db.flush()
        first_run = create_training_run(db, first_task, "fresh")
        second_run = create_training_run(db, second_task, "fresh")
        first_run.queued_at = utcnow() - timedelta(seconds=2)
        second_run.queued_at = utcnow() - timedelta(seconds=1)
        first_run_id, second_run_id = first_run.id, second_run.id
        db.commit()

    report = sample()
    report["agent_version"] = "1.8.0"
    report["training"] = {"enabled": True, "ready": True, "supported_parameters": []}
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=report).status_code == 200
    first_claim = client.post(f"/api/compute/nodes/{node_id}/agent/poll",
                              headers=headers(credential)).json()["task"]
    assert first_claim["run_id"] == first_run_id
    assert client.post("/api/tasks/expired-stop/stop").status_code == 200
    with factory() as db:
        stopped = db.get(TrainingRun, first_run_id)
        stopped.claim_expires_at = utcnow() - timedelta(seconds=1)
        db.commit()

    next_claim = client.post(f"/api/compute/nodes/{node_id}/agent/poll",
                             headers=headers(credential)).json()["task"]
    assert next_claim["run_id"] == second_run_id
    with factory() as db:
        stopped = db.get(TrainingRun, first_run_id)
        task = db.get(TrainingTask, "expired-stop")
        assert stopped.status == "stopped" and stopped.claim_confirmed is False
        assert task.status == "stopped"
        assert stopped.error_message == "平台已取消远程训练启动，训练进程尚未启动"


def test_confirmed_remote_run_can_be_manually_interrupted_after_stop_stalls(monitoring_client):
    client, factory = monitoring_client
    credential = create(client, "Recovery Agent")
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="stalled-remote", name="stalled", task_type="detect",
                            execution_node_id=node_id, model_path="/remote/model.pt",
                            data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run_id = run.id
        db.commit()

    report = sample()
    report["agent_version"] = "1.8.0"
    report["training"] = {"enabled": True, "ready": True, "supported_parameters": []}
    assert client.post(f"/api/compute/nodes/{node_id}/report",
                       headers=headers(credential), json=report).status_code == 200
    claimed = client.post(f"/api/compute/nodes/{node_id}/agent/poll",
                          headers=headers(credential)).json()["task"]
    lease = claimed["claim_lease"]
    assert client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/confirm",
                       headers=headers(credential), json={"lease": lease}).status_code == 200
    assert client.post("/api/tasks/stalled-remote/stop").json()["status"] == "stopping"

    confirmation = {"run_id": run_id, "confirm_process_stopped": True}
    too_soon = client.post("/api/tasks/stalled-remote/force-interrupt", json=confirmation)
    assert too_soon.status_code == 409
    assert client.post("/api/tasks/stalled-remote/force-interrupt", json={
        "run_id": run_id, "confirm_process_stopped": False,
    }).status_code == 422

    with factory() as db:
        run = db.get(TrainingRun, run_id)
        stop_event = db.scalar(select(TaskEvent).where(
            TaskEvent.run_id == run_id, TaskEvent.event_type == "stop_requested"))
        stop_event.created_at = utcnow() - timedelta(seconds=31)
        # A continuing Agent heartbeat must not restart the recovery delay.
        run.progress = 1
        db.commit()
        assert run.updated_at > stop_event.created_at
    stale_run = client.post("/api/tasks/stalled-remote/force-interrupt", json={
        "run_id": "0" * 32, "confirm_process_stopped": True,
    })
    assert stale_run.status_code == 409
    released = client.post("/api/tasks/stalled-remote/force-interrupt", json=confirmation)
    assert released.status_code == 200 and released.json()["status"] == "interrupted"
    repeated_release = client.post("/api/tasks/stalled-remote/force-interrupt", json=confirmation)
    assert repeated_release.status_code == 409

    late_update = client.post(
        f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update",
        headers={**headers(credential), "X-Training-Lease": lease},
        json={"status": "running", "pid": 9999},
    )
    assert late_update.status_code == 200
    assert late_update.json() == {"status": "interrupted", "stop_requested": True}
    with factory() as db:
        run = db.get(TrainingRun, run_id)
        events = db.scalars(select(TaskEvent).where(
            TaskEvent.run_id == run_id, TaskEvent.event_type == "force_interrupted")).all()
        assert run.status == "interrupted" and run.pid is None and len(events) == 1


def test_remote_training_trace_upload_is_authenticated_and_stored(monitoring_client, tmp_path, monkeypatch):
    client, factory = monitoring_client
    monkeypatch.setattr(monitoring, "DATA_ROOT", tmp_path)
    credential = create(client)
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="remote-trace", name="trace", task_type="detect", execution_node_id=node_id,
                            model_path="/remote/model.pt", data_path="/remote/data.yaml",
                            config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = "running"
        run.run_dir = "/remote/runs/detect/remote-trace/attempt-001"
        run_id = run.id
        db.commit()
    payload = {"trace": {"updated_at": 1, "maximum": 24, "interval": 10, "samples": [
        {"epoch": 1, "batch": 10, "images": {"shape": [2, 3, 640, 640]}, "loss": {"box": 1.2}, "layers": {}, "updates": {},
         "image": "batch-00010.jpg"}
    ]}, "image_name": "batch-00010.jpg", "image_base64": base64.b64encode(b"\xff\xd8\xfftrace").decode()}
    url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/trace"
    assert client.post(url, json=payload).status_code == 401
    response = client.post(url, headers=headers(credential), json=payload)
    assert response.status_code == 200, response.text
    trace = client.get(f"/api/tasks/remote-trace/runs/{run_id}/trace")
    assert trace.status_code == 200 and trace.json()["available"]
    assert trace.json()["samples"][0]["image_url"].endswith("/batch-00010.jpg")
    image = client.get(f"/api/tasks/remote-trace/runs/{run_id}/trace/images/batch-00010.jpg")
    assert image.status_code == 200 and image.content.startswith(b"\xff\xd8\xff")
    assert client.post(url, headers=headers(credential), json={**payload, "image_name": "../escape.jpg"}).status_code == 422


def test_agent_rejects_outside_paths_and_keeps_training_disabled_by_default(monitoring_client, tmp_path):
    client, _ = monitoring_client
    credential = create(client)
    script = Path(__file__).resolve().parents[2] / "agent" / "monitor_agent.py"
    spec = importlib.util.spec_from_file_location("remote_agent_test", script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    path = tmp_path / "agent-config.json"
    path.write_text(json.dumps({"center_url": "https://example.com", "node_id": credential["node"]["id"],
                                "token": credential["token"]}))
    path.chmod(0o600)
    assert agent.load_config(path)["enable_training"] is False
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="允许目录"):
        agent.within_allowed(str(outside), [str(allowed)])


def test_remote_command_uses_allowed_files_and_separate_attempt_directory(tmp_path):
    script = Path(__file__).resolve().parents[2] / "agent" / "monitor_agent.py"
    spec = importlib.util.spec_from_file_location("remote_command_test", script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    root = tmp_path / "training"
    root.mkdir()
    model = root / "model.pt"
    dataset = root / "data.yaml"
    model.write_bytes(b"model")
    dataset.write_text("train: images/train\nval: images/val\n")
    yolo = root / "yolo"
    yolo.write_text("#!/bin/sh\n")
    yolo.chmod(0o700)
    run_dir = root / "runs" / "detect" / "task-id" / "attempt-001"
    run_dir.parent.mkdir(parents=True)
    payload = {"model_path": str(model), "data_path": str(dataset), "run_dir": str(run_dir),
               "task_id": "task-id", "attempt_number": 1,
               "log_path": str(root / "runs/.platform-logs/20261012/task-id/attempt-001.log"),
               "task_type": "detect", "checkpoint_path": None,
               "parameters": {"epochs": 2, "batch": 1, "imgsz": 640, "device": "0", "workers": 2,
                              "optimizer": "auto", "remote_output_root": str(root / "runs"),
                              "config": {"custom_args": {"label_smoothing": 0.1}}}}
    config = {"allowed_training_roots": [str(root)], "yolo_bin": str(yolo)}
    command = agent.build_training_command(config, payload)
    assert command[0] == str(yolo)
    assert "label_smoothing=0.1" in command
    assert f"name={run_dir.name}" in command
    with pytest.raises(ValueError, match="已存在"):
        agent.build_training_command(config, payload)
    payload["run_dir"] = str(root / "runs" / "detect" / "task-id" / "attempt-002")
    payload["attempt_number"] = 2
    payload["log_path"] = str(Path(payload["log_path"]).with_name("attempt-002.log"))
    payload["parameters"]["config"]["custom_args"] = {"project": "/tmp/elsewhere"}
    with pytest.raises(ValueError, match="不允许"):
        agent.build_training_command(config, payload)
    assert not Path(payload["run_dir"]).exists()
    payload["parameters"]["config"]["custom_args"] = {}
    payload["log_path"] = str(Path(payload["run_dir"]) / "platform.log")
    with pytest.raises(ValueError, match="日志"):
        agent.build_training_command(config, payload)


@pytest.mark.parametrize("reason, expected", [
    ("agent_shutdown", "Agent 退出，训练进程已停止"),
    (None, "远程训练已停止，Agent 未提供停止原因"),
])
def test_remote_stop_does_not_invent_user_request(monitoring_client, reason, expected):
    client, factory = monitoring_client
    credential = create(client)
    with factory() as db:
        task = TrainingTask(id="shutdown", name="remote", task_type="detect",
                            execution_node_id=credential["node"]["id"], model_path="/remote/model.pt",
                            data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = "running"
        run_id = run.id
        db.commit()
    response = client.post(f"/api/compute/nodes/{credential['node']['id']}/agent/runs/{run_id}/update",
                           headers=headers(credential), json={"status": "stopped", "stop_reason": reason, "exit_code": -15})
    assert response.status_code == 200
    with factory() as db:
        assert db.get(TrainingRun, run_id).error_message == expected


@pytest.mark.parametrize("with_info", [False, True])
def test_remote_completion_preserves_unknown_exit_code_and_early_stop_details(monitoring_client, with_info):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="early-stop", name="smoke", task_type="detect", execution_node_id=node_id,
                            model_path="/remote/model.pt", data_path="/remote/data.yaml", epochs=100,
                            config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = "running"
        run.error_message = "stale recovery error"
        run_id = run.id
        db.commit()
    info = {"source": "legacy_log", "reason": "early_stopping", "completed_epochs": 56,
            "expected_epochs": 100, "best_epoch": 21, "patience": 35,
            "final_metrics": {"metrics/mAP50(B)": 0.818}}
    payload = {"status": "completed", "completed_epochs": 56, "progress": 56,
               "latest_metrics": {"metrics/mAP50(B)": 0.8016}, "exit_code": None}
    if with_info:
        payload["completion_info"] = info
    url = f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update"
    response = client.post(url, headers=headers(credential), json=payload)
    assert response.status_code == 200, response.text
    with factory() as db:
        task, run = db.get(TrainingTask, "early-stop"), db.get(TrainingRun, run_id)
        assert task.status == run.status == "completed"
        assert run.exit_code is task.exit_code is None
        assert run.error_message is task.error_message is None
        assert run.completed_epochs == 56 and run.progress == 100
        assert run.completion_info == task.completion_info == (info if with_info else {})
        assert run.latest_metrics == {"metrics/mAP50(B)": 0.8016}
        event = db.scalar(select(TaskEvent).where(TaskEvent.run_id == run_id, TaskEvent.event_type == "completed"))
        assert event.payload["completion_info"] == (info if with_info else {})
    assert client.get("/api/tasks/early-stop").json()["completion_info"] == (info if with_info else {})
    # A delayed heartbeat must not roll a completed Attempt back to running.
    assert client.post(url, headers=headers(credential), json={"status": "running"}).json()["status"] == "completed"


@pytest.mark.parametrize("changes", [
    {"exit_code": 1},
    {"completion_info": {"source": "training_result", "reason": "exception", "completed_epochs": 56, "expected_epochs": 100}},
    {"completion_info": {"source": "unknown", "reason": "early_stopping", "completed_epochs": 56, "expected_epochs": 100}},
    {"completion_info": {"source": "legacy_log", "reason": "early_stopping", "completed_epochs": True, "expected_epochs": 100}},
    {"completion_info": {"source": "legacy_log", "reason": "early_stopping", "completed_epochs": 56, "expected_epochs": 100, "best_epoch": 57}},
])
def test_remote_completion_rejects_inconsistent_or_invalid_details(monitoring_client, changes):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="invalid-result", name="smoke", task_type="detect", execution_node_id=node_id,
                            model_path="/remote/model.pt", data_path="/remote/data.yaml",
                            config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = "running"
        run_id = run.id
        db.commit()
    response = client.post(f"/api/compute/nodes/{node_id}/agent/runs/{run_id}/update",
                           headers=headers(credential), json={"status": "completed", **changes})
    assert response.status_code == 422
    with factory() as db:
        assert db.get(TrainingRun, run_id).status == "running"


def test_active_remote_run_blocks_node_disabling_and_token_rotation(monitoring_client):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    with factory() as db:
        task = TrainingTask(id="active-remote", name="remote", task_type="detect", execution_node_id=node_id,
                            model_path="/remote/model.pt", data_path="/remote/data.yaml",
                            config={"remote_output_root": "/remote/runs"})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = "running"
        db.commit()
    assert client.post(f"/api/compute/nodes/{node_id}/token").status_code == 409
    assert client.patch(f"/api/compute/nodes/{node_id}", json={"name": "remote", "enabled": False}).status_code == 409
    assert client.delete(f"/api/compute/nodes/{node_id}").status_code == 409
