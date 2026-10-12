import importlib.util
import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.compute_nodes as monitoring
from app.database import Base, get_db
from app.main import app
from app.models import ComputeNode, utcnow


def sample(gpus=None):
    return {"agent_version": "1.0.0", "compute": {
        "available": bool(gpus), "sampled_at": utcnow().isoformat(), "provider": "NVIDIA" if gpus else None,
        "message": None, "host": {"hostname": "remote-host", "platform": "Linux", "cpu_count": 8,
                                 "memory_total": 16000, "memory_used": 4000, "load_average": [1, 2, 3]},
        "gpus": gpus or [],
    }}


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
    assert client.get(f"/api/compute/nodes/{bid}").json()["compute"] is None
    listing = client.get("/api/compute/nodes").json()
    assert listing[0]["id"] == "local"
    assert all("compute" not in item and "token_hash" not in item and "token" not in item for item in listing)
    assert a["token"] not in json.dumps(detail)


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
    assert client.post(url, headers=headers(credential), json=older).status_code == 409
    detail = client.get(url.removesuffix("/report")).json()
    assert detail["status"] == "online" and detail["compute"]["available"] is False
    payload["compute"]["host"]["load_average"] = [-1]
    assert client.post(url, headers=headers(credential), json=payload).status_code == 422
    assert client.post(url, headers=headers(credential), content=b"x" * (monitoring.MAX_REPORT_BYTES + 1)).status_code == 413


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
    assert agent.load_config(config_path)["report_interval_seconds"] == 5
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
