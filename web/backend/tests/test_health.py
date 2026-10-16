from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main_module


class _Connection:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def exec_driver_sql(self, statement: str):
        assert statement == "SELECT 1"
        return SimpleNamespace(scalar_one=lambda: 1)


class _Engine:
    def connect(self):
        return _Connection()


def test_health_reports_each_core_worker(monkeypatch):
    monkeypatch.setattr(main_module, "engine", _Engine())
    monkeypatch.setattr(main_module, "scheduler", SimpleNamespace(
        _thread=SimpleNamespace(is_alive=lambda: True),
    ))
    monkeypatch.setattr(main_module, "local_deletion_worker", SimpleNamespace(is_alive=True))
    monkeypatch.setattr(main_module, "inference_worker", SimpleNamespace(is_alive=True))

    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health")
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["database"] == "ok"
    assert payload["scheduler"] == "running"
    assert payload["local_deletion_worker"] == "running"
    assert payload["inference_worker"] == "running"
    assert isinstance(payload["database_latency_ms"], (int, float))
    assert payload["instance"]
    assert payload["runs_root"]


def test_health_reports_degraded_when_a_worker_stops(monkeypatch):
    monkeypatch.setattr(main_module, "engine", _Engine())
    monkeypatch.setattr(main_module, "scheduler", SimpleNamespace(
        _thread=SimpleNamespace(is_alive=lambda: True),
    ))
    monkeypatch.setattr(main_module, "local_deletion_worker", SimpleNamespace(is_alive=True))
    monkeypatch.setattr(main_module, "inference_worker", SimpleNamespace(is_alive=False))

    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["inference_worker"] == "stopped"


def test_health_returns_service_unavailable_when_database_check_fails(monkeypatch):
    class BrokenEngine:
        def connect(self):
            raise RuntimeError("database offline")

    monkeypatch.setattr(main_module, "engine", BrokenEngine())
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health")
    finally:
        client.close()

    assert response.status_code == 503
    assert response.json() == {"detail": "数据库暂不可用"}
