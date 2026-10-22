from datetime import UTC, datetime
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
    assert payload["version"] == main_module.PLATFORM_VERSION
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


def test_health_reports_active_scheduler_error(monkeypatch):
    failed_at = datetime(2026, 10, 19, 8, 30, tzinfo=UTC)
    monkeypatch.setattr(main_module, "engine", _Engine())
    monkeypatch.setattr(main_module, "scheduler", SimpleNamespace(
        is_alive=True,
        active_error="RuntimeError: scheduler failed",
        last_error="RuntimeError: scheduler failed",
        last_error_at=failed_at,
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
    assert payload["status"] == "degraded"
    assert payload["scheduler"] == "error"
    assert payload["scheduler_error"] == "RuntimeError: scheduler failed"
    assert payload["scheduler_last_error"] == "RuntimeError: scheduler failed"
    assert payload["scheduler_last_error_at"] == failed_at.isoformat()


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


def test_storage_health_reports_configured_directory_without_writing(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "RUNS_ROOT", tmp_path)
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health/storage")
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == str(tmp_path)
    assert payload["status"] in {"writable", "read_only"}
    assert isinstance(payload["free_bytes"], int)
    assert isinstance(payload["total_bytes"], int)
    assert list(tmp_path.iterdir()) == []


def test_storage_health_reports_missing_directory(monkeypatch, tmp_path):
    missing = tmp_path / "not-created"
    monkeypatch.setattr(main_module, "RUNS_ROOT", missing)
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health/storage")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {
        "path": str(missing),
        "status": "missing",
        "free_bytes": None,
        "total_bytes": None,
    }
    assert not missing.exists()


def test_storage_health_reports_path_that_is_not_a_directory(monkeypatch, tmp_path):
    file_path = tmp_path / "runs.txt"
    file_path.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(main_module, "RUNS_ROOT", file_path)
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health/storage")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {
        "path": str(file_path),
        "status": "not_directory",
        "free_bytes": None,
        "total_bytes": None,
    }


def test_storage_health_reports_read_only_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "RUNS_ROOT", tmp_path)
    monkeypatch.setattr(main_module.os, "access", lambda *_: False)
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health/storage")
    finally:
        client.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["path"] == str(tmp_path)
    assert payload["status"] == "read_only"
    assert isinstance(payload["free_bytes"], int)
    assert isinstance(payload["total_bytes"], int)


def test_storage_health_reports_unavailable_when_disk_usage_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(main_module, "RUNS_ROOT", tmp_path)

    def fail_disk_usage(_):
        raise OSError("storage unavailable")

    monkeypatch.setattr(main_module.shutil, "disk_usage", fail_disk_usage)
    client = TestClient(main_module.app)
    try:
        response = client.get("/api/health/storage")
    finally:
        client.close()

    assert response.status_code == 200
    assert response.json() == {
        "path": str(tmp_path),
        "status": "unavailable",
        "free_bytes": None,
        "total_bytes": None,
    }
