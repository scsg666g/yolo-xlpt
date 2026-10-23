"""Registration removal preserves source files, history and exact-version reruns."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import sqlite3
import threading

import pytest
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

import app.database as database
import app.inference as inference
import app.model_assets as assets
import app.remote_operations as operations
from app.database import Base, get_db
from app.main import app
from app.models import ComputeNode, ModelAsset, ModelTest, RemoteOperation, utcnow
from test_compute_nodes import create as create_node, monitoring_client
from test_inference import (
    configure_standalone_models, image_bytes, image_test_client,
    install_fake_process, upload_all,
)
from test_remote_model_assets import claim, register, report_remote_model_node, upload_single_model_chunk


def test_payload(asset_id, **kwargs):
    return {"model_asset_id": asset_id, "files": [{"name": "input.png", "size": len(image_bytes())}], **kwargs}


test_payload.__test__ = False


def local_asset(client, tmp_path, monkeypatch):
    _, root = configure_standalone_models(tmp_path, monkeypatch)
    checkpoint = root / "best.pt"
    checkpoint.write_bytes(b"original-checkpoint")
    response = client.post("/api/model-assets/local", json={"path": str(checkpoint)})
    assert response.status_code == 201, response.text
    return response.json(), checkpoint


def test_local_removal_persists_and_can_restore_same_identity(image_test_client, tmp_path, monkeypatch):
    client, db, _, factory, _ = image_test_client
    asset, checkpoint = local_asset(client, tmp_path, monkeypatch)
    assert client.delete(f"/api/model-assets/{asset['id']}").status_code == 204
    assert checkpoint.read_bytes() == b"original-checkpoint"
    assert client.get("/api/model-assets").json()["items"] == []
    with factory() as reopened:
        removed_at = reopened.get(ModelAsset, asset["id"]).deleted_at
        assert removed_at is not None
    assert client.delete(f"/api/model-assets/{asset['id']}").status_code == 204
    db.expire_all()
    assert db.get(ModelAsset, asset["id"]).deleted_at == removed_at
    assert client.delete("/api/model-assets/nonexistent").status_code == 404
    response = client.post("/api/model-tests", json=test_payload(asset["id"]))
    assert response.status_code == 409 and "已移除" in response.json()["detail"]
    assert client.post(f"/api/model-assets/{asset['id']}/prepare").status_code == 409

    # Validation failures must never restore an invalid source.
    checkpoint.unlink()
    assert client.post("/api/model-assets/local", json={"path": asset["path"]}).status_code == 404
    assert client.get("/api/model-assets").json()["items"] == []
    checkpoint.write_bytes(b"replacement-checkpoint")
    restored = client.post("/api/model-assets/local", json={"path": asset["path"]})
    assert restored.status_code == 201
    assert restored.json()["id"] == asset["id"]
    assert restored.json()["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert [item["id"] for item in client.get("/api/model-assets").json()["items"]] == [asset["id"]]


@pytest.mark.parametrize("status", sorted(inference.ACTIVE_STATUSES))
def test_busy_test_blocks_removal(image_test_client, tmp_path, monkeypatch, status):
    client, db, _, _, _ = image_test_client
    asset, checkpoint = local_asset(client, tmp_path, monkeypatch)
    created = client.post("/api/model-tests", json=test_payload(asset["id"])).json()
    db.get(ModelTest, created["id"]).status = status
    db.commit()
    listed = client.get("/api/model-assets").json()["items"][0]
    assert listed["history_count"] == 1
    assert "尚未结束" in listed["removal_blocked_reason"]
    response = client.delete(f"/api/model-assets/{asset['id']}")
    assert response.status_code == 409
    assert checkpoint.read_bytes() == b"original-checkpoint"
    db.expire_all()
    assert db.get(ModelAsset, asset["id"]).deleted_at is None


def test_removed_model_keeps_history_downloads_and_exact_rerun(image_test_client, tmp_path, monkeypatch):
    client, db, _, _, worker = image_test_client
    asset, checkpoint = local_asset(client, tmp_path, monkeypatch)
    created = client.post("/api/model-tests", json=test_payload(asset["id"])).json()
    upload_all(client, created["id"])
    assert client.post(f"/api/model-tests/{created['id']}/start").status_code == 202
    install_fake_process(monkeypatch)
    assert worker.process_one()
    prefix = f"/api/model-tests/{created['id']}"
    original = client.get(prefix + "/images/0/original").content
    result = client.get(prefix + "/images/0/result").content
    assert client.get("/api/model-assets").json()["items"][0]["history_count"] == 1
    assert client.delete(f"/api/model-assets/{asset['id']}").status_code == 204
    assert client.get(prefix).json()["status"] == "completed"
    assert client.get(prefix + "/images/0/original").content == original
    assert client.get(prefix + "/images/0/result").content == result
    assert client.get(prefix + "/download").status_code == 200
    assert client.get("/api/model-tests").json()["items"][0]["id"] == created["id"]
    rerun = client.post("/api/model-tests", json=test_payload(asset["id"], rerun_of_test_id=created["id"]))
    assert rerun.status_code == 201, rerun.text
    assert rerun.json()["model_sha256"] == created["model_sha256"]
    upload_all(client, rerun.json()["id"])
    assert client.post(f"/api/model-tests/{rerun.json()['id']}/start").status_code == 202
    assert worker.process_one()
    assert client.get("/api/model-assets").json()["items"] == []

    # Local historical bytes are not silently replaced by a newer source file.
    checkpoint.write_bytes(b"changed")
    failed = client.post("/api/model-tests", json=test_payload(asset["id"], rerun_of_test_id=created["id"]))
    assert failed.status_code == 409 and "版本已变化" in failed.json()["detail"]
    db.expire_all()
    assert db.get(ModelTest, created["id"]).model_asset_id == asset["id"]


@pytest.mark.parametrize("first", ["remove", "create"])
def test_remove_and_create_are_serialized(image_test_client, tmp_path, monkeypatch, first):
    client, _, _, factory, _ = image_test_client
    asset, _ = local_asset(client, tmp_path, monkeypatch)
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    hook_module = assets if first == "remove" else inference
    hook_name = "asset_removal_metadata" if first == "remove" else "_available_model_asset_version"
    original = getattr(hook_module, hook_name)

    def paused(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)

    def separate_sessions():
        with factory() as session:
            yield session

    monkeypatch.setattr(hook_module, hook_name, paused)
    app.dependency_overrides[get_db] = separate_sessions

    def request(action, second=False):
        if second:
            second_started.set()
        if action == "remove":
            return client.delete(f"/api/model-assets/{asset['id']}")
        return client.post("/api/model-tests", json=test_payload(asset["id"]))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_request = pool.submit(request, first)
        try:
            assert entered.wait(5)
            second_request = pool.submit(request, "create" if first == "remove" else "remove", True)
            assert second_started.wait(5)
        finally:
            release.set()
        assert first_request.result(timeout=5).status_code == (204 if first == "remove" else 201)
        assert second_request.result(timeout=5).status_code == 409


@pytest.mark.parametrize("status", ["pending", "running", "finalizing"])
def test_remote_transfer_blocks_removal(monitoring_client, tmp_path, monkeypatch, status):
    client, factory = monitoring_client
    credential = create_node(client)
    report_remote_model_node(client, credential, tmp_path)
    asset = register(client, credential, tmp_path / "remote.pt").json()
    with factory() as db:
        operation = db.scalar(select(RemoteOperation).where(RemoteOperation.kind == "fetch_model"))
        operation.status = status
        db.commit()
    response = client.delete(f"/api/model-assets/{asset['id']}")
    assert response.status_code == 409 and "传输" in response.json()["detail"]
    listed = client.get("/api/model-assets").json()["items"][0]
    assert listed["history_count"] == 0
    assert "传输" in listed["removal_blocked_reason"]


def test_remote_offline_removal_preserves_historical_cache_and_restores_with_refresh(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create_node(client)
    report_remote_model_node(client, credential, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 0)
    checkpoint = tmp_path / "remote.pt"
    checkpoint.write_bytes(b"remote-original")
    asset = register(client, credential, checkpoint).json()
    old_work = claim(client, credential)
    assert upload_single_model_chunk(client, credential, old_work, checkpoint.read_bytes()).status_code == 200
    with factory() as db:
        current = db.get(ModelAsset, asset["id"])
        db.add(ModelTest(
            id="historical", model_asset_id=current.id, model_source_type="remote_path",
            model_sha256=current.sha256, model_size=current.size, status="completed",
            artifact_name=current.path, model_name=current.name, confidence=0.25, iou=0.7,
        ))
        db.commit()
    checkpoint.write_bytes(b"remote-new-version")
    register(client, credential, checkpoint)
    new_work = claim(client, credential)
    assert upload_single_model_chunk(client, credential, new_work, checkpoint.read_bytes()).status_code == 200
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).last_seen_at = utcnow() - timedelta(days=1)
        db.commit()
    assert client.delete(f"/api/model-assets/{asset['id']}").status_code == 204
    assert checkpoint.read_bytes() == b"remote-new-version"
    assert client.get("/api/model-assets").json()["items"] == []
    with factory() as db:
        assert operations.reclaim_unreferenced_model_cache(db) == len(b"remote-new-version")
        db.commit()
        current = db.get(ModelAsset, asset["id"])
        historical = db.get(ModelTest, "historical")
        cache, _, _ = operations.verified_remote_model_cache(
            db, current, expected_sha256=historical.model_sha256, expected_size=historical.model_size,
        )
        assert cache.read_bytes() == b"remote-original"
        assert {operation.kind for operation in db.scalars(select(RemoteOperation))} == {"fetch_model"}
    # Offline restoration cannot accidentally unhide the model.
    assert register(client, credential, checkpoint).status_code == 409
    assert client.get("/api/model-assets").json()["items"] == []
    report_remote_model_node(client, credential, tmp_path)
    restored = register(client, credential, checkpoint)
    assert restored.status_code == 201
    assert restored.json()["id"] == asset["id"]
    assert restored.json()["status"] == "preparing"
    assert claim(client, credential)["id"] not in {old_work["id"], new_work["id"]}


def test_removed_remote_model_can_rerun_from_offline_history(image_test_client, tmp_path, monkeypatch):
    client, db, _, _, worker = image_test_client
    credential = create_node(client)
    report_remote_model_node(client, credential, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 0)
    asset = register(client, credential, tmp_path / "remote.pt").json()
    work = claim(client, credential)
    assert upload_single_model_chunk(client, credential, work, b"remote-checkpoint").status_code == 200
    created = client.post("/api/model-tests", json=test_payload(asset["id"])).json()
    upload_all(client, created["id"])
    assert client.post(f"/api/model-tests/{created['id']}/start").status_code == 202
    seen_models = []
    install_fake_process(monkeypatch, seen_models=seen_models)
    assert worker.process_one()
    db.expire_all()
    db.get(ComputeNode, credential["node"]["id"]).last_seen_at = utcnow() - timedelta(days=1)
    db.commit()
    assert client.delete(f"/api/model-assets/{asset['id']}").status_code == 204
    assert operations.reclaim_unreferenced_model_cache(db) == 0
    db.commit()
    rerun = client.post("/api/model-tests", json=test_payload(asset["id"], rerun_of_test_id=created["id"]))
    assert rerun.status_code == 201, rerun.text
    upload_all(client, rerun.json()["id"])
    assert client.post(f"/api/model-tests/{rerun.json()['id']}/start").status_code == 202
    assert worker.process_one()
    assert seen_models == [b"remote-checkpoint", b"remote-checkpoint"]
    assert client.get("/api/model-assets").json()["items"] == []
    assert client.get(f"/api/model-tests/{created['id']}/download").status_code == 200


def test_migration_preserves_legacy_model_and_history(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as db:
        db.add(ModelAsset(id="legacy", name="old", source_type="local_path", execution_node_id="local", path="pt/old.pt"))
        db.flush()
        db.add(ModelTest(id="old-test", model_asset_id="legacy", artifact_name="old.pt", model_name="old", confidence=0.25, iou=0.7, status="completed"))
        db.commit()
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_model_assets_deleted_at"))
        connection.execute(text("ALTER TABLE model_assets DROP COLUMN deleted_at"))
    monkeypatch.setattr(database, "engine", engine)
    backup = database.migrate_database()
    assert backup is not None and backup.is_file()
    assert database.migrate_database() is None
    with sessionmaker(bind=engine)() as db:
        assert db.get(ModelAsset, "legacy").deleted_at is None
        assert db.get(ModelTest, "old-test").model_asset_id == "legacy"
    assert "deleted_at" in {column["name"] for column in inspect(engine).get_columns("model_assets")}
    with sqlite3.connect(backup) as original:
        assert "deleted_at" not in {row[1] for row in original.execute("PRAGMA table_info(model_assets)")}
        assert original.execute("SELECT model_asset_id FROM model_tests").fetchone() == ("legacy",)
    engine.dispose()
