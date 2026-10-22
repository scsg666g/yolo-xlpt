"""Standalone remote-model protocol and immutable-cache integration."""
from __future__ import annotations

import base64
import hashlib
import stat
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import app.inference as inference
import app.remote_operations as operations
from app.models import ComputeNode, ModelAsset, ModelTest, RemoteOperation, utcnow
from test_compute_nodes import create, headers, monitoring_client, sample
from test_remote_agent import agent


def report_remote_model_node(client, credential, root: Path) -> None:
    report = sample()
    report["agent_version"] = "2.0.3"
    report["training"]["model_test_roots"] = [str(root)]
    response = client.post(
        f"/api/compute/nodes/{credential['node']['id']}/report",
        headers=headers(credential),
        json=report,
    )
    assert response.status_code == 200, response.text


def agent_api(client, credential):
    def api(_config, path, method="GET", payload=None, **_kwargs):
        response = client.request(method, path, json=payload, headers=headers(credential))
        if response.status_code >= 400:
            response.raise_for_status()
        return response.json()
    return api


def claim(client, credential):
    return client.post(
        f"/api/compute/nodes/{credential['node']['id']}/agent/operations/poll",
        headers=headers(credential),
    ).json()["operation"]


def register(client, credential, model: Path):
    return client.post("/api/model-assets/remote", json={
        "execution_node_id": credential["node"]["id"],
        "path": str(model),
    })


def upload_single_model_chunk(client, credential, work, content: bytes):
    return client.post(
        f"/api/compute/nodes/{credential['node']['id']}"
        f"/agent/operations/{work['id']}/chunk",
        headers=headers(credential),
        json={
            "lease": work["lease"],
            "offset": 0,
            "data": base64.b64encode(content).decode(),
            "final": True,
            "sha256": hashlib.sha256(content).hexdigest(),
            "size": len(content),
            "modified_at": utcnow().isoformat(),
            "mtime_ns": 100,
        },
    )


def test_remote_model_cache_quota_rejects_before_writing(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Quota node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "quota.pt"
    content = b"0123456789"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_QUOTA_BYTES", len(content) - 1)
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 0)

    asset = register(client, credential, model).json()
    work = claim(client, credential)
    response = upload_single_model_chunk(client, credential, work, content)
    assert response.status_code == 507
    assert "缓存总配额" in response.json()["detail"]

    with factory() as db:
        stored_operation = db.get(RemoteOperation, work["id"])
        stored_asset = db.get(ModelAsset, asset["id"])
        assert stored_operation.status == "failed"
        assert stored_asset.status == "failed"
    assert not operations.operation_file(stored_operation, False).exists()
    assert not operations.operation_file(stored_operation).exists()


def test_remote_model_cache_minimum_free_space_counts_unreceived_bytes(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Disk reserve node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "reserve.pt"
    content = b"remote-model"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_QUOTA_BYTES", 0)
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 100)
    monkeypatch.setattr(operations.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100))

    asset = register(client, credential, model).json()
    work = claim(client, credential)
    response = upload_single_model_chunk(client, credential, work, content)
    assert response.status_code == 507
    assert "磁盘空间不足" in response.json()["detail"]
    with factory() as db:
        assert db.get(RemoteOperation, work["id"]).status == "failed"
        assert db.get(ModelAsset, asset["id"]).status == "failed"


def test_remote_model_chunk_rejects_cache_symlink(
    monitoring_client, tmp_path, monkeypatch,
):
    client, _ = monitoring_client
    credential = create(client, "Symlink cache node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "safe.pt"
    content = b"trusted-remote-model"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    cache_root = tmp_path / "remote-cache"
    cache_root.mkdir()
    monkeypatch.setattr(operations, "CACHE_ROOT", cache_root)
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_QUOTA_BYTES", 0)
    monkeypatch.setattr(operations, "REMOTE_MODEL_CACHE_MIN_FREE_BYTES", 0)

    register(client, credential, model)
    work = claim(client, credential)
    outside = tmp_path / "outside"
    outside.write_bytes(b"do-not-change")
    (cache_root / f"{work['id']}.part").symlink_to(outside)

    response = upload_single_model_chunk(client, credential, work, content)
    assert response.status_code == 409
    assert "缓存路径无效" in response.json()["detail"]
    assert outside.read_bytes() == b"do-not-change"


def test_remote_model_cache_reclaims_only_unreferenced_versions(
    monitoring_client, tmp_path, monkeypatch,
):
    _, factory = monitoring_client
    cache_root = tmp_path / "remote-cache"
    cache_root.mkdir()
    monkeypatch.setattr(operations, "CACHE_ROOT", cache_root)
    asset_id = "a" * 32
    current = b"current"
    historical = b"historical"
    stale = b"stale"
    active = b"partial"
    current_sha = hashlib.sha256(current).hexdigest()
    historical_sha = hashlib.sha256(historical).hexdigest()
    stale_sha = hashlib.sha256(stale).hexdigest()
    asset = ModelAsset(
        id=asset_id,
        name="remote.pt",
        source_type="remote_path",
        execution_node_id="remote-node",
        path="/models/remote.pt",
        sha256=current_sha,
        size=len(current),
        status="ready",
    )

    def operation(identifier: str, status: str, content: bytes, sha256: str | None):
        payload = {
            "model_asset_id": asset_id,
            "path": asset.path,
            "size": len(content),
        }
        if sha256 is not None:
            payload["sha256"] = sha256
        return RemoteOperation(
            id=identifier * 32,
            node_id=asset.execution_node_id,
            task_id=f"model-asset:{asset_id}",
            kind="fetch_model",
            status=status,
            payload=payload,
        )

    current_operation = operation("1", "succeeded", current, current_sha)
    historical_operation = operation("2", "succeeded", historical, historical_sha)
    stale_operation = operation("3", "succeeded", stale, stale_sha)
    active_operation = operation("4", "running", active, None)
    failed_operation = operation("5", "failed", b"failed", None)
    historical_test = ModelTest(
        id="b" * 32,
        task_id=None,
        model_asset_id=asset_id,
        model_source_type="remote_path",
        execution_node_id=asset.execution_node_id,
        model_source_path=asset.path,
        artifact_name=asset.path,
        model_name=asset.name,
        model_sha256=historical_sha,
        model_size=len(historical),
        status="completed",
        confidence=0.25,
        iou=0.7,
        model_snapshot={},
    )
    with factory() as db:
        db.add_all([
            asset,
            current_operation,
            historical_operation,
            stale_operation,
            active_operation,
            failed_operation,
            historical_test,
        ])
        db.commit()
        operations.operation_file(current_operation).write_bytes(current)
        operations.operation_file(historical_operation).write_bytes(historical)
        operations.operation_file(stale_operation).write_bytes(stale)
        operations.operation_file(active_operation, False).write_bytes(active)
        operations.operation_file(failed_operation, False).write_bytes(b"failed")

        reclaimed = operations.reclaim_unreferenced_model_cache(db)
        db.commit()
        assert reclaimed == len(stale) + len(b"failed")
        assert db.get(RemoteOperation, stale_operation.id).status == "cancelled"
        assert db.get(RemoteOperation, current_operation.id).status == "succeeded"
        assert db.get(RemoteOperation, historical_operation.id).status == "succeeded"
        assert db.get(RemoteOperation, active_operation.id).status == "running"

    assert operations.operation_file(current_operation).read_bytes() == current
    assert operations.operation_file(historical_operation).read_bytes() == historical
    assert operations.operation_file(active_operation, False).read_bytes() == active
    assert not operations.operation_file(stale_operation).exists()
    assert not operations.operation_file(failed_operation, False).exists()


def test_remote_model_is_prepared_and_cached_history_survives_refresh_and_offline(
    monitoring_client, agent, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Remote model node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "standalone.pt"
    version_one = b"remote-v1" * 80_000
    model.write_bytes(version_one)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    config = {
        "node_id": credential["node"]["id"],
        "training_cwd": str(tmp_path),
        "model_test_roots": [str(model_root)],
    }

    created = register(client, credential, model)
    assert created.status_code == 201, created.text
    asset = created.json()
    assert asset["status"] == "preparing"
    first_prepare = client.post(f"/api/model-assets/{asset['id']}/prepare").json()
    assert first_prepare["status"] == "preparing"
    work = claim(client, credential)
    assert work["kind"] == "fetch_model"
    assert work["payload"] == {"model_asset_id": asset["id"], "path": str(model)}
    agent.execute_operation(config, work, threading.Event())

    current = next(item for item in client.get("/api/model-assets").json()["items"] if item["id"] == asset["id"])
    first_sha = hashlib.sha256(version_one).hexdigest()
    assert current["status"] == "ready"
    assert current["sha256"] == first_sha and current["size"] == len(version_one)
    with factory() as db:
        first_operation = db.get(RemoteOperation, work["id"])
        assert first_operation.task_id == f"model-asset:{asset['id']}"
        first_cache = operations.operation_file(first_operation)
        assert first_cache.read_bytes() == version_one
        assert stat.S_IMODE(first_cache.stat().st_mode) == 0o600

    # Refresh creates a new immutable cache; it does not mutate the bytes used
    # by a test that already captured v1's digest and size.
    version_two = b"remote-v2" * 90_000
    model.write_bytes(version_two)
    refreshed = register(client, credential, model)
    assert refreshed.status_code == 201 and refreshed.json()["status"] == "preparing"
    second = claim(client, credential)
    assert second["id"] != work["id"]
    agent.execute_operation(config, second, threading.Event())
    with factory() as db:
        stored = db.get(ModelAsset, asset["id"])
        assert stored.sha256 == hashlib.sha256(version_two).hexdigest()
        old_path, old_asset, old_sha, old_size = inference._verified_model_asset_version(
            db, asset["id"], expected_sha256=first_sha, expected_size=len(version_one),
        )
        assert old_asset.id == asset["id"] and old_sha == first_sha and old_size == len(version_one)
        assert old_path.read_bytes() == version_one
        node = db.get(ComputeNode, credential["node"]["id"])
        node.last_seen_at = utcnow() - timedelta(hours=1)
        db.commit()
        current_path, _, _, _ = inference._verified_model_asset_version(db, asset["id"])
        assert current_path.read_bytes() == version_two


def test_creating_test_while_remote_model_is_transferring_does_not_fail_asset(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Concurrent model node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "concurrent.pt"
    content = b"remote-model-in-flight"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")

    asset = register(client, credential, model).json()
    work = claim(client, credential)
    assert asset["status"] == "preparing"
    assert work["kind"] == "fetch_model"

    create_test = client.post("/api/model-tests", json={
        "model_asset_id": asset["id"],
        "files": [{"name": "sample.jpg", "size": 1}],
    })
    assert create_test.status_code == 409
    assert create_test.json()["detail"] == "远程模型正在传输，请稍后重试"
    with factory() as db:
        assert db.get(ModelAsset, asset["id"]).status == "preparing"
        assert db.get(RemoteOperation, work["id"]).status == "running"

    uploaded = upload_single_model_chunk(client, credential, work, content)
    assert uploaded.status_code == 200, uploaded.text
    assert uploaded.json()["status"] == "succeeded"
    with factory() as db:
        stored_asset = db.get(ModelAsset, asset["id"])
        stored_operation = db.get(RemoteOperation, work["id"])
        assert stored_asset.status == "ready"
        assert stored_asset.sha256 == hashlib.sha256(content).hexdigest()
        assert stored_asset.size == len(content)
        assert stored_operation.status == "succeeded"


def test_remote_model_chunk_metadata_failure_is_retryable(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Metadata node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "metadata.pt"
    model.write_bytes(b"0123456789")
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")

    asset = register(client, credential, model).json()
    work = claim(client, credential)
    endpoint = (
        f"/api/compute/nodes/{credential['node']['id']}"
        f"/agent/operations/{work['id']}"
    )
    metadata = {
        "lease": work["lease"], "size": 10,
        "modified_at": utcnow().isoformat(), "mtime_ns": 100,
    }
    first = client.post(endpoint + "/chunk", headers=headers(credential), json={
        **metadata, "offset": 0, "data": base64.b64encode(b"01234").decode(), "final": False,
    })
    assert first.status_code == 200
    changed = client.post(endpoint + "/chunk", headers=headers(credential), json={
        **metadata, "mtime_ns": 101, "offset": 5,
        "data": base64.b64encode(b"56789").decode(), "final": True,
        "sha256": hashlib.sha256(b"0123456789").hexdigest(),
    })
    assert changed.status_code == 409
    failed = client.post(endpoint + "/result", headers=headers(credential), json={
        "lease": work["lease"], "success": False, "error": "model changed",
    })
    assert failed.status_code == 200 and failed.json()["status"] == "failed"
    listed = next(item for item in client.get("/api/model-assets").json()["items"] if item["id"] == asset["id"])
    assert listed["status"] == "failed" and listed["error"] == "model changed"
    assert client.post(f"/api/model-assets/{asset['id']}/prepare").json()["status"] == "failed"
    retried = client.post(f"/api/model-assets/{asset['id']}/prepare?retry=true")
    assert retried.status_code == 200 and retried.json()["status"] == "preparing"
    with factory() as db:
        operations_for_asset = db.query(RemoteOperation).filter_by(
            task_id=f"model-asset:{asset['id']}", kind="fetch_model",
        ).all()
        assert len(operations_for_asset) == 2


def test_remote_model_final_chunk_hashes_once_and_duplicate_is_idempotent(
    monitoring_client, tmp_path, monkeypatch,
):
    client, _ = monitoring_client
    credential = create(client, "Single hash node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "single-hash.pt"
    content = b"one-complete-checkpoint"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")

    register(client, credential, model)
    work = claim(client, credential)
    original_digest = operations.cached_fetch_digest
    digest_paths = []

    def counted_digest(operation, *, path=None, heartbeat=None):
        digest_paths.append(path or operations.operation_file(operation))
        return original_digest(operation, path=path, heartbeat=heartbeat)

    monkeypatch.setattr(operations, "cached_fetch_digest", counted_digest)
    first = upload_single_model_chunk(client, credential, work, content)
    repeated = upload_single_model_chunk(client, credential, work, content)

    assert first.status_code == 200 and first.json()["status"] == "succeeded"
    assert repeated.status_code == 200 and repeated.json() == first.json()
    assert len(digest_paths) == 1
    assert digest_paths[0].suffix == ".part"


def test_remote_model_finalization_uses_db_claim_and_heartbeats_across_sessions(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Cross worker finalization node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "cross-worker.pt"
    content = b"cross-worker-checkpoint"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")

    register(client, credential, model)
    work = claim(client, credential)
    original_digest = operations.cached_fetch_digest
    duplicate_statuses = []

    def inspect_digest(operation, *, path=None, heartbeat=None):
        duplicate = upload_single_model_chunk(client, credential, work, content)
        assert duplicate.status_code == 200
        duplicate_statuses.append(duplicate.json()["status"])

        stale = utcnow() - timedelta(minutes=10)
        with factory() as other:
            current = other.get(RemoteOperation, operation.id)
            assert current.status == "finalizing"
            current.updated_at = stale
            other.commit()

        result = original_digest(operation, path=path, heartbeat=heartbeat)
        with factory() as other:
            refreshed = other.get(RemoteOperation, operation.id)
            assert refreshed.status == "finalizing"
            assert operations.as_utc(refreshed.updated_at) > stale
        polled = client.post(
            f"/api/compute/nodes/{credential['node']['id']}/agent/operations/poll",
            headers=headers(credential),
        )
        assert polled.status_code == 200 and polled.json()["operation"] is None
        return result

    monkeypatch.setattr(operations, "cached_fetch_digest", inspect_digest)
    response = upload_single_model_chunk(client, credential, work, content)

    assert response.status_code == 200 and response.json()["status"] == "succeeded"
    assert duplicate_statuses == ["finalizing"]


def test_remote_model_final_chunk_does_not_publish_after_lease_reassignment(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client, "Lease CAS node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "lease.pt"
    content = b"lease-protected-checkpoint"
    model.write_bytes(content)
    report_remote_model_node(client, credential, model_root)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "remote-cache")

    asset = register(client, credential, model).json()
    work = claim(client, credential)
    original_digest = operations.cached_fetch_digest
    replacement_lease = "f" * 32

    def reassign_after_digest(operation, *, path=None, heartbeat=None):
        result = original_digest(operation, path=path, heartbeat=heartbeat)
        with factory() as other:
            current = other.get(RemoteOperation, operation.id)
            current.status = "running"
            current.lease = replacement_lease
            current.updated_at = utcnow()
            other.commit()
        return result

    monkeypatch.setattr(operations, "cached_fetch_digest", reassign_after_digest)
    response = upload_single_model_chunk(client, credential, work, content)

    assert response.status_code == 409
    assert response.json()["detail"] == "远程操作已重新分配"
    with factory() as db:
        stored_operation = db.get(RemoteOperation, work["id"])
        stored_asset = db.get(ModelAsset, asset["id"])
        assert stored_operation.status == "running"
        assert stored_operation.lease == replacement_lease
        assert stored_asset.status == "preparing"
        assert operations.operation_file(stored_operation, False).read_bytes() == content
        assert not operations.operation_file(stored_operation).exists()


def test_remote_model_registration_enforces_capability_roots_and_canonical_lowercase_path(
    monitoring_client, tmp_path,
):
    client, _ = monitoring_client
    credential = create(client, "Path policy node")
    model_root = tmp_path / "pt"
    model_root.mkdir()
    model = model_root / "ok.pt"
    model.write_bytes(b"model")

    old = sample()
    old["agent_version"] = "2.0.2"
    old["training"]["model_test_roots"] = [str(model_root)]
    assert client.post(
        f"/api/compute/nodes/{credential['node']['id']}/report",
        headers=headers(credential), json=old,
    ).status_code == 200
    response = register(client, credential, model)
    assert response.status_code == 409 and "2.0.3" in response.json()["detail"]

    report_remote_model_node(client, credential, model_root)
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"outside")
    assert register(client, credential, outside).status_code == 403
    assert register(client, credential, model_root / "MODEL.PT").status_code == 422
    noncanonical = client.post("/api/model-assets/remote", json={
        "execution_node_id": credential["node"]["id"],
        "path": str(model_root) + "/nested/../ok.pt",
    })
    assert noncanonical.status_code == 422
