from datetime import timedelta
import hashlib
from io import BytesIO
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import sessionmaker
from starlette.requests import ClientDisconnect

import app.inference as inference
import app.main as main_module
import app.model_assets as model_assets
import app.model_test_storage as storage
import app.services as services_module
from app.local_deletions import LocalDeletionWorker
from app.database import get_db
from app.main import app
from app.models import (
    Artifact,
    ComputeNode,
    ModelAsset,
    ModelTest,
    ModelTestImage,
    MetricPoint,
    RemoteOperation,
    TaskGroup,
    TrainingRun,
    utcnow,
)
from test_services import make_session, make_task


@pytest.fixture
def image_test_client(tmp_path, monkeypatch):
    history_root = tmp_path / "history"
    work_root = tmp_path / "work"
    monkeypatch.setattr(storage, "HISTORY_ROOT", history_root)
    monkeypatch.setattr(storage, "WORK_ROOT", work_root)
    monkeypatch.setattr(inference, "HISTORY_ROOT", history_root)
    monkeypatch.setattr(inference, "WORK_ROOT", work_root)
    monkeypatch.setattr(storage, "MODEL_TEST_HISTORY_QUOTA_BYTES", 0)
    monkeypatch.setattr(storage, "MODEL_TEST_TASK_QUOTA_BYTES", 0)
    monkeypatch.setattr(storage, "MODEL_TEST_MIN_FREE_BYTES", 0)
    monkeypatch.setattr(storage, "MODEL_TEST_IMAGE_RESERVE_BYTES", 1024)
    with make_session(tmp_path) as db:
        factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
        test_worker = inference.InferenceWorker(session_factory=factory, interval=0.01)
        monkeypatch.setattr(inference, "worker", test_worker)
        monkeypatch.setattr(services_module, "RUNS_ROOT", tmp_path / "runs")
        monkeypatch.setattr(services_module, "RUNTIME_ROOT", tmp_path / "runtime")
        monkeypatch.setattr(services_module, "LOG_ROOT", tmp_path / "logs")
        run_dir = tmp_path / "runs" / "detect" / "task-test" / "attempt-001"
        run_dir.mkdir(parents=True)
        model = run_dir / "best.pt"
        model.write_bytes(b"test-checkpoint")
        task = make_task(status="completed")
        run = TrainingRun(
            id="run-test", task=task, attempt_number=1, launch_mode="fresh", status="completed",
            model_path="base.pt", data_path="data.yaml", run_dir=str(run_dir), parameter_snapshot={},
        )
        artifact = Artifact(
            task=task, run=run, kind="weight", name="weights/best.pt", path=str(model),
            size=model.stat().st_size, modified_at=utcnow(),
        )
        db.add_all([task, run, artifact])
        db.commit()

        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        try:
            yield TestClient(app), db, artifact, factory, test_worker
        finally:
            app.dependency_overrides.pop(get_db, None)


def image_bytes(color="white"):
    stream = BytesIO()
    Image.new("RGB", (40, 30), color).save(stream, "PNG")
    return stream.getvalue()


def create(client, artifact, names=None, **overrides):
    return client.post("/api/model-tests", json={
        "artifact_id": artifact.id,
        "confidence": 0.4,
        "files": [
            {"name": name, "size": len(image_bytes())}
            for name in (names or ["input.png"])
        ],
        **overrides,
    })


def upload_all(client, test_id, count=1):
    for index in range(count):
        assert client.put(f"/api/model-tests/{test_id}/images/{index}", content=image_bytes()).status_code == 204


def install_fake_process(monkeypatch, expected_images=1, seen_models=None, detected_task=None):
    class FakeProcess:
        def __init__(self, command, **_kwargs):
            model = Path(command[command.index("--model") + 1])
            if seen_models is not None:
                seen_models.append(model.read_bytes())
            output = Path(command[command.index("--output") + 1])
            summary = Path(command[command.index("--summary") + 1])
            output.mkdir()
            task_type = detected_task or (
                command[command.index("--expected-task") + 1] if "--expected-task" in command else "detect"
            )
            images = []
            for index in range(expected_images):
                Image.new("RGB", (40, 30), "red").save(output / f"{index}.jpg")
                objects = ([{
                    "class_id": 0, "class_name": "person", "confidence": 0.91,
                    "bbox_xyxy": [1, 2, 20, 25],
                }] if index == 0 else [])
                if objects:
                    if task_type == "classify":
                        objects[0].pop("bbox_xyxy")
                    elif task_type == "segment":
                        objects[0].update(mask_available=True, mask_polygon_xy=[[1, 2], [20, 2], [20, 25]])
                    elif task_type == "pose":
                        objects[0]["keypoints"] = [{"index": 0, "x": 4, "y": 5, "confidence": 0.8}]
                    elif task_type == "obb":
                        objects[0].update(obb_points=[[1, 2], [20, 2], [20, 25], [1, 25]], obb_xywhr=[10, 10, 5, 5, 0.2])
                images.append({
                    "index": index, "prediction_count": len(objects), "objects": objects,
                    "timings": {"preprocess": 3, "inference": 25, "postprocess": 2},
                })
            summary.write_text(json.dumps({
                "task_type": task_type,
                "device": "CPU", "imgsz": 640,
                "timings": {
                    "model_load": 120, "preprocess": 3 * expected_images,
                    "inference": 25 * expected_images, "postprocess": 2 * expected_images,
                    "total": 160,
                },
                "images": images,
            }), encoding="utf-8")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(inference.subprocess, "Popen", FakeProcess)


def configure_standalone_models(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    model_root = project_root / "pt"
    model_root.mkdir(parents=True)
    monkeypatch.setattr(model_assets, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(model_assets, "MODEL_TEST_ROOTS", [model_root])
    return project_root, model_root


def test_local_model_assets_are_portable_listed_and_restricted(image_test_client, tmp_path, monkeypatch):
    client, db, _, _, _ = image_test_client
    project_root, model_root = configure_standalone_models(tmp_path, monkeypatch)
    checkpoint = model_root / "yan-best.pt"
    checkpoint.write_bytes(b"yan-checkpoint")

    created = client.post("/api/model-assets/local", json={
        "path": str(checkpoint), "name": "yan v1",
    })
    assert created.status_code == 201
    asset = created.json()
    assert asset["path"] == "pt/yan-best.pt"
    assert asset["name"] == "yan v1"
    assert asset["source_type"] == "local_path"
    assert asset["execution_node_id"] == "local"
    assert asset["sha256"] == hashlib.sha256(b"yan-checkpoint").hexdigest()
    assert asset["size"] == len(b"yan-checkpoint")
    db.expire_all()
    assert db.get(ModelAsset, asset["id"]).path == "pt/yan-best.pt"

    listed = client.get("/api/model-assets").json()
    assert [item["id"] for item in listed["items"]] == [asset["id"]]
    assert listed["local_root"] == "pt"
    assert listed["local_roots"] == ["pt"]

    # Re-registering one logical path refreshes the same reusable source.
    checkpoint.write_bytes(b"yan-checkpoint-v2")
    refreshed = client.post("/api/model-assets/local", json={"path": "pt/yan-best.pt"})
    assert refreshed.status_code == 201
    assert refreshed.json()["id"] == asset["id"]
    assert refreshed.json()["sha256"] == hashlib.sha256(b"yan-checkpoint-v2").hexdigest()

    outside = project_root / "outside.pt"
    outside.write_bytes(b"outside")
    assert client.post("/api/model-assets/local", json={"path": str(outside)}).status_code == 403
    wrong_suffix = model_root / "model.onnx"
    wrong_suffix.write_bytes(b"onnx")
    assert client.post("/api/model-assets/local", json={"path": str(wrong_suffix)}).status_code == 422
    symlink = model_root / "linked.pt"
    symlink.symlink_to(checkpoint)
    response = client.post("/api/model-assets/local", json={"path": str(symlink)})
    assert response.status_code == 422
    assert "符号链接" in response.json()["detail"]


def test_training_history_group_is_snapshotted_and_refreshed_for_reruns(image_test_client):
    client, db, artifact, _, _ = image_test_client
    original_group = TaskGroup(id="group-original", name="烟草初始实验")
    current_group = TaskGroup(id="group-current", name="烟草新增数据")
    artifact.task.group_id = original_group.id
    db.add_all([original_group, current_group])
    db.commit()

    source = create(client, artifact).json()
    assert source["model"]["history_group"] == {
        "kind": "task_group", "id": original_group.id, "name": original_group.name,
    }

    source_record = db.get(ModelTest, source["id"])
    source_record.status = "completed"
    source_record.finished_at = utcnow()
    artifact.task.group_id = current_group.id
    db.commit()

    rerun_response = create(client, artifact, rerun_of_test_id=source["id"])

    assert rerun_response.status_code == 201
    rerun = rerun_response.json()
    assert rerun["model"]["history_group"] == {
        "kind": "task_group", "id": current_group.id, "name": current_group.name,
    }
    assert rerun["model"]["weight_name"] == source["model"]["weight_name"]
    assert rerun["model"]["source_path"] == source["model"]["source_path"]
    db.refresh(source_record)
    assert source_record.model_snapshot["history_group"]["id"] == original_group.id


def test_standalone_history_groups_include_model_source_and_remote_node(
    image_test_client, tmp_path, monkeypatch,
):
    client, db, _, _, _ = image_test_client
    _, model_root = configure_standalone_models(tmp_path, monkeypatch)
    local_model = model_root / "local.pt"
    local_model.write_bytes(b"local-model")
    local_asset = client.post("/api/model-assets/local", json={"path": str(local_model)}).json()
    local_test = client.post("/api/model-tests", json={
        "model_asset_id": local_asset["id"],
        "files": [{"name": "local.png", "size": len(image_bytes())}],
    }).json()
    assert local_test["model"]["history_group"] == {
        "kind": "standalone_model",
        "id": "local_path:local",
        "name": "独立模型 · 本机模型",
        "source_type": "local_path",
        "execution_node_id": "local",
        "execution_node_name": None,
    }

    node = ComputeNode(id="remote-node", name="远程 GPU 节点")
    remote_asset = ModelAsset(
        id="a" * 32,
        name="remote-best.pt",
        source_type="remote_path",
        execution_node_id=node.id,
        path="/models/remote-best.pt",
        sha256=hashlib.sha256(b"remote-model").hexdigest(),
        size=len(b"remote-model"),
        status="ready",
    )
    remote_model = tmp_path / "remote-best.pt"
    remote_model.write_bytes(b"remote-model")
    db.add_all([node, remote_asset])
    db.commit()
    monkeypatch.setattr(
        inference,
        "_verified_model_asset_version",
        lambda _db, _asset_id, **_kwargs: (
            remote_model, remote_asset, remote_asset.sha256, remote_asset.size,
        ),
    )

    remote_test_response = client.post("/api/model-tests", json={
        "model_asset_id": remote_asset.id,
        "files": [{"name": "remote.png", "size": len(image_bytes())}],
    })

    assert remote_test_response.status_code == 201
    assert remote_test_response.json()["model"]["history_group"] == {
        "kind": "standalone_model",
        "id": "remote_path:remote-node",
        "name": "独立模型 · 远程 GPU 节点",
        "source_type": "remote_path",
        "execution_node_id": "remote-node",
        "execution_node_name": "远程 GPU 节点",
    }


@pytest.mark.parametrize("task_type", ["detect", "segment", "classify", "pose", "obb"])
def test_standalone_model_test_runs_without_training_task_and_has_global_history(
    image_test_client, tmp_path, monkeypatch, task_type,
):
    client, db, artifact, _, test_worker = image_test_client
    _, model_root = configure_standalone_models(tmp_path, monkeypatch)
    checkpoint = model_root / "standalone.pt"
    checkpoint.write_bytes(b"standalone-checkpoint")
    asset = client.post("/api/model-assets/local", json={"path": str(checkpoint)}).json()

    created_response = client.post("/api/model-tests", json={
        "model_asset_id": asset["id"],
        "confidence": 0.3,
        "iou": 0.6,
        "files": [{"name": "standalone.png", "size": len(image_bytes())}],
    })
    assert created_response.status_code == 201
    created = created_response.json()
    assert created["task_id"] is None
    assert created["run_id"] is None
    assert created["artifact_id"] is None
    assert created["model_asset_id"] == asset["id"]
    assert created["model_source_type"] == "local_path"
    assert created["model_source_path"] == "pt/standalone.pt"
    assert created["execution_node_id"] == "local"
    assert created["model_source_name"] == "standalone.pt"
    assert created["task_type"] is None
    test_id = created["id"]
    directory = storage.test_history_root(storage.STANDALONE_HISTORY_SCOPE, test_id)
    assert directory.is_dir()

    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    install_fake_process(monkeypatch, detected_task=task_type)
    assert test_worker.process_one()

    detail = client.get(f"/api/model-tests/{test_id}").json()
    assert detail["status"] == "completed"
    assert detail["task_type"] == task_type
    assert detail["files"][0]["detections"] == 1
    global_history = client.get("/api/model-tests", params={"source_type": "local_path"}).json()
    assert [item["id"] for item in global_history["items"]] == [test_id]
    assert global_history["items"][0]["task_id"] is None
    task_history = client.get(f"/api/tasks/{artifact.task_id}/model-tests").json()
    assert all(item["id"] != test_id for item in task_history["items"])
    assert client.get(f"/api/model-tests/{test_id}/download").status_code == 200

    assert client.delete(f"/api/model-tests/{test_id}").status_code == 204
    assert not directory.exists()
    db.expire_all()
    assert db.get(ModelTest, test_id) is None
    assert db.get(ModelAsset, asset["id"]) is not None


def test_historical_local_rerun_failure_does_not_poison_current_asset(
    image_test_client, tmp_path, monkeypatch,
):
    client, db, _, _, test_worker = image_test_client
    _, model_root = configure_standalone_models(tmp_path, monkeypatch)
    checkpoint = model_root / "versioned.pt"
    version_one = b"local-version-one"
    version_two = b"local-version-two"
    checkpoint.write_bytes(version_one)
    asset = client.post("/api/model-assets/local", json={"path": str(checkpoint)}).json()

    source = client.post("/api/model-tests", json={
        "model_asset_id": asset["id"],
        "files": [{"name": "source.png", "size": len(image_bytes())}],
    }).json()
    upload_all(client, source["id"])
    assert client.post(f"/api/model-tests/{source['id']}/start").status_code == 202
    install_fake_process(monkeypatch)
    assert test_worker.process_one()
    assert client.get(f"/api/model-tests/{source['id']}").json()["status"] == "completed"

    checkpoint.write_bytes(version_two)
    refreshed = client.post("/api/model-assets/local", json={"path": str(checkpoint)})
    assert refreshed.status_code == 201
    assert refreshed.json()["sha256"] == hashlib.sha256(version_two).hexdigest()

    rerun = client.post("/api/model-tests", json={
        "model_asset_id": asset["id"],
        "rerun_of_test_id": source["id"],
        "files": [{"name": "rerun.png", "size": len(image_bytes())}],
    })
    assert rerun.status_code == 409
    assert rerun.json()["detail"] == "原模型权重版本已变化，无法按原权重重测"

    db.expire_all()
    current = db.get(ModelAsset, asset["id"])
    assert current.status == "ready"
    assert current.error is None
    assert current.sha256 == hashlib.sha256(version_two).hexdigest()
    assert current.size == len(version_two)


def test_standalone_start_rejects_a_model_replaced_after_test_creation(
    image_test_client, tmp_path, monkeypatch,
):
    client, db, _, _, _ = image_test_client
    _, model_root = configure_standalone_models(tmp_path, monkeypatch)
    checkpoint = model_root / "replaceable.pt"
    checkpoint.write_bytes(b"version-one")
    asset = client.post("/api/model-assets/local", json={"path": str(checkpoint)}).json()
    created = client.post("/api/model-tests", json={
        "model_asset_id": asset["id"],
        "files": [{"name": "image.png", "size": len(image_bytes())}],
    }).json()
    upload_all(client, created["id"])

    checkpoint.write_bytes(b"version-two")
    response = client.post(f"/api/model-tests/{created['id']}/start")
    assert response.status_code == 409
    assert "版本已变化" in response.json()["detail"]
    db.expire_all()
    record = db.get(ModelTest, created["id"])
    assert record.status == "failed"
    assert "版本已变化" in record.error


def test_completed_test_persists_and_is_listed_after_reopen(image_test_client, monkeypatch):
    client, db, artifact, _, test_worker = image_test_client
    response = create(client, artifact, ["重复.png", "重复.png"])
    assert response.status_code == 201
    created = response.json()
    assert created["task_id"] == artifact.task_id
    assert created["run_id"] == artifact.run_id
    assert created["source_run_id"] == artifact.run_id
    assert created["artifact_name"] == "weights/best.pt"
    assert created["model_name"] == "best.pt"
    assert created["attempt_number"] == 1
    assert created["source_attempt_number"] == 1
    assert created["model"]["weight_name"] == "best.pt"
    assert created["model"]["history_group"] == {
        "kind": "ungrouped", "id": None, "name": "未分组",
    }
    expected_sha256 = hashlib.sha256(b"test-checkpoint").hexdigest()
    assert created["model_sha256"] == expected_sha256
    assert created["model_size"] == len(b"test-checkpoint")
    assert created["model"]["sha256"] == expected_sha256
    test_id = created["id"]
    directory = storage.test_history_root(artifact.task_id, test_id)
    assert directory.parent.name == artifact.task_id
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 409
    upload_all(client, test_id, 2)
    assert client.post(f"/api/model-tests/{test_id}/start").json()["status"] == "queued"

    install_fake_process(monkeypatch, expected_images=2)
    assert test_worker.process_one()

    detail = client.get(f"/api/model-tests/{test_id}").json()
    assert detail["status"] == "completed"
    assert detail["files"][0]["detections"] == 1
    assert detail["files"][1]["detections"] == 0
    assert detail["timings"]["inference"] == 50
    assert detail["storage_bytes"] > 0
    listed = client.get(f"/api/tasks/{artifact.task_id}/model-tests").json()
    assert [item["id"] for item in listed["items"]] == [test_id]
    assert listed["next_cursor"] is None
    assert "timings" not in listed["items"][0]
    assert "objects" not in listed["items"][0]["files"][0]
    assert "timings" not in listed["items"][0]["files"][0]
    scoped = client.get(f"/api/tasks/{artifact.task_id}/model-tests/{test_id}")
    assert scoped.status_code == 200
    assert client.get(f"/api/tasks/another-task/model-tests/{test_id}").status_code == 404

    original = client.get(f"/api/model-tests/{test_id}/images/0/original")
    result = client.get(f"/api/model-tests/{test_id}/images/0/result?download=true")
    assert Image.open(BytesIO(original.content)).size == (40, 30)
    assert result.headers["content-disposition"] == 'attachment; filename="detection-1.jpg"'
    download = client.get(f"/api/model-tests/{test_id}/download")
    assert download.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(BytesIO(download.content)) as archive:
        assert set(archive.namelist()) == {
            "detections.json", "results/1-重复.png.jpg", "results/2-重复.png.jpg",
        }
        exported = json.loads(archive.read("detections.json"))
        assert exported["files"][0]["objects"][0]["confidence"] == 0.91
        assert exported["model_version"] == {
            "sha256": expected_sha256,
            "size": len(b"test-checkpoint"),
        }

    # A new worker reconstructs state from the database and keeps terminal files.
    restarted = inference.InferenceWorker(session_factory=test_worker.session_factory)
    restarted._recover()
    db.expire_all()
    assert db.get(ModelTest, test_id).status == "completed"
    assert directory.is_dir()
    assert client.get(f"/api/model-tests/{test_id}").status_code == 200


def test_soft_deleted_task_hides_direct_model_test_resources_until_restore(
    image_test_client, monkeypatch,
):
    client, _, artifact, _, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    install_fake_process(monkeypatch)
    assert test_worker.process_one()

    direct_resources = (
        f"/api/model-tests/{test_id}",
        f"/api/model-tests/{test_id}/images/0/original",
        f"/api/model-tests/{test_id}/images/0/result",
        f"/api/model-tests/{test_id}/download",
    )
    assert all(client.get(path).status_code == 200 for path in direct_resources)

    assert client.delete(f"/api/tasks/{artifact.task_id}").status_code == 204
    for path in direct_resources:
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["detail"] == "模型检测记录不存在"

    assert client.post(f"/api/tasks/{artifact.task_id}/restore").status_code == 200
    assert all(client.get(path).status_code == 200 for path in direct_resources)


def test_history_list_uses_stable_compact_cursor_pages(image_test_client):
    client, db, artifact, _, _ = image_test_client
    created_at = utcnow()
    records = []
    for index in range(51):
        record = ModelTest(
            id=f"{index:032x}", task_id=artifact.task_id, run_id=artifact.run_id,
            source_run_id=artifact.run_id, artifact_id=artifact.id,
            artifact_name="weights/best.pt", model_name="best.pt",
            attempt_number=1, source_attempt_number=1, status="completed",
            confidence=0.4, iou=0.7, model_snapshot={},
            timings={"inference": index}, created_at=created_at,
        )
        record.images.append(ModelTestImage(
            ordinal=0, original_name=f"image-{index}.png", declared_size=100,
            uploaded=True, detections=1,
            objects=[{"class_id": 0, "class_name": "person", "confidence": 0.9}],
            timings={"inference": 1},
        ))
        records.append(record)
    db.add_all(records)
    db.commit()
    expected_ids = [f"{index:032x}" for index in reversed(range(51))]

    first = client.get(f"/api/tasks/{artifact.task_id}/model-tests").json()
    assert [item["id"] for item in first["items"]] == expected_ids[:50]
    assert first["next_cursor"]
    assert "timings" not in first["items"][0]
    assert "objects" not in first["items"][0]["files"][0]
    assert "timings" not in first["items"][0]["files"][0]

    second = client.get(
        f"/api/tasks/{artifact.task_id}/model-tests",
        params={"cursor": first["next_cursor"], "limit": 50},
    ).json()
    assert [item["id"] for item in second["items"]] == expected_ids[50:]
    assert second["next_cursor"] is None

    paged_ids = []
    cursor = None
    while True:
        response = client.get(
            f"/api/tasks/{artifact.task_id}/model-tests",
            params={"limit": 17, **({"cursor": cursor} if cursor else {})},
        )
        assert response.status_code == 200
        page = response.json()
        paged_ids.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break
    assert paged_ids == expected_ids
    assert len(paged_ids) == len(set(paged_ids))

    detail = client.get(f"/api/tasks/{artifact.task_id}/model-tests/{expected_ids[0]}").json()
    assert detail["timings"]["inference"] == 50
    assert detail["files"][0]["objects"][0]["class_name"] == "person"
    assert detail["files"][0]["timings"]["inference"] == 1

    assert client.get(
        f"/api/tasks/{artifact.task_id}/model-tests", params={"cursor": "invalid"},
    ).status_code == 422
    assert client.get(
        f"/api/tasks/{artifact.task_id}/model-tests", params={"limit": 0},
    ).status_code == 422
    assert client.get(
        f"/api/tasks/{artifact.task_id}/model-tests", params={"limit": 101},
    ).status_code == 422


def test_history_list_filters_and_paginates_by_context_run(image_test_client):
    client, db, artifact, _, _ = image_test_client
    other_run = TrainingRun(
        id="run-other", task=artifact.task, attempt_number=2, launch_mode="resume", status="completed",
        model_path="base.pt", data_path="data.yaml", run_dir=str(Path(artifact.path).parent),
        parameter_snapshot={},
    )
    db.add(other_run)
    created_at = utcnow()
    records = []
    for index in range(8):
        context_run = artifact.run if index % 2 == 0 else other_run
        records.append(ModelTest(
            id=f"{index + 100:032x}", task_id=artifact.task_id, run_id=context_run.id,
            source_run_id=artifact.run_id, artifact_id=artifact.id,
            artifact_name="weights/best.pt", model_name="best.pt",
            attempt_number=context_run.attempt_number, source_attempt_number=1,
            status="completed", confidence=0.4, iou=0.7, model_snapshot={},
            created_at=created_at,
        ))
    db.add_all(records)
    db.commit()

    ids = []
    cursor = None
    while True:
        response = client.get(f"/api/tasks/{artifact.task_id}/model-tests", params={
            "run_id": other_run.id,
            "limit": 2,
            **({"cursor": cursor} if cursor else {}),
        })
        assert response.status_code == 200
        page = response.json()
        assert all(item["run_id"] == other_run.id for item in page["items"])
        ids.extend(item["id"] for item in page["items"])
        cursor = page["next_cursor"]
        if cursor is None:
            break

    assert ids == [f"{index + 100:032x}" for index in (7, 5, 3, 1)]
    assert client.get(
        f"/api/tasks/{artifact.task_id}/model-tests", params={"run_id": "bad/run"},
    ).status_code == 422


def test_explicit_delete_removes_database_row_and_owned_files(image_test_client):
    client, db, artifact, _, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    directory = storage.test_history_root(artifact.task_id, test_id)
    assert directory.is_dir()

    assert client.delete(f"/api/model-tests/{test_id}").status_code == 204
    assert client.get(f"/api/model-tests/{test_id}").status_code == 404
    db.expire_all()
    assert db.get(ModelTest, test_id) is None
    assert not directory.exists()
    assert Path(artifact.path).read_bytes() == b"test-checkpoint"
    assert test_id not in inference.worker._cancelled_ids


def test_abandon_upload_preserves_images_and_frees_active_capacity(image_test_client):
    client, db, artifact, _, _ = image_test_client
    test_id = create(client, artifact, ["kept.png", "missing.png"]).json()["id"]
    upload_all(client, test_id)
    directory = storage.test_history_root(artifact.task_id, test_id)
    kept_image = directory / "images" / "0.jpg"
    assert kept_image.is_file()

    response = client.post(f"/api/model-tests/{test_id}/abandon")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert response.json()["files"][0]["uploaded"] is True
    assert response.json()["files"][1]["uploaded"] is False
    assert response.json()["finished_at"] is not None
    assert kept_image.is_file()
    assert client.post(f"/api/model-tests/{test_id}/abandon").status_code == 200

    # A cancelled upload no longer consumes one of the twenty active slots.
    for index in range(20):
        assert create(client, artifact, [f"slot-{index}.png"]).status_code == 201
    assert create(client, artifact, ["over-limit.png"]).status_code == 429
    db.expire_all()
    assert db.get(ModelTest, test_id).status == "cancelled"


def test_abandon_does_not_cancel_a_submitted_test(image_test_client):
    client, db, artifact, _, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202

    response = client.post(f"/api/model-tests/{test_id}/abandon")
    assert response.status_code == 409
    assert response.json()["detail"] == "检测已提交，关闭页面不会取消检测"
    db.expire_all()
    assert db.get(ModelTest, test_id).status == "queued"


def test_abandon_during_normalization_does_not_publish_inflight_image(image_test_client, monkeypatch):
    client, _, artifact, factory, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    directory = storage.test_history_root(artifact.task_id, test_id)
    normalization_started = threading.Event()
    release_normalization = threading.Event()
    real_normalize = inference.normalize_image
    responses = []

    def blocking_normalize(source, destination):
        normalization_started.set()
        if not release_normalization.wait(3):
            raise RuntimeError("test did not release image normalization")
        real_normalize(source, destination)

    def run_upload() -> None:
        responses.append(client.put(
            f"/api/model-tests/{test_id}/images/0", content=image_bytes(),
        ))

    monkeypatch.setattr(inference, "normalize_image", blocking_normalize)
    upload_thread = threading.Thread(target=run_upload, name="model-test-upload", daemon=True)
    upload_thread.start()
    assert normalization_started.wait(2)
    try:
        with factory() as abandon_db:
            abandoned = inference.abandon_test(test_id, abandon_db)
        assert abandoned["status"] == "cancelled"
        assert abandoned["files"][0]["uploaded"] is False
        assert not (directory / "images" / "0.jpg").exists()
    finally:
        release_normalization.set()
        upload_thread.join(3)

    assert not upload_thread.is_alive()
    assert [response.status_code for response in responses] == [409]
    assert not (directory / "images" / "0.jpg").exists()
    assert not (directory / ".0.normalized.part").exists()
    with factory() as check_db:
        record = check_db.get(ModelTest, test_id)
        assert record.status == "cancelled"
        assert record.images[0].uploaded is False


def test_upload_disconnect_marks_upload_only_record_cancelled(image_test_client, monkeypatch):
    client, db, artifact, _, _ = image_test_client
    test_id = create(client, artifact).json()["id"]

    async def disconnected_stream():
        raise ClientDisconnect()
        yield b""  # pragma: no cover - keeps this an async generator

    class DisconnectingRequest:
        async def stream(self):
            async for chunk in disconnected_stream():
                yield chunk

    response = None
    try:
        import asyncio

        asyncio.run(inference.upload_image(test_id, 0, DisconnectingRequest(), db))
    except inference.HTTPException as exc:
        response = exc
    assert response is not None
    assert response.status_code == 400
    db.expire_all()
    record = db.get(ModelTest, test_id)
    assert record.status == "cancelled"
    assert record.error == "图片上传连接已中断，检测未启动"


def test_delete_waits_for_worker_cleanup_before_removing_history(image_test_client, monkeypatch):
    client, db, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    directory = storage.test_history_root(artifact.task_id, test_id)
    process_waiting = threading.Event()
    process_killed = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()

    class BlockingProcess:
        def __init__(self, *_args, **_kwargs):
            pass

        def wait(self, timeout=None):
            process_waiting.set()
            if not process_killed.wait(3):
                raise inference.subprocess.TimeoutExpired("inference", timeout)
            return -9

        def poll(self):
            return -9 if process_killed.is_set() else None

        def kill(self):
            process_killed.set()

    monkeypatch.setattr(inference.subprocess, "Popen", BlockingProcess)
    real_delete_work = inference.delete_test_work

    def blocking_delete_work(candidate: str) -> None:
        if (
            candidate == test_id
            and process_killed.is_set()
            and threading.current_thread().name == "model-test-worker"
        ):
            cleanup_entered.set()
            if not release_cleanup.wait(3):
                raise RuntimeError("test did not release worker cleanup")
        real_delete_work(candidate)

    monkeypatch.setattr(inference, "delete_test_work", blocking_delete_work)
    worker_errors: list[BaseException] = []
    delete_errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            test_worker.process_one()
        except BaseException as exc:
            worker_errors.append(exc)

    def run_delete() -> None:
        try:
            with factory() as delete_db:
                inference.delete_test(test_id, delete_db)
        except BaseException as exc:
            delete_errors.append(exc)

    worker_thread = threading.Thread(target=run_worker, name="model-test-worker", daemon=True)
    delete_thread = threading.Thread(target=run_delete, name="model-test-delete", daemon=True)
    worker_thread.start()
    assert process_waiting.wait(2)
    delete_thread.start()
    try:
        assert cleanup_entered.wait(2)
        delete_thread.join(0.1)
        assert delete_thread.is_alive()
        assert directory.is_dir()
        with factory() as check_db:
            assert check_db.get(ModelTest, test_id) is not None
    finally:
        release_cleanup.set()
        worker_thread.join(3)
        delete_thread.join(3)

    assert not worker_thread.is_alive()
    assert not delete_thread.is_alive()
    assert worker_errors == []
    assert delete_errors == []
    with factory() as check_db:
        assert check_db.get(ModelTest, test_id) is None
    assert not directory.exists()


def test_task_purge_waits_for_pre_popen_worker_and_prevents_launch(image_test_client, monkeypatch):
    client, _, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    directory = storage.test_history_root(artifact.task_id, test_id)
    environment_started = threading.Event()
    release_environment = threading.Event()
    cancel_waiting = threading.Event()
    popen_called = threading.Event()
    real_wait_for_exit = test_worker.wait_for_test_exit

    def blocking_environment():
        environment_started.set()
        if not release_environment.wait(3):
            raise RuntimeError("test did not release environment setup")
        return {}

    def observed_wait(test_id_arg: str, timeout: float = 5.0) -> bool:
        cancel_waiting.set()
        return real_wait_for_exit(test_id_arg, timeout)

    def unexpected_popen(*_args, **_kwargs):
        popen_called.set()
        raise AssertionError("cancelled inference must not launch a subprocess")

    monkeypatch.setattr(inference, "training_environment", blocking_environment)
    monkeypatch.setattr(test_worker, "wait_for_test_exit", observed_wait)
    monkeypatch.setattr(inference.subprocess, "Popen", unexpected_popen)
    worker_errors: list[BaseException] = []
    cancel_errors: list[BaseException] = []

    def run_worker() -> None:
        try:
            test_worker.process_one()
        except BaseException as exc:
            worker_errors.append(exc)

    def run_cancel() -> None:
        try:
            test_worker.cancel_task_tests(artifact.task_id, purge=True)
        except BaseException as exc:
            cancel_errors.append(exc)

    worker_thread = threading.Thread(target=run_worker, name="model-test-pre-popen", daemon=True)
    cancel_thread = threading.Thread(target=run_cancel, name="model-test-cancel", daemon=True)
    worker_thread.start()
    assert environment_started.wait(2)
    cancel_thread.start()
    try:
        assert cancel_waiting.wait(2)
        cancel_thread.join(0.1)
        assert cancel_thread.is_alive()
        assert directory.is_dir()
        assert not popen_called.is_set()
    finally:
        release_environment.set()
        worker_thread.join(3)
        cancel_thread.join(3)

    assert not worker_thread.is_alive()
    assert not cancel_thread.is_alive()
    assert worker_errors == []
    assert cancel_errors == []
    assert not popen_called.is_set()
    assert not storage.test_work_root(test_id).exists()
    with factory() as check_db:
        record = check_db.get(ModelTest, test_id)
        assert record.status == "cancelled"
        assert record.delete_requested


def test_delete_timeout_keeps_record_and_history_for_retry(image_test_client, monkeypatch):
    client, db, artifact, _, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    directory = storage.test_history_root(artifact.task_id, test_id)
    with test_worker._state_changed:
        test_worker._current_id = test_id
        test_worker._current_task_id = artifact.task_id
    monkeypatch.setattr(test_worker, "wait_for_test_exit", lambda *_args, **_kwargs: False)

    try:
        response = client.delete(f"/api/model-tests/{test_id}")
        assert response.status_code == 409
        assert response.json()["detail"] == "正在停止模型检测，请稍后重试删除"
        db.expire_all()
        record = db.get(ModelTest, test_id)
        assert record is not None
        assert record.delete_requested
        assert directory.is_dir()
    finally:
        with test_worker._state_changed:
            test_worker._current_id = None
            test_worker._current_task_id = None
            test_worker._cancelled_ids.discard(test_id)
            test_worker._state_changed.notify_all()


def test_worker_records_structured_detections_and_keeps_model_snapshot(image_test_client, monkeypatch):
    client, db, artifact, _, test_worker = image_test_client
    context = TrainingRun(
        id="run-context", task=artifact.task, attempt_number=2, launch_mode="resume", status="completed",
        model_path="base.pt", data_path="data.yaml", run_dir=str(Path(artifact.path).parent / "attempt-2"),
        parameter_snapshot={},
    )
    db.add(context)
    db.commit()
    test_id = create(client, artifact, ["street.png"], run_id=context.id).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    install_fake_process(monkeypatch)
    test_worker.process_one()

    payload = client.get(f"/api/model-tests/{test_id}").json()
    assert payload["status"] == "completed"
    assert payload["run_id"] == context.id
    assert payload["source_run_id"] == artifact.run_id
    assert payload["attempt_number"] == 2
    assert payload["source_attempt_number"] == 1
    assert payload["files"][0]["objects"][0]["class_name"] == "person"

    # History remains readable from snapshots if the live artifact link disappears.
    record = db.get(ModelTest, test_id)
    record.artifact_id = None
    db.commit()
    payload = client.get(f"/api/model-tests/{test_id}").json()
    assert payload["artifact_name"] == "weights/best.pt"
    assert payload["model"]["attempt_number"] == 1


def test_recovery_requeues_interrupted_test_with_valid_weight_pin(image_test_client):
    client, db, artifact, factory, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").json()["status"] == "queued"
    record = db.get(ModelTest, test_id)
    record.status = "running"
    record.started_at = utcnow()
    db.commit()
    work = storage.test_work_root(test_id)
    assert (work / "model.pt").read_bytes() == b"test-checkpoint"

    inference.InferenceWorker(session_factory=factory)._recover()

    db.expire_all()
    recovered = db.get(ModelTest, test_id)
    assert recovered.status == "queued"
    assert recovered.started_at is None
    assert (work / "model.pt").read_bytes() == b"test-checkpoint"
    assert storage.test_history_root(artifact.task_id, test_id).is_dir()


def test_recovery_fails_corrupt_weight_pin_without_live_artifact_fallback(image_test_client):
    client, db, artifact, factory, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").json()["status"] == "queued"
    record = db.get(ModelTest, test_id)
    record.status = "running"
    record.started_at = utcnow()
    db.commit()
    work = storage.test_work_root(test_id)
    (work / "model.pt").write_bytes(b"corrupt-pin")
    Path(artifact.path).write_bytes(b"replacement-that-must-not-be-used")

    inference.InferenceWorker(session_factory=factory)._recover()

    db.expire_all()
    recovered = db.get(ModelTest, test_id)
    assert recovered.status == "failed"
    assert recovered.error == "固定的模型权重副本校验失败"
    assert recovered.finished_at is not None
    assert not work.exists()
    assert storage.test_history_root(artifact.task_id, test_id).is_dir()


def test_expired_upload_cleanup_serializes_with_upload(image_test_client, monkeypatch):
    client, db, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    record = db.get(ModelTest, test_id)
    record.created_at = utcnow() - timedelta(seconds=inference.UPLOAD_TTL + 1)
    db.commit()
    directory = storage.test_history_root(artifact.task_id, test_id)
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    upload_started = threading.Event()
    real_delete_history = inference.delete_test_history

    def blocking_delete_history(task_id: str, candidate: str) -> None:
        cleanup_entered.set()
        if not release_cleanup.wait(3):
            raise RuntimeError("test did not release expiry cleanup")
        real_delete_history(task_id, candidate)

    monkeypatch.setattr(inference, "delete_test_history", blocking_delete_history)
    expiry_errors: list[BaseException] = []
    upload_responses = []

    def run_expiry() -> None:
        try:
            test_worker._expire_uploads()
        except BaseException as exc:
            expiry_errors.append(exc)

    def run_upload() -> None:
        upload_started.set()
        upload_responses.append(client.put(
            f"/api/model-tests/{test_id}/images/0", content=image_bytes(),
        ))

    expiry_thread = threading.Thread(target=run_expiry, name="model-test-expiry", daemon=True)
    upload_thread = threading.Thread(target=run_upload, name="model-test-upload", daemon=True)
    expiry_thread.start()
    assert cleanup_entered.wait(2)
    upload_thread.start()
    try:
        assert upload_started.wait(1)
        upload_thread.join(0.1)
        assert upload_thread.is_alive()
        assert directory.is_dir()
    finally:
        release_cleanup.set()
        expiry_thread.join(3)
        upload_thread.join(3)

    assert not expiry_thread.is_alive()
    assert not upload_thread.is_alive()
    assert expiry_errors == []
    assert [response.status_code for response in upload_responses] == [404]
    with factory() as check_db:
        assert check_db.get(ModelTest, test_id) is None
    assert not directory.exists()


def test_cancel_task_tests_stops_active_records_but_keeps_history(image_test_client):
    client, db, artifact, _, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    client.post(f"/api/model-tests/{test_id}/start")
    directory = storage.test_history_root(artifact.task_id, test_id)

    test_worker.cancel_task_tests(artifact.task_id)

    db.expire_all()
    record = db.get(ModelTest, test_id)
    assert record.status == "cancelled"
    assert not record.delete_requested
    assert directory.is_dir()
    assert test_id not in test_worker._cancelled_ids


@pytest.mark.parametrize("status", ["queued", "running"])
@pytest.mark.parametrize("finish", ["rollback", "close"])
def test_staged_task_cancellation_has_no_side_effect_before_commit(
    image_test_client, status, finish,
):
    client, _, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    pin = storage.test_work_root(test_id) / "model.pt"

    class LiveProcess:
        killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = LiveProcess()
    if status == "running":
        with factory() as setup_db:
            setup_db.get(ModelTest, test_id).status = "running"
            setup_db.commit()
        with test_worker._state_changed:
            test_worker._current_id = test_id
            test_worker._current_task_id = artifact.task_id
            test_worker._process = process

    owner = factory()
    try:
        assert inference.stage_task_test_cancellation(artifact.task_id, owner)
        assert owner.get(ModelTest, test_id).status == "cancel_requested"
        getattr(owner, finish)()
    finally:
        owner.close()
        with test_worker._state_changed:
            test_worker._current_id = None
            test_worker._current_task_id = None
            test_worker._process = None
            test_worker._state_changed.notify_all()

    with factory() as check_db:
        assert check_db.get(ModelTest, test_id).status == status
    assert pin.is_file()
    assert not process.killed
    assert test_worker._cancelled_ids == set()


@pytest.mark.parametrize(("action", "expected_statuses"), [
    ("create", {404}),
    ("start", {404, 409}),
])
def test_task_delete_serializes_model_test_create_and_start(
    image_test_client, monkeypatch, action, expected_statuses,
):
    client, _, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    if action == "start":
        upload_all(client, test_id)
    task_id = artifact.task_id
    artifact_id = artifact.id
    cancellation_entered = threading.Event()
    release_cancellation = threading.Event()
    action_started = threading.Event()
    real_cancel = test_worker.cancel_task_tests

    def blocking_cancel(*args, **kwargs):
        cancellation_entered.set()
        if not release_cancellation.wait(3):
            raise RuntimeError("test did not release task cancellation")
        return real_cancel(*args, **kwargs)

    monkeypatch.setattr(test_worker, "cancel_task_tests", blocking_cancel)
    delete_errors: list[BaseException] = []
    action_errors: list[BaseException] = []

    def run_delete() -> None:
        try:
            with factory() as delete_db:
                main_module.delete_task(task_id, delete_db)
        except BaseException as exc:
            delete_errors.append(exc)

    def run_action() -> None:
        action_started.set()
        try:
            with factory() as action_db:
                if action == "create":
                    inference.create_test(inference.TestCreate(
                        artifact_id=artifact_id,
                        files=[inference.ImageSpec(name="later.png", size=len(image_bytes()))],
                    ), action_db)
                else:
                    inference.start_test(test_id, action_db)
        except BaseException as exc:
            action_errors.append(exc)

    delete_thread = threading.Thread(target=run_delete, daemon=True)
    action_thread = threading.Thread(target=run_action, daemon=True)
    delete_thread.start()
    assert cancellation_entered.wait(2)
    action_thread.start()
    try:
        assert action_started.wait(1)
        action_thread.join(0.1)
        assert action_thread.is_alive()
    finally:
        release_cancellation.set()
        delete_thread.join(3)
        action_thread.join(3)

    assert not delete_thread.is_alive()
    assert not action_thread.is_alive()
    assert delete_errors == []
    assert len(action_errors) == 1
    assert isinstance(action_errors[0], HTTPException)
    assert action_errors[0].status_code in expected_statuses
    with factory() as check_db:
        assert check_db.get(ModelTest, test_id).status == "cancelled"
        assert len(check_db.scalars(select(ModelTest)).all()) == 1


def test_permanent_delete_upload_timeout_keeps_durable_retryable_job(
    image_test_client, monkeypatch,
):
    client, _, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    task_id = artifact.task_id
    with test_worker._state_changed:
        test_worker.uploading.add((test_id, 0))
    monkeypatch.setattr(test_worker, "wait_for_uploads", lambda *_args, **_kwargs: False)

    try:
        with factory() as delete_db:
            response = main_module.permanently_delete_task(task_id, delete_db)
        assert response.status_code == 202
        deletion_worker = LocalDeletionWorker(session_factory=factory)
        assert deletion_worker.process_once()
        with factory() as check_db:
            task = check_db.get(inference.TrainingTask, task_id)
            assert task.deleted_at is not None
            assert task.deleted_reason == "local_delete_failed"
            operation = check_db.scalar(select(RemoteOperation))
            assert operation.status == "failed"
            assert "上传" in operation.error
            assert Path(artifact.path).is_file()
    finally:
        with test_worker._state_changed:
            test_worker.uploading.discard((test_id, 0))
            test_worker._state_changed.notify_all()


def test_soft_delete_retry_settles_reservation_after_upload_timeout(
    image_test_client, monkeypatch,
):
    client, _, artifact, factory, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    task_id = artifact.task_id
    marker = storage.reservation_path(task_id, test_id)
    real_wait_for_uploads = test_worker.wait_for_uploads
    with test_worker._state_changed:
        test_worker.uploading.add((test_id, 0))
    monkeypatch.setattr(test_worker, "wait_for_uploads", lambda *_args, **_kwargs: False)

    with factory() as delete_db, pytest.raises(HTTPException) as error:
        main_module.delete_task(task_id, delete_db)
    assert error.value.status_code == 409
    assert marker.is_file()

    with test_worker._state_changed:
        test_worker.uploading.discard((test_id, 0))
        test_worker._state_changed.notify_all()
    monkeypatch.setattr(test_worker, "wait_for_uploads", real_wait_for_uploads)
    with factory() as restore_db:
        main_module.restore_task(task_id, restore_db)
    with factory() as retry_db:
        main_module.delete_task(task_id, retry_db)

    assert not marker.exists()
    with factory() as check_db:
        assert check_db.get(ModelTest, test_id).status == "cancelled"


def test_permanent_delete_sweeps_stale_work_for_terminal_record(image_test_client):
    client, _, artifact, factory, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    with factory() as setup_db:
        record = setup_db.get(ModelTest, test_id)
        record.status = "cancelled"
        record.finished_at = utcnow()
        setup_db.commit()
    work = storage.test_work_root(test_id)
    work.mkdir(parents=True)
    (work / "model.pt").write_bytes(b"stale-pin")

    with factory() as delete_db:
        response = main_module.permanently_delete_task(artifact.task_id, delete_db)

    assert response.status_code == 202
    assert work.exists()
    assert LocalDeletionWorker(session_factory=factory).process_once()
    assert not work.exists()
    with factory() as check_db:
        operation = check_db.scalar(select(RemoteOperation).where(
            RemoteOperation.task_id == artifact.task_id,
        ))
        assert operation is not None
        assert operation.kind == "local_delete"
        assert operation.status == "succeeded"


@pytest.mark.parametrize("confidence", [-0.1, 0, 1.01, "nan"])
def test_rejects_invalid_confidence(image_test_client, confidence):
    client, _, artifact, _, _ = image_test_client
    assert create(client, artifact, confidence=confidence).status_code == 422


@pytest.mark.parametrize("iou", [-0.1, 0, 1.01, "nan"])
def test_rejects_invalid_iou(image_test_client, iou):
    client, _, artifact, _, _ = image_test_client
    assert create(client, artifact, iou=iou).status_code == 422


def test_upload_validation_and_no_path_injection(image_test_client):
    client, _, artifact, _, _ = image_test_client
    test_id = create(client, artifact, ["../../outside.png"]).json()["id"]
    base = f"/api/model-tests/{test_id}"
    assert client.put(base + "/images/0", content=b"x" * len(image_bytes())).status_code == 422
    assert client.post(base + "/start").status_code == 409
    assert client.put(base + "/images/0", content=image_bytes() + b"extra").status_code == 413
    assert client.put(base + "/images/0", content=image_bytes()[:-1]).status_code == 422
    assert client.put(base + "/images/0", content=image_bytes()).status_code == 204
    assert client.get(base + "/images/-1/original").status_code == 404
    assert client.get(base + "/images/0/model.pt").status_code == 404
    assert create(client, artifact, files=[]).status_code == 422
    assert create(client, artifact, names=["a.png"] * 11).status_code == 422
    assert create(client, artifact, files=[{"name": "a.png", "size": 20 * 1024**2}] * 6).status_code == 413


def test_rejects_active_unsupported_deleted_or_outside_weights(image_test_client, tmp_path):
    client, db, artifact, _, _ = image_test_client
    artifact.run.status = "running"
    db.commit()
    assert create(client, artifact).status_code == 409
    artifact.run.status = "completed"
    artifact.task.task_type = "segment"
    db.commit()
    segment_test = create(client, artifact)
    assert segment_test.status_code == 201
    assert segment_test.json()["task_type"] == "segment"
    artifact.task.task_type = "detect"
    artifact.task.deleted_at = utcnow()
    db.commit()
    assert create(client, artifact).status_code == 404
    artifact.task.deleted_at = None
    artifact.path = str(tmp_path / "outside.pt")
    db.commit()
    assert create(client, artifact).status_code == 403


@pytest.mark.parametrize("task_type", ["detect", "segment", "classify", "pose", "obb"])
def test_training_artifacts_run_through_generic_result_history(image_test_client, monkeypatch, task_type):
    client, _, artifact, _, test_worker = image_test_client
    artifact.task.task_type = task_type
    test = create(client, artifact).json()
    test_id = test["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    install_fake_process(monkeypatch)

    assert test_worker.process_one()

    detail = client.get(f"/api/model-tests/{test_id}").json()
    assert detail["task_type"] == task_type
    assert detail["model"]["task_type"] == task_type
    assert detail["files"][0]["prediction_count"] == 1
    prediction = detail["files"][0]["objects"][0]
    if task_type == "classify":
        assert "bbox_xyxy" not in prediction
    elif task_type == "segment":
        assert prediction["mask_polygon_xy"] == [[1, 2], [20, 2], [20, 25]]
    elif task_type == "pose":
        assert prediction["keypoints"][0]["confidence"] == 0.8
    elif task_type == "obb":
        assert len(prediction["obb_points"]) == 4
    download = client.get(f"/api/model-tests/{test_id}/download")
    archive_name = "detection-results.zip" if task_type == "detect" else "model-test-results.zip"
    results_name = "detections.json" if task_type == "detect" else "results.json"
    assert download.headers["content-disposition"].endswith(f'filename="{archive_name}"')
    with zipfile.ZipFile(BytesIO(download.content)) as archive:
        assert results_name in archive.namelist()
        exported = json.loads(archive.read(results_name))
        assert exported["task_type"] == task_type
        assert exported["files"][0]["objects"] == [prediction]


def test_training_model_summary_uses_task_specific_metric(image_test_client):
    client, db, artifact, _, _ = image_test_client
    artifact.task.task_type = "classify"
    db.add_all([
        MetricPoint(
            task_id=artifact.task_id,
            run_id=artifact.run_id,
            epoch=4,
            elapsed=10,
            values={"metrics/accuracy_top1": 0.82, "metrics/mAP50-95(B)": 0.99},
        ),
        MetricPoint(
            task_id=artifact.task_id,
            run_id=artifact.run_id,
            epoch=5,
            elapsed=12,
            values={"metrics/accuracy_top1": 0.78, "metrics/mAP50-95(B)": 1.0},
        ),
    ])
    db.commit()

    response = create(client, artifact)

    assert response.status_code == 201
    model = response.json()["model"]
    assert model["metric_name"] == "Top-1 accuracy"
    assert model["best_epoch"] == 4
    assert model["metric_value"] == 0.82


def test_legacy_detection_history_without_task_type_remains_downloadable(image_test_client, monkeypatch):
    client, db, artifact, _, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    install_fake_process(monkeypatch)
    assert test_worker.process_one()
    record = db.get(ModelTest, test_id)
    db.refresh(record)
    record.model_snapshot = {key: value for key, value in record.model_snapshot.items() if key != "task_type"}
    db.commit()

    detail = client.get(f"/api/model-tests/{test_id}").json()
    assert "task_type" not in detail["model"]
    assert detail["files"][0]["detections"] == detail["files"][0]["prediction_count"] == 1
    download = client.get(f"/api/model-tests/{test_id}/download")
    assert download.status_code == 200
    with zipfile.ZipFile(BytesIO(download.content)) as archive:
        assert "detections.json" in archive.namelist()


def test_start_rejects_weight_changed_after_record_creation(image_test_client):
    client, db, artifact, _, test_worker = image_test_client
    created = create(client, artifact).json()
    test_id = created["id"]
    upload_all(client, test_id)
    replacement = b"replacement-checkpoint"
    Path(artifact.path).write_bytes(replacement)

    response = client.post(f"/api/model-tests/{test_id}/start")

    assert response.status_code == 409
    assert response.json()["detail"] == "原模型权重版本已变化，无法按原权重重测"
    db.expire_all()
    record = db.get(ModelTest, test_id)
    assert record.status == "failed"
    assert record.error == response.json()["detail"]
    assert record.model_sha256 == created["model_sha256"]
    assert not storage.test_work_root(test_id).exists()
    assert not test_worker.process_one()


def test_worker_uses_pin_when_live_weight_changes_after_start(image_test_client, monkeypatch):
    client, _, artifact, _, test_worker = image_test_client
    original = Path(artifact.path).read_bytes()
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    Path(artifact.path).write_bytes(b"new-live-checkpoint")
    seen_models = []
    install_fake_process(monkeypatch, seen_models=seen_models)

    assert test_worker.process_one()

    assert seen_models == [original]
    assert client.get(f"/api/model-tests/{test_id}").json()["status"] == "completed"


def test_start_rejects_weight_mutated_during_pin_copy(image_test_client, monkeypatch):
    client, db, artifact, _, _ = image_test_client
    test_id = create(client, artifact).json()["id"]
    upload_all(client, test_id)
    real_copy_and_digest = inference._copy_and_digest

    def copy_then_mutate(source: Path, destination: Path):
        result = real_copy_and_digest(source, destination)
        source.write_bytes(b"mutated-during-copy")
        return result

    monkeypatch.setattr(inference, "_copy_and_digest", copy_then_mutate)

    response = client.post(f"/api/model-tests/{test_id}/start")

    assert response.status_code == 409
    assert response.json()["detail"] == "模型权重在复制期间发生变化，请刷新产物后重试"
    db.expire_all()
    assert db.get(ModelTest, test_id).status == "failed"
    assert not storage.test_work_root(test_id).exists()


def test_legacy_history_without_digest_cannot_claim_original_weight_rerun(image_test_client):
    client, db, artifact, _, _ = image_test_client
    source_id = create(client, artifact).json()["id"]
    source = db.get(ModelTest, source_id)
    source.status = "completed"
    source.model_sha256 = None
    source.finished_at = utcnow()
    db.commit()

    response = create(client, artifact, rerun_of_test_id=source_id)

    assert response.status_code == 409
    assert response.json()["detail"] == "该历史记录没有精确权重版本，请改用当前权重新建检测"


def test_remote_rerun_uses_exact_legacy_cache_after_artifact_refresh(
    image_test_client, tmp_path, monkeypatch,
):
    client, db, artifact, _, test_worker = image_test_client
    old_weight = b"old-remote-model"
    new_weight = b"new-remote-model-version"
    artifact.run.execution_node_id = "remote-node"
    artifact.task.execution_node_id = "remote-node"
    artifact.size = len(old_weight)
    old_operation = RemoteOperation(
        id="a" * 32,
        node_id="remote-node",
        task_id=artifact.task_id,
        kind="fetch",
        status="succeeded",
        created_at=utcnow() - timedelta(minutes=1),
        payload={
            "artifact_id": artifact.id,
            "run_id": artifact.run_id,
            "name": artifact.name,
            "size": len(old_weight),
            "modified_at": artifact.modified_at.isoformat(),
        },
    )
    old_cache = tmp_path / "old-remote.pt"
    old_cache.write_bytes(old_weight)
    cache_paths = {old_operation.id: old_cache}
    monkeypatch.setattr(inference, "operation_file", lambda operation: cache_paths[operation.id])
    db.add(old_operation)
    db.commit()

    source_payload = create(client, artifact).json()
    source = db.get(ModelTest, source_payload["id"])
    source.status = "completed"
    source.finished_at = utcnow()
    # Simulate a cache record written before fetch digests were persisted.
    old_operation.payload = {
        key: value for key, value in old_operation.payload.items() if key != "sha256"
    }
    artifact.size = len(new_weight)
    artifact.modified_at = utcnow() + timedelta(seconds=1)
    new_operation = RemoteOperation(
        id="b" * 32,
        node_id="remote-node",
        task_id=artifact.task_id,
        kind="fetch",
        status="succeeded",
        payload={
            "artifact_id": artifact.id,
            "run_id": artifact.run_id,
            "name": artifact.name,
            "size": len(new_weight),
            "modified_at": artifact.modified_at.isoformat(),
            "sha256": hashlib.sha256(new_weight).hexdigest(),
        },
    )
    new_cache = tmp_path / "new-remote.pt"
    new_cache.write_bytes(new_weight)
    cache_paths[new_operation.id] = new_cache
    db.add(new_operation)
    db.commit()

    rerun_response = create(
        client, artifact, rerun_of_test_id=source.id, confidence=0.55, iou=0.6,
    )

    assert rerun_response.status_code == 201
    rerun = rerun_response.json()
    assert rerun["rerun_of_test_id"] == source.id
    assert rerun["model_sha256"] == hashlib.sha256(old_weight).hexdigest()
    assert rerun["model_size"] == len(old_weight)
    assert rerun["artifact_name"] == source.artifact_name
    assert rerun["source_attempt_number"] == source.source_attempt_number
    db.refresh(old_operation)
    assert old_operation.payload["sha256"] == rerun["model_sha256"]

    upload_all(client, rerun["id"])
    assert client.post(f"/api/model-tests/{rerun['id']}/start").status_code == 202
    seen_models = []
    install_fake_process(monkeypatch, seen_models=seen_models)
    assert test_worker.process_one()
    assert seen_models == [old_weight]


def test_remote_history_cache_falls_back_after_newest_candidate_is_corrupt(
    image_test_client, tmp_path, monkeypatch,
):
    _, db, artifact, _, _ = image_test_client
    expected_weight = b"stable-remote-model"
    expected_sha256 = hashlib.sha256(expected_weight).hexdigest()
    artifact.run.execution_node_id = "remote-node"
    artifact.task.execution_node_id = "remote-node"
    artifact.size = len(expected_weight)
    common_payload = {
        "artifact_id": artifact.id,
        "run_id": artifact.run_id,
        "name": artifact.name,
        "size": len(expected_weight),
        "modified_at": artifact.modified_at.isoformat(),
        "sha256": expected_sha256,
    }
    valid = RemoteOperation(
        id="c" * 32, node_id="remote-node", task_id=artifact.task_id,
        kind="fetch", status="succeeded", payload=common_payload,
        created_at=utcnow() - timedelta(minutes=2),
    )
    corrupt = RemoteOperation(
        id="d" * 32, node_id="remote-node", task_id=artifact.task_id,
        kind="fetch", status="succeeded", payload=common_payload,
        created_at=utcnow() - timedelta(minutes=1),
    )
    valid_cache = tmp_path / "valid-remote.pt"
    corrupt_cache = tmp_path / "corrupt-remote.pt"
    valid_cache.write_bytes(expected_weight)
    corrupt_cache.write_bytes(b"x" * len(expected_weight))
    cache_paths = {valid.id: valid_cache, corrupt.id: corrupt_cache}
    monkeypatch.setattr(inference, "operation_file", lambda operation: cache_paths[operation.id])
    db.add_all([valid, corrupt])
    db.commit()

    selected = inference._remote_weight_for_version(
        db, artifact, artifact.run, expected_sha256, len(expected_weight),
    )

    assert selected == valid_cache


@pytest.mark.parametrize("task_type", ["detect", "segment", "classify", "pose", "obb"])
def test_remote_weight_uses_verified_center_cache(image_test_client, tmp_path, monkeypatch, task_type):
    client, db, artifact, _, test_worker = image_test_client
    remote_weight = b"verified-remote-weight"
    artifact.run.execution_node_id = "remote-node"
    artifact.task.execution_node_id = "remote-node"
    artifact.task.task_type = task_type
    artifact.size = len(remote_weight)
    db.commit()
    assert create(client, artifact).status_code == 409
    operation = RemoteOperation(
        id="a" * 32, node_id="remote-node", task_id=artifact.task_id, kind="fetch",
        status="succeeded", payload={
            "artifact_id": artifact.id, "run_id": artifact.run_id,
            "name": artifact.name, "size": artifact.size,
            "modified_at": artifact.modified_at.isoformat(),
        },
    )
    db.add(operation)
    db.commit()
    cached = tmp_path / "cache.bin"
    monkeypatch.setattr(inference, "operation_file", lambda _: cached)
    assert create(client, artifact).status_code == 404
    cached.write_bytes(remote_weight)
    created = create(client, artifact)
    assert created.status_code == 201
    payload = created.json()
    expected_sha256 = hashlib.sha256(remote_weight).hexdigest()
    assert payload["model_sha256"] == expected_sha256
    assert payload["model_size"] == len(remote_weight)
    db.refresh(operation)
    assert operation.payload["sha256"] == expected_sha256
    test_id = payload["id"]
    upload_all(client, test_id)
    assert client.post(f"/api/model-tests/{test_id}/start").status_code == 202
    seen_models = []
    install_fake_process(monkeypatch, seen_models=seen_models)
    test_worker.process_one()
    assert seen_models == [remote_weight]
    assert client.get(f"/api/model-tests/{test_id}").json()["status"] == "completed"


def test_task_history_cleanup_rejects_task_symlink(tmp_path, monkeypatch):
    history = tmp_path / "history"
    history.mkdir()
    monkeypatch.setattr(storage, "HISTORY_ROOT", history)
    owned = history / "task-b"
    owned.mkdir()
    marker = owned / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    (history / "task-a").symlink_to(owned, target_is_directory=True)

    with pytest.raises(ValueError, match="符号链接"):
        storage.delete_task_history("task-a")

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(("quota_name", "message"), [
    ("MODEL_TEST_HISTORY_QUOTA_BYTES", "历史总配额不足"),
    ("MODEL_TEST_TASK_QUOTA_BYTES", "当前训练任务的模型测试历史配额不足"),
])
def test_create_rejects_model_test_history_quota(
    image_test_client, monkeypatch, quota_name, message,
):
    client, db, artifact, _, _ = image_test_client
    payload_size = len(image_bytes())
    estimate = storage.estimate_test_storage(
        model_bytes=Path(artifact.path).stat().st_size,
        declared_upload_bytes=payload_size,
        image_count=1,
    )
    monkeypatch.setattr(storage, quota_name, estimate.history_bytes - 1)

    response = create(client, artifact)

    assert response.status_code == 507
    assert message in response.json()["detail"]
    assert db.scalar(select(ModelTest.id)) is None


def test_create_rejects_low_model_test_disk_space(image_test_client, monkeypatch):
    client, db, artifact, _, _ = image_test_client
    payload_size = len(image_bytes())
    estimate = storage.estimate_test_storage(
        model_bytes=Path(artifact.path).stat().st_size,
        declared_upload_bytes=payload_size,
        image_count=1,
    )
    monkeypatch.setattr(storage, "MODEL_TEST_MIN_FREE_BYTES", 100)
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=estimate.peak_bytes + 99),
    )

    response = create(client, artifact)

    assert response.status_code == 507
    assert "服务器模型测试存储空间不足" in response.json()["detail"]
    assert db.scalar(select(ModelTest.id)) is None


def test_pending_reservation_is_counted_until_test_is_abandoned(image_test_client, monkeypatch):
    client, _, artifact, _, _ = image_test_client
    payload_size = len(image_bytes())
    estimate = storage.estimate_test_storage(
        model_bytes=Path(artifact.path).stat().st_size,
        declared_upload_bytes=payload_size,
        image_count=1,
    )
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=estimate.peak_bytes + estimate.peak_bytes // 2),
    )
    first = create(client, artifact)
    assert first.status_code == 201
    first_id = first.json()["id"]
    marker = storage.reservation_path(artifact.task_id, first_id)
    assert marker.is_file()

    refused = create(client, artifact)
    assert refused.status_code == 507
    assert "现有测试尚需" in refused.json()["detail"]

    assert client.post(f"/api/model-tests/{first_id}/abandon").status_code == 200
    assert not marker.exists()
    assert create(client, artifact).status_code == 201


def test_completed_and_failed_tests_release_storage_reservations(image_test_client, monkeypatch):
    client, db, artifact, _, test_worker = image_test_client
    completed_id = create(client, artifact).json()["id"]
    completed_marker = storage.reservation_path(artifact.task_id, completed_id)
    assert completed_marker.is_file()
    upload_all(client, completed_id)
    assert client.post(f"/api/model-tests/{completed_id}/start").status_code == 202
    install_fake_process(monkeypatch)
    assert test_worker.process_one()
    assert not completed_marker.exists()

    failed_id = create(client, artifact).json()["id"]
    failed_marker = storage.reservation_path(artifact.task_id, failed_id)
    upload_all(client, failed_id)

    def fail_pin(*_args, **_kwargs):
        raise inference.HTTPException(507, "模拟权重固定失败")

    monkeypatch.setattr(inference, "_pin_weight", fail_pin)
    response = client.post(f"/api/model-tests/{failed_id}/start")
    assert response.status_code == 507
    assert not failed_marker.exists()
    db.expire_all()
    for test_id in (completed_id, failed_id):
        record = db.get(ModelTest, test_id)
        assert record.status in {"completed", "failed"}
        assert record.storage_bytes == storage.directory_size(
            storage.test_history_root(artifact.task_id, test_id),
        )


def test_task_cancel_releases_reservation_only_after_owner_commit(image_test_client):
    client, db, artifact, _, test_worker = image_test_client
    test_id = create(client, artifact).json()["id"]
    marker = storage.reservation_path(artifact.task_id, test_id)

    inference.stage_task_test_cancellation(artifact.task_id, db)
    assert marker.is_file()
    db.rollback()
    assert marker.is_file()
    db.expire_all()
    assert db.get(ModelTest, test_id).status == "uploading"

    inference.stage_task_test_cancellation(artifact.task_id, db)
    assert marker.is_file()
    db.commit()
    assert marker.is_file()
    test_worker.cancel_task_tests(artifact.task_id)
    assert not marker.exists()
    db.expire_all()
    record = db.get(ModelTest, test_id)
    assert record.status == "cancelled"
    assert record.storage_bytes == storage.directory_size(
        storage.test_history_root(artifact.task_id, test_id),
    )


def test_recovery_keeps_active_reservation_and_releases_stale_markers(image_test_client):
    client, db, artifact, factory, _ = image_test_client
    active_id = create(client, artifact).json()["id"]
    terminal_id = create(client, artifact).json()["id"]
    terminal = db.get(ModelTest, terminal_id)
    terminal.status = "failed"
    terminal.finished_at = utcnow()
    db.commit()
    orphan_id = "f" * 32
    estimate = storage.ModelTestStorageEstimate(100, 100, 0, 0, 0, 0)
    storage.prepare_test_history(artifact.task_id, orphan_id, estimate)

    inference.InferenceWorker(session_factory=factory)._recover()

    assert storage.reservation_path(artifact.task_id, active_id).is_file()
    assert not storage.reservation_path(artifact.task_id, terminal_id).exists()
    assert not storage.reservation_path(artifact.task_id, orphan_id).exists()
