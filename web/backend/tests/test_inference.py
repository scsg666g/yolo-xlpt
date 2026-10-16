from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.inference as inference
from app.database import get_db
from app.main import app
from app.models import Artifact, RemoteOperation, TrainingRun, utcnow
from test_services import make_session, make_task


@pytest.fixture
def image_test_client(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "TEST_ROOT", tmp_path / "tests")
    monkeypatch.setattr(inference, "worker", inference.InferenceWorker())
    with make_session(tmp_path) as db:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        model = run_dir / "best.pt"
        model.write_bytes(b"test-checkpoint")
        task = make_task(status="completed")
        run = TrainingRun(id="run-test", task=task, attempt_number=1, launch_mode="fresh", status="completed",
                          model_path="base.pt", data_path="data.yaml", run_dir=str(run_dir), parameter_snapshot={})
        artifact = Artifact(task=task, run=run, kind="weight", name="weights/best.pt", path=str(model),
                            size=model.stat().st_size, modified_at=utcnow())
        db.add_all([task, run, artifact])
        db.commit()

        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        try:
            yield TestClient(app), db, artifact
        finally:
            app.dependency_overrides.pop(get_db, None)


def image_bytes():
    stream = BytesIO()
    Image.new("RGB", (40, 30), "white").save(stream, "PNG")
    return stream.getvalue()


def create(client, artifact, names=None, **overrides):
    return client.post("/api/model-tests", json={
        "artifact_id": artifact.id, "confidence": 0.4,
        "files": [{"name": name, "size": len(image_bytes())} for name in (names or ["input.png"])],
        **overrides,
    })


def test_batch_upload_isolated_from_training_and_result_download(image_test_client):
    client, _, artifact = image_test_client
    response = create(client, artifact, ["重复.png", "重复.png"])
    assert response.status_code == 201
    assert response.json()["iou"] == 0.7
    assert response.json()["model"]["weight_name"] == "best.pt"
    assert response.json()["model"]["attempt_number"] == 1
    test_id = response.json()["id"]
    base = f"/api/model-tests/{test_id}"
    assert client.post(base + "/start").status_code == 409
    for index in range(2):
        assert client.put(base + f"/images/{index}", content=image_bytes()).status_code == 204
    original = client.get(base + "/images/0/original")
    assert original.status_code == 200
    assert Image.open(BytesIO(original.content)).size == (40, 30)
    assert client.get(base + "/images/0/result").status_code == 409
    assert client.post(base + "/start").json()["status"] == "queued"
    job = inference.worker.jobs[test_id]
    assert job.confidence == 0.4
    assert (job.directory / "model.pt").read_bytes() == b"test-checkpoint"
    assert client.put(base + "/images/0", content=image_bytes()).status_code == 409
    (job.directory / "results").mkdir()
    Image.new("RGB", (40, 30), "red").save(job.directory / "results/0.jpg")
    job.status = "completed"
    job.files[0]["detections"] = 0
    (job.directory / "detection-results.zip").write_bytes(b"zip")
    result = client.get(base + "/images/0/result?download=true")
    assert result.status_code == 200
    assert 'attachment; filename="detection-1.jpg"' == result.headers["content-disposition"]
    assert client.get(base).json()["files"][0]["detections"] == 0
    download = client.get(base + "/download")
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"
    assert client.delete(base).status_code == 204
    assert client.get(base).status_code == 404
    assert not job.directory.exists()
    assert artifact.path.endswith("best.pt")


def test_can_cancel_queued_test_without_touching_checkpoint(image_test_client):
    client, _, artifact = image_test_client
    test_id = create(client, artifact).json()["id"]
    base = f"/api/model-tests/{test_id}"
    assert client.put(base + "/images/0", content=image_bytes()).status_code == 204
    assert client.post(base + "/start").status_code == 202
    job = inference.worker.jobs[test_id]
    assert client.delete(base).status_code == 204
    assert not job.directory.exists()
    assert Path(artifact.path).read_bytes() == b"test-checkpoint"


def test_worker_records_structured_detections_timings_and_batch_export(image_test_client, monkeypatch):
    client, _, artifact = image_test_client
    test_id = create(client, artifact, ["street.png"]).json()["id"]
    base = f"/api/model-tests/{test_id}"
    assert client.put(base + "/images/0", content=image_bytes()).status_code == 204
    assert client.post(base + "/start").status_code == 202
    job = inference.worker.jobs[test_id]

    class FakeProcess:
        def __init__(self, command, **_kwargs):
            output = Path(command[command.index("--output") + 1])
            summary = Path(command[command.index("--summary") + 1])
            output.mkdir()
            Image.new("RGB", (40, 30), "red").save(output / "0.jpg")
            summary.write_text(json.dumps({
                "device": "CPU", "imgsz": 640,
                "timings": {"model_load": 120, "preprocess": 3, "inference": 25, "postprocess": 2, "total": 160},
                "images": [{"index": 0, "objects": [{"class_id": 0, "class_name": "person",
                            "confidence": 0.91, "bbox_xyxy": [1, 2, 20, 25]}],
                            "timings": {"preprocess": 3, "inference": 25, "postprocess": 2}}],
            }), encoding="utf-8")

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(inference.subprocess, "Popen", FakeProcess)
    inference.worker.execute(job)

    payload = client.get(base).json()
    assert payload["status"] == "completed"
    assert payload["files"][0]["detections"] == 1
    assert payload["files"][0]["objects"][0]["class_name"] == "person"
    assert payload["timings"]["inference"] == 25
    archive_path = job.directory / "detection-results.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"detections.json", "results/1-street.png.jpg"}
        exported = json.loads(archive.read("detections.json"))
        assert exported["iou"] == 0.7
        assert exported["files"][0]["objects"][0]["confidence"] == 0.91


@pytest.mark.parametrize("confidence", [-0.1, 0, 1.01, "nan"])
def test_rejects_invalid_confidence(image_test_client, confidence):
    client, _, artifact = image_test_client
    assert create(client, artifact, confidence=confidence).status_code == 422


@pytest.mark.parametrize("iou", [-0.1, 0, 1.01, "nan"])
def test_rejects_invalid_iou(image_test_client, iou):
    client, _, artifact = image_test_client
    assert create(client, artifact, iou=iou).status_code == 422


def test_upload_validation_and_no_path_injection(image_test_client):
    client, _, artifact = image_test_client
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
    client, db, artifact = image_test_client
    artifact.run.status = "running"
    db.commit()
    assert create(client, artifact).status_code == 409
    artifact.run.status = "completed"
    artifact.task.task_type = "segment"
    db.commit()
    assert create(client, artifact).status_code == 422
    artifact.task.task_type = "detect"
    artifact.task.deleted_at = utcnow()
    db.commit()
    assert create(client, artifact).status_code == 404
    artifact.task.deleted_at = None
    artifact.path = str(tmp_path / "outside.pt")
    db.commit()
    assert create(client, artifact).status_code == 403


def test_remote_weight_requires_verified_cache(image_test_client, tmp_path, monkeypatch):
    client, db, artifact = image_test_client
    artifact.run.execution_node_id = "remote-node"
    artifact.task.execution_node_id = "remote-node"
    db.commit()
    # Even if a same-named path exists on the center, remote weights require a completed fetch.
    assert create(client, artifact).status_code == 409
    operation = RemoteOperation(id="a" * 32, node_id="remote-node", task_id=artifact.task_id, kind="fetch",
                                status="succeeded", payload={"artifact_id": artifact.id, "run_id": artifact.run_id,
                                "name": artifact.name, "size": artifact.size,
                                "modified_at": artifact.modified_at.isoformat()})
    db.add(operation)
    db.commit()
    cached = tmp_path / "cache.bin"
    monkeypatch.setattr(inference, "operation_file", lambda _: cached)
    assert create(client, artifact).status_code == 404
    cached.write_bytes(b"verified-remote-weight")
    test_id = create(client, artifact).json()["id"]
    base = f"/api/model-tests/{test_id}"
    assert client.put(base + "/images/0", content=image_bytes()).status_code == 204
    assert client.post(base + "/start").status_code == 202
    assert (inference.worker.jobs[test_id].directory / "model.pt").read_bytes() == b"verified-remote-weight"
