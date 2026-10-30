"""Exercise directory publication with real image bytes and isolated filesystem roots."""
import io
import fcntl
import json
import os
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.dataset_uploads as uploads
import app.services as services
from app.database import Base, get_db
from app.models import TrainingRun, TrainingTask, utcnow


@pytest.fixture
def environment(tmp_path, monkeypatch):
    dataset_root = tmp_path / "数据集"
    monkeypatch.setattr(uploads, "UPLOAD_ROOT", tmp_path / "staging")
    monkeypatch.setattr(uploads, "DATASET_ROOTS", [dataset_root])
    monkeypatch.setattr(services, "DATASET_ROOTS", [dataset_root])
    engine = create_engine(f"sqlite:///{tmp_path / 'datasets.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db():
        with local_session() as db:
            yield db

    isolated = FastAPI()
    isolated.include_router(uploads.router)
    isolated.include_router(uploads.management_router)
    isolated.dependency_overrides[get_db] = override_db
    isolated.state.session_factory = local_session
    with TestClient(isolated) as client:
        yield client, dataset_root
    engine.dispose()


def dataset_files():
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "red").save(output, format="PNG")
    return {
        "images/train/样本.png": output.getvalue(),
        "images/val/样本.png": output.getvalue(),
        "labels/train/样本.txt": b"0 0.5 0.5 0.2 0.3\n",
        "labels/val/样本.txt": b"1 0.5 0.5 0.2 0.3\n",
    }


def create(client, files, name="中文 训练集"):
    return client.post("/api/datasets/uploads", json={
        "name": name,
        "files": [{"path": path, "size": len(content)} for path, content in files.items()],
    })


def stage(client, files, name="中文 训练集"):
    response = create(client, files, name)
    assert response.status_code == 201, response.text
    upload_id = response.json()["upload_id"]
    for index, content in enumerate(files.values()):
        result = client.put(f"/api/datasets/uploads/{upload_id}/files/{index}", content=content)
        assert result.status_code == 204, result.text
    return upload_id


def complete(client, upload_id, names=None):
    return client.post(f"/api/datasets/uploads/{upload_id}/complete", json={"names": names or []})


def test_chinese_name_absolute_path_and_success_receipt(environment):
    client, root = environment
    files = dataset_files()
    upload_id = stage(client, files)
    response = complete(client, upload_id, ["人员", "安全帽"])
    assert response.status_code == 200, response.text
    result = response.json()
    destination = root / "中文 训练集"
    assert result["name"] == destination.name
    assert result["directory"] == str(destination.resolve())
    config = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
    assert config == {"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
                      "names": {0: "人员", 1: "安全帽"}}
    assert services.validate_dataset_config(result["resource"]["path"]) == destination / "data.yaml"
    assert services.discover_files([root], {".yaml"}) == [result["resource"]]
    assert result["stats"] == {"train_images": 1, "val_images": 1, "labels": 2, "classes": 2, "background_images": 0}
    assert result["warnings"] == []
    for path, content in files.items():
        assert (destination / path).read_bytes() == content
    # The server acknowledges retries even when the final HTTP response was lost.
    assert complete(client, upload_id, ["different"]).json() == result
    status = client.get(f"/api/datasets/uploads/{upload_id}").json()
    assert status["result"] == result
    assert status["uploaded_indices"] == list(range(len(files)))
    assert not (uploads.UPLOAD_ROOT / upload_id / "files").exists()
    assert client.put(f"/api/datasets/uploads/{upload_id}/files/0", content=b"changed").status_code == 409
    assert client.delete(f"/api/datasets/uploads/{upload_id}").status_code == 204
    assert (destination / "data.yaml").exists()


@pytest.mark.parametrize("metadata", [
    {"data.yaml": "path: /another/server\ntrain: incorrect\nval: incorrect\nnames:\n  0: 人员\n  1: 安全帽\n"},
    {"data.yml": "names: [人员, 安全帽]\n"},
    {"classes.txt": "\ufeff人员\n安全帽\n"},
])
def test_classes_from_metadata_and_server_path_overrides(environment, metadata):
    client, root = environment
    files = dataset_files() | {path: text.encode("utf-8") for path, text in metadata.items()}
    upload_id = stage(client, files)
    result = complete(client, upload_id)
    assert result.status_code == 200, result.text
    assert result.json()["names"] == ["人员", "安全帽"]
    config = yaml.safe_load((root / "中文 训练集" / "data.yaml").read_text(encoding="utf-8"))
    assert config["path"] == str((root / "中文 训练集").resolve())
    assert config["train"] == "images/train"


@pytest.mark.parametrize("name", ["../outside", "folder/nested", "folder\\nested", ".hidden", " x ", "x\n", "中" * 81])
def test_invalid_folder_names(environment, name):
    client, root = environment
    assert create(client, dataset_files(), name).status_code == 422
    assert not root.exists()


@pytest.mark.parametrize("invalid", [
    "../outside.txt", "/images/train/a.png", "images/train/../../a.png", "images//train/a.png",
    "images/train/.hidden.png", "images/train\\a.png", "other/a.txt", "images/test/a.png", "labels/train/a.json",
    "images/train/样本.png/nested.png", "labels/train/样本.txt/nested.txt", "images/train/样本.jpg",
    "labels/train/another.TXT",
])
def test_invalid_paths_conflicts_and_ambiguous_image_names(environment, invalid):
    client, _ = environment
    files = dataset_files() | {invalid: b"data"}
    assert create(client, files).status_code == 422


def test_same_name_rejected_before_upload_and_before_publication(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    existing = root / "中文 训练集"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("unchanged")
    assert create(client, dataset_files()).status_code == 409
    assert complete(client, upload_id, ["person", "hat"]).status_code == 409
    assert list(existing.iterdir()) == [marker]
    assert marker.read_text() == "unchanged"


def test_incomplete_upload_size_mismatch_and_resume(environment):
    client, root = environment
    files = dataset_files()
    upload_id = create(client, files).json()["upload_id"]
    endpoint = f"/api/datasets/uploads/{upload_id}"
    image = next(iter(files.values()))
    assert client.put(f"{endpoint}/files/0", content=image[:-1]).status_code == 422
    assert client.put(f"{endpoint}/files/0", content=image + b"x").status_code == 413
    assert client.get(endpoint).json()["uploaded_indices"] == []
    assert client.put(f"{endpoint}/files/0", content=image).status_code == 204
    assert client.get(endpoint).json()["uploaded_indices"] == [0]
    assert complete(client, upload_id, ["person", "hat"]).status_code == 409
    assert not root.exists()
    for index, content in list(enumerate(files.values()))[1:]:
        assert client.put(f"{endpoint}/files/{index}", content=content).status_code == 204
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


@pytest.mark.parametrize("label", [
    "2 0.5 0.5 0.2 0.3", "-1 0.5 0.5 0.2 0.3", "0.5 0.5 0.5 0.2 0.3", "0 1.1 0.5 0.2 0.3",
    "0 0.5 0.5 0 0.3", "0 nan 0.5 0.2 0.3", "0 inf 0.5 0.2 0.3", "0 0.5 0.5", "text", "0 0.5 0.5 0.2 0.3 0.4",
])
def test_invalid_detection_labels_not_published(environment, label):
    client, root = environment
    files = dataset_files() | {"labels/train/样本.txt": label.encode()}
    upload_id = stage(client, files)
    result = complete(client, upload_id, ["person", "hat"])
    assert result.status_code == 422, result.text
    assert "标签格式错误" in result.json()["detail"]
    assert not (root / "中文 训练集").exists()


def test_corrupted_image_not_published(environment):
    client, root = environment
    files = dataset_files() | {"images/train/样本.png": b"not an image"}
    upload_id = stage(client, files)
    result = complete(client, upload_id, ["person", "hat"])
    assert result.status_code == 422
    assert "图片损坏" in result.json()["detail"]
    assert not (root / "中文 训练集").exists()


@pytest.mark.parametrize("config", ["names: {1: person, 2: hat}", "names: {0.5: person, 1: hat}", "names: {false: person, 1: hat}", "names: [person, person]", "names: [person, 3]", "names: [person, '']", "names: [broken"])
def test_invalid_class_metadata(environment, config):
    client, root = environment
    files = dataset_files() | {"data.yaml": config.encode()}
    upload_id = stage(client, files)
    assert complete(client, upload_id).status_code == 422
    assert not (root / "中文 训练集").exists()


def test_class_names_required_and_can_correct_after_validation_failure(environment):
    client, _ = environment
    upload_id = stage(client, dataset_files())
    assert complete(client, upload_id).status_code == 422
    assert complete(client, upload_id, ["person"]).status_code == 422
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


def test_background_and_orphan_labels_counted_correctly(environment):
    client, root = environment
    files = dataset_files()
    del files["labels/train/样本.txt"]
    files["labels/val/样本.txt"] = b""
    files["labels/val/orphan.txt"] = b""
    upload_id = stage(client, files)
    response = complete(client, upload_id, ["person"])
    assert response.status_code == 200, response.text
    assert response.json()["stats"]["background_images"] == 2
    assert len(response.json()["warnings"]) == 3
    assert (root / "中文 训练集" / "labels" / "train").is_dir()


def test_cancel_only_deletes_staging_and_expired_sessions_are_cleaned(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    directory = uploads.UPLOAD_ROOT / upload_id
    assert client.delete(f"/api/datasets/uploads/{upload_id}").status_code == 204
    assert not directory.exists()
    assert client.get(f"/api/datasets/uploads/{upload_id}").status_code == 404
    assert not root.exists()
    expired_id = create(client, dataset_files()).json()["upload_id"]
    expired_manifest = uploads.UPLOAD_ROOT / expired_id / "manifest.json"
    expiry = time.time() - uploads.SESSION_TTL - 10
    os.utime(expired_manifest.parent / ".lock", (expiry, expiry))
    assert create(client, dataset_files(), "another").status_code == 201
    assert not expired_manifest.exists()
    assert client.get("/api/datasets/uploads/..%2Foutside").status_code == 404


def test_publication_io_failure_removes_new_target_and_allows_retry(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    destination = root / "中文 训练集"
    publication = root / f"{uploads.MANAGED_PUBLICATION_PREFIX}{upload_id}"
    original = uploads.shutil.copytree
    with monkeypatch.context() as patch:
        def fail_copy(*args, **kwargs):
            assert not destination.exists()
            assert services.discover_files([root], {".yaml"}) == []
            raise OSError("disk full")
        patch.setattr(uploads.shutil, "copytree", fail_copy)
        assert complete(client, upload_id, ["person", "hat"]).status_code == 507
    assert uploads.shutil.copytree is original
    assert not destination.exists()
    assert not publication.exists()
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


def test_receipt_write_failure_does_not_leave_a_cached_success(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    destination = root / "中文 训练集"
    publication = root / f"{uploads.MANAGED_PUBLICATION_PREFIX}{upload_id}"
    original = Path.write_text
    with monkeypatch.context() as patch:
        def fail_receipt(path, *args, **kwargs):
            if path.name == ".manifest.tmp":
                assert not destination.exists()
                assert services.discover_files([root], {".yaml"}) == []
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        patch.setattr(Path, "write_text", fail_receipt)
        assert complete(client, upload_id, ["person", "hat"]).status_code == 507
    assert not destination.exists()
    assert not publication.exists()
    assert "result" not in client.get(f"/api/datasets/uploads/{upload_id}").json()
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


@pytest.mark.parametrize("retry_names", [[], ["已改名类别一", "已改名类别二"]], ids=["without-names", "renamed-names"])
def test_retry_recovers_published_dataset_and_cleans_staging_outside_mutation_lock(
        environment, monkeypatch, retry_names):
    client, root = environment
    upload_id = stage(client, dataset_files())
    directory = uploads.UPLOAD_ROOT / upload_id
    destination = root / "中文 训练集"
    original_replace = Path.replace

    with monkeypatch.context() as patch:
        def interrupt_receipt_replace(path, target):
            if path.name == ".manifest.tmp":
                raise OSError("simulated process interruption after publication")
            return original_replace(path, target)

        patch.setattr(Path, "replace", interrupt_receipt_replace)
        response = complete(client, upload_id, ["person", "hat"])
    assert response.status_code == 507
    assert destination.is_dir()
    assert (destination / uploads.MANAGED_MARKER).is_file()
    assert "result" not in client.get(f"/api/datasets/uploads/{upload_id}").json()
    assert (directory / "files").is_dir()
    marker_before = (destination / uploads.MANAGED_MARKER).read_bytes()
    config_before = (destination / "data.yaml").read_bytes()
    managed_before = client.get("/api/datasets/managed").json()

    real_rmtree = uploads.shutil.rmtree
    cleanup_ran = False

    def assert_cleanup_is_outside_lock(path, *args, **kwargs):
        nonlocal cleanup_ran
        if Path(path) == directory / "files":
            lock_path = uploads.UPLOAD_ROOT.parent / ".managed-datasets.lock"
            with lock_path.open("a+b") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(lock, fcntl.LOCK_UN)
            cleanup_ran = True
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(uploads.shutil, "rmtree", assert_cleanup_is_outside_lock)
    recovered = complete(client, upload_id, retry_names)
    assert recovered.status_code == 200, recovered.text
    result = recovered.json()
    assert result["directory"] == str(destination.resolve())
    assert result["names"] == ["person", "hat"]
    assert cleanup_ran
    assert not (directory / "files").exists()
    receipt = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    marker = json.loads((destination / uploads.MANAGED_MARKER).read_text(encoding="utf-8"))
    config = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
    managed = client.get("/api/datasets/managed").json()
    assert receipt["result"] == result
    assert client.get(f"/api/datasets/uploads/{upload_id}").json()["result"] == result
    assert (destination / uploads.MANAGED_MARKER).read_bytes() == marker_before
    assert (destination / "data.yaml").read_bytes() == config_before
    assert managed == managed_before
    assert result["names"] == marker["classes"] == list(config["names"].values()) == managed[0]["classes"]
    assert result["stats"] == marker["stats"] == managed[0]["stats"]
    assert result["warnings"] == marker["warnings"] == managed[0]["warnings"]


def test_session_lock_prevents_cleanup_and_concurrent_mutation(environment):
    client, _ = environment
    upload_id = stage(client, dataset_files())
    directory = uploads.UPLOAD_ROOT / upload_id
    expiry = time.time() - uploads.SESSION_TTL - 10
    os.utime(directory / ".lock", (expiry, expiry))
    with (directory / ".lock").open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert create(client, dataset_files(), "another").status_code == 201
        assert directory.exists()
        assert client.delete(f"/api/datasets/uploads/{upload_id}").status_code == 409
        assert complete(client, upload_id, ["person", "hat"]).status_code == 409
    assert client.delete(f"/api/datasets/uploads/{upload_id}").status_code == 204


def test_retry_removes_partial_publication_before_disk_space_check(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    publication = root / f"{uploads.MANAGED_PUBLICATION_PREFIX}{upload_id}"
    publication.mkdir(parents=True)
    (publication / "partial.bin").write_bytes(b"partial")
    real_disk_usage = uploads.shutil.disk_usage
    checked = False

    def assert_partial_was_removed(path):
        nonlocal checked
        if Path(path) == root:
            assert not publication.exists()
            checked = True
        return real_disk_usage(path)

    monkeypatch.setattr(uploads.shutil, "disk_usage", assert_partial_was_removed)
    response = complete(client, upload_id, ["person", "hat"])
    assert response.status_code == 200, response.text
    assert checked


def test_cancel_upload_removes_partial_publication(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    publication = root / f"{uploads.MANAGED_PUBLICATION_PREFIX}{upload_id}"
    publication.mkdir(parents=True)
    (publication / "partial.bin").write_bytes(b"partial")

    assert client.delete(f"/api/datasets/uploads/{upload_id}").status_code == 204
    assert not publication.exists()
    assert not (uploads.UPLOAD_ROOT / upload_id).exists()


def test_publication_cleanup_respects_recent_and_locked_upload_sessions(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    publication = root / f"{uploads.MANAGED_PUBLICATION_PREFIX}{upload_id}"
    publication.mkdir(parents=True)
    expiry = time.time() - uploads.SESSION_TTL - 10
    os.utime(publication, (expiry, expiry))
    lock_path = uploads.UPLOAD_ROOT / upload_id / ".lock"

    uploads.cleanup_managed_dataset_workdirs()
    assert publication.is_dir()

    os.utime(lock_path, (expiry, expiry))
    with lock_path.open("r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        uploads.cleanup_managed_dataset_workdirs()
        assert publication.is_dir()

    uploads.cleanup_managed_dataset_workdirs()
    assert not publication.exists()


def test_managed_dataset_listing_and_deletion_cleans_receipt(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    result = complete(client, upload_id, ["person", "hat"])
    assert result.status_code == 200, result.text
    destination = root / "中文 训练集"
    assert (destination / uploads.MANAGED_MARKER).is_file()

    response = client.get("/api/datasets/managed")
    assert response.status_code == 200
    assert response.json() == [{
        "name": "中文 训练集",
        "directory": str(destination.resolve()),
        "config_path": str((destination / "data.yaml").resolve()),
        "created_at": response.json()[0]["created_at"],
        "total_bytes": sum(map(len, dataset_files().values())),
        "stats": {"train_images": 1, "val_images": 1, "labels": 2, "classes": 2, "background_images": 0},
        "classes": ["person", "hat"],
        "warnings": [],
        "references": [],
    }]

    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 204
    assert not destination.exists()
    assert not (uploads.UPLOAD_ROOT / upload_id).exists()
    assert client.get("/api/datasets/managed").json() == []


def test_legacy_delete_unknown_active_training_fails_closed(environment):
    client, root = environment
    assert complete(client, stage(client, dataset_files()), ['person', 'hat']).status_code == 200
    with client.app.state.session_factory() as db:
        db.add(TrainingTask(id='unknown-active', name='unknown active', status='running',
                            model_path='model.pt', data_path=str(root / 'missing.yaml')))
        db.commit()
    response = client.delete('/api/datasets/managed/中文 训练集')
    assert response.status_code == 409
    assert (root / '中文 训练集').is_dir()


def test_legacy_same_name_reupload_uses_current_catalogue_row(environment):
    from app.models import DatasetEntry
    client, root = environment
    assert complete(client, stage(client, dataset_files()), ['person', 'hat']).status_code == 200
    with client.app.state.session_factory() as db:
        # Legacy uploads now publish their catalogue row immediately. Model the
        # old tombstone alongside that actual current registration.
        current = db.scalar(select(DatasetEntry).where(
            DatasetEntry.path == str(root / '中文 训练集'), DatasetEntry.status == 'ready'))
        current_id = current.id
        db.add(DatasetEntry(id='old-deleted', location_id='legacy', node_id='local', name='中文 训练集',
                            path=str(root / '中文 训练集'), ownership='managed', status='deleted'))
        db.commit()
    assert client.delete('/api/datasets/managed/中文 训练集').status_code == 204
    with client.app.state.session_factory() as db:
        assert db.get(DatasetEntry, current_id).status == 'deleted'


def test_managed_dataset_lock_serializes_mutations(environment):
    acquired = threading.Event()

    def acquire_in_thread():
        with uploads.managed_dataset_mutation():
            acquired.set()

    with uploads.managed_dataset_mutation():
        thread = threading.Thread(target=acquire_in_thread)
        thread.start()
        assert not acquired.wait(0.1)
    assert acquired.wait(1)
    thread.join(timeout=1)
    assert not thread.is_alive()


def test_managed_dataset_delete_is_reference_protected(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    result = complete(client, upload_id, ["person", "hat"]).json()
    local_session = client.app.state.session_factory
    with local_session() as db:
        db.add(TrainingTask(id="referencing-task", name="正在使用数据集", task_type="detect",
                            model_path="/models/yolo.pt", data_path=result["resource"]["path"]))
        db.commit()

    listed = client.get("/api/datasets/managed").json()[0]
    assert listed["references"] == [{"id": "referencing-task", "name": "正在使用数据集",
                                      "status": "draft", "deleted": False}]
    blocked = client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86")
    assert blocked.status_code == 409
    assert "正在使用数据集" in blocked.json()["detail"]
    assert (root / "中文 训练集" / "data.yaml").is_file()

    with local_session() as db:
        task = db.get(TrainingTask, "referencing-task")
        task.deleted_at = utcnow()
        task.deleted_reason = "soft_delete"
        db.commit()

    recycled = client.get("/api/datasets/managed").json()[0]
    assert recycled["references"] == [{"id": "referencing-task", "name": "正在使用数据集",
                                        "status": "draft", "deleted": True}]
    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 409

    with local_session() as db:
        db.delete(db.get(TrainingTask, "referencing-task"))
        db.add(TrainingTask(id="remote-task", name="远程同路径任务", task_type="detect",
                            execution_node_id="remote-node", model_path="/models/yolo.pt",
                            data_path=result["resource"]["path"]))
        db.commit()
    assert client.get("/api/datasets/managed").json()[0]["references"] == []
    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 204


def test_managed_dataset_reference_matches_equivalent_symlink_path(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    result = complete(client, upload_id, ["person", "hat"]).json()
    alias = root.parent / "dataset-alias"
    alias.symlink_to(root / result["name"], target_is_directory=True)
    local_session = client.app.state.session_factory
    with local_session() as db:
        db.add(TrainingTask(id="symlink-reference", name="软链接引用", task_type="detect",
                            model_path="/models/yolo.pt", data_path=str(alias / "data.yaml")))
        db.commit()

    listed = client.get("/api/datasets/managed").json()[0]
    assert listed["references"][0]["id"] == "symlink-reference"
    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 409


def test_managed_dataset_reference_in_historical_run_remains_protected(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    result = complete(client, upload_id, ["person", "hat"]).json()
    unrelated = root.parent / "new-data.yaml"
    unrelated.write_text("train: images/train\n", encoding="utf-8")
    local_session = client.app.state.session_factory
    with local_session() as db:
        task = TrainingTask(
            id="historical-reference",
            name="历史轮次引用",
            task_type="detect",
            model_path="/models/yolo.pt",
            data_path=str(unrelated),
        )
        db.add(task)
        db.flush()
        db.add(TrainingRun(
            id="historical-reference-run",
            task_id=task.id,
            attempt_number=1,
            status="completed",
            model_path=task.model_path,
            data_path=result["resource"]["path"],
        ))
        db.commit()

    listed = client.get("/api/datasets/managed").json()[0]
    assert [reference["id"] for reference in listed["references"]] == [task.id]
    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 409


def test_managed_dataset_detects_external_yaml_and_manifest_references_without_sibling_false_positive(environment):
    client, root = environment
    upload_id = stage(client, dataset_files())
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200
    target = root / "中文 训练集"
    sibling = root / "sibling"
    (sibling / "images" / "train").mkdir(parents=True)
    configs = root.parent / "external-configs"
    configs.mkdir()
    rooted = configs / "rooted.yaml"
    rooted.write_text(
        yaml.safe_dump({"path": str(target), "train": "images/train", "val": "images/val"}),
        encoding="utf-8",
    )
    split = configs / "split.yaml"
    split.write_text(
        yaml.safe_dump({"path": str(root), "train": "中文 训练集/images/train", "val": "sibling/images/train"}),
        encoding="utf-8",
    )
    manifest = configs / "train.txt"
    manifest.write_text(str(target / "images" / "train" / "样本.png") + "\n", encoding="utf-8")
    listed = configs / "listed.yaml"
    listed.write_text(yaml.safe_dump({"train": "train.txt"}), encoding="utf-8")
    unrelated = configs / "unrelated.yaml"
    unrelated.write_text(
        yaml.safe_dump({"path": str(root), "train": "sibling/images/train"}),
        encoding="utf-8",
    )

    local_session = client.app.state.session_factory
    with local_session() as db:
        for task_id, data_path in (
            ("rooted-reference", rooted),
            ("split-reference", split),
            ("manifest-reference", listed),
            ("sibling-reference", unrelated),
        ):
            db.add(TrainingTask(
                id=task_id,
                name=task_id,
                task_type="detect",
                model_path="/models/yolo.pt",
                data_path=str(data_path),
            ))
        db.commit()

    references = client.get("/api/datasets/managed").json()[0]["references"]
    assert {reference["id"] for reference in references} == {
        "rooted-reference", "split-reference", "manifest-reference",
    }
    assert client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86").status_code == 409


def test_managed_dataset_cleanup_failure_remains_detached(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200
    destination = root / "中文 训练集"
    real_rmtree = uploads.shutil.rmtree

    def fail_quarantine_cleanup(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent.name == ".platform-deletions":
            raise OSError("simulated cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(uploads.shutil, "rmtree", fail_quarantine_cleanup)
    response = client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86")
    assert response.status_code == 507
    assert "已从平台移除" in response.json()["detail"]
    assert not destination.exists()
    assert client.get("/api/datasets/managed").json() == []
    detached = list((root / ".platform-deletions").iterdir())
    assert len(detached) == 1
    assert (detached[0] / "data.yaml").is_file()
    assert services.discover_files([root], {".yaml"}) == []
    with pytest.raises(ValueError, match="隐藏目录"):
        services.validate_dataset_config(str(detached[0] / "data.yaml"))


def test_managed_dataset_delete_succeeds_when_housekeeping_finishes_cleanup(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200
    real_rmtree = uploads.shutil.rmtree

    def cleanup_first(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent.name == ".platform-deletions":
            real_rmtree(path, *args, **kwargs)
            raise FileNotFoundError(candidate)
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(uploads.shutil, "rmtree", cleanup_first)
    response = client.delete("/api/datasets/managed/%E4%B8%AD%E6%96%87%20%E8%AE%AD%E7%BB%83%E9%9B%86")
    assert response.status_code == 204
    assert client.get("/api/datasets/managed").json() == []
    assert not (root / "中文 训练集").exists()


def test_upload_creation_continues_when_managed_cleanup_is_temporarily_unavailable(environment, monkeypatch, caplog):
    client, _ = environment

    def fail_cleanup():
        raise PermissionError("simulated cleanup failure")

    monkeypatch.setattr(uploads, "cleanup_managed_dataset_workdirs", fail_cleanup)
    with caplog.at_level("WARNING", logger=uploads.__name__):
        response = create(client, dataset_files())
    assert response.status_code == 201
    assert "managed_dataset_workdir_cleanup_failed" in caplog.text


def test_unmarked_or_invalid_managed_dataset_is_never_deletable(environment):
    client, root = environment
    unmarked = root / "manual-dataset"
    unmarked.mkdir(parents=True)
    (unmarked / "data.yaml").write_text("train: images/train\nval: images/val\n")
    invalid = root / "invalid-marker"
    invalid.mkdir()
    (invalid / "data.yaml").write_text("train: images/train\nval: images/val\n")
    (invalid / uploads.MANAGED_MARKER).write_text("{}")

    assert client.get("/api/datasets/managed").json() == []
    assert client.delete("/api/datasets/managed/manual-dataset").status_code == 404
    assert client.delete("/api/datasets/managed/invalid-marker").status_code == 404
    assert unmarked.is_dir() and invalid.is_dir()

    damaged = root / "damaged-metadata"
    damaged.mkdir()
    (damaged / "data.yaml").write_text("train: images/train\nval: images/val\n")
    (damaged / uploads.MANAGED_MARKER).write_text(json.dumps({
        "version": 1, "name": damaged.name, "created_at": 42, "total_bytes": "bad",
        "stats": ["bad"], "classes": [1, "person"], "warnings": ["warning", None],
        "upload_id": 123,
    }))
    listed = client.get("/api/datasets/managed")
    assert listed.status_code == 200
    assert listed.json()[0] == {
        "name": damaged.name, "directory": str(damaged.resolve()),
        "config_path": str((damaged / "data.yaml").resolve()), "created_at": None,
        "total_bytes": 0, "stats": {}, "classes": ["person"], "warnings": ["warning"],
        "references": [],
    }
