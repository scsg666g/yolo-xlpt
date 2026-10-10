"""Exercise directory publication with real image bytes and isolated filesystem roots."""
import io
import fcntl
import os
import time
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import app.dataset_uploads as uploads
import app.services as services


@pytest.fixture
def environment(tmp_path, monkeypatch):
    dataset_root = tmp_path / "数据集"
    monkeypatch.setattr(uploads, "UPLOAD_ROOT", tmp_path / "staging")
    monkeypatch.setattr(uploads, "DATASET_ROOTS", [dataset_root])
    monkeypatch.setattr(services, "DATASET_ROOTS", [dataset_root])
    isolated = FastAPI()
    isolated.include_router(uploads.router)
    with TestClient(isolated) as client:
        yield client, dataset_root


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
    original = uploads.shutil.copytree
    with monkeypatch.context() as patch:
        def fail_copy(*args, **kwargs):
            raise OSError("disk full")
        patch.setattr(uploads.shutil, "copytree", fail_copy)
        assert complete(client, upload_id, ["person", "hat"]).status_code == 507
    assert uploads.shutil.copytree is original
    assert not (root / "中文 训练集").exists()
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


def test_receipt_write_failure_does_not_leave_a_cached_success(environment, monkeypatch):
    client, root = environment
    upload_id = stage(client, dataset_files())
    original = Path.write_text
    with monkeypatch.context() as patch:
        def fail_receipt(path, *args, **kwargs):
            if path.name == ".manifest.tmp":
                raise OSError("disk full")
            return original(path, *args, **kwargs)
        patch.setattr(Path, "write_text", fail_receipt)
        assert complete(client, upload_id, ["person", "hat"]).status_code == 507
    assert not (root / "中文 训练集").exists()
    assert "result" not in client.get(f"/api/datasets/uploads/{upload_id}").json()
    assert complete(client, upload_id, ["person", "hat"]).status_code == 200


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
