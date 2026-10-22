"""Remote lifecycle integration with a real Agent and files isolated in tmp_path."""
import hashlib
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError

import pytest
from sqlalchemy import event, select

import app.model_test_storage as model_test_storage
import app.remote_operations as operations
from app.models import Artifact, ComputeNode, MetricPoint, ModelTest, RemoteOperation, TaskEvent, TrainingRun, TrainingTask, utcnow
from app.parameter_catalog import training_parameter_keys
from app.run_deletions import validate_attempt_deletion
from app.services import create_training_run
from test_compute_nodes import create, headers, monitoring_client, sample
from test_remote_agent import agent, training


def remote_run(client, factory, tmp_path):
    credential = create(client)
    root = tmp_path / "output"
    root.mkdir()
    model, data = tmp_path / "model.pt", tmp_path / "data.yaml"
    model.write_bytes(b"base model")
    data.write_text("train: images/train\n")
    with factory() as db:
        node = db.get(ComputeNode, credential["node"]["id"])
        node.agent_version = "1.8.0"
        node.last_seen_at = utcnow()
        node.snapshot = {"sampled_at": utcnow().isoformat(),
                         "training": {"enabled": True, "ready": True, "supported_parameters": []}}
        task = TrainingTask(id="remote-lifecycle", name="Remote lifecycle", task_type="detect",
                            execution_node_id=credential["node"]["id"], model_path=str(model), data_path=str(data),
                            epochs=20, config={"remote_output_root": str(root)})
        db.add(task)
        db.flush()
        run = create_training_run(db, task, "fresh")
        run.status = task.status = "stopped"
        run.completed_epochs = task.completed_epochs = 3
        run.progress = task.progress = 15
        db.commit()
        payload = operations.run_payload(run, task)
    run_dir = Path(payload["run_dir"])
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights/last.pt").write_bytes(b"remote weight\0" * 90000)
    (run_dir / "results.csv").write_text("epoch,time,score\n3,1,0.8\n")
    log = Path(payload["log_path"])
    log.parent.mkdir(parents=True)
    log.write_text("independent log")
    config = {"node_id": credential["node"]["id"], "allowed_training_roots": [str(tmp_path)],
              "training_output_roots": [str(root)], "training_cwd": str(tmp_path),
              "yolo_bin": "/usr/bin/python3", "_workspace_state_dir": str(tmp_path / ".workspace-state")}
    return credential, config, payload


def agent_api(client, credential):
    def api(config, path, method="GET", payload=None, extra_headers=None, *, timeout_seconds=20):
        response = client.request(method, path, json=payload, headers=headers(credential))
        if response.status_code >= 400:
            raise HTTPError(path, response.status_code, response.text, {}, None)
        return response.json()
    return api


def claim(client, credential):
    return client.post(f"/api/compute/nodes/{credential['node']['id']}/agent/operations/poll",
                       json={}, headers=headers(credential)).json()["operation"]


def execute_through(client, credential, config, agent, operation_id):
    while work := claim(client, credential):
        agent.execute_operation(config, work, threading.Event())
        if work["id"] == operation_id:
            return work
    raise AssertionError(f"remote operation was not claimed: {operation_id}")


def test_old_agent_cannot_claim_remote_file_operations(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    client.get(f"/api/tasks/{payload['task_id']}/artifacts")
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "1.7.0"
        db.commit()
    blocked = client.post(
        f"/api/compute/nodes/{credential['node']['id']}/agent/operations/poll",
        json={}, headers=headers(credential),
    ).json()
    assert blocked["operation"] is None
    assert blocked["required_agent_version"] == "1.8.0"
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "1.8.0"
        db.commit()
    assert claim(client, credential) is not None


def test_remote_delete_confirm_retry_and_local_isolation(monitoring_client, agent, tmp_path, monkeypatch):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", tmp_path / "model-test-history")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    task_id = payload["task_id"]
    history_file = model_test_storage.test_history_root(task_id, "c" * 32) / "results" / "0.jpg"
    history_file.parent.mkdir(parents=True)
    history_file.write_bytes(b"remote detection result")
    task_url = f"/api/tasks/{task_id}"
    other_dir = tmp_path / "output/detect/another-task/attempt-001"
    other_dir.mkdir(parents=True)
    (other_dir / "keep.txt").write_text("untouched")
    assert client.post(task_url + "/trash").status_code == 204
    assert history_file.is_file()
    assert client.post(task_url + "/restore").status_code == 200
    assert history_file.is_file()
    assert client.delete(task_url + "/permanent").status_code == 202
    assert client.delete(task_url + "/permanent").status_code == 202
    with factory() as db:
        assert db.get(TrainingTask, task_id).deleted_reason == "remote_delete_pending"
        assert len(db.scalars(select(RemoteOperation)).all()) == 1
    assert client.post(task_url + "/restore").status_code == 409
    assert Path(payload["run_dir"]).exists()  # Offline: neither local nor remote files deleted.
    work = claim(client, credential)
    other = create(client, "Other")
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{work['id']}/result"
    result = {"lease": work["lease"], "success": False, "error": "permission denied"}
    assert client.post(endpoint, json=result, headers=headers(other)).status_code == 401
    assert client.post(endpoint, json=result, headers=headers(credential)).status_code == 200
    with factory() as db:
        assert db.get(TrainingTask, task_id).deleted_reason == "remote_delete_failed"
    assert history_file.is_file()
    assert client.delete(task_url + "/permanent").status_code == 202
    work = claim(client, credential)
    agent.execute_operation(config, work, threading.Event())
    assert not Path(payload["run_dir"]).exists() and not Path(payload["log_path"]).exists()
    assert (other_dir / "keep.txt").read_text() == "untouched"
    assert (tmp_path / "model.pt").read_bytes() == b"base model"
    assert (tmp_path / "data.yaml").is_file()
    assert not model_test_storage.task_history_root(task_id).exists()
    with factory() as db:
        assert db.get(TrainingTask, task_id) is None
        assert not db.scalars(select(TrainingRun)).all()
    # Retry of a lost result ACK still succeeds after the task record is gone.
    result = {"lease": work["lease"], "success": True}
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{work['id']}/result"
    assert client.post(endpoint, json=result, headers=headers(credential)).json()["status"] == "succeeded"


def test_remote_delete_keeps_task_retryable_when_center_cleanup_fails(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", tmp_path / "model-test-history")
    task_id = payload["task_id"]
    history_file = model_test_storage.test_history_root(task_id, "d" * 32) / "detections.json"
    history_file.parent.mkdir(parents=True)
    history_file.write_text("{}", encoding="utf-8")

    assert client.delete(f"/api/tasks/{task_id}/permanent").status_code == 202
    work = claim(client, credential)
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{work['id']}/result"

    def cleanup_failure(_task_id: str) -> None:
        raise PermissionError("history is read-only")

    monkeypatch.setattr(operations, "delete_task_history", cleanup_failure)
    result = client.post(endpoint, json={"lease": work["lease"], "success": True}, headers=headers(credential))
    assert result.status_code == 200
    assert result.json()["status"] == "failed"
    with factory() as db:
        task = db.get(TrainingTask, task_id)
        assert task is not None
        assert task.deleted_reason == "remote_delete_failed"
        assert "中心关联数据清理失败" in task.error_message
    assert history_file.is_file()

    monkeypatch.setattr(operations, "delete_task_history", model_test_storage.delete_task_history)
    assert client.delete(f"/api/tasks/{task_id}/permanent").status_code == 202
    retry = claim(client, credential)
    retry_endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{retry['id']}/result"
    retried = client.post(
        retry_endpoint, json={"lease": retry["lease"], "success": True}, headers=headers(credential),
    )
    assert retried.status_code == 200
    assert retried.json()["status"] == "succeeded"
    with factory() as db:
        assert db.get(TrainingTask, task_id) is None
    assert not model_test_storage.task_history_root(task_id).exists()


def test_remote_attempt_delete_retries_and_preserves_history_and_other_attempt(
    monitoring_client, agent, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    task_id, run_id = payload["task_id"], payload["run_id"]
    task_url = f"/api/tasks/{task_id}"
    second_dir = Path(payload["output_root"]) / "detect" / task_id / "attempt-002"
    second_log = Path(payload["output_root"]) / ".platform-logs" / "20261021" / task_id / "attempt-002.log"
    second_dir.mkdir(parents=True)
    (second_dir / "keep.txt").write_text("other Attempt")
    second_log.parent.mkdir(parents=True, exist_ok=True)
    second_log.write_text("other Attempt log")

    with factory() as db:
        task = db.get(TrainingTask, task_id)
        first = db.get(TrainingRun, run_id)
        # Simulate an older remote record whose root and run-level node field
        # were not persisted. Its canonical run_dir is enough to recover root.
        first.execution_node_id = None
        first.parameter_snapshot = {}
        task.config = {}
        second = TrainingRun(
            id="remote-attempt-two", task_id=task_id, attempt_number=2, status="stopped",
            execution_node_id=credential["node"]["id"], model_path=task.model_path,
            data_path=task.data_path, run_dir=str(second_dir), log_path=str(second_log),
        )
        task.current_run_id = second.id
        task.run_dir = str(second_dir)
        task.log_path = str(second_log)
        artifact = Artifact(
            task_id=task_id, run_id=first.id, kind="weight", name="weights/last.pt",
            path=str(Path(payload["run_dir"]) / "weights/last.pt"), size=12,
        )
        db.add_all([
            second,
            artifact,
            MetricPoint(task_id=task_id, run_id=first.id, epoch=1, elapsed=1, values={"map": 0.5}),
            TaskEvent(task_id=task_id, run_id=first.id, event_type="epoch", message="epoch 1", payload={}),
        ])
        db.flush()
        db.add(ModelTest(
            id="e" * 32, task_id=task_id, run_id=first.id, source_run_id=first.id,
            artifact_id=artifact.id, artifact_name=artifact.name, model_name=artifact.name,
            confidence=0.25, iou=0.7, status="completed",
        ))
        db.commit()

    delete_url = f"{task_url}/runs/{run_id}"
    response = client.delete(delete_url)
    assert response.status_code == 202, response.text
    failed_work = claim(client, credential)
    assert failed_work["payload"]["scope"] == "attempt"
    assert failed_work["payload"]["runs"][0]["output_root"] == payload["output_root"]
    result_url = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{failed_work['id']}/result"
    failed = client.post(
        result_url,
        json={"lease": failed_work["lease"], "success": False, "error": "temporary offline"},
        headers=headers(credential),
    )
    assert failed.status_code == 200 and failed.json()["status"] == "failed"
    with factory() as db:
        assert db.get(TrainingRun, run_id) is not None
        assert db.get(TrainingTask, task_id) is not None

    # A failed remote cleanup remains retryable from the same UI action.
    retried = client.delete(delete_url)
    assert retried.status_code == 202, retried.text
    retry_work = claim(client, credential)
    assert retry_work["payload"]["scope"] == "attempt"
    agent.execute_operation(config, retry_work, threading.Event())

    with factory() as db:
        assert db.get(TrainingTask, task_id) is not None
        assert db.get(TrainingRun, run_id) is None
        assert db.get(TrainingRun, "remote-attempt-two") is not None
        assert db.scalar(select(MetricPoint).where(MetricPoint.run_id == run_id)) is None
        assert db.scalar(select(TaskEvent).where(TaskEvent.run_id == run_id)) is None
        assert db.scalar(select(Artifact).where(Artifact.run_id == run_id)) is None
        history = db.get(ModelTest, "e" * 32)
        assert history is not None
        assert history.run_id is None and history.source_run_id is None and history.artifact_id is None
    assert not Path(payload["run_dir"]).exists()
    assert not Path(payload["log_path"]).exists()
    assert (second_dir / "keep.txt").read_text() == "other Attempt"
    assert second_log.read_text() == "other Attempt log"


def test_remote_attempt_delete_rejects_unresolvable_legacy_directory(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    with factory() as db:
        task = db.get(TrainingTask, payload["task_id"])
        run = db.get(TrainingRun, payload["run_id"])
        task.config = {}
        run.parameter_snapshot = {}
        run.run_dir = str(tmp_path / "not-a-canonical-attempt")
        db.commit()

    response = client.delete(f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}")
    assert response.status_code == 409
    assert "输出根目录" in response.json()["detail"]
    with factory() as db:
        assert db.get(TrainingRun, payload["run_id"]) is not None
        assert not db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == payload["task_id"],
            RemoteOperation.kind == "delete",
        )).all()


def test_remote_attempt_delete_requires_supported_agent_version(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "1.7.0"
        db.commit()

    response = client.delete(f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}")
    assert response.status_code == 409
    assert "1.8.0" in response.json()["detail"]
    with factory() as db:
        assert not db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == payload["task_id"],
            RemoteOperation.kind == "delete",
        )).all()


def test_remote_legacy_flat_attempt_requires_agent_201(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    with factory() as db:
        node = db.get(ComputeNode, credential["node"]["id"])
        node.agent_version = "2.0.0"
        task = db.get(TrainingTask, payload["task_id"])
        run = db.get(TrainingRun, payload["run_id"])
        legacy_root = Path(payload["run_dir"]).parent
        run.run_dir = task.run_dir = str(legacy_root)
        db.commit()

    response = client.delete(f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}")
    assert response.status_code == 409
    assert "2.0.1" in response.json()["detail"]
    with factory() as db:
        assert not db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == payload["task_id"],
            RemoteOperation.kind == "delete",
        )).all()


def test_remote_legacy_flat_attempt_delete_preserves_canonical_child(
    monitoring_client, agent, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    task_id, first_id = payload["task_id"], payload["run_id"]
    task_root = Path(payload["run_dir"]).parent
    first_dir = Path(payload["run_dir"])
    for child in list(first_dir.iterdir()):
        child.rename(task_root / child.name)
    first_dir.rmdir()
    first_weight = task_root / "weights" / "last.pt"
    second_dir = task_root / "attempt-002"
    second_log = Path(payload["output_root"]) / ".platform-logs" / "20261021" / task_id / "attempt-002.log"
    second_dir.mkdir()
    marker = second_dir / "keep.txt"
    marker.write_text("canonical child")
    second_log.parent.mkdir(parents=True, exist_ok=True)
    second_log.write_text("canonical log")

    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "2.0.1"
        task = db.get(TrainingTask, task_id)
        first = db.get(TrainingRun, first_id)
        first.run_dir = str(task_root)
        first.log_path = str(task_root / "platform.log")
        (task_root / "platform.log").write_text("legacy log")
        second = TrainingRun(
            id="remote-flat-second", task_id=task_id, attempt_number=2, status="stopped",
            execution_node_id=credential["node"]["id"], model_path=task.model_path,
            data_path=task.data_path, run_dir=str(second_dir), log_path=str(second_log),
            parameter_snapshot={"remote_output_root": payload["output_root"]},
        )
        task.current_run_id = second.id
        task.run_dir = str(second_dir)
        task.log_path = str(second_log)
        db.add_all([
            second,
            Artifact(
                task_id=task_id, run_id=first.id, kind="weight", name="weights/last.pt",
                path=str(first_weight), size=first_weight.stat().st_size,
            ),
        ])
        db.flush()
        # Deleting the canonical child first must also pass despite the legacy
        # run storing its task-container path.
        validate_attempt_deletion(db, task, second)
        db.commit()

    response = client.delete(f"/api/tasks/{task_id}/runs/{first_id}")
    assert response.status_code == 202, response.text
    work = claim(client, credential)
    assert work["payload"]["runs"][0]["layout"] == "legacy_flat"
    agent.execute_operation(config, work, threading.Event())

    with factory() as db:
        assert db.get(TrainingRun, first_id) is None
        assert db.get(TrainingRun, "remote-flat-second") is not None
    assert not first_weight.exists()
    assert not (task_root / "platform.log").exists()
    assert marker.read_text() == "canonical child"
    assert second_log.read_text() == "canonical log"


def test_remote_delete_refreshes_references_before_dispatch(monitoring_client, agent, tmp_path, monkeypatch):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    task_id = payload["task_id"]
    best = Path(payload["run_dir"]) / "weights/best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"referenced model")

    assert client.post(f"/api/tasks/{task_id}/trash").status_code == 204
    assert client.delete(f"/api/tasks/{task_id}/permanent").status_code == 202

    # The Agent is offline while another task starts referencing this model.
    # The queued payload is deliberately stale until the operation is claimed.
    with factory() as db:
        operation = db.scalar(select(RemoteOperation).where(RemoteOperation.task_id == task_id))
        assert str(best) not in operation.payload["protected_paths"]
        dependent = TrainingTask(
            id="new-dependent-task", name="Dependent", task_type="detect",
            execution_node_id=credential["node"]["id"], model_path=str(best),
            data_path=str(tmp_path / "data.yaml"), epochs=1,
            config={"remote_output_root": str(tmp_path / "dependent-output")},
        )
        db.add(dependent)
        db.flush()
        dependent_run = create_training_run(db, dependent, "fresh")
        dependent_run.checkpoint_path = str(best)
        dependent_run.status = dependent.status = "stopped"
        db.commit()

    work = claim(client, credential)
    assert str(best) in work["payload"]["protected_paths"]
    agent.execute_operation(config, work, threading.Event())

    assert best.read_bytes() == b"referenced model"
    with factory() as db:
        assert db.get(TrainingTask, task_id).deleted_reason == "remote_delete_failed"


def test_remote_delete_serializes_same_node_references_until_completion(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, _config, payload = remote_run(client, factory, tmp_path)
    task_id = payload["task_id"]
    best = Path(payload["run_dir"]) / "weights/best.pt"
    best.parent.mkdir(parents=True, exist_ok=True)
    best.write_bytes(b"model scheduled for deletion")

    assert client.post(f"/api/tasks/{task_id}/trash").status_code == 204
    assert client.delete(f"/api/tasks/{task_id}/permanent").status_code == 202

    # The center cannot resolve symlinks on a remote server. Treat this
    # lexically unrelated path as a possible alias and serialize all new
    # references on the node while deletion is pending.
    possible_alias = str(tmp_path / "mounted-alias/weights/best.pt")
    create_response = client.post("/api/tasks", json={
        "name": "Pending alias dependency",
        "execution_node_id": credential["node"]["id"],
        "model_path": possible_alias,
        "data_path": str(tmp_path / "data.yaml"),
        "remote_output_root": str(tmp_path / "dependent-output"),
    })
    assert create_response.status_code == 409
    assert "正在清理任务文件" in create_response.json()["detail"]

    with factory() as db:
        editable = TrainingTask(
            id="editable-dependent", name="Editable", task_type="detect",
            execution_node_id=credential["node"]["id"], model_path=str(tmp_path / "model.pt"),
            data_path=str(tmp_path / "data.yaml"), config={"remote_output_root": str(tmp_path / "editable")},
        )
        clone_source = TrainingTask(
            id="clone-dependent", name="Clone source", task_type="detect",
            execution_node_id=credential["node"]["id"], model_path=str(tmp_path / "unrelated-model.pt"),
            data_path=str(tmp_path / "data.yaml"), config={"remote_output_root": str(tmp_path / "clone")},
        )
        db.add_all([editable, clone_source])
        db.commit()

    update_response = client.patch("/api/tasks/editable-dependent/resources", json={"model_path": possible_alias})
    clone_response = client.post("/api/tasks/clone-dependent/clone")
    assert update_response.status_code == 409
    assert clone_response.status_code == 409
    assert "正在清理任务文件" in update_response.json()["detail"]
    assert "正在清理任务文件" in clone_response.json()["detail"]

    # Deletion on one server must not pause normal work on another server.
    other = create(client, "Other deletion-independent node")
    other_report = sample()
    other_report["training"]["supported_parameters"] = sorted(training_parameter_keys())
    assert client.post(
        f"/api/compute/nodes/{other['node']['id']}/report",
        headers=headers(other), json=other_report,
    ).status_code == 200
    other_response = client.post("/api/tasks", json={
        "name": "Other node task",
        "execution_node_id": other["node"]["id"],
        "model_path": possible_alias,
        "data_path": str(tmp_path / "data.yaml"),
        "remote_output_root": str(tmp_path / "other-output"),
    })
    assert other_response.status_code == 201, other_response.text

    work = claim(client, credential)
    assert work["kind"] == "delete"
    running_response = client.post("/api/tasks", json={
        "name": "Running alias dependency",
        "execution_node_id": credential["node"]["id"],
        "model_path": str(tmp_path / "another-alias/model.pt"),
        "data_path": str(tmp_path / "another-alias/data.yaml"),
        "remote_output_root": str(tmp_path / "another-output"),
    })
    assert running_response.status_code == 409
    assert "正在清理任务文件" in running_response.json()["detail"]

    with factory() as db:
        assert db.get(TrainingTask, "editable-dependent").model_path == str(tmp_path / "model.pt")
        assert db.get(RemoteOperation, work["id"]).status == "running"
    assert best.is_file()


def test_manifest_resume_and_chunked_download(monitoring_client, agent, tmp_path, monkeypatch):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    api = agent_api(client, credential)
    monkeypatch.setattr(agent, "api_json", api)
    agent.report_manifest(config, payload)
    artifacts = client.get(f"/api/tasks/{payload['task_id']}/artifacts").json()
    weight = next(a for a in artifacts if a["name"] == "weights/last.pt")
    assert weight["path"] == str(Path(payload["run_dir"]) / "weights/last.pt")
    assert client.get(f"/api/artifacts/{weight['id']}/download").status_code == 409
    prepare_url = f"/api/artifacts/{weight['id']}/prepare"
    assert client.post(prepare_url).json()["status"] == "pending"
    lost = False
    def unreliable_api(config, path, method="GET", payload=None, extra_headers=None, *, timeout_seconds=20):
        nonlocal lost
        response = api(
            config,
            path,
            method,
            payload,
            extra_headers,
            timeout_seconds=timeout_seconds,
        )
        if path.endswith("/chunk") and not lost:
            lost = True
            raise OSError("lost ACK after successful write")
        return response
    monkeypatch.setattr(agent, "api_json", unreliable_api)
    while work := claim(client, credential):
        agent.execute_operation(config, work, threading.Event())
    assert lost
    assert client.post(prepare_url).json()["status"] == "ready"
    download = client.get(f"/api/artifacts/{weight['id']}/download")
    assert download.status_code == 200
    assert download.content == (Path(payload["run_dir"]) / "weights/last.pt").read_bytes()
    with factory() as db:
        fetched = db.scalar(select(RemoteOperation).where(
            RemoteOperation.kind == "fetch", RemoteOperation.status == "succeeded",
        ))
        assert fetched is not None
        assert fetched.payload["sha256"] == hashlib.sha256(download.content).hexdigest()
    response = client.post(f"/api/tasks/{payload['task_id']}/resume")
    assert response.status_code == 200, response.text
    assert response.json()["completed_epochs"] == 3
    with factory() as db:
        new_run = db.get(TrainingRun, response.json()["current_run_id"])
        assert new_run.attempt_number == 2 and new_run.launch_mode == "resume"
        assert new_run.checkpoint_path == weight["path"]
        assert new_run.run_dir != payload["run_dir"]
    assert client.delete(f"/api/tasks/{payload['task_id']}/permanent").status_code == 409


def test_prepare_refetches_same_size_corrupt_succeeded_cache(
    monitoring_client, agent, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    agent.report_manifest(config, payload)
    weight = next(
        artifact for artifact in client.get(f"/api/tasks/{payload['task_id']}/artifacts").json()
        if artifact["name"] == "weights/last.pt"
    )
    prepare_url = f"/api/artifacts/{weight['id']}/prepare"
    first = client.post(prepare_url).json()
    work = execute_through(client, credential, config, agent, first["operation_id"])
    assert work["id"] == first["operation_id"]
    assert client.post(prepare_url).json()["status"] == "ready"

    with factory() as db:
        original = db.get(RemoteOperation, work["id"])
        cached = operations.operation_file(original)
        expected = (Path(payload["run_dir"]) / "weights/last.pt").read_bytes()
        cached.write_bytes(b"x" * len(expected))

    replacement = client.post(prepare_url).json()
    assert replacement["status"] == "pending"
    assert replacement["operation_id"] != work["id"]
    repeated = client.post(prepare_url).json()
    assert repeated["operation_id"] == replacement["operation_id"]
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == payload["task_id"],
            RemoteOperation.kind == "fetch",
        )))
        assert len(fetches) == 2

    retry = execute_through(client, credential, config, agent, replacement["operation_id"])
    assert retry["id"] == replacement["operation_id"]
    assert client.post(prepare_url).json()["status"] == "ready"
    assert client.get(f"/api/artifacts/{weight['id']}/download").content == expected


def test_prepare_backfills_legacy_cache_digest_without_refetch(
    monitoring_client, agent, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    agent.report_manifest(config, payload)
    weight = next(
        artifact for artifact in client.get(f"/api/tasks/{payload['task_id']}/artifacts").json()
        if artifact["name"] == "weights/last.pt"
    )
    prepare_url = f"/api/artifacts/{weight['id']}/prepare"
    first = client.post(prepare_url).json()
    execute_through(client, credential, config, agent, first["operation_id"])
    with factory() as db:
        original = db.get(RemoteOperation, first["operation_id"])
        original.payload = {key: value for key, value in original.payload.items() if key != "sha256"}
        old_cache = operations.operation_file(original)
        old_bytes = old_cache.read_bytes()
        db.commit()

    assert client.post(prepare_url).json()["status"] == "ready"
    assert old_cache.read_bytes() == old_bytes
    assert client.post(prepare_url).json()["status"] == "ready"
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == payload["task_id"],
            RemoteOperation.kind == "fetch",
        )))
        assert len(fetches) == 1
        assert fetches[0].payload["sha256"] == hashlib.sha256(old_bytes).hexdigest()


def test_agent_delete_validates_entire_plan_and_symlinks(agent, training, tmp_path):
    config, task = training
    agent.build_training_command(config, task)
    run = {**task, "output_root": task["parameters"]["remote_output_root"]}
    log = Path(task["log_path"])
    log.write_text("keep until all targets validated")
    bad = dict(run, run_dir=str(tmp_path))
    with pytest.raises(ValueError):
        agent.delete_remote_files(config, {"runs": [run, bad]})
    assert Path(run["run_dir"]).is_dir() and log.is_file()
    with pytest.raises(ValueError):
        agent.delete_remote_files(config, {"runs": [run], "protected_paths": [str(Path(run["run_dir"]) / "dataset.yaml")]})
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("keep")
    # Internal directory links are removed, never followed.
    (Path(run["run_dir"]) / "external").symlink_to(outside, target_is_directory=True)
    agent.delete_remote_files(config, {"runs": [run]})
    assert (outside / "keep").read_text() == "keep"
    agent.delete_remote_files(config, {"runs": [run]})  # repeat after partial success/lost ACK
    Path(run["run_dir"]).parent.mkdir(parents=True)
    Path(run["run_dir"]).symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        agent.operation_paths(config, run)


def test_remote_delete_rejects_missing_or_unmounted_output_root(monitoring_client, agent, tmp_path, monkeypatch):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    output_root = Path(payload["output_root"])
    task_id = payload["task_id"]
    shutil.rmtree(output_root)
    config["allowed_training_roots"] = [str(tmp_path / "this-server-only")]

    assert client.delete(f"/api/tasks/{task_id}/permanent").status_code == 202
    work = claim(client, credential)
    agent.execute_operation(config, work, threading.Event())
    with factory() as db:
        task = db.get(TrainingTask, task_id)
        assert task is not None and task.deleted_reason == "remote_delete_failed"
        assert "输出根目录不存在" in (task.error_message or "")

    # A missing or unmounted root cannot prove that remote data was cleaned.
    # Scan/fetch also reject it, and a root outside the current server's
    # configured roots must never become a delete target.
    with pytest.raises(ValueError, match="输出根目录不存在"):
        agent.operation_paths(config, payload)
    output_root.mkdir()
    with pytest.raises(ValueError, match="允许目录"):
        agent.delete_remote_files(config, {"runs": [payload]})
    output_root.rmdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    output_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        agent.delete_remote_files(config, {"runs": [payload]})


def test_operation_lease_and_bad_artifact_paths(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/runs/{payload['run_id']}/artifacts"
    for name in ("../model.pt", "/etc/passwd", "weights/../last.pt", "weights//last.pt", "weights\\last.pt"):
        assert client.post(endpoint, headers=headers(credential), json={"items": [{"name": name, "size": 1, "modified_at": utcnow().isoformat()}]}).status_code == 422
    client.get(f"/api/tasks/{payload['task_id']}/artifacts")
    work = claim(client, credential)
    with factory() as db:
        operation = db.get(RemoteOperation, work["id"])
        operation.updated_at = utcnow() - timedelta(minutes=6)
        db.commit()
    reclaimed = claim(client, credential)
    assert reclaimed["id"] == work["id"] and reclaimed["lease"] != work["lease"]
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{work['id']}/result"
    assert client.post(endpoint, headers=headers(credential), json={"lease": work["lease"], "success": True}).status_code == 409


def test_resume_command_keeps_old_attempt_and_checkpoint(agent, training):
    config, task = training
    old = Path(task["run_dir"])
    (old / "weights").mkdir(parents=True)
    checkpoint = old / "weights/last.pt"
    checkpoint.write_bytes(b"checkpoint")
    task.update(attempt_number=2, run_dir=str(old.parent / "attempt-002"), checkpoint_path=str(checkpoint),
                log_path=str(Path(task["log_path"]).with_name("attempt-002.log")))
    command = agent.build_training_command(config, task)
    assert f"resume={checkpoint}" in command
    assert f"save_dir={task['run_dir']}" in command
    assert checkpoint.read_bytes() == b"checkpoint"


def test_concurrent_polls_keep_one_active_run_per_server(monitoring_client):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential["node"]["id"]
    with factory() as db:
        node = db.get(ComputeNode, node_id)
        node.agent_version = "1.8.0"
        node.last_seen_at = utcnow()
        node.snapshot = {
            "sampled_at": utcnow().isoformat(),
            "training": {"enabled": True, "ready": True, "supported_parameters": []},
        }
        for number in range(2):
            task = TrainingTask(id=f"queued-{number}", name=f"Queued {number}", task_type="detect",
                                execution_node_id=node_id, model_path="/remote/model.pt",
                                data_path="/remote/data.yaml", config={"remote_output_root": "/remote/runs"})
            db.add(task)
            db.flush()
            create_training_run(db, task, "fresh")
        db.commit()
    barrier = threading.Barrier(2)
    endpoint = f"/api/compute/nodes/{node_id}/agent/poll"

    def poll_together():
        barrier.wait(timeout=5)
        return client.post(endpoint, headers=headers(credential))

    # Force both requests to begin their claim at the same time. The node
    # claim lock must serialize them so the legacy Agent gets only one Run.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(poll_together) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]
    assert all(response.status_code == 200 for response in responses)
    claims = [response.json()["task"] for response in responses]
    assert sum(claim is not None for claim in claims) == 1
    with factory() as db:
        statuses = [run.status for run in db.scalars(select(TrainingRun))]
        assert sorted(statuses) == ["queued", "starting"]
