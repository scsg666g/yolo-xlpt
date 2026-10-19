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
from app.models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask, utcnow
from app.parameter_catalog import training_parameter_keys
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
    def api(config, path, method="GET", payload=None):
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
    def unreliable_api(config, path, method="GET", payload=None):
        nonlocal lost
        response = api(config, path, method, payload)
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
    waiting, release = threading.Event(), threading.Event()
    engine = factory.kw["bind"]

    def delay_first_queue_read(conn, cursor, statement, parameters, context, executemany):
        # Pause after the occupancy check. Another poll claims the first task,
        # so this poll subsequently sees the second queued task.
        if "ORDER BY training_runs.queued_at" in statement and not waiting.is_set():
            waiting.set()
            assert release.wait(5), "concurrent poll did not finish"

    event.listen(engine, "before_cursor_execute", delay_first_queue_read)
    endpoint = f"/api/compute/nodes/{node_id}/agent/poll"
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            delayed = executor.submit(client.post, endpoint, headers=headers(credential))
            try:
                assert waiting.wait(5)
                first = client.post(endpoint, headers=headers(credential))
                assert first.status_code == 200 and first.json()["task"]
            finally:
                release.set()
            second = delayed.result(timeout=5)
            assert second.status_code == 200 and second.json()["task"] is None
    finally:
        release.set()
        event.remove(engine, "before_cursor_execute", delay_first_queue_read)
    with factory() as db:
        statuses = [run.status for run in db.scalars(select(TrainingRun))]
        assert sorted(statuses) == ["queued", "starting"]
