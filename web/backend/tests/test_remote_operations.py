"""Remote lifecycle integration with a real Agent and files isolated in tmp_path."""
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from urllib.error import HTTPError

import pytest
from sqlalchemy import event, select

import app.remote_operations as operations
from app.models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask, utcnow
from app.services import create_training_run
from test_compute_nodes import create, headers, monitoring_client
from test_remote_agent import agent, training


def remote_run(client, factory, tmp_path):
    credential = create(client)
    root = tmp_path / "output"
    root.mkdir()
    model, data = tmp_path / "model.pt", tmp_path / "data.yaml"
    model.write_bytes(b"base model")
    data.write_text("train: images/train\n")
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "1.3.0"
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
              "training_cwd": str(tmp_path), "yolo_bin": "/usr/bin/python3"}
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


def test_remote_delete_confirm_retry_and_local_isolation(monitoring_client, agent, tmp_path, monkeypatch):
    client, factory = monitoring_client
    credential, config, payload = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    task_id = payload["task_id"]
    task_url = f"/api/tasks/{task_id}"
    other_dir = tmp_path / "output/detect/another-task/attempt-001"
    other_dir.mkdir(parents=True)
    (other_dir / "keep.txt").write_text("untouched")
    assert client.post(task_url + "/trash").status_code == 204
    assert client.post(task_url + "/restore").status_code == 200
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
    assert client.delete(task_url + "/permanent").status_code == 202
    work = claim(client, credential)
    agent.execute_operation(config, work, threading.Event())
    assert not Path(payload["run_dir"]).exists() and not Path(payload["log_path"]).exists()
    assert (other_dir / "keep.txt").read_text() == "untouched"
    assert (tmp_path / "model.pt").read_bytes() == b"base model"
    assert (tmp_path / "data.yaml").is_file()
    with factory() as db:
        assert db.get(TrainingTask, task_id) is None
        assert not db.scalars(select(TrainingRun)).all()
    # Retry of a lost result ACK still succeeds after the task record is gone.
    result = {"lease": work["lease"], "success": True}
    endpoint = f"/api/compute/nodes/{credential['node']['id']}/agent/operations/{work['id']}/result"
    assert client.post(endpoint, json=result, headers=headers(credential)).json()["status"] == "succeeded"


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
    response = client.post(f"/api/tasks/{payload['task_id']}/resume")
    assert response.status_code == 200, response.text
    assert response.json()["completed_epochs"] == 3
    with factory() as db:
        new_run = db.get(TrainingRun, response.json()["current_run_id"])
        assert new_run.attempt_number == 2 and new_run.launch_mode == "resume"
        assert new_run.checkpoint_path == weight["path"]
        assert new_run.run_dir != payload["run_dir"]
    assert client.delete(f"/api/tasks/{payload['task_id']}/permanent").status_code == 409


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
        db.get(ComputeNode, node_id).agent_version = "1.3.0"
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
