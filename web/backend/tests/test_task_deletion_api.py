"""Whole-task deletion API, durable progress and non-mutating previews."""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import app.model_test_storage as storage
import app.services as services
from app.database import get_db
from app.deletion_jobs import list_attempt_deletion_jobs
from app.main import app
from app.models import ComputeNode, ModelTest, RemoteOperation, TrainingRun, TrainingTask, utcnow
from test_services import make_session, make_task


@pytest.fixture
def deletion_client(tmp_path, monkeypatch):
    monkeypatch.setattr(services, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(services, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(services, "LOG_ROOT", tmp_path / "log")
    monkeypatch.setattr(storage, "HISTORY_ROOT", tmp_path / "history")
    monkeypatch.setattr(storage, "WORK_ROOT", tmp_path / "work")
    with make_session(tmp_path) as db:
        task = make_task()
        db.add(task)
        db.commit()

        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        try:
            yield TestClient(app), db, task
        finally:
            app.dependency_overrides.pop(get_db, None)


def test_preview_is_read_only_and_delete_queues_atomic_intent(deletion_client):
    client, db, task = deletion_client
    preview = client.get(f"/api/tasks/{task.id}/deletion-preview")
    assert preview.status_code == 200
    assert preview.json()["can_delete"]
    assert preview.json()["attempt_count"] == 0
    assert task.deleted_at is None
    assert db.scalar(select(RemoteOperation)) is None

    response = client.delete(f"/api/tasks/{task.id}/permanent")
    assert response.status_code == 202
    operation_id = response.json()["operation_id"]
    operation = db.get(RemoteOperation, operation_id)
    assert operation.status == "pending"
    assert operation.payload["scope"] == "task"
    assert task.deleted_reason == "local_delete_pending"
    repeat = client.delete(f"/api/tasks/{task.id}/permanent")
    assert repeat.status_code == 202
    assert repeat.json()["operation_id"] == operation_id
    jobs = client.get("/api/task-deletions").json()
    assert len(jobs) == 1
    assert jobs[0]["executor"] == "local"
    assert jobs[0]["phase"] == "stopping_tests"
    assert jobs[0]["task_name"] == task.name


def test_invalid_plan_rejection_preserves_visibility_and_test_state(deletion_client):
    client, db, task = deletion_client
    task.run_dir = str(services.RUNS_ROOT / "unowned")
    record = ModelTest(id="test-upload", task_id=task.id, status="uploading",
                       artifact_name="best.pt", model_name="best.pt", confidence=0.25, iou=0.7)
    db.add(record)
    db.commit()
    preview = client.get(f"/api/tasks/{task.id}/deletion-preview").json()
    assert not preview["can_delete"]
    response = client.delete(f"/api/tasks/{task.id}/permanent")
    assert response.status_code == 409
    db.refresh(task)
    db.refresh(record)
    assert task.deleted_at is None
    assert record.status == "uploading"
    assert db.scalar(select(RemoteOperation)) is None


def test_completed_receipt_survives_task_removal_and_excludes_attempt_jobs(deletion_client):
    client, db, task = deletion_client
    task_id = task.id
    receipt = {"verified_target_count": 2, "test_history_removed": True}
    db.add_all([
        RemoteOperation(id="whole", node_id="__local__", task_id=task_id, kind="local_delete",
                        status="succeeded", payload={"scope": "task", "task_name": "已清理的任务", "receipt": receipt}),
        RemoteOperation(id="attempt", node_id="__local__", task_id=task_id, kind="local_delete",
                        status="succeeded", payload={"scope": "attempt", "run_id": "run"}),
    ])
    db.delete(task)
    db.commit()
    jobs = client.get("/api/task-deletions").json()
    assert len(jobs) == 1
    assert jobs[0]["task_name"] == "已清理的任务"
    assert jobs[0]["receipt"] == receipt
    assert jobs[0]["phase"] == "completed"
    assert not jobs[0]["retryable"]


def test_remote_center_cleanup_does_not_claim_agent_is_required(deletion_client):
    client, db, task = deletion_client
    node = ComputeNode(id="offline-node", name="测试服务器", kind="agent", enabled=True,
                       agent_version="2.0.4", report_interval_seconds=5,
                       last_seen_at=utcnow() - timedelta(hours=1))
    task.execution_node_id = node.id
    task.deleted_at = utcnow()
    task.deleted_reason = "remote_delete_failed"
    operation = RemoteOperation(id="center-retry", node_id=node.id, task_id=task.id,
                                kind="delete", status="failed", error="中心存储暂不可写",
                                payload={"scope": "task", "phase": "center_cleanup_failed",
                                         "remote_verified": True, "receipt": {"verified_absent": True}})
    db.add_all([node, operation])
    db.commit()
    jobs = client.get("/api/task-deletions").json()
    assert jobs[0]["node_status"] == "offline"
    assert jobs[0]["retryable"]
    assert jobs[0]["message"] == "中心存储暂不可写"
    assert client.get(f"/api/tasks/{task.id}/deletion-preview").json()["can_delete"]
    response = client.delete(f"/api/tasks/{task.id}/permanent")
    assert response.status_code == 202
    assert response.json()["operation_id"] == operation.id
    job = client.get("/api/task-deletions").json()[0]
    assert job["status"] == "finalizing"
    assert job["phase"] == "center_cleanup"
    assert "Agent" not in job["message"]


def test_history_limit_cannot_hide_latest_failure_of_trashed_task(deletion_client):
    client, db, task = deletion_client
    task.deleted_at = utcnow()
    task.deleted_reason = "local_delete_failed"
    task.error_message = "原来的训练中断原因"
    db.add(RemoteOperation(id="failed-current", node_id="__local__", task_id=task.id,
                           kind="local_delete", status="failed", error="文件暂不可写",
                           payload={"scope": "task"}, updated_at=utcnow() - timedelta(days=2)))
    for index in range(3):
        db.add(RemoteOperation(id=f"old-{index}", node_id="__local__", task_id=f"deleted-{index}",
                               kind="local_delete", status="succeeded", payload={"scope": "task"}))
    db.commit()
    jobs = client.get("/api/task-deletions?limit=1").json()
    assert len(jobs) == 2
    current = next(job for job in jobs if job["task_id"] == task.id)
    assert current["error"] == "文件暂不可写"
    assert task.error_message == "原来的训练中断原因"


def test_attempt_deletion_list_keeps_active_and_failed_jobs_outside_history_limit(deletion_client):
    _, db, task = deletion_client
    run = TrainingRun(
        id="attempt-list-run", task=task, attempt_number=3, status="completed",
        model_path=task.model_path, data_path=task.data_path,
    )
    db.add(run)
    db.flush()
    db.add_all([
        RemoteOperation(
            id="attempt-failed", node_id="__local__", task_id=task.id,
            kind="local_delete", status="failed", error="disk busy",
            payload={"scope": "attempt", "run_id": run.id, "attempt_number": 3},
            updated_at=utcnow() - timedelta(days=3),
        ),
        RemoteOperation(
            id="attempt-active", node_id="__local__", task_id="missing-active-task",
            kind="local_delete", status="running",
            payload={"scope": "attempt", "run_id": "missing-active-run", "attempt_number": 2},
            updated_at=utcnow() - timedelta(days=2),
        ),
        *[
            RemoteOperation(
                id=f"attempt-history-{index}", node_id="__local__", task_id=f"history-task-{index}",
                kind="local_delete", status="succeeded",
                payload={"scope": "attempt", "run_id": f"history-run-{index}", "attempt_number": index + 1},
                updated_at=utcnow() - timedelta(hours=index),
            )
            for index in range(3)
        ],
        RemoteOperation(
            id="whole-task-history", node_id="__local__", task_id=task.id,
            kind="local_delete", status="failed", payload={"scope": "task"},
        ),
    ])
    db.commit()

    jobs = list_attempt_deletion_jobs(db, limit=1)
    assert {job["id"] for job in jobs} == {
        "attempt-failed", "attempt-active", "attempt-history-0",
    }
    failed = next(job for job in jobs if job["id"] == "attempt-failed")
    assert failed["scope"] == "attempt"
    assert failed["run_id"] == run.id
    assert failed["attempt_number"] == 3
    assert failed["retryable"] is True


@pytest.mark.parametrize(
    ("layout", "agent_version", "required_version", "upgrade_required"),
    [
        ("canonical", "1.8.0", "1.8.0", False),
        ("legacy_flat", "1.8.0", "2.0.1", True),
    ],
)
def test_attempt_deletion_list_uses_attempt_agent_requirement(
    deletion_client, layout, agent_version, required_version, upgrade_required,
):
    _, db, task = deletion_client
    node = ComputeNode(
        id=f"attempt-node-{layout}", name="Attempt node", kind="agent", enabled=True,
        agent_version=agent_version, report_interval_seconds=5, last_seen_at=utcnow(), snapshot={},
    )
    task.execution_node_id = node.id
    run = TrainingRun(
        id=f"attempt-run-{layout}", task=task, attempt_number=1, status="completed",
        execution_node_id=node.id, model_path=task.model_path, data_path=task.data_path,
    )
    operation = RemoteOperation(
        id=f"attempt-operation-{layout}", node_id=node.id, task_id=task.id,
        kind="delete", status="pending",
        payload={
            "scope": "attempt", "run_id": run.id, "attempt_number": 1,
            "runs": [{"layout": layout}],
        },
    )
    db.add_all([node, run, operation])
    db.commit()

    job = next(item for item in list_attempt_deletion_jobs(db) if item["id"] == operation.id)
    assert job["required_agent_version"] == required_version
    assert (job["phase"] == "agent_upgrade_required") is upgrade_required
    if not upgrade_required:
        assert "完整任务清理" not in job["message"]
