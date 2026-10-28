"""Configured and actual Batch values remain separate across old and live Attempts."""
import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException
from starlette.requests import Request
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import app.database as database_module
import app.compute_nodes as compute_nodes
import app.remote_operations as remote_operations
from app.batch_history import BatchLogReader, append_reductions, scan_batch_log
from app.database import Base
from app.models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask
from app.services import mirror_run_to_task, sync_local_batch_history


FIRST = "CUDA out of memory with batch=10. Reducing to batch=5 and retrying (1/3).\n"
SECOND = "CUDA out of memory with batch=5. Reducing to batch=2 and retrying (2/3).\n"


def agent_module():
    path = Path(__file__).resolve().parents[2] / "agent/monitor_agent.py"
    spec = importlib.util.spec_from_file_location("batch_history_agent_test", path)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    return agent


def test_legacy_log_recovers_complete_reduction_chain(tmp_path):
    path = tmp_path / "attempt.log"
    path.write_text("Starting training for 100 epochs...\n" + FIRST + "more progress\n" + SECOND,
                    encoding="utf-8")
    assert scan_batch_log(path, 10) == ([10, 5, 2], True)
    assert append_reductions([], [SECOND], 10) == []  # Tail cannot prove the missing first step.
    assert scan_batch_log(tmp_path / "missing.log", 10) == ([], False)


def test_live_log_updates_each_complete_step_without_reading_future_partial_line(tmp_path):
    path = tmp_path / "attempt.log"
    path.write_text("startup\n" + FIRST, encoding="utf-8")
    reader = BatchLogReader()
    assert reader.read(path, 10) == [10, 5]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(SECOND[:-1])
    assert reader.read(path, 10) == [10, 5]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert reader.read(path, 10) == [10, 5, 2]
    assert reader.read(path, 10) == [10, 5, 2]


def test_terminal_local_scan_keeps_last_warning_without_newline(tmp_path):
    path = tmp_path / "attempt.log"
    path.write_text(FIRST + SECOND[:-1], encoding="utf-8")
    task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml", batch=10)
    run = TrainingRun(id="run", task_id="task", attempt_number=1, model_path="model.pt",
                      data_path="data.yaml", parameter_snapshot={"batch": 10}, log_path=str(path),
                      batch_history=[10, 5])
    task.current_run_id = run.id
    sync_local_batch_history(task, run, final=True)
    assert run.batch_history == task.batch_history == [10, 5, 2]
    assert run.batch_history_scanned is True


def test_agent_reports_batch_only_to_negotiated_center_and_scans_old_log(tmp_path):
    agent = agent_module()
    path = tmp_path / "remote.log"
    path.write_text(FIRST, encoding="utf-8")
    task = {"log_path": str(path), "parameters": {"batch": 10}}
    assert "batch_history" not in agent.training_report_payload(task, {"status": "running"})
    task["batch_history_version"] = 1
    assert agent.training_report_payload(task, {"status": "running"})["batch_history"] == [10, 5]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(SECOND)
    assert agent.training_report_payload(task, {"status": "running"})["batch_history"] == [10, 5, 2]
    assert agent.read_batch_history(path, 10, full=True) == [10, 5, 2]


def test_new_attempt_mirror_drops_previous_attempt_actual_batch():
    task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml",
                        batch=10, batch_history=[10, 5, 2])
    run = TrainingRun(id="new", task_id="task", attempt_number=2, model_path="model.pt",
                      data_path="data.yaml", parameter_snapshot={"batch": 10})
    mirror_run_to_task(task, run)
    assert task.current_run_id == "new"
    assert task.batch == 10  # Requested configuration is untouched.
    assert task.batch_history == []


def test_old_database_adds_batch_columns_without_losing_runs(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml",
                            batch=10, status="completed")
        run = TrainingRun(id="run", task=task, attempt_number=1, model_path="model.pt",
                          data_path="data.yaml", parameter_snapshot={"batch": 10}, status="completed")
        db.add_all([task, run])
        db.commit()
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE training_tasks DROP COLUMN batch_history"))
        connection.execute(text("ALTER TABLE training_runs DROP COLUMN batch_history"))
        connection.execute(text("ALTER TABLE training_runs DROP COLUMN batch_history_scanned"))
    monkeypatch.setattr(database_module, "engine", engine)
    assert database_module.migrate_database() is not None
    assert database_module.migrate_database() is None
    assert {"batch_history"} <= {column["name"] for column in inspect(engine).get_columns("training_tasks")}
    assert {"batch_history", "batch_history_scanned"} <= {
        column["name"] for column in inspect(engine).get_columns("training_runs")}
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT batch FROM training_tasks WHERE id='task'")) == 10
        assert connection.scalar(text("SELECT status FROM training_runs WHERE id='run'")) == "completed"
    engine.dispose()


def test_remote_live_update_and_historical_scan_are_attempt_scoped(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'batch-remote.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    root = tmp_path / "runs"
    node = ComputeNode(id="node", name="Remote", kind="agent", agent_version="2.5.3")
    task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml",
                        execution_node_id=node.id, config={"remote_output_root": str(root)}, batch=10)
    run = TrainingRun(id="run", task=task, attempt_number=1, model_path="model.pt", data_path="data.yaml",
                      execution_node_id=node.id, parameter_snapshot={"batch": 10, "remote_output_root": str(root)},
                      run_dir=str(root / "detect/task/attempt-001"),
                      log_path=str(root / ".platform-logs/20261028/task/attempt-001.log"), status="running")
    task.current_run_id = run.id
    with factory() as db:
        db.add_all([node, task, run])
        db.commit()

    monkeypatch.setattr(compute_nodes, "authenticated_agent", lambda *_args: node)
    monkeypatch.setattr(remote_operations, "authenticated_agent", lambda *_args: node)
    request = Request({"type": "http", "headers": []})
    with factory() as db:
        first = compute_nodes.RemoteRunUpdate(status="running", batch_history=[10, 5])
        compute_nodes.update_remote_run(node.id, run.id, first, request, db)
        assert db.get(TrainingRun, run.id).batch_history == [10, 5]
        assert db.get(TrainingTask, task.id).batch_history == [10, 5]
        second = compute_nodes.RemoteRunUpdate(status="running", batch_history=[10, 5, 2])
        compute_nodes.update_remote_run(node.id, run.id, second, request, db)
        assert db.get(TrainingRun, run.id).batch_history == [10, 5, 2]
        with pytest.raises(HTTPException) as invalid:
            compute_nodes.update_remote_run(node.id, run.id,
                                            compute_nodes.RemoteRunUpdate(status="running", batch_history=[8, 4]),
                                            request, db)
        assert invalid.value.status_code == 422
        compute_nodes.update_remote_run(node.id, run.id,
                                        compute_nodes.RemoteRunUpdate(status="failed", batch_history=[10, 5, 2]),
                                        request, db)
        assert db.get(TrainingRun, run.id).batch_history_scanned is True
    with factory() as db:
        db.get(TrainingRun, run.id).batch_history_scanned = False
        db.commit()
        assert remote_operations.queue_batch_history_scan(db, node)
        operation = db.query(RemoteOperation).filter_by(kind="batch_history").one()
        assert operation.payload["run_id"] == run.id
        assert not remote_operations.queue_batch_history_scan(db, node)
        operation.lease = "a" * 32
        operation.status = "running"
        db.commit()
        result = remote_operations.OperationResult(lease=operation.lease, success=True,
                                                   batch_history=[10, 5, 2])
        assert remote_operations.operation_result(node.id, operation.id, result, request, db)["status"] == "succeeded"
        assert db.get(TrainingRun, run.id).batch_history_scanned is True
        assert db.get(TrainingTask, task.id).batch_history == [10, 5, 2]
    engine.dispose()


def test_historical_scan_waits_for_new_agent_and_yields_to_user_operations(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'batch-queue.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    root = tmp_path / "runs"
    node = ComputeNode(id="node", name="Remote", kind="agent", agent_version="2.5.1")
    task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml",
                        execution_node_id=node.id, config={"remote_output_root": str(root)}, batch=10)
    run = TrainingRun(id="run", task=task, attempt_number=1, model_path="model.pt", data_path="data.yaml",
                      execution_node_id=node.id, parameter_snapshot={"batch": 10, "remote_output_root": str(root)},
                      run_dir=str(root / "detect/task/attempt-001"),
                      log_path=str(root / ".platform-logs/20261028/task/attempt-001.log"), status="completed")
    with factory() as db:
        db.add_all([node, task, run])
        db.commit()
        assert not remote_operations.queue_batch_history_scan(db, node)
        assert db.query(RemoteOperation).count() == 0
        node.agent_version = "2.5.3"
        assert remote_operations.queue_batch_history_scan(db, node)
        user_operation = remote_operations.enqueue(db, node.id, task.id, "scan", {"task_id": task.id})
        db.commit()
        monkeypatch.setattr(remote_operations, "authenticated_agent", lambda *_args: node)
        picked = remote_operations.poll_operations(node.id, Request({"type": "http", "headers": []}), db)
        assert picked["operation"]["id"] == user_operation.id
        node.agent_version = "2.5.1"
        db.commit()
        picked = remote_operations.poll_operations(node.id, Request({"type": "http", "headers": []}), db)
        assert picked["operation"] is None  # Pending new-only work is never sent to old Agents.
    engine.dispose()


def test_upgraded_agent_recovers_active_old_run_then_tail_extends_path(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'active-upgrade.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    root = tmp_path / "runs"
    node = ComputeNode(id="node", name="Remote", kind="agent", agent_version="2.5.1")
    task = TrainingTask(id="task", name="smoke", model_path="model.pt", data_path="data.yaml",
                        execution_node_id=node.id, config={"remote_output_root": str(root)}, batch=10)
    run = TrainingRun(id="run", task=task, attempt_number=1, model_path="model.pt", data_path="data.yaml",
                      execution_node_id=node.id, parameter_snapshot={"batch": 10, "remote_output_root": str(root)},
                      run_dir=str(root / "detect/task/attempt-001"),
                      log_path=str(root / ".platform-logs/20261028/task/attempt-001.log"), status="running")
    task.current_run_id = run.id
    with factory() as db:
        db.add_all([node, task, run])
        db.commit()
        assert not remote_operations.queue_batch_history_scan(db, node)
        node.agent_version = "2.5.3"
        assert remote_operations.queue_batch_history_scan(db, node)
        operation = db.query(RemoteOperation).filter_by(kind="batch_history").one()
        operation.lease, operation.status = "a" * 32, "running"
        db.commit()
        request = Request({"type": "http", "headers": []})
        monkeypatch.setattr(remote_operations, "authenticated_agent", lambda *_args: node)
        monkeypatch.setattr(compute_nodes, "authenticated_agent", lambda *_args: node)
        result = remote_operations.OperationResult(lease=operation.lease, success=True, batch_history=[10, 5])
        remote_operations.operation_result(node.id, operation.id, result, request, db)
        assert db.get(TrainingRun, run.id).batch_history == [10, 5]
        assert db.get(TrainingTask, task.id).batch_history == [10, 5]
        assert db.get(TrainingRun, run.id).batch_history_scanned is False
        update = compute_nodes.RemoteRunUpdate(status="running", log_tail=[SECOND])
        compute_nodes.update_remote_run(node.id, run.id, update, request, db)
        assert db.get(TrainingRun, run.id).batch_history == [10, 5, 2]
        assert db.get(TrainingTask, task.id).batch_history == [10, 5, 2]
        # A later truncated scan can only confirm existing knowledge.
        operation2 = remote_operations.enqueue(db, node.id, task.id, "batch_history", operation.payload)
        operation2.lease, operation2.status = "b" * 32, "running"
        db.commit()
        remote_operations.operation_result(node.id, operation2.id,
                                           remote_operations.OperationResult(lease=operation2.lease,
                                                                             success=True, batch_history=[]),
                                           request, db)
        assert db.get(TrainingRun, run.id).batch_history == [10, 5, 2]
        assert db.get(TrainingTask, task.id).batch_history == [10, 5, 2]
    engine.dispose()
