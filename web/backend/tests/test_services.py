from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
import yaml
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

import app.config as config_module
import app.database as database_module
import app.local_deletions as local_deletions_module
import app.model_test_storage as model_test_storage
import app.services as services_module
from app.database import Base, configure_sqlite_connection
from app.database import get_db
from app.main import app
import app.main as main_module
from app.local_deletions import LocalDeletionWorker
from app.models import Artifact, MetricPoint, TaskEvent, TrainingRun, TrainingTask, utcnow
from app.schemas import TaskConfigUpdate, TaskCreate, TaskRead
from app.services import (
    TrainingScheduler,
    backfill_training_runs,
    create_training_run,
    failure_detail,
    import_existing_runs,
    migrate_training_logs,
    mirror_run_to_task,
    path_is_within,
    permanently_delete_task_files,
    prepare_dataset_config,
    process_matches_training,
    resume_checkpoint,
    sync_artifacts,
    sync_metrics,
    training_environment,
)


def make_session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)()


def make_task(task_id: str = "task-test", **values):
    defaults = dict(
        id=task_id,
        name="test",
        task_type="detect",
        model_path="model.pt",
        data_path="data.yaml",
        config={},
    )
    defaults.update(values)
    return TrainingTask(**defaults)


@pytest.mark.parametrize("status", ["starting", "running", "stopping"])
@pytest.mark.parametrize("node_id,active_device,requested_device,available", [
    ("remote-server", "0", "0", True),
    (None, "0", "0", False),
    (None, "0", "1", True),
    (None, "0,1", "1", False),
    (None, "0", "cpu", True),
    (None, "cpu", "0", True),
])
def test_local_scheduler_gpu_occupancy_is_scoped_to_local_tasks(
    tmp_path: Path, status, node_id, active_device, requested_device, available,
):
    with make_session(tmp_path) as session:
        queued = make_task("queued-local", status="queued", device=requested_device)
        active = make_task("active-task", status=status, execution_node_id=node_id, device=active_device)
        session.add_all([queued, active])
        session.commit()

        assert TrainingScheduler()._available(session, queued) is available


def test_scheduler_claim_is_atomic_across_scheduler_instances(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'atomic-claim.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as db:
        task = make_task("task-atomic-claim", status="queued")
        run = TrainingRun(
            id="run-atomic-claim", task=task, attempt_number=1, launch_mode="fresh", status="queued",
            model_path=task.model_path, data_path=task.data_path, parameter_snapshot={"epochs": 10},
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    first = TrainingScheduler(session_factory=session_factory)
    second = TrainingScheduler(session_factory=session_factory)
    with session_factory() as db:
        assert first._claim_queued_run(db, "run-atomic-claim")
    with session_factory() as db:
        assert not second._claim_queued_run(db, "run-atomic-claim")
        assert db.get(TrainingRun, "run-atomic-claim").status == "starting"
        assert db.get(TrainingTask, "task-atomic-claim").status == "starting"


def test_popen_happens_after_durable_claim_and_final_db_failure_is_recoverable(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'launch-recovery.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    run_dir = tmp_path / "runs" / "attempt-001"
    log_path = tmp_path / "logs" / "attempt-001.log"
    with session_factory() as db:
        task = make_task("task-launch-recovery", status="starting")
        run = TrainingRun(
            id="run-launch-recovery", task=task, attempt_number=1, launch_mode="fresh", status="starting",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir), log_path=str(log_path),
            parameter_snapshot={"epochs": 10, "config": {}},
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    command = ["fake-yolo", "detect", "train"]
    monkeypatch.setattr(scheduler, "_command", lambda *_args: command)
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(services_module, "prepare_dataset_config", lambda _raw, destination: destination)

    class FakeProcess:
        pid = 4312

        def __init__(self, *_args, **_kwargs):
            with session_factory() as observer:
                persisted = observer.get(TrainingRun, "run-launch-recovery")
                assert persisted.status == "starting"
                assert persisted.command == command

        def poll(self):
            return None

    monkeypatch.setattr(services_module.subprocess, "Popen", FakeProcess)
    with session_factory() as db:
        run = db.get(TrainingRun, "run-launch-recovery")
        task = db.get(TrainingTask, "task-launch-recovery")
        real_commit = db.commit
        commits = 0

        def fail_final_commit():
            nonlocal commits
            commits += 1
            if commits == 2:
                raise RuntimeError("simulated final commit failure")
            real_commit()

        monkeypatch.setattr(db, "commit", fail_final_commit)
        with pytest.raises(RuntimeError, match="simulated final commit failure"):
            scheduler._launch(db, task, run)
        db.rollback()

    with session_factory() as db:
        persisted = db.get(TrainingRun, "run-launch-recovery")
        assert persisted.status == "starting"
        assert persisted.command == command
        assert not TrainingScheduler(session_factory=session_factory)._claim_queued_run(db, persisted.id)

    with session_factory() as db:
        scheduler._poll_processes(db)
        db.commit()
        assert db.get(TrainingRun, "run-launch-recovery").status == "running"
        assert db.get(TrainingRun, "run-launch-recovery").pid == FakeProcess.pid


def test_stop_winning_between_prepared_commit_and_popen_return_terminates_process(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'stop-launch-race.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    run_dir = tmp_path / "runs" / "attempt-001"
    with session_factory() as db:
        task = make_task("task-stop-launch-race", status="queued")
        run = TrainingRun(
            id="run-stop-launch-race", task=task, attempt_number=1, launch_mode="fresh", status="queued",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir),
            log_path=str(tmp_path / "logs" / "attempt-001.log"), parameter_snapshot={"epochs": 10, "config": {}},
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    with session_factory() as db:
        assert scheduler._claim_queued_run(db, "run-stop-launch-race")
    monkeypatch.setattr(scheduler, "_command", lambda *_args: ["fake-yolo", "train"])
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(services_module, "prepare_dataset_config", lambda _raw, destination: destination)
    signals = []
    monkeypatch.setattr(services_module.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal)))

    class FakeProcess:
        pid = 5432

        def __init__(self, *_args, **_kwargs):
            assert scheduler.stop_task("task-stop-launch-race")

        def poll(self):
            return -15

    monkeypatch.setattr(services_module.subprocess, "Popen", FakeProcess)
    with session_factory() as db:
        scheduler._launch(
            db, db.get(TrainingTask, "task-stop-launch-race"), db.get(TrainingRun, "run-stop-launch-race"),
        )

    assert scheduler.stop_task("task-stop-launch-race")  # repeated requests are idempotent
    assert signals == [(FakeProcess.pid, services_module.signal.SIGTERM),
                       (FakeProcess.pid, services_module.signal.SIGTERM)]
    with session_factory() as db:
        assert db.get(TrainingRun, "run-stop-launch-race").status == "stopping"
        scheduler._poll_processes(db)
        db.commit()
        assert db.get(TrainingRun, "run-stop-launch-race").status == "stopped"
        assert db.get(TrainingTask, "task-stop-launch-race").status == "stopped"


def test_stop_rereads_process_registered_during_status_transition(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'stop-process-registration-race.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as db:
        task = make_task("task-stop-registration-race", status="running", pid=6543)
        run = TrainingRun(
            id="run-stop-registration-race", task=task, attempt_number=1, launch_mode="fresh", status="running",
            model_path=task.model_path, data_path=task.data_path, pid=6543,
            parameter_snapshot={"epochs": 10, "config": {}},
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    managed = services_module.ManagedTrainingProcess(
        run_id="run-stop-registration-race", pid=6543, command=["fake-yolo"], started_at=utcnow(),
    )
    db = session_factory()
    original_execute = db.execute
    execute_calls = 0

    def register_between_status_update_and_signal(*args, **kwargs):
        nonlocal execute_calls
        result = original_execute(*args, **kwargs)
        execute_calls += 1
        if execute_calls == 2:
            with scheduler._process_lock:
                scheduler.processes["task-stop-registration-race"] = managed
        return result

    monkeypatch.setattr(db, "execute", register_between_status_update_and_signal)
    scheduler.session_factory = lambda: db
    signals = []
    monkeypatch.setattr(services_module.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal)))

    assert scheduler.stop_task("task-stop-registration-race")
    assert signals == [(managed.pid, services_module.signal.SIGTERM)]


def test_process_match_requires_command_and_start_time(monkeypatch):
    started_at = utcnow()
    monkeypatch.setattr(
        services_module, "_process_identity",
        lambda _pid: (["python", "train.py", "task-1"], started_at),
    )

    assert process_matches_training(123, ["python", "train.py", "task-1"], started_at)
    assert not process_matches_training(123, ["python", "train.py", "task-2"], started_at)
    assert not process_matches_training(123, ["python", "train.py", "task-1"], started_at.replace(year=2025))


def test_scheduler_recovers_training_process_after_restart(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'recovery.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    started_at = utcnow()
    with session_factory() as db:
        task = make_task("task-recovery", status="interrupted", pid=None)
        run = TrainingRun(
            id="run-recovery", task=task, attempt_number=2, launch_mode="retry", status="interrupted",
            model_path=task.model_path, data_path=task.data_path, command=["python", "train.py", "task-recovery"],
            parameter_snapshot={"epochs": 20}, process_started_at=started_at,
            error_message="平台服务重启后未发现受管训练进程",
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    monkeypatch.setattr(scheduler, "_find_run_process", lambda _run: 4321)
    scheduler._recover_processes()

    with session_factory() as db:
        recovered = db.get(TrainingRun, "run-recovery")
        recovered_task = db.get(TrainingTask, "task-recovery")
        event = db.scalar(select(TaskEvent).where(TaskEvent.task_id == "task-recovery", TaskEvent.event_type == "reattached"))
        assert (recovered.status, recovered.pid, recovered.error_message) == ("running", 4321, None)
        assert (recovered_task.status, recovered_task.pid) == ("running", 4321)
        assert event is not None
    assert scheduler.processes["task-recovery"].adopted
    assert scheduler.processes["task-recovery"].pid == 4321


def test_scheduler_resumes_pending_stop_after_restart(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'recover-stopping.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    started_at = utcnow()
    with session_factory() as db:
        task = make_task("task-recover-stopping", status="stopping", pid=4321)
        run = TrainingRun(
            id="run-recover-stopping", task=task, attempt_number=1, launch_mode="fresh", status="stopping", pid=4321,
            model_path=task.model_path, data_path=task.data_path,
            command=["python", "train.py", "task-recover-stopping"],
            parameter_snapshot={"epochs": 20}, process_started_at=started_at,
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    signals = []
    scheduler = TrainingScheduler(session_factory=session_factory)
    monkeypatch.setattr(scheduler, "_find_run_process", lambda _run: 4321)
    monkeypatch.setattr(services_module.os, "killpg", lambda pid, sent_signal: signals.append((pid, sent_signal)))

    scheduler._recover_processes()

    with session_factory() as db:
        recovered = db.get(TrainingRun, "run-recover-stopping")
        recovered_task = db.get(TrainingTask, "task-recover-stopping")
        assert (recovered.status, recovered.pid) == ("stopping", 4321)
        assert (recovered_task.status, recovered_task.pid) == ("stopping", 4321)
    assert signals == [(4321, services_module.signal.SIGTERM)]
    assert scheduler.processes["task-recover-stopping"].adopted


def test_scheduler_marks_missing_active_process_interrupted(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'missing-process.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    run_dir = tmp_path / "run-missing"
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"completed-checkpoint")
    with session_factory() as db:
        task = make_task("task-missing", status="running", pid=9876, run_dir=str(run_dir))
        run = TrainingRun(
            id="run-missing", task=task, attempt_number=1, launch_mode="fresh", status="running", pid=9876,
            model_path=task.model_path, data_path=task.data_path, command=["python", "train.py", "task-missing"],
            parameter_snapshot={"epochs": 20}, process_started_at=utcnow(), run_dir=str(run_dir),
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    monkeypatch.setattr(scheduler, "_find_run_process", lambda _run: None)
    scheduler._recover_processes()

    with session_factory() as db:
        missing = db.get(TrainingRun, "run-missing")
        assert missing.status == "interrupted"
        assert missing.pid is None
        assert missing.error_message == "平台服务重启后未发现受管训练进程"
        artifact = db.scalar(select(Artifact).where(Artifact.run_id == missing.id, Artifact.name == "weights/best.pt"))
        assert artifact is not None
        assert artifact.path == str(checkpoint.resolve())
    assert "task-missing" not in scheduler.processes


def test_scheduler_reconciles_artifacts_for_previously_interrupted_run(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'interrupted-artifacts.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    run_dir = tmp_path / "run-interrupted"
    checkpoint = run_dir / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"completed-checkpoint")
    with session_factory() as db:
        task = make_task(
            "task-interrupted-artifacts", status="interrupted", run_dir=str(run_dir),
            error_message="平台服务重启后未发现受管训练进程",
        )
        run = TrainingRun(
            id="run-interrupted-artifacts", task=task, attempt_number=1, launch_mode="fresh",
            status="interrupted", model_path=task.model_path, data_path=task.data_path,
            command=["python", "train.py", task.id], parameter_snapshot={"epochs": 20},
            process_started_at=utcnow(), run_dir=str(run_dir),
            error_message="平台服务重启后未发现受管训练进程",
        )
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()

    scheduler = TrainingScheduler(session_factory=session_factory)
    monkeypatch.setattr(scheduler, "_find_run_process", lambda _run: None)
    scheduler._recover_processes()

    with session_factory() as db:
        artifact = db.scalar(select(Artifact).where(Artifact.run_id == "run-interrupted-artifacts"))
        assert artifact is not None
        assert artifact.name == "weights/best.pt"
        assert artifact.path == str(checkpoint.resolve())


@pytest.mark.parametrize("completed_epochs,expected_status", [(3, "interrupted"), (20, "completed")])
def test_adopted_process_exit_uses_recorded_epochs_without_faking_exit_code(
    tmp_path, completed_epochs, expected_status,
):
    session = make_session(tmp_path)
    task = make_task("task-adopted-exit", status="running", completed_epochs=completed_epochs)
    run = TrainingRun(
        id="run-adopted-exit", task=task, attempt_number=1, launch_mode="fresh", status="running",
        model_path=task.model_path, data_path=task.data_path, completed_epochs=completed_epochs,
        parameter_snapshot={"epochs": 20}, process_started_at=utcnow(),
    )
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()
    scheduler = TrainingScheduler()
    scheduler.processes[task.id] = SimpleNamespace(run_id=run.id, adopted=True, poll=lambda: 0)

    scheduler._poll_processes(session)

    assert run.status == expected_status
    assert run.exit_code is None
    assert run.pid is None
    if expected_status == "interrupted":
        assert "无法取得原进程退出码" in run.error_message
    session.close()


def test_sqlite_connection_uses_wal_and_bounded_busy_timeout(tmp_path: Path):
    connection = sqlite3.connect(tmp_path / "sqlite-pragmas.db")
    try:
        configure_sqlite_connection(connection)
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
    finally:
        connection.close()


def test_rename_task_preserves_training_state(tmp_path: Path):
    session = make_session(tmp_path)
    task = make_task(
        "task-rename", name="火", config={"lr0": 0.001, "_resume_checkpoint": "last.pt"},
        run_dir="existing-run", log_path="existing.log", completed_epochs=20,
        latest_metrics={"metrics/mAP50(B)": 0.6508},
    )
    run = TrainingRun(
        id="rename-run", task=task, attempt_number=1, launch_mode="fresh", status="completed",
        model_path=task.model_path, data_path=task.data_path,
        parameter_snapshot={"epochs": 100, "name": "火"},
    )
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        for status in ("draft", "running", "completed"):
            task.status = status
            session.commit()
            session.refresh(task)
            before = TaskRead.model_validate(task).model_dump(mode="json")
            response = client.patch(f"/api/tasks/{task.id}/name", json={"name": f"  火焰训练 {status}  "})
            assert response.status_code == 200, response.text
            after = response.json()
            assert after["name"] == f"火焰训练 {status}"
            assert {k: v for k, v in after.items() if k not in {"name", "updated_at"}} == {
                k: v for k, v in before.items() if k not in {"name", "updated_at"}
            }
            session.expire_all()
            assert session.get(TrainingTask, task.id).name == after["name"]
            assert run.parameter_snapshot == {"epochs": 100, "name": "火"}
        events = session.scalars(select(TaskEvent).where(TaskEvent.task_id == task.id)).all()
        assert len(events) == 3
        assert events[0].payload == {"old_name": "火", "new_name": "火焰训练 draft"}
        # Saving the current name does not add another rename event.
        assert client.patch(f"/api/tasks/{task.id}/name", json={"name": task.name}).status_code == 200
        assert len(session.scalars(select(TaskEvent).where(TaskEvent.task_id == task.id)).all()) == 3
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()


def test_rename_rejects_invalid_names_and_missing_tasks(tmp_path: Path):
    session = make_session(tmp_path)
    task = make_task("task-invalid-rename", name="火")
    session.add(task)
    session.commit()

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        for payload in ({"name": ""}, {"name": " \t\n "}, {"name": "火" * 161},
                        {"name": None}, {"name": 123}, {"name": "新名称", "epochs": 1}):
            assert client.patch(f"/api/tasks/{task.id}/name", json=payload).status_code == 422
        assert task.name == "火"
        assert client.patch("/api/tasks/missing/name", json={"name": "新名称"}).status_code == 404
        task.deleted_at = utcnow()
        session.commit()
        assert client.patch(f"/api/tasks/{task.id}/name", json={"name": "新名称"}).status_code == 404
        assert session.scalar(select(TaskEvent).where(TaskEvent.task_id == task.id)) is None
    finally:
        app.dependency_overrides.pop(get_db, None)
        session.close()


@pytest.mark.parametrize("field,value", [
    ("name", "  \t "),
    ("optimizer", "   "),
    ("device", "0,00"),
])
def test_task_create_rejects_blank_text_and_duplicate_gpus(field, value):
    payload = {"name": "training", "model_path": "model.pt", "data_path": "data.yaml", field: value}
    with pytest.raises(ValidationError):
        TaskCreate(**payload)


@pytest.mark.parametrize("field,value", [("optimizer", " \t"), ("device", "1,1")])
def test_task_config_rejects_blank_optimizer_and_duplicate_gpus(field, value):
    payload = {"epochs": 10, "batch": 2, "imgsz": 640, "device": "0", "workers": 1, "optimizer": "auto"}
    payload[field] = value
    with pytest.raises(ValidationError):
        TaskConfigUpdate(**payload)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_task_create_rejects_nonfinite_lr0(value):
    with pytest.raises(ValidationError):
        TaskCreate(name="training", model_path="model.pt", data_path="data.yaml", lr0=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 10**400])
def test_task_config_rejects_nonfinite_custom_numbers(tmp_path: Path, value):
    session = make_session(tmp_path)
    task = make_task("task-custom-nonfinite", status="draft")
    session.add(task)
    session.commit()
    payload = TaskConfigUpdate(
        epochs=10, batch=2, imgsz=640, device="0", workers=1, optimizer="auto",
        config={}, custom_config={"custom_value": value},
    )

    with pytest.raises(HTTPException) as error:
        main_module.update_task_config(task.id, payload, session)

    assert error.value.status_code == 422
    session.close()


@pytest.mark.parametrize("changed_field", ["epochs", "data_path"])
def test_local_resume_rejects_checkpoint_snapshot_mismatch(tmp_path: Path, changed_field: str):
    session = make_session(tmp_path)
    run_dir = tmp_path / "attempt-001"
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    task = make_task(
        "task-resume-mismatch", status="stopped", epochs=20, completed_epochs=5,
        data_path=str(tmp_path / "original-data.yaml"), run_dir=str(run_dir),
    )
    run = TrainingRun(
        id="run-resume-mismatch", task=task, attempt_number=1, launch_mode="fresh", status="stopped",
        model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir),
        parameter_snapshot={"epochs": 20, "config": {}}, completed_epochs=5,
    )
    task.current_run_id = run.id
    if changed_field == "epochs":
        task.epochs = 21
    else:
        task.data_path = str(tmp_path / "replacement-data.yaml")
    session.add_all([task, run])
    session.commit()

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).post(f"/api/tasks/{task.id}/resume")
        assert response.status_code == 409
        assert "上一 Attempt" in response.json()["detail"]
        assert len(session.scalars(select(TrainingRun).where(TrainingRun.task_id == task.id)).all()) == 1
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_path_is_within(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert path_is_within(allowed / "model.pt", [allowed])
    assert not path_is_within(tmp_path / "outside.pt", [allowed])


def test_discover_files_prunes_hidden_directories_before_traversal(tmp_path: Path, monkeypatch):
    visible = tmp_path / "visible"
    visible.mkdir()
    (visible / "data.yaml").write_text("visible", encoding="utf-8")
    for name in (".platform-upload-abc", ".platform-deletions"):
        hidden = tmp_path / name
        hidden.mkdir()
        (hidden / "data.yaml").write_text("hidden", encoding="utf-8")

    real_scandir = services_module.os.scandir

    def reject_hidden_traversal(path):
        assert not Path(path).name.startswith("."), f"hidden directory was traversed: {path}"
        return real_scandir(path)

    monkeypatch.setattr(services_module.os, "scandir", reject_hidden_traversal)
    assert services_module.discover_files([tmp_path], {".yaml"}) == [{
        "name": "data.yaml",
        "path": str(visible / "data.yaml"),
        "size": len("visible"),
        "root": str(tmp_path),
    }]


def test_discover_files_skips_file_removed_between_check_and_stat(tmp_path: Path, monkeypatch):
    target = tmp_path / "dataset" / "data.yaml"
    target.parent.mkdir()
    target.write_text("train: images/train\n", encoding="utf-8")
    real_is_file = Path.is_file

    def remove_after_check(path):
        result = real_is_file(path)
        if path == target and result:
            path.unlink()
        return result

    monkeypatch.setattr(Path, "is_file", remove_after_check)
    assert services_module.discover_files([tmp_path], {".yaml"}) == []


def test_prepare_dataset_config_makes_dataset_root_absolute(tmp_path: Path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    (dataset / "images" / "val").mkdir(parents=True)
    source = dataset / "data.yaml"
    source.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")

    monkeypatch.setattr("app.services.DATASET_ROOTS", [tmp_path])
    destination = tmp_path / "runtime" / "dataset.yaml"
    prepare_dataset_config(str(source), destination)

    generated = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert generated["path"] == str(dataset.resolve())
    assert destination.parent.name == "runtime"


def test_training_environment_uses_gnu_mkl_threading_layer():
    assert training_environment()["MKL_THREADING_LAYER"] == "GNU"


def test_yolo_bin_resolution_prefers_config_then_python_environment_then_path(tmp_path, monkeypatch):
    explicit = tmp_path / "configured" / "yolo"
    assert config_module.resolve_yolo_bin(str(explicit)) == explicit.resolve()

    environment_python = tmp_path / "environment" / "bin" / "python"
    environment_python.parent.mkdir(parents=True)
    environment_cli = environment_python.with_name("yolo")
    environment_cli.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(config_module.sys, "executable", str(environment_python))
    monkeypatch.setattr(config_module.shutil, "which", lambda _name: (_ for _ in ()).throw(
        AssertionError("PATH lookup must not run when the environment CLI exists")
    ))
    assert config_module.resolve_yolo_bin("") == environment_cli.resolve()

    environment_cli.unlink()
    path_cli = tmp_path / "path-bin" / "yolo"
    monkeypatch.setattr(config_module.shutil, "which", lambda name: str(path_cli) if name == "yolo" else None)
    assert config_module.resolve_yolo_bin("") == path_cli.resolve()


def test_failure_detail_prefers_root_error(tmp_path: Path):
    log = tmp_path / "platform.log"
    log.write_text(
        "Error: mkl-service is incompatible with libgomp.so.1\n"
        "subprocess.CalledProcessError: worker failed\n",
        encoding="utf-8",
    )
    assert failure_detail(str(log)) == "Error: mkl-service is incompatible with libgomp.so.1"


def test_failure_detail_recognizes_ddp_rank_prefix(tmp_path: Path):
    log = tmp_path / "ddp.log"
    log.write_text(
        "[rank1]: FileNotFoundError: trace.json.tmp\n"
        "subprocess.CalledProcessError: distributed worker failed\n",
        encoding="utf-8",
    )
    assert failure_detail(str(log)) == "FileNotFoundError: trace.json.tmp"


def test_sync_artifacts_is_idempotent(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    results_file = run_dir / "results.csv"
    results_file.write_text("epoch,metrics\n", encoding="utf-8")
    session = make_session(tmp_path)
    task = make_task("task-idempotent", run_dir=str(run_dir))
    run = TrainingRun(
        id="run-idempotent", task=task, attempt_number=1, launch_mode="fresh", status="completed",
        model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir), parameter_snapshot={"epochs": 100},
    )
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()

    sync_artifacts(session, task, run)
    session.commit()
    sync_artifacts(session, task, run)
    session.commit()
    results_file.write_text("epoch,metrics\n1,0.5\n", encoding="utf-8")
    sync_artifacts(session, task, run)
    session.commit()

    artifact = session.scalar(select(Artifact).where(Artifact.task_id == task.id))
    assert artifact.path == str(results_file.resolve())
    assert artifact.size == results_file.stat().st_size
    session.close()


def test_sync_artifacts_removes_records_for_disappeared_local_files(tmp_path: Path):
    run_dir = tmp_path / "run-stale"
    run_dir.mkdir()
    result = run_dir / "results.csv"
    result.write_text("epoch,metric\n", encoding="utf-8")
    session = make_session(tmp_path)
    task = make_task("task-stale-artifact", run_dir=str(run_dir))
    run = TrainingRun(
        id="run-stale-artifact", task=task, attempt_number=1, launch_mode="fresh", status="completed",
        model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir), parameter_snapshot={},
    )
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()
    sync_artifacts(session, task, run)
    session.commit()
    assert session.scalar(select(Artifact).where(Artifact.run_id == run.id)) is not None

    result.unlink()
    sync_artifacts(session, task, run)
    session.commit()

    assert session.scalar(select(Artifact).where(Artifact.run_id == run.id)) is None
    session.close()


def test_import_existing_runs_isolates_corrupt_args_yaml(tmp_path: Path, monkeypatch):
    ultralytics_root = tmp_path / "ultralytics"
    bad = ultralytics_root / "runs" / "00-bad" / "args.yaml"
    good = ultralytics_root / "runs" / "01-good" / "args.yaml"
    bad.parent.mkdir(parents=True)
    good.parent.mkdir(parents=True)
    bad.write_text("mode: train\nepochs: definitely-not-a-number\n", encoding="utf-8")
    good.write_text(
        "mode: train\nname: valid-run\ntask: detect\nepochs: 3\nbatch: 2\nimgsz: 320\nworkers: 1\n",
        encoding="utf-8",
    )
    engine = create_engine(f"sqlite:///{tmp_path / 'imports.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    monkeypatch.setattr(services_module, "ULTRALYTICS_ROOT", ultralytics_root)
    monkeypatch.setattr(services_module, "SessionLocal", session_factory)

    assert import_existing_runs() == 1
    with session_factory() as db:
        tasks = db.scalars(select(TrainingTask)).all()
        assert len(tasks) == 1
        assert tasks[0].name == "valid-run"


def test_resume_checkpoint_and_command(tmp_path: Path):
    run_dir = tmp_path / "run"
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    task = make_task("task-resume", run_dir=str(run_dir))
    run = TrainingRun(
        id="run-resume", task_id=task.id, attempt_number=2, launch_mode="resume", status="queued",
        model_path=task.model_path, data_path=task.data_path, checkpoint_path=str(checkpoint),
        run_dir=str(tmp_path / "attempt-002"), parameter_snapshot={"epochs": 100, "config": {}},
    )
    assert resume_checkpoint(task) == checkpoint
    command = TrainingScheduler()._command(task, run, Path("generated.yaml"))
    assert f"resume={checkpoint}" in command
    assert f"save_dir={run.run_dir}" in command
    assert "name=attempt-002" in command


def test_attempt_numbering_and_task_mirror(tmp_path: Path, monkeypatch):
    session = make_session(tmp_path)
    monkeypatch.setattr("app.services.RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr("app.services.RUNTIME_ROOT", tmp_path / "runtime")
    log_root = tmp_path / "log"
    monkeypatch.setattr("app.services.LOG_ROOT", log_root)
    task = make_task("task-numbering", epochs=20, batch=4)
    session.add(task)
    session.commit()

    first = create_training_run(session, task, "fresh")
    first.status = "completed"
    second = create_training_run(session, task, "retry")

    assert (first.attempt_number, second.attempt_number) == (1, 2)
    assert second.run_dir.endswith("detect/task-numbering/attempt-002")
    assert Path(first.log_path).parent == log_root
    assert Path(second.log_path).parent == log_root
    assert Path(first.log_path).name.endswith("_task-numbering_attempt-001.log")
    assert Path(second.log_path).name.endswith("_task-numbering_attempt-002.log")
    assert len(Path(second.log_path).name.split("_", 1)[0]) == 8
    assert Path(second.log_path).name.split("_", 1)[0].isdigit()
    assert task.current_run_id == second.id
    assert task.status == "queued"
    assert second.parameter_snapshot["batch"] == 4
    session.close()


def test_migrate_training_logs_moves_only_inactive_runs(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'logs.db'}")
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)
    old_root = tmp_path / "runtime"
    log_root = tmp_path / "log"
    old_root.mkdir()
    completed_log = old_root / "completed.log"
    running_log = old_root / "running.log"
    completed_log.write_text("completed output", encoding="utf-8")
    running_log.write_text("running output", encoding="utf-8")
    with local_session() as session:
        completed_task = make_task("task-completed", status="completed", log_path=str(completed_log))
        running_task = make_task("task-running", status="running", log_path=str(running_log))
        completed_run = TrainingRun(
            id="run-completed", task=completed_task, attempt_number=1, launch_mode="fresh", status="completed",
            model_path="model.pt", data_path="data.yaml", log_path=str(completed_log), parameter_snapshot={},
        )
        running_run = TrainingRun(
            id="run-running", task=running_task, attempt_number=1, launch_mode="fresh", status="running",
            model_path="model.pt", data_path="data.yaml", log_path=str(running_log), parameter_snapshot={},
        )
        completed_task.current_run_id = completed_run.id
        running_task.current_run_id = running_run.id
        session.add_all([completed_task, running_task, completed_run, running_run])
        session.commit()
    monkeypatch.setattr(services_module, "SessionLocal", local_session)
    monkeypatch.setattr(services_module, "LOG_ROOT", log_root)

    assert migrate_training_logs() == 1

    with local_session() as session:
        completed_run = session.get(TrainingRun, "run-completed")
        running_run = session.get(TrainingRun, "run-running")
        destination = Path(completed_run.log_path)
        assert destination.parent == log_root
        assert destination.name.endswith("_task-completed_attempt-001.log")
        assert destination.read_text(encoding="utf-8") == "completed output"
        assert not completed_log.exists()
        assert running_run.log_path == str(running_log)
        assert running_log.exists()


def test_same_epoch_is_stored_for_each_run(tmp_path: Path):
    session = make_session(tmp_path)
    task = make_task("task-metrics", epochs=10)
    runs = []
    for attempt in (1, 2):
        run_dir = tmp_path / f"attempt-{attempt:03d}"
        run_dir.mkdir()
        (run_dir / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,1.5,0.5\n", encoding="utf-8")
        runs.append(TrainingRun(
            id=f"run-{attempt}", task=task, attempt_number=attempt, launch_mode="fresh", status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir), parameter_snapshot={"epochs": 10},
        ))
    session.add_all([task, *runs])
    session.commit()

    for run in runs:
        sync_metrics(session, task, run)
    session.commit()

    points = session.scalars(select(MetricPoint).where(MetricPoint.task_id == task.id)).all()
    assert [(point.run_id, point.epoch) for point in points] == [("run-1", 1), ("run-2", 1)]
    session.close()


def test_mirror_run_to_task_uses_latest_run_state():
    task = make_task("task-mirror", status="completed", completed_epochs=9)
    run = TrainingRun(
        id="run-current", task_id=task.id, attempt_number=2, launch_mode="retry", status="running",
        model_path=task.model_path, data_path=task.data_path, run_dir="/runs/attempt-002", log_path="/logs/attempt-002.log",
        completed_epochs=3, progress=30, latest_metrics={"score": 0.4}, parameter_snapshot={"epochs": 10}, pid=123,
    )
    mirror_run_to_task(task, run)
    assert task.current_run_id == run.id
    assert (task.status, task.completed_epochs, task.pid) == ("running", 3, 123)
    assert task.latest_metrics == {"score": 0.4}


def test_legacy_schema_migration_creates_backup_and_run_columns(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "legacy.db"
    legacy_engine = create_engine(f"sqlite:///{database_path}")
    with legacy_engine.begin() as connection:
        connection.execute(text("CREATE TABLE training_tasks (id VARCHAR(64) PRIMARY KEY)"))
        connection.execute(text("CREATE TABLE metric_points (id INTEGER PRIMARY KEY, task_id VARCHAR(64) NOT NULL, epoch INTEGER NOT NULL, elapsed FLOAT NOT NULL, \"values\" JSON NOT NULL, created_at DATETIME NOT NULL, UNIQUE(task_id, epoch))"))
        connection.execute(text("CREATE TABLE artifacts (id INTEGER PRIMARY KEY, task_id VARCHAR(64) NOT NULL, kind VARCHAR(32) NOT NULL, name VARCHAR(255) NOT NULL, path TEXT NOT NULL, size INTEGER NOT NULL, modified_at DATETIME)"))
        connection.execute(text("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id VARCHAR(64) NOT NULL, level VARCHAR(16) NOT NULL, event_type VARCHAR(40) NOT NULL, message TEXT NOT NULL, payload JSON NOT NULL, created_at DATETIME NOT NULL)"))
        connection.execute(text("CREATE TABLE model_tests (id VARCHAR(32) PRIMARY KEY, task_id VARCHAR(64) NOT NULL, created_at DATETIME NOT NULL)"))
        connection.execute(text("INSERT INTO model_tests (id, task_id, created_at) VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'legacy-task', '2026-01-01 00:00:00')"))
    monkeypatch.setattr(database_module, "engine", legacy_engine)

    backup = database_module.migrate_database()
    inspector = inspect(legacy_engine)

    assert backup and backup.exists()
    assert "current_run_id" in {column["name"] for column in inspector.get_columns("training_tasks")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("metric_points")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("artifacts")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("task_events")}
    model_test_columns = {column["name"] for column in inspector.get_columns("model_tests")}
    assert {"model_sha256", "model_size", "rerun_of_test_id"} <= model_test_columns
    model_test_indexes = {index["name"] for index in inspector.get_indexes("model_tests")}
    assert {"ix_model_tests_task_created_id", "ix_model_tests_rerun_of_test_id"} <= model_test_indexes
    with legacy_engine.connect() as connection:
        assert connection.scalar(text("SELECT COUNT(*) FROM model_tests")) == 1


def test_model_test_migration_repairs_missing_history_indexes(tmp_path: Path, monkeypatch):
    database_path = tmp_path / "missing-history-indexes.db"
    legacy_engine = create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(legacy_engine)
    with legacy_engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_model_tests_task_created_id"))
        connection.execute(text("DROP INDEX ix_model_tests_rerun_of_test_id"))
    monkeypatch.setattr(database_module, "engine", legacy_engine)

    backup = database_module.migrate_database()

    assert backup and backup.exists()
    indexes = {index["name"] for index in inspect(legacy_engine).get_indexes("model_tests")}
    assert {"ix_model_tests_task_created_id", "ix_model_tests_rerun_of_test_id"} <= indexes
    assert database_module.migrate_database() is None


def test_backfill_registers_existing_task_as_attempt_one(tmp_path: Path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'backfill.db'}")
    Base.metadata.create_all(engine)
    local_session = sessionmaker(bind=engine, expire_on_commit=False)
    with local_session() as session:
        task = make_task("legacy-task", status="interrupted", completed_epochs=7, progress=35)
        session.add(task)
        session.flush()
        session.add(MetricPoint(task_id=task.id, epoch=7, elapsed=10, values={"score": 0.3}))
        session.commit()
    monkeypatch.setattr(services_module, "SessionLocal", local_session)

    assert backfill_training_runs() == 1
    with local_session() as session:
        task = session.get(TrainingTask, "legacy-task")
        run = session.get(TrainingRun, task.current_run_id)
        point = session.scalar(select(MetricPoint).where(MetricPoint.task_id == task.id))
        assert run and run.attempt_number == 1
        assert (run.status, run.completed_epochs) == ("interrupted", 7)
        assert point.run_id == run.id


def test_historical_run_api_returns_selected_attempt(tmp_path: Path):
    session = make_session(tmp_path)
    task = make_task("task-api", status="completed")
    first = TrainingRun(
        id="api-run-1", task=task, attempt_number=1, launch_mode="fresh", status="failed",
        model_path=task.model_path, data_path=task.data_path, parameter_snapshot={"epochs": 10},
    )
    second = TrainingRun(
        id="api-run-2", task=task, attempt_number=2, launch_mode="retry", status="completed",
        model_path=task.model_path, data_path=task.data_path, parameter_snapshot={"epochs": 20},
    )
    task.current_run_id = second.id
    session.add_all([task, first, second])
    session.flush()
    session.add_all([
        MetricPoint(task_id=task.id, run_id=first.id, epoch=1, elapsed=1, values={"score": 0.1}),
        MetricPoint(task_id=task.id, run_id=second.id, epoch=1, elapsed=2, values={"score": 0.9}),
    ])
    session.commit()

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        runs = client.get(f"/api/tasks/{task.id}/runs")
        metrics = client.get(f"/api/tasks/{task.id}/runs/{first.id}/metrics")
        assert runs.status_code == 200
        assert [item["attempt_number"] for item in runs.json()] == [2, 1]
        assert metrics.status_code == 200
        assert metrics.json()[0]["values"] == {"score": 0.1}
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_permanent_delete_removes_only_run_and_runtime_directories(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    log_root = tmp_path / "log"
    run_dir = runs_root / "detect" / "task-files" / "attempt-001"
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    (runtime_root / "task-files" / "attempt-001").mkdir(parents=True)
    (runtime_root / "task-files" / "attempt-001" / "platform.log").write_text("log", encoding="utf-8")
    log_root.mkdir()
    log_file = log_root / "20261010_task-files_attempt-001.log"
    log_file.write_text("training log", encoding="utf-8")
    monkeypatch.setattr(services_module, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(services_module, "LOG_ROOT", log_root)
    task = make_task("task-files", run_dir=str(run_dir))
    task.runs = [TrainingRun(
        id="run-files", task_id=task.id, attempt_number=1, launch_mode="fresh", status="completed",
        model_path="/models/base.pt", data_path="/datasets/data.yaml", run_dir=str(run_dir), log_path=str(log_file),
        parameter_snapshot={},
    )]

    deleted = permanently_delete_task_files(task)

    assert str(run_dir.resolve()) in deleted
    assert not run_dir.exists()
    assert not (runtime_root / "task-files").exists()
    assert not log_file.exists()


def test_permanent_delete_rejects_outside_run_directory(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.setattr(services_module, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", runtime_root)
    task = make_task("task-outside", run_dir=str(outside))

    try:
        permanently_delete_task_files(task)
    except ValueError as error:
        assert "允许的删除范围" in str(error)
    else:
        raise AssertionError("outside run directory should be rejected")


def test_permanent_delete_unlinks_top_level_symlink_without_following_target(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    outside = tmp_path / "outside-training-data"
    outside.mkdir()
    protected = outside / "keep.pt"
    protected.write_bytes(b"must remain")
    run_link = runs_root / "detect" / "task-symlink" / "attempt-001"
    run_link.parent.mkdir(parents=True)
    run_link.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(services_module, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", runtime_root)
    task = make_task("task-symlink", run_dir=str(run_link))
    task.runs = [TrainingRun(
        id="run-symlink", task_id=task.id, attempt_number=1, launch_mode="fresh", status="completed",
        model_path=task.model_path, data_path=task.data_path, run_dir=str(run_link), parameter_snapshot={},
    )]

    deleted = permanently_delete_task_files(task)

    assert str(run_link) in deleted
    assert not run_link.exists()
    assert not run_link.is_symlink()
    assert protected.read_bytes() == b"must remain"


def test_soft_delete_filters_task_list_and_restore(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", tmp_path / "model-test-history")
    session = make_session(tmp_path)
    task = make_task("task-delete", status="completed")
    session.add(task)
    session.commit()
    history_file = model_test_storage.test_history_root(task.id, "a" * 32) / "detections.json"
    history_file.parent.mkdir(parents=True)
    history_file.write_text("{}", encoding="utf-8")

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        assert client.post(f"/api/tasks/{task.id}/trash").status_code == 204
        assert history_file.is_file()
        assert client.get("/api/tasks").json() == []
        deleted = client.get("/api/tasks?include_deleted=true").json()
        assert deleted[0]["deleted_at"] is not None
        assert client.post(f"/api/tasks/{task.id}/restore").status_code == 200
        assert history_file.is_file()
        assert client.get("/api/tasks").json()[0]["id"] == task.id
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_task_listing_does_not_scan_or_write_metrics(tmp_path: Path, monkeypatch):
    session = make_session(tmp_path)
    run_dir = tmp_path / "runs" / "attempt-001"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,2,0.5\n", encoding="utf-8")
    task = make_task("task-read-only", status="completed", run_dir=str(run_dir))
    run = TrainingRun(id="run-read-only", task=task, attempt_number=1, launch_mode="fresh",
                      status="completed", model_path=task.model_path, data_path=task.data_path,
                      run_dir=str(run_dir), parameter_snapshot={"epochs": 10})
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()

    def override_db():
        yield session

    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("dashboard list endpoint must not scan training files")

    monkeypatch.setattr(main_module, "sync_metrics", forbidden_sync)
    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).get("/api/tasks")
        assert response.status_code == 200
        assert response.json()[0]["completed_epochs"] == 0
        assert session.scalar(select(MetricPoint).where(MetricPoint.run_id == run.id)) is None
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_read_only_run_endpoints_do_not_scan_local_files(tmp_path: Path, monkeypatch):
    session = make_session(tmp_path)
    run_dir = tmp_path / "runs" / "attempt-001"
    run_dir.mkdir(parents=True)
    (run_dir / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,2,0.5\n", encoding="utf-8")
    (run_dir / "best.pt").write_bytes(b"weights")
    task = make_task("task-read-only-runs", status="completed", run_dir=str(run_dir))
    run = TrainingRun(id="run-read-only-runs", task=task, attempt_number=1, launch_mode="fresh",
                      status="completed", model_path=task.model_path, data_path=task.data_path,
                      run_dir=str(run_dir), parameter_snapshot={"epochs": 10})
    task.current_run_id = run.id
    session.add_all([task, run])
    session.commit()

    def forbidden_sync(*_args, **_kwargs):
        raise AssertionError("read endpoints must not scan training files")

    monkeypatch.setattr(main_module, "sync_metrics", forbidden_sync)

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        for suffix in ("metrics", "artifacts", f"runs/{run.id}/metrics", f"runs/{run.id}/artifacts"):
            response = client.get(f"/api/tasks/{task.id}/{suffix}")
            assert response.status_code == 200, response.text
            assert response.json() == []
        assert session.scalar(select(MetricPoint).where(MetricPoint.run_id == run.id)) is None
        assert session.scalar(select(Artifact).where(Artifact.run_id == run.id)) is None
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_effective_best_weight_follows_exact_resume_chain(tmp_path: Path):
    session = make_session(tmp_path)
    task = make_task("task-effective-best", status="completed")
    run_one = TrainingRun(
        id="run-best-1", task=task, attempt_number=1, launch_mode="fresh", status="interrupted",
        model_path=task.model_path, data_path=task.data_path, run_dir="runs/attempt-001", parameter_snapshot={},
    )
    run_two = TrainingRun(
        id="run-best-2", task=task, attempt_number=2, launch_mode="resume", status="interrupted",
        model_path=task.model_path, data_path=task.data_path, run_dir="runs/attempt-002",
        checkpoint_path="runs/attempt-001/weights/last.pt", parameter_snapshot={},
    )
    run_three = TrainingRun(
        id="run-best-3", task=task, attempt_number=3, launch_mode="resume", status="completed",
        model_path=task.model_path, data_path=task.data_path, run_dir="runs/attempt-003",
        checkpoint_path="runs/attempt-002/weights/last.pt", parameter_snapshot={},
    )
    unrelated = TrainingRun(
        id="run-best-4", task=task, attempt_number=4, launch_mode="retry", status="completed",
        model_path=task.model_path, data_path=task.data_path, run_dir="runs/attempt-004", parameter_snapshot={},
    )
    task.current_run_id = run_three.id
    session.add_all([task, run_one, run_two, run_three, unrelated])
    session.flush()
    session.add_all([
        Artifact(task_id=task.id, run_id=run_one.id, kind="weight", name="weights/best.pt",
                 path="runs/attempt-001/weights/best.pt", size=100),
        Artifact(task_id=task.id, run_id=run_one.id, kind="weight", name="weights/last.pt",
                 path=run_two.checkpoint_path, size=110),
        Artifact(task_id=task.id, run_id=run_two.id, kind="weight", name="weights/last.pt",
                 path=run_three.checkpoint_path, size=120),
        Artifact(task_id=task.id, run_id=unrelated.id, kind="weight", name="weights/best.pt",
                 path="runs/attempt-004/weights/best.pt", size=400),
    ])
    session.commit()

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        current = client.get(f"/api/tasks/{task.id}/runs/{run_three.id}/effective-best-weight")
        assert current.status_code == 200
        assert current.json()["artifact"]["path"] == "runs/attempt-001/weights/best.pt"
        assert current.json()["source_run_id"] == run_one.id
        assert current.json()["source_attempt_number"] == 1
        assert current.json()["inherited"] is True

        direct = client.get(f"/api/tasks/{task.id}/runs/{run_one.id}/effective-best-weight").json()
        assert direct["source_run_id"] == run_one.id
        assert direct["inherited"] is False

        isolated = client.get(f"/api/tasks/{task.id}/runs/{unrelated.id}/effective-best-weight").json()
        assert isolated["source_run_id"] == unrelated.id
        assert isolated["inherited"] is False
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_permanent_delete_endpoint_removes_database_records_and_owned_files(tmp_path: Path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    run_dir = runs_root / "detect" / "task-permanent" / "attempt-001"
    runtime_dir = runtime_root / "task-permanent"
    run_dir.mkdir(parents=True)
    runtime_dir.mkdir(parents=True)
    artifact_path = run_dir / "best.pt"
    artifact_path.write_bytes(b"trained weights")
    (runtime_dir / "platform.log").write_text("training log", encoding="utf-8")
    model_path = tmp_path / "base.pt"
    data_path = tmp_path / "data.yaml"
    model_path.write_bytes(b"base model")
    data_path.write_text("names: [object]\n", encoding="utf-8")

    monkeypatch.setattr(services_module, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", tmp_path / "model-test-history")
    session = make_session(tmp_path)
    task = make_task(
        "task-permanent", status="completed", model_path=str(model_path),
        data_path=str(data_path), run_dir=str(run_dir),
    )
    run = TrainingRun(
        id="run-permanent", task=task, attempt_number=1, launch_mode="fresh", status="completed",
        model_path=str(model_path), data_path=str(data_path), run_dir=str(run_dir), parameter_snapshot={},
    )
    task.current_run_id = run.id
    session.add_all([task, run])
    session.flush()
    session.add_all([
        MetricPoint(task_id=task.id, run_id=run.id, epoch=1, elapsed=1, values={"score": 0.5}),
        Artifact(task_id=task.id, run_id=run.id, kind="weight", name="best.pt", path=str(artifact_path), size=15),
        TaskEvent(task_id=task.id, run_id=run.id, event_type="completed", message="done", payload={}),
    ])
    session.commit()
    history_file = model_test_storage.test_history_root(task.id, "b" * 32) / "results" / "0.jpg"
    history_file.parent.mkdir(parents=True)
    history_file.write_bytes(b"detection result")

    def override_db():
        yield session

    app.dependency_overrides[get_db] = override_db
    try:
        response = TestClient(app).delete(f"/api/tasks/{task.id}/permanent")
        assert response.status_code == 202
        assert response.json()["status"] == "pending"
        assert task.deleted_reason == "local_delete_pending"
        worker = LocalDeletionWorker(session_factory=lambda: session)
        assert worker.process_one()
        assert session.get(TrainingTask, task.id) is None
        assert session.get(TrainingRun, run.id) is None
        assert session.scalar(select(MetricPoint).where(MetricPoint.task_id == task.id)) is None
        assert session.scalar(select(Artifact).where(Artifact.task_id == task.id)) is None
        assert session.scalar(select(TaskEvent).where(TaskEvent.task_id == task.id)) is None
        assert not run_dir.exists()
        assert not runtime_dir.exists()
        assert not model_test_storage.task_history_root(task.id).exists()
        assert model_path.exists()
        assert data_path.exists()
    finally:
        app.dependency_overrides.clear()
        session.close()


def test_local_history_cleanup_failure_keeps_task_retryable(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(services_module, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setattr(services_module, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", tmp_path / "model-test-history")
    session = make_session(tmp_path)
    task = make_task("task-history-retry", status="completed")
    session.add(task)
    session.commit()
    history_file = model_test_storage.test_history_root(task.id, "e" * 32) / "detections.json"
    history_file.parent.mkdir(parents=True)
    history_file.write_text("{}", encoding="utf-8")

    def override_db():
        yield session

    def cleanup_failure(_task_id: str) -> None:
        raise PermissionError("history is read-only")

    app.dependency_overrides[get_db] = override_db
    try:
        client = TestClient(app)
        assert client.delete(f"/api/tasks/{task.id}/permanent").status_code == 202
        monkeypatch.setattr(local_deletions_module, "delete_task_history", cleanup_failure)
        worker = LocalDeletionWorker(session_factory=lambda: session)
        assert worker.process_one()
        retained = session.get(TrainingTask, task.id)
        assert retained is not None
        assert retained.deleted_reason == "local_delete_failed"
        assert "PermissionError" in retained.error_message
        assert history_file.is_file()

        monkeypatch.setattr(
            local_deletions_module, "delete_task_history", model_test_storage.delete_task_history,
        )
        assert client.delete(f"/api/tasks/{task.id}/permanent").status_code == 202
        assert worker.process_one()
        assert session.get(TrainingTask, task.id) is None
        assert not model_test_storage.task_history_root(task.id).exists()
    finally:
        app.dependency_overrides.clear()
        session.close()
