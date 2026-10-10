from pathlib import Path

import yaml
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

import app.database as database_module
import app.services as services_module
from app.database import Base
from app.database import get_db
from app.main import app
from app.models import Artifact, MetricPoint, TrainingRun, TrainingTask
from app.services import (
    TrainingScheduler,
    backfill_training_runs,
    create_training_run,
    failure_detail,
    mirror_run_to_task,
    path_is_within,
    prepare_dataset_config,
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


def test_path_is_within(tmp_path: Path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    assert path_is_within(allowed / "model.pt", [allowed])
    assert not path_is_within(tmp_path / "outside.pt", [allowed])


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


def test_failure_detail_prefers_root_error(tmp_path: Path):
    log = tmp_path / "platform.log"
    log.write_text(
        "Error: mkl-service is incompatible with libgomp.so.1\n"
        "subprocess.CalledProcessError: worker failed\n",
        encoding="utf-8",
    )
    assert failure_detail(str(log)) == "Error: mkl-service is incompatible with libgomp.so.1"


def test_sync_artifacts_is_idempotent(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("epoch,metrics\n", encoding="utf-8")
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

    paths = session.scalars(select(Artifact.path).where(Artifact.task_id == task.id)).all()
    assert paths == [str((run_dir / "results.csv").resolve())]
    session.close()


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
    task = make_task("task-numbering", epochs=20, batch=4)
    session.add(task)
    session.commit()

    first = create_training_run(session, task, "fresh")
    first.status = "completed"
    second = create_training_run(session, task, "retry")

    assert (first.attempt_number, second.attempt_number) == (1, 2)
    assert second.run_dir.endswith("detect/task-numbering/attempt-002")
    assert second.log_path.endswith("task-numbering/attempt-002/platform.log")
    assert task.current_run_id == second.id
    assert task.status == "queued"
    assert second.parameter_snapshot["batch"] == 4
    session.close()


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
    monkeypatch.setattr(database_module, "engine", legacy_engine)

    backup = database_module.migrate_database()
    inspector = inspect(legacy_engine)

    assert backup and backup.exists()
    assert "current_run_id" in {column["name"] for column in inspector.get_columns("training_tasks")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("metric_points")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("artifacts")}
    assert "run_id" in {column["name"] for column in inspector.get_columns("task_events")}


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
