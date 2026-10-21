import hashlib
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.run_deletions as run_deletions
from app.database import Base
from app.local_deletions import LocalDeletionWorker, queue_local_run_delete
from app.models import Artifact, MetricPoint, ModelTest, RemoteOperation, TaskEvent, TrainingRun, TrainingTask
from app.services import create_training_run


def session_factory(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'runs.db'}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


def test_local_attempt_delete_preserves_other_run_and_detection_history(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    log_root = tmp_path / "logs"
    ultralytics_root = tmp_path / "ultralytics"
    for root in (runs_root, runtime_root, log_root, ultralytics_root):
        root.mkdir()
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(run_deletions, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(run_deletions, "LOG_ROOT", log_root)
    monkeypatch.setattr(run_deletions, "ULTRALYTICS_ROOT", ultralytics_root)
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id="task-attempt-delete", name="Attempt delete", task_type="detect", status="completed",
            model_path="model.pt", data_path="data.yaml", config={},
        )
        first_dir = runs_root / "detect" / task.id / "attempt-001"
        second_dir = runs_root / "detect" / task.id / "attempt-002"
        first_dir.mkdir(parents=True)
        second_dir.mkdir(parents=True)
        (first_dir / "weights.pt").write_bytes(b"first")
        (second_dir / "weights.pt").write_bytes(b"second")
        first = TrainingRun(
            id="run-attempt-1", task=task, attempt_number=1, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(first_dir),
        )
        second = TrainingRun(
            id="run-attempt-2", task=task, attempt_number=2, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(second_dir),
        )
        task.current_run_id = second.id
        db.add_all([task, first, second])
        db.flush()
        artifact = Artifact(
            task_id=task.id, run_id=first.id, kind="weight", name="weights.pt",
            path=str(first_dir / "weights.pt"), size=5,
        )
        db.add_all([
            artifact,
            MetricPoint(task_id=task.id, run_id=first.id, epoch=1, elapsed=1, values={"score": 0.1}),
            TaskEvent(task_id=task.id, run_id=first.id, event_type="finished", message="done", payload={}),
            ModelTest(
                id="a" * 32, task_id=task.id, run_id=first.id, source_run_id=first.id,
                artifact_id=artifact.id, artifact_name="weights.pt", model_name="weights.pt",
                confidence=0.25, iou=0.7, status="completed",
            ),
        ])
        db.commit()
        operation = queue_local_run_delete(db, task, first)
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingRun, first.id) is None
        assert db.get(TrainingRun, second.id) is not None
        assert db.scalar(select(MetricPoint).where(MetricPoint.run_id == first.id)) is None
        assert db.scalar(select(Artifact).where(Artifact.run_id == first.id)) is None
        assert db.scalar(select(TaskEvent).where(TaskEvent.run_id == first.id)) is None
        history = db.get(ModelTest, "a" * 32)
        assert history is not None
        assert history.run_id is None and history.source_run_id is None and history.artifact_id is None
        assert db.get(TrainingTask, task.id).current_run_id == second.id
        assert db.get(RemoteOperation, operation.id).status == "succeeded"
    assert not first_dir.exists()
    assert second_dir.is_dir()


def test_deleted_attempt_number_is_not_reused(tmp_path, monkeypatch):
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id="task-attempt-sequence", name="Sequence", task_type="detect", status="draft",
            model_path="model.pt", data_path="data.yaml", config={},
        )
        db.add(task)
        db.flush()
        db.add(RemoteOperation(
            id="a" * 32, node_id="__local__", task_id=task.id, kind="local_delete",
            status="succeeded", payload={"scope": "attempt", "run_id": "old", "attempt_number": 4},
        ))
        db.commit()
        run = create_training_run(db, task, "fresh")
        assert run.attempt_number == 5


def test_local_legacy_flat_attempt_delete_preserves_canonical_child(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    log_root = tmp_path / "logs"
    ultralytics_root = tmp_path / "ultralytics"
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(run_deletions, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(run_deletions, "LOG_ROOT", log_root)
    monkeypatch.setattr(run_deletions, "ULTRALYTICS_ROOT", ultralytics_root)
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id="task-flat-legacy", name="Flat legacy", task_type="detect", status="stopped",
            model_path="model.pt", data_path="data.yaml", config={},
        )
        parent = runs_root / task.task_type / task.id
        weight = parent / "weights" / "last.pt"
        second_dir = parent / "attempt-002"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"legacy weight")
        second_dir.mkdir()
        marker = second_dir / "keep.txt"
        marker.write_text("must remain")
        first = TrainingRun(
            id="run-flat-legacy", task=task, attempt_number=1, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(parent),
        )
        second = TrainingRun(
            id="run-flat-second", task=task, attempt_number=2, status="stopped",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(second_dir),
        )
        task.current_run_id = second.id
        task.run_dir = str(second_dir)
        task.config = {"_resume_checkpoint": str(weight)}
        db.add_all([task, first, second])
        db.flush()
        db.add(Artifact(
            task_id=task.id, run_id=first.id, kind="weight", name="weights/last.pt",
            path=str(weight), size=len(b"legacy weight"),
        ))
        db.commit()
        operation = queue_local_run_delete(db, task, first)
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingRun, first.id) is None
        assert db.get(TrainingRun, second.id) is not None
        assert db.get(RemoteOperation, operation.id).status == "succeeded"
        assert "_resume_checkpoint" not in db.get(TrainingTask, task.id).config
    assert not weight.exists()
    assert marker.read_text() == "must remain"


def test_local_canonical_attempt_delete_ignores_legacy_container_overlap(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    log_root = tmp_path / "logs"
    ultralytics_root = tmp_path / "ultralytics"
    for root in (runtime_root, log_root, ultralytics_root):
        root.mkdir()
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(run_deletions, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(run_deletions, "LOG_ROOT", log_root)
    monkeypatch.setattr(run_deletions, "ULTRALYTICS_ROOT", ultralytics_root)
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id="task-flat-container", name="Flat container", task_type="detect", status="stopped",
            model_path="model.pt", data_path="data.yaml", config={},
        )
        parent = runs_root / task.task_type / task.id
        legacy_weight = parent / "weights" / "last.pt"
        second_dir = parent / "attempt-002"
        legacy_weight.parent.mkdir(parents=True)
        legacy_weight.write_bytes(b"keep legacy")
        second_dir.mkdir()
        (second_dir / "results.csv").write_text("delete second")
        first = TrainingRun(
            id="run-container-first", task=task, attempt_number=1, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(parent),
        )
        second = TrainingRun(
            id="run-container-second", task=task, attempt_number=2, status="stopped",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(second_dir),
        )
        task.current_run_id = second.id
        task.run_dir = str(second_dir)
        db.add_all([task, first, second])
        db.flush()
        db.add(Artifact(
            task_id=task.id, run_id=first.id, kind="weight", name="weights/last.pt",
            path=str(legacy_weight), size=len(b"keep legacy"),
        ))
        db.commit()
        operation = queue_local_run_delete(db, task, second)
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingRun, first.id) is not None
        assert db.get(TrainingRun, second.id) is None
        assert db.get(RemoteOperation, operation.id).status == "succeeded"
    assert legacy_weight.read_bytes() == b"keep legacy"
    assert not second_dir.exists()


def test_imported_legacy_attempt_remains_deletable(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    runtime_root = tmp_path / "runtime"
    log_root = tmp_path / "logs"
    ultralytics_root = tmp_path / "ultralytics"
    legacy_dir = ultralytics_root / "runs" / "detect" / "train7"
    for root in (runs_root, runtime_root, log_root, ultralytics_root):
        root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", runs_root)
    monkeypatch.setattr(run_deletions, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(run_deletions, "LOG_ROOT", log_root)
    monkeypatch.setattr(run_deletions, "ULTRALYTICS_ROOT", ultralytics_root)
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "args.yaml").write_text("mode: train\ntask: detect\n", encoding="utf-8")
    (legacy_dir / "results.csv").write_text("epoch,metrics\n1,0.5\n", encoding="utf-8")
    (legacy_dir / "platform.log").write_text("legacy log", encoding="utf-8")
    legacy_id = "legacy-" + hashlib.sha1(str(legacy_dir.resolve()).encode()).hexdigest()[:12]
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id=legacy_id, name="Imported legacy", task_type="detect", status="interrupted",
            source="imported", model_path="model.pt", data_path="data.yaml", run_dir=str(legacy_dir), config={},
        )
        run = TrainingRun(
            id="run-imported-legacy", task=task, attempt_number=1, status="interrupted",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(legacy_dir),
            log_path=str(legacy_dir / "platform.log"),
        )
        db.add_all([task, run])
        db.flush()
        operation = queue_local_run_delete(db, task, run)
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingTask, legacy_id) is not None
        assert db.get(TrainingRun, "run-imported-legacy") is None
        assert db.get(RemoteOperation, operation.id).status == "succeeded"
    assert not legacy_dir.exists()


def test_local_attempt_delete_preserves_path_referenced_by_other_task(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", runs_root)
    factory = session_factory(tmp_path)
    with factory() as db:
        task = TrainingTask(
            id="task-shared-attempt", name="Shared attempt", task_type="detect", status="completed",
            model_path="base.pt", data_path="data.yaml", config={},
        )
        run_dir = runs_root / task.task_type / task.id / "attempt-001"
        weight = run_dir / "weights" / "best.pt"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"shared model")
        run = TrainingRun(
            id="run-shared-attempt", task=task, attempt_number=1, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir),
        )
        dependent = TrainingTask(
            id="task-dependent-model", name="Dependent model", task_type="detect", status="draft",
            model_path=str(weight), data_path="data.yaml", config={},
        )
        db.add_all([task, run, dependent])
        db.flush()

        try:
            queue_local_run_delete(db, task, run)
        except ValueError as error:
            assert "引用" in str(error)
        else:
            raise AssertionError("an Attempt referenced by another task must not be deleted")

    assert weight.read_bytes() == b"shared model"
