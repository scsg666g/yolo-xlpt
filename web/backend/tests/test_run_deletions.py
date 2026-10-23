import hashlib
import shutil
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.local_deletions as local_deletions
import app.run_deletions as run_deletions
from app.database import Base
from app.local_deletions import LocalDeletionWorker, queue_local_run_delete
from app.models import Artifact, MetricPoint, ModelAsset, ModelTest, RemoteOperation, TaskEvent, TrainingRun, TrainingTask, utcnow
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


def _isolated_attempt(tmp_path, monkeypatch, suffix="protected"):
    roots = {
        "runs": tmp_path / "runs",
        "runtime": tmp_path / "runtime",
        "logs": tmp_path / "logs",
        "ultralytics": tmp_path / "ultralytics",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run_deletions, "RUNS_ROOT", roots["runs"])
    monkeypatch.setattr(run_deletions, "RUNTIME_ROOT", roots["runtime"])
    monkeypatch.setattr(run_deletions, "LOG_ROOT", roots["logs"])
    monkeypatch.setattr(run_deletions, "ULTRALYTICS_ROOT", roots["ultralytics"])
    monkeypatch.setattr(local_deletions, "managed_dataset_mutation", nullcontext)
    factory = session_factory(tmp_path)
    task_id = f"task-{suffix}"
    run_id = f"run-{suffix}"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    weight = run_dir / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    runtime = roots["runtime"] / task_id / "attempt-001"
    runtime.mkdir(parents=True)
    with factory() as db:
        task = TrainingTask(
            id=task_id, name=suffix, task_type="detect", status="completed",
            model_path="base.pt", data_path="data.yaml", config={},
        )
        run = TrainingRun(
            id=run_id, task=task, attempt_number=1, status="completed",
            model_path=task.model_path, data_path=task.data_path, run_dir=str(run_dir),
        )
        task.current_run_id = run.id
        task.run_dir = str(run_dir)
        db.add_all([task, run])
        db.flush()
        artifact = Artifact(
            task_id=task.id, run_id=run.id, kind="weight", name="weights/best.pt",
            path=str(weight), size=weight.stat().st_size,
        )
        db.add(artifact)
        db.commit()
    return factory, roots, task_id, run_id, weight


def test_local_attempt_delete_rejects_registered_model_asset(tmp_path, monkeypatch):
    factory, _, task_id, run_id, weight = _isolated_attempt(tmp_path, monkeypatch, "asset")
    with factory() as db:
        task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
        db.add(ModelAsset(
            id="asset-protected", name="正在使用的模型", source_type="training_artifact",
            execution_node_id="local", path=str(weight), status="ready",
        ))
        db.flush()
        with pytest.raises(ValueError, match="模型来源"):
            queue_local_run_delete(db, task, run)
    assert weight.exists()


def test_local_attempt_delete_rejects_removed_asset_retained_by_external_history(tmp_path, monkeypatch):
    factory, _, task_id, run_id, weight = _isolated_attempt(tmp_path, monkeypatch, "history")
    with factory() as db:
        task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
        asset = ModelAsset(
            id="asset-history", name="历史模型", source_type="training_artifact",
            execution_node_id="local", path=str(weight), status="removed", deleted_at=utcnow(),
        )
        db.add(asset)
        db.flush()
        db.add(ModelTest(
            id="b" * 32, task_id=None, model_asset_id=asset.id,
            model_source_type="registered_model", model_source_path=str(weight),
            artifact_name="best.pt", model_name="best.pt", confidence=0.25, iou=0.7,
            status="completed",
        ))
        db.flush()
        with pytest.raises(ValueError, match="历史引用"):
            queue_local_run_delete(db, task, run)
    assert weight.exists()


def test_local_attempt_delete_rejects_external_active_model_test(tmp_path, monkeypatch):
    factory, _, task_id, run_id, weight = _isolated_attempt(tmp_path, monkeypatch, "active-test")
    with factory() as db:
        task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
        db.add(ModelTest(
            id="c" * 32, task_id=None, execution_node_id="local",
            model_source_type="registered_model", model_source_path=str(weight),
            artifact_name="best.pt", model_name="best.pt", confidence=0.25, iou=0.7,
            status="running",
        ))
        db.flush()
        with pytest.raises(ValueError, match="其他模型检测"):
            queue_local_run_delete(db, task, run)
    assert weight.exists()


def test_local_attempt_delete_allows_removed_unreferenced_asset(tmp_path, monkeypatch):
    factory, _, task_id, run_id, weight = _isolated_attempt(tmp_path, monkeypatch, "removed-asset")
    with factory() as db:
        task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
        db.add(ModelAsset(
            id="asset-removed", name="已移除模型", source_type="training_artifact",
            execution_node_id="local", path=str(weight), status="removed", deleted_at=utcnow(),
        ))
        db.flush()
        operation = queue_local_run_delete(db, task, run)
        assert operation.payload["plan"]["run_id"] == run.id


def test_attempt_files_are_deleted_without_holding_sqlite_writer_lock(tmp_path, monkeypatch):
    factory, _, task_id, run_id, _ = _isolated_attempt(tmp_path, monkeypatch, "writer-lock")
    with factory() as db:
        operation = queue_local_run_delete(db, db.get(TrainingTask, task_id), db.get(TrainingRun, run_id))
        db.commit()

    execute = local_deletions.execute_local_attempt_plan

    def execute_while_writing(plan):
        with factory() as other:
            other.add(TrainingTask(
                id="parallel-write", name="Parallel write", task_type="detect",
                status="draft", model_path="base.pt", data_path="data.yaml", config={},
            ))
            other.commit()
        return execute(plan)

    monkeypatch.setattr(local_deletions, "execute_local_attempt_plan", execute_while_writing)
    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingTask, "parallel-write") is not None
        assert db.get(RemoteOperation, operation.id).status == "succeeded"


def test_stale_attempt_cleanup_resumes_after_partial_file_removal(tmp_path, monkeypatch):
    factory, roots, task_id, run_id, _ = _isolated_attempt(tmp_path, monkeypatch, "stale-partial")
    with factory() as db:
        operation = queue_local_run_delete(db, db.get(TrainingTask, task_id), db.get(TrainingRun, run_id))
        db.commit()
        runtime_target = Path(operation.payload["plan"]["runtime_target"])
        shutil.rmtree(runtime_target)
        operation.status = "running"
        operation.lease = "stale-lease"
        operation.updated_at = utcnow() - timedelta(hours=1)
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        completed = db.get(RemoteOperation, operation.id)
        assert completed.status == "succeeded"
        assert str(runtime_target) in completed.payload["receipt"]["already_missing_targets"]
        assert db.get(TrainingRun, run_id) is None
    assert not (roots["runs"] / "detect" / task_id / "attempt-001").exists()


def test_legacy_attempt_plan_rejects_new_root_entry_after_queue(tmp_path, monkeypatch):
    factory, roots, task_id, run_id, weight = _isolated_attempt(tmp_path, monkeypatch, "legacy-freeze")
    task_root = roots["runs"] / "detect" / task_id
    canonical = task_root / "attempt-001"
    legacy_weight = task_root / "weights" / "best.pt"
    legacy_weight.parent.mkdir(parents=True, exist_ok=True)
    legacy_weight.write_bytes(weight.read_bytes())
    shutil.rmtree(canonical)
    with factory() as db:
        task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
        run.run_dir = task.run_dir = str(task_root)
        artifact = db.scalar(select(Artifact).where(Artifact.run_id == run.id))
        artifact.path = str(legacy_weight)
        db.commit()
        operation = queue_local_run_delete(db, task, run)
        db.commit()
    unexpected = task_root / "created-after-queue.txt"
    unexpected.write_text("keep", encoding="utf-8")

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(RemoteOperation, operation.id).status == "failed"
        assert db.get(TrainingRun, run_id) is not None
    assert unexpected.read_text(encoding="utf-8") == "keep"
    assert legacy_weight.exists()


def test_attempt_cleanup_finishes_database_records_from_saved_receipt(tmp_path, monkeypatch):
    factory, _, task_id, run_id, _ = _isolated_attempt(tmp_path, monkeypatch, "saved-receipt")
    with factory() as db:
        operation = queue_local_run_delete(db, db.get(TrainingTask, task_id), db.get(TrainingRun, run_id))
        db.commit()
        plan = operation.payload["plan"]
    receipt = run_deletions.execute_local_attempt_plan(plan)
    receipt.update({
        "plan_version": plan["version"], "task_id": task_id, "run_id": run_id,
        "attempt_number": plan["attempt_number"], "completed_at": utcnow().isoformat(),
    })
    with factory() as db:
        operation = db.get(RemoteOperation, operation.id)
        operation.status = "running"
        operation.lease = "expired-receipt-lease"
        operation.updated_at = utcnow() - timedelta(hours=1)
        operation.payload = {**operation.payload, "phase": "recording", "receipt": receipt}
        db.commit()

    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        completed = db.get(RemoteOperation, operation.id)
        assert completed.status == "succeeded"
        assert completed.payload["receipt"] == receipt
        assert db.get(TrainingRun, run_id) is None
