import hashlib
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

import app.dataset_uploads as dataset_uploads
import app.local_deletions as local_deletions
import app.model_assets as model_asset_service
import app.model_test_storage as model_test_storage
import app.services as services
from app.database import Base, configure_sqlite_connection
from app.local_deletions import LocalDeletionWorker, queue_local_delete
from app.models import (
    Artifact,
    ComputeNode,
    ModelAsset,
    ModelTest,
    RemoteOperation,
    TrainingRun,
    TrainingTask,
    utcnow,
)
from app.task_deletions import build_local_task_plan, execute_local_task_plan


@pytest.fixture
def deletion_env(tmp_path, monkeypatch):
    roots = {
        "runs": tmp_path / "runs",
        "runtime": tmp_path / "runtime",
        "logs": tmp_path / "logs",
        "ultralytics": tmp_path / "ultralytics",
        "history": tmp_path / "test-history",
        "work": tmp_path / "test-work",
        "uploads": tmp_path / "uploads",
    }
    for root in roots.values():
        root.mkdir(parents=True)
    monkeypatch.setattr(services, "RUNS_ROOT", roots["runs"])
    monkeypatch.setattr(services, "RUNTIME_ROOT", roots["runtime"])
    monkeypatch.setattr(services, "LOG_ROOT", roots["logs"])
    monkeypatch.setattr(services, "ULTRALYTICS_ROOT", roots["ultralytics"])
    monkeypatch.setattr(model_test_storage, "HISTORY_ROOT", roots["history"])
    monkeypatch.setattr(model_test_storage, "WORK_ROOT", roots["work"])
    monkeypatch.setattr(dataset_uploads, "UPLOAD_ROOT", roots["uploads"])
    engine = create_engine(f"sqlite:///{tmp_path / 'deletions.db'}")
    event.listen(engine, "connect", configure_sqlite_connection)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        db.add(ComputeNode(id="local", name="本机服务器", kind="local", snapshot={}))
        db.commit()
    return factory, roots, tmp_path


def add_task(db, roots, tmp_path, task_id="task-delete", *, status="completed", source="platform"):
    resources = tmp_path / "resources"
    resources.mkdir(exist_ok=True)
    model = resources / "base.pt"
    data = resources / "data.yaml"
    model.write_bytes(b"model")
    data.write_text("names: [object]\n", encoding="utf-8")
    task = TrainingTask(
        id=task_id,
        name="Delete task",
        task_type="detect",
        status=status,
        source=source,
        model_path=str(model),
        data_path=str(data),
        config={},
    )
    db.add(task)
    return task, model, data


def add_run(db, task, run_id="run-delete", attempt=1, *, status="completed", run_dir=None):
    run = TrainingRun(
        id=run_id,
        task=task,
        attempt_number=attempt,
        launch_mode="fresh",
        status=status,
        model_path=task.model_path,
        data_path=task.data_path,
        run_dir=str(run_dir) if run_dir else None,
        parameter_snapshot={},
    )
    db.add(run)
    return run


def test_whole_local_cleanup_records_receipt_and_preserves_inputs(deletion_env):
    factory, roots, tmp_path = deletion_env
    task_id = "task-delete-complete"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    runtime = roots["runtime"] / task_id / "attempt-001"
    log = roots["logs"] / f"20261023_{task_id}_attempt-001.log"
    weight = run_dir / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    runtime.mkdir(parents=True)
    (runtime / "args.yaml").write_text("task: detect\n", encoding="utf-8")
    log.write_text("finished", encoding="utf-8")
    test_id = "a" * 32

    with factory() as db:
        task, model, data = add_task(db, roots, tmp_path, task_id)
        run = add_run(db, task, run_dir=run_dir)
        run.log_path = str(log)
        task.current_run_id = run.id
        task.run_dir = str(run_dir)
        task.log_path = str(log)
        db.flush()
        artifact = Artifact(
            task_id=task.id,
            run_id=run.id,
            kind="weight",
            name="best.pt",
            path=str(weight),
            size=6,
        )
        db.add(artifact)
        db.flush()
        db.add(ModelTest(
            id=test_id,
            task_id=task.id,
            run_id=run.id,
            source_run_id=run.id,
            artifact_id=artifact.id,
            artifact_name="best.pt",
            model_name="best.pt",
            model_source_path=str(weight),
            execution_node_id="local",
            confidence=0.25,
            iou=0.7,
            status="completed",
        ))
        operation = queue_local_delete(db, task)
        db.commit()
        operation_id = operation.id

    history = model_test_storage.test_history_root(task_id, test_id)
    work = model_test_storage.test_work_root(test_id)
    history.mkdir(parents=True)
    work.mkdir(parents=True)
    (history / "detections.json").write_text("{}", encoding="utf-8")
    (work / "model.pt").write_bytes(b"pin")

    assert LocalDeletionWorker(factory).process_once()

    with factory() as db:
        assert db.get(TrainingTask, task_id) is None
        completed = db.get(RemoteOperation, operation_id)
        assert completed.status == "succeeded"
        assert completed.payload["phase"] == "complete"
        receipt = completed.payload["receipt"]
        assert receipt["verified_target_count"] >= 3
        assert receipt["cleaned_test_ids"] == [test_id]
        assert receipt["test_history_removed"] is True
    assert not run_dir.exists()
    assert not (roots["runtime"] / task_id).exists()
    assert not log.exists()
    assert not history.exists()
    assert not work.exists()
    assert model.exists() and data.exists()


@pytest.mark.parametrize("active_status", ["queued", "running", "stopping"])
def test_whole_plan_rejects_active_attempt_even_if_task_is_terminal(deletion_env, active_status):
    factory, roots, tmp_path = deletion_env
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path)
        add_run(db, task, status=active_status)
        db.flush()
        with pytest.raises(ValueError, match="Attempt"):
            build_local_task_plan(db, task)


def test_whole_plan_rejects_remote_or_malformed_run(deletion_env):
    factory, roots, tmp_path = deletion_env
    with factory() as db:
        remote_task, _, _ = add_task(db, roots, tmp_path, "task-remote-run")
        remote_run = add_run(db, remote_task, "run-remote")
        remote_run.execution_node_id = "remote-a"
        malformed_task, _, _ = add_task(db, roots, tmp_path, "task-wrong-path")
        add_run(db, malformed_task, "run-wrong", run_dir=tmp_path / "other-task" / "attempt-001")
        db.flush()
        with pytest.raises(ValueError, match="远程服务器"):
            build_local_task_plan(db, remote_task)
        with pytest.raises(ValueError, match="不匹配"):
            build_local_task_plan(db, malformed_task)


def test_current_task_mirror_cannot_supply_an_outside_run_directory(deletion_env):
    factory, roots, tmp_path = deletion_env
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, "task-outside-mirror")
        run = add_run(db, task, "run-outside-mirror")
        task.current_run_id = run.id
        task.run_dir = str(tmp_path / "outside" / "attempt-001")
        db.flush()
        with pytest.raises(ValueError, match="不匹配"):
            build_local_task_plan(db, task)


def test_queue_replaces_active_legacy_operation_without_a_frozen_plan(deletion_env):
    factory, roots, tmp_path = deletion_env
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, "task-old-delete-job")
        legacy = RemoteOperation(
            id="7" * 32,
            node_id="__local__",
            task_id=task.id,
            kind="local_delete",
            status="pending",
            payload={},
        )
        db.add(legacy)
        db.flush()
        replacement = queue_local_delete(db, task)
        db.commit()
        assert replacement.id != legacy.id
        assert legacy.status == "cancelled"
        assert replacement.payload["scope"] == "task"
        assert replacement.payload["plan"]["task_id"] == task.id


def test_whole_plan_allows_internal_checkpoint_but_blocks_other_task_reference(deletion_env):
    factory, roots, tmp_path = deletion_env
    task_id = "task-dependency"
    first_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    second_dir = roots["runs"] / "detect" / task_id / "attempt-002"
    checkpoint = first_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    second_dir.mkdir(parents=True)
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, task_id)
        first = add_run(db, task, "run-first", 1, run_dir=first_dir)
        second = add_run(db, task, "run-second", 2, run_dir=second_dir)
        second.checkpoint_path = str(checkpoint)
        db.flush()
        assert build_local_task_plan(db, task)["attempt_count"] == 2

        dependent, _, _ = add_task(db, roots, tmp_path, "task-dependent")
        dependent.model_path = str(checkpoint)
        db.flush()
        with pytest.raises(ValueError, match="引用"):
            build_local_task_plan(db, task)


@pytest.mark.parametrize(("deleted", "referenced", "allowed"), [
    (False, False, False),
    (True, False, True),
    (True, True, False),
])
def test_model_asset_reference_policy(deletion_env, deleted, referenced, allowed):
    factory, roots, tmp_path = deletion_env
    task_id = f"task-asset-{int(deleted)}-{int(referenced)}"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    weight = run_dir / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, task_id)
        add_run(db, task, run_dir=run_dir)
        asset = ModelAsset(
            id="1" * 32,
            name="registered weight",
            source_type="local_path",
            execution_node_id="local",
            path=str(weight),
            status="ready",
            deleted_at=utcnow() if deleted else None,
        )
        db.add(asset)
        if referenced:
            db.add(ModelTest(
                id="2" * 32,
                task_id=None,
                model_asset=asset,
                artifact_name="best.pt",
                model_name="registered weight",
                model_source_type="local_path",
                execution_node_id="local",
                model_source_path=str(weight),
                confidence=0.25,
                iou=0.7,
                status="completed",
            ))
        db.flush()
        if allowed:
            assert build_local_task_plan(db, task)["task_id"] == task.id
        else:
            message = "历史引用" if deleted else "模型来源"
            with pytest.raises(ValueError, match=message):
                build_local_task_plan(db, task)


def test_relative_model_asset_path_is_resolved_from_project_root(deletion_env, monkeypatch):
    factory, roots, tmp_path = deletion_env
    task_id = "task-relative-asset"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    weight = run_dir / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    monkeypatch.setattr(model_asset_service, "PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "different-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, task_id)
        add_run(db, task, run_dir=run_dir)
        db.add(ModelAsset(
            id="6" * 32,
            name="portable weight",
            source_type="local_path",
            execution_node_id="local",
            path=weight.relative_to(tmp_path).as_posix(),
            status="ready",
        ))
        db.flush()
        with pytest.raises(ValueError, match="模型来源"):
            build_local_task_plan(db, task)


def test_external_symlink_alias_to_weight_blocks_task_deletion(deletion_env):
    factory, roots, tmp_path = deletion_env
    task_id = "task-symlink-reference"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    weight = run_dir / "weights" / "best.pt"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"weight")
    alias = tmp_path / "resources" / "alias.pt"
    alias.parent.mkdir(exist_ok=True)
    alias.symlink_to(weight)
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, task_id)
        add_run(db, task, run_dir=run_dir)
        dependent, _, _ = add_task(db, roots, tmp_path, "task-alias-dependent")
        dependent.model_path = str(alias)
        db.flush()
        with pytest.raises(ValueError, match="引用"):
            build_local_task_plan(db, task)


def test_imported_and_flat_layouts_are_deleted_idempotently(deletion_env):
    factory, roots, tmp_path = deletion_env
    imported_dir = roots["ultralytics"] / "runs" / "detect" / "train9"
    imported_dir.mkdir(parents=True)
    (imported_dir / "results.csv").write_text("epoch\n", encoding="utf-8")
    imported_id = "legacy-" + hashlib.sha1(str(imported_dir.resolve()).encode()).hexdigest()[:12]
    flat_id = "task-flat"
    flat_dir = roots["runs"] / "detect" / flat_id
    (flat_dir / "weights").mkdir(parents=True)
    (flat_dir / "weights" / "best.pt").write_bytes(b"weight")

    with factory() as db:
        imported, _, _ = add_task(db, roots, tmp_path, imported_id, source="imported")
        imported.run_dir = str(imported_dir)
        imported_run = add_run(db, imported, "run-imported", run_dir=imported_dir)
        imported.current_run_id = imported_run.id
        flat, _, _ = add_task(db, roots, tmp_path, flat_id)
        flat.run_dir = str(flat_dir)
        flat_run = add_run(db, flat, "run-flat", run_dir=flat_dir)
        flat.current_run_id = flat_run.id
        db.flush()
        imported_plan = build_local_task_plan(db, imported)
        flat_plan = build_local_task_plan(db, flat)

    first = execute_local_task_plan(imported_plan)
    second = execute_local_task_plan(imported_plan)
    execute_local_task_plan(flat_plan)
    assert str(imported_dir) in first["deleted_targets"]
    assert second["deleted_targets"] == []
    assert not imported_dir.exists()
    assert not flat_dir.exists()


def test_changed_plan_fails_without_replacing_training_error(deletion_env):
    factory, roots, tmp_path = deletion_env
    task_id = "task-plan-changed"
    run_dir = roots["runs"] / "detect" / task_id / "attempt-001"
    run_dir.mkdir(parents=True)
    with factory() as db:
        task, _, _ = add_task(db, roots, tmp_path, task_id)
        task.error_message = "original training failure"
        add_run(db, task, run_dir=run_dir)
        operation = queue_local_delete(db, task)
        db.commit()
        operation_id = operation.id
        db.add(ModelTest(
            id="3" * 32,
            task_id=task.id,
            artifact_name="best.pt",
            model_name="best.pt",
            confidence=0.25,
            iou=0.7,
            status="completed",
        ))
        db.commit()

    assert LocalDeletionWorker(factory).process_once()
    with factory() as db:
        task = db.get(TrainingTask, task_id)
        operation = db.get(RemoteOperation, operation_id)
        assert task.deleted_reason == "local_delete_failed"
        assert task.error_message == "original training failure"
        assert operation.status == "failed"
        assert "发生变化" in operation.error
    assert run_dir.is_dir()


def test_expired_lease_is_reclaimed_but_live_process_lease_is_not(deletion_env):
    factory, _, _ = deletion_env
    expired_lease = "e" * 32
    live_lease = "f" * 32
    with factory() as db:
        db.add_all([
            RemoteOperation(
                id="4" * 32,
                node_id="__local__",
                task_id="missing-expired",
                kind="local_delete",
                status="running",
                lease=expired_lease,
                payload={"scope": "task"},
                updated_at=utcnow() - timedelta(hours=1),
            ),
            RemoteOperation(
                id="5" * 32,
                node_id="__local__",
                task_id="missing-live",
                kind="local_delete",
                status="running",
                lease=live_lease,
                payload={"scope": "task"},
                updated_at=utcnow() - timedelta(hours=1),
            ),
        ])
        db.commit()
    with local_deletions._active_leases_lock:
        local_deletions._active_leases.add(live_lease)
    try:
        claim = LocalDeletionWorker(factory)._claim()
        assert claim is not None
        assert claim[0] == "4" * 32
        with factory() as db:
            assert db.get(RemoteOperation, "5" * 32).lease == live_lease
    finally:
        with local_deletions._active_leases_lock:
            local_deletions._active_leases.discard(live_lease)
            if claim:
                local_deletions._active_leases.discard(claim[2])
