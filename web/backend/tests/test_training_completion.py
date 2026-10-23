import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import app.services as services
from app.database import Base
from app.models import GpuReservation, TaskEvent, TrainingRun, TrainingTask, utcnow
from app.services import TrainingScheduler


@pytest.fixture
def completed_attempt(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'completion.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    started = utcnow() - timedelta(minutes=5)
    directory = tmp_path / "attempt-001"
    (directory / "weights").mkdir(parents=True)
    (directory / "weights" / "best.pt").write_bytes(b"checkpoint")
    (directory / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n56,300,0.8016\n")
    log = tmp_path / "attempt-001.log"
    log.write_text(
        "[training-trace] collection disabled: OutOfMemoryError: CUDA out of memory\n"
        "EarlyStopping: Training stopped early as no improvement observed in last 35 epochs. "
        "Best results observed at epoch 21, best model saved as best.pt.\n"
        "56 epochs completed in 25.666 hours.\n"
        "Optimizer stripped from weights/best.pt\nValidating weights/best.pt...\n"
        f"Results saved to {directory}\n"
    )
    with factory() as db:
        task = TrainingTask(id="task", name="smoke", task_type="detect", model_path="model.pt", data_path="data.yaml",
                            status="running", epochs=100, current_run_id="run")
        run = TrainingRun(id="run", task=task, attempt_number=1, model_path=task.model_path, data_path=task.data_path,
                          status="running", run_dir=str(directory), log_path=str(log), parameter_snapshot={"epochs": 100},
                          process_started_at=started, started_at=started, error_message="stale recovery error")
        db.add_all([task, run, GpuReservation(resource_key="local:GPU-0", run_id=run.id, node_id="local")])
        db.commit()
    scheduler = TrainingScheduler(session_factory=factory)
    monkeypatch.setattr(services, "gpu_inventory", lambda: [])
    monkeypatch.setattr(scheduler, "_find_run_process", lambda _: None)
    yield SimpleNamespace(factory=factory, scheduler=scheduler, directory=directory, log=log, started=started)
    engine.dispose()


def write_receipt(attempt, **changes):
    record = dict(schema_version=1, run_id="run", task_id="task", status="completed", exit_code=0,
                  reason="early_stopping", completed_epochs=56, expected_epochs=100, best_epoch=21, patience=35,
                  finished_at=(utcnow() - timedelta(seconds=1)).isoformat(), final_metrics={"metrics/mAP50(B)": 0.818})
    record.update(changes)
    (attempt.directory / "training-result.json").write_text(json.dumps(record))


@pytest.mark.parametrize("adopted,with_receipt", [(True, False), (True, True), (False, True)])
def test_scheduler_early_stopping_clears_error_and_releases_gpu(completed_attempt, adopted, with_receipt):
    attempt = completed_attempt
    if with_receipt:
        write_receipt(attempt)
    attempt.scheduler.processes["task"] = SimpleNamespace(run_id="run", adopted=adopted, poll=lambda: 0)
    with attempt.factory() as db:
        attempt.scheduler._poll_processes(db)
        db.commit()
        run, task = db.get(TrainingRun, "run"), db.get(TrainingTask, "task")
        assert run.status == task.status == "completed"
        assert run.error_message is task.error_message is None
        assert run.completed_epochs == task.completed_epochs == 56
        assert run.progress == 100
        assert run.exit_code == (0 if with_receipt else None)
        assert run.completion_info == task.completion_info
        assert run.completion_info["reason"] == "early_stopping"
        assert run.completion_info["best_epoch"] == 21
        assert run.latest_metrics["metrics/mAP50(B)"] == 0.8016
        assert not db.scalars(select(GpuReservation)).all()
        assert db.scalar(select(TaskEvent)).payload["completion_info"]["patience"] == 35
        services.sync_metrics(db, task, run)
        assert run.progress == task.progress == 100
        assert run.completed_epochs == 56
    assert not attempt.scheduler.processes


def test_metrics_refresh_uses_actual_training_epoch_target(completed_attempt):
    with completed_attempt.factory() as db:
        run, task = db.get(TrainingRun, "run"), db.get(TrainingTask, "task")
        run.status = "failed"
        run.completion_info = {"expected_epochs": 200}
        services.sync_metrics(db, task, run)
        assert run.progress == task.progress == 28
        assert run.completed_epochs == 56


@pytest.mark.parametrize("historical_error", [None, "平台服务重启后未发现受管训练进程", "平台接管的训练进程已退出，无法取得原进程退出码"])
def test_recovery_reconciles_finished_or_misclassified_attempt_once(completed_attempt, historical_error):
    attempt = completed_attempt
    with attempt.factory() as db:
        run = db.get(TrainingRun, "run")
        if historical_error:
            run.status = "interrupted"
            run.error_message = historical_error
        db.commit()
    attempt.scheduler._recover_processes()
    attempt.scheduler._recover_processes()
    with attempt.factory() as db:
        run = db.get(TrainingRun, "run")
        assert run.status == db.get(TrainingTask, "task").status == "completed"
        assert run.exit_code is None
        assert run.completion_info["reason"] == "early_stopping"
        events = db.scalars(select(TaskEvent)).all()
        assert len(events) == 1
        assert events[0].event_type == ("reconciled" if historical_error else "completed")
        assert not db.scalars(select(GpuReservation)).all()
    assert not attempt.scheduler.processes


def test_historical_reconciliation_does_not_replace_new_attempt(completed_attempt):
    attempt = completed_attempt
    with attempt.factory() as db:
        run, task = db.get(TrainingRun, "run"), db.get(TrainingTask, "task")
        run.status, run.error_message = "interrupted", "平台服务重启后未发现受管训练进程"
        new_run = TrainingRun(id="new-run", task=task, attempt_number=2, status="queued", model_path="model.pt",
                              data_path="data.yaml", parameter_snapshot={"epochs": 100})
        task.current_run_id, task.status = new_run.id, "queued"
        db.add(new_run)
        db.commit()
    attempt.scheduler._recover_processes()
    with attempt.factory() as db:
        assert db.get(TrainingRun, "run").status == "completed"
        task = db.get(TrainingTask, "task")
        assert (task.current_run_id, task.status, task.completion_info) == ("new-run", "queued", {})


@pytest.mark.parametrize("kind", ["remote", "deleted", "manual_stop", "other_interruption"])
def test_recovery_does_not_rewrite_unrelated_terminal_attempts(completed_attempt, kind):
    attempt = completed_attempt
    with attempt.factory() as db:
        run, task = db.get(TrainingRun, "run"), db.get(TrainingTask, "task")
        run.status = "interrupted"
        run.error_message = "平台服务重启后未发现受管训练进程"
        if kind == "remote":
            run.execution_node_id = "remote-node"
        elif kind == "deleted":
            task.deleted_at = utcnow()
        elif kind == "manual_stop":
            run.stop_requested = True
        else:
            run.error_message = "人工标记中断"
        db.commit()
    attempt.scheduler._recover_processes()
    with attempt.factory() as db:
        assert db.get(TrainingRun, "run").status == "interrupted"
        assert not db.scalars(select(TaskEvent)).all()


@pytest.mark.parametrize("recover", [False, True])
def test_stop_request_takes_precedence_over_completion_receipt(completed_attempt, recover):
    attempt = completed_attempt
    write_receipt(attempt)
    with attempt.factory() as db:
        run = db.get(TrainingRun, "run")
        run.status, run.stop_requested = "stopping", True
        db.commit()
    if recover:
        attempt.scheduler._recover_processes()
    else:
        attempt.scheduler.processes["task"] = SimpleNamespace(
            run_id="run", adopted=True, pid=1234, command=[], started_at=attempt.started, poll=lambda: 0
        )
        with attempt.factory() as db:
            attempt.scheduler._poll_processes(db)
            db.commit()
    with attempt.factory() as db:
        run = db.get(TrainingRun, "run")
        assert run.status == db.get(TrainingTask, "task").status == "stopped"
        assert run.completion_info == {}


@pytest.mark.parametrize("adopted", [False, True])
def test_validation_failure_cannot_be_hidden_by_epochs_or_checkpoints(completed_attempt, adopted):
    attempt = completed_attempt
    attempt.log.write_text(attempt.log.read_text() + "Traceback (most recent call last):\nRuntimeError: final validation failed\n")
    attempt.scheduler.processes["task"] = SimpleNamespace(run_id="run", adopted=adopted, poll=lambda: 1)
    with attempt.factory() as db:
        attempt.scheduler._poll_processes(db)
        assert db.get(TrainingRun, "run").status == db.get(TrainingTask, "task").status == "failed"
        assert "final validation failed" in db.get(TrainingRun, "run").error_message


@pytest.mark.parametrize("task_type", ["detect", "segment", "classify", "pose", "obb"])
def test_all_local_training_commands_use_result_runner(tmp_path, task_type):
    task = TrainingTask(id="task", name="test", task_type=task_type, model_path="model.pt", data_path="data.yaml",
                        epochs=10, batch=1, imgsz=640, device="cpu", workers=0, optimizer="auto")
    run = TrainingRun(id="run", task_id=task.id, attempt_number=1, model_path="model.pt", data_path="data.yaml",
                      run_dir=str(tmp_path / "attempt-001"), parameter_snapshot={"epochs": 10, "config": {}})
    command = TrainingScheduler()._command(task, run)
    assert Path(command[1]).name == "training_trace.py"
    assert ("--disable-trace" in command) == (task_type != "detect")
    assert task_type in command
