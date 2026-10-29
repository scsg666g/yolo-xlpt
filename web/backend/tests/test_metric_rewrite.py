"""A rewritten results.csv replaces only the current Attempt's stale epochs."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

import app.compute_nodes as compute_nodes
from app.database import Base
from app.models import MetricPoint, TrainingRun, TrainingTask
from app.services import sync_metrics


def _training(db, run_dir, *, task_id, run_id, node_id=None):
    task = TrainingTask(id=task_id, name=task_id, task_type="detect", status="running",
                        model_path="model.pt", data_path="data.yaml",
                        execution_node_id=node_id)
    run = TrainingRun(id=run_id, task=task, attempt_number=1, launch_mode="fresh",
                      status="running", execution_node_id=node_id,
                      model_path=task.model_path, data_path=task.data_path,
                      run_dir=str(run_dir), parameter_snapshot={"epochs": 10})
    task.current_run_id = run.id
    db.add_all([task, run])
    db.commit()
    return task, run


def test_local_rewrite_removes_stale_epochs_only_from_current_attempt(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'local.db'}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            run_dir = tmp_path / "attempt"
            run_dir.mkdir()
            csv = run_dir / "results.csv"
            csv.write_text("epoch,time,score\n1,1,0.1\n2,2,0.2\n3,3,0.3\n")
            task, run = _training(db, run_dir, task_id="local-rewrite", run_id="local-run")
            _, other = _training(db, tmp_path / "other", task_id="other-local", run_id="other-run")
            db.add(MetricPoint(task_id=other.task_id, run_id=other.id, epoch=3,
                               elapsed=3, values={"score": 0.7}))
            sync_metrics(db, task, run)
            db.commit()
            csv.write_text("epoch,time,score\n1,4,0.9\n")
            sync_metrics(db, task, run)
            db.commit()
            points = db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id)).all()
            assert [(point.epoch, point.values) for point in points] == [(1, {"score": 0.9})]
            assert db.scalar(select(MetricPoint).where(MetricPoint.run_id == other.id)).epoch == 3
            assert run.completed_epochs == task.completed_epochs == 1
            assert run.latest_metrics == task.latest_metrics == {"score": 0.9}
            csv.unlink()
            sync_metrics(db, task, run)
            db.commit()
            assert len(db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id)).all()) == 1
    finally:
        engine.dispose()


def test_remote_reset_is_optional_and_scoped_to_one_attempt(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'remote.db'}")
    Base.metadata.create_all(engine)
    monkeypatch.setattr(compute_nodes, "authenticated_agent", lambda node_id, request, db: None)
    try:
        with Session(engine) as db:
            node_id = "remote-node"
            task, run = _training(db, "/remote/attempt", task_id="remote-rewrite",
                                  run_id="remote-run", node_id=node_id)
            _, old_run = _training(db, "/remote/old", task_id="old-remote",
                                   run_id="old-run", node_id=node_id)
            db.add(MetricPoint(task_id=old_run.task_id, run_id=old_run.id, epoch=3,
                               elapsed=3, values={"score": 0.7}))
            db.commit()
            request = SimpleNamespace(headers={})

            def update(payload):
                return compute_nodes.update_remote_run(
                    node_id, run.id, compute_nodes.RemoteRunUpdate(**payload), request, db)

            update({"status": "running", "completed_epochs": 3, "progress": 30,
                    "latest_metrics": {"score": 0.3},
                    "metrics": [{"epoch": epoch, "values": {"score": epoch / 10}}
                                for epoch in (1, 2, 3)]})
            update({"status": "running", "completed_epochs": 1, "progress": 10,
                    "latest_metrics": {"score": 0.9}, "metrics_reset": True,
                    "metrics": [{"epoch": 1, "values": {"score": 0.9}}]})
            points = db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id)).all()
            assert [(point.epoch, point.values) for point in points] == [(1, {"score": 0.9})]
            assert db.scalar(select(MetricPoint).where(MetricPoint.run_id == old_run.id)).epoch == 3
            assert run.completed_epochs == task.completed_epochs == 1
            with pytest.raises(HTTPException) as invalid:
                update({"status": "running", "completed_epochs": 1,
                        "metrics_reset": True,
                        "metrics": [{"epoch": 99, "values": {"score": 1}}]})
            assert invalid.value.status_code == 422
            assert db.scalar(select(MetricPoint).where(MetricPoint.run_id == run.id)).epoch == 1
            # Legacy Agent updates omit the optional flag and retain the old upsert behavior.
            update({"status": "running", "completed_epochs": 2,
                    "metrics": [{"epoch": 2, "values": {"score": 0.5}}]})
            assert [point.epoch for point in db.scalars(
                select(MetricPoint).where(MetricPoint.run_id == run.id).order_by(MetricPoint.epoch))] == [1, 2]
    finally:
        engine.dispose()
