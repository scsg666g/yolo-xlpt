"""Connection freshness must never rewrite training outcomes or reservations."""
from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database import Base
from app.main import serialize_run
from app.models import ComputeNode, TrainingTask, utcnow
from app.services import create_training_run


@pytest.mark.parametrize("case,expected", [
    ("local", "local"), ("online", "online"),
    ("offline", "reconnecting"), ("never_connected", "reconnecting"),
    ("disabled", "unavailable"), ("historical", None),
])
def test_run_monitoring_is_derived_and_backward_compatible(tmp_path, case, expected):
    engine = create_engine(f"sqlite:///{tmp_path / 'monitoring.db'}")
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            node = None
            if case != "local":
                node = ComputeNode(id="b" * 32, name="remote", kind="agent",
                                   enabled=case != "disabled", report_interval_seconds=5,
                                   agent_version="1.8.0", snapshot={})
                if case != "never_connected":
                    node.last_seen_at = utcnow() - timedelta(seconds=90 if case == "offline" else 1)
                db.add(node)
            task = TrainingTask(id="monitor-test", name="test", task_type="detect",
                                status="draft", model_path="model.pt", data_path="data.yaml",
                                execution_node_id=node.id if node else None)
            db.add(task)
            db.flush()
            run = create_training_run(db, task, "fresh")
            run.status = "completed" if case == "historical" else "running"
            run.completed_epochs = 7
            run.progress = 35
            db.commit()
            data = serialize_run(db, run)
            assert data.monitoring_status == expected
            assert data.completed_epochs == 7 and data.progress == 35
            assert run.status == ("completed" if case == "historical" else "running")
            if node and case != "historical":
                node.last_seen_at = utcnow()
                node.enabled = True
                db.flush()
                assert serialize_run(db, run).monitoring_status == "online"
    finally:
        engine.dispose()
