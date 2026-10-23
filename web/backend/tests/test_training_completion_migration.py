import json
import sqlite3

from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.orm import sessionmaker

import app.database as database_module
from app.database import Base
from app.models import TrainingRun, TrainingTask


def test_completion_metadata_upgrade_preserves_legacy_history_and_is_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "legacy-completion.db"
    engine = create_engine(f"sqlite:///{path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as db:
        task = TrainingTask(id="task", name="smoke", task_type="detect", model_path="model.pt", data_path="data.yaml",
                            status="interrupted", completed_epochs=56, display_order=1000,
                            latest_metrics={"metrics/mAP50(B)": 0.8016}, config={"patience": 35})
        run = TrainingRun(id="run", task=task, attempt_number=1, status="interrupted", model_path="model.pt",
                          data_path="data.yaml", completed_epochs=56, parameter_snapshot={"epochs": 100},
                          error_message="original interruption", run_dir="runs/attempt-001")
        task.current_run_id = run.id
        db.add_all([task, run])
        db.commit()
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE training_tasks DROP COLUMN completion_info"))
        connection.execute(text("ALTER TABLE training_runs DROP COLUMN completion_info"))
    monkeypatch.setattr(database_module, "engine", engine)

    backup = database_module.migrate_database()

    assert backup is not None and backup.is_file()
    for table in ("training_tasks", "training_runs"):
        column = next(item for item in inspect(engine).get_columns(table) if item["name"] == "completion_info")
        assert not column["nullable"]
    with sqlite3.connect(backup) as saved:
        assert "completion_info" not in {row[1] for row in saved.execute("PRAGMA table_info(training_runs)")}
        assert saved.execute("SELECT status, completed_epochs FROM training_runs WHERE id='run'").fetchone() == ("interrupted", 56)
    with factory() as db:
        task = db.scalar(select(TrainingTask).where(TrainingTask.id == "task"))
        run = db.scalar(select(TrainingRun).where(TrainingRun.id == "run"))
        assert task.completion_info == run.completion_info == {}
        assert task.latest_metrics == {"metrics/mAP50(B)": 0.8016}
        assert task.config == {"patience": 35}
        assert (task.status, task.current_run_id, run.status, run.completed_epochs) == ("interrupted", "run", "interrupted", 56)
        assert (run.run_dir, run.error_message) == ("runs/attempt-001", "original interruption")
        task.completion_info = run.completion_info = {"source": "legacy_log", "reason": "early_stopping"}
        db.commit()

    assert database_module.migrate_database() is None
    assert len(list(tmp_path.glob("legacy-completion.db.backup-*"))) == 1
    with engine.connect() as connection:
        for table in ("training_tasks", "training_runs"):
            stored = connection.scalar(text(f"SELECT completion_info FROM {table}"))
            assert json.loads(stored) == {"source": "legacy_log", "reason": "early_stopping"}
    engine.dispose()
