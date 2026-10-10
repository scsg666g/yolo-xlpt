from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def migrate_database() -> Path | None:
    """Upgrade the original task-only SQLite schema without touching run files."""
    if engine.dialect.name != "sqlite" or not inspect(engine).has_table("training_tasks"):
        return None
    database = Path(str(engine.url.database)).resolve()
    inspector = inspect(engine)
    task_columns = {column["name"] for column in inspector.get_columns("training_tasks")}
    metric_columns = {column["name"] for column in inspector.get_columns("metric_points")}
    artifact_columns = {column["name"] for column in inspector.get_columns("artifacts")}
    event_columns = {column["name"] for column in inspector.get_columns("task_events")}
    display_order_needs_repair = False
    if "display_order" in task_columns:
        with engine.connect() as connection:
            display_order_needs_repair = bool(connection.scalar(text(
                "SELECT 1 FROM training_tasks WHERE display_order IS NULL OR display_order = 0 LIMIT 1"
            )))
    needs_migration = (
        "display_order" not in task_columns
        or display_order_needs_repair
        or "current_run_id" not in task_columns
        or "deleted_at" not in task_columns
        or "deleted_reason" not in task_columns
        or "run_id" not in metric_columns
        or "run_id" not in artifact_columns
        or "run_id" not in event_columns
    )
    if not needs_migration:
        return None

    backup = database.with_name(f"{database.name}.backup-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    if database.exists():
        shutil.copy2(database, backup)

    with engine.begin() as connection:
        if "display_order" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN display_order INTEGER NOT NULL DEFAULT 0"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_tasks_display_order ON training_tasks (display_order)"))
        # Existing cards keep their current newest-first order. Repair rows left at
        # the SQL default as well, which covers databases migrated by an older build.
        connection.execute(text("""
            WITH ranked AS (
                SELECT id, (ROW_NUMBER() OVER (
                    ORDER BY
                        CASE WHEN display_order IS NULL OR display_order = 0 THEN 1 ELSE 0 END,
                        display_order,
                        created_at DESC,
                        id DESC
                ) * 1000) AS position
                FROM training_tasks
            )
            UPDATE training_tasks
            SET display_order = (SELECT position FROM ranked WHERE ranked.id = training_tasks.id)
            WHERE display_order IS NULL OR display_order = 0
        """))
        if "current_run_id" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN current_run_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_tasks_current_run_id ON training_tasks (current_run_id)"))
        if "deleted_at" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN deleted_at DATETIME"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_tasks_deleted_at ON training_tasks (deleted_at)"))
        if "deleted_reason" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN deleted_reason VARCHAR(40)"))
        if "run_id" not in artifact_columns:
            connection.execute(text("ALTER TABLE artifacts ADD COLUMN run_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_artifacts_run_id ON artifacts (run_id)"))
        if "run_id" not in event_columns:
            connection.execute(text("ALTER TABLE task_events ADD COLUMN run_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_task_events_run_id ON task_events (run_id)"))
        if "run_id" not in metric_columns:
            connection.execute(text("PRAGMA foreign_keys=OFF"))
            connection.execute(text("""
                CREATE TABLE metric_points_new (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    task_id VARCHAR(64) NOT NULL,
                    run_id VARCHAR(64),
                    epoch INTEGER NOT NULL,
                    elapsed FLOAT NOT NULL,
                    "values" JSON NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_metric_run_epoch UNIQUE (run_id, epoch),
                    FOREIGN KEY(task_id) REFERENCES training_tasks (id),
                    FOREIGN KEY(run_id) REFERENCES training_runs (id)
                )
            """))
            connection.execute(text("INSERT INTO metric_points_new (id, task_id, epoch, elapsed, \"values\", created_at) SELECT id, task_id, epoch, elapsed, \"values\", created_at FROM metric_points"))
            connection.execute(text("DROP TABLE metric_points"))
            connection.execute(text("ALTER TABLE metric_points_new RENAME TO metric_points"))
            connection.execute(text("CREATE INDEX ix_metric_points_task_id ON metric_points (task_id)"))
            connection.execute(text("CREATE INDEX ix_metric_points_run_id ON metric_points (run_id)"))
            connection.execute(text("PRAGMA foreign_keys=ON"))
    return backup


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
