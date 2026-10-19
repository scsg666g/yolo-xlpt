from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from sqlalchemy import event, inspect, text
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False, "timeout": 5.0} if is_sqlite else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)


def configure_sqlite_connection(connection, _record=None) -> None:
    """Enable concurrent readers and a bounded wait for SQLite writers."""
    cursor = connection.cursor()
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


if is_sqlite:
    event.listen(engine, "connect", configure_sqlite_connection)

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
    run_columns = {column["name"] for column in inspector.get_columns("training_runs")} if inspector.has_table("training_runs") else set()
    model_test_columns = (
        {column["name"] for column in inspector.get_columns("model_tests")}
        if inspector.has_table("model_tests") else set()
    )
    model_test_indexes = (
        {index["name"] for index in inspector.get_indexes("model_tests")}
        if model_test_columns else set()
    )
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
        or "execution_node_id" not in task_columns
        or (bool(run_columns) and "execution_node_id" not in run_columns)
        or (bool(run_columns) and "stop_requested" not in run_columns)
        or (bool(run_columns) and "remote_log_tail" not in run_columns)
        or (bool(run_columns) and "claim_lease" not in run_columns)
        or (bool(run_columns) and "claim_expires_at" not in run_columns)
        or (bool(run_columns) and "claim_confirmed" not in run_columns)
        or (bool(model_test_columns) and "model_sha256" not in model_test_columns)
        or (bool(model_test_columns) and "model_size" not in model_test_columns)
        or (bool(model_test_columns) and "rerun_of_test_id" not in model_test_columns)
        or (bool(model_test_columns) and "ix_model_tests_task_created_id" not in model_test_indexes)
        or (bool(model_test_columns) and "ix_model_tests_rerun_of_test_id" not in model_test_indexes)
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
        if "execution_node_id" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN execution_node_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_tasks_execution_node_id ON training_tasks (execution_node_id)"))
        if run_columns and "execution_node_id" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN execution_node_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_runs_execution_node_id ON training_runs (execution_node_id)"))
        if run_columns and "stop_requested" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN stop_requested BOOLEAN NOT NULL DEFAULT 0"))
        if run_columns and "remote_log_tail" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN remote_log_tail JSON NOT NULL DEFAULT '[]'"))
        if run_columns and "claim_lease" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN claim_lease VARCHAR(32)"))
        if run_columns and "claim_expires_at" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN claim_expires_at DATETIME"))
        if run_columns and "claim_confirmed" not in run_columns:
            connection.execute(text("ALTER TABLE training_runs ADD COLUMN claim_confirmed BOOLEAN NOT NULL DEFAULT 0"))
        # Existing cards keep their current newest-first order. Repair rows left at
        # the SQL default as well, which covers databases migrated by an older build.
        order_timestamp = "created_at DESC," if "created_at" in task_columns else ""
        connection.execute(text(f"""
            WITH ranked AS (
                SELECT id, (ROW_NUMBER() OVER (
                    ORDER BY
                        CASE WHEN display_order IS NULL OR display_order = 0 THEN 1 ELSE 0 END,
                        display_order,
                        {order_timestamp}
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
        if model_test_columns and "model_sha256" not in model_test_columns:
            connection.execute(text("ALTER TABLE model_tests ADD COLUMN model_sha256 VARCHAR(64)"))
        if model_test_columns and "model_size" not in model_test_columns:
            connection.execute(text("ALTER TABLE model_tests ADD COLUMN model_size INTEGER"))
        if model_test_columns and "rerun_of_test_id" not in model_test_columns:
            connection.execute(text("ALTER TABLE model_tests ADD COLUMN rerun_of_test_id VARCHAR(32)"))
        if model_test_columns:
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_model_tests_task_created_id "
                "ON model_tests (task_id, created_at, id)"
            ))
            connection.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_model_tests_rerun_of_test_id "
                "ON model_tests (rerun_of_test_id)"
            ))
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
