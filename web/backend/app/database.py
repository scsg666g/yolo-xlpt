from __future__ import annotations

import os
import sqlite3
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


_MODEL_TEST_COLUMNS = (
    "id", "task_id", "run_id", "source_run_id", "artifact_id", "model_asset_id",
    "model_source_type", "execution_node_id", "model_source_path", "artifact_name",
    "model_name", "model_sha256", "model_size", "rerun_of_test_id", "attempt_number",
    "source_attempt_number", "status", "confidence", "iou", "imgsz", "device",
    "model_snapshot", "timings", "error", "storage_bytes", "delete_requested",
    "created_at", "started_at", "finished_at", "updated_at",
)

_MODEL_ASSET_COLUMNS = {
    "id", "name", "source_type", "execution_node_id", "path", "artifact_id",
    "sha256", "size", "status", "error", "created_at", "updated_at",
}
_MODEL_ASSET_INDEXES = {
    "ix_model_assets_source_type",
    "ix_model_assets_execution_node_id",
    "ix_model_assets_artifact_id",
    "ix_model_assets_status",
}
_MODEL_ASSET_UNIQUE_COLUMNS = ("source_type", "execution_node_id", "path")
_MODEL_ASSET_UNIQUE_INDEX = "uq_model_asset_source_path_idx"


def _backup_sqlite_database(database: Path) -> Path:
    """Create a transactionally consistent migration backup, including WAL pages."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup = database.with_name(f"{database.name}.backup-{stamp}")
    temporary = backup.with_suffix(backup.suffix + ".tmp")
    source = destination = None
    try:
        source = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)
        destination = sqlite3.connect(temporary)
        source.backup(destination)
        result = destination.execute("PRAGMA integrity_check").fetchone()
        if not result or result[0] != "ok":
            raise RuntimeError("数据库迁移备份完整性检查失败")
        destination.close()
        destination = None
        source.close()
        source = None
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(backup)
    finally:
        if destination is not None:
            destination.close()
        if source is not None:
            source.close()
        temporary.unlink(missing_ok=True)
    return backup


def _create_model_assets_table(connection, *, add_unique_index: bool = False) -> None:
    connection.execute(text("""
        CREATE TABLE IF NOT EXISTS model_assets (
            id VARCHAR(32) NOT NULL PRIMARY KEY,
            name VARCHAR(255) NOT NULL,
            source_type VARCHAR(32) NOT NULL,
            execution_node_id VARCHAR(64) NOT NULL,
            path TEXT NOT NULL,
            artifact_id INTEGER,
            sha256 VARCHAR(64),
            size INTEGER,
            status VARCHAR(24) NOT NULL DEFAULT 'ready',
            error TEXT,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            CONSTRAINT uq_model_asset_source_path UNIQUE (source_type, execution_node_id, path),
            FOREIGN KEY(artifact_id) REFERENCES artifacts (id) ON DELETE SET NULL
        )
    """))
    for name, column in (
        ("ix_model_assets_source_type", "source_type"),
        ("ix_model_assets_execution_node_id", "execution_node_id"),
        ("ix_model_assets_artifact_id", "artifact_id"),
        ("ix_model_assets_status", "status"),
    ):
        connection.execute(text(f"CREATE INDEX IF NOT EXISTS {name} ON model_assets ({column})"))
    if add_unique_index:
        connection.execute(text(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {_MODEL_ASSET_UNIQUE_INDEX} "
            "ON model_assets (source_type, execution_node_id, path)"
        ))


def _rebuild_model_tests_for_standalone(existing_columns: set[str]) -> None:
    """Make task ownership optional while preserving every legacy history row."""
    expressions = {
        "id": "id",
        "task_id": "task_id",
        "run_id": "run_id" if "run_id" in existing_columns else "NULL",
        "source_run_id": "source_run_id" if "source_run_id" in existing_columns else "NULL",
        "artifact_id": "artifact_id" if "artifact_id" in existing_columns else "NULL",
        "model_asset_id": "model_asset_id" if "model_asset_id" in existing_columns else "NULL",
        "model_source_type": "COALESCE(model_source_type, 'training_artifact')" if "model_source_type" in existing_columns else "'training_artifact'",
        "execution_node_id": "execution_node_id" if "execution_node_id" in existing_columns else "NULL",
        "model_source_path": "model_source_path" if "model_source_path" in existing_columns else "NULL",
        "artifact_name": "COALESCE(artifact_name, '')" if "artifact_name" in existing_columns else "''",
        "model_name": "COALESCE(model_name, '')" if "model_name" in existing_columns else "''",
        "model_sha256": "model_sha256" if "model_sha256" in existing_columns else "NULL",
        "model_size": "model_size" if "model_size" in existing_columns else "NULL",
        "rerun_of_test_id": "rerun_of_test_id" if "rerun_of_test_id" in existing_columns else "NULL",
        "attempt_number": "attempt_number" if "attempt_number" in existing_columns else "NULL",
        "source_attempt_number": "source_attempt_number" if "source_attempt_number" in existing_columns else "NULL",
        "status": "COALESCE(status, 'cancelled')" if "status" in existing_columns else "'cancelled'",
        "confidence": "COALESCE(confidence, 0.25)" if "confidence" in existing_columns else "0.25",
        "iou": "COALESCE(iou, 0.7)" if "iou" in existing_columns else "0.7",
        "imgsz": "COALESCE(imgsz, 640)" if "imgsz" in existing_columns else "640",
        "device": "COALESCE(device, 'CPU')" if "device" in existing_columns else "'CPU'",
        "model_snapshot": "COALESCE(model_snapshot, '{}')" if "model_snapshot" in existing_columns else "'{}'",
        "timings": "timings" if "timings" in existing_columns else "NULL",
        "error": "error" if "error" in existing_columns else "NULL",
        "storage_bytes": "COALESCE(storage_bytes, 0)" if "storage_bytes" in existing_columns else "0",
        "delete_requested": "COALESCE(delete_requested, 0)" if "delete_requested" in existing_columns else "0",
        "created_at": "created_at",
        "started_at": "started_at" if "started_at" in existing_columns else "NULL",
        "finished_at": "finished_at" if "finished_at" in existing_columns else "NULL",
        "updated_at": "COALESCE(updated_at, created_at)" if "updated_at" in existing_columns else "created_at",
    }
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=OFF")
        connection.commit()
        try:
            with connection.begin():
                connection.execute(text("DROP TABLE IF EXISTS model_tests_standalone_new"))
                connection.execute(text("""
                    CREATE TABLE model_tests_standalone_new (
                        id VARCHAR(32) NOT NULL PRIMARY KEY,
                        task_id VARCHAR(64),
                        run_id VARCHAR(64),
                        source_run_id VARCHAR(64),
                        artifact_id INTEGER,
                        model_asset_id VARCHAR(32),
                        model_source_type VARCHAR(32) NOT NULL DEFAULT 'training_artifact',
                        execution_node_id VARCHAR(64),
                        model_source_path TEXT,
                        artifact_name VARCHAR(255) NOT NULL,
                        model_name VARCHAR(255) NOT NULL,
                        model_sha256 VARCHAR(64),
                        model_size INTEGER,
                        rerun_of_test_id VARCHAR(32),
                        attempt_number INTEGER,
                        source_attempt_number INTEGER,
                        status VARCHAR(24) NOT NULL,
                        confidence FLOAT NOT NULL,
                        iou FLOAT NOT NULL,
                        imgsz INTEGER NOT NULL DEFAULT 640,
                        device VARCHAR(32) NOT NULL DEFAULT 'CPU',
                        model_snapshot JSON NOT NULL,
                        timings JSON,
                        error TEXT,
                        storage_bytes INTEGER NOT NULL DEFAULT 0,
                        delete_requested BOOLEAN NOT NULL DEFAULT 0,
                        created_at DATETIME NOT NULL,
                        started_at DATETIME,
                        finished_at DATETIME,
                        updated_at DATETIME NOT NULL,
                        FOREIGN KEY(task_id) REFERENCES training_tasks (id) ON DELETE CASCADE,
                        FOREIGN KEY(run_id) REFERENCES training_runs (id) ON DELETE SET NULL,
                        FOREIGN KEY(source_run_id) REFERENCES training_runs (id) ON DELETE SET NULL,
                        FOREIGN KEY(artifact_id) REFERENCES artifacts (id) ON DELETE SET NULL,
                        FOREIGN KEY(model_asset_id) REFERENCES model_assets (id) ON DELETE SET NULL
                    )
                """))
                columns = ", ".join(_MODEL_TEST_COLUMNS)
                values = ", ".join(expressions[column] for column in _MODEL_TEST_COLUMNS)
                connection.execute(text(
                    f"INSERT INTO model_tests_standalone_new ({columns}) SELECT {values} FROM model_tests"
                ))
                connection.execute(text("DROP TABLE model_tests"))
                connection.execute(text("ALTER TABLE model_tests_standalone_new RENAME TO model_tests"))
                indexes = (
                    ("ix_model_tests_task_id", "task_id"),
                    ("ix_model_tests_run_id", "run_id"),
                    ("ix_model_tests_source_run_id", "source_run_id"),
                    ("ix_model_tests_artifact_id", "artifact_id"),
                    ("ix_model_tests_model_asset_id", "model_asset_id"),
                    ("ix_model_tests_model_source_type", "model_source_type"),
                    ("ix_model_tests_execution_node_id", "execution_node_id"),
                    ("ix_model_tests_rerun_of_test_id", "rerun_of_test_id"),
                    ("ix_model_tests_status", "status"),
                    ("ix_model_tests_created_at", "created_at"),
                    ("ix_model_tests_task_created_id", "task_id, created_at, id"),
                    ("ix_model_tests_created_id", "created_at, id"),
                )
                for name, columns_sql in indexes:
                    connection.execute(text(f"CREATE INDEX {name} ON model_tests ({columns_sql})"))
        finally:
            connection.exec_driver_sql("PRAGMA foreign_keys=ON")
            connection.commit()


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
    model_test_column_details = (
        inspector.get_columns("model_tests") if inspector.has_table("model_tests") else []
    )
    model_test_columns = {column["name"] for column in model_test_column_details}
    model_test_task_nullable = next(
        (bool(column.get("nullable")) for column in model_test_column_details if column["name"] == "task_id"),
        False,
    )
    model_test_indexes = (
        {index["name"] for index in inspector.get_indexes("model_tests")}
        if model_test_columns else set()
    )
    group_table_exists = inspector.has_table("task_groups")
    model_assets_table_exists = inspector.has_table("model_assets")
    model_asset_columns = (
        {column["name"] for column in inspector.get_columns("model_assets")}
        if model_assets_table_exists else set()
    )
    model_asset_index_details = (
        inspector.get_indexes("model_assets") if model_assets_table_exists else []
    )
    model_asset_indexes = {index["name"] for index in model_asset_index_details}
    model_asset_unique_constraints = (
        {
            tuple(constraint.get("column_names") or ())
            for constraint in inspector.get_unique_constraints("model_assets")
        }
        if model_assets_table_exists else set()
    )
    model_asset_unique_layouts = model_asset_unique_constraints | {
        tuple(index.get("column_names") or ())
        for index in model_asset_index_details
        if index.get("unique")
    }
    model_asset_unique_present = _MODEL_ASSET_UNIQUE_COLUMNS in model_asset_unique_layouts
    missing_model_asset_columns = _MODEL_ASSET_COLUMNS - model_asset_columns
    model_assets_need_repair = (
        not model_assets_table_exists
        or bool(missing_model_asset_columns)
        or not _MODEL_ASSET_INDEXES.issubset(model_asset_indexes)
        or not model_asset_unique_present
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
        or "group_id" not in task_columns
        or not group_table_exists
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
        or (bool(model_test_columns) and "ix_model_tests_created_id" not in model_test_indexes)
        or (bool(model_test_columns) and "ix_model_tests_rerun_of_test_id" not in model_test_indexes)
        or (bool(model_test_columns) and not model_test_task_nullable)
        or (bool(model_test_columns) and not set(_MODEL_TEST_COLUMNS).issubset(model_test_columns))
        or model_assets_need_repair
    )
    if not needs_migration:
        return None

    backup = _backup_sqlite_database(database)

    if model_assets_need_repair:
        if model_assets_table_exists and missing_model_asset_columns:
            missing = ", ".join(sorted(missing_model_asset_columns))
            raise RuntimeError(f"model_assets 缺少必需字段：{missing}；已保留迁移前备份 {backup}")
        with engine.begin() as connection:
            _create_model_assets_table(
                connection,
                add_unique_index=model_assets_table_exists and not model_asset_unique_present,
            )
    if model_test_columns and (
        not model_test_task_nullable or not set(_MODEL_TEST_COLUMNS).issubset(model_test_columns)
    ):
        _rebuild_model_tests_for_standalone(model_test_columns)
        model_test_columns = set(_MODEL_TEST_COLUMNS)
        model_test_indexes = {
            "ix_model_tests_task_created_id", "ix_model_tests_created_id",
            "ix_model_tests_rerun_of_test_id",
        }

    with engine.begin() as connection:
        if not group_table_exists:
            connection.execute(text("""
                CREATE TABLE task_groups (
                    id VARCHAR(64) NOT NULL PRIMARY KEY,
                    name VARCHAR(160) NOT NULL,
                    display_order INTEGER NOT NULL DEFAULT 0,
                    created_at DATETIME NOT NULL,
                    updated_at DATETIME NOT NULL
                )
            """))
            connection.execute(text("CREATE INDEX ix_task_groups_name ON task_groups (name)"))
            connection.execute(text("CREATE INDEX ix_task_groups_display_order ON task_groups (display_order)"))
        if "group_id" not in task_columns:
            connection.execute(text("ALTER TABLE training_tasks ADD COLUMN group_id VARCHAR(64)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_training_tasks_group_id ON training_tasks (group_id)"))
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
                "CREATE INDEX IF NOT EXISTS ix_model_tests_created_id "
                "ON model_tests (created_at, id)"
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
