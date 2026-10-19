"""Durable background cleanup for local training-task files."""
from __future__ import annotations

import logging
import threading
import time
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from .database import SessionLocal
from .model_test_storage import delete_task_history
from .models import RemoteOperation, TrainingTask, utcnow
from .services import add_event, permanently_delete_task_files

logger = logging.getLogger("yolo-platform.local-deletions")
LOCAL_NODE_ID = "__local__"


def queue_local_delete(db, task: TrainingTask) -> RemoteOperation:
    existing = db.scalar(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id,
        RemoteOperation.kind == "local_delete",
        RemoteOperation.status.in_(["pending", "running"]),
    ))
    if existing:
        return existing
    operation = RemoteOperation(
        id=uuid.uuid4().hex, node_id=LOCAL_NODE_ID, task_id=task.id,
        kind="local_delete", status="pending", payload={},
    )
    db.add(operation)
    task.deleted_at = task.deleted_at or utcnow()
    task.deleted_reason = "local_delete_pending"
    task.error_message = None
    add_event(db, task.id, "local_delete_queued", "已提交本机训练文件清理")
    return operation


class LocalDeletionWorker:
    def __init__(self, session_factory=SessionLocal, interval: float = 2.0):
        self.session_factory = session_factory
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        # File removal is idempotent. Resume a cleanup interrupted by a process restart.
        with self.session_factory() as db:
            db.execute(update(RemoteOperation).where(
                RemoteOperation.node_id == LOCAL_NODE_ID,
                RemoteOperation.kind == "local_delete",
                RemoteOperation.status == "running",
            ).values(status="pending", updated_at=utcnow()))
            db.commit()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="local-deletion-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def is_alive(self) -> bool:
        """Expose worker health without exposing the thread object to callers."""
        return bool(self._thread and self._thread.is_alive())

    def _finish(self, operation_id: str, task_id: str | None, error: str | None) -> None:
        """Persist the result, retrying transient SQLite locks before giving up."""
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self.session_factory() as db:
                    operation = db.get(RemoteOperation, operation_id)
                    task = db.get(TrainingTask, task_id) if task_id else None
                    if operation:
                        operation.status = "failed" if error else "succeeded"
                        operation.error = error
                        operation.updated_at = utcnow()
                    if task:
                        if error:
                            task.deleted_reason = "local_delete_failed"
                            task.error_message = f"本机文件清理失败：{error}"
                            add_event(db, task.id, "local_delete_failed", task.error_message, "error")
                        else:
                            add_event(db, task.id, "local_delete_succeeded", "本机训练文件清理完成")
                            db.delete(task)
                    db.commit()
                return
            except Exception as exc:  # SQLite can be busy while the API commits a request.
                last_error = exc
                if attempt < 2:
                    time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"删除结果无法写入数据库：{last_error}") from last_error

    def _requeue(self, operation_id: str, reason: str) -> None:
        """Make a claimed operation retryable after an unexpected worker error."""
        try:
            with self.session_factory() as db:
                operation = db.get(RemoteOperation, operation_id)
                if operation and operation.status == "running":
                    operation.status = "pending"
                    operation.error = reason
                    operation.updated_at = utcnow()
                    db.commit()
        except Exception:
            logger.exception("local_delete_requeue_error operation_id=%s", operation_id)

    def process_one(self) -> bool:
        """Claim one operation, remove files without holding a DB transaction, then record result."""
        task_snapshot = None
        operation_id = None
        try:
            with self.session_factory() as db:
                operation = db.scalar(select(RemoteOperation).where(
                    RemoteOperation.node_id == LOCAL_NODE_ID,
                    RemoteOperation.kind == "local_delete",
                    RemoteOperation.status == "pending",
                ).order_by(RemoteOperation.created_at).limit(1))
                if not operation:
                    return False
                claim = db.execute(update(RemoteOperation).where(
                    RemoteOperation.id == operation.id,
                    RemoteOperation.status == "pending",
                ).values(status="running", updated_at=utcnow()))
                if claim.rowcount != 1:
                    db.rollback()
                    return False
                operation_id = operation.id
                db.commit()
                task_snapshot = db.scalar(select(TrainingTask).where(
                    TrainingTask.id == operation.task_id,
                ).options(selectinload(TrainingTask.runs)))
                if task_snapshot:
                    db.expunge(task_snapshot)
        except Exception:
            # A failure after claiming must not strand the row in "running".
            if operation_id:
                self._requeue(operation_id, "删除任务领取失败，已重新排队")
            raise

        error = None
        if task_snapshot is None:
            error = "待清理任务记录不存在"
        else:
            try:
                permanently_delete_task_files(task_snapshot)
                delete_task_history(task_snapshot.id)
            except Exception as exc:
                # A malformed path or an unexpected filesystem error must become
                # a visible failed operation, not an invisible "running" hang.
                error = f"{type(exc).__name__}: {exc}"

        try:
            self._finish(operation_id, task_snapshot.id if task_snapshot else None, error)
        except Exception as exc:
            # The file operation is idempotent, so retrying is safer than leaving
            # the UI in an unbounded "cleaning" state after a DB write failure.
            self._requeue(operation_id, f"删除结果写回失败，将自动重试：{exc}")
            raise
        if error:
            logger.warning("local_delete_failed task_id=%s error=%s", task_snapshot.id if task_snapshot else "missing", error)
        return True

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                if not self.process_one():
                    continue
            except Exception:
                logger.exception("local_delete_worker_error")


local_deletion_worker = LocalDeletionWorker()
