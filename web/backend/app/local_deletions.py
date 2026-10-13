"""Durable background cleanup for local training-task files."""
from __future__ import annotations

import logging
import threading
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from .database import SessionLocal
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

    def process_one(self) -> bool:
        """Claim one operation, remove files without holding a DB transaction, then record result."""
        task_snapshot = None
        operation_id = None
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

        error = None
        if task_snapshot is None:
            error = "待清理任务记录不存在"
        else:
            try:
                permanently_delete_task_files(task_snapshot)
            except Exception as exc:
                # A malformed path or an unexpected filesystem error must become
                # a visible failed operation, not an invisible "running" hang.
                error = f"{type(exc).__name__}: {exc}"

        with self.session_factory() as db:
            operation = db.get(RemoteOperation, operation_id)
            task = db.get(TrainingTask, task_snapshot.id) if task_snapshot else None
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
