"""Durable background cleanup for local training-task files."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta, timezone

from sqlalchemy import select, update

from .database import SessionLocal
from .dataset_uploads import managed_dataset_mutation
from .gpu_reservations import acquire_node_claim_lock
from .model_test_storage import delete_task_history
from .models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask, utcnow
from .run_deletions import (
    delete_attempt_files,
    delete_attempt_records,
    is_attempt_payload,
    validate_attempt_deletion,
)
from .services import add_event
from .task_deletions import (
    build_local_task_plan,
    cleanup_task_test_files,
    execute_local_task_plan,
)


logger = logging.getLogger("yolo-platform.local-deletions")
LOCAL_NODE_ID = "__local__"
LOCAL_COMPUTE_NODE_ID = "local"
LEASE_TIMEOUT = timedelta(minutes=15)
HEARTBEAT_SECONDS = 30
ACTIVE_OPERATION_STATUSES = {"pending", "running"}
_active_leases: set[str] = set()
_active_leases_lock = threading.Lock()


def _whole_task_operation(operation: RemoteOperation) -> bool:
    return (operation.payload or {}).get("scope") != "attempt"


def queue_local_delete(db, task: TrainingTask) -> RemoteOperation:
    """Validate and queue an immutable whole-task cleanup plan."""
    operations = list(db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id,
        RemoteOperation.kind == "local_delete",
    ).order_by(RemoteOperation.created_at.desc())))
    existing = next((operation for operation in operations if _whole_task_operation(operation)), None)
    if existing and existing.status in ACTIVE_OPERATION_STATUSES:
        payload = existing.payload or {}
        if payload.get("scope") == "task" and isinstance(payload.get("plan"), dict):
            return existing
        existing.status = "cancelled"
        existing.error = "旧版本机删除任务缺少冻结计划，已由新任务替换"
        existing.updated_at = utcnow()

    plan = build_local_task_plan(db, task)
    node = db.get(ComputeNode, LOCAL_COMPUTE_NODE_ID)
    operation = RemoteOperation(
        id=uuid.uuid4().hex,
        node_id=LOCAL_NODE_ID,
        task_id=task.id,
        kind="local_delete",
        status="pending",
        payload={
            "scope": "task",
            "phase": "stopping_tests",
            "task_name": task.name,
            "node_name": node.name if node else "本机服务器",
            "test_ids": plan["test_ids"],
            "plan": plan,
            "receipt": None,
        },
    )
    db.add(operation)
    task.deleted_at = task.deleted_at or utcnow()
    task.deleted_reason = "local_delete_pending"
    add_event(db, task.id, "local_delete_queued", "已提交本机训练文件清理")
    return operation


def queue_local_run_delete(db, task: TrainingTask, run: TrainingRun) -> RemoteOperation:
    """Queue cleanup for one local Attempt without moving the task to trash."""
    validate_attempt_deletion(db, task, run)
    operation = RemoteOperation(
        id=uuid.uuid4().hex,
        node_id=LOCAL_NODE_ID,
        task_id=task.id,
        kind="local_delete",
        status="pending",
        payload={
            "scope": "attempt",
            "run_id": run.id,
            "attempt_number": run.attempt_number,
            "run_dir": run.run_dir,
            "log_path": run.log_path,
        },
    )
    db.add(operation)
    add_event(
        db,
        task.id,
        "run_delete_queued",
        f"已提交本机 Attempt {run.attempt_number:03d} 文件清理",
        run_id=run.id,
    )
    return operation


def _as_utc(value):
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _phase_payload(payload: dict | None, phase: str, **values) -> dict:
    updated = dict(payload or {})
    updated.update(values)
    updated["phase"] = phase
    return updated


class LocalDeletionWorker:
    def __init__(self, session_factory=SessionLocal, interval: float = 2.0):
        self.session_factory = session_factory
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="local-deletion-worker",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def _claim(self) -> tuple[str, str, str, dict] | None:
        """Claim pending work or reclaim a lease whose owner has disappeared."""
        with self.session_factory() as db:
            now = utcnow()
            candidates = list(db.scalars(select(RemoteOperation).where(
                RemoteOperation.node_id == LOCAL_NODE_ID,
                RemoteOperation.kind == "local_delete",
                RemoteOperation.status.in_(ACTIVE_OPERATION_STATUSES),
            ).order_by(RemoteOperation.created_at)))
            for operation in candidates:
                if operation.status == "running":
                    with _active_leases_lock:
                        active_here = bool(operation.lease and operation.lease in _active_leases)
                    if active_here or _as_utc(operation.updated_at) >= now - LEASE_TIMEOUT:
                        continue
                old_status = operation.status
                old_updated_at = operation.updated_at
                lease = uuid.uuid4().hex
                payload = dict(operation.payload or {})
                claimed = db.execute(update(RemoteOperation).where(
                    RemoteOperation.id == operation.id,
                    RemoteOperation.status == old_status,
                    RemoteOperation.updated_at == old_updated_at,
                ).values(
                    status="running",
                    lease=lease,
                    error=None,
                    payload=payload,
                    updated_at=now,
                ))
                if claimed.rowcount != 1:
                    db.rollback()
                    continue
                db.commit()
                with _active_leases_lock:
                    _active_leases.add(lease)
                return operation.id, operation.task_id, lease, payload
        return None

    def _heartbeat(self, operation_id: str, lease: str, phase: str | None = None) -> bool:
        """Refresh ownership between cleanup phases without accepting a stale worker."""
        with self.session_factory() as db:
            operation = db.get(RemoteOperation, operation_id)
            if not operation or operation.status != "running" or operation.lease != lease:
                return False
            values = {"updated_at": utcnow()}
            if phase:
                values["payload"] = _phase_payload(operation.payload, phase)
            changed = db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation_id,
                RemoteOperation.status == "running",
                RemoteOperation.lease == lease,
            ).values(**values))
            db.commit()
            return changed.rowcount == 1

    @contextmanager
    def _lease_keepalive(self, operation_id: str, lease: str):
        """Keep long recursive filesystem cleanup from looking abandoned."""
        stopped = threading.Event()

        def refresh() -> None:
            while not stopped.wait(HEARTBEAT_SECONDS):
                try:
                    if not self._heartbeat(operation_id, lease):
                        return
                except Exception:
                    logger.warning(
                        "local_delete_heartbeat_failed operation_id=%s",
                        operation_id,
                        exc_info=True,
                    )

        thread = threading.Thread(
            target=refresh,
            name=f"local-delete-heartbeat-{operation_id[:8]}",
            daemon=True,
        )
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=2)

    @staticmethod
    def _leased_operation(db, operation_id: str, lease: str) -> RemoteOperation:
        operation = db.get(RemoteOperation, operation_id)
        if not operation or operation.status != "running" or operation.lease != lease:
            raise RuntimeError("本机删除任务租约已失效，停止继续清理")
        return operation

    def _finish_error(
        self,
        operation_id: str,
        task_id: str,
        lease: str,
        error: str,
        *,
        attempt: bool,
    ) -> None:
        last_error: Exception | None = None
        for retry in range(3):
            try:
                with self.session_factory() as db:
                    operation = db.get(RemoteOperation, operation_id)
                    if not operation or operation.status != "running" or operation.lease != lease:
                        return
                    operation.status = "failed"
                    operation.error = error[-2000:]
                    operation.payload = _phase_payload(operation.payload, "failed")
                    operation.updated_at = utcnow()
                    task = db.get(TrainingTask, task_id)
                    if task:
                        if attempt:
                            add_event(db, task.id, "run_delete_failed", f"Attempt 文件清理失败：{error}", "error")
                        else:
                            task.deleted_reason = "local_delete_failed"
                            add_event(db, task.id, "local_delete_failed", f"本机文件清理失败：{error}", "error")
                    db.commit()
                return
            except Exception as exc:
                last_error = exc
                if retry < 2:
                    time.sleep(0.2 * (retry + 1))
        raise RuntimeError(f"删除失败结果无法写入数据库：{last_error}") from last_error

    def _process_attempt(self, operation_id: str, task_id: str, lease: str, payload: dict) -> None:
        run_id = payload.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("待清理 Attempt 标识无效")
        with managed_dataset_mutation():
            with self.session_factory() as db:
                acquire_node_claim_lock(db, LOCAL_COMPUTE_NODE_ID)
                operation = self._leased_operation(db, operation_id, lease)
                task = db.get(TrainingTask, task_id)
                run = db.get(TrainingRun, run_id)
                if not task:
                    raise ValueError("待清理任务记录不存在")
                if not run or run.task_id != task.id:
                    raise ValueError("待清理 Attempt 记录不存在")
                validate_attempt_deletion(db, task, run, ignore_operation_id=operation.id)
                operation.payload = _phase_payload(operation.payload, "deleting_files")
                operation.updated_at = utcnow()
                db.flush()
                deleted = delete_attempt_files(task, run)
                delete_attempt_records(db, task.id, run.id)
                operation.status = "succeeded"
                operation.error = None
                operation.payload = _phase_payload(
                    operation.payload,
                    "complete",
                    receipt={
                        "deleted_targets": sorted(deleted),
                        "completed_at": utcnow().isoformat(),
                    },
                )
                operation.updated_at = utcnow()
                add_event(
                    db,
                    task.id,
                    "run_deleted",
                    f"Attempt {payload.get('attempt_number', '')} 已删除",
                )
                db.commit()

    def _process_whole_task(self, operation_id: str, task_id: str, lease: str, payload: dict) -> None:
        from .inference import cancel_task_tests

        with self.session_factory() as bind_session:
            bind = bind_session.get_bind()
        self._heartbeat(operation_id, lease, "stopping_tests")
        cancel_task_tests(task_id, purge=True, bind=bind)

        with managed_dataset_mutation():
            if not self._heartbeat(operation_id, lease, "validating"):
                raise RuntimeError("本机删除任务租约已失效，停止继续清理")
            with self.session_factory() as db:
                acquire_node_claim_lock(db, LOCAL_COMPUTE_NODE_ID)
                operation = self._leased_operation(db, operation_id, lease)
                task = db.get(TrainingTask, task_id)
                if task is None:
                    raise ValueError("待清理任务记录不存在")
                current_plan = build_local_task_plan(
                    db,
                    task,
                    ignore_operation_id=operation.id,
                )
                frozen_plan = (operation.payload or {}).get("plan")
                frozen_test_ids = sorted((operation.payload or {}).get("test_ids") or [])
                if current_plan != frozen_plan or current_plan["test_ids"] != frozen_test_ids:
                    raise ValueError("任务文件或引用关系在排队后发生变化，请重新提交永久删除")
                # End the node claim before recursive filesystem work.  The
                # cross-process mutation guard remains held, so no local route
                # can create a new reference to the validated targets.
                db.commit()

            if not self._heartbeat(operation_id, lease, "deleting_files"):
                raise RuntimeError("本机删除任务租约已失效，停止继续清理")
            with self._lease_keepalive(operation_id, lease):
                receipt = execute_local_task_plan(current_plan)

                if not self._heartbeat(operation_id, lease, "verifying"):
                    raise RuntimeError("本机删除任务租约已失效，停止继续清理")
                cleaned_test_ids = cleanup_task_test_files(
                    task_id,
                    current_plan["test_ids"],
                    delete_history=delete_task_history,
                )
            receipt.update({
                "cleaned_test_ids": cleaned_test_ids,
                "test_history_removed": True,
                "completed_at": utcnow().isoformat(),
            })

            # Persist the verified receipt independently.  If final record
            # cleanup is interrupted, a retry can prove that physical cleanup
            # completed without guessing from missing paths.
            with self.session_factory() as db:
                operation = self._leased_operation(db, operation_id, lease)
                operation.payload = _phase_payload(
                    operation.payload,
                    "recording",
                    receipt=receipt,
                )
                operation.updated_at = utcnow()
                db.commit()

            with self.session_factory() as db:
                operation = self._leased_operation(db, operation_id, lease)
                task = db.get(TrainingTask, task_id)
                if task is None:
                    raise ValueError("待清理任务记录不存在")
                operation.status = "succeeded"
                operation.error = None
                operation.payload = _phase_payload(operation.payload, "complete", receipt=receipt)
                operation.updated_at = utcnow()
                db.delete(task)
                db.commit()

    def process_one(self) -> bool:
        """Process remote center cleanup first, then one leased local deletion."""
        from .remote_operations import process_remote_delete_center

        if process_remote_delete_center(self.session_factory):
            return True
        claim = self._claim()
        if claim is None:
            return False
        operation_id, task_id, lease, payload = claim
        attempt = is_attempt_payload(payload, payload.get("run_id", ""))
        try:
            if attempt:
                self._process_attempt(operation_id, task_id, lease, payload)
            else:
                self._process_whole_task(operation_id, task_id, lease, payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                self._finish_error(
                    operation_id,
                    task_id,
                    lease,
                    error,
                    attempt=attempt,
                )
            except Exception:
                logger.exception("local_delete_finish_error operation_id=%s", operation_id)
            logger.warning("local_delete_failed task_id=%s error=%s", task_id, error)
        finally:
            with _active_leases_lock:
                _active_leases.discard(lease)
        return True

    def process_once(self) -> bool:
        """Compatibility alias used by deterministic worker tests."""
        return self.process_one()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.process_one()
            except Exception:
                logger.exception("local_delete_worker_error")


local_deletion_worker = LocalDeletionWorker()
