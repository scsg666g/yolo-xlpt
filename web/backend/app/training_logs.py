"""Download full local logs or verified snapshots fetched by the remote Agent."""
import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .compute_nodes import agent_version_at_least
from .config import LOG_ROOT, RUNS_ROOT, ULTRALYTICS_ROOT, nonnegative_int_env
from .database import get_db
from .gpu_reservations import acquire_node_claim_lock
from .models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask
from .remote_operations import cached_fetch_digest, enqueue, operation_file, run_payload
from .run_deletions import attempt_delete_operation
from .services import path_is_within

router = APIRouter()
LOG_AGENT_VERSION = "2.0.6"
MAX_LOG_BYTES = nonnegative_int_env("PLATFORM_LOG_DOWNLOAD_MAX_BYTES", 256 * 1024**2)
ACTIVE_FETCH_STATUSES = frozenset({"pending", "running", "finalizing"})


def log_context(db: Session, task_id: str, run_id: str):
    task, run = db.get(TrainingTask, task_id), db.get(TrainingRun, run_id)
    if not task or task.deleted_at or not run or run.task_id != task.id:
        raise HTTPException(404, "训练运行不存在")
    if attempt_delete_operation(db, run, active_only=True):
        raise HTTPException(409, "该训练运行正在清理，不能下载日志")
    if not run.log_path:
        raise HTTPException(404, "该训练运行尚未产生日志")
    return task, run


def local_log(run: TrainingRun) -> Path:
    path = Path(run.log_path).expanduser()
    if path.is_symlink() or not path_is_within(path.resolve(), [LOG_ROOT, RUNS_ROOT, ULTRALYTICS_ROOT / "runs"]):
        raise HTTPException(403, "日志路径不在允许的训练目录内")
    if not path.is_file():
        raise HTTPException(404, "日志文件不存在")
    if MAX_LOG_BYTES and path.stat().st_size > MAX_LOG_BYTES:
        raise HTTPException(413, "日志超过下载大小限制，请调整 PLATFORM_LOG_DOWNLOAD_MAX_BYTES 或在服务器读取")
    return path


def effective_node_id(task: TrainingTask, run: TrainingRun) -> str | None:
    """Keep historical remote Runs usable when only their Task stored the node."""
    return run.execution_node_id or task.execution_node_id


def log_fetches(db: Session, run: TrainingRun) -> list[RemoteOperation]:
    return list(db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == run.task_id,
        RemoteOperation.kind == "fetch_log",
        RemoteOperation.payload["run_id"].as_string() == run.id,
    ).order_by(RemoteOperation.created_at.desc(), RemoteOperation.id.desc())))


def matches_log_source(operation: RemoteOperation, node_id: str, run: TrainingRun) -> bool:
    payload = operation.payload or {}
    return operation.node_id == node_id and payload.get("log_path") == run.log_path


def usable_snapshot(
    operations: list[RemoteOperation], node_id: str, run: TrainingRun,
) -> RemoteOperation | None:
    for operation in operations:
        if operation.status != "succeeded" or not matches_log_source(operation, node_id, run):
            continue
        digest, error = cached_fetch_digest(operation)
        if digest and not error:
            return operation
    return None


def active_fetch(
    operations: list[RemoteOperation], node_id: str, run: TrainingRun,
) -> RemoteOperation | None:
    return next((operation for operation in operations
                 if operation.status in ACTIVE_FETCH_STATUSES
                 and matches_log_source(operation, node_id, run)), None)


def latest_failure(
    operations: list[RemoteOperation], node_id: str, run: TrainingRun,
) -> RemoteOperation | None:
    return next((operation for operation in operations
                 if operation.status in {"failed", "cancelled"}
                 and matches_log_source(operation, node_id, run)), None)


def prune_terminal_fetches(
    db: Session, operations: list[RemoteOperation], keep_ids: set[str],
) -> bool:
    """Retain the selected snapshot and every operation that may still be running."""
    changed = False
    for operation in operations:
        if operation.id in keep_ids or operation.status in ACTIVE_FETCH_STATUSES:
            continue
        removable = True
        for path in (operation_file(operation), operation_file(operation, False)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                removable = False
        if removable:
            db.delete(operation)
            changed = True
    return changed


def ready_response(operation: RemoteOperation) -> dict:
    return {"status": "ready", "snapshot_at": operation.created_at}


def file_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def open_remote_snapshot(operation: RemoteOperation) -> tuple[BinaryIO, int]:
    """Open and verify the exact inode that the response will stream."""
    payload = operation.payload or {}
    expected_size = payload.get("size")
    expected_sha256 = payload.get("sha256")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        raise HTTPException(409, "远程日志缓存缺少有效大小，请重新获取")
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise HTTPException(409, "远程日志缓存摘要无效，请重新获取")
    descriptor = None
    handle = None
    try:
        descriptor = os.open(operation_file(operation), (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        ))
        handle = os.fdopen(descriptor, "rb")
        descriptor = None
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("远程日志缓存不是普通文件")
        if before.st_size != expected_size:
            raise ValueError("远程日志缓存大小校验失败")
        if MAX_LOG_BYTES and before.st_size > MAX_LOG_BYTES:
            raise HTTPException(413, "日志超过下载大小限制，请调整 PLATFORM_LOG_DOWNLOAD_MAX_BYTES 或在服务器读取")
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(512 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
        if file_identity(before) != file_identity(after):
            raise ValueError("远程日志缓存在校验期间发生变化")
        if expected_sha256 is not None and not secrets.compare_digest(digest.hexdigest(), expected_sha256):
            raise ValueError("远程日志缓存摘要校验失败")
        handle.seek(0)
        return handle, before.st_size
    except HTTPException:
        if handle is not None:
            handle.close()
        raise
    except (OSError, ValueError) as exc:
        if handle is not None:
            handle.close()
        message = str(exc) if isinstance(exc, ValueError) else "远程日志缓存文件不可用"
        raise HTTPException(409, message + "，请重新获取") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@router.post("/api/tasks/{task_id}/runs/{run_id}/logs/prepare")
def prepare_log(task_id: str, run_id: str, refresh: bool = False, db: Session = Depends(get_db)):
    task, run = log_context(db, task_id, run_id)
    node_id = effective_node_id(task, run)
    if not node_id:
        local_log(run)
        return {"status": "ready"}
    operations = log_fetches(db, run)
    operation = active_fetch(operations, node_id, run)
    if operation:
        return {"status": operation.status, "message": "正在获取远程完整日志快照"}
    snapshot = usable_snapshot(operations, node_id, run)
    if snapshot and not refresh:
        if prune_terminal_fetches(db, operations, {snapshot.id}):
            db.commit()
        return ready_response(snapshot)
    failure = latest_failure(operations, node_id, run)
    if failure and not snapshot and not refresh:
        return {"status": "failed", "message": failure.error or "日志传输失败，请重试"}
    node = db.get(ComputeNode, node_id)
    if not node or not node.enabled:
        raise HTTPException(409, "远程节点不存在或已禁用")
    if not agent_version_at_least(node.agent_version, (2, 0, 6)):
        raise HTTPException(409, f"远程完整日志下载需要 Agent {LOG_AGENT_VERSION} 或以上版本；仍可导出当前日志片段")
    acquire_node_claim_lock(db, node_id)
    db.expire_all()
    task, run = log_context(db, task_id, run_id)
    node_id = effective_node_id(task, run)
    if not node_id:
        local_log(run)
        return {"status": "ready"}
    operations = log_fetches(db, run)
    operation = active_fetch(operations, node_id, run)
    if operation:
        return {"status": operation.status, "message": "正在获取远程完整日志快照"}
    snapshot = usable_snapshot(operations, node_id, run)
    if snapshot and not refresh:
        if prune_terminal_fetches(db, operations, {snapshot.id}):
            db.commit()
        return ready_response(snapshot)
    failure = latest_failure(operations, node_id, run)
    if failure and not snapshot and not refresh:
        return {"status": "failed", "message": failure.error or "日志传输失败，请重试"}
    operation = enqueue(db, node_id, task.id, "fetch_log", {
        **run_payload(run, task), "max_bytes": MAX_LOG_BYTES,
    })
    keep_ids = {operation.id}
    if snapshot:
        keep_ids.add(snapshot.id)
    prune_terminal_fetches(db, operations, keep_ids)
    db.commit()
    return {"status": "pending", "message": "等待远程 Agent 传输完整日志快照"}


@router.get("/api/tasks/{task_id}/runs/{run_id}/logs/download")
def download_log(task_id: str, run_id: str, db: Session = Depends(get_db)):
    task, run = log_context(db, task_id, run_id)
    filename = f"{task_id}-attempt-{run.attempt_number:03d}.log"
    node_id = effective_node_id(task, run)
    if node_id:
        operations = log_fetches(db, run)
        operation = usable_snapshot(operations, node_id, run)
        if not operation:
            raise HTTPException(409, "请先准备远程完整日志")
        if prune_terminal_fetches(db, operations, {operation.id}):
            db.commit()
        handle, size = open_remote_snapshot(operation)

        def stream_remote():
            try:
                remaining = size
                while remaining:
                    block = handle.read(min(65536, remaining))
                    if not block:
                        break
                    remaining -= len(block)
                    yield block
            finally:
                handle.close()

        return StreamingResponse(stream_remote(), media_type="text/plain; charset=utf-8", headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
            "Content-Length": str(size),
        })
    path = local_log(run)
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        handle = os.fdopen(descriptor, "rb")
    except OSError as exc:
        raise HTTPException(409, "日志文件暂时无法读取") from exc
    metadata = os.fstat(handle.fileno())
    if not stat.S_ISREG(metadata.st_mode):
        handle.close()
        raise HTTPException(409, "日志必须是普通文件")
    if MAX_LOG_BYTES and metadata.st_size > MAX_LOG_BYTES:
        handle.close()
        raise HTTPException(413, "日志超过下载大小限制，请调整 PLATFORM_LOG_DOWNLOAD_MAX_BYTES 或在服务器读取")

    def stream():
        try:
            remaining = metadata.st_size
            while remaining:
                block = handle.read(min(65536, remaining))
                if not block:
                    break
                remaining -= len(block)
                yield block
        finally:
            handle.close()

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers={
        "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
    })
