"""Agent-initiated file operations. Never interpret remote paths as local paths."""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
import shutil
import stat
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import DATA_ROOT
from .compute_nodes import (
    REQUIRED_AGENT_VERSION,
    agent_version_at_least,
    as_utc,
    authenticated_agent,
    remote_trace_root,
)
from .database import get_db
from .model_test_storage import delete_task_history
from .models import Artifact, ComputeNode, RemoteOperation, TrainingRun, TrainingTask, utcnow
from .run_deletions import (
    LEGACY_FLAT_LAYOUT,
    attempt_delete_operation,
    delete_attempt_records,
    is_attempt_payload,
    remote_attempt_layout,
    remote_output_root_for_run,
    validate_attempt_deletion,
)
from .services import add_event, artifact_kind

router = APIRouter()
CACHE_ROOT = DATA_ROOT / "remote-artifacts"
ACTIVE = {"queued", "starting", "running", "stopping"}
LEGACY_FLAT_DELETE_AGENT_VERSION = "2.0.1"
LEGACY_FLAT_DELETE_AGENT_VERSION_PARTS = (2, 0, 1)


def operation_file(operation: RemoteOperation, complete: bool = True) -> Path:
    return CACHE_ROOT / (operation.id + (".bin" if complete else ".part"))


def cached_fetch_digest(
    operation: RemoteOperation,
    *,
    path: Path | None = None,
) -> tuple[str | None, str | None]:
    """Return a stable cache digest and any validation error without changing it."""
    payload = operation.payload or {}
    expected_size = payload.get("size")
    expected_sha256 = payload.get("sha256")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0:
        return None, "远程产物缓存缺少有效大小"
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        return None, "远程产物缓存摘要无效"
    path = path or operation_file(operation)
    if path.is_symlink():
        return None, "远程产物缓存路径无效"
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None, "远程产物缓存不是普通文件"
            if before.st_size != expected_size:
                return None, "远程产物缓存大小校验失败"
            digest = hashlib.sha256()
            for block in iter(lambda: handle.read(512 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
        current = path.lstat()
    except OSError:
        return None, "远程产物缓存文件不可用"
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if (
        not stat.S_ISREG(current.st_mode)
        or before_identity != after_identity
        or after_identity != current_identity
    ):
        return None, "远程产物缓存在校验期间发生变化"
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and not secrets.compare_digest(actual_sha256, expected_sha256):
        return None, "远程产物缓存摘要校验失败"
    return actual_sha256, None


def run_payload(run: TrainingRun, task: TrainingTask) -> dict:
    output_root = remote_output_root_for_run(task, run)
    layout = None
    if output_root:
        try:
            layout = remote_attempt_layout(task, run)[0]
        except ValueError:
            pass
    return {"task_id": task.id, "task_type": task.task_type, "run_id": run.id,
            "attempt_number": run.attempt_number, "run_dir": run.run_dir, "log_path": run.log_path,
            "output_root": output_root, **({"layout": layout} if layout else {})}


def enqueue(db: Session, node_id: str, task_id: str, kind: str, payload: dict) -> RemoteOperation:
    operation = RemoteOperation(id=uuid.uuid4().hex, node_id=node_id, task_id=task_id,
                                kind=kind, payload=payload, status="pending")
    db.add(operation)
    db.flush()
    return operation


def queue_scan(db: Session, task: TrainingTask, run: TrainingRun) -> None:
    if not run.execution_node_id or not run.run_dir or task.deleted_at:
        return
    try:
        if remote_attempt_layout(task, run)[0] == LEGACY_FLAT_LAYOUT:
            # Scanning the legacy task container would incorrectly attribute
            # newer attempt-NNN children to Attempt 001. Legacy outputs already
            # indexed by the center remain available until deletion.
            return
    except ValueError:
        return
    if any(
        operation.kind == "delete" and operation.status in {"pending", "running"}
        and is_attempt_payload(operation.payload, run.id)
        for operation in db.scalars(select(RemoteOperation).where(RemoteOperation.task_id == task.id))
    ):
        return
    latest = db.scalar(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id, RemoteOperation.kind == "scan",
        RemoteOperation.payload["run_id"].as_string() == run.id,
    ).order_by(RemoteOperation.created_at.desc()).limit(1))
    if latest and latest.payload.get("run_id") == run.id and (
        latest.status in {"pending", "running"} or
        (utcnow() - as_utc(latest.updated_at)).total_seconds() < (30 if run.status in ACTIVE else 120)
    ):
        return
    enqueue(db, run.execution_node_id, task.id, "scan", run_payload(run, task))


def queue_delete(db: Session, task: TrainingTask) -> None:
    existing = db.scalar(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id, RemoteOperation.kind == "delete",
        RemoteOperation.status.in_(["pending", "running"])))
    if existing:
        return
    # Freeze the exact set of Attempts; a pending deletion cannot be restored or launched.
    runs = [run_payload(run, task) for run in task.runs if run.execution_node_id]
    enqueue(db, task.execution_node_id, task.id, "delete", {
        "runs": runs,
        "protected_paths": live_protected_paths(db, task.execution_node_id, task.id),
    })
    task.deleted_at = task.deleted_at or utcnow()
    task.deleted_reason = "remote_delete_pending"
    add_event(db, task.id, "remote_delete_queued", "等待远程 Agent 清理该任务文件；确认后删除记录")


def queue_attempt_delete(db: Session, task: TrainingTask, run: TrainingRun) -> RemoteOperation:
    """Queue one remote Attempt for Agent cleanup while retaining its task."""
    validate_attempt_deletion(db, task, run)
    if not task.execution_node_id:
        raise ValueError("本机 Attempt 应使用本机删除队列")
    if run.execution_node_id not in {None, task.execution_node_id}:
        raise ValueError("Attempt 所属远程节点与任务不一致，已拒绝清理")
    node = db.get(ComputeNode, task.execution_node_id)
    if not node or not agent_version_at_least(node.agent_version):
        raise ValueError(f"远程 Agent 版本过低或节点不存在，请升级到 {REQUIRED_AGENT_VERSION} 或以上版本后重试")
    payload_run = run_payload(run, task)
    if not payload_run["output_root"]:
        raise ValueError("远程 Attempt 缺少可验证的输出根目录，请检查远程训练路径配置")
    if payload_run.get("layout") == LEGACY_FLAT_LAYOUT and not agent_version_at_least(
        node.agent_version, LEGACY_FLAT_DELETE_AGENT_VERSION_PARTS,
    ):
        raise ValueError(
            f"旧版平铺 Attempt 需要远程 Agent {LEGACY_FLAT_DELETE_AGENT_VERSION} "
            "或以上版本才能安全单独清理"
        )
    if active_remote_delete(db, task.execution_node_id):
        raise ValueError("该远程服务器正在清理其他任务文件，请稍后重试")
    payload = {
        "scope": "attempt",
        "run_id": run.id,
        "attempt_number": run.attempt_number,
        "runs": [payload_run],
        "protected_paths": live_protected_paths(db, task.execution_node_id, task.id, deleting_run_id=run.id),
    }
    operation = enqueue(db, task.execution_node_id, task.id, "delete", payload)
    add_event(db, task.id, "run_delete_queued", f"已提交远程 Attempt {run.attempt_number:03d} 文件清理", run_id=run.id)
    return operation


def live_protected_paths(
    db: Session,
    node_id: str,
    deleting_task_id: str,
    *,
    deleting_run_id: str | None = None,
) -> list[str]:
    """Return current references that a queued remote deletion must preserve."""
    protected: set[str] = set()
    tasks = db.scalars(select(TrainingTask).where(TrainingTask.execution_node_id == node_id))
    for task in tasks:
        if task.id == deleting_task_id and deleting_run_id is None:
            continue
        # Task resources are shared by all its Attempts and must never be
        # removed by an Attempt-scoped operation.
        protected.update(path for path in (task.model_path, task.data_path) if path)
        for run in task.runs:
            if task.id == deleting_task_id and deleting_run_id and run.id == deleting_run_id:
                continue
            protected.update(path for path in (run.model_path, run.data_path, run.checkpoint_path) if path)
            protected.update(
                artifact.path for artifact in db.scalars(select(Artifact).where(Artifact.run_id == run.id))
            )
    return sorted(protected)


def active_remote_delete(db: Session, node_id: str) -> RemoteOperation | None:
    """Return active deletion work for a node, if any.

    Remote paths may contain symlink aliases that the center cannot resolve.
    Callers must therefore serialize new references at node scope instead of
    comparing path strings with the directories in a deletion payload.
    """
    return db.scalar(select(RemoteOperation).where(
        RemoteOperation.node_id == node_id,
        RemoteOperation.kind == "delete",
        RemoteOperation.status.in_(["pending", "running"]),
    ).order_by(RemoteOperation.created_at).limit(1))


class ArtifactItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(ge=0, le=8 * 1024**3)
    modified_at: datetime


class Manifest(BaseModel):
    items: list[ArtifactItem] = Field(max_length=1024)


@router.post("/api/compute/nodes/{node_id}/agent/runs/{run_id}/artifacts")
def report_artifacts(node_id: str, run_id: str, payload: Manifest, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id or not run.run_dir:
        raise HTTPException(404, "远程运行不存在")
    task = db.get(TrainingTask, run.task_id)
    if not task or task.deleted_at:
        raise HTTPException(409, "任务已进入回收站，停止同步产物")
    if attempt_delete_operation(db, run, active_only=True):
        raise HTTPException(409, "该 Attempt 已进入清理队列，停止同步产物")
    paths = []
    for item in payload.items:
        relative = PurePosixPath(item.name)
        if relative.is_absolute() or str(relative) != item.name or any(part in {".", ".."} or part.startswith(".") for part in relative.parts) or "\\" in item.name:
            raise HTTPException(422, "产物名称必须是运行目录内的相对路径")
        if item.modified_at.tzinfo is None:
            raise HTTPException(422, "产物时间必须包含时区")
        paths.append(str(PurePosixPath(run.run_dir) / relative))
    if len(set(paths)) != len(paths):
        raise HTTPException(422, "产物不能重复")
    known = {a.path: a for a in db.scalars(select(Artifact).where(Artifact.run_id == run.id))}
    for item, path in zip(payload.items, paths):
        artifact = known.get(path)
        if artifact is None:
            try:
                with db.begin_nested():
                    artifact = Artifact(task_id=task.id, run_id=run.id, path=path, name=item.name,
                                        kind=artifact_kind(Path(item.name)), size=item.size, modified_at=item.modified_at)
                    db.add(artifact)
                    db.flush()
            except IntegrityError:
                artifact = db.scalar(select(Artifact).where(Artifact.task_id == task.id, Artifact.path == path))
                if artifact is None:
                    raise
        artifact.name, artifact.size = item.name, item.size
        artifact.kind = artifact_kind(Path(item.name))
        artifact.modified_at = item.modified_at
    for path, artifact in known.items():
        if path not in paths:
            db.delete(artifact)
    db.commit()
    return {"status": "ok"}


@router.post("/api/compute/nodes/{node_id}/agent/operations/poll")
def poll_operations(node_id: str, request: Request, db: Session = Depends(get_db)):
    node = authenticated_agent(node_id, request, db)
    if not agent_version_at_least(node.agent_version):
        return {"operation": None, "required_agent_version": REQUIRED_AGENT_VERSION,
                "message": f"请将远程 Agent 升级到 {REQUIRED_AGENT_VERSION} 或以上版本"}
    cutoff = utcnow() - timedelta(minutes=5)
    operation = db.scalar(select(RemoteOperation).where(
        RemoteOperation.node_id == node_id,
        (RemoteOperation.status == "pending") |
        ((RemoteOperation.status == "running") & (RemoteOperation.updated_at < cutoff)),
    ).order_by(RemoteOperation.created_at).limit(1))
    if not operation:
        return {"operation": None}
    operation_payload = operation.payload
    if operation.kind == "delete":
        task = db.get(TrainingTask, operation.task_id)
        attempt_scope = (operation.payload or {}).get("scope") == "attempt"
        target_run = db.get(TrainingRun, (operation.payload or {}).get("run_id")) if attempt_scope else None
        invalid_attempt = (
            attempt_scope and (
                not task or not target_run or target_run.task_id != operation.task_id
                or task.execution_node_id != node_id
                or target_run.execution_node_id not in {None, node_id}
                or target_run.status in ACTIVE
            )
        )
        invalid_task = (
            not attempt_scope and (
                not task or task.execution_node_id != node_id or task.status in ACTIVE
                or task.deleted_reason != "remote_delete_pending"
            )
        )
        if invalid_attempt or invalid_task:
            cancelled = db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation.id,
                RemoteOperation.status == operation.status,
                RemoteOperation.updated_at == operation.updated_at,
            ).values(status="cancelled", error="任务状态已变更，删除指令已取消", updated_at=utcnow()))
            if cancelled.rowcount != 1:
                db.rollback()
                return {"operation": None}
            db.commit()
            return {"operation": None}
        # A node can be offline for a long time after deletion was queued. Refresh
        # references immediately before dispatch so newly-created tasks are safe.
        operation_payload = {
            **operation.payload,
            "protected_paths": live_protected_paths(
                db,
                node_id,
                operation.task_id,
                deleting_run_id=(operation.payload or {}).get("run_id") if attempt_scope else None,
            ),
        }
    lease = secrets.token_hex(16)
    claimed = db.execute(update(RemoteOperation).where(
        RemoteOperation.id == operation.id, RemoteOperation.status == operation.status,
        RemoteOperation.updated_at == operation.updated_at,
    ).values(status="running", lease=lease, payload=operation_payload, updated_at=utcnow()))
    if claimed.rowcount != 1:
        db.rollback()
        return {"operation": None}
    db.commit()
    db.refresh(operation)
    return {"operation": {"id": operation.id, "kind": operation.kind, "lease": lease, "payload": operation.payload}}


def authorized_operation(db: Session, node_id: str, operation_id: str, request: Request, lease: str) -> RemoteOperation:
    authenticated_agent(node_id, request, db)
    operation = db.get(RemoteOperation, operation_id)
    if not operation or operation.node_id != node_id:
        raise HTTPException(404, "远程操作不存在")
    if not operation.lease or not secrets.compare_digest(operation.lease, lease):
        raise HTTPException(409, "远程操作已重新分配")
    return operation


class OperationResult(BaseModel):
    lease: str = Field(pattern=r"^[0-9a-f]{32}$")
    success: bool
    error: str | None = Field(default=None, max_length=2000)


@router.post("/api/compute/nodes/{node_id}/agent/operations/{operation_id}/result")
def operation_result(node_id: str, operation_id: str, payload: OperationResult, request: Request, db: Session = Depends(get_db)):
    operation = authorized_operation(db, node_id, operation_id, request, payload.lease)
    if operation.status in {"succeeded", "failed", "cancelled"}:
        return {"status": operation.status}
    if operation.kind == "fetch" and payload.success:
        raise HTTPException(409, "产物传输必须通过分块接口完成")
    operation.status = "succeeded" if payload.success else "failed"
    operation.error = payload.error if not payload.success else None
    operation.updated_at = utcnow()
    if not payload.success and operation.kind == "fetch":
        operation_file(operation, False).unlink(missing_ok=True)
    task = db.get(TrainingTask, operation.task_id)
    attempt_scope = operation.kind == "delete" and (operation.payload or {}).get("scope") == "attempt"
    target_run = db.get(TrainingRun, (operation.payload or {}).get("run_id")) if attempt_scope else None
    if attempt_scope:
        if (
            not task or not target_run or target_run.task_id != operation.task_id
            or task.execution_node_id != node_id
            or target_run.execution_node_id not in {None, node_id}
            or target_run.status in ACTIVE
        ):
            raise HTTPException(409, "该 Attempt 当前不能永久删除")
        try:
            validate_attempt_deletion(db, task, target_run, ignore_operation_id=operation.id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
    if operation.kind == "delete" and task:
        if attempt_scope:
            if payload.success:
                try:
                    trace_root = remote_trace_root(target_run.id)
                    if trace_root.exists() or trace_root.is_symlink():
                        shutil.rmtree(trace_root)
                    related_operations = list(db.scalars(select(RemoteOperation).where(
                        RemoteOperation.task_id == task.id,
                    )))
                    for other in related_operations:
                        # Fetch operations keep the run id in their payload but
                        # are not deletion payloads. Remove their center cache
                        # as well so a deleted Attempt cannot leave a stale
                        # weight available for later downloads.
                        same_run = (other.payload or {}).get("run_id") == target_run.id
                        if other.id == operation.id or is_attempt_payload(other.payload, target_run.id) or same_run:
                            for path in (operation_file(other), operation_file(other, False)):
                                path.unlink(missing_ok=True)
                    delete_attempt_records(db, task.id, target_run.id)
                    add_event(db, task.id, "run_deleted", f"远程 Attempt {operation.payload.get('attempt_number', '')} 已删除")
                except Exception as exc:
                    detail = f"{type(exc).__name__}: {exc}"[-1600:]
                    operation.status = "failed"
                    operation.error = f"中心关联数据清理失败：{detail}"
                    add_event(db, task.id, "run_delete_failed", operation.error, "error")
            else:
                operation.status = "failed"
                operation.error = "远程文件清理失败：" + (payload.error or "未知错误")
                add_event(db, task.id, "run_delete_failed", operation.error, "error")
            db.commit()
            return {"status": operation.status}
        if task.status in ACTIVE or task.deleted_reason != "remote_delete_pending":
            raise HTTPException(409, "任务当前不能永久删除")
        if payload.success:
            related_operations = list(db.scalars(select(RemoteOperation).where(
                RemoteOperation.task_id == task.id,
            )))
            try:
                # The Agent ACK covers remote files only. Center-owned history,
                # traces and fetched checkpoints must be gone before the task row.
                delete_task_history(task.id)
                for historical in task.runs:
                    trace_root = remote_trace_root(historical.id)
                    if trace_root.exists() or trace_root.is_symlink():
                        shutil.rmtree(trace_root)
                for other in related_operations:
                    for path in (operation_file(other), operation_file(other, False)):
                        path.unlink(missing_ok=True)
            except Exception as exc:
                detail = f"{type(exc).__name__}: {exc}"[-1600:]
                operation.status = "failed"
                operation.error = f"中心关联数据清理失败：{detail}"
                task.deleted_reason = "remote_delete_failed"
                task.error_message = operation.error + "；远程文件已确认清理，可重试永久删除"
                add_event(db, task.id, "remote_delete_failed", task.error_message, "error")
            else:
                for other in related_operations:
                    if other.id != operation.id:
                        other.status = "cancelled"
                db.delete(task)
        else:
            task.deleted_reason = "remote_delete_failed"
            task.error_message = "远程文件清理失败：" + (payload.error or "未知错误")
            add_event(db, task.id, "remote_delete_failed", task.error_message, "error")
    db.commit()
    return {"status": operation.status}


def matching_fetch(db: Session, artifact: Artifact) -> RemoteOperation | None:
    for operation in db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == artifact.task_id, RemoteOperation.kind == "fetch",
    ).order_by(RemoteOperation.created_at.desc())):
        p = operation.payload
        if (p.get("artifact_id") == artifact.id and p.get("run_id") == artifact.run_id and
                p.get("name") == artifact.name and p.get("size") == artifact.size and
                p.get("modified_at") == as_utc(artifact.modified_at).isoformat()):
            return operation
    return None


@router.post("/api/artifacts/{artifact_id}/prepare")
def prepare_artifact(artifact_id: int, retry: bool = False, db: Session = Depends(get_db)):
    artifact = db.get(Artifact, artifact_id)
    task = db.get(TrainingTask, artifact.task_id) if artifact else None
    if not artifact or not task or task.deleted_at:
        raise HTTPException(404, "产物不存在")
    run = db.get(TrainingRun, artifact.run_id)
    if not run or not run.execution_node_id:
        return {"status": "ready"}
    if attempt_delete_operation(db, run, active_only=True):
        raise HTTPException(409, "该 Attempt 已进入清理队列，不能准备远程权重")
    operation = matching_fetch(db, artifact)
    if operation and operation.status == "succeeded":
        digest, cache_error = cached_fetch_digest(operation)
        if cache_error is None and digest is not None:
            if (operation.payload or {}).get("sha256") is None:
                operation.payload = {**(operation.payload or {}), "sha256": digest}
                db.commit()
            return {"status": "ready"}
    if operation and operation.status == "failed" and not retry:
        return {"status": "failed", "message": operation.error or "远程文件传输失败"}
    if not operation or operation.status in {"failed", "cancelled", "succeeded"}:
        p = {**run_payload(run, task), "artifact_id": artifact.id, "name": artifact.name,
             "size": artifact.size, "modified_at": as_utc(artifact.modified_at).isoformat()}
        operation = enqueue(db, run.execution_node_id, task.id, "fetch", p)
        db.commit()
    return {"status": operation.status, "operation_id": operation.id, "message": "等待远程 Agent 传输文件"}


class ArtifactChunk(BaseModel):
    lease: str = Field(pattern=r"^[0-9a-f]{32}$")
    offset: int = Field(ge=0)
    data: str = Field(max_length=720000)
    final: bool = False
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


@router.post("/api/compute/nodes/{node_id}/agent/operations/{operation_id}/chunk")
def upload_chunk(node_id: str, operation_id: str, payload: ArtifactChunk, request: Request, db: Session = Depends(get_db)):
    operation = authorized_operation(db, node_id, operation_id, request, payload.lease)
    if operation.kind != "fetch" or operation.status not in {"running", "succeeded"}:
        raise HTTPException(409, "远程操作当前不能传输文件")
    if operation.status == "succeeded":
        return {"status": "succeeded", "offset": operation.payload["size"]}
    task = db.get(TrainingTask, operation.task_id)
    if not task or task.deleted_at:
        raise HTTPException(409, "任务已删除，停止传输文件")
    try:
        data = base64.b64decode(payload.data, validate=True)
    except ValueError as exc:
        raise HTTPException(422, "文件分块编码错误") from exc
    if payload.offset + len(data) > operation.payload["size"]:
        raise HTTPException(413, "传输文件超出登记大小")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = operation_file(operation, False)
    current = path.stat().st_size if path.exists() else 0
    # Restarting a leased operation starts at zero; retries of other chunks are idempotent.
    if payload.offset == 0:
        with path.open("wb") as handle:
            handle.write(data)
    elif current == payload.offset:
        with path.open("ab") as handle:
            handle.write(data)
    elif current == payload.offset + len(data):
        with path.open("rb") as handle:
            handle.seek(payload.offset)
            if handle.read(len(data)) != data:
                raise HTTPException(409, "重试分块内容不一致")
    else:
        raise HTTPException(409, "文件分块偏移不一致，请重新传输")
    operation.updated_at = utcnow()
    if payload.final:
        if path.stat().st_size != operation.payload["size"] or not payload.sha256:
            raise HTTPException(409, "传输文件不完整")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(512 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != payload.sha256:
            raise HTTPException(409, "文件校验失败，请重新传输")
        os.replace(path, operation_file(operation))
        # Keep the Agent-provided content digest with the durable fetch record.
        # Model-test history uses it to request the exact checkpoint bytes even
        # when a later manifest updates the same Artifact row in place.
        operation.payload = {**operation.payload, "sha256": payload.sha256}
        operation.status = "succeeded"
    db.commit()
    return {"status": operation.status, "offset": payload.offset + len(data)}
