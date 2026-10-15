"""Agent-initiated file operations. Never interpret remote paths as local paths."""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import DATA_ROOT
from .compute_nodes import as_utc, authenticated_agent, remote_trace_root
from .database import get_db
from .models import Artifact, RemoteOperation, TrainingRun, TrainingTask, utcnow
from .services import add_event, artifact_kind

router = APIRouter()
CACHE_ROOT = DATA_ROOT / "remote-artifacts"
ACTIVE = {"queued", "starting", "running", "stopping"}


def operation_file(operation: RemoteOperation, complete: bool = True) -> Path:
    return CACHE_ROOT / (operation.id + (".bin" if complete else ".part"))


def run_payload(run: TrainingRun, task: TrainingTask) -> dict:
    return {"task_id": task.id, "task_type": task.task_type, "run_id": run.id,
            "attempt_number": run.attempt_number, "run_dir": run.run_dir, "log_path": run.log_path,
            "output_root": (run.parameter_snapshot or {}).get("remote_output_root")}


def enqueue(db: Session, node_id: str, task_id: str, kind: str, payload: dict) -> RemoteOperation:
    operation = RemoteOperation(id=uuid.uuid4().hex, node_id=node_id, task_id=task_id,
                                kind=kind, payload=payload, status="pending")
    db.add(operation)
    db.flush()
    return operation


def queue_scan(db: Session, task: TrainingTask, run: TrainingRun) -> None:
    if not run.execution_node_id or not run.run_dir or task.deleted_at:
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
    protected = []
    for other in db.scalars(select(TrainingTask).where(TrainingTask.execution_node_id == task.execution_node_id)):
        protected.extend([other.model_path, other.data_path])
        for historical in other.runs:
            protected.extend([historical.model_path, historical.data_path])
    enqueue(db, task.execution_node_id, task.id, "delete", {"runs": runs, "protected_paths": protected})
    task.deleted_at = task.deleted_at or utcnow()
    task.deleted_reason = "remote_delete_pending"
    add_event(db, task.id, "remote_delete_queued", "等待远程 Agent 清理该任务文件；确认后删除记录")


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
    try:
        compatible = tuple(int(p) for p in (node.agent_version or "0").split(".")) >= (1, 3, 0)
    except ValueError:
        compatible = False
    if not compatible:
        return {"operation": None, "required_agent_version": "1.3.0"}
    cutoff = utcnow() - timedelta(minutes=5)
    operation = db.scalar(select(RemoteOperation).where(
        RemoteOperation.node_id == node_id,
        (RemoteOperation.status == "pending") |
        ((RemoteOperation.status == "running") & (RemoteOperation.updated_at < cutoff)),
    ).order_by(RemoteOperation.created_at).limit(1))
    if not operation:
        return {"operation": None}
    lease = secrets.token_hex(16)
    claimed = db.execute(update(RemoteOperation).where(
        RemoteOperation.id == operation.id, RemoteOperation.status == operation.status,
        RemoteOperation.updated_at == operation.updated_at,
    ).values(status="running", lease=lease, updated_at=utcnow()))
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
    if operation.kind == "delete" and task:
        if task.status in ACTIVE or task.deleted_reason != "remote_delete_pending":
            raise HTTPException(409, "任务当前不能永久删除")
        if payload.success:
            for historical in task.runs:
                shutil.rmtree(remote_trace_root(historical.id), ignore_errors=True)
            for other in db.scalars(select(RemoteOperation).where(RemoteOperation.task_id == task.id)):
                if other.id != operation.id:
                    other.status = "cancelled"
                for path in (operation_file(other), operation_file(other, False)):
                    path.unlink(missing_ok=True)
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
    operation = matching_fetch(db, artifact)
    if operation and operation.status == "succeeded" and operation_file(operation).is_file():
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
        operation.status = "succeeded"
    db.commit()
    return {"status": operation.status, "offset": payload.offset + len(data)}
