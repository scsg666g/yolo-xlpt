"""Read-only multi-host compute monitoring. Agent credentials are node-scoped."""
from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
import uuid
import math
from pathlib import Path
from datetime import datetime, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .config import WEB_ROOT
from .database import get_db
from .models import ComputeNode, MetricPoint, TaskEvent, TrainingRun, TrainingTask, utcnow
from .services import add_event, compute_inventory, mirror_run_to_task, remote_log_path

router = APIRouter(prefix="/api/compute", tags=["compute nodes"])
MAX_REPORT_BYTES = 1024 * 1024
LOCAL_ID = "local"
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Percentage = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]


class NodeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=160)
    address: str = Field(default="", max_length=255)
    notes: str = Field(default="", max_length=1000)
    enabled: bool = True
    report_interval_seconds: int = Field(default=5, ge=5, le=300)

    @field_validator("name", "address", "notes")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if any(ord(c) < 32 and c not in "\n\t" for c in value):
            raise ValueError("不能包含控制字符")
        return value

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, value: str) -> str:
        if not value or "\n" in value or "\t" in value:
            raise ValueError("请输入有效的服务器名称")
        return value


class ProcessReport(BaseModel):
    pid: int = Field(ge=1)
    name: str = Field(max_length=255)
    memory_used: Nonnegative | None = None


class GpuReport(BaseModel):
    index: int = Field(ge=0, le=1023)
    uuid: str = Field(default="", max_length=160)
    name: str = Field(max_length=255)
    driver_version: str = Field(default="", max_length=100)
    memory_total: Nonnegative | None = None
    memory_used: Nonnegative | None = None
    memory_free: Nonnegative | None = None
    utilization: Percentage | None = None
    memory_utilization: Percentage | None = None
    temperature: float | None = Field(default=None, ge=-100, le=250, allow_inf_nan=False)
    power_draw: Nonnegative | None = None
    power_limit: Nonnegative | None = None
    graphics_clock: Nonnegative | None = None
    memory_clock: Nonnegative | None = None
    processes: list[ProcessReport] = Field(default_factory=list, max_length=512)


class HostReport(BaseModel):
    hostname: str = Field(min_length=1, max_length=255)
    platform: str = Field(max_length=1000)
    cpu_count: int | None = Field(default=None, ge=1, le=65536)
    load_average: list[Nonnegative] = Field(default_factory=list, max_length=3)
    memory_total: Nonnegative | None = None
    memory_used: Nonnegative | None = None


class ComputeReport(BaseModel):
    available: bool
    provider: str | None = Field(default=None, max_length=100)
    message: str | None = Field(default=None, max_length=1000)
    sampled_at: datetime
    host: HostReport
    gpus: list[GpuReport] = Field(default_factory=list, max_length=128)

    @field_validator("sampled_at")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("采样时间必须包含时区")
        if (value - utcnow()).total_seconds() > 300:
            raise ValueError("服务器时间超前超过 5 分钟，请校准时间")
        return value

    @field_validator("gpus")
    @classmethod
    def unique_indices(cls, value: list[GpuReport]) -> list[GpuReport]:
        if len({gpu.index for gpu in value}) != len(value):
            raise ValueError("GPU 编号不能重复")
        return value


class AgentReport(BaseModel):
    agent_version: str = Field(default="1.0.0", max_length=40)
    compute: ComputeReport


class RemoteRunUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    pid: int | None = Field(default=None, ge=1)
    completed_epochs: int = Field(default=0, ge=0)
    progress: float = Field(default=0, ge=0, le=100, allow_inf_nan=False)
    latest_metrics: dict[str, Any] = Field(default_factory=dict)
    metrics: list[dict[str, Any]] = Field(default_factory=list, max_length=256)
    log_tail: list[str] = Field(default_factory=list, max_length=200)
    exit_code: int | None = None
    error_message: str | None = Field(default=None, max_length=4000)
    stop_reason: str | None = Field(default=None, pattern=r"^(platform_stop|agent_shutdown)$")
    run_dir: str | None = Field(default=None, max_length=2000)
    command: list[str] = Field(default_factory=list, max_length=256)

    @field_validator("status")
    @classmethod
    def allowed_status(cls, value: str) -> str:
        if value not in {"starting", "running", "stopping", "stopped", "completed", "failed"}:
            raise ValueError("不支持的远程运行状态")
        return value

    @field_validator("log_tail")
    @classmethod
    def clean_logs(cls, value: list[str]) -> list[str]:
        return [line[:4000] for line in value]


_local_lock = threading.Lock()
_local_cache: dict = {}
_local_sampled = 0.0


def local_snapshot() -> dict:
    global _local_cache, _local_sampled
    with _local_lock:
        if not _local_cache or time.monotonic() - _local_sampled >= 5:
            _local_cache = compute_inventory()
            _local_sampled = time.monotonic()
        return _local_cache


def ensure_local_node(db: Session) -> ComputeNode:
    node = db.get(ComputeNode, LOCAL_ID)
    if node is None:
        node = ComputeNode(id=LOCAL_ID, name="本机服务器", kind="local")
        db.add(node)
        db.commit()
        db.refresh(node)
    return node


def node_or_404(db: Session, node_id: str) -> ComputeNode:
    node = db.get(ComputeNode, node_id)
    if node is None:
        raise HTTPException(404, "服务器不存在")
    return node


def as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def serialize_node(node: ComputeNode, *, detail: bool = False) -> dict:
    snapshot = local_snapshot() if node.kind == "local" else node.snapshot or {}
    last_seen = datetime.fromisoformat(snapshot["sampled_at"]) if node.kind == "local" else node.last_seen_at
    timeout = max(30, node.report_interval_seconds * 3)
    if node.kind == "local":
        status = "online"
    elif not node.enabled:
        status = "disabled"
    elif last_seen is None:
        status = "pending"
    else:
        sampled = datetime.fromisoformat(snapshot["sampled_at"])
        age = max((utcnow() - as_utc(last_seen)).total_seconds(), (utcnow() - as_utc(sampled)).total_seconds())
        status = "online" if age <= timeout else "offline"
    gpus = snapshot.get("gpus", [])
    utils = [gpu["utilization"] for gpu in gpus if gpu.get("utilization") is not None]
    result = {
        "id": node.id, "name": node.name, "address": node.address, "notes": node.notes,
        "kind": node.kind, "enabled": node.enabled, "status": status,
        "report_interval_seconds": node.report_interval_seconds, "offline_after_seconds": timeout,
        "last_seen_at": as_utc(last_seen).isoformat() if last_seen else None,
        "sampled_at": snapshot.get("sampled_at"), "agent_version": node.agent_version,
        "hostname": snapshot.get("host", {}).get("hostname"),
        "summary": {
            "gpu_count": len(gpus), "utilization": round(sum(utils) / len(utils)) if utils else None,
            "memory_used": sum(gpu.get("memory_used") or 0 for gpu in gpus),
            "memory_total": sum(gpu.get("memory_total") or 0 for gpu in gpus),
            "process_count": sum(len(gpu.get("processes", [])) for gpu in gpus),
        } if snapshot else None,
    }
    if detail:
        result["compute"] = snapshot or None
    return result


def issue_token(node: ComputeNode) -> str:
    token = "ylp_" + secrets.token_urlsafe(32)
    node.token_hash = hashlib.sha256(token.encode()).hexdigest()
    return token


def authenticated_agent(node_id: str, request: Request, db: Session) -> ComputeNode:
    authorization = request.headers.get("authorization", "")
    scheme, _, token = authorization.partition(" ")
    node = db.get(ComputeNode, node_id)
    digest = hashlib.sha256(token.encode()).hexdigest()
    if scheme.lower() != "bearer" or not token or not node or node.kind != "agent" or not node.token_hash or not hmac.compare_digest(digest, node.token_hash):
        raise HTTPException(401, "Agent 身份验证失败", headers={"WWW-Authenticate": "Bearer"})
    if not node.enabled:
        raise HTTPException(403, "服务器监控已禁用")
    return node


@router.get("/nodes")
def list_nodes(db: Session = Depends(get_db)):
    nodes = db.scalars(select(ComputeNode).order_by(ComputeNode.kind.desc(), ComputeNode.created_at, ComputeNode.id)).all()
    return [serialize_node(node) for node in nodes]


@router.post("/nodes", status_code=201)
def create_node(payload: NodeSettings, db: Session = Depends(get_db)):
    node = ComputeNode(id=uuid.uuid4().hex, kind="agent", **payload.model_dump())
    token = issue_token(node)
    db.add(node)
    db.commit()
    db.refresh(node)
    return {"node": serialize_node(node), "token": token}


@router.get("/agent/download")
def download_agent():
    return FileResponse(WEB_ROOT / "agent" / "monitor_agent.py", filename="monitor_agent.py", media_type="text/x-python")


@router.get("/nodes/{node_id}")
def node_detail(node_id: str, db: Session = Depends(get_db)):
    return serialize_node(node_or_404(db, node_id), detail=True)


@router.patch("/nodes/{node_id}")
def update_node(node_id: str, payload: NodeSettings, db: Session = Depends(get_db)):
    node = node_or_404(db, node_id)
    if node.kind == "local" and not payload.enabled:
        raise HTTPException(422, "本机监控不能禁用")
    if not payload.enabled and db.scalar(select(TrainingRun.id).where(
        TrainingRun.execution_node_id == node_id,
        TrainingRun.status.in_(["starting", "running", "stopping"]),
    ).limit(1)):
        raise HTTPException(409, "该服务器正在执行训练，请先停止并等待状态上报后再禁用")
    for key, value in payload.model_dump().items():
        setattr(node, key, value)
    db.commit()
    db.refresh(node)
    return serialize_node(node)


@router.post("/nodes/{node_id}/token")
def rotate_token(node_id: str, db: Session = Depends(get_db)):
    node = node_or_404(db, node_id)
    if node.kind != "agent":
        raise HTTPException(409, "本机无需 Agent Token")
    if db.scalar(select(TrainingRun.id).where(
        TrainingRun.execution_node_id == node_id,
        TrainingRun.status.in_(["starting", "running", "stopping"]),
    ).limit(1)):
        raise HTTPException(409, "该服务器正在执行训练，请等待完成后再更换 Token")
    token = issue_token(node)
    # Old observations do not establish the connection of the replacement Agent.
    node.last_seen_at = None
    db.commit()
    db.refresh(node)
    return {"node": serialize_node(node), "token": token}


@router.delete("/nodes/{node_id}", status_code=204)
def delete_node(node_id: str, db: Session = Depends(get_db)):
    node = node_or_404(db, node_id)
    if node.kind == "local":
        raise HTTPException(409, "不能移除本机服务器")
    if db.scalar(select(TrainingTask.id).where(TrainingTask.execution_node_id == node_id).limit(1)):
        raise HTTPException(409, "该服务器仍有关联任务，请先处理任务后再移除服务器")
    db.delete(node)
    db.commit()


@router.post("/nodes/{node_id}/report")
async def report(node_id: str, request: Request, db: Session = Depends(get_db)):
    node = authenticated_agent(node_id, request, db)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_REPORT_BYTES:
            raise HTTPException(413, "采样数据不能超过 1 MB")
    try:
        payload = AgentReport.model_validate_json(body)
    except ValidationError as exc:
        raise HTTPException(422, "采样数据格式错误：" + "; ".join(error["msg"] for error in exc.errors(include_input=False))) from exc
    previous = (node.snapshot or {}).get("sampled_at")
    if previous and payload.compute.sampled_at < datetime.fromisoformat(previous):
        raise HTTPException(409, "采样数据早于上次采样，已忽略")
    node.snapshot = payload.compute.model_dump(mode="json")
    node.agent_version = payload.agent_version
    node.last_seen_at = utcnow()
    db.commit()
    return {"status": "ok", "report_interval_seconds": node.report_interval_seconds}


@router.post("/nodes/{node_id}/agent/poll")
def poll_training(node_id: str, request: Request, db: Session = Depends(get_db)):
    node = authenticated_agent(node_id, request, db)
    # Older Agents require platform.log inside save_dir, which DDP deletes.
    try:
        version = tuple(int(part) for part in (node.agent_version or "0").split("."))
    except ValueError:
        version = (0,)
    if version < (1, 2, 0):
        return {"task": None, "poll_interval_seconds": 5, "required_agent_version": "1.2.0",
                "message": "请升级 Agent 至 1.2.0 或以上版本后领取训练任务"}
    if db.scalar(select(TrainingRun.id).where(TrainingRun.execution_node_id == node_id,
                                            TrainingRun.status.in_(["starting", "running", "stopping"]))):
        return {"task": None, "poll_interval_seconds": 5}
    # A task is delivered once, by atomically switching queued -> starting.
    run = db.scalar(select(TrainingRun).where(
        TrainingRun.execution_node_id == node_id, TrainingRun.status == "queued"
    ).order_by(TrainingRun.queued_at, TrainingRun.id).limit(1))
    if not run:
        return {"task": None, "poll_interval_seconds": 5}
    if run.launch_mode == "resume" and version < (1, 3, 0):
        return {"task": None, "poll_interval_seconds": 5, "required_agent_version": "1.3.0",
                "message": "断点续训需要 Agent 1.3.0 或以上版本"}
    task = db.get(TrainingTask, run.task_id)
    if not task or task.deleted_at or task.current_run_id != run.id:
        run.status = "failed"
        run.error_message = "任务已删除或不再是当前运行，未执行"
        run.finished_at = utcnow()
        db.commit()
        return {"task": None, "poll_interval_seconds": 5}
    snapshot = run.parameter_snapshot or {}
    run_dir = str(Path(snapshot.get("remote_output_root") or "runs") / task.task_type / task.id / f"attempt-{run.attempt_number:03d}")
    claimed_at = utcnow()
    # Check node occupancy again in the atomic claim, since another poll may
    # start a different queued run between the earlier SELECTs.
    occupied = select(TrainingRun.id).where(
        TrainingRun.execution_node_id == node_id,
        TrainingRun.status.in_(["starting", "running", "stopping"]),
    ).exists()
    claimed = db.execute(update(TrainingRun).where(
        TrainingRun.id == run.id, TrainingRun.status == "queued", ~occupied,
    ).values(status="starting", started_at=claimed_at, process_started_at=claimed_at,
             run_dir=run_dir, log_path=str(remote_log_path(snapshot["remote_output_root"], task.id, run.attempt_number))))
    if claimed.rowcount != 1:
        db.rollback()
        return {"task": None, "poll_interval_seconds": 5}
    db.refresh(run)
    mirror_run_to_task(task, run)
    add_event(db, task.id, "remote_claimed", f"远程服务器已领取 Attempt {run.attempt_number:03d}", run_id=run.id)
    db.commit()
    return {"poll_interval_seconds": 5, "task": {
        "task_id": task.id, "run_id": run.id, "attempt_number": run.attempt_number,
        "task_type": task.task_type, "launch_mode": run.launch_mode,
        "model_path": run.model_path, "data_path": run.data_path,
        "checkpoint_path": run.checkpoint_path, "run_dir": run.run_dir,
        "completed_epochs": run.completed_epochs, "progress": run.progress, "latest_metrics": run.latest_metrics,
        "log_path": run.log_path, "parameters": snapshot,
    }}


@router.post("/nodes/{node_id}/agent/runs/{run_id}/update")
def update_remote_run(node_id: str, run_id: str, payload: RemoteRunUpdate, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id:
        raise HTTPException(404, "远程训练运行不存在")
    task = db.get(TrainingTask, run.task_id)
    if not task:
        raise HTTPException(404, "训练任务不存在")
    prior_status = run.status
    terminal = {"stopped", "completed", "failed"}
    if prior_status in terminal:
        return {"status": prior_status, "stop_requested": False}
    if payload.status not in {"starting", "running", "stopping", *terminal}:
        raise HTTPException(422, "非法状态")
    effective_status = "stopping" if run.stop_requested and payload.status in {"starting", "running"} else payload.status
    run.status = effective_status
    run.pid = payload.pid if payload.status in {"starting", "running", "stopping"} else None
    run.completed_epochs = payload.completed_epochs
    run.progress = payload.progress
    run.latest_metrics = payload.latest_metrics
    run.remote_log_tail = payload.log_tail[-200:]
    if payload.run_dir and payload.run_dir != run.run_dir:
        raise HTTPException(422, "远程运行目录不得更改")
    if payload.command:
        run.command = payload.command
    if payload.error_message:
        run.error_message = payload.error_message
    if payload.exit_code is not None:
        run.exit_code = payload.exit_code
    for point in payload.metrics:
        try:
            epoch = int(point["epoch"])
            values = point["values"]
            elapsed = float(point.get("elapsed", 0))
            if epoch < 0 or epoch > payload.completed_epochs or not isinstance(values, dict) or not math.isfinite(elapsed) or elapsed < 0:
                continue
            exists = db.scalar(select(MetricPoint.id).where(MetricPoint.run_id == run.id, MetricPoint.epoch == epoch))
            if not exists:
                db.add(MetricPoint(task_id=task.id, run_id=run.id, epoch=epoch, elapsed=elapsed, values=values))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    if effective_status in terminal:
        run.finished_at = utcnow()
        if effective_status == "completed":
            run.progress = 100
            run.exit_code = payload.exit_code if payload.exit_code is not None else 0
        if effective_status == "stopped" and not run.error_message:
            if payload.stop_reason == "agent_shutdown":
                run.error_message = "Agent 退出，训练进程已停止"
            elif run.stop_requested or payload.stop_reason == "platform_stop":
                run.error_message = "收到平台停止指令，训练进程已停止"
            else:
                run.error_message = "远程训练已停止，Agent 未提供停止原因"
        add_event(db, task.id, effective_status, run.error_message or f"远程训练{effective_status}",
                  "error" if effective_status == "failed" else "info",
                  payload={"exit_code": run.exit_code, "stop_reason": payload.stop_reason,
                           "stop_requested": bool(run.stop_requested)}, run_id=run.id)
    elif effective_status != prior_status:
        add_event(db, task.id, effective_status, f"远程训练状态：{effective_status}", run_id=run.id)
    mirror_run_to_task(task, run)
    db.commit()
    return {"status": run.status, "stop_requested": bool(run.stop_requested)}


@router.get("/nodes/{node_id}/agent/runs/{run_id}/control")
def remote_run_control(node_id: str, run_id: str, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id:
        raise HTTPException(404, "远程训练运行不存在")
    return {"stop": bool(run.stop_requested), "status": run.status}
