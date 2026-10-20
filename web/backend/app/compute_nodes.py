"""Read-only multi-host compute monitoring. Agent credentials are node-scoped."""
from __future__ import annotations

import hashlib
import hmac
import base64
import json
import logging
import os
import platform
import posixpath
import secrets
import socket
import threading
import time
import uuid
import math
import re
from pathlib import Path, PurePosixPath
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from .config import DATA_ROOT, WEB_ROOT, platform_instance_id
from .database import get_db
from .gpu_reservations import (
    acquire_node_claim_lock,
    node_serial_assignment,
    release_gpu_reservations,
    reserve_gpu_resources,
    resolve_gpu_assignments,
)
from .models import ComputeNode, MetricPoint, TaskEvent, TrainingRun, TrainingTask, utcnow
from .services import add_event, compute_inventory, mirror_run_to_task, remote_log_path
from .process_attribution import inspect_process_owner

router = APIRouter(prefix="/api/compute", tags=["compute nodes"])
MAX_REPORT_BYTES = 1024 * 1024
MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_TRACE_SAMPLES = 24
MAX_TRACE_IMAGE_BYTES = 512 * 1024
LOCAL_ID = "local"
REQUIRED_AGENT_VERSION = "1.8.0"
REQUIRED_AGENT_VERSION_PARTS = (1, 8, 0)
PROCESS_ATTRIBUTION_AGENT_VERSION = "1.9.0"
PROCESS_ATTRIBUTION_AGENT_VERSION_PARTS = (1, 9, 0)
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
    owner_hint: dict[str, Any] | None = None


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


class TrainingCapability(BaseModel):
    enabled: bool
    ready: bool
    reason: str | None = Field(default=None, max_length=400)
    supported_parameters: list[str] = Field(default_factory=list, max_length=256)
    yolo_version: str | None = Field(default=None, max_length=100)
    python_version: str | None = Field(default=None, max_length=100)
    output_writable: bool = False
    lock_occupied: bool = False
    gpu_parallelism: bool = False
    output_roots: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(default_factory=list, max_length=128)

    @field_validator("output_roots")
    @classmethod
    def absolute_output_roots(cls, value: list[str]) -> list[str]:
        if any(not PurePosixPath(root).is_absolute() or "\x00" in root for root in value):
            raise ValueError("训练输出目录必须是远程服务器上的有效绝对路径")
        return list(dict.fromkeys(posixpath.normpath(root) for root in value))


class AgentReport(BaseModel):
    agent_version: str = Field(default="1.0.0", max_length=40)
    compute: ComputeReport
    training: TrainingCapability | None = None


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
        if value not in {"starting", "running", "stopping", "stopped", "completed", "failed", "interrupted"}:
            raise ValueError("不支持的远程运行状态")
        return value

    @field_validator("log_tail")
    @classmethod
    def clean_logs(cls, value: list[str]) -> list[str]:
        return [line[:4000] for line in value]


def remote_trace_root(run_id: str) -> Path:
    return DATA_ROOT / "remote-traces" / run_id


def _trace_sample_key(sample: dict) -> str:
    return f"{sample.get('epoch', '')}:{sample.get('batch', '')}"


class RemoteTraceUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trace: dict[str, Any] = Field(default_factory=dict)
    image_name: str | None = Field(default=None, max_length=80, pattern=r"^batch-[0-9]{5}\.jpg$")
    image_base64: str | None = Field(default=None, max_length=MAX_TRACE_IMAGE_BYTES * 2)


_local_lock = threading.Lock()
_logger = logging.getLogger("yolo-platform.compute")
_local_cache: dict = {
    "available": False,
    "provider": None,
    "message": "本机算力信息正在采样",
    "sampled_at": utcnow().isoformat(),
    "host": {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "load_average": [],
        "memory_total": None,
        "memory_used": None,
    },
    "gpus": [],
}
_local_sampled = 0.0
_local_refreshing = False


def _refresh_local_snapshot() -> None:
    global _local_cache, _local_sampled, _local_refreshing
    try:
        snapshot = compute_inventory()
    except Exception:
        _logger.exception("local_compute_sampling_failed")
        snapshot = {**_local_cache, "sampled_at": utcnow().isoformat(),
                    "message": "本机算力信息采集失败，请稍后刷新"}
    with _local_lock:
        _local_cache = snapshot
        _local_sampled = time.monotonic()
        _local_refreshing = False


def local_snapshot() -> dict:
    global _local_refreshing
    with _local_lock:
        if not _local_refreshing and time.monotonic() - _local_sampled >= 5:
            # nvidia-smi may block for several seconds. Serve the last sample now
            # and refresh in the background so monitoring never holds an API worker.
            _local_refreshing = True
            threading.Thread(target=_refresh_local_snapshot, name="local-compute-sampler", daemon=True).start()
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


def agent_version_at_least(value: str | None, minimum: tuple[int, int, int] = REQUIRED_AGENT_VERSION_PARTS) -> bool:
    """Compare the numeric core of an Agent version without trusting arbitrary suffixes."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+._a-zA-Z0-9]*)?", value or "")
    return bool(match and tuple(int(part) for part in match.groups()) >= minimum)


def remote_training_unavailable_reason(node: ComputeNode | None, *, now: datetime | None = None) -> str | None:
    """Return a user-facing reason when a remote node cannot safely accept work."""
    if not node or node.kind != "agent":
        return "远程服务器不存在"
    if not node.enabled:
        return "远程服务器已禁用"
    if not agent_version_at_least(node.agent_version):
        return f"请将远程 Agent 升级到 {REQUIRED_AGENT_VERSION} 或以上版本"
    if node.last_seen_at is None:
        return "远程 Agent 尚未连接"
    current = now or utcnow()
    snapshot = node.snapshot or {}
    try:
        sampled_at = datetime.fromisoformat(snapshot["sampled_at"])
    except (KeyError, TypeError, ValueError):
        return "远程 Agent 尚未上报有效状态"
    timeout = max(30, node.report_interval_seconds * 3)
    # Connectivity is established by the center's receive time. The remote
    # sampled_at value is schema-validated but must not make a fresh request
    # look offline when the server clock lags behind the center.
    age = (current - as_utc(node.last_seen_at)).total_seconds()
    if age > timeout:
        return "远程 Agent 已离线"
    capability = snapshot.get("training")
    if not isinstance(capability, dict):
        return "远程 Agent 未上报训练能力"
    if not capability.get("enabled"):
        return "远程 Agent 未启用训练"
    if not capability.get("ready"):
        return str(capability.get("reason") or "远程训练环境未就绪")[:400]
    return None


def remote_gpu_parallelism_supported(node: ComputeNode) -> bool:
    capability = (node.snapshot or {}).get("training")
    return bool(isinstance(capability, dict) and capability.get("gpu_parallelism") is True)


def remote_output_root_unavailable_reason(node: ComputeNode | None, output_root: Any) -> str | None:
    """Check reported remote POSIX roots without consulting the center's filesystem.

    Older Agents omit output_roots; their authoritative path checks remain on
    the Agent. Symlinks and real filesystem permissions also remain remote.
    """
    if not isinstance(output_root, str) or "\x00" in output_root or not PurePosixPath(output_root).is_absolute():
        return "训练输出目录必须是远程服务器上的有效绝对路径"
    capability = (node.snapshot or {}).get("training") if node else None
    roots = capability.get("output_roots") if isinstance(capability, dict) else None
    if not roots:
        return None
    candidate = PurePosixPath(posixpath.normpath(output_root))
    if any(candidate.is_relative_to(PurePosixPath(posixpath.normpath(root))) for root in roots):
        return None
    return (
        f"训练输出目录 {output_root} 不在远程 Agent 允许的训练输出目录内：{'、'.join(roots)}。"
        "请选择其中一个目录或其子目录；如需调整范围，请修改该 Agent 的 training_output_roots，"
        "同时确保它在 allowed_training_roots 内，并重启该 Agent 后等待重新上报"
    )


def _run_query_for_node(db: Session, node: ComputeNode, run_ids: set[str] | None = None):
    node_filter = TrainingRun.execution_node_id.is_(None) if node.kind == "local" else TrainingRun.execution_node_id == node.id
    active_for_node = and_(node_filter, TrainingRun.status.in_(["starting", "running", "stopping"]))
    query_filter = or_(active_for_node, TrainingRun.id.in_(run_ids)) if run_ids else active_for_node
    return db.execute(
        select(TrainingRun, TrainingTask).join(TrainingTask, TrainingTask.id == TrainingRun.task_id)
        .where(query_filter)
    ).all()


def _association_from_hint(hint: dict[str, Any] | None, *, node: ComputeNode,
                           current_instance_id: str, runs_by_id: dict[str, tuple[TrainingRun, TrainingTask]],
                           sampled_at: str | None, supported: bool) -> dict[str, Any]:
    unknown = {"state": "unknown", "reason": "无法确认该进程的训练归属"}
    if not supported:
        unknown["reason"] = f"远程 Agent 需升级到 {PROCESS_ATTRIBUTION_AGENT_VERSION} 才能关联 GPU 进程"
        return {**unknown, "sampled_at": sampled_at}
    if not isinstance(hint, dict):
        unknown["reason"] = "Agent 未提供进程归属线索"
        return {**unknown, "sampled_at": sampled_at}

    method = hint.get("match_method")
    if method == "ambiguous":
        return {"state": "ambiguous", "reason": "检测到多个训练归属候选", "sampled_at": sampled_at}
    if method == "none":
        state = "unmanaged" if hint.get("scan_complete") is True else "unknown"
        reason = ("未发现本实例运行标识；旧任务或其他实例任务可能无法识别"
                  if state == "unmanaged" else "进程信息读取不完整")
        return {"state": state, "reason": reason, "sampled_at": sampled_at}
    if method not in {"runtime_marker", "ancestor", "launcher"}:
        return {**unknown, "sampled_at": sampled_at}

    owner_instance = hint.get("instance_id")
    owner_node = hint.get("node_id")
    run_id = hint.get("run_id")
    if owner_instance and owner_instance != current_instance_id:
        return {"state": "other_platform", "reason": "由其他 YOLO Forge 实例管理", "sampled_at": sampled_at}
    if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
        return {**unknown, "sampled_at": sampled_at}

    record = runs_by_id.get(run_id)
    if record is None:
        return {"state": "orphaned", "run_id": run_id, "reason": "进程仍带有平台标识，但当前实例没有对应 Run",
                "match_method": method, "sampled_at": sampled_at}
    run, task = record
    if hint.get("task_id") and hint.get("task_id") != run.task_id:
        return {"state": "ambiguous", "reason": "进程标识与 Run 的任务 ID 不一致", "sampled_at": sampled_at}
    expected_node = "local" if run.execution_node_id is None else run.execution_node_id
    if owner_node and owner_node != expected_node:
        return {"state": "ambiguous", "reason": "进程标识与 Run 的执行节点不一致", "sampled_at": sampled_at}
    active = run.status in {"starting", "running", "stopping"} and task.deleted_at is None
    if not active:
        return {"state": "orphaned", "run_id": run.id, "root_pid": run.pid,
                "reason": "平台 Run 已结束或任务已删除，但进程仍出现在 GPU 快照中",
                "match_method": method, "sampled_at": sampled_at}

    own_node = "local" if node.kind == "local" else node.id
    state = "managed" if expected_node == own_node else "other_node"
    return {"state": state, "task_id": task.id, "task_name": task.name, "run_id": run.id,
            "attempt_number": run.attempt_number, "root_pid": run.pid,
            "match_method": method, "sampled_at": sampled_at}


def _associate_snapshot(node: ComputeNode, snapshot: dict, db: Session, *, stale: bool = False) -> dict:
    result = {**snapshot, "gpus": []}
    process_items = [process for gpu in snapshot.get("gpus", [])
                     for process in gpu.get("processes", []) if isinstance(process, dict)]
    hinted_run_ids = {process.get("owner_hint", {}).get("run_id") for process in process_items
                      if isinstance(process.get("owner_hint"), dict)
                      and isinstance(process["owner_hint"].get("run_id"), str)}
    rows = _run_query_for_node(db, node, hinted_run_ids)
    runs_by_id = {run.id: (run, task) for run, task in rows}
    instance_id = platform_instance_id()
    validated_roots: dict[int, list[dict[str, Any]]] = {}
    if node.kind == "local":
        from .services import process_matches_training

        for run, task in rows:
            if run.status not in {"starting", "running", "stopping"} or not run.pid:
                continue
            if process_matches_training(run.pid, list(run.command or []), run.process_started_at):
                validated_roots.setdefault(run.pid, []).append({
                    "instance_id": instance_id, "node_id": "local", "run_id": run.id, "task_id": task.id,
                    "root_pid": run.pid,
                })

    supported = node.kind == "local" or agent_version_at_least(
        node.agent_version, PROCESS_ATTRIBUTION_AGENT_VERSION_PARTS,
    )
    sampled_at = snapshot.get("sampled_at")
    for gpu in snapshot.get("gpus", []):
        gpu_copy = {**gpu, "processes": []}
        for process in gpu.get("processes", []):
            process_copy = dict(process)
            if node.kind == "local":
                hint = inspect_process_owner(
                    int(process_copy.get("pid") or 0), validated_roots=validated_roots,
                )
            else:
                hint = process_copy.get("owner_hint")
            association = _association_from_hint(
                hint, node=node, current_instance_id=instance_id, runs_by_id=runs_by_id,
                sampled_at=sampled_at, supported=supported,
            )
            if stale:
                association = {**association, "last_state": association["state"], "state": "stale"}
                association["reason"] = "服务器已离线，显示最后一次采样的归属结果"
            process_copy["association"] = association
            gpu_copy["processes"].append(process_copy)
        result["gpus"].append(gpu_copy)
    return result


def serialize_node(node: ComputeNode, *, detail: bool = False, db: Session | None = None) -> dict:
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
        age = (utcnow() - as_utc(last_seen)).total_seconds()
        status = "online" if age <= timeout else "offline"
    gpus = snapshot.get("gpus", [])
    utils = [gpu["utilization"] for gpu in gpus if gpu.get("utilization") is not None]
    result = {
        "id": node.id, "name": node.name, "address": node.address, "notes": node.notes,
        "kind": node.kind, "enabled": node.enabled, "status": status,
        "report_interval_seconds": node.report_interval_seconds, "offline_after_seconds": timeout,
        "last_seen_at": as_utc(last_seen).isoformat() if last_seen else None,
        "sampled_at": snapshot.get("sampled_at"), "agent_version": node.agent_version,
        "required_agent_version": REQUIRED_AGENT_VERSION,
        "training_unavailable_reason": (
            remote_training_unavailable_reason(node) if node.kind == "agent" else None
        ),
        "hostname": snapshot.get("host", {}).get("hostname"),
        "training": snapshot.get("training") if node.kind == "agent" else None,
        "summary": {
            "gpu_count": len(gpus), "utilization": round(sum(utils) / len(utils)) if utils else None,
            "memory_used": sum(gpu.get("memory_used") or 0 for gpu in gpus),
            "memory_total": sum(gpu.get("memory_total") or 0 for gpu in gpus),
            "process_count": sum(len(gpu.get("processes", [])) for gpu in gpus),
        } if snapshot else None,
    }
    result["process_attribution_supported"] = (
        node.kind == "local" or agent_version_at_least(
            node.agent_version, PROCESS_ATTRIBUTION_AGENT_VERSION_PARTS,
        )
    )
    result["required_process_attribution_agent_version"] = PROCESS_ATTRIBUTION_AGENT_VERSION
    if detail:
        result["compute"] = (
            _associate_snapshot(node, snapshot, db, stale=node.kind == "agent" and status != "online")
            if snapshot and db else snapshot or None
        )
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
    return serialize_node(node_or_404(db, node_id), detail=True, db=db)


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
    node.snapshot = {**payload.compute.model_dump(mode="json"),
                     **({"training": payload.training.model_dump(mode="json")} if payload.training else {})}
    node.agent_version = payload.agent_version
    node.last_seen_at = utcnow()
    db.commit()
    return {"status": "ok", "report_interval_seconds": node.report_interval_seconds}


@router.post("/nodes/{node_id}/agent/poll")
def poll_training(node_id: str, request: Request, db: Session = Depends(get_db)):
    node = authenticated_agent(node_id, request, db)
    unavailable = remote_training_unavailable_reason(node)
    if unavailable:
        return {"task": None, "poll_interval_seconds": 5,
                "required_agent_version": REQUIRED_AGENT_VERSION, "message": unavailable}
    # Serialize all claims for this node. Reservation keys alone cannot
    # arbitrate a node-wide legacy/automatic claim against a UUID claim.
    acquire_node_claim_lock(db, node_id)
    # Terminal rows can be left behind when the process exited between two
    # heartbeats. Reconcile their durable GPU ownership before selecting work.
    from .gpu_reservations import release_terminal_gpu_reservations
    release_terminal_gpu_reservations(db)
    now = utcnow()
    # If a stop arrived after delivery but before confirmation, the Agent is
    # contractually forbidden to start the process. Retire an abandoned claim
    # after its lease expires so it cannot occupy the node forever.
    expired_stopping_id = db.scalar(select(TrainingRun.id).where(
        TrainingRun.execution_node_id == node_id, TrainingRun.status == "stopping",
        TrainingRun.stop_requested.is_(True), TrainingRun.claim_lease.is_not(None),
        TrainingRun.claim_confirmed.is_(False), TrainingRun.claim_expires_at < now,
    ).order_by(TrainingRun.queued_at).limit(1))
    if expired_stopping_id:
        retired = db.execute(update(TrainingRun).where(
            TrainingRun.id == expired_stopping_id, TrainingRun.status == "stopping",
            TrainingRun.stop_requested.is_(True), TrainingRun.claim_confirmed.is_(False),
            TrainingRun.claim_expires_at < now,
        ).values(status="stopped", pid=None, finished_at=now,
                 error_message="平台已取消远程训练启动，训练进程尚未启动"))
        if retired.rowcount == 1:
            db.expire_all()
            retired_run = db.get(TrainingRun, expired_stopping_id)
            retired_task = db.get(TrainingTask, retired_run.task_id) if retired_run else None
            if retired_run and retired_task and retired_task.current_run_id == retired_run.id:
                mirror_run_to_task(retired_task, retired_run)
                add_event(db, retired_task.id, "stopped", retired_run.error_message, run_id=retired_run.id)
            release_gpu_reservations(db, expired_stopping_id)
            db.commit()
            acquire_node_claim_lock(db, node_id)
    expired = db.scalar(select(TrainingRun).where(
        TrainingRun.execution_node_id == node_id, TrainingRun.status == "starting",
        TrainingRun.claim_lease.is_not(None), TrainingRun.claim_confirmed.is_(False),
        TrainingRun.claim_expires_at < now,
    ).order_by(TrainingRun.queued_at).limit(1))
    capability_parallel = remote_gpu_parallelism_supported(node)
    remote_gpus = ((node.snapshot or {}).get("gpus") or [])
    queued = db.scalars(select(TrainingRun).where(
        TrainingRun.execution_node_id == node_id,
        TrainingRun.status == "queued",
    ).order_by(TrainingRun.queued_at, TrainingRun.id)).all()
    candidates = ([expired] if expired else []) + [item for item in queued if not expired or item.id != expired.id]
    last_message = None
    for run in candidates:
        task = db.get(TrainingTask, run.task_id)
        if not task or task.deleted_at or task.current_run_id != run.id:
            run.status = "failed"
            run.error_message = "任务已删除或不再是当前运行，未执行"
            run.finished_at = utcnow()
            release_gpu_reservations(db, run.id)
            db.commit()
            acquire_node_claim_lock(db, node_id)
            continue
        snapshot = run.parameter_snapshot or {}
        output_error = remote_output_root_unavailable_reason(node, snapshot.get("remote_output_root"))
        if output_error:
            rejected = db.execute(update(TrainingRun).where(
                TrainingRun.id == run.id,
                ((TrainingRun.status == "queued") | ((TrainingRun.status == "starting") &
                  TrainingRun.claim_confirmed.is_(False) & (TrainingRun.claim_expires_at < now))),
            ).values(status="failed", pid=None, finished_at=now, error_message=output_error)
              .execution_options(synchronize_session=False))
            if rejected.rowcount == 1:
                db.refresh(run)
                mirror_run_to_task(task, run)
                add_event(db, task.id, "failed", output_error, run_id=run.id)
                release_gpu_reservations(db, run.id)
                db.commit()
                acquire_node_claim_lock(db, node_id)
            else:
                db.rollback()
                acquire_node_claim_lock(db, node_id)
            last_message = output_error
            continue

        device = snapshot.get("device", task.device)
        normalized_device = str(device or "").strip().lower()
        is_gpu_request = normalized_device not in {"", "cpu", "mps"}
        if not capability_parallel:
            assignments = [node_serial_assignment(node_id)]
            exclusive_node = True
        elif is_gpu_request:
            # A parallel-capable Agent must provide a UUID for every requested
            # CUDA index. Unknown indices and missing inventory fail closed.
            parts = [part.strip() for part in normalized_device.split(",")]
            valid_indices = bool(parts) and all(part.isdigit() for part in parts)
            indices = list(dict.fromkeys(int(part) for part in parts)) if valid_indices else []
            inventory_indices = {
                gpu.get("index") for gpu in remote_gpus
                if isinstance(gpu, dict) and type(gpu.get("index")) is int
            }
            if not indices or any(index not in inventory_indices for index in indices):
                last_message = "任务请求的 GPU 编号未出现在远程 Agent 的设备清单中，任务继续排队"
                continue
            assignments = resolve_gpu_assignments(
                device, remote_gpus, node_id=node_id, require_uuid=True,
            )
            if assignments is None:
                last_message = "远程 Agent 未上报请求 GPU 的 UUID，任务继续排队"
                continue
            exclusive_node = False
        else:
            # The current Agent treats a task without gpu_assignment as a
            # node-exclusive legacy claim; avoid claiming CPU/MPS alongside
            # another run until the Agent protocol can explicitly acknowledge
            # independent non-CUDA execution.
            assignments = [node_serial_assignment(node_id)]
            exclusive_node = True

        claimed_at = utcnow()
        lease = secrets.token_hex(16)
        run_dir = str(Path(snapshot.get("remote_output_root") or "runs") / task.task_type / task.id / f"attempt-{run.attempt_number:03d}")
        with db.begin_nested() as candidate_tx:
            claimed = db.execute(update(TrainingRun).where(
                TrainingRun.id == run.id,
                ((TrainingRun.status == "queued") | ((TrainingRun.status == "starting") &
                  TrainingRun.claim_confirmed.is_(False) & (TrainingRun.claim_expires_at < claimed_at))),
            ).values(status="starting", started_at=claimed_at, process_started_at=claimed_at,
                     claim_lease=lease, claim_expires_at=claimed_at + timedelta(seconds=90),
                     claim_confirmed=False, run_dir=run_dir,
                     log_path=str(remote_log_path(snapshot["remote_output_root"], task.id, run.attempt_number))))
            if claimed.rowcount != 1 or not reserve_gpu_resources(
                db, run_id=run.id, node_id=node_id, assignments=assignments,
                exclusive_node=exclusive_node,
            ):
                candidate_tx.rollback()
                continue
        db.refresh(run)
        mirror_run_to_task(task, run)
        add_event(db, task.id, "remote_claimed", f"远程服务器已领取 Attempt {run.attempt_number:03d}", run_id=run.id)
        db.commit()
        return {"poll_interval_seconds": 5, "task": {
            "task_id": task.id, "run_id": run.id, "attempt_number": run.attempt_number,
            "platform_instance_id": platform_instance_id(),
            "task_type": task.task_type, "launch_mode": run.launch_mode,
            "model_path": run.model_path, "data_path": run.data_path,
            "checkpoint_path": run.checkpoint_path, "run_dir": run.run_dir, "claim_lease": lease,
            "completed_epochs": run.completed_epochs, "progress": run.progress, "latest_metrics": run.latest_metrics,
            "log_path": run.log_path, "parameters": snapshot,
            "gpu_assignment": [
                {"uuid": item.get("uuid"), "index": item.get("index")}
                for item in assignments if item.get("uuid")
            ],
        }}
    db.commit()
    result = {"task": None, "poll_interval_seconds": 5}
    if last_message:
        result["message"] = last_message
    return result


class ClaimConfirmation(BaseModel):
    lease: str = Field(min_length=32, max_length=32)


@router.post("/nodes/{node_id}/agent/runs/{run_id}/confirm")
def confirm_remote_claim(node_id: str, run_id: str, payload: ClaimConfirmation,
                         request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    now = utcnow()
    confirmed = db.execute(update(TrainingRun).where(
        TrainingRun.id == run_id, TrainingRun.execution_node_id == node_id,
        TrainingRun.claim_lease == payload.lease, TrainingRun.claim_confirmed.is_(False),
        TrainingRun.status.in_(["starting", "stopping"]),
        TrainingRun.claim_expires_at.is_not(None), TrainingRun.claim_expires_at >= now,
    ).values(claim_confirmed=True))
    if confirmed.rowcount == 1:
        db.commit()
        run = db.get(TrainingRun, run_id)
        return {"confirmed": True, "stop": bool(run and run.stop_requested)}

    # A client may retry after the server committed but its response was lost.
    # The same node and lease receive the original successful outcome even if
    # the lease timestamp has since elapsed or the run has reached a new state.
    db.rollback()
    run = db.get(TrainingRun, run_id)
    if (run and run.execution_node_id == node_id and run.claim_lease == payload.lease
            and run.claim_confirmed):
        if run.status in {"completed", "failed", "stopped", "interrupted"}:
            return {"confirmed": True, "stop": True, "status": run.status}
        return {"confirmed": True, "stop": bool(run.stop_requested)}
    if (run and run.execution_node_id == node_id and run.claim_lease == payload.lease
            and run.claim_expires_at and as_utc(run.claim_expires_at) < now):
        raise HTTPException(409, "训练领取已过期，请等待下一次任务轮询")
    raise HTTPException(409, "训练领取已失效，请等待下一次任务轮询")


@router.post("/nodes/{node_id}/agent/runs/{run_id}/update")
def update_remote_run(node_id: str, run_id: str, payload: RemoteRunUpdate, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id:
        raise HTTPException(404, "远程训练运行不存在")
    if run.claim_lease and request.headers.get("X-Training-Lease") != run.claim_lease:
        raise HTTPException(409, "训练领取已失效")
    if run.claim_lease and not run.claim_confirmed:
        raise HTTPException(409, "请先确认训练领取")
    task = db.get(TrainingTask, run.task_id)
    if not task:
        raise HTTPException(404, "训练任务不存在")
    prior_status = run.status
    terminal = {"stopped", "completed", "failed", "interrupted"}
    if prior_status in terminal:
        return {"status": prior_status, "stop_requested": bool(run.stop_requested)}
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
        release_gpu_reservations(db, run.id)
    elif effective_status != prior_status:
        add_event(db, task.id, effective_status, f"远程训练状态：{effective_status}", run_id=run.id)
    mirror_run_to_task(task, run)
    db.commit()
    return {"status": run.status, "stop_requested": bool(run.stop_requested)}


@router.post("/nodes/{node_id}/agent/runs/{run_id}/trace")
def upload_remote_trace(node_id: str, run_id: str, payload: RemoteTraceUpload, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id:
        raise HTTPException(404, "远程训练运行不存在")
    samples = payload.trace.get("samples", [])
    if not isinstance(samples, list) or len(samples) > MAX_TRACE_SAMPLES:
        raise HTTPException(422, "训练过程采样数量超出限制")
    if len(json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)) > MAX_TRACE_BYTES:
        raise HTTPException(413, "训练过程数据不能超过 4 MB")
    clean_samples = []
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get("epoch"), int) or not isinstance(sample.get("batch"), int):
            raise HTTPException(422, "训练过程采样格式错误")
        clean_samples.append(sample)
    encoded = payload.image_base64
    image_name = payload.image_name
    if bool(encoded) != bool(image_name):
        raise HTTPException(422, "训练过程图片必须同时提供文件名和内容")
    trace_root = remote_trace_root(run_id)
    trace_root.mkdir(parents=True, exist_ok=True)
    previous = {}
    previous_path = trace_root / "trace.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
    if encoded and image_name:
        try:
            image = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            raise HTTPException(422, "训练过程图片编码无效") from None
        if len(image) > MAX_TRACE_IMAGE_BYTES or not image.startswith(b"\xff\xd8\xff"):
            raise HTTPException(422, "训练过程图片大小或格式不符合要求")
        temporary = trace_root / f".{image_name}.tmp"
        temporary.write_bytes(image)
        temporary.replace(trace_root / image_name)
    merged = {"version": 1, "available": bool(clean_samples), "source": "remote training callback",
              "updated_at": payload.trace.get("updated_at"), "sample_count": len(clean_samples),
              "maximum": min(int(payload.trace.get("maximum", MAX_TRACE_SAMPLES)), MAX_TRACE_SAMPLES),
              "interval": max(1, int(payload.trace.get("interval", 10))), "samples": clean_samples}
    if not isinstance(merged["updated_at"], (int, float)):
        merged["updated_at"] = time.time()
    for sample in merged["samples"]:
        if sample.get("image"):
            image_path = trace_root / Path(str(sample["image"])).name
            if not image_path.is_file():
                old = next((item for item in previous.get("samples", []) if _trace_sample_key(item) == _trace_sample_key(sample)), {})
                sample["image"] = old.get("image") if old.get("image") and (trace_root / Path(str(old["image"])).name).is_file() else None
    temporary = trace_root / ".trace.json.tmp"
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(trace_root / "trace.json")
    return {"status": "ok", "sample_count": len(clean_samples)}


@router.get("/nodes/{node_id}/agent/runs/{run_id}/control")
def remote_run_control(node_id: str, run_id: str, request: Request, db: Session = Depends(get_db)):
    authenticated_agent(node_id, request, db)
    run = db.get(TrainingRun, run_id)
    if not run or run.execution_node_id != node_id:
        raise HTTPException(404, "远程训练运行不存在")
    return {"stop": bool(run.stop_requested), "status": run.status}
