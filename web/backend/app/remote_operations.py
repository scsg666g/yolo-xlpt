"""Agent-initiated file operations. Never interpret remote paths as local paths."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import and_, case, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from .config import (
    DATA_ROOT,
    REMOTE_MODEL_CACHE_MIN_FREE_BYTES,
    REMOTE_MODEL_CACHE_QUOTA_BYTES,
    nonnegative_int_env,
)
from .compute_nodes import (
    REQUIRED_AGENT_VERSION,
    REMOTE_MODEL_TEST_AGENT_VERSION,
    agent_version_at_least,
    as_utc,
    authenticated_agent,
    remote_model_test_unavailable_reason,
    remote_trace_root,
)
from .database import get_db
from .gpu_reservations import acquire_node_claim_lock
from .model_test_storage import delete_task_history
from .models import (
    Artifact,
    ComputeNode,
    ModelAsset,
    ModelTest,
    RemoteOperation,
    TrainingRun,
    TrainingTask,
    utcnow,
)
from .run_deletions import (
    LEGACY_FLAT_LAYOUT,
    attempt_delete_operation,
    delete_attempt_records,
    is_attempt_payload,
    remote_attempt_layout,
    remote_attempt_path_candidates,
    remote_output_root_for_run,
    validate_attempt_deletion,
    checkpoint_releases,
    checkpoint_required_absent_files,
)
from .services import add_event, artifact_kind

router = APIRouter()
CACHE_ROOT = DATA_ROOT / "remote-artifacts"
ACTIVE = {"queued", "starting", "running", "stopping"}
LEGACY_FLAT_DELETE_AGENT_VERSION = "2.0.1"
LEGACY_FLAT_DELETE_AGENT_VERSION_PARTS = (2, 0, 1)
TASK_DELETE_AGENT_VERSION = "2.0.4"
TASK_DELETE_AGENT_VERSION_PARTS = (2, 0, 4)
ABSENT_BEST_AGENT_VERSION = "2.6.2"
ABSENT_BEST_AGENT_VERSION_PARTS = (2, 6, 2)
DELETE_RECEIPT_VERSION = 1
MODEL_ASSET_OPERATION_PREFIX = "model-asset:"
MAX_REMOTE_MODEL_BYTES = 8 * 1024**3
MAX_FILE_CHUNK_BYTES = 512 * 1024
ACTIVE_MODEL_FETCH_STATUSES = {"pending", "running", "finalizing"}
_MODEL_CACHE_CAPACITY_LOCK = threading.Lock()
OPERATION_LEASE_TIMEOUT = timedelta(minutes=5)
MODEL_FINALIZE_HEARTBEAT_SECONDS = 30
CENTER_CLEANUP_LEASE_TIMEOUT = timedelta(minutes=5)
CENTER_CLEANUP_HEARTBEAT_SECONDS = 30
REMOTE_LOG_CACHE_QUOTA_BYTES = nonnegative_int_env("PLATFORM_REMOTE_LOG_CACHE_QUOTA_BYTES", 2 * 1024**3)
REMOTE_LOG_MIN_FREE_BYTES = nonnegative_int_env("PLATFORM_REMOTE_LOG_MIN_FREE_BYTES", 2 * 1024**3)


def _delete_plan_run(run: dict) -> dict:
    """Return the stable subset covered by a remote deletion receipt."""
    return {
        key: run.get(key)
        for key in (
            "task_id", "task_type", "run_id", "attempt_number", "run_dir",
            "log_path", "output_root", "layout",
        )
    }


def deletion_plan_digest(runs: list[dict]) -> str:
    document = {
        "version": DELETE_RECEIPT_VERSION,
        "scope": "task",
        "runs": [_delete_plan_run(run) for run in runs],
    }
    encoded = json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def deletion_plan_targets(runs: list[dict]) -> list[str]:
    targets: list[str] = []
    seen: set[str] = set()
    for run in runs:
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id or len(run_id) > 160 or run_id in seen:
            raise ValueError("远程删除计划包含无效或重复的 Attempt 标识")
        seen.add(run_id)
        targets.append(f"{run_id}:output")
        if run.get("log_path"):
            targets.append(f"{run_id}:log")
    return targets


class RemoteModelCacheError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(detail)


def model_asset_operation_scope(asset_id: str) -> str:
    return MODEL_ASSET_OPERATION_PREFIX + asset_id


def operation_model_asset_id(operation: RemoteOperation) -> str | None:
    value = (operation.payload or {}).get("model_asset_id")
    return value if isinstance(value, str) and operation.task_id == model_asset_operation_scope(value) else None


def operation_file(operation: RemoteOperation, complete: bool = True) -> Path:
    return CACHE_ROOT / (operation.id + (".bin" if complete else ".part"))


def validate_log_fetch(db: Session, operation: RemoteOperation) -> None:
    task = db.get(TrainingTask, operation.task_id)
    run = db.get(TrainingRun, (operation.payload or {}).get("run_id"))
    if (not task or task.deleted_at or not run or run.task_id != task.id
            or (run.execution_node_id or task.execution_node_id) != operation.node_id
            or run.log_path != operation.payload.get("log_path")
            or attempt_delete_operation(db, run, active_only=True)):
        raise HTTPException(409, "训练日志来源已变更或正在清理，停止传输")


def reserve_log_cache(db: Session) -> None:
    """Reserve complete snapshot sizes before accepting their first chunks."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    used = outstanding = 0
    for operation in db.scalars(select(RemoteOperation).where(RemoteOperation.kind == "fetch_log")):
        stored = sum(_regular_file_size(path) for path in (operation_file(operation), operation_file(operation, False)))
        reserved = max(stored, int((operation.payload or {}).get("size") or 0)) if operation.status in {"pending", "running", "finalizing"} else stored
        used += reserved
        outstanding += max(0, reserved - stored)
    if REMOTE_LOG_CACHE_QUOTA_BYTES and used > REMOTE_LOG_CACHE_QUOTA_BYTES:
        raise HTTPException(507, "远程日志缓存已达到容量限制，请清理不再需要的训练记录或调整日志缓存配额")
    if REMOTE_LOG_MIN_FREE_BYTES and shutil.disk_usage(CACHE_ROOT).free - outstanding < REMOTE_LOG_MIN_FREE_BYTES:
        raise HTTPException(507, "平台磁盘余量不足，不能接收完整日志")


def _cache_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns


def _same_regular_cache_file(path: Path, opened: os.stat_result) -> bool:
    try:
        current = path.lstat()
    except OSError:
        return False
    return (
        stat.S_ISREG(opened.st_mode)
        and stat.S_ISREG(current.st_mode)
        and _cache_identity(opened) == _cache_identity(current)
    )


def write_cache_chunk(path: Path, *, offset: int, data: bytes) -> int:
    """Write or verify one chunk through a no-follow file descriptor."""
    flags = (
        os.O_RDWR
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    if offset == 0:
        flags |= os.O_CREAT
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags, 0o600)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("cache path is not a regular file")
        os.fchmod(descriptor, 0o600)
        if offset == 0:
            os.ftruncate(descriptor, 0)
            current = 0
        else:
            current = opened.st_size
        if current == offset:
            os.lseek(descriptor, offset, os.SEEK_SET)
            view = memoryview(data)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("cache chunk write failed")
                view = view[written:]
            os.fsync(descriptor)
        elif current == offset + len(data):
            os.lseek(descriptor, offset, os.SEEK_SET)
            received = bytearray()
            while len(received) < len(data):
                block = os.read(descriptor, len(data) - len(received))
                if not block:
                    break
                received.extend(block)
            if bytes(received) != data:
                raise RemoteModelCacheError(409, "重试分块内容不一致")
        else:
            raise RemoteModelCacheError(409, "文件分块偏移不一致，请重新传输")
        after = os.fstat(descriptor)
        if not _same_regular_cache_file(path, after):
            raise OSError("cache path changed while open")
        return after.st_size
    except RemoteModelCacheError:
        raise
    except OSError as exc:
        raise RemoteModelCacheError(409, "远程文件缓存路径无效或无法安全写入") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def cached_fetch_digest(
    operation: RemoteOperation,
    *,
    path: Path | None = None,
    heartbeat: Callable[[], None] | None = None,
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
        descriptor = os.open(path, (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        ))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                return None, "远程产物缓存不是普通文件"
            if before.st_size != expected_size:
                return None, "远程产物缓存大小校验失败"
            digest = hashlib.sha256()
            if heartbeat is not None:
                heartbeat()
            next_heartbeat = time.monotonic() + MODEL_FINALIZE_HEARTBEAT_SECONDS
            for block in iter(lambda: handle.read(512 * 1024), b""):
                digest.update(block)
                if heartbeat is not None and time.monotonic() >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = time.monotonic() + MODEL_FINALIZE_HEARTBEAT_SECONDS
            if heartbeat is not None:
                heartbeat()
            after = os.fstat(handle.fileno())
        current = path.lstat()
    except OSError:
        return None, "远程产物缓存文件不可用"
    finally:
        if descriptor is not None:
            os.close(descriptor)
    before_identity = _cache_identity(before)
    after_identity = _cache_identity(after)
    current_identity = _cache_identity(current)
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


def model_fetches(db: Session, asset: ModelAsset) -> list[RemoteOperation]:
    """Return newest first operations that belong to the exact model source."""
    scope = model_asset_operation_scope(asset.id)
    return [
        operation for operation in db.scalars(select(RemoteOperation).where(
            RemoteOperation.task_id == scope,
            RemoteOperation.kind == "fetch_model",
        ).order_by(RemoteOperation.created_at.desc(), RemoteOperation.id.desc()))
        if operation_model_asset_id(operation) == asset.id
        and operation.node_id == asset.execution_node_id
        and (operation.payload or {}).get("path") == asset.path
    ]


def matching_model_fetch(
    db: Session,
    asset: ModelAsset,
    *,
    sha256: str | None = None,
    size: int | None = None,
) -> RemoteOperation | None:
    for operation in model_fetches(db, asset):
        payload = operation.payload or {}
        if sha256 is not None and payload.get("sha256") != sha256:
            continue
        if size is not None and payload.get("size") != size:
            continue
        return operation
    return None


def verified_remote_model_cache(
    db: Session,
    asset: ModelAsset,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[Path, str, int]:
    """Resolve immutable cached bytes, including an older version for reruns."""
    if asset.source_type != "remote_path":
        raise RemoteModelCacheError(422, "该模型来源不是远程路径")
    if expected_sha256 is None and expected_size is None and asset.status != "ready":
        if asset.status == "preparing":
            raise RemoteModelCacheError(409, "远程模型正在传输，请稍后重试")
        raise RemoteModelCacheError(409, asset.error or "远程模型尚未准备完成")
    sha256 = expected_sha256 if expected_sha256 is not None else asset.sha256
    size = expected_size if expected_size is not None else asset.size
    if not isinstance(sha256, str) or len(sha256) != 64 or not isinstance(size, int) or size <= 0:
        if asset.status == "preparing":
            raise RemoteModelCacheError(409, "远程模型正在传输，请稍后重试")
        raise RemoteModelCacheError(409, asset.error or "远程模型尚未准备完成")
    for operation in model_fetches(db, asset):
        payload = operation.payload or {}
        if operation.status != "succeeded" or payload.get("sha256") != sha256 or payload.get("size") != size:
            continue
        digest, cache_error = cached_fetch_digest(operation)
        if cache_error is None and digest is not None and secrets.compare_digest(digest, sha256):
            return operation_file(operation), digest, size
    raise RemoteModelCacheError(409, "远程模型的固定缓存已不可用，请重新登记或准备模型")


def _mark_model_fetch_failed(
    db: Session,
    operation: RemoteOperation,
    message: str,
    *,
    status: str = "failed",
) -> None:
    operation.status = status
    operation.error = message[-2000:]
    operation.updated_at = utcnow()
    operation_file(operation, False).unlink(missing_ok=True)
    asset_id = operation_model_asset_id(operation)
    asset = db.get(ModelAsset, asset_id) if asset_id else None
    if asset is not None:
        newest = matching_model_fetch(db, asset)
        # A delayed result from an older lease must not overwrite a newer
        # successful refresh of the same reusable model source.
        if newest is not None and newest.id == operation.id:
            asset.status = "failed" if status == "failed" else asset.status
            asset.error = message[-1000:] if status == "failed" else asset.error
            asset.updated_at = utcnow()


def _regular_file_size(path: Path) -> int:
    """Return a cache file's logical size without following a symlink."""
    try:
        metadata = path.lstat()
    except OSError:
        return 0
    return metadata.st_size if stat.S_ISREG(metadata.st_mode) else 0


def _referenced_model_versions(db: Session) -> set[tuple[str, str, int]]:
    """Return immutable remote-model versions that must remain rerunnable."""
    references: set[tuple[str, str, int]] = set()
    for asset_id, sha256, size in db.execute(select(
        ModelAsset.id, ModelAsset.sha256, ModelAsset.size,
    ).where(ModelAsset.source_type == "remote_path", ModelAsset.deleted_at.is_(None))):
        if isinstance(sha256, str) and isinstance(size, int) and not isinstance(size, bool) and size > 0:
            references.add((asset_id, sha256, size))
    for asset_id, sha256, size in db.execute(select(
        ModelTest.model_asset_id, ModelTest.model_sha256, ModelTest.model_size,
    ).where(ModelTest.model_asset_id.is_not(None))):
        if (
            isinstance(asset_id, str)
            and isinstance(sha256, str)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size > 0
        ):
            references.add((asset_id, sha256, size))
    return references


def _model_fetch_version(operation: RemoteOperation) -> tuple[str, str, int] | None:
    payload = operation.payload or {}
    asset_id = operation_model_asset_id(operation)
    sha256 = payload.get("sha256")
    size = payload.get("size")
    if (
        not asset_id
        or not isinstance(sha256, str)
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        return None
    return asset_id, sha256, size


def reclaim_unreferenced_model_cache(db: Session) -> int:
    """Remove only cache bytes that no current asset or history row can use."""
    references = _referenced_model_versions(db)
    reclaimed = 0
    operations = list(db.scalars(select(RemoteOperation).where(
        RemoteOperation.kind == "fetch_model",
    ).order_by(RemoteOperation.created_at, RemoteOperation.id)))
    for operation in operations:
        if operation.status in ACTIVE_MODEL_FETCH_STATUSES:
            continue
        version = _model_fetch_version(operation)
        if operation.status == "succeeded" and version in references:
            continue
        for path in (operation_file(operation), operation_file(operation, False)):
            size = _regular_file_size(path)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue
            reclaimed += size
        if operation.status == "succeeded":
            operation.status = "cancelled"
            operation.error = "远程模型缓存已按容量策略安全回收"
            operation.updated_at = utcnow()
    return reclaimed


def remote_model_cache_usage(db: Session) -> tuple[int, int]:
    """Return quota usage and bytes still needed by all active model fetches."""
    quota_bytes = 0
    remaining_bytes = 0
    for operation in db.scalars(select(RemoteOperation).where(
        RemoteOperation.kind == "fetch_model",
    )):
        complete_size = _regular_file_size(operation_file(operation))
        partial_size = _regular_file_size(operation_file(operation, False))
        payload_size = (operation.payload or {}).get("size")
        if (
            operation.status in ACTIVE_MODEL_FETCH_STATUSES
            and isinstance(payload_size, int)
            and not isinstance(payload_size, bool)
            and payload_size > 0
        ):
            # Reserve the full declared object, not merely the first received
            # chunk. A stray complete file, if present, is counted separately.
            quota_bytes += complete_size + max(payload_size, partial_size)
            remaining_bytes += max(payload_size - partial_size, 0)
        else:
            quota_bytes += complete_size + partial_size
    return quota_bytes, remaining_bytes


def ensure_remote_model_cache_capacity(db: Session) -> None:
    """Reclaim safe candidates and reject transfers that cannot fit."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    reclaim_unreferenced_model_cache(db)
    quota_bytes, remaining_bytes = remote_model_cache_usage(db)
    if REMOTE_MODEL_CACHE_QUOTA_BYTES and quota_bytes > REMOTE_MODEL_CACHE_QUOTA_BYTES:
        raise RemoteModelCacheError(507, "远程模型缓存空间不足：本次模型会超过缓存总配额")
    if REMOTE_MODEL_CACHE_MIN_FREE_BYTES:
        try:
            free_bytes = shutil.disk_usage(CACHE_ROOT).free
        except OSError as exc:
            raise RemoteModelCacheError(507, "无法读取远程模型缓存磁盘容量") from exc
        if free_bytes - remaining_bytes < REMOTE_MODEL_CACHE_MIN_FREE_BYTES:
            raise RemoteModelCacheError(507, "远程模型缓存所在磁盘空间不足，请清理空间后重试")


def prepare_remote_model_asset(
    db: Session,
    asset: ModelAsset,
    *,
    retry: bool = False,
    refresh: bool = False,
) -> dict:
    """Idempotently enqueue transfer of a standalone remote checkpoint."""
    if asset.deleted_at is not None:
        raise HTTPException(409, "该模型登记已移除，请重新登记后再准备模型")
    if asset.source_type != "remote_path":
        raise HTTPException(422, "该模型来源不是远程路径")
    operations = model_fetches(db, asset)
    latest = operations[0] if operations else None
    if latest and latest.status in ACTIVE_MODEL_FETCH_STATUSES:
        asset.status = "preparing"
        asset.error = None
        asset.updated_at = utcnow()
        return {
            "status": "preparing",
            "operation_id": latest.id,
            "message": "等待远程 Agent 传输模型",
        }
    if latest and latest.status == "succeeded" and not refresh:
        digest, cache_error = cached_fetch_digest(latest)
        payload = latest.payload or {}
        size = payload.get("size")
        if (
            cache_error is None and digest is not None
            and isinstance(size, int) and size > 0
            and (asset.sha256 is None or secrets.compare_digest(asset.sha256, digest))
            and (asset.size is None or asset.size == size)
        ):
            asset.sha256 = digest
            asset.size = size
            asset.status = "ready"
            asset.error = None
            asset.updated_at = utcnow()
            return {"status": "ready", "operation_id": latest.id}
        refresh = True
    if latest and latest.status == "failed" and not retry and not refresh:
        asset.status = "failed"
        asset.error = latest.error or "远程模型传输失败"
        asset.updated_at = utcnow()
        return {
            "status": "failed",
            "operation_id": latest.id,
            "message": asset.error,
        }

    node = db.get(ComputeNode, asset.execution_node_id)
    unavailable = remote_model_test_unavailable_reason(node)
    if unavailable:
        asset.status = "failed"
        asset.error = unavailable
        asset.updated_at = utcnow()
        raise HTTPException(409, unavailable)
    assert node is not None
    from .model_assets import ModelAssetPathError, validate_remote_model_path
    try:
        validate_remote_model_path(node, asset.path)
    except ModelAssetPathError as exc:
        asset.status = "failed"
        asset.error = str(exc)
        asset.updated_at = utcnow()
        raise HTTPException(exc.status_code, str(exc)) from exc

    operation = enqueue(
        db,
        node.id,
        model_asset_operation_scope(asset.id),
        "fetch_model",
        {"model_asset_id": asset.id, "path": asset.path},
    )
    asset.status = "preparing"
    asset.error = None
    asset.updated_at = utcnow()
    return {
        "status": "preparing",
        "operation_id": operation.id,
        "message": "等待远程 Agent 传输模型",
    }


def run_payload(run: TrainingRun, task: TrainingTask) -> dict:
    output_root = remote_output_root_for_run(task, run)
    layout = None
    run_dir = run.run_dir
    if output_root:
        try:
            layout, _, validated_run_dir = remote_attempt_layout(task, run)
            run_dir = str(validated_run_dir)
        except ValueError:
            pass
    return {"task_id": task.id, "task_type": task.task_type, "run_id": run.id,
            "attempt_number": run.attempt_number, "run_dir": run_dir, "log_path": run.log_path,
            "output_root": output_root, **({"layout": layout} if layout else {})}


def enqueue(db: Session, node_id: str, task_id: str, kind: str, payload: dict) -> RemoteOperation:
    operation = RemoteOperation(id=uuid.uuid4().hex, node_id=node_id, task_id=task_id,
                                kind=kind, payload=payload, status="pending")
    db.add(operation)
    db.flush()
    return operation


def queue_batch_history_scan(db: Session, node: ComputeNode) -> bool:
    """Queue one small read-only historical scan per heartbeat after Agent upgrade."""
    if not agent_version_at_least(node.agent_version, (2, 5, 3)):
        return False
    if db.scalar(select(RemoteOperation.id).where(
        RemoteOperation.node_id == node.id,
        RemoteOperation.status.in_(["pending", "running", "finalizing"]),
    ).limit(1)) is not None:
        return False
    scannable = {"running", "stopping", "completed", "failed", "stopped", "interrupted"}
    existing = select(RemoteOperation.id).where(
        RemoteOperation.kind == "batch_history",
        RemoteOperation.payload["run_id"].as_string() == TrainingRun.id,
    ).correlate(TrainingRun).exists()
    candidates = db.execute(select(TrainingRun, TrainingTask).join(
        TrainingTask, TrainingTask.id == TrainingRun.task_id,
    ).where(
        or_(TrainingRun.execution_node_id == node.id,
            and_(TrainingRun.execution_node_id.is_(None), TrainingTask.execution_node_id == node.id)),
        TrainingTask.deleted_at.is_(None), TrainingRun.status.in_(scannable),
        TrainingRun.batch_history_scanned.is_(False), TrainingRun.log_path.is_not(None),
        ~existing,
    ).order_by(TrainingRun.created_at, TrainingRun.id).limit(16)).all()
    from .run_deletions import remote_attempt_layout
    for run, task in candidates:
        try:
            remote_attempt_layout(task, run)
            snapshot = run.parameter_snapshot if isinstance(run.parameter_snapshot, dict) else {}
            payload = {**run_payload(run, task), "batch": int(snapshot.get("batch") or task.batch)}
            if not 1 <= payload["batch"] <= 1024:
                raise ValueError("Batch 配置无效")
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            # A durable failed receipt keeps malformed legacy rows from blocking
            # every later Attempt on this node. The UI continues to show unknown.
            operation = enqueue(db, node.id, task.id, "batch_history", {"run_id": run.id})
            operation.status, operation.error = "failed", f"历史 Batch 扫描不可用：{exc}"[:2000]
            continue
        enqueue(db, node.id, task.id, "batch_history", payload)
        return True
    return False


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


def _remote_path_overlaps(first: str, second: str) -> bool:
    left, right = PurePosixPath(first), PurePosixPath(second)
    return left == right or left in right.parents or right in left.parents


def _protected_remote_model_assets(
    db: Session,
    node_id: str,
    deleting_task_id: str,
) -> list[tuple[ModelAsset, bool]]:
    """Return live sources and removed sources retained by external test history."""
    protected: list[tuple[ModelAsset, bool]] = []
    assets = db.scalars(select(ModelAsset).where(
        ModelAsset.execution_node_id == node_id,
        ModelAsset.source_type == "remote_path",
    )).all()
    for asset in assets:
        retained_by_history = asset.deleted_at is not None and db.scalar(select(ModelTest.id).where(
            ModelTest.model_asset_id == asset.id,
            (ModelTest.task_id.is_(None)) | (ModelTest.task_id != deleting_task_id),
        ).limit(1)) is not None
        if asset.deleted_at is None or retained_by_history:
            protected.append((asset, retained_by_history))
    return protected


def build_remote_task_delete_plan(
    db: Session,
    task: TrainingTask,
    *,
    acquire_lock: bool = True,
) -> dict:
    """Validate and freeze the complete remote portion of a whole-task deletion."""
    task_id, node_id = task.id, task.execution_node_id
    if not node_id:
        raise ValueError("本机任务不能进入远程删除队列")
    if acquire_lock:
        acquire_node_claim_lock(db, node_id)
        task = db.get(TrainingTask, task_id)
        if task is None or task.execution_node_id != node_id:
            raise ValueError("远程任务归属已变更，请刷新后重试")
    node = db.get(ComputeNode, node_id)
    if not node:
        raise ValueError("远程服务器不存在，无法验证删除计划")
    if task.status in ACTIVE:
        raise ValueError("运行中或排队中的任务不能永久删除")

    plans: list[dict] = []
    output_evidence = bool(task.run_dir or task.log_path or task.current_run_id)
    artifacts = list(db.scalars(select(Artifact).where(Artifact.task_id == task.id)))
    output_evidence = output_evidence or any(artifact.path for artifact in artifacts)
    for run in sorted(task.runs, key=lambda item: (item.attempt_number, item.id)):
        if run.status in ACTIVE:
            raise ValueError(f"Attempt {run.attempt_number:03d} 仍在运行或排队，不能永久删除")
        if run.execution_node_id not in {None, node_id}:
            raise ValueError(
                f"Attempt {run.attempt_number:03d} 属于其他远程服务器，已拒绝清理"
            )
        output_evidence = output_evidence or bool(
            run.run_dir or run.log_path or run.checkpoint_path
            or any(artifact.run_id == run.id and artifact.path for artifact in artifacts)
        )
        try:
            layout, output_root, validated_run_dir = remote_attempt_layout(task, run)
        except ValueError as exc:
            raise ValueError(f"Attempt {run.attempt_number:03d} 无法形成安全删除计划：{exc}") from exc
        payload = run_payload(run, task)
        payload.update({
            "output_root": str(output_root),
            "run_dir": str(validated_run_dir),
            "layout": layout,
        })
        plans.append(payload)

    if not plans and output_evidence:
        raise ValueError("任务存在训练目录、日志或产物记录，但没有可验证的远程 Attempt，已拒绝清理")
    if plans and not agent_version_at_least(node.agent_version, TASK_DELETE_AGENT_VERSION_PARTS):
        raise ValueError(f"整任务永久删除需要远程 Agent {TASK_DELETE_AGENT_VERSION} 或以上版本")
    if task.current_run_id and all(run.id != task.current_run_id for run in task.runs):
        raise ValueError("当前 Attempt 记录不存在，已拒绝清理")

    protected = live_protected_paths(db, node_id, task.id)
    target_paths: list[str] = []
    for run, plan in zip(sorted(task.runs, key=lambda item: (item.attempt_number, item.id)), plans):
        target_paths.extend(str(path) for path in remote_attempt_path_candidates(
            task,
            run,
            [artifact.path for artifact in artifacts if artifact.run_id == run.id],
        ))
        if plan.get("layout") == LEGACY_FLAT_LAYOUT:
            target_paths.append(plan["run_dir"])
    for target in target_paths:
        for asset, retained_by_history in _protected_remote_model_assets(db, node_id, task.id):
            if asset.path and _remote_path_overlaps(target, asset.path):
                if retained_by_history:
                    raise ValueError(
                        f"已移除的模型登记「{asset.name}」仍有模型测试历史引用；"
                        "请先删除相关历史记录后再永久删除任务"
                    )
                raise ValueError(f"训练产物仍被模型来源「{asset.name}」登记，请先移除该模型来源")
        if any(_remote_path_overlaps(target, path) for path in protected):
            raise ValueError("待清理远程目录仍被模型、数据集或其他任务引用，已拒绝永久删除")

    targets = deletion_plan_targets(plans)
    test_ids = sorted(db.scalars(select(ModelTest.id).where(ModelTest.task_id == task.id)).all())
    from .final_receipts import receipt_proofs
    return {
        "scope": "task",
        "phase": "stopping_tests",
        "task_name": task.name,
        "node_name": node.name,
        "test_ids": test_ids,
        "runs": plans,
        "final_receipt_proofs": receipt_proofs(task.runs),
        "protected_paths": protected,
        "receipt_required": True,
        "receipt_version": DELETE_RECEIPT_VERSION,
        "plan_digest": deletion_plan_digest(plans),
        "plan_targets": targets,
    }


def queue_delete(db: Session, task: TrainingTask) -> RemoteOperation:
    task_id, node_id = task.id, task.execution_node_id
    if not node_id:
        raise ValueError("本机任务不能进入远程删除队列")
    acquire_node_claim_lock(db, node_id)
    task = db.get(TrainingTask, task_id)
    if task is None or task.execution_node_id != node_id:
        raise ValueError("远程任务归属已变更，请刷新后重试")
    operations = list(db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id,
        RemoteOperation.kind == "delete",
    ).order_by(RemoteOperation.created_at.desc())))
    existing = next((operation for operation in operations if (operation.payload or {}).get("scope") != "attempt"), None)
    if existing and existing.status in {"pending", "running", "finalizing", "center_cleaning"}:
        existing_payload = existing.payload or {}
        modern = (
            existing_payload.get("scope") == "task"
            and existing_payload.get("receipt_required") is True
            and existing_payload.get("receipt_version") == DELETE_RECEIPT_VERSION
        )
        if modern:
            return existing
        existing.status = "cancelled"
        existing.error = "旧版删除指令缺少验证回执要求，已由新协议替换"
        existing.updated_at = utcnow()
    if existing and (existing.payload or {}).get("remote_verified"):
        payload = dict(existing.payload or {})
        payload["phase"] = "center_cleanup"
        existing.payload = payload
        existing.status = "finalizing"
        existing.error = None
        existing.updated_at = utcnow()
        task.deleted_at = task.deleted_at or utcnow()
        task.deleted_reason = "remote_delete_pending"
        add_event(db, task.id, "remote_delete_center_retry", "远程文件已确认清理，正在重试平台关联数据清理")
        return existing
    active = active_remote_delete(db, node_id)
    if active and active.task_id != task.id:
        raise ValueError("该远程服务器正在清理其他任务文件，请稍后重试")

    payload = build_remote_task_delete_plan(db, task, acquire_lock=False)
    status = "pending"
    if not payload["runs"]:
        receipt = {
            "version": DELETE_RECEIPT_VERSION,
            "plan_digest": payload["plan_digest"],
            "targets": [],
            "removed": [],
            "already_absent": [],
            "verified_absent": True,
        }
        payload.update({
            "phase": "center_cleanup",
            "remote_verified": True,
            "receipt": receipt,
        })
        status = "finalizing"
    operation = enqueue(db, node_id, task.id, "delete", payload)
    operation.status = status
    task.deleted_at = task.deleted_at or utcnow()
    task.deleted_reason = "remote_delete_pending"
    message = (
        "任务尚未产生远程训练文件，正在清理平台关联数据"
        if status == "finalizing"
        else "等待远程 Agent 清理该任务文件；确认后删除记录"
    )
    add_event(db, task.id, "remote_delete_queued", message)
    return operation


def prepare_attempt_delete_payload(db: Session, task: TrainingTask, run: TrainingRun) -> dict:
    """Read-only preflight shared by previews and actual delete submissions."""
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
    releases = checkpoint_releases(db, task, run)
    if releases and not agent_version_at_least(node.agent_version, (2, 0, 8)):
        raise ValueError("释放历史续训检查点依赖需要 Agent 2.0.8 或以上版本")
    absent_files = checkpoint_required_absent_files(db, task, run)
    if absent_files and not agent_version_at_least(node.agent_version, ABSENT_BEST_AGENT_VERSION_PARTS):
        raise ValueError(f"验证历史最佳权重缺失需要远程 Agent {ABSENT_BEST_AGENT_VERSION} 或以上版本")
    from .final_receipts import receipt_proofs
    payload = {
        "required_files": [file for release in releases for file in release["required_files"]],
        "required_absent_files": absent_files,
        "checkpoint_releases": releases,
        "scope": "attempt",
        "run_id": run.id,
        "attempt_number": run.attempt_number,
        "runs": [payload_run],
        "final_receipt_proofs": receipt_proofs([run]),
        "protected_paths": live_protected_paths(db, task.execution_node_id, task.id, deleting_run_id=run.id),
    }
    return payload


def queue_attempt_delete(db: Session, task: TrainingTask, run: TrainingRun) -> RemoteOperation:
    """Queue one remote Attempt for Agent cleanup while retaining its task."""
    payload = prepare_attempt_delete_payload(db, task, run)
    operation = enqueue(db, task.execution_node_id, task.id, "delete", payload)
    add_event(db, task.id, "run_delete_queued", f"已提交远程 Attempt {run.attempt_number:03d} 文件清理", run_id=run.id)
    return operation


def live_protected_paths(
    db: Session,
    node_id: str,
    deleting_task_id: str,
    *,
    deleting_run_id: str | None = None,
    released_run_ids: set[str] | None = None,
) -> list[str]:
    """Return current references that a queued remote deletion must preserve."""
    protected: set[str] = set()
    source = db.get(TrainingRun, deleting_run_id) if deleting_run_id else None
    source_task = db.get(TrainingTask, deleting_task_id) if source else None
    released = (
        set(released_run_ids) if released_run_ids is not None
        else {item["run_id"] for item in checkpoint_releases(db, source_task, source)} if source_task else set()
    )
    tasks = db.scalars(select(TrainingTask).where(TrainingTask.execution_node_id == node_id))
    for task in tasks:
        # Task resources are shared by all its Attempts and must never be
        # removed by an Attempt-scoped operation.
        protected.update(path for path in (task.model_path, task.data_path) if path)
        if task.id == deleting_task_id and deleting_run_id is None:
            continue
        for run in task.runs:
            if task.id == deleting_task_id and deleting_run_id and run.id == deleting_run_id:
                continue
            protected.update(path for path in (run.model_path, run.data_path) if path)
            if run.checkpoint_path and run.id not in released:
                protected.add(run.checkpoint_path)
            protected.update(
                artifact.path for artifact in db.scalars(select(Artifact).where(Artifact.run_id == run.id))
            )
    protected.update(
        asset.path for asset, _retained in _protected_remote_model_assets(
            db, node_id, deleting_task_id,
        ) if asset.path
    )
    protected.update(
        test.model_source_path for test in db.scalars(select(ModelTest).where(
            ModelTest.execution_node_id == node_id,
            ModelTest.status.in_(["uploading", "queued", "running", "cancel_requested"]),
            (ModelTest.task_id.is_(None)) | (ModelTest.task_id != deleting_task_id),
        )) if test.model_source_path
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
        RemoteOperation.status.in_(["pending", "running", "finalizing", "center_cleaning"]),
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
    # Share the node row/write lock with node management and model-source
    # registration so Token rotation cannot race a pending fetch into running.
    acquire_node_claim_lock(db, node_id)
    node = authenticated_agent(node_id, request, db)
    if not agent_version_at_least(node.agent_version):
        return {"operation": None, "required_agent_version": REQUIRED_AGENT_VERSION,
                "message": f"请将远程 Agent 升级到 {REQUIRED_AGENT_VERSION} 或以上版本"}
    cutoff = utcnow() - OPERATION_LEASE_TIMEOUT
    query = select(RemoteOperation).where(
        RemoteOperation.node_id == node_id,
        (RemoteOperation.status == "pending") |
        ((RemoteOperation.status.in_(["running", "finalizing"])) & (RemoteOperation.updated_at < cutoff)),
    )
    if not agent_version_at_least(node.agent_version, (2, 5, 3)):
        query = query.where(RemoteOperation.kind != "batch_history")
    operation = db.scalar(query.order_by(
        case((RemoteOperation.kind == "batch_history", 1), else_=0),
        RemoteOperation.created_at,
    ).limit(1))
    if not operation:
        return {"operation": None}
    operation_payload = operation.payload
    if operation.kind == "fetch_log":
        try:
            validate_log_fetch(db, operation)
            if not agent_version_at_least(node.agent_version, (2, 0, 6)):
                raise HTTPException(409, "远程完整日志下载需要 Agent 2.0.6 或以上版本")
        except HTTPException as exc:
            operation.status, operation.error = "failed", str(exc.detail)
            operation_file(operation, False).unlink(missing_ok=True)
            db.commit()
            return {"operation": None, "message": str(exc.detail)}
        # A fresh lease takes a fresh bounded snapshot; stale chunks cannot mix.
        operation_payload = {key: value for key, value in operation.payload.items()
                             if key not in {"size", "modified_at", "mtime_ns", "sha256"}}
    task_delete = (
        operation.kind == "delete"
        and (operation.payload or {}).get("scope") == "task"
    )
    if task_delete and (operation.payload or {}).get("remote_verified"):
        return {"operation": None, "message": "远程文件已确认清理，正在清理平台关联数据"}
    if task_delete and not agent_version_at_least(node.agent_version, TASK_DELETE_AGENT_VERSION_PARTS):
        task = db.get(TrainingTask, operation.task_id)
        message = f"整任务永久删除需要远程 Agent {TASK_DELETE_AGENT_VERSION} 或以上版本"
        _mark_task_delete_failed(db, operation, task, message)
        db.commit()
        return {
            "operation": None,
            "required_agent_version": TASK_DELETE_AGENT_VERSION,
            "message": message,
        }
    if operation.kind == "fetch_model":
        asset_id = operation_model_asset_id(operation)
        asset = db.get(ModelAsset, asset_id) if asset_id else None
        latest = matching_model_fetch(db, asset) if asset is not None else None
        invalid = (
            asset is None
            or asset.source_type != "remote_path"
            or asset.execution_node_id != node_id
            or asset.path != (operation.payload or {}).get("path")
            or latest is None
            or latest.id != operation.id
            or asset.status != "preparing"
        )
        if invalid:
            _mark_model_fetch_failed(db, operation, "模型来源已变更，远程传输指令已取消", status="cancelled")
            db.commit()
            return {"operation": None}
        unavailable = remote_model_test_unavailable_reason(node)
        if unavailable:
            _mark_model_fetch_failed(db, operation, unavailable)
            db.commit()
            return {
                "operation": None,
                "required_agent_version": REMOTE_MODEL_TEST_AGENT_VERSION,
                "message": unavailable,
            }
        from .model_assets import ModelAssetPathError, validate_remote_model_path
        try:
            validate_remote_model_path(node, asset.path)
        except ModelAssetPathError as exc:
            _mark_model_fetch_failed(db, operation, str(exc))
            db.commit()
            return {"operation": None, "message": str(exc)}
    if operation.kind == "delete":
        task = db.get(TrainingTask, operation.task_id)
        attempt_scope = (operation.payload or {}).get("scope") == "attempt"
        task_scope = (operation.payload or {}).get("scope") == "task"
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
                or task.deleted_reason != "remote_delete_pending" or not task_scope
            )
        )
        if invalid_attempt or invalid_task:
            message = "任务状态或删除协议已变更，删除指令已取消"
            cancelled = db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation.id,
                RemoteOperation.status == operation.status,
                RemoteOperation.updated_at == operation.updated_at,
            ).values(status="cancelled", error=message, updated_at=utcnow()))
            if cancelled.rowcount != 1:
                db.rollback()
                return {"operation": None}
            if invalid_task and task is not None:
                task.deleted_reason = "remote_delete_failed"
                add_event(db, task.id, "remote_delete_failed", message, "error")
            db.commit()
            return {"operation": None, "message": message}
        # A node can be offline for a long time after deletion was queued. Refresh
        # references immediately before dispatch so newly-created tasks are safe.
        if attempt_scope:
            try:
                validate_attempt_deletion(db, task, target_run, ignore_operation_id=operation.id)
                releases = checkpoint_releases(db, task, target_run)
                if releases and not agent_version_at_least(node.agent_version, (2, 0, 8)):
                    raise ValueError("释放历史续训检查点依赖需要 Agent 2.0.8 或以上版本")
                absent_files = checkpoint_required_absent_files(db, task, target_run)
                if absent_files and not agent_version_at_least(node.agent_version, ABSENT_BEST_AGENT_VERSION_PARTS):
                    raise ValueError(f"验证历史最佳权重缺失需要远程 Agent {ABSENT_BEST_AGENT_VERSION} 或以上版本")
            except ValueError as exc:
                operation.status, operation.error = "failed", str(exc)
                db.commit()
                return {"operation": None, "message": str(exc)}
        operation_payload = {
            **operation.payload,
            **({"required_files": [file for release in releases for file in release["required_files"]]} if attempt_scope else {}),
            **({"required_absent_files": absent_files} if attempt_scope else {}),
            **({"checkpoint_releases": releases} if attempt_scope else {}),
            "protected_paths": live_protected_paths(
                db,
                node_id,
                operation.task_id,
                deleting_run_id=(operation.payload or {}).get("run_id") if attempt_scope else None,
                released_run_ids={item["run_id"] for item in releases} if attempt_scope else None,
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
    if operation.kind == "fetch_log":
        # Only the worker that owns the new lease may remove a previous
        # partial snapshot. The row remains locked until this reset is durable.
        for path in (operation_file(operation), operation_file(operation, False)):
            path.unlink(missing_ok=True)
    db.commit()
    db.refresh(operation)
    if task_delete:
        try:
            from .inference import cancel_task_tests
            cancel_task_tests(operation.task_id, purge=True, bind=db.get_bind())
        except Exception as exc:
            detail = getattr(exc, "detail", None) or str(exc) or type(exc).__name__
            message = f"停止模型检测失败：{detail}"[-2000:]
            db.expire_all()
            current = db.get(RemoteOperation, operation.id)
            task = db.get(TrainingTask, operation.task_id)
            if current is not None and current.status == "running" and current.lease == lease:
                _mark_task_delete_failed(db, current, task, message)
                db.commit()
            return {"operation": None, "message": message}
        db.expire_all()
        operation = db.get(RemoteOperation, operation.id)
        if operation is None or operation.status != "running" or operation.lease != lease:
            return {"operation": None}
        dispatched_payload = dict(operation.payload or {})
        dispatched_payload["phase"] = "remote_cleanup"
        operation.payload = dispatched_payload
        operation.updated_at = utcnow()
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


class DeleteReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    targets: list[str] = Field(max_length=4096)
    removed: list[str] = Field(max_length=4096)
    already_absent: list[str] = Field(max_length=4096)
    verified_absent: bool


class OperationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lease: str = Field(pattern=r"^[0-9a-f]{32}$")
    success: bool
    error: str | None = Field(default=None, max_length=2000)
    receipt: DeleteReceipt | None = None
    batch_history: list[int] | None = Field(default=None, max_length=8)


def _verified_delete_receipt(operation: RemoteOperation, receipt: DeleteReceipt | None) -> dict:
    expected = operation.payload or {}
    if receipt is None:
        raise ValueError("远程 Agent 未返回删除验证回执")
    value = receipt.model_dump()
    if value["version"] != DELETE_RECEIPT_VERSION or value["version"] != expected.get("receipt_version"):
        raise ValueError("远程删除验证回执版本不匹配")
    if not value["verified_absent"]:
        raise ValueError("远程 Agent 未确认全部目标已经清理")
    expected_digest = expected.get("plan_digest")
    if value["plan_digest"] != expected_digest:
        raise ValueError("远程删除验证回执与冻结计划不匹配")
    targets = value["targets"]
    removed = value["removed"]
    absent = value["already_absent"]
    if (
        len(targets) != len(set(targets))
        or len(removed) != len(set(removed))
        or len(absent) != len(set(absent))
        or set(removed) & set(absent)
        or set(targets) != set(expected.get("plan_targets") or [])
        or set(removed) | set(absent) != set(targets)
    ):
        raise ValueError("远程删除验证回执没有完整覆盖冻结计划")
    return value


def _mark_task_delete_failed(
    db: Session,
    operation: RemoteOperation,
    task: TrainingTask | None,
    message: str,
) -> None:
    operation.status = "failed"
    operation.error = message[-2000:]
    operation.updated_at = utcnow()
    if task is not None:
        task.deleted_reason = "remote_delete_failed"
        add_event(db, task.id, "remote_delete_failed", operation.error, "error")


def _remove_center_path(path: Path) -> bool:
    """Remove one center-owned path and prove it is absent afterward."""
    present = path.is_symlink() or path.exists()
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.exists():
        shutil.rmtree(path)
    if path.is_symlink() or path.exists():
        raise RuntimeError(f"中心清理后路径仍然存在：{path}")
    return present


def _remove_center_cache_file(path: Path) -> bool:
    """Delete only a regular cache file (or its link), never an unexpected directory."""
    if not path.is_symlink() and not path.exists():
        return False
    if not path.is_symlink() and not path.is_file():
        raise RuntimeError(f"远程产物缓存路径不是普通文件：{path}")
    path.unlink(missing_ok=True)
    if path.is_symlink() or path.exists():
        raise RuntimeError(f"远程产物缓存清理后仍然存在：{path}")
    return True


def _owns_center_cleanup(operation: RemoteOperation | None, claim_token: str) -> bool:
    token = (operation.payload or {}).get("center_claim_token") if operation is not None else None
    return (
        operation is not None
        and operation.status == "center_cleaning"
        and isinstance(token, str)
        and secrets.compare_digest(token, claim_token)
    )


def _heartbeat_center_cleanup(session_factory, operation_id: str, claim_token: str) -> bool:
    """Renew a center-cleanup claim only while this exact worker still owns it."""
    with session_factory() as db:
        operation = db.get(RemoteOperation, operation_id)
        if not _owns_center_cleanup(operation, claim_token):
            return False
        previous_update = operation.updated_at
        changed = db.execute(update(RemoteOperation).where(
            RemoteOperation.id == operation_id,
            RemoteOperation.status == "center_cleaning",
            RemoteOperation.updated_at == previous_update,
        ).values(updated_at=utcnow()))
        if changed.rowcount != 1:
            db.rollback()
            return False
        db.commit()
        return True


def _center_cleanup_heartbeat_loop(
    session_factory,
    operation_id: str,
    claim_token: str,
    stop: threading.Event,
    claim_lost: threading.Event,
) -> None:
    while not stop.wait(CENTER_CLEANUP_HEARTBEAT_SECONDS):
        try:
            if not _heartbeat_center_cleanup(session_factory, operation_id, claim_token):
                claim_lost.set()
                return
        except Exception:
            # A transient database error must not terminate filesystem cleanup.
            # The final compare-and-swap still prevents a stale worker from
            # committing success or failure after another worker reclaims it.
            continue


def process_remote_delete_center(session_factory) -> bool:
    """Claim and idempotently finish one Agent-verified task deletion."""
    operation_id: str | None = None
    task_id: str | None = None
    payload: dict = {}
    claim_token: str | None = None
    heartbeat_stop: threading.Event | None = None
    heartbeat_lost: threading.Event | None = None
    heartbeat_thread: threading.Thread | None = None
    try:
        with session_factory() as db:
            candidates = list(db.scalars(select(RemoteOperation).where(
                RemoteOperation.kind == "delete",
                RemoteOperation.status.in_(["finalizing", "center_cleaning"]),
            ).order_by(RemoteOperation.updated_at, RemoteOperation.created_at)))
            operation = next((candidate for candidate in candidates if (
                (candidate.payload or {}).get("scope") == "task"
                and (candidate.payload or {}).get("remote_verified") is True
                and (
                    candidate.status == "finalizing"
                    or as_utc(candidate.updated_at) < utcnow() - CENTER_CLEANUP_LEASE_TIMEOUT
                )
            )), None)
            if operation is None:
                return False
            claim_token = secrets.token_hex(16)
            claimed_at = utcnow()
            claimed_payload = dict(operation.payload or {})
            claimed_payload.update({
                "phase": "center_cleanup",
                "center_claim_token": claim_token,
                "center_claimed_at": claimed_at.isoformat(),
            })
            claimed = db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation.id,
                RemoteOperation.status == operation.status,
                RemoteOperation.updated_at == operation.updated_at,
            ).values(
                status="center_cleaning",
                payload=claimed_payload,
                error=None,
                updated_at=claimed_at,
            ))
            if claimed.rowcount != 1:
                db.rollback()
                return False
            operation_id, task_id, payload = operation.id, operation.task_id, claimed_payload
            db.commit()

            task = db.get(TrainingTask, task_id)
            current_test_ids = (
                list(db.scalars(select(ModelTest.id).where(ModelTest.task_id == task_id)))
                if task is not None else []
            )
            test_ids = sorted(set(payload.get("test_ids") or []) | set(current_test_ids))
            run_ids = {
                run.get("run_id") for run in payload.get("runs") or []
                if isinstance(run, dict) and isinstance(run.get("run_id"), str)
            }
            if task is not None:
                run_ids.update(run.id for run in task.runs)
            related_operation_ids = list(db.scalars(select(RemoteOperation.id).where(
                RemoteOperation.task_id == task_id,
            )))

        heartbeat_stop = threading.Event()
        heartbeat_lost = threading.Event()
        heartbeat_thread = threading.Thread(
            target=_center_cleanup_heartbeat_loop,
            args=(session_factory, operation_id, claim_token, heartbeat_stop, heartbeat_lost),
            name=f"remote-center-cleanup-{operation_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()

        from .task_deletions import cleanup_task_test_files
        cleanup_task_test_files(task_id, test_ids, delete_history=delete_task_history)
        if heartbeat_lost.is_set():
            raise RuntimeError("中心清理租约已失效，已停止提交清理结果")
        removed_trace_count = 0
        for run_id in run_ids:
            removed_trace_count += int(_remove_center_path(remote_trace_root(run_id)))
        removed_cache_count = 0
        cache_target_count = 0
        for related_id in related_operation_ids:
            for complete in (True, False):
                cache_path = CACHE_ROOT / (related_id + (".bin" if complete else ".part"))
                cache_target_count += 1
                removed_cache_count += int(_remove_center_cache_file(cache_path))

        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5)
        heartbeat_thread = None
        if heartbeat_lost.is_set() or not _heartbeat_center_cleanup(
            session_factory, operation_id, claim_token,
        ):
            raise RuntimeError("中心清理租约已失效，已停止提交清理结果")

        with session_factory() as db:
            operation = db.get(RemoteOperation, operation_id)
            task = db.get(TrainingTask, task_id)
            if operation is None:
                return True
            if not _owns_center_cleanup(operation, claim_token):
                return True
            if not (operation.payload or {}).get("remote_verified"):
                raise RuntimeError("远程验证状态已丢失，已拒绝删除任务记录")
            previous_update = operation.updated_at
            completed_payload = dict(operation.payload or {})
            completed_payload.pop("center_claim_token", None)
            completed_payload.update({
                "phase": "complete",
                "center_cleanup": {
                    "started_at": completed_payload.pop("center_claimed_at", None),
                    "completed_at": utcnow().isoformat(),
                    "test_count": len(test_ids),
                    "run_count": len(run_ids),
                    "trace_target_count": len(run_ids),
                    "removed_trace_count": removed_trace_count,
                    "cache_target_count": cache_target_count,
                    "removed_cache_count": removed_cache_count,
                },
            })
            completed = db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation_id,
                RemoteOperation.status == "center_cleaning",
                RemoteOperation.updated_at == previous_update,
            ).values(
                payload=completed_payload,
                status="succeeded",
                error=None,
                updated_at=utcnow(),
            ))
            if completed.rowcount != 1:
                db.rollback()
                return True
            for other in db.scalars(select(RemoteOperation).where(
                RemoteOperation.task_id == task_id,
                RemoteOperation.id != operation_id,
                RemoteOperation.status.in_(["pending", "running", "finalizing", "center_cleaning"]),
            )):
                other.status = "cancelled"
                other.error = "任务永久删除已完成"
                other.updated_at = utcnow()
            if task is not None:
                db.delete(task)
            db.commit()
        return True
    except Exception as exc:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=5)
            heartbeat_thread = None
        if operation_id is None:
            raise
        detail = f"中心关联数据清理失败：{type(exc).__name__}: {exc}"[-2000:]
        try:
            with session_factory() as db:
                operation = db.get(RemoteOperation, operation_id)
                task = db.get(TrainingTask, task_id) if task_id else None
                if claim_token is not None and _owns_center_cleanup(operation, claim_token):
                    previous_update = operation.updated_at
                    failed_payload = dict(operation.payload or payload)
                    failed_payload.pop("center_claim_token", None)
                    failed_payload.pop("center_claimed_at", None)
                    failed_payload.update({"phase": "center_cleanup_failed", "remote_verified": True})
                    failed = db.execute(update(RemoteOperation).where(
                        RemoteOperation.id == operation_id,
                        RemoteOperation.status == "center_cleaning",
                        RemoteOperation.updated_at == previous_update,
                    ).values(
                        payload=failed_payload,
                        status="failed",
                        error=detail,
                        updated_at=utcnow(),
                    ))
                    if failed.rowcount != 1:
                        db.rollback()
                        return True
                    if task is not None:
                        task.deleted_reason = "remote_delete_failed"
                        add_event(db, task.id, "remote_delete_failed", detail, "error")
                    db.commit()
        except Exception:
            pass
        return True
    finally:
        if heartbeat_stop is not None:
            heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=5)


@router.post("/api/compute/nodes/{node_id}/agent/operations/{operation_id}/result")
def operation_result(node_id: str, operation_id: str, payload: OperationResult, request: Request, db: Session = Depends(get_db)):
    operation = authorized_operation(db, node_id, operation_id, request, payload.lease)
    if operation.status in {"succeeded", "failed", "cancelled"}:
        return {"status": operation.status}
    task_scope = operation.kind == "delete" and (operation.payload or {}).get("scope") == "task"
    if task_scope and (operation.payload or {}).get("remote_verified"):
        return {"status": operation.status}
    if operation.kind in {"fetch", "fetch_model", "fetch_log"} and payload.success:
        raise HTTPException(409, "文件传输必须通过分块接口完成")
    task = db.get(TrainingTask, operation.task_id)
    if operation.kind == "batch_history" and payload.success:
        run = db.get(TrainingRun, (operation.payload or {}).get("run_id"))
        configured = (operation.payload or {}).get("batch")
        history = payload.batch_history
        if (not task or task.deleted_at or not run or run.task_id != task.id
                or (run.execution_node_id or task.execution_node_id) != node_id
                or run.log_path != (operation.payload or {}).get("log_path")
                or history is None or (history and (len(history) < 2 or history[0] != configured))
                or any(type(value) is not int or not 1 <= value <= 1024 for value in history)
                or any(a <= b for a, b in zip(history, history[1:]))):
            raise HTTPException(422, "历史 Batch 扫描结果无效")
        known = list(run.batch_history or [])
        if history[:len(known)] == known and len(history) > len(known):
            run.batch_history = list(history)
        # A running log can still grow after this one-time scan. A terminal
        # Run needs no repeat scan, even when no reduction was found.
        run.batch_history_scanned = run.status in {"completed", "failed", "stopped", "interrupted"}
        if task.current_run_id == run.id:
            task.batch_history = list(run.batch_history or [])
    if task_scope:
        if (
            task is None or task.execution_node_id != node_id or task.status in ACTIVE
            or task.deleted_reason != "remote_delete_pending"
        ):
            raise HTTPException(409, "任务当前不能永久删除")
        if not payload.success:
            _mark_task_delete_failed(
                db,
                operation,
                task,
                "远程文件清理失败：" + (payload.error or "未知错误"),
            )
            db.commit()
            return {"status": operation.status}
        try:
            receipt = _verified_delete_receipt(operation, payload.receipt)
        except ValueError as exc:
            _mark_task_delete_failed(db, operation, task, f"远程删除回执验证失败：{exc}")
            db.commit()
            return {"status": operation.status}
        verified_payload = dict(operation.payload or {})
        verified_payload.update({
            "phase": "center_cleanup",
            "remote_verified": True,
            "receipt": receipt,
            "remote_verified_at": utcnow().isoformat(),
        })
        operation.payload = verified_payload
        operation.status = "finalizing"
        operation.error = None
        operation.updated_at = utcnow()
        db.commit()

        factory = sessionmaker(bind=db.get_bind(), expire_on_commit=False)
        process_remote_delete_center(factory)
        db.expire_all()
        current = db.get(RemoteOperation, operation_id)
        return {"status": current.status if current is not None else "succeeded"}

    operation.status = "succeeded" if payload.success else "failed"
    operation.error = payload.error if not payload.success else None
    operation.updated_at = utcnow()
    if not payload.success and operation.kind in {"fetch", "fetch_log"}:
        operation_file(operation, False).unlink(missing_ok=True)
    if not payload.success and operation.kind == "fetch_model":
        _mark_model_fetch_failed(db, operation, payload.error or "远程模型传输失败")
        db.commit()
        return {"status": operation.status}
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
        if payload.success:
            try:
                validate_attempt_deletion(
                    db, task, target_run, ignore_operation_id=operation.id,
                    checkpoint_release_records=(operation.payload or {}).get("checkpoint_releases"),
                )
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
                    delete_attempt_records(
                        db, task.id, target_run.id,
                        checkpoint_release_records=(operation.payload or {}).get("checkpoint_releases"),
                    )
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
        # Whole-task deletion uses the receipt-gated branch above. A legacy
        # instruction must be retried so it can be rebuilt under the new plan.
        _mark_task_delete_failed(db, operation, task, "旧版整任务删除指令缺少验证计划，请重试永久删除")
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
    size: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None
    mtime_ns: int | None = Field(default=None, ge=0)


@router.post("/api/compute/nodes/{node_id}/agent/operations/{operation_id}/chunk")
def upload_chunk(node_id: str, operation_id: str, payload: ArtifactChunk, request: Request, db: Session = Depends(get_db)):
    operation = authorized_operation(db, node_id, operation_id, request, payload.lease)
    return _upload_chunk(operation, node_id, payload, db)


def _upload_chunk(operation: RemoteOperation, node_id: str, payload: ArtifactChunk, db: Session):
    if operation.kind == "fetch_log" and "size" not in (operation.payload or {}):
        operation_id = operation.id
        # All nodes reserve against the same cache. A DB lock, rather than a
        # process-only mutex, serializes reservations across API workers too.
        acquire_node_claim_lock(db, "local")
        db.expire_all()
        operation = db.get(RemoteOperation, operation_id)
        if operation is None or operation.lease != payload.lease or operation.node_id != node_id:
            raise HTTPException(409, "远程日志传输租约已失效")
    if operation.kind not in {"fetch", "fetch_model", "fetch_log"} or operation.status not in {
        "running", "finalizing", "succeeded",
    }:
        raise HTTPException(409, "远程操作当前不能传输文件")
    if operation.status == "succeeded":
        return {"status": "succeeded", "offset": operation.payload["size"]}
    if operation.status == "finalizing" and not payload.final:
        raise HTTPException(409, "远程操作正在完成文件校验")
    model_asset: ModelAsset | None = None
    if operation.kind == "fetch_log":
        validate_log_fetch(db, operation)
    elif operation.kind == "fetch":
        task = db.get(TrainingTask, operation.task_id)
        if not task or task.deleted_at:
            raise HTTPException(409, "任务已删除，停止传输文件")
    else:
        asset_id = operation_model_asset_id(operation)
        model_asset = db.get(ModelAsset, asset_id) if asset_id else None
        latest = matching_model_fetch(db, model_asset) if model_asset is not None else None
        if (
            model_asset is None
            or model_asset.source_type != "remote_path"
            or model_asset.execution_node_id != node_id
            or model_asset.path != (operation.payload or {}).get("path")
            or latest is None
            or latest.id != operation.id
            or model_asset.status != "preparing"
        ):
            raise HTTPException(409, "模型来源已变更，停止传输文件")

    if operation.status == "finalizing":
        # The status is a cross-worker claim. A duplicate request with the same
        # lease waits while its owner heartbeats; after a crashed worker misses
        # the normal lease window, this request may return it to running and
        # claim finalization again below.
        if as_utc(operation.updated_at) >= utcnow() - OPERATION_LEASE_TIMEOUT:
            return {"status": "finalizing", "offset": (operation.payload or {}).get("size", 0)}
        resumed = db.execute(update(RemoteOperation).where(
            RemoteOperation.id == operation.id,
            RemoteOperation.node_id == node_id,
            RemoteOperation.status == "finalizing",
            RemoteOperation.lease == payload.lease,
            RemoteOperation.updated_at == operation.updated_at,
        ).values(status="running", updated_at=utcnow()).execution_options(synchronize_session=False))
        if resumed.rowcount != 1:
            db.rollback()
            current = db.get(RemoteOperation, operation.id)
            if current is not None and current.lease == payload.lease:
                if current.status == "succeeded":
                    return {"status": "succeeded", "offset": current.payload["size"]}
                if current.status == "finalizing":
                    return {"status": "finalizing", "offset": (current.payload or {}).get("size", 0)}
            raise HTTPException(409, "远程操作已重新分配")
        db.commit()
        db.refresh(operation)
    try:
        data = base64.b64decode(payload.data, validate=True)
    except ValueError as exc:
        raise HTTPException(422, "文件分块编码错误") from exc
    if len(data) > MAX_FILE_CHUNK_BYTES:
        raise HTTPException(413, "文件分块不能超过 512 KiB")

    if operation.kind in {"fetch_model", "fetch_log"}:
        if not data and operation.kind == "fetch_model":
            raise HTTPException(422, "远程模型不能为空")
        if payload.size is None or payload.modified_at is None or payload.mtime_ns is None:
            raise HTTPException(422, "远程文件分块缺少稳定文件元数据")
        if payload.modified_at.tzinfo is None or payload.modified_at.utcoffset() is None:
            raise HTTPException(422, "远程模型修改时间必须包含时区")
        metadata = {
            "size": payload.size,
            "modified_at": as_utc(payload.modified_at).isoformat(),
            "mtime_ns": payload.mtime_ns,
        }
        registered = operation.payload or {}
        first_metadata = "size" not in registered
        if "size" not in registered:
            operation.payload = {**registered, **metadata}
        elif any(registered.get(key) != value for key, value in metadata.items()):
            raise HTTPException(409, "远程模型在传输期间发生变化，请重试")
    else:
        first_metadata = False

    expected_size = (operation.payload or {}).get("size")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        raise HTTPException(409, "远程文件缺少有效大小")
    if operation.kind == "fetch_model" and not 0 < expected_size <= MAX_REMOTE_MODEL_BYTES:
        raise HTTPException(413, "远程模型大小必须在 1 字节到 8 GiB 之间")
    if operation.kind == "fetch_log":
        maximum = int(operation.payload.get("max_bytes") or 0)
        if maximum and expected_size > maximum:
            raise HTTPException(413, "日志超过下载大小限制")
    if payload.offset + len(data) > expected_size:
        raise HTTPException(413, "传输文件超出登记大小")
    if operation.kind in {"fetch_model", "fetch_log"} and (payload.offset + len(data) == expected_size) != payload.final:
        raise HTTPException(409, "远程模型分块结束标记与文件大小不一致")
    if first_metadata:
        # Commit the full-size reservation before accepting bytes. The process
        # lock makes concurrent first chunks observe every earlier admission;
        # database state then keeps the reservation durable across restarts.
        with (nullcontext() if operation.kind == "fetch_log" else _MODEL_CACHE_CAPACITY_LOCK):
            db.flush()
            try:
                if operation.kind == "fetch_log":
                    reserve_log_cache(db)
                else:
                    ensure_remote_model_cache_capacity(db)
            except HTTPException as exc:
                operation.status, operation.error = "failed", str(exc.detail)
                db.commit()
                raise
            except RemoteModelCacheError as exc:
                _mark_model_fetch_failed(db, operation, str(exc))
                db.commit()
                raise HTTPException(exc.status_code, str(exc)) from exc
            db.commit()

    if payload.final:
        claimed = db.execute(update(RemoteOperation).where(
            RemoteOperation.id == operation.id,
            RemoteOperation.node_id == node_id,
            RemoteOperation.status == "running",
            RemoteOperation.lease == payload.lease,
        ).values(status="finalizing", updated_at=utcnow()).execution_options(synchronize_session=False))
        if claimed.rowcount != 1:
            db.rollback()
            current = db.get(RemoteOperation, operation.id)
            if current is not None and current.lease == payload.lease:
                if current.status == "succeeded":
                    return {"status": "succeeded", "offset": current.payload["size"]}
                if current.status == "finalizing":
                    return {"status": "finalizing", "offset": (current.payload or {}).get("size", 0)}
            raise HTTPException(409, "远程操作已重新分配")
        db.commit()
        db.refresh(operation)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = operation_file(operation, False)
    complete_path = operation_file(operation)
    if not payload.final:
        try:
            written_size = write_cache_chunk(path, offset=payload.offset, data=data)
        except RemoteModelCacheError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        operation.updated_at = utcnow()
        db.commit()
        return {"status": operation.status, "offset": written_size}

    operation_id = operation.id

    def heartbeat_finalization() -> None:
        heartbeat = db.execute(update(RemoteOperation).where(
            RemoteOperation.id == operation_id,
            RemoteOperation.node_id == node_id,
            RemoteOperation.status == "finalizing",
            RemoteOperation.lease == payload.lease,
        ).values(updated_at=utcnow()).execution_options(synchronize_session=False))
        if heartbeat.rowcount != 1:
            db.rollback()
            raise RemoteModelCacheError(409, "远程操作已重新分配")
        db.commit()

    def release_finalization() -> None:
        """Make a failed final request retryable without disturbing a new lease."""
        db.rollback()
        try:
            db.execute(update(RemoteOperation).where(
                RemoteOperation.id == operation_id,
                RemoteOperation.node_id == node_id,
                RemoteOperation.status == "finalizing",
                RemoteOperation.lease == payload.lease,
            ).values(status="running", updated_at=utcnow()).execution_options(synchronize_session=False))
            db.commit()
        except Exception:
            db.rollback()

    try:
        part_exists = path.exists() or path.is_symlink()
        complete_exists = complete_path.exists() or complete_path.is_symlink()
        recovering_published_file = not part_exists and complete_exists
        if recovering_published_file:
            # A process can publish immediately before its database commit is
            # interrupted. Verify those exact bytes instead of retransmitting.
            written_size = expected_size
        else:
            try:
                written_size = write_cache_chunk(path, offset=payload.offset, data=data)
            except RemoteModelCacheError as exc:
                raise HTTPException(exc.status_code, str(exc)) from exc
        if written_size != expected_size or not payload.sha256:
            raise HTTPException(409, "传输文件不完整")

        digest_path = complete_path if recovering_published_file else path
        try:
            digest, cache_error = cached_fetch_digest(
                operation,
                path=digest_path,
                heartbeat=heartbeat_finalization,
            )
        except RemoteModelCacheError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        if cache_error or digest is None or not secrets.compare_digest(digest, payload.sha256):
            raise HTTPException(409, "文件校验失败，请重新传输")

        # A test or another worker may have replaced the lease immediately
        # after the digest callback returned. Renew once more before publish.
        try:
            heartbeat_finalization()
        except RemoteModelCacheError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        if not recovering_published_file:
            try:
                published_identity = _cache_identity(path.lstat())
                os.replace(path, complete_path)
                published = complete_path.lstat()
            except OSError as exc:
                raise HTTPException(409, "远程文件缓存无法安全发布") from exc
            if not stat.S_ISREG(published.st_mode) or _cache_identity(published) != published_identity:
                complete_path.unlink(missing_ok=True)
                raise HTTPException(409, "远程文件缓存在发布期间发生变化")

        db.rollback()
        current = db.get(RemoteOperation, operation_id)
        if (
            current is None
            or current.node_id != node_id
            or current.status != "finalizing"
            or not current.lease
            or not secrets.compare_digest(current.lease, payload.lease)
        ):
            if current is not None and current.status == "succeeded" and current.lease == payload.lease:
                return {"status": "succeeded", "offset": current.payload["size"]}
            raise HTTPException(409, "远程操作已重新分配")

        model_asset = None
        if current.kind == "fetch_log":
            validate_log_fetch(db, current)
        if current.kind == "fetch_model":
            asset_id = operation_model_asset_id(current)
            model_asset = db.get(ModelAsset, asset_id) if asset_id else None
            latest = matching_model_fetch(db, model_asset) if model_asset is not None else None
            if (
                model_asset is None
                or model_asset.source_type != "remote_path"
                or model_asset.execution_node_id != node_id
                or model_asset.path != (current.payload or {}).get("path")
                or latest is None
                or latest.id != current.id
                or model_asset.status != "preparing"
            ):
                raise HTTPException(409, "模型来源已变更，停止传输文件")

        final_payload = {**(current.payload or {}), "sha256": payload.sha256}
        finalized = db.execute(update(RemoteOperation).where(
            RemoteOperation.id == current.id,
            RemoteOperation.node_id == node_id,
            RemoteOperation.status == "finalizing",
            RemoteOperation.lease == payload.lease,
        ).values(
            status="succeeded", error=None, payload=final_payload, updated_at=utcnow(),
        ).execution_options(synchronize_session=False))
        if finalized.rowcount != 1:
            db.rollback()
            raise HTTPException(409, "远程操作已重新分配")
        if model_asset is not None:
            asset_updated = db.execute(update(ModelAsset).where(
                ModelAsset.id == model_asset.id,
                ModelAsset.source_type == "remote_path",
                ModelAsset.execution_node_id == node_id,
                ModelAsset.path == (current.payload or {}).get("path"),
                ModelAsset.status == "preparing",
            ).values(
                sha256=payload.sha256,
                size=expected_size,
                status="ready",
                error=None,
                updated_at=utcnow(),
            ).execution_options(synchronize_session=False))
            if asset_updated.rowcount != 1:
                db.rollback()
                raise HTTPException(409, "模型来源已变更，停止传输文件")
        db.commit()
        return {"status": "succeeded", "offset": expected_size}
    except Exception:
        release_finalization()
        raise
