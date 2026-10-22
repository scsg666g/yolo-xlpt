"""Persistent image tests, serialized on CPU and isolated from training outputs."""
from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.orm import Session, defer, selectinload, sessionmaker
from starlette.requests import ClientDisconnect

from .config import ULTRALYTICS_ROOT, YOLO_BIN
from .database import SessionLocal, get_db
from .model_assets import (
    LOCAL_NODE_ID,
    ModelAssetPathError,
    copy_stable_regular_file,
    digest_stable_regular_file,
    verify_local_asset,
)
from .model_test_storage import (
    HISTORY_ROOT,
    WORK_ROOT,
    ModelTestCapacityError,
    delete_test_history,
    delete_test_work,
    directory_size,
    estimate_test_storage,
    history_scope,
    prepare_test_history,
    reconcile_test_reservations,
    release_test_reservation,
    test_history_root,
    test_work_root,
)
from .models import (
    Artifact,
    ComputeNode,
    MetricPoint,
    ModelAsset,
    ModelTest,
    ModelTestImage,
    RemoteOperation,
    TrainingRun,
    TrainingTask,
    utcnow,
)
from .remote_operations import cached_fetch_digest, matching_fetch, operation_file
from .run_deletions import attempt_delete_operation
from .services import failure_detail, path_is_within, training_environment

router = APIRouter(tags=["model-tests"])
logger = logging.getLogger("yolo-platform.inference")
MAX_IMAGE_BYTES = 20 * 1024**2
MAX_TOTAL_BYTES = 100 * 1024**2
UPLOAD_TTL = 24 * 60 * 60
ACTIVE_STATUSES = {"uploading", "preparing", "queued", "running", "cancel_requested"}


class ImageSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=MAX_IMAGE_BYTES)


class TestCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_id: int | None = Field(default=None, gt=0)
    model_asset_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    run_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    rerun_of_test_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    confidence: float = Field(default=0.25, ge=0.01, le=1)
    iou: float = Field(default=0.7, ge=0.01, le=1)
    files: list[ImageSpec] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def one_model_source(self):
        if (self.artifact_id is None) == (self.model_asset_id is None):
            raise ValueError("artifact_id 与 model_asset_id 必须且只能提供一个")
        return self


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _as_utc(value).isoformat() if value else None


def model_summary(db: Session, artifact: Artifact, run: TrainingRun) -> dict:
    best_epoch = None
    best_metric = None
    points = db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id)).all()
    for point in points:
        value = point.values.get("metrics/mAP50-95(B)")
        if isinstance(value, (int, float)) and math.isfinite(value) and (best_metric is None or value > best_metric):
            best_epoch, best_metric = point.epoch, float(value)
    task = db.get(TrainingTask, artifact.task_id)
    return {
        "weight_name": Path(artifact.name).name,
        "weight_size": artifact.size,
        "modified_at": artifact.modified_at.isoformat() if artifact.modified_at else None,
        "attempt_number": run.attempt_number,
        "best_epoch": best_epoch,
        "metric_name": "mAP50-95",
        "metric_value": best_metric,
        "device": "CPU",
        "imgsz": 640,
        "source_type": "training_artifact",
        "source_name": Path(artifact.name).name,
        "source_path": artifact.path,
        "execution_node_id": run.execution_node_id or LOCAL_NODE_ID,
        "task_name": task.name if task else None,
    }


def asset_model_summary(db: Session, asset: ModelAsset, *, sha256: str, size: int) -> dict:
    node = db.get(ComputeNode, asset.execution_node_id)
    return {
        "weight_name": asset.name,
        "weight_size": size,
        "sha256": sha256,
        "modified_at": None,
        "attempt_number": None,
        "best_epoch": None,
        "metric_name": None,
        "metric_value": None,
        "device": "CPU",
        "imgsz": 640,
        "source_type": asset.source_type,
        "source_name": asset.name,
        "source_path": asset.path,
        "execution_node_id": asset.execution_node_id,
        "execution_node_name": node.name if node else None,
        "task_name": None,
    }


def _weight_context(db: Session, artifact_id: int | None) -> tuple[Artifact, TrainingTask, TrainingRun]:
    artifact = db.get(Artifact, artifact_id) if artifact_id is not None else None
    task = db.get(TrainingTask, artifact.task_id) if artifact else None
    run = db.get(TrainingRun, artifact.run_id) if artifact and artifact.run_id else None
    if not artifact or not task or task.deleted_at or not run or run.task_id != task.id:
        raise HTTPException(404, "模型权重所属的训练运行不存在")
    if task.task_type != "detect" or artifact.kind != "weight" or Path(artifact.name).suffix.lower() != ".pt":
        raise HTTPException(422, "图片测试目前支持目标检测任务的 .pt 权重")
    if run.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "请等待本次训练结束或停止后再测试模型")
    return artifact, task, run


def _local_weight_path(artifact: Artifact, run: TrainingRun) -> Path:
    path = Path(artifact.path).resolve()
    if not run.run_dir or not path_is_within(path, [Path(run.run_dir)]):
        raise HTTPException(403, "模型权重路径不受信任")
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "模型权重文件不存在，请刷新后重试")
    return path


def _remote_weight_for_version(
    db: Session,
    artifact: Artifact,
    run: TrainingRun,
    sha256: str,
    size: int,
) -> Path:
    operations = db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == artifact.task_id,
        RemoteOperation.kind == "fetch",
        RemoteOperation.status == "succeeded",
    ).order_by(RemoteOperation.created_at.desc()))
    for operation in operations:
        payload = operation.payload or {}
        if (
            payload.get("artifact_id") == artifact.id
            and payload.get("run_id") == run.id
            and payload.get("size") == size
        ):
            path = operation_file(operation)
            if not path.is_file() or path.is_symlink():
                continue
            cached_sha256 = payload.get("sha256")
            if cached_sha256 is not None and cached_sha256 != sha256:
                continue
            actual_sha256, cache_error = cached_fetch_digest(operation, path=path)
            if cache_error is not None or actual_sha256 != sha256:
                continue
            if cached_sha256 is None:
                operation.payload = {**payload, "sha256": actual_sha256}
            return path
    raise HTTPException(409, "原远程权重的精确版本已不在中心缓存，无法按原权重重测")


def _weight_source(
    db: Session,
    artifact_id: int | None,
    *,
    sha256: str | None = None,
    size: int | None = None,
) -> tuple[Path, Artifact, TrainingRun, str | None]:
    artifact, _, run = _weight_context(db, artifact_id)
    if run.execution_node_id:
        if sha256 is not None and size is not None:
            return _remote_weight_for_version(db, artifact, run, sha256, size), artifact, run, sha256
        operation = matching_fetch(db, artifact)
        if not operation or operation.status != "succeeded":
            raise HTTPException(409, "远程权重尚未准备完成，请先准备模型")
        path = operation_file(operation)
        advertised_sha256 = (operation.payload or {}).get("sha256")
        if not isinstance(advertised_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", advertised_sha256):
            advertised_sha256 = None
    else:
        return _local_weight_path(artifact, run), artifact, run, None
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "模型权重文件不存在，请刷新后重试")
    return path, artifact, run, advertised_sha256


def weight_path(db: Session, artifact_id: int | None) -> Path:
    """Return the currently registered source path after ownership checks."""
    return _weight_source(db, artifact_id)[0]


def _file_identity(stat_result: os.stat_result) -> tuple[int, int, int, int]:
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_size, stat_result.st_mtime_ns)


def _digest_stable_file(path: Path) -> tuple[str, int]:
    try:
        return digest_stable_regular_file(path)
    except ModelAssetPathError as exc:
        raise HTTPException(409, "固定的模型权重版本已不可用，请重新选择当前权重") from exc


def _verified_weight_version(
    db: Session,
    artifact_id: int | None,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[Path, Artifact, TrainingRun, str, int]:
    path, artifact, run, advertised_sha256 = _weight_source(
        db, artifact_id, sha256=expected_sha256, size=expected_size,
    )
    actual_sha256, actual_size = _digest_stable_file(path)
    if advertised_sha256 and actual_sha256 != advertised_sha256:
        raise HTTPException(409, "远程权重缓存校验失败，请重新准备模型")
    if run.execution_node_id and expected_size is None and actual_size != artifact.size:
        raise HTTPException(409, "远程权重缓存大小与产物登记不一致，请重新准备模型")
    if (
        (expected_sha256 is not None and actual_sha256 != expected_sha256)
        or (expected_size is not None and actual_size != expected_size)
    ):
        raise HTTPException(409, "原模型权重版本已变化，无法按原权重重测")
    if run.execution_node_id and advertised_sha256 is None and expected_sha256 is None:
        # Fetch records created by older center builds have no digest. Once the
        # center has independently verified the complete cache file, persist it
        # so later historical reruns can identify these exact bytes directly.
        operation = matching_fetch(db, artifact)
        if operation and operation.status == "succeeded":
            operation.payload = {**(operation.payload or {}), "sha256": actual_sha256}
    return path, artifact, run, actual_sha256, actual_size


def _verified_model_asset_version(
    db: Session,
    model_asset_id: str | None,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[Path, ModelAsset, str, int]:
    asset = db.get(ModelAsset, model_asset_id) if model_asset_id else None
    if asset is None:
        raise HTTPException(404, "模型来源不存在，请重新选择模型")
    if asset.source_type == "remote_path":
        from .remote_operations import RemoteModelCacheError, verified_remote_model_cache
        try:
            path, sha256, size = verified_remote_model_cache(
                db,
                asset,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
        except RemoteModelCacheError as exc:
            # Historical tests may target an older successful cache while a
            # newer version is being prepared. Do not overwrite that current
            # reusable-asset state from a historical validation path. Likewise,
            # a new test that arrives during transfer must leave the active
            # preparation intact so the Agent can continue uploading chunks.
            current_version = (
                expected_sha256 is None and expected_size is None
            ) or (
                asset.status == "ready"
                and asset.sha256 == expected_sha256
                and asset.size == expected_size
            )
            if current_version and asset.status == "ready":
                # Compare-and-set prevents a stale request from replacing a
                # concurrent refresh's preparing/ready state with failed.
                db.execute(update(ModelAsset).where(
                    ModelAsset.id == asset.id,
                    ModelAsset.status == "ready",
                    ModelAsset.sha256 == asset.sha256,
                    ModelAsset.size == asset.size,
                ).values(
                    status="failed",
                    error=str(exc)[-1000:],
                    updated_at=utcnow(),
                ))
                db.commit()
            raise HTTPException(exc.status_code, str(exc)) from exc
    elif asset.source_type == "local_path":
        try:
            path, sha256, size = verify_local_asset(
                asset, expected_sha256=expected_sha256, expected_size=expected_size,
            )
        except ModelAssetPathError as exc:
            # A historical rerun can ask for v1 after this reusable source has
            # already been refreshed to v2. Its expected failure must not mark
            # the valid current v2 asset as failed for every future test.
            current_version = (
                expected_sha256 is None and expected_size is None
            ) or (
                asset.status == "ready"
                and asset.sha256 == expected_sha256
                and asset.size == expected_size
            )
            if current_version:
                asset.status = "failed"
                asset.error = str(exc)[-1000:]
                asset.updated_at = utcnow()
                db.commit()
            raise HTTPException(exc.status_code, str(exc)) from exc
    else:
        raise HTTPException(422, "当前模型来源类型不支持独立测试")
    return path, asset, sha256, size


def _copy_and_digest(source: Path, destination: Path) -> tuple[str, int]:
    try:
        return copy_stable_regular_file(source, destination)
    except ModelAssetPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc


def _pin_weight(source: Path, test_id: str, sha256: str, size: int) -> Path:
    work = test_work_root(test_id)
    delete_test_work(test_id)
    work.mkdir(parents=True, exist_ok=False)
    temporary = work / "model.pt.part"
    destination = work / "model.pt"
    try:
        before = source.stat()
        actual_sha256, actual_size = _copy_and_digest(source, temporary)
        after = source.stat()
        if _file_identity(before) != _file_identity(after):
            raise HTTPException(409, "模型权重在复制期间发生变化，请刷新产物后重试")
        if actual_sha256 != sha256 or actual_size != size:
            raise HTTPException(409, "原模型权重版本已变化，无法启动检测")
        os.replace(temporary, destination)
        return destination
    except HTTPException:
        delete_test_work(test_id)
        raise
    except OSError as exc:
        try:
            delete_test_work(test_id)
        except (OSError, ValueError):
            logger.warning("model_test_pin_cleanup_failed test_id=%s", test_id)
        raise HTTPException(507, "模型权重副本保存失败，请检查服务器磁盘空间") from exc


def _pin_validation_error(test_id: str, sha256: str | None, size: int | None) -> str | None:
    if not sha256 or size is None:
        return "检测记录缺少固定权重的版本信息"
    try:
        model = test_work_root(test_id) / "model.pt"
    except ValueError:
        return "固定的模型权重副本路径无效"
    if not model.is_file() or model.is_symlink():
        return "固定的模型权重副本缺失"
    try:
        actual_sha256, actual_size = _digest_stable_file(model)
    except HTTPException:
        return "固定的模型权重副本无法读取"
    if actual_sha256 != sha256 or actual_size != size:
        return "固定的模型权重副本校验失败"
    return None


def normalize_image(source: Path, destination: Path) -> None:
    try:
        with Image.open(source) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "BMP"}:
                raise ValueError("仅支持 JPG、PNG、WEBP 和 BMP 图片")
            if image.width * image.height > 20_000_000:
                raise ValueError("单张图片不能超过 2000 万像素")
            image.load()
            ImageOps.exif_transpose(image).convert("RGB").save(destination, "JPEG", quality=95)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("图片损坏或无法读取，请重新选择") from exc


def _record_query(test_id: str):
    return (
        select(ModelTest)
        .where(ModelTest.id == test_id)
        .options(selectinload(ModelTest.images))
        .execution_options(populate_existing=True)
    )


def get_record(db: Session, test_id: str, *, task_id: str | None = None) -> ModelTest:
    record = db.scalar(_record_query(test_id))
    if not record or (task_id is not None and record.task_id != task_id):
        raise HTTPException(404, "模型检测记录不存在")
    if record.task_id is not None:
        task = db.scalar(select(TrainingTask).where(
            TrainingTask.id == record.task_id,
        ).execution_options(populate_existing=True))
        if not task or task.deleted_at:
            raise HTTPException(404, "模型检测记录不存在")
    return record


def image_public(image: ModelTestImage) -> dict:
    return {
        "name": image.original_name,
        "size": image.declared_size,
        "uploaded": image.uploaded,
        "detections": image.detections,
        "objects": list(image.objects or []),
        "timings": dict(image.timings or {}),
    }


def record_public(record: ModelTest) -> dict:
    expires_at = _as_utc(record.created_at).timestamp() + UPLOAD_TTL if record.status == "uploading" else None
    snapshot = dict(record.model_snapshot or {})
    return {
        "id": record.id,
        "task_id": record.task_id,
        "run_id": record.run_id,
        "source_run_id": record.source_run_id,
        "artifact_id": record.artifact_id,
        "model_asset_id": record.model_asset_id,
        "artifact_name": record.artifact_name,
        "model_name": record.model_name,
        "model_source_type": record.model_source_type,
        "model_source_name": snapshot.get("source_name") or record.model_name,
        "model_source_path": record.model_source_path,
        "execution_node_id": record.execution_node_id,
        "execution_node_name": snapshot.get("execution_node_name"),
        "task_name": snapshot.get("task_name"),
        "model_sha256": record.model_sha256,
        "model_size": record.model_size,
        "rerun_of_test_id": record.rerun_of_test_id,
        "attempt_number": record.attempt_number,
        "source_attempt_number": record.source_attempt_number,
        "status": record.status,
        "error": record.error,
        "confidence": record.confidence,
        "iou": record.iou,
        "model": snapshot,
        "timings": dict(record.timings) if record.timings else None,
        "files": [image_public(image) for image in sorted(record.images, key=lambda item: item.ordinal)],
        "created_at": _iso(record.created_at),
        "started_at": _iso(record.started_at),
        "finished_at": _iso(record.finished_at),
        "storage_bytes": record.storage_bytes,
        "expires_at": expires_at,
    }


def record_summary(record: ModelTest) -> dict:
    expires_at = _as_utc(record.created_at).timestamp() + UPLOAD_TTL if record.status == "uploading" else None
    snapshot = dict(record.model_snapshot or {})
    return {
        "id": record.id,
        "task_id": record.task_id,
        "run_id": record.run_id,
        "source_run_id": record.source_run_id,
        "artifact_id": record.artifact_id,
        "model_asset_id": record.model_asset_id,
        "artifact_name": record.artifact_name,
        "model_name": record.model_name,
        "model_source_type": record.model_source_type,
        "model_source_name": snapshot.get("source_name") or record.model_name,
        "model_source_path": record.model_source_path,
        "execution_node_id": record.execution_node_id,
        "execution_node_name": snapshot.get("execution_node_name"),
        "task_name": snapshot.get("task_name"),
        "model_sha256": record.model_sha256,
        "model_size": record.model_size,
        "rerun_of_test_id": record.rerun_of_test_id,
        "attempt_number": record.attempt_number,
        "source_attempt_number": record.source_attempt_number,
        "status": record.status,
        "error": record.error,
        "confidence": record.confidence,
        "iou": record.iou,
        "model": snapshot,
        "files": [
            {
                "name": image.original_name,
                "size": image.declared_size,
                "uploaded": image.uploaded,
                "detections": image.detections,
            }
            for image in sorted(record.images, key=lambda item: item.ordinal)
        ],
        "created_at": _iso(record.created_at),
        "started_at": _iso(record.started_at),
        "finished_at": _iso(record.finished_at),
        "storage_bytes": record.storage_bytes,
        "expires_at": expires_at,
    }


def _encode_history_cursor(record: ModelTest) -> str:
    payload = json.dumps(
        {"created_at": _iso(record.created_at), "id": record.id}, separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_history_cursor(cursor: str) -> tuple[datetime, str]:
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded.decode("utf-8"))
        if not isinstance(payload, dict) or set(payload) != {"created_at", "id"}:
            raise ValueError
        created_at = datetime.fromisoformat(payload["created_at"])
        test_id = payload["id"]
        if not isinstance(test_id, str) or not re.fullmatch(r"[0-9a-f]{32}", test_id):
            raise ValueError
        return _as_utc(created_at), test_id
    except (binascii.Error, json.JSONDecodeError, TypeError, UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(422, "历史记录游标无效") from exc


def _error_message(exc: Exception) -> str:
    if isinstance(exc, HTTPException):
        return str(exc.detail)[-1000:]
    return (str(exc) or type(exc).__name__)[-1000:]


def _settle_test_storage(task_id: str | None, test_id: str) -> int:
    """Release active capacity and return the terminal record's real size."""
    scope = history_scope(task_id)
    try:
        release_test_reservation(scope, test_id)
    except (OSError, ValueError):
        logger.warning("model_test_reservation_release_failed test_id=%s", test_id)
    return directory_size(test_history_root(scope, test_id))


def _persist_settled_storage(db: Session, record: ModelTest) -> None:
    """Best-effort accounting after the terminal status is already durable."""
    test_id = record.id
    try:
        record.storage_bytes = _settle_test_storage(record.task_id, record.id)
        db.commit()
    except Exception:
        db.rollback()
        logger.warning("model_test_storage_accounting_failed test_id=%s", test_id, exc_info=True)


class _WorkerRestart(RuntimeError):
    pass


class _TestCancelled(RuntimeError):
    pass


class InferenceWorker:
    def __init__(self, session_factory=SessionLocal, interval: float = 1.0):
        self.session_factory = session_factory
        self.interval = interval
        self.lifecycle_lock = threading.RLock()
        self.lock = threading.RLock()
        self._state_changed = threading.Condition(self.lock)
        self.uploading: set[tuple[str, int]] = set()
        self._cancelled_ids: set[str] = set()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._current_id: str | None = None
        self._current_task_id: str | None = None

    def start(self) -> None:
        HISTORY_ROOT.mkdir(parents=True, exist_ok=True)
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        self._recover()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="image-inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self.lock:
            if self._process and self._process.poll() is None:
                self._process.kill()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def wait_for_test_exit(self, test_id: str, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._state_changed:
            while self._current_id == test_id:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state_changed.wait(remaining)
        return True

    def wait_for_uploads(self, test_ids: set[str], timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        with self._state_changed:
            while any(test_id in test_ids for test_id, _ in self.uploading):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state_changed.wait(remaining)
        return True

    def _recover(self) -> None:
        retained_pins: set[str] = set()
        transitioned: set[tuple[str, str]] = set()
        with self.session_factory() as db:
            records = db.scalars(select(ModelTest).where(
                ModelTest.status.in_(ACTIVE_STATUSES)
            )).all()
            for record in records:
                task = db.get(TrainingTask, record.task_id) if record.task_id is not None else None
                task_unavailable = record.task_id is not None and (not task or task.deleted_at)
                scope = history_scope(record.task_id)
                if task_unavailable or record.delete_requested or record.status == "cancel_requested":
                    record.status = "cancelled"
                    record.finished_at = record.finished_at or utcnow()
                    transitioned.add((scope, record.id))
                elif record.status == "uploading":
                    continue
                else:
                    pin_error = _pin_validation_error(record.id, record.model_sha256, record.model_size)
                    if pin_error:
                        record.status = "failed"
                        record.error = pin_error
                        record.finished_at = record.finished_at or utcnow()
                        transitioned.add((scope, record.id))
                    else:
                        retained_pins.add(record.id)
                        record.status = "queued"
                        record.error = None
                        record.started_at = None
                        record.finished_at = None
            db.commit()
            active_tests = {
                (history_scope(task_id), test_id)
                for task_id, test_id in db.execute(select(ModelTest.task_id, ModelTest.id).where(
                    ModelTest.status.in_(ACTIVE_STATUSES),
                ))
            }
            try:
                reconcile_test_reservations(active_tests)
            except (OSError, ValueError):
                logger.warning("model_test_reservation_recovery_failed", exc_info=True)
            for scope, test_id in transitioned:
                record = db.get(ModelTest, test_id)
                if record:
                    record.storage_bytes = directory_size(test_history_root(scope, test_id))
            if transitioned:
                db.commit()
        if WORK_ROOT.is_dir() and not WORK_ROOT.is_symlink():
            for path in WORK_ROOT.iterdir():
                if re.fullmatch(r"[0-9a-f]{32}", path.name) and path.name not in retained_pins:
                    try:
                        delete_test_work(path.name)
                    except (OSError, ValueError):
                        logger.warning("model_test_work_recovery_failed test_id=%s", path.name)

    def _expire_uploads(self) -> None:
        cutoff = utcnow() - timedelta(seconds=UPLOAD_TTL)
        with self.lock:
            with self.session_factory() as db:
                records = db.scalars(select(ModelTest).where(
                    ModelTest.status == "uploading", ModelTest.created_at < cutoff,
                ).options(selectinload(ModelTest.images))).all()
                for record in records:
                    if any(test_id == record.id for test_id, _ in self.uploading):
                        continue
                    try:
                        delete_test_history(history_scope(record.task_id), record.id)
                        delete_test_work(record.id)
                    except (OSError, ValueError):
                        logger.warning("model_test_expiry_cleanup_failed test_id=%s", record.id)
                        continue
                    db.delete(record)
                db.commit()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._expire_uploads()
                if not self.process_one():
                    continue
            except Exception:
                logger.exception("image_inference_worker_error")

    def process_one(self) -> bool:
        with self.lifecycle_lock:
            with self.session_factory() as db:
                test_id = db.scalar(select(ModelTest.id).outerjoin(
                    TrainingTask, TrainingTask.id == ModelTest.task_id,
                ).where(
                    ModelTest.status == "queued",
                    ModelTest.delete_requested.is_(False),
                    or_(
                        ModelTest.task_id.is_(None),
                        and_(TrainingTask.id.is_not(None), TrainingTask.deleted_at.is_(None)),
                    ),
                ).order_by(ModelTest.created_at, ModelTest.id).limit(1))
                if not test_id:
                    return False
                claimed = db.execute(update(ModelTest).where(
                    ModelTest.id == test_id, ModelTest.status == "queued", ModelTest.delete_requested.is_(False),
                ).values(status="running", started_at=utcnow(), finished_at=None, error=None))
                if claimed.rowcount != 1:
                    db.rollback()
                    return False
                db.commit()
        self.execute(test_id)
        return True

    def execute(self, test_id: str) -> None:
        directory: Path | None = None
        work: Path | None = None
        output: Path | None = None
        retain_pin = False
        with self._state_changed:
            if test_id in self._cancelled_ids:
                return
            self._current_id = test_id
            self._current_task_id = None
        try:
            with self.session_factory() as db:
                record = db.scalar(_record_query(test_id))
                if not record:
                    return
                if record.status == "queued":
                    claimed = db.execute(update(ModelTest).where(
                        ModelTest.id == test_id, ModelTest.status == "queued", ModelTest.delete_requested.is_(False),
                    ).values(status="running", started_at=utcnow(), finished_at=None, error=None))
                    if claimed.rowcount != 1:
                        db.rollback()
                        return
                    db.commit()
                    record = db.scalar(_record_query(test_id))
                if not record or record.status != "running" or record.delete_requested:
                    return
                directory = test_history_root(history_scope(record.task_id), record.id)
                work = test_work_root(record.id)
                output = directory / "results"
                pin_error = _pin_validation_error(record.id, record.model_sha256, record.model_size)
                if pin_error:
                    raise RuntimeError(pin_error)
                model = work / "model.pt"
                confidence, iou = record.confidence, record.iou
                task_id = record.task_id

            with self._state_changed:
                if test_id in self._cancelled_ids:
                    raise _TestCancelled()
                self._current_task_id = task_id
            if output.exists():
                if output.is_symlink() or not output.is_dir():
                    raise RuntimeError("检测结果目录不安全")
                shutil.rmtree(output)
            for stale in (directory / "summary.json", directory / "detections.json", directory / "detection-results.zip"):
                stale.unlink(missing_ok=True)

            log = directory / "inference.log"
            summary_path = directory / "summary.json"
            python_bin = YOLO_BIN.with_name("python")
            command = [
                str(python_bin), str(Path(__file__).with_name("inference_runner.py")),
                "--model", str(work / "model.pt"), "--source", str(directory / "images"),
                "--output", str(output), "--summary", str(summary_path),
                "--ultralytics-root", str(ULTRALYTICS_ROOT), "--confidence", str(confidence),
                "--iou", str(iou), "--imgsz", "640",
            ]
            environment = {
                **training_environment(), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
            }
            with log.open("w", encoding="utf-8") as handle:
                with self.lock:
                    with self.session_factory() as db:
                        current = db.get(ModelTest, test_id)
                        if self._stop.is_set():
                            raise _WorkerRestart()
                        if (
                            test_id in self._cancelled_ids
                            or not current
                            or current.status != "running"
                            or current.delete_requested
                        ):
                            raise _TestCancelled()
                    self._process = subprocess.Popen(
                        command, cwd=ULTRALYTICS_ROOT, env=environment,
                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                    )
                    process = self._process
                try:
                    code = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("推理超过 5 分钟，请减少图片数量或尺寸后重试") from None
            if self._stop.is_set():
                raise _WorkerRestart()
            with self.lock:
                if test_id in self._cancelled_ids:
                    raise _TestCancelled()
            if code:
                raise RuntimeError(failure_detail(str(log)) or "模型推理失败，请检查权重是否兼容当前训练环境")

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            image_results = summary.get("images", [])
            with self.lock:
                with self.session_factory() as db:
                    record = db.scalar(_record_query(test_id))
                    if not record or record.delete_requested or record.status in {"cancel_requested", "cancelled"}:
                        raise _TestCancelled()
                    images = sorted(record.images, key=lambda item: item.ordinal)
                    if len(image_results) != len(images):
                        raise RuntimeError("推理没有生成完整结果")
                    for index, (image, item) in enumerate(zip(images, image_results)):
                        if not (output / f"{index}.jpg").is_file():
                            raise RuntimeError("推理没有生成完整结果，请确认模型为目标检测模型")
                        objects = item.get("objects", [])
                        image.detections = len(objects)
                        image.objects = objects
                        image.timings = item.get("timings", {})
                    record.timings = summary.get("timings", {})
                    record.device = str(summary.get("device", "CPU"))[:32]
                    record.imgsz = int(summary.get("imgsz", 640))
                    files = [image_public(image) for image in images]
                    export = {
                        "model": record.model_snapshot,
                        "model_version": {"sha256": record.model_sha256, "size": record.model_size},
                        "rerun_of_test_id": record.rerun_of_test_id,
                        "confidence": record.confidence, "iou": record.iou,
                        "device": record.device, "imgsz": record.imgsz,
                        "timings": record.timings, "files": files,
                    }
                    (directory / "detections.json").write_text(
                        json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8",
                    )
                    with zipfile.ZipFile(directory / "detection-results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                        archive.write(directory / "detections.json", "detections.json")
                        for index, image in enumerate(images):
                            original_name = re.split(r"[/\\]", image.original_name)[-1]
                            safe_name = re.sub(r"[^\w.-]+", "_", original_name).strip("._") or "image"
                            archive.write(output / f"{index}.jpg", f"results/{index + 1}-{safe_name}.jpg")
                    record.status = "completed"
                    record.error = None
                    record.finished_at = utcnow()
                    db.commit()
                    _persist_settled_storage(db, record)
        except _WorkerRestart:
            retain_pin = True
            with self.session_factory() as db:
                record = db.get(ModelTest, test_id)
                if record and record.status == "running" and not record.delete_requested:
                    record.status = "queued"
                    record.error = None
                    record.started_at = None
                    db.commit()
        except _TestCancelled:
            if output and output.exists() and output.is_dir() and not output.is_symlink():
                shutil.rmtree(output, ignore_errors=True)
            with self.session_factory() as db:
                record = db.get(ModelTest, test_id)
                if record and record.status not in ACTIVE_STATUSES:
                    _persist_settled_storage(db, record)
        except Exception as exc:
            with self.lock:
                cancelled_in_process = test_id in self._cancelled_ids
            if cancelled_in_process:
                return
            logger.exception("image_inference_failed test_id=%s", test_id)
            with self.session_factory() as db:
                record = db.get(ModelTest, test_id)
                if record and not record.delete_requested and record.status not in {"cancel_requested", "cancelled"}:
                    record.status = "failed"
                    record.error = _error_message(exc)
                    record.finished_at = utcnow()
                    directory = directory or test_history_root(history_scope(record.task_id), record.id)
                    db.commit()
                    _persist_settled_storage(db, record)
        finally:
            if not retain_pin:
                try:
                    delete_test_work(test_id)
                except (OSError, ValueError):
                    logger.warning("model_test_work_cleanup_failed test_id=%s", test_id)
            with self._state_changed:
                self._process = None
                self._current_id = None
                self._current_task_id = None
                self._cancelled_ids.discard(test_id)
                self._state_changed.notify_all()

    def cancel_task_tests(self, task_id: str, *, purge: bool = False, session_factory=None) -> None:
        factory = session_factory or self.session_factory
        current_id: str | None = None
        cancelled_ids: tuple[str, ...] = ()
        all_test_ids: set[str] = set()
        terminal_records: tuple[tuple[str, str], ...] = ()
        try:
            # Keep the worker from publishing a result between the durable status
            # transition and the in-memory cancellation signal.
            with self._state_changed:
                with factory() as db:
                    records = db.scalars(select(ModelTest).where(ModelTest.task_id == task_id)).all()
                    all_test_ids = {record.id for record in records}
                    active_records = [record for record in records if record.status in ACTIVE_STATUSES]
                    terminal_records = tuple((record.task_id, record.id) for record in records)
                    current_belongs_to_task = (
                        self._current_task_id == task_id or self._current_id in all_test_ids
                    )
                    current_id = self._current_id if current_belongs_to_task else None
                    cancelled_ids = tuple(dict.fromkeys([
                        *(record.id for record in active_records),
                        *([current_id] if current_id else []),
                    ]))
                    for record in active_records:
                        record.status = "cancelled"
                        record.error = record.error or "训练任务状态已变更，检测已取消"
                        record.finished_at = record.finished_at or utcnow()
                        if purge:
                            record.delete_requested = True
                    db.commit()
                self._cancelled_ids.update(cancelled_ids)
                if current_id and self._process and self._process.poll() is None:
                    self._process.kill()

            if current_id and not self.wait_for_test_exit(current_id):
                logger.warning("model_test_cancel_timeout task_id=%s test_id=%s", task_id, current_id)
                raise HTTPException(409, "正在停止模型检测，请稍后重试")
            if all_test_ids and not self.wait_for_uploads(all_test_ids):
                logger.warning("model_test_upload_cancel_timeout task_id=%s", task_id)
                raise HTTPException(409, "正在停止图片上传，请稍后重试")

            if terminal_records:
                try:
                    with factory() as cleanup_db:
                        for record_task_id, record_id in terminal_records:
                            record = cleanup_db.get(ModelTest, record_id)
                            if record and record.status not in ACTIVE_STATUSES:
                                record.storage_bytes = _settle_test_storage(record_task_id, record_id)
                        cleanup_db.commit()
                except Exception:
                    logger.warning("model_test_cancel_storage_finalize_failed task_id=%s", task_id, exc_info=True)
            for test_id in all_test_ids:
                try:
                    delete_test_work(test_id)
                except (OSError, ValueError):
                    logger.warning("model_test_cancel_work_cleanup_failed test_id=%s", test_id)
        finally:
            with self._state_changed:
                self._cancelled_ids.difference_update(
                    test_id for test_id in cancelled_ids if test_id != self._current_id
                )


worker = InferenceWorker()


@contextmanager
def model_test_lifecycle():
    """Serialize task visibility changes with model-test creation and launch."""
    with worker.lifecycle_lock:
        yield


def stage_task_test_cancellation(task_id: str, db: Session, *, purge: bool = False) -> bool:
    """Stage only database state; the caller may still safely roll back or close."""
    records = db.scalars(select(ModelTest).where(
        ModelTest.task_id == task_id,
        ModelTest.status.in_(ACTIVE_STATUSES),
    )).all()
    for record in records:
        record.status = "cancel_requested"
        record.error = record.error or "训练任务状态已变更，检测已取消"
        if purge:
            record.delete_requested = True
    return bool(records)


def cancel_task_tests(task_id: str, *, purge: bool = False, bind=None) -> None:
    factory = sessionmaker(bind=bind, expire_on_commit=False) if bind is not None else None
    worker.cancel_task_tests(task_id, purge=purge, session_factory=factory)


def abandon_upload_record(test_id: str, db: Session, reason: str) -> ModelTest:
    with worker.lock:
        record = get_record(db, test_id)
        if record.status == "cancelled":
            return record
        if record.status != "uploading":
            raise HTTPException(409, "检测已提交，关闭页面不会取消检测")
        record.status = "cancelled"
        record.error = reason
        record.finished_at = utcnow()
        db.commit()
        _persist_settled_storage(db, record)
        return get_record(db, test_id)


@router.post("/api/model-tests", status_code=201)
def create_test(payload: TestCreate, db: Session = Depends(get_db)):
    with model_test_lifecycle():
        return _create_test(payload, db)


def _create_test(payload: TestCreate, db: Session):
    declared_upload_bytes = sum(item.size for item in payload.files)
    if declared_upload_bytes > MAX_TOTAL_BYTES:
        raise HTTPException(413, "一次测试的图片总大小不能超过 100 MB")
    active = db.scalar(select(func.count()).select_from(ModelTest).where(ModelTest.status.in_(ACTIVE_STATUSES))) or 0
    if active >= 20:
        raise HTTPException(429, "同时处理的测试记录已达上限，请稍后重试")
    rerun_source = get_record(db, payload.rerun_of_test_id) if payload.rerun_of_test_id else None
    expected_sha256 = None
    expected_size = None
    if rerun_source:
        if rerun_source.status != "completed":
            raise HTTPException(409, "只能使用已完成的历史记录重测")
        if not rerun_source.model_sha256 or rerun_source.model_size is None:
            raise HTTPException(409, "该历史记录没有精确权重版本，请改用当前权重新建检测")
        expected_sha256 = rerun_source.model_sha256
        expected_size = rerun_source.model_size

    task: TrainingTask | None = None
    context_run: TrainingRun | None = None
    source_run: TrainingRun | None = None
    artifact: Artifact | None = None
    asset: ModelAsset | None = None
    if payload.artifact_id is not None:
        artifact = db.get(Artifact, payload.artifact_id)
        source_run = db.get(TrainingRun, artifact.run_id) if artifact and artifact.run_id else None
        task = db.scalar(select(TrainingTask).where(
            TrainingTask.id == artifact.task_id,
        ).execution_options(populate_existing=True)) if artifact else None
        if not artifact or not source_run or not task or task.deleted_at:
            raise HTTPException(404, "模型权重所属的训练运行不存在")
        context_run = db.get(TrainingRun, payload.run_id) if payload.run_id else source_run
        if not context_run or context_run.task_id != task.id:
            raise HTTPException(422, "当前 Attempt 与模型权重不属于同一训练任务")
        if (
            attempt_delete_operation(db, source_run, active_only=True)
            or attempt_delete_operation(db, context_run, active_only=True)
        ):
            raise HTTPException(409, "当前 Attempt 已进入清理队列，不能新建模型检测")
        if rerun_source:
            if rerun_source.task_id != task.id:
                raise HTTPException(422, "历史检测记录与当前训练任务不一致")
            if rerun_source.artifact_id is None or rerun_source.artifact_id != artifact.id:
                raise HTTPException(409, "历史记录的原权重关联已失效，无法按原权重重测")
        model, artifact, source_run, model_sha256, model_size = _verified_weight_version(
            db, artifact.id, expected_sha256=expected_sha256, expected_size=expected_size,
        )
        snapshot = (
            dict(rerun_source.model_snapshot or {})
            if rerun_source else model_summary(db, artifact, source_run)
        )
        artifact_name = rerun_source.artifact_name if rerun_source else artifact.name
        model_name = rerun_source.model_name if rerun_source else Path(artifact.name).name
        source_run_id = rerun_source.source_run_id if rerun_source else source_run.id
        source_attempt_number = (
            rerun_source.source_attempt_number if rerun_source else source_run.attempt_number
        )
        model_source_type = "training_artifact"
        execution_node_id = source_run.execution_node_id or LOCAL_NODE_ID
        model_source_path = rerun_source.model_source_path if rerun_source else artifact.path
    else:
        if payload.run_id is not None:
            raise HTTPException(422, "独立模型测试不能指定训练 Attempt")
        asset = db.get(ModelAsset, payload.model_asset_id)
        if asset is None:
            raise HTTPException(404, "模型来源不存在，请重新选择模型")
        if rerun_source:
            if rerun_source.task_id is not None:
                raise HTTPException(422, "历史检测记录不是独立模型测试")
            if rerun_source.model_asset_id is None or rerun_source.model_asset_id != asset.id:
                raise HTTPException(409, "历史记录的原模型来源已失效，无法按原权重重测")
        model, asset, model_sha256, model_size = _verified_model_asset_version(
            db, asset.id, expected_sha256=expected_sha256, expected_size=expected_size,
        )
        snapshot = (
            dict(rerun_source.model_snapshot or {})
            if rerun_source else asset_model_summary(db, asset, sha256=model_sha256, size=model_size)
        )
        artifact_name = rerun_source.artifact_name if rerun_source else asset.path
        model_name = rerun_source.model_name if rerun_source else asset.name
        source_run_id = None
        source_attempt_number = None
        model_source_type = asset.source_type
        execution_node_id = asset.execution_node_id
        model_source_path = rerun_source.model_source_path if rerun_source else asset.path
    snapshot.update({"weight_size": model_size, "sha256": model_sha256})

    test_id = uuid.uuid4().hex
    estimate = estimate_test_storage(
        model_bytes=model_size,
        declared_upload_bytes=declared_upload_bytes,
        image_count=len(payload.files),
    )
    scope = history_scope(task.id if task else None)
    try:
        directory = prepare_test_history(scope, test_id, estimate)
    except ModelTestCapacityError as exc:
        raise HTTPException(507, str(exc)) from exc
    try:
        record = ModelTest(
            id=test_id, task_id=task.id if task else None,
            run_id=context_run.id if context_run else None, source_run_id=source_run_id,
            artifact_id=artifact.id if artifact else None,
            model_asset_id=asset.id if asset else None,
            model_source_type=model_source_type, execution_node_id=execution_node_id,
            model_source_path=model_source_path,
            artifact_name=artifact_name, model_name=model_name,
            model_sha256=model_sha256, model_size=model_size,
            rerun_of_test_id=rerun_source.id if rerun_source else None,
            attempt_number=context_run.attempt_number if context_run else None,
            source_attempt_number=source_attempt_number,
            confidence=payload.confidence, iou=payload.iou,
            model_snapshot=snapshot, status="uploading",
        )
        record.images = [
            ModelTestImage(ordinal=index, original_name=item.name, declared_size=item.size)
            for index, item in enumerate(payload.files)
        ]
        db.add(record)
        db.commit()
        return record_public(get_record(db, test_id))
    except Exception:
        db.rollback()
        try:
            delete_test_history(scope, test_id)
        except (OSError, ValueError):
            logger.warning("model_test_create_cleanup_failed test_id=%s", test_id)
        raise


@router.put("/api/model-tests/{test_id}/images/{index}", status_code=204)
async def upload_image(test_id: str, index: int, request: Request, db: Session = Depends(get_db)):
    key = (test_id, index)
    with worker.lock:
        record = get_record(db, test_id)
        task = db.get(TrainingTask, record.task_id) if record.task_id is not None else None
        images = sorted(record.images, key=lambda item: item.ordinal)
        if record.task_id is not None and (not task or task.deleted_at):
            raise HTTPException(409, "训练任务已进入回收站")
        if record.status != "uploading" or key in worker.uploading:
            raise HTTPException(409, "当前测试不能上传图片")
        if not 0 <= index < len(images):
            raise HTTPException(404, "图片不存在")
        image = images[index]
        image.uploaded = False
        image.stored_size = 0
        db.commit()
        worker.uploading.add(key)
        declared_size = image.declared_size
        directory = test_history_root(history_scope(record.task_id), record.id)
        destination = directory / "images" / f"{index}.jpg"
    temporary = directory / f".{index}.upload.part"
    normalized = directory / f".{index}.normalized.part"
    try:
        size = 0
        with temporary.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > declared_size:
                    raise HTTPException(413, "图片大小超过申报大小")
                handle.write(chunk)
        if size != declared_size:
            raise HTTPException(422, "图片上传不完整，请重试")
        await asyncio.to_thread(normalize_image, temporary, normalized)
        with worker.lock:
            record = get_record(db, test_id)
            image = next((item for item in record.images if item.ordinal == index), None)
            if not image or record.status != "uploading":
                raise HTTPException(409, "当前测试不能上传图片")
            # Publish the normalized image only while the record is still accepting
            # uploads. Abandoning the test and this replace therefore cannot race.
            os.replace(normalized, destination)
            image.uploaded = True
            image.stored_size = destination.stat().st_size
            record.storage_bytes = directory_size(directory)
            db.commit()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ClientDisconnect as exc:
        try:
            abandon_upload_record(test_id, db, "图片上传连接已中断，检测未启动")
        except Exception:
            logger.exception("model_test_disconnect_cancel_failed test_id=%s", test_id)
        raise HTTPException(400, "上传连接已中断，请重新测试") from exc
    except OSError as exc:
        raise HTTPException(507, "图片保存失败，请检查服务器磁盘空间") from exc
    finally:
        temporary.unlink(missing_ok=True)
        normalized.unlink(missing_ok=True)
        with worker._state_changed:
            worker.uploading.discard(key)
            worker._state_changed.notify_all()


@router.post("/api/model-tests/{test_id}/start", status_code=202)
def start_test(test_id: str, db: Session = Depends(get_db)):
    with model_test_lifecycle():
        return _start_test(test_id, db)


def _start_test(test_id: str, db: Session):
    with worker.lock:
        record = get_record(db, test_id)
        if record.task_id is not None:
            task = db.scalar(select(TrainingTask).where(
                TrainingTask.id == record.task_id,
            ).execution_options(populate_existing=True))
            if not task or task.deleted_at:
                raise HTTPException(409, "训练任务已进入回收站")
        if record.status != "uploading":
            return record_public(record)
        images = sorted(record.images, key=lambda item: item.ordinal)
        directory = test_history_root(history_scope(record.task_id), record.id)
        if any(key[0] == test_id for key in worker.uploading) or not all(
            image.uploaded and (directory / "images" / f"{image.ordinal}.jpg").is_file() for image in images
        ):
            raise HTTPException(409, "请先上传全部图片")
        if not record.model_sha256 or record.model_size is None:
            raise HTTPException(409, "检测记录没有固定权重版本，请重新创建检测")
        record.status = "preparing"
        record.error = None
        record.finished_at = None
        db.commit()
        try:
            if record.model_asset_id is not None:
                model, _, _, _ = _verified_model_asset_version(
                    db, record.model_asset_id,
                    expected_sha256=record.model_sha256,
                    expected_size=record.model_size,
                )
            else:
                model, _, _, _, _ = _verified_weight_version(
                    db, record.artifact_id,
                    expected_sha256=record.model_sha256,
                    expected_size=record.model_size,
                )
            _pin_weight(model, record.id, record.model_sha256, record.model_size)
        except Exception as exc:
            db.rollback()
            failed = db.get(ModelTest, test_id)
            if failed and failed.status == "preparing":
                failed.status = "failed"
                failed.error = _error_message(exc)
                failed.finished_at = utcnow()
                db.commit()
                _persist_settled_storage(db, failed)
            try:
                delete_test_work(test_id)
            except (OSError, ValueError):
                logger.warning("model_test_start_cleanup_failed test_id=%s", test_id)
            raise
        record = get_record(db, test_id)
        if record.status != "preparing" or record.delete_requested:
            delete_test_work(test_id)
            raise HTTPException(409, "当前测试不能启动")
        record.status = "queued"
        db.commit()
        return record_public(get_record(db, test_id))


@router.post("/api/model-tests/{test_id}/abandon")
def abandon_test(test_id: str, db: Session = Depends(get_db)):
    """Finish an upload-only record without deleting its completed images."""
    return record_public(abandon_upload_record(
        test_id, db, "图片上传未完成，检测未启动",
    ))


@router.get("/api/model-tests/{test_id}")
def test_status(test_id: str, db: Session = Depends(get_db)):
    return record_public(get_record(db, test_id))


@router.get("/api/model-tests")
def list_model_tests(
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=50, ge=1, le=100),
    source_type: str | None = Query(default=None, max_length=32, pattern=r"^[A-Za-z0-9_-]+$"),
    status: str | None = Query(default=None, max_length=24, pattern=r"^[A-Za-z0-9_-]+$"),
    execution_node_id: str | None = Query(default=None, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    db: Session = Depends(get_db),
):
    """List task-bound and standalone history with one stable cursor."""
    query = select(ModelTest).outerjoin(TrainingTask, TrainingTask.id == ModelTest.task_id).where(
        or_(
            ModelTest.task_id.is_(None),
            and_(TrainingTask.id.is_not(None), TrainingTask.deleted_at.is_(None)),
        ),
    )
    if source_type:
        query = query.where(ModelTest.model_source_type == source_type)
    if status:
        query = query.where(ModelTest.status == status)
    if execution_node_id:
        query = query.where(ModelTest.execution_node_id == execution_node_id)
    if cursor:
        cursor_created_at, cursor_id = _decode_history_cursor(cursor)
        query = query.where(
            (ModelTest.created_at < cursor_created_at)
            | ((ModelTest.created_at == cursor_created_at) & (ModelTest.id < cursor_id))
        )
    records = list(db.scalars(
        query.options(
            defer(ModelTest.timings),
            selectinload(ModelTest.images).load_only(
                ModelTestImage.ordinal,
                ModelTestImage.original_name,
                ModelTestImage.declared_size,
                ModelTestImage.uploaded,
                ModelTestImage.detections,
            ),
        )
        .order_by(ModelTest.created_at.desc(), ModelTest.id.desc())
        .limit(limit + 1)
        .execution_options(populate_existing=True)
    ).all())
    has_more = len(records) > limit
    records = records[:limit]
    return {
        "items": [record_summary(record) for record in records],
        "next_cursor": _encode_history_cursor(records[-1]) if has_more else None,
    }


@router.get("/api/tasks/{task_id}/model-tests")
def list_task_tests(
    task_id: str,
    cursor: str | None = Query(default=None, max_length=512),
    limit: int = Query(default=50, ge=1, le=100),
    run_id: str | None = Query(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
    db: Session = Depends(get_db),
):
    task = db.get(TrainingTask, task_id)
    if not task or task.deleted_at:
        raise HTTPException(404, "训练任务不存在")
    query = select(ModelTest).where(ModelTest.task_id == task_id)
    if run_id:
        query = query.where(ModelTest.run_id == run_id)
    if cursor:
        cursor_created_at, cursor_id = _decode_history_cursor(cursor)
        query = query.where(
            (ModelTest.created_at < cursor_created_at)
            | ((ModelTest.created_at == cursor_created_at) & (ModelTest.id < cursor_id))
        )
    records = list(db.scalars(
        query.options(
            defer(ModelTest.timings),
            selectinload(ModelTest.images).load_only(
                ModelTestImage.ordinal,
                ModelTestImage.original_name,
                ModelTestImage.declared_size,
                ModelTestImage.uploaded,
                ModelTestImage.detections,
            ),
        )
        .order_by(ModelTest.created_at.desc(), ModelTest.id.desc())
        .limit(limit + 1)
        .execution_options(populate_existing=True)
    ).all())
    has_more = len(records) > limit
    records = records[:limit]
    return {
        "items": [record_summary(record) for record in records],
        "next_cursor": _encode_history_cursor(records[-1]) if has_more else None,
    }


@router.get("/api/tasks/{task_id}/model-tests/{test_id}")
def task_test_detail(task_id: str, test_id: str, db: Session = Depends(get_db)):
    task = db.get(TrainingTask, task_id)
    if not task or task.deleted_at:
        raise HTTPException(404, "训练任务不存在")
    return record_public(get_record(db, test_id, task_id=task_id))


@router.get("/api/model-tests/{test_id}/images/{index}/{kind}")
def test_image(test_id: str, index: int, kind: str, download: bool = False, db: Session = Depends(get_db)):
    record = get_record(db, test_id)
    images = sorted(record.images, key=lambda item: item.ordinal)
    if not 0 <= index < len(images) or kind not in {"original", "result"}:
        raise HTTPException(404, "图片不存在")
    if kind == "result" and record.status != "completed":
        raise HTTPException(409, "检测结果尚未生成")
    path = test_history_root(history_scope(record.task_id), record.id) / ("images" if kind == "original" else "results") / f"{index}.jpg"
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "图片尚未上传")
    filename = f"detection-{index + 1}.jpg" if download else None
    return FileResponse(path, media_type="image/jpeg", filename=filename)


@router.get("/api/model-tests/{test_id}/download")
def download_test(test_id: str, db: Session = Depends(get_db)):
    record = get_record(db, test_id)
    if record.status != "completed":
        raise HTTPException(409, "检测结果尚未生成")
    path = test_history_root(history_scope(record.task_id), record.id) / "detection-results.zip"
    if not path.is_file() or path.is_symlink():
        raise HTTPException(404, "检测结果压缩包不存在")
    return FileResponse(path, media_type="application/zip", filename="detection-results.zip")


@router.delete("/api/model-tests/{test_id}", status_code=204)
def delete_test(test_id: str, db: Session = Depends(get_db)):
    with worker._state_changed:
        record = get_record(db, test_id)
        if any(key[0] == test_id for key in worker.uploading):
            raise HTTPException(409, "图片正在上传，请稍后删除")
        record.delete_requested = True
        record.status = "cancel_requested" if worker._current_id == test_id else "cancelled"
        record.finished_at = record.finished_at or utcnow()
        worker._cancelled_ids.add(test_id)
        db.commit()
        if worker._current_id == test_id and worker._process and worker._process.poll() is None:
            worker._process.kill()
    if not worker.wait_for_test_exit(test_id):
        raise HTTPException(409, "正在停止模型检测，请稍后重试删除")
    try:
        record = get_record(db, test_id)
        delete_test_work(test_id)
        delete_test_history(history_scope(record.task_id), test_id)
        db.delete(record)
        db.commit()
    except (OSError, ValueError) as exc:
        raise HTTPException(507, "模型检测文件清理失败，请稍后重试") from exc
    finally:
        with worker._state_changed:
            worker._cancelled_ids.discard(test_id)
