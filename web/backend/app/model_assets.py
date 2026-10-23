"""Safe, persistent model sources for the standalone model-test centre."""
from __future__ import annotations

import hashlib
import os
import posixpath
import stat
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .config import MODEL_TEST_ROOTS, PROJECT_ROOT
from .compute_nodes import remote_model_test_unavailable_reason
from .database import get_db
from .dataset_uploads import managed_dataset_mutation
from .deletion_jobs import reject_local_task_cleanup
from .gpu_reservations import acquire_node_claim_lock
from .models import ComputeNode, ModelAsset, ModelTest, RemoteOperation, utcnow


router = APIRouter(prefix="/api/model-assets", tags=["model-assets"])
LOCAL_NODE_ID = "local"
LOCAL_SOURCE_TYPE = "local_path"
REMOTE_SOURCE_TYPE = "remote_path"


class LocalModelAssetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=255)

    @field_validator("path")
    @classmethod
    def clean_path(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("模型路径不合法")
        return value

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("模型名称不能为空")
        return value


class RemoteModelAssetCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_node_id: str = Field(
        min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$",
    )
    path: str = Field(min_length=1, max_length=4096)
    name: str | None = Field(default=None, max_length=255)

    @field_validator("path")
    @classmethod
    def clean_path(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value or any(ord(character) < 32 for character in value):
            raise ValueError("模型路径不合法")
        return value

    @field_validator("name")
    @classmethod
    def clean_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("模型名称不能为空")
        return value

class ModelAssetPathError(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(detail)


def _identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def digest_stable_regular_file(path: Path) -> tuple[str, int]:
    """Hash one immutable view of a regular file without following a final symlink."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.stat(path, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ModelAssetPathError(422, "模型路径必须指向普通文件，不能使用符号链接")
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or _identity(opened) != _identity(before):
                raise ModelAssetPathError(409, "模型文件在读取前发生变化，请重试")
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
            after_fd = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_path = os.stat(path, follow_symlinks=False)
    except ModelAssetPathError:
        raise
    except FileNotFoundError as exc:
        raise ModelAssetPathError(404, "模型文件不存在") from exc
    except OSError as exc:
        raise ModelAssetPathError(409, "模型文件无法安全读取，请检查权限后重试") from exc
    if (
        size != after_fd.st_size
        or _identity(before) != _identity(after_fd)
        or _identity(after_fd) != _identity(after_path)
    ):
        raise ModelAssetPathError(409, "模型文件在读取期间发生变化，请重试")
    return digest.hexdigest(), size


def copy_stable_regular_file(source: Path, destination: Path) -> tuple[str, int]:
    """Copy and hash a regular source while rejecting swaps and final symlinks."""
    source_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    source_fd: int | None = None
    destination_fd: int | None = None
    try:
        before = os.stat(source, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise ModelAssetPathError(422, "模型路径必须指向普通文件，不能使用符号链接")
        source_fd = os.open(source, source_flags)
        destination_fd = os.open(destination, destination_flags, 0o600)
        opened = os.fstat(source_fd)
        if _identity(opened) != _identity(before):
            raise ModelAssetPathError(409, "模型文件在复制前发生变化，请重试")
        digest = hashlib.sha256()
        size = 0
        with (
            os.fdopen(source_fd, "rb", closefd=False) as reader,
            os.fdopen(destination_fd, "wb", closefd=False) as writer,
        ):
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                digest.update(block)
                size += len(block)
            writer.flush()
            os.fsync(destination_fd)
        after_fd = os.fstat(source_fd)
        after_path = os.stat(source, follow_symlinks=False)
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
    if (
        size != after_fd.st_size
        or _identity(before) != _identity(after_fd)
        or _identity(after_fd) != _identity(after_path)
    ):
        destination.unlink(missing_ok=True)
        raise ModelAssetPathError(409, "模型文件在复制期间发生变化，请重试")
    return digest.hexdigest(), size


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root.resolve() or root.resolve() in path.parents for root in roots)


def resolve_local_model_path(value: str) -> Path:
    """Resolve a user supplied local path inside the configured test roots."""
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    if candidate.is_symlink():
        raise ModelAssetPathError(422, "模型文件不能使用符号链接")
    try:
        path = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ModelAssetPathError(404, "模型文件不存在") from exc
    except OSError as exc:
        raise ModelAssetPathError(409, "模型路径无法读取") from exc
    if path.suffix.lower() != ".pt":
        raise ModelAssetPathError(422, "模型测试仅支持 .pt 权重")
    if not _within(path, MODEL_TEST_ROOTS):
        raise ModelAssetPathError(403, "模型路径不在平台配置的测试模型目录中")
    try:
        mode = os.stat(path, follow_symlinks=False).st_mode
    except OSError as exc:
        raise ModelAssetPathError(404, "模型文件不存在") from exc
    if not stat.S_ISREG(mode) or path.is_symlink():
        raise ModelAssetPathError(422, "模型路径必须指向普通文件，不能使用符号链接")
    return path


def portable_local_model_path(path: Path) -> str:
    """Store checkout-owned paths relatively so a cloned project remains portable."""
    try:
        return path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def local_asset_file(asset: ModelAsset) -> Path:
    if asset.source_type != LOCAL_SOURCE_TYPE or asset.execution_node_id != LOCAL_NODE_ID:
        raise ModelAssetPathError(422, "该模型来源不是本机路径")
    return resolve_local_model_path(asset.path)


def verify_local_asset(
    asset: ModelAsset,
    *,
    expected_sha256: str | None = None,
    expected_size: int | None = None,
) -> tuple[Path, str, int]:
    path = local_asset_file(asset)
    sha256, size = digest_stable_regular_file(path)
    if (
        (expected_sha256 is not None and sha256 != expected_sha256)
        or (expected_size is not None and size != expected_size)
    ):
        raise ModelAssetPathError(409, "原模型权重版本已变化，无法按原权重重测")
    asset.sha256 = sha256
    asset.size = size
    asset.status = "ready"
    asset.error = None
    asset.updated_at = utcnow()
    return path, sha256, size


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return value.isoformat()


def asset_removal_metadata(db: Session, asset_ids: list[str]) -> dict[str, dict]:
    """Batch history counts and authoritative busy reasons without N+1 queries."""
    from .inference import ACTIVE_STATUSES
    from .remote_operations import ACTIVE_MODEL_FETCH_STATUSES, model_asset_operation_scope

    metadata = {asset_id: {"history_count": 0, "removal_blocked_reason": None} for asset_id in asset_ids}
    if not asset_ids:
        return metadata
    for asset_id, status, count in db.execute(select(
        ModelTest.model_asset_id, ModelTest.status, func.count(),
    ).where(ModelTest.model_asset_id.in_(asset_ids)).group_by(ModelTest.model_asset_id, ModelTest.status)):
        metadata[asset_id]["history_count"] += count
        if status in ACTIVE_STATUSES:
            metadata[asset_id]["removal_blocked_reason"] = "该模型有尚未结束的测试，请完成或取消测试后再移除登记"
    scope_assets = {model_asset_operation_scope(asset_id): asset_id for asset_id in asset_ids}
    for scope in db.scalars(select(RemoteOperation.task_id).where(
        RemoteOperation.task_id.in_(scope_assets),
        RemoteOperation.kind == "fetch_model",
        RemoteOperation.status.in_(ACTIVE_MODEL_FETCH_STATUSES),
    )):
        metadata[scope_assets[scope]]["removal_blocked_reason"] = "模型正在准备或传输，请等待完成后再移除登记"
    return metadata


def asset_public(asset: ModelAsset, *, node_name: str | None = None, removal: dict | None = None) -> dict:
    return {
        "id": asset.id,
        "name": asset.name,
        "source_type": asset.source_type,
        "execution_node_id": asset.execution_node_id,
        "execution_node_name": node_name,
        "path": asset.path,
        "artifact_id": asset.artifact_id,
        "status": asset.status,
        "error": asset.error,
        "size": asset.size,
        "sha256": asset.sha256,
        "created_at": _iso(asset.created_at),
        "updated_at": _iso(asset.updated_at),
        **(removal or {}),
    }


def model_test_root_labels() -> list[str]:
    labels = []
    for root in MODEL_TEST_ROOTS:
        try:
            labels.append(root.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix())
        except ValueError:
            labels.append(str(root))
    return labels


def validate_remote_model_path(node: ComputeNode, value: str) -> str:
    """Validate one remote path lexically against Agent-reported roots.

    The center must never resolve a remote path with its own filesystem.  The
    Agent repeats this check against the real remote filesystem, including all
    symlink and stable-file checks, before it opens the checkpoint.
    """
    if "\\" in value or not PurePosixPath(value).is_absolute():
        raise ModelAssetPathError(422, "远程模型必须使用服务器上的绝对 POSIX 路径")
    normalized = posixpath.normpath(value)
    if normalized != value or "//" in value:
        raise ModelAssetPathError(422, "远程模型路径必须是已规范化的绝对路径")
    path = PurePosixPath(value)
    if path.suffix != ".pt":
        raise ModelAssetPathError(422, "模型测试仅支持小写 .pt 权重")
    capability = (node.snapshot or {}).get("training")
    roots = capability.get("model_test_roots") if isinstance(capability, dict) else None
    if not isinstance(roots, list) or not roots:
        raise ModelAssetPathError(409, "远程 Agent 未上报 model_test_roots，请升级并重启 Agent")
    allowed = []
    for root_value in roots:
        if not isinstance(root_value, str):
            continue
        root_normalized = posixpath.normpath(root_value)
        root = PurePosixPath(root_normalized)
        if root.is_absolute() and root_normalized == root_value:
            allowed.append(root)
    if not any(root in path.parents for root in allowed):
        raise ModelAssetPathError(403, "远程模型路径不在该 Agent 上报的 model_test_roots 内")
    return value


@router.get("")
def list_model_assets(
    source_type: str | None = Query(default=None, max_length=32),
    db: Session = Depends(get_db),
):
    query = select(ModelAsset).where(ModelAsset.deleted_at.is_(None))
    if source_type:
        query = query.where(ModelAsset.source_type == source_type)
    assets = list(db.scalars(query.order_by(ModelAsset.updated_at.desc(), ModelAsset.id.desc())).all())
    node_ids = {asset.execution_node_id for asset in assets}
    nodes = {
        node.id: node.name for node in db.scalars(
            select(ComputeNode).where(ComputeNode.id.in_(node_ids)),
        ).all()
    } if node_ids else {}
    roots = model_test_root_labels()
    removal = asset_removal_metadata(db, [asset.id for asset in assets])
    return {
        "items": [asset_public(asset, node_name=nodes.get(asset.execution_node_id), removal=removal[asset.id]) for asset in assets],
        "local_root": roots[0],
        "local_roots": roots,
    }


@router.post("/local", status_code=201)
def register_local_model_asset(payload: LocalModelAssetCreate, db: Session = Depends(get_db)):
    # Keep reference registration serialized with the full local deletion
    # transaction, including the filesystem verification performed by workers.
    with managed_dataset_mutation():
        reject_local_task_cleanup(db)
        return _register_local_model_asset(payload, db)


def _register_local_model_asset(payload: LocalModelAssetCreate, db: Session):
    try:
        path = resolve_local_model_path(payload.path)
        sha256, size = digest_stable_regular_file(path)
    except ModelAssetPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    stored_path = portable_local_model_path(path)
    # Serialize restoring a registration with removal, without holding a write
    # lock during potentially expensive checkpoint hashing.
    acquire_node_claim_lock(db, LOCAL_NODE_ID)
    asset = db.scalar(select(ModelAsset).where(
        ModelAsset.source_type == LOCAL_SOURCE_TYPE,
        ModelAsset.execution_node_id == LOCAL_NODE_ID,
        ModelAsset.path == stored_path,
    ))
    if asset is None:
        asset = ModelAsset(
            id=uuid.uuid4().hex,
            name=payload.name or path.name,
            source_type=LOCAL_SOURCE_TYPE,
            execution_node_id=LOCAL_NODE_ID,
            path=stored_path,
        )
        db.add(asset)
    elif payload.name is not None:
        asset.name = payload.name
    asset.sha256 = sha256
    asset.size = size
    asset.status = "ready"
    asset.error = None
    asset.deleted_at = None
    asset.updated_at = utcnow()
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        asset = db.scalar(select(ModelAsset).where(
            ModelAsset.source_type == LOCAL_SOURCE_TYPE,
            ModelAsset.execution_node_id == LOCAL_NODE_ID,
            ModelAsset.path == stored_path,
        ))
        if asset is None:
            raise
        if payload.name is not None:
            asset.name = payload.name
        asset.sha256 = sha256
        asset.size = size
        asset.status = "ready"
        asset.error = None
        asset.deleted_at = None
        asset.updated_at = utcnow()
        db.commit()
    db.refresh(asset)
    node = db.get(ComputeNode, LOCAL_NODE_ID)
    return asset_public(asset, node_name=node.name if node else "本机服务器", removal=asset_removal_metadata(db, [asset.id])[asset.id])


@router.post("/remote", status_code=201)
def register_remote_model_asset(payload: RemoteModelAssetCreate, db: Session = Depends(get_db)):
    # Serialize source registration with node disable/removal and operation
    # polling. Otherwise a concurrent request could enqueue work immediately
    # after node management had concluded that no transfer was active.
    acquire_node_claim_lock(db, payload.execution_node_id)
    from .remote_operations import active_remote_delete
    if active_remote_delete(db, payload.execution_node_id):
        raise HTTPException(409, "该远程服务器正在清理训练文件，请等待完成后再登记模型")
    node = db.get(ComputeNode, payload.execution_node_id)
    unavailable = remote_model_test_unavailable_reason(node)
    if unavailable:
        raise HTTPException(409, unavailable)
    assert node is not None
    try:
        path = validate_remote_model_path(node, payload.path)
    except ModelAssetPathError as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc

    asset = db.scalar(select(ModelAsset).where(
        ModelAsset.source_type == REMOTE_SOURCE_TYPE,
        ModelAsset.execution_node_id == node.id,
        ModelAsset.path == path,
    ))
    force_refresh = False
    if asset is None:
        asset = ModelAsset(
            id=uuid.uuid4().hex,
            name=payload.name or PurePosixPath(path).name,
            source_type=REMOTE_SOURCE_TYPE,
            execution_node_id=node.id,
            path=path,
            status="preparing",
        )
        db.add(asset)
    else:
        # Registering the same remote path means "read its current bytes".
        # Preserve older successful operation caches for already-created tests,
        # while preparing a fresh immutable version for new tests.
        force_refresh = asset.deleted_at is not None or asset.status in {"ready", "failed"}
        asset.deleted_at = None
        if payload.name is not None:
            asset.name = payload.name
        asset.updated_at = utcnow()
    try:
        db.flush()
        # Local import keeps the model-source module independent from the
        # operation router during application startup.
        from .remote_operations import prepare_remote_model_asset
        prepare_remote_model_asset(db, asset, retry=force_refresh, refresh=force_refresh)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "该远程模型已被其他请求登记，请刷新后重试") from None
    db.refresh(asset)
    return asset_public(asset, node_name=node.name, removal=asset_removal_metadata(db, [asset.id])[asset.id])


@router.post("/{asset_id}/prepare")
def prepare_model_asset(
    asset_id: str,
    retry: bool = False,
    db: Session = Depends(get_db),
):
    asset = db.get(ModelAsset, asset_id)
    if asset is None:
        raise HTTPException(404, "模型来源不存在")
    if asset.deleted_at is not None:
        raise HTTPException(409, "该模型登记已移除，请重新登记后再准备模型")
    if asset.source_type != REMOTE_SOURCE_TYPE:
        node = db.get(ComputeNode, asset.execution_node_id)
        return asset_public(asset, node_name=node.name if node else None, removal=asset_removal_metadata(db, [asset.id])[asset.id])
    acquire_node_claim_lock(db, asset.execution_node_id)
    asset = db.get(ModelAsset, asset_id, populate_existing=True)
    if asset is None:
        raise HTTPException(404, "模型来源不存在")
    if asset.deleted_at is not None:
        raise HTTPException(409, "该模型登记已移除，请重新登记后再准备模型")
    from .remote_operations import prepare_remote_model_asset
    try:
        prepare_remote_model_asset(db, asset, retry=retry)
    except HTTPException:
        # Preserve the actionable failure on the asset while returning the
        # protocol status (for example, an Agent going offline before retry).
        db.commit()
        raise
    db.commit()
    db.refresh(asset)
    node = db.get(ComputeNode, asset.execution_node_id)
    return asset_public(asset, node_name=node.name if node else None, removal=asset_removal_metadata(db, [asset.id])[asset.id])


@router.delete("/{asset_id}", status_code=204)
def remove_model_asset(asset_id: str, db: Session = Depends(get_db)):
    """Hide a registration, keeping source bytes and immutable test histories."""
    from .inference import model_test_lifecycle

    # Use the same lock order as test launch/task mutations: lifecycle first,
    # then database. The node lock also serializes Agent polling and restores.
    with model_test_lifecycle():
        asset = db.get(ModelAsset, asset_id)
        if asset is None:
            raise HTTPException(404, "模型来源不存在")
        acquire_node_claim_lock(db, asset.execution_node_id)
        asset = db.get(ModelAsset, asset_id, populate_existing=True)
        if asset is None:
            raise HTTPException(404, "模型来源不存在")
        if asset.deleted_at is not None:
            return
        reason = asset_removal_metadata(db, [asset.id])[asset.id]["removal_blocked_reason"]
        if reason:
            raise HTTPException(409, reason)
        asset.deleted_at = utcnow()
        asset.updated_at = asset.deleted_at
        db.commit()
