"""Directory uploads: stage files privately, validate, then publish a named dataset."""
from __future__ import annotations

import hashlib
import fcntl
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
import warnings as python_warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml
from fastapi import APIRouter, Depends, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from starlette.requests import ClientDisconnect

from .config import DATA_ROOT, DATASET_ROOTS
from .database import get_db
from .models import TrainingRun, TrainingTask

router = APIRouter(prefix="/api/datasets/uploads", tags=["dataset uploads"])
management_router = APIRouter(prefix="/api/datasets/managed", tags=["managed datasets"])
logger = logging.getLogger(__name__)
UPLOAD_ROOT = DATA_ROOT / "dataset_uploads"
MANAGED_MARKER = ".yolo-platform-dataset.json"
MANAGED_DELETION_DIR = ".platform-deletions"
MANAGED_PUBLICATION_PREFIX = ".platform-upload-"
MAX_FILES = 200_000
MAX_TOTAL_BYTES = 50 * 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_TEXT_BYTES = 4 * 1024**2
SESSION_TTL = 24 * 60 * 60
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
METADATA_FILES = {"data.yaml", "data.yml", "classes.txt"}
MANAGED_STAT_KEYS = {"train_images", "val_images", "labels", "classes", "background_images"}


class UploadFileSpec(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=MAX_FILE_BYTES)
    sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')


class UploadCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    files: list[UploadFileSpec] = Field(min_length=1, max_length=MAX_FILES)
    location_id: str | None = None


class UploadComplete(BaseModel):
    names: list[str] = Field(default_factory=list, max_length=1000)


def managed_marker(directory: Path) -> dict | None:
    marker = directory / MANAGED_MARKER
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("name") != directory.name:
        return None
    stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
    return {
        "version": 1,
        "name": directory.name,
        "upload_id": payload.get("upload_id") if isinstance(payload.get("upload_id"), str) else None,
        "created_at": payload.get("created_at") if isinstance(payload.get("created_at"), str) else None,
        "total_bytes": payload.get("total_bytes") if type(payload.get("total_bytes")) is int and payload["total_bytes"] >= 0 else 0,
        "stats": {key: value for key, value in stats.items()
                  if key in MANAGED_STAT_KEYS and type(value) is int and value >= 0},
        "classes": [value for value in payload.get("classes", [])
                    if isinstance(value, str)][:1000] if isinstance(payload.get("classes"), list) else [],
        "warnings": [value for value in payload.get("warnings", [])
                     if isinstance(value, str)][:100] if isinstance(payload.get("warnings"), list) else [],
    }


@contextmanager
def managed_dataset_mutation():
    """Serialize local task references with managed-dataset deletion across workers."""
    lock_directory = UPLOAD_ROOT.parent
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / ".managed-datasets.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def local_tasks_by_reference_priority(db: Session) -> list[TrainingTask]:
    return list(db.scalars(select(TrainingTask).where(
        TrainingTask.execution_node_id.is_(None),
    ).order_by(
        TrainingTask.deleted_at.is_not(None), TrainingTask.created_at.desc(), TrainingTask.id,
    )).all())


def canonical_path(raw: str | Path) -> Path:
    try:
        return Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return Path(os.path.abspath(os.path.expanduser(str(raw))))


def _paths_intersect(first: Path, second: Path) -> bool:
    left, right = canonical_path(first), canonical_path(second)
    return left == right or left in right.parents or right in left.parents


def _resolve_dataset_entry(raw: str, base: Path) -> Path | None:
    value = raw.strip()
    if not value or "://" in value or "\x00" in value:
        return None
    candidate = Path(value).expanduser()
    return canonical_path(candidate if candidate.is_absolute() else base / candidate)


def _manifest_references_directory(manifest: Path, target_directory: Path) -> bool:
    try:
        with manifest.open("r", encoding="utf-8-sig") as stream:
            for line in stream:
                value = line.strip()
                if not value or value.startswith("#"):
                    continue
                entry = _resolve_dataset_entry(value, manifest.parent)
                if entry is not None and _paths_intersect(entry, target_directory):
                    return True
    except (OSError, UnicodeError):
        return False
    return False


def _dataset_config_references_directory(config_path: Path, target_directory: Path) -> bool:
    config_path = canonical_path(config_path)
    target_directory = canonical_path(target_directory)
    if _paths_intersect(config_path, target_directory):
        return True
    if config_path.suffix.lower() not in {".yaml", ".yml"} or not config_path.is_file():
        return False
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False

    root_value = payload.get("path")
    root = _resolve_dataset_entry(str(root_value), config_path.parent) if root_value else canonical_path(config_path.parent)
    if root is None:
        return False
    # A YAML rooted inside the managed dataset necessarily consumes that
    # dataset. A root above it is only a reference when one of the split paths
    # actually intersects the managed directory.
    if root == target_directory or target_directory in root.parents:
        return True
    for key in ("train", "val", "test"):
        raw_entries = payload.get(key)
        entries = raw_entries if isinstance(raw_entries, list) else [raw_entries]
        for raw_entry in entries:
            if not isinstance(raw_entry, str):
                continue
            entry = _resolve_dataset_entry(raw_entry, root)
            if entry is None:
                continue
            if _paths_intersect(entry, target_directory):
                return True
            if entry.suffix.lower() == ".txt" and entry.is_file():
                if _manifest_references_directory(entry, target_directory):
                    return True
    return False


def dataset_references(db: Session, config_path: str,
                       local_tasks: list[TrainingTask] | None = None) -> list[TrainingTask]:
    target = canonical_path(config_path)
    target_directory = target.parent
    tasks = local_tasks if local_tasks is not None else local_tasks_by_reference_priority(db)
    task_ids = [task.id for task in tasks]
    historical: dict[str, list[str]] = {task_id: [] for task_id in task_ids}
    if task_ids:
        for run in db.scalars(select(TrainingRun).where(TrainingRun.task_id.in_(task_ids))):
            if run.data_path:
                historical[run.task_id].append(run.data_path)

    reference_cache: dict[str, bool] = {}
    references: list[TrainingTask] = []
    for task in tasks:
        paths = [task.data_path, *historical.get(task.id, [])]
        referenced = False
        for raw_path in dict.fromkeys(path for path in paths if path):
            candidate = canonical_path(raw_path)
            cache_key = str(candidate)
            if cache_key not in reference_cache:
                reference_cache[cache_key] = (
                    candidate == target
                    or _dataset_config_references_directory(candidate, target_directory)
                )
            if reference_cache[cache_key]:
                referenced = True
                break
        if referenced:
            references.append(task)
    return references


@management_router.get("")
def list_managed_datasets(db: Session = Depends(get_db)):
    root = DATASET_ROOTS[0]
    if not root.is_dir():
        return []
    datasets = []
    local_tasks = local_tasks_by_reference_priority(db)
    for directory in root.iterdir():
        if not directory.is_dir() or directory.is_symlink():
            continue
        metadata = managed_marker(directory)
        config_path = (directory / "data.yaml").resolve()
        if not metadata or not config_path.is_file():
            continue
        references = dataset_references(db, str(config_path), local_tasks)
        datasets.append({
            "name": directory.name,
            "directory": str(directory.resolve()),
            "config_path": str(config_path),
            "created_at": metadata.get("created_at"),
            "total_bytes": int(metadata.get("total_bytes") or 0),
            "stats": metadata.get("stats") or {},
            "classes": metadata.get("classes") or [],
            "warnings": metadata.get("warnings") or [],
            "references": [{"id": task.id, "name": task.name, "status": task.status,
                            "deleted": task.deleted_at is not None} for task in references],
        })
    return sorted(datasets, key=lambda item: (item["created_at"] or "", item["name"]), reverse=True)


@management_router.delete("/{name}", status_code=204)
def delete_managed_dataset(name: str, db: Session = Depends(get_db)):
    validate_name(name)
    root = DATASET_ROOTS[0].resolve()
    detached: Path | None = None
    upload_id: str | None = None
    with managed_dataset_mutation():
        directory = root / name
        metadata = managed_marker(directory) if directory.is_dir() and not directory.is_symlink() else None
        if metadata is None:
            raise HTTPException(404, "平台上传的数据集不存在")
        from .models import DatasetEntry
        from .dataset_store import busy_entry
        entry = db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == 'local', DatasetEntry.path == str(directory)))
        if entry and busy_entry(db, entry.id):
            raise HTTPException(409, '该数据集已有未完成操作')
        config_path = str((directory / "data.yaml").resolve())
        references = dataset_references(db, config_path)
        if references:
            task_names = "、".join(task.name for task in references[:3])
            suffix = f"等 {len(references)} 个任务" if len(references) > 3 else ""
            if suffix:
                task_names += "、"
            raise HTTPException(409, f"数据集仍被任务引用：{task_names}{suffix}；请先更换数据集或永久删除任务")
        quarantine = root / MANAGED_DELETION_DIR
        detached = quarantine / f"{uuid.uuid4().hex}-{name}"
        try:
            quarantine.mkdir(parents=True, exist_ok=True)
            directory.replace(detached)
            if entry:
                entry.status = 'deleted'
                db.commit()
        except OSError as exc:
            raise HTTPException(507, "数据集无法移入清理区，请检查目录权限和磁盘状态") from exc
        marker_upload_id = metadata.get("upload_id")
        if isinstance(marker_upload_id, str) and re.fullmatch(r"[0-9a-f]{32}", marker_upload_id):
            upload_id = marker_upload_id

    try:
        shutil.rmtree(detached)
    except FileNotFoundError:
        # A concurrent housekeeping pass may have already finished the detached deletion.
        pass
    except OSError as exc:
        raise HTTPException(507, "数据集已从平台移除，但部分文件清理失败；请检查磁盘和目录权限") from exc
    finally:
        if upload_id:
            shutil.rmtree(UPLOAD_ROOT / upload_id, ignore_errors=True)
        try:
            detached.parent.rmdir()
        except OSError:
            pass


def validate_name(name: str) -> None:
    if (name.startswith(".") or name != name.strip() or
            any(c in name for c in "/\\") or
            any(ord(c) < 32 or ord(c) == 127 for c in name) or
            len(name.encode("utf-8")) > 240):
        raise HTTPException(422, "文件夹名称不能包含路径分隔符、控制字符、前后空格或以点开头，且不能过长")


def validate_file_path(raw: str) -> PurePosixPath:
    parts = raw.split("/")
    if (any(not p or p.startswith(".") or len(p.encode("utf-8")) > 240 for p in parts) or
            "\\" in raw or any(ord(c) < 32 or ord(c) == 127 for c in raw)):
        raise HTTPException(422, f"文件相对路径不合法：{raw}")
    path = PurePosixPath(raw)
    if raw in METADATA_FILES:
        return path
    if len(parts) < 3 or parts[0] not in {"images", "labels"} or parts[1] not in {"train", "val"}:
        raise HTTPException(422, f"目录格式错误：{raw}。请选择包含 images/train、images/val 和 labels 的最外层文件夹")
    suffixes = IMAGE_SUFFIXES if parts[0] == "images" else {".txt"}
    if path.suffix.lower() not in suffixes:
        raise HTTPException(422, f"不支持的文件类型：{raw}")
    if parts[0] == "labels" and path.suffix != ".txt":
        raise HTTPException(422, f"标签扩展名必须是小写 .txt，以便训练框架匹配图片：{raw}")
    return path


@contextmanager
def upload_session(upload_id: str, *, touch: bool = True):
    if not re.fullmatch(r"[0-9a-f]{32}", upload_id):
        raise HTTPException(404, "上传记录不存在")
    directory = UPLOAD_ROOT / upload_id
    try:
        lock = (directory / ".lock").open("r+")
    except FileNotFoundError as exc:
        raise HTTPException(404, "上传记录不存在或已过期，请重新选择文件夹") from exc
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise HTTPException(409, "该上传正在处理，请稍后重试") from exc
        try:
            try:
                manifest_path = directory / "manifest.json"
                manifest = load_manifest(manifest_path, manifest_path.stat().st_mtime_ns)
                if touch:
                    os.utime(directory / ".lock", None)
            except FileNotFoundError as exc:
                raise HTTPException(404, "上传记录不存在或已过期，请重新选择文件夹") from exc
            yield directory, manifest
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@lru_cache(maxsize=2)
def load_manifest(path: Path, modified_ns: int) -> dict:
    # A large dataset has one request per file; parse its immutable manifest once.
    # Atomic receipt replacement changes the timestamp and invalidates this key.
    return json.loads(path.read_text(encoding="utf-8"))


def cleanup_expired_uploads() -> None:
    if not UPLOAD_ROOT.exists():
        return
    for directory in UPLOAD_ROOT.iterdir():
        if not re.fullmatch(r"[0-9a-f]{32}", directory.name) or not directory.is_dir() or directory.is_symlink():
            continue
        try:
            if time.time() - (directory / ".lock").stat().st_mtime < SESSION_TTL:
                continue
            with upload_session(directory.name, touch=False) as (_, manifest):
                if manifest.get('operation_id'):
                    continue  # The durable operation owns its staging until cleanup.
                if time.time() - (directory / ".lock").stat().st_mtime >= SESSION_TTL:
                    shutil.rmtree(directory)
        except (OSError, HTTPException):
            continue


def remove_publication_workdir(path: Path) -> None:
    """Remove one upload-owned hidden publication path without following symlinks."""
    try:
        if path.is_symlink():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    except FileNotFoundError:
        pass


def cleanup_managed_dataset_workdirs() -> None:
    """Retry detached deletions and remove stale, never-published upload directories."""
    root = DATASET_ROOTS[0]
    if not root.is_dir():
        return
    quarantine = root / MANAGED_DELETION_DIR
    if quarantine.is_dir() and not quarantine.is_symlink():
        for candidate in quarantine.iterdir():
            if candidate.is_dir() and not candidate.is_symlink():
                shutil.rmtree(candidate, ignore_errors=True)
        try:
            quarantine.rmdir()
        except OSError:
            pass
    now = time.time()
    for candidate in root.glob(f"{MANAGED_PUBLICATION_PREFIX}*"):
        upload_id = candidate.name.removeprefix(MANAGED_PUBLICATION_PREFIX)
        if (not re.fullmatch(r"[0-9a-f]{32}", upload_id) or not candidate.is_dir()
                or candidate.is_symlink()):
            continue
        try:
            if now - candidate.stat().st_mtime < SESSION_TTL:
                continue
            lock_path = UPLOAD_ROOT / upload_id / ".lock"
            try:
                lock = lock_path.open("r+")
            except FileNotFoundError:
                remove_publication_workdir(candidate)
                continue
            with lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                try:
                    try:
                        session_is_active = now - lock_path.stat().st_mtime < SESSION_TTL
                    except FileNotFoundError:
                        session_is_active = False
                    if not session_is_active:
                        remove_publication_workdir(candidate)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
        except OSError:
            continue


def completed_indices(directory: Path, manifest: dict) -> list[int]:
    return [index for index, spec in enumerate(manifest["files"])
            if (path := directory / "files" / spec["path"]).is_file() and path.stat().st_size == spec["size"]]


@router.post("", status_code=201)
def create_upload(payload: UploadCreate, db: Session = Depends(get_db)):
    validate_name(payload.name)
    location = None
    if payload.location_id:
        from .dataset_store import location_get
        location = location_get(db, payload.location_id, ready=True)
    destination = Path(location.resolved_path) / payload.name if location else DATASET_ROOTS[0] / payload.name
    if (not location or location.node_id == "local") and (destination.exists() or destination.is_symlink()):
        raise HTTPException(409, f"服务器已存在同名目录“{payload.name}”，请重命名本地文件夹后上传")
    seen: set[str] = set()
    image_keys: set[str] = set()
    counts = {"train": 0, "val": 0}
    total = 0
    for spec in payload.files:
        path = validate_file_path(spec.path)
        if spec.path in seen:
            raise HTTPException(422, f"存在重复文件：{spec.path}")
        seen.add(spec.path)
        if path.parts[0] == "images":
            if not spec.size:
                raise HTTPException(422, f"图片文件为空：{spec.path}")
            key = str(path.with_suffix(""))
            if key in image_keys:
                raise HTTPException(422, f"图片有同名但不同扩展名的文件，无法唯一匹配标签：{spec.path}")
            image_keys.add(key)
            counts[path.parts[1]] += 1
        elif spec.size > MAX_TEXT_BYTES:
            raise HTTPException(413, f"标签或配置文件不能超过 4 MB：{spec.path}")
        total += spec.size
    for raw in seen:
        if any(str(parent) in seen for parent in PurePosixPath(raw).parents):
            raise HTTPException(422, f"文件和目录路径发生冲突：{raw}")
    if not all(counts.values()):
        raise HTTPException(422, "images/train 和 images/val 都需要包含至少一张图片")
    if total > MAX_TOTAL_BYTES:
        raise HTTPException(413, "数据集总大小不能超过 50 GB")
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    cleanup_expired_uploads()
    try:
        cleanup_managed_dataset_workdirs()
    except OSError as exc:
        logger.warning("managed_dataset_workdir_cleanup_failed error=%s", type(exc).__name__)
    if shutil.disk_usage(UPLOAD_ROOT).free < total + 100 * 1024**2:
        raise HTTPException(507, "服务器磁盘空间不足，无法保存该数据集")
    upload_id = uuid.uuid4().hex
    directory = UPLOAD_ROOT / upload_id
    (directory / "files").mkdir(parents=True)
    (directory / ".lock").touch()
    manifest = {"name": payload.name, "files": [file.model_dump() for file in payload.files]}
    if location:
        manifest.update(location_id=location.id, node_id=location.node_id, root=location.resolved_path)
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return {"upload_id": upload_id, "destination": str(destination.resolve()), "total_bytes": total}


@router.get("/{upload_id}")
def get_upload(upload_id: str, db: Session = Depends(get_db)):
    with upload_session(upload_id) as (directory, manifest):
        if manifest.get('operation_id'):
            from .dataset_store import get_operation
            op = get_operation(manifest['operation_id'], db)
            return {'uploaded_indices': list(range(len(manifest['files']))), 'operation': op,
                    **({'result': op['result']} if op['status'] == 'completed' else {})}
        if "result" in manifest:
            return {"uploaded_indices": list(range(len(manifest["files"]))), "result": manifest["result"]}
        return {"uploaded_indices": completed_indices(directory, manifest)}


@router.put("/{upload_id}/files/{index}", status_code=204)
async def upload_file(upload_id: str, index: int, request: Request):
    with upload_session(upload_id) as (directory, manifest):
        if "result" in manifest or manifest.get("operation_id"):
            raise HTTPException(409, "该数据集已提交处理，不能继续修改上传文件")
        if index < 0 or index >= len(manifest["files"]):
            raise HTTPException(404, "上传文件不存在")
        spec = manifest["files"][index]
        target = directory / "files" / spec["path"]
        temporary = directory / f".part-{index}"
        size = 0
        checksum = hashlib.sha256()
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > spec["size"]:
                        raise HTTPException(413, "文件大小超过申报大小，请重新选择文件夹")
                    handle.write(chunk)
                    checksum.update(chunk)
            if spec.get('sha256') and checksum.hexdigest() != spec['sha256']:
                raise HTTPException(422, '文件 SHA-256 与上传清单不一致，请重新选择文件夹')
            if size != spec["size"]:
                raise HTTPException(422, "文件传输不完整，请重试")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.replace(target)
        except ClientDisconnect as exc:
            raise HTTPException(400, "上传连接已中断") from exc
        except OSError as exc:
            raise HTTPException(507, "文件保存失败，请检查服务器磁盘空间") from exc
        finally:
            temporary.unlink(missing_ok=True)


def class_names(files_root: Path, supplied: list[str]) -> list[str]:
    names = supplied
    if not names:
        for name in ("data.yaml", "data.yml"):
            source = files_root / name
            if not source.exists():
                continue
            try:
                config = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
                values = config.get("names") if isinstance(config, dict) else None
                if isinstance(values, dict):
                    if any(type(key) is not int and not (isinstance(key, str) and re.fullmatch(r"0|[1-9][0-9]*", key)) for key in values):
                        raise ValueError("类别编号必须为非负整数")
                    indexed = {int(key): value for key, value in values.items()}
                    if len(indexed) != len(values) or set(indexed) != set(range(len(indexed))):
                        raise ValueError("类别编号必须从 0 开始连续排列")
                    names = [indexed[index] for index in range(len(indexed))]
                elif isinstance(values, list):
                    names = values
            except (ValueError, TypeError, UnicodeError, yaml.YAMLError) as exc:
                raise HTTPException(422, f"{name} 中的类别配置无法读取：{exc}") from exc
            if names:
                break
        if not names and (source := files_root / "classes.txt").exists():
            try:
                names = source.read_text(encoding="utf-8-sig").splitlines()
            except UnicodeError as exc:
                raise HTTPException(422, "classes.txt 必须采用 UTF-8 编码") from exc
    if not isinstance(names, list) or not names or len(names) > 1000:
        raise HTTPException(422, "请填写类别名称（每行一个，编号从 0 开始），或上传带 names 的 data.yaml")
    if any(not isinstance(name, str) or not name.strip() or len(name) > 160 or
           any(ord(c) < 32 for c in name) for name in names):
        raise HTTPException(422, "类别名称不能为空、包含控制字符或超过 160 个字符")
    names = [name.strip() for name in names]
    if len(set(names)) != len(names):
        raise HTTPException(422, "类别名称不能重复")
    return names


def validate_contents(files_root: Path, manifest: dict, names: list[str], check=None) -> tuple[dict, list[str]]:
    images: dict[str, set[str]] = {"train": set(), "val": set()}
    image_paths: dict[str, set[str]] = {"train": set(), "val": set()}
    labels: dict[str, set[str]] = {"train": set(), "val": set()}
    empty_labels: dict[str, set[str]] = {"train": set(), "val": set()}
    from .dataset_files import report_progress
    total_files = len(manifest['files'])
    report_progress(check, 'content_validate', 0, total_files)
    for index, spec in enumerate(manifest["files"]):
        if check:
            check()
            # Reloads can provide a byte budget for image decoding as well as
            # archive I/O. Other callers have no hook and keep their behavior.
            throttle = getattr(check, 'throttle_bytes', None)
            if throttle:
                throttle(spec['size'] * (2 if spec['path'].startswith('images/') else 1))
        report_progress(check, 'content_validate', index, total_files, current_file=spec['path'])
        relative = PurePosixPath(spec["path"])
        path = files_root / relative
        if relative.parts[0] == "images":
            try:
                with python_warnings.catch_warnings():
                    python_warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
            except (OSError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, SyntaxError) as exc:
                raise HTTPException(422, f"图片损坏或无法读取：{relative}") from exc
            split = relative.parts[1]
            image_path = str(PurePosixPath(*relative.parts[2:]))
            key = str(PurePosixPath(image_path).with_suffix(""))
            if key in images[split]:
                raise HTTPException(422, f"图片有同名但不同扩展名的文件，无法唯一匹配标签：{relative}")
            image_paths[split].add(image_path)
            images[split].add(key)
        elif relative.parts[0] == "labels":
            labels[relative.parts[1]].add(str(PurePosixPath(*relative.parts[2:]).with_suffix("")))
            targets = 0
            try:
                with path.open(encoding="utf-8-sig") as handle:
                    for number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        values = [float(value) for value in line.split()]
                        if (len(values) != 5 or not all(math.isfinite(v) for v in values) or
                                not values[0].is_integer() or not 0 <= values[0] < len(names) or
                                not all(0 <= v <= 1 for v in values[1:]) or
                                values[3] <= 0 or values[4] <= 0):
                            raise ValueError(f"第 {number} 行应为：类别编号 中心x 中心y 宽 高；坐标需归一化，宽高需大于 0")
                        targets += 1
            except (UnicodeError, ValueError) as exc:
                raise HTTPException(422, f"标签格式错误：{relative}，{exc}") from exc
            if not targets:
                empty_labels[relative.parts[1]].add(str(PurePosixPath(*relative.parts[2:]).with_suffix("")))
    report_progress(check, 'content_validate', total_files, total_files)
    missing = sum(len(images[split] - labels[split]) for split in images)
    orphaned = sum(len(labels[split] - images[split]) for split in labels)
    empty_backgrounds = sum(len(images[split] & empty_labels[split]) for split in images)
    warnings = []
    if missing:
        warnings.append(f"{missing} 张图片没有对应标签，将作为背景图片参与训练，请确认是否符合预期。")
    if empty_backgrounds:
        warnings.append(f"{empty_backgrounds} 张图片的标签文件为空，将作为背景图片。")
    if orphaned:
        warnings.append(f"{orphaned} 个标签没有对应图片，训练时不会使用。")
    stats = {"train_images": len(image_paths["train"]), "val_images": len(image_paths["val"]),
             "labels": sum(len(values) for values in labels.values()), "classes": len(names), "background_images": missing + empty_backgrounds}
    return stats, warnings


def published_upload_result(destination: Path, upload_id: str) -> dict | None:
    """Rebuild a lost receipt from the immutable, published dataset metadata."""
    if not destination.is_dir() or destination.is_symlink():
        return None
    metadata = managed_marker(destination)
    if not metadata or metadata.get("upload_id") != upload_id:
        return None
    config_path = destination / "data.yaml"
    try:
        if not config_path.is_file():
            raise OSError("data.yaml is missing")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or not isinstance(config.get("names"), (dict, list)) or not config["names"]:
            raise ValueError("data.yaml names are missing")
        names = class_names(destination, [])
        config_size = config_path.stat().st_size
    except (OSError, ValueError, TypeError, UnicodeError, yaml.YAMLError, HTTPException) as exc:
        raise HTTPException(500, "已发布的数据集元数据损坏，无法恢复上传状态") from exc
    stats = metadata["stats"]
    if (metadata["classes"] != names or set(stats) != MANAGED_STAT_KEYS
            or stats.get("classes") != len(names)):
        raise HTTPException(500, "已发布的数据集元数据不一致，无法恢复上传状态")
    return {
        "name": destination.name,
        "directory": str(destination.resolve()),
        "names": names,
        "stats": stats,
        "warnings": metadata["warnings"],
        "resource": {
            "name": destination.name,
            "path": str(config_path.resolve()),
            "root": str(DATASET_ROOTS[0]),
            "size": config_size,
        },
    }


@router.post("/{upload_id}/complete")
def complete_upload(upload_id: str, payload: UploadComplete, db: Session = Depends(get_db)):
    with upload_session(upload_id) as (directory, manifest):
        if "result" in manifest:
            return manifest["result"]
        files_root = directory / "files"
        if manifest.get('location_id'):
            from .dataset_store import queue_upload, get_operation
            if manifest.get('operation_id'):
                op = get_operation(manifest['operation_id'], db)
                return op['result'] if op['status'] == 'completed' else {'operation': op}
            if len(completed_indices(directory, manifest)) != len(manifest['files']):
                raise HTTPException(409, '文件尚未全部上传')
            names = class_names(files_root, payload.names)
            stats, warnings = validate_contents(files_root, manifest, names)
            with managed_dataset_mutation():
                from .gpu_reservations import acquire_node_claim_lock
                acquire_node_claim_lock(db, manifest['node_id'])
                op = queue_upload(db, upload_id, directory, manifest, names, stats, warnings)
            receipt = directory / '.manifest.tmp'
            receipt.write_text(json.dumps({**manifest, 'operation_id': op['id']}, ensure_ascii=False), encoding='utf-8')
            receipt.replace(directory / 'manifest.json')
            return {'operation': op}
        destination = DATASET_ROOTS[0] / manifest["name"]
        receipt = directory / ".manifest.tmp"

        # A process may have published the directory and stopped before replacing
        # the upload receipt. Recover from published facts before consulting this
        # retry's class-name payload or the remaining staging files.
        recovered_result = None
        with managed_dataset_mutation():
            recovered_result = published_upload_result(destination, upload_id)
            if recovered_result is not None:
                try:
                    receipt.write_text(
                        json.dumps({**manifest, "result": recovered_result}, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    receipt.replace(directory / "manifest.json")
                except OSError as exc:
                    raise HTTPException(507, "数据集已发布，但上传状态保存失败，请重试") from exc
            elif destination.exists() or destination.is_symlink():
                raise HTTPException(409, f"服务器已存在同名目录“{manifest['name']}”，请重命名本地文件夹后上传")
        if recovered_result is not None:
            shutil.rmtree(files_root, ignore_errors=True)
            return recovered_result

        if len(completed_indices(directory, manifest)) != len(manifest["files"]):
            raise HTTPException(409, "文件尚未全部上传，请继续上传或重试")
        names = class_names(files_root, payload.names)
        stats, warnings = validate_contents(files_root, manifest, names)
        total_bytes = sum(spec["size"] for spec in manifest["files"])
        config_path = destination / "data.yaml"
        config_text = yaml.safe_dump({
            "path": str(destination.resolve()), "train": "images/train", "val": "images/val",
            "names": dict(enumerate(names)),
        }, allow_unicode=True, sort_keys=False)
        result = {"name": manifest["name"], "directory": str(destination.resolve()), "names": names,
                  "stats": stats, "warnings": warnings,
                  "resource": {"name": manifest["name"], "path": str(config_path.resolve()),
                               "root": str(DATASET_ROOTS[0]), "size": len(config_text.encode("utf-8"))}}

        publication = destination.parent / f"{MANAGED_PUBLICATION_PREFIX}{upload_id}"
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A stopped process may have left a large partial copy. Reclaim it
            # before checking whether there is room to rebuild the full dataset.
            remove_publication_workdir(publication)
            if shutil.disk_usage(destination.parent).free < total_bytes + 100 * 1024**2:
                raise HTTPException(507, "数据集目标目录所在磁盘空间不足")
            publication.mkdir()
        except OSError as exc:
            raise HTTPException(507, "无法创建数据集目录，请检查服务器权限和磁盘空间") from exc
        try:
            # Build under a hidden sibling, then expose the complete dataset with
            # one same-filesystem directory rename.
            for folder in ("images", "labels"):
                source = files_root / folder
                if source.exists():
                    shutil.copytree(source, publication / folder)
                for split in ("train", "val"):
                    (publication / folder / split).mkdir(parents=True, exist_ok=True)
            (publication / "data.yaml").write_text(config_text, encoding="utf-8")
            (publication / MANAGED_MARKER).write_text(json.dumps({
                "version": 1,
                "upload_id": upload_id,
                "name": manifest["name"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_bytes": total_bytes,
                "stats": stats,
                "classes": names,
                "warnings": warnings,
            }, ensure_ascii=False), encoding="utf-8")
            # Retain a success receipt so a lost response can be safely retried.
            receipt.write_text(json.dumps({**manifest, "result": result}, ensure_ascii=False), encoding="utf-8")
            with managed_dataset_mutation():
                if destination.exists() or destination.is_symlink():
                    raise HTTPException(409, f"服务器已存在同名目录“{manifest['name']}”，请重命名本地文件夹后上传")
                publication.replace(destination)
                receipt.replace(directory / "manifest.json")
        except HTTPException:
            shutil.rmtree(publication, ignore_errors=True)
            raise
        except OSError as exc:
            shutil.rmtree(publication, ignore_errors=True)
            raise HTTPException(507, "数据集保存失败，请检查磁盘空间后重试") from exc
        shutil.rmtree(files_root, ignore_errors=True)
        return result


@router.delete("/{upload_id}", status_code=204)
def cancel_upload(upload_id: str, db: Session = Depends(get_db)):
    with upload_session(upload_id) as (directory, manifest):
        if manifest.get('operation_id'):
            from .dataset_store import control_operation
            control_operation(manifest['operation_id'], 'cancel', db)
        publication = DATASET_ROOTS[0] / f"{MANAGED_PUBLICATION_PREFIX}{upload_id}"
        try:
            remove_publication_workdir(publication)
            shutil.rmtree(directory)
        except OSError as exc:
            raise HTTPException(507, "上传临时文件清理失败，请检查目录权限后重试") from exc
