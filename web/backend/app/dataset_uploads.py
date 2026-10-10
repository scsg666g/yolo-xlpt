"""Directory uploads: stage files privately, validate, then publish a named dataset."""
from __future__ import annotations

import fcntl
import json
import math
import os
import re
import shutil
import time
import uuid
import warnings as python_warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath

import yaml
from fastapi import APIRouter, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from starlette.requests import ClientDisconnect

from .config import DATA_ROOT, DATASET_ROOTS

router = APIRouter(prefix="/api/datasets/uploads", tags=["dataset uploads"])
UPLOAD_ROOT = DATA_ROOT / "dataset_uploads"
MAX_FILES = 200_000
MAX_TOTAL_BYTES = 50 * 1024**3
MAX_FILE_BYTES = 256 * 1024**2
MAX_TEXT_BYTES = 4 * 1024**2
SESSION_TTL = 24 * 60 * 60
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
METADATA_FILES = {"data.yaml", "data.yml", "classes.txt"}


class UploadFileSpec(BaseModel):
    path: str = Field(min_length=1, max_length=1024)
    size: int = Field(ge=0, le=MAX_FILE_BYTES)


class UploadCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    files: list[UploadFileSpec] = Field(min_length=1, max_length=MAX_FILES)


class UploadComplete(BaseModel):
    names: list[str] = Field(default_factory=list, max_length=1000)


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
            with upload_session(directory.name, touch=False):
                if time.time() - (directory / ".lock").stat().st_mtime >= SESSION_TTL:
                    shutil.rmtree(directory)
        except (OSError, HTTPException):
            continue


def completed_indices(directory: Path, manifest: dict) -> list[int]:
    return [index for index, spec in enumerate(manifest["files"])
            if (path := directory / "files" / spec["path"]).is_file() and path.stat().st_size == spec["size"]]


@router.post("", status_code=201)
def create_upload(payload: UploadCreate):
    validate_name(payload.name)
    destination = DATASET_ROOTS[0] / payload.name
    if destination.exists() or destination.is_symlink():
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
    if shutil.disk_usage(UPLOAD_ROOT).free < total + 100 * 1024**2:
        raise HTTPException(507, "服务器磁盘空间不足，无法保存该数据集")
    upload_id = uuid.uuid4().hex
    directory = UPLOAD_ROOT / upload_id
    (directory / "files").mkdir(parents=True)
    (directory / ".lock").touch()
    manifest = {"name": payload.name, "files": [file.model_dump() for file in payload.files]}
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return {"upload_id": upload_id, "destination": str(destination.resolve()), "total_bytes": total}


@router.get("/{upload_id}")
def get_upload(upload_id: str):
    with upload_session(upload_id) as (directory, manifest):
        if "result" in manifest:
            return {"uploaded_indices": list(range(len(manifest["files"]))), "result": manifest["result"]}
        return {"uploaded_indices": completed_indices(directory, manifest)}


@router.put("/{upload_id}/files/{index}", status_code=204)
async def upload_file(upload_id: str, index: int, request: Request):
    with upload_session(upload_id) as (directory, manifest):
        if "result" in manifest:
            raise HTTPException(409, "该数据集已保存，不能继续修改上传文件")
        if index < 0 or index >= len(manifest["files"]):
            raise HTTPException(404, "上传文件不存在")
        spec = manifest["files"][index]
        target = directory / "files" / spec["path"]
        temporary = directory / f".part-{index}"
        size = 0
        try:
            with temporary.open("wb") as handle:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > spec["size"]:
                        raise HTTPException(413, "文件大小超过申报大小，请重新选择文件夹")
                    handle.write(chunk)
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


def validate_contents(files_root: Path, manifest: dict, names: list[str]) -> tuple[dict, list[str]]:
    images: dict[str, set[str]] = {"train": set(), "val": set()}
    labels: dict[str, set[str]] = {"train": set(), "val": set()}
    empty_labels: dict[str, set[str]] = {"train": set(), "val": set()}
    for spec in manifest["files"]:
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
            images[relative.parts[1]].add(str(PurePosixPath(*relative.parts[2:]).with_suffix("")))
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
    stats = {"train_images": len(images["train"]), "val_images": len(images["val"]),
             "labels": sum(len(values) for values in labels.values()), "classes": len(names), "background_images": missing + empty_backgrounds}
    return stats, warnings


@router.post("/{upload_id}/complete")
def complete_upload(upload_id: str, payload: UploadComplete):
    with upload_session(upload_id) as (directory, manifest):
        if "result" in manifest:
            return manifest["result"]
        if len(completed_indices(directory, manifest)) != len(manifest["files"]):
            raise HTTPException(409, "文件尚未全部上传，请继续上传或重试")
        files_root = directory / "files"
        names = class_names(files_root, payload.names)
        stats, warnings = validate_contents(files_root, manifest, names)
        destination = DATASET_ROOTS[0] / manifest["name"]
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(destination.parent).free < sum(spec["size"] for spec in manifest["files"]) + 100 * 1024**2:
                raise HTTPException(507, "数据集目标目录所在磁盘空间不足")
            destination.mkdir()
        except FileExistsError as exc:
            raise HTTPException(409, f"服务器已存在同名目录“{manifest['name']}”，请重命名本地文件夹后上传") from exc
        except OSError as exc:
            raise HTTPException(507, "无法创建数据集目录，请检查服务器权限和磁盘空间") from exc
        try:
            # Keep staged files until publication succeeds, allowing a safe retry on IO failure.
            for folder in ("images", "labels"):
                source = files_root / folder
                if source.exists():
                    shutil.copytree(source, destination / folder)
                for split in ("train", "val"):
                    (destination / folder / split).mkdir(parents=True, exist_ok=True)
            config_path = destination / "data.yaml"
            config = {"path": str(destination.resolve()), "train": "images/train", "val": "images/val",
                      "names": dict(enumerate(names))}
            config_temporary = destination / ".data.yaml.tmp"
            config_temporary.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
            result = {"name": manifest["name"], "directory": str(destination.resolve()), "names": names,
                      "stats": stats, "warnings": warnings,
                      "resource": {"name": manifest["name"], "path": str(config_path.resolve()),
                                   "root": str(DATASET_ROOTS[0]), "size": config_temporary.stat().st_size}}
            config_temporary.replace(config_path)
            # Retain a success receipt so a lost response can be safely retried.
            receipt = directory / ".manifest.tmp"
            receipt.write_text(json.dumps({**manifest, "result": result}, ensure_ascii=False), encoding="utf-8")
            receipt.replace(directory / "manifest.json")
        except OSError as exc:
            shutil.rmtree(destination, ignore_errors=True)
            raise HTTPException(507, "数据集保存失败，请检查磁盘空间后重试") from exc
        shutil.rmtree(files_root, ignore_errors=True)
        return result


@router.delete("/{upload_id}", status_code=204)
def cancel_upload(upload_id: str):
    with upload_session(upload_id) as (directory, _):
        shutil.rmtree(directory)
