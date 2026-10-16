"""Temporary image tests, serialized on CPU and isolated from training outputs."""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.requests import ClientDisconnect

from .config import DATA_ROOT, ULTRALYTICS_ROOT, YOLO_BIN
from .database import get_db
from .models import Artifact, TrainingRun, TrainingTask
from .remote_operations import matching_fetch, operation_file
from .services import failure_detail, path_is_within, training_environment

router = APIRouter(prefix="/api/model-tests", tags=["model-tests"])
logger = logging.getLogger("yolo-platform.inference")
TEST_ROOT = DATA_ROOT / "model-tests"
MAX_IMAGE_BYTES = 20 * 1024**2
MAX_TOTAL_BYTES = 100 * 1024**2
TTL = 24 * 60 * 60


class ImageSpec(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=MAX_IMAGE_BYTES)


class TestCreate(BaseModel):
    artifact_id: int
    confidence: float = Field(default=0.25, ge=0.01, le=1)
    files: list[ImageSpec] = Field(min_length=1, max_length=10)


@dataclass
class ImageTest:
    id: str
    artifact_id: int
    confidence: float
    files: list[dict]
    directory: Path
    created_at: float = field(default_factory=time.time)
    status: str = "uploading"
    error: str | None = None
    uploading: set[int] = field(default_factory=set)

    def public(self) -> dict:
        return {"id": self.id, "status": self.status, "error": self.error,
                "confidence": self.confidence, "files": [dict(item) for item in self.files],
                "expires_at": self.created_at + TTL}


def weight_path(db: Session, artifact_id: int) -> Path:
    artifact = db.get(Artifact, artifact_id)
    task = db.get(TrainingTask, artifact.task_id) if artifact else None
    run = db.get(TrainingRun, artifact.run_id) if artifact and artifact.run_id else None
    if not artifact or not task or task.deleted_at or not run or run.task_id != task.id:
        raise HTTPException(404, "模型权重所属的训练运行不存在")
    if task.task_type != "detect" or artifact.kind != "weight" or Path(artifact.name).suffix.lower() != ".pt":
        raise HTTPException(422, "图片测试目前支持目标检测任务的 .pt 权重")
    if run.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "请等待本次训练结束或停止后再测试模型")
    if run.execution_node_id:
        operation = matching_fetch(db, artifact)
        if not operation or operation.status != "succeeded":
            raise HTTPException(409, "远程权重尚未准备完成，请先准备模型")
        path = operation_file(operation)
    else:
        path = Path(artifact.path).resolve()
        if not run.run_dir or not path_is_within(path, [Path(run.run_dir)]):
            raise HTTPException(403, "模型权重路径不受信任")
    if not path.is_file():
        raise HTTPException(404, "模型权重文件不存在，请刷新后重试")
    return path


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


class InferenceWorker:
    def __init__(self):
        self.jobs: dict[str, ImageTest] = {}
        self.lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen | None = None
        self._current_id: str | None = None

    def get(self, test_id: str) -> ImageTest:
        job = self.jobs.get(test_id)
        if not job:
            raise HTTPException(404, "测试记录已过期或服务已重启，请重新测试")
        return job

    def start(self) -> None:
        # These are temporary previews, not training artifacts or durable task records.
        TEST_ROOT.mkdir(parents=True, exist_ok=True)
        self.jobs.clear()
        for path in TEST_ROOT.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", path.name) and path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
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

    def _loop(self) -> None:
        while not self._stop.wait(1):
            with self.lock:
                for test_id, job in list(self.jobs.items()):
                    if time.time() - job.created_at > TTL and job.status not in {"preparing", "queued", "running"} and not job.uploading:
                        shutil.rmtree(job.directory, ignore_errors=True)
                        del self.jobs[test_id]
                job = next((job for job in self.jobs.values() if job.status == "queued"), None)
                if job:
                    job.status = "running"
            if job:
                self.execute(job)

    def execute(self, job: ImageTest) -> None:
        output = job.directory / "results"
        log = job.directory / "inference.log"
        command = [str(YOLO_BIN), "detect", "predict", f"model={job.directory / 'model.pt'}",
                   f"source={job.directory / 'images'}", f"project={job.directory}", "name=results",
                   "device=cpu", "batch=1", "imgsz=640", f"conf={job.confidence}",
                   "save=True", "save_txt=True", "save_conf=True", "exist_ok=True", "verbose=False"]
        try:
            environment = {**training_environment(), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2",
                           "MKL_NUM_THREADS": "2"}
            with log.open("w", encoding="utf-8") as handle:
                with self.lock:
                    if self._stop.is_set() or job.status == "cancelled":
                        raise RuntimeError("服务正在重启，请重新测试")
                    self._process = subprocess.Popen(command, cwd=ULTRALYTICS_ROOT, env=environment,
                                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    process = self._process
                    self._current_id = job.id
                try:
                    code = process.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("推理超过 5 分钟，请减少图片数量或尺寸后重试") from None
            if code:
                raise RuntimeError(failure_detail(str(log)) or "模型推理失败，请检查权重是否兼容当前训练环境")
            counts = []
            for index in range(len(job.files)):
                if not (output / f"{index}.jpg").is_file():
                    raise RuntimeError("推理没有生成完整结果，请确认模型为目标检测模型")
                label = output / "labels" / f"{index}.txt"
                counts.append(len(label.read_text().splitlines()) if label.is_file() else 0)
            with self.lock:
                if job.status != "cancelled":
                    for item, count in zip(job.files, counts):
                        item["detections"] = count
                    job.status = "completed"
        except Exception as exc:
            with self.lock:
                if job.status != "cancelled":
                    logger.exception("image_inference_failed test_id=%s", job.id)
                    job.status, job.error = "failed", str(exc)[-1000:]
        finally:
            (job.directory / "model.pt").unlink(missing_ok=True)
            with self.lock:
                self._process = None
                self._current_id = None
                if job.status == "cancelled":
                    shutil.rmtree(job.directory, ignore_errors=True)


worker = InferenceWorker()


@router.post("", status_code=201)
def create_test(payload: TestCreate, db: Session = Depends(get_db)):
    weight_path(db, payload.artifact_id)
    if sum(item.size for item in payload.files) > MAX_TOTAL_BYTES:
        raise HTTPException(413, "一次测试的图片总大小不能超过 100 MB")
    with worker.lock:
        if len(worker.jobs) >= 20:
            raise HTTPException(429, "测试记录已达上限，请清除旧测试后重试")
        test_id = uuid.uuid4().hex
        directory = TEST_ROOT / test_id
        (directory / "images").mkdir(parents=True)
        job = ImageTest(test_id, payload.artifact_id, payload.confidence,
                        [{**item.model_dump(), "uploaded": False, "detections": None} for item in payload.files], directory)
        worker.jobs[test_id] = job
        return job.public()


@router.put("/{test_id}/images/{index}", status_code=204)
async def upload_image(test_id: str, index: int, request: Request):
    with worker.lock:
        job = worker.get(test_id)
        if job.status != "uploading" or index in job.uploading:
            raise HTTPException(409, "当前测试不能上传图片")
        if not 0 <= index < len(job.files):
            raise HTTPException(404, "图片不存在")
        job.uploading.add(index)
        job.files[index]["uploaded"] = False
    temporary = job.directory / f"{index}.part"
    try:
        size = 0
        with temporary.open("wb") as handle:
            async for chunk in request.stream():
                size += len(chunk)
                if size > job.files[index]["size"]:
                    raise HTTPException(413, "图片大小超过申报大小")
                handle.write(chunk)
        if size != job.files[index]["size"]:
            raise HTTPException(422, "图片上传不完整，请重试")
        await asyncio.to_thread(normalize_image, temporary, job.directory / "images" / f"{index}.jpg")
        with worker.lock:
            job.files[index]["uploaded"] = True
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except ClientDisconnect as exc:
        raise HTTPException(400, "上传连接已中断，请重新测试") from exc
    except OSError as exc:
        raise HTTPException(507, "图片保存失败，请检查服务器磁盘空间") from exc
    finally:
        temporary.unlink(missing_ok=True)
        with worker.lock:
            job.uploading.discard(index)


@router.post("/{test_id}/start", status_code=202)
def start_test(test_id: str, db: Session = Depends(get_db)):
    with worker.lock:
        job = worker.get(test_id)
        if job.status != "uploading":
            return job.public()
        if job.uploading or not all(item["uploaded"] for item in job.files):
            raise HTTPException(409, "请先上传全部图片")
        model = weight_path(db, job.artifact_id)
        if shutil.disk_usage(job.directory).free < model.stat().st_size + MAX_TOTAL_BYTES:
            raise HTTPException(507, "磁盘空间不足，无法准备模型副本")
        job.status = "preparing"
    # Copying a large checkpoint must not hold the API/worker lock.
    try:
        shutil.copyfile(model, job.directory / "model.pt")
    except OSError as exc:
        (job.directory / "model.pt").unlink(missing_ok=True)
        with worker.lock:
            if job.status == "cancelled":
                shutil.rmtree(job.directory, ignore_errors=True)
            else:
                job.status = "uploading"
        raise HTTPException(507, "无法准备模型副本，请检查文件和磁盘空间") from exc
    with worker.lock:
        if job.status == "cancelled":
            shutil.rmtree(job.directory, ignore_errors=True)
            raise HTTPException(409, "测试已取消")
        job.status = "queued"
        return job.public()


@router.get("/{test_id}")
def test_status(test_id: str):
    with worker.lock:
        return worker.get(test_id).public()


@router.get("/{test_id}/images/{index}/{kind}")
def test_image(test_id: str, index: int, kind: str, download: bool = False):
    with worker.lock:
        job = worker.get(test_id)
        if not 0 <= index < len(job.files) or kind not in {"original", "result"}:
            raise HTTPException(404, "图片不存在")
        if kind == "result" and job.status != "completed":
            raise HTTPException(409, "检测结果尚未生成")
        path = job.directory / ("images" if kind == "original" else "results") / f"{index}.jpg"
        if not path.is_file():
            raise HTTPException(404, "图片尚未上传")
        filename = f"detection-{index + 1}.jpg" if download else None
        return FileResponse(path, media_type="image/jpeg", filename=filename)


@router.delete("/{test_id}", status_code=204)
def delete_test(test_id: str):
    with worker.lock:
        job = worker.get(test_id)
        if job.uploading:
            raise HTTPException(409, "图片正在上传，请稍后清除")
        deferred_cleanup = job.status in {"preparing", "running"}
        job.status = "cancelled"
        if worker._current_id == test_id and worker._process and worker._process.poll() is None:
            worker._process.kill()
        if not deferred_cleanup:
            shutil.rmtree(job.directory, ignore_errors=True)
        del worker.jobs[test_id]
