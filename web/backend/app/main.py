from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import DATASET_ROOTS, MODEL_ROOTS, RUNS_ROOT, WEB_ROOT
from .database import Base, SessionLocal, engine, get_db
from .models import Artifact, MetricPoint, TaskEvent, TrainingTask, utcnow
from .schemas import ArtifactRead, EventRead, LayoutUpdate, MetricRead, TaskConfigUpdate, TaskCreate, TaskRead, TaskResourceUpdate
from .services import (
    add_event,
    compute_inventory,
    discover_files,
    gpu_inventory,
    import_existing_runs,
    path_is_within,
    scheduler,
    sync_artifacts,
    sync_metrics,
    validate_input_path,
)

EDITABLE_TRAINING_KEYS = {
    "patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay",
    "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction",
    "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl",
}
RESERVED_CUSTOM_KEYS = {
    "model", "data", "project", "name", "device", "epochs", "batch", "imgsz", "workers", "optimizer",
    "task", "mode", "exist_ok", "save_dir", "resume", "source",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(engine)
    import_existing_runs()
    scheduler.start()
    yield
    scheduler.stop()


app = FastAPI(title="YOLO 模型训练平台", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_task_or_404(db: Session, task_id: str) -> TrainingTask:
    task = db.get(TrainingTask, task_id)
    if not task:
        raise HTTPException(404, "训练任务不存在")
    return task


@app.get("/api/health")
def health():
    return {"status": "ok", "scheduler": "running", "runs_root": str(RUNS_ROOT)}


@app.get("/api/resources")
def resources():
    return {
        "gpus": gpu_inventory(),
        "models": discover_files(MODEL_ROOTS, {".pt", ".pth", ".onnx", ".yaml", ".yml"}),
        "datasets": discover_files(DATASET_ROOTS, {".yaml", ".yml"}),
    }


@app.get("/api/compute")
def compute_resources():
    return compute_inventory()


@app.get("/api/tasks", response_model=list[TaskRead])
def list_tasks(status: str | None = Query(default=None), db: Session = Depends(get_db)):
    query = select(TrainingTask).order_by(TrainingTask.created_at.desc())
    if status:
        query = query.where(TrainingTask.status == status)
    tasks = db.scalars(query).all()
    for task in tasks:
        if task.status not in {"running", "starting"}:
            sync_metrics(db, task)
    db.commit()
    return tasks


@app.post("/api/tasks", response_model=TaskRead, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    try:
        model_path = validate_input_path(payload.model_path, MODEL_ROOTS, "模型文件")
        data_path = validate_input_path(payload.data_path, DATASET_ROOTS, "数据集配置")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    task_id = uuid.uuid4().hex[:16]
    config = payload.model_dump(exclude={"name", "task_type", "model_path", "data_path", "epochs", "batch", "imgsz", "device", "workers", "optimizer", "advanced"})
    config.update({key: value for key, value in payload.advanced.items() if key in {
        "cos_lr", "weight_decay", "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed",
        "deterministic", "fraction", "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate",
        "scale", "shear", "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl",
    }})
    task = TrainingTask(
        id=task_id,
        name=payload.name.strip(),
        task_type=payload.task_type,
        model_path=str(model_path),
        data_path=str(data_path),
        epochs=payload.epochs,
        batch=payload.batch,
        imgsz=payload.imgsz,
        device=payload.device,
        workers=payload.workers,
        optimizer=payload.optimizer,
        config=config,
    )
    db.add(task)
    db.flush()
    add_event(db, task.id, "created", "训练任务已创建")
    db.commit()
    db.refresh(task)
    return task


@app.get("/api/tasks/{task_id}", response_model=TaskRead)
def get_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    sync_metrics(db, task)
    sync_artifacts(db, task)
    db.commit()
    return task


@app.post("/api/tasks/{task_id}/start", response_model=TaskRead)
def start_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status not in {"draft", "stopped", "failed", "interrupted"}:
        raise HTTPException(409, f"当前状态 {task.status} 不能启动")
    if not Path(task.model_path).exists() or not Path(task.data_path).exists():
        raise HTTPException(409, "模型或数据集路径已经失效，请克隆任务并选择有效路径")
    task.status = "queued"
    task.queued_at = utcnow()
    task.finished_at = None
    task.exit_code = None
    task.error_message = None
    add_event(db, task.id, "queued", "任务已加入训练队列")
    db.commit()
    db.refresh(task)
    return task


@app.post("/api/tasks/{task_id}/stop", response_model=TaskRead)
def stop_task(task_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    if not scheduler.stop_task(task_id):
        raise HTTPException(409, "该任务当前无法停止")
    with SessionLocal() as refreshed_db:
        return get_task_or_404(refreshed_db, task_id)


@app.post("/api/tasks/{task_id}/clone", response_model=TaskRead, status_code=201)
def clone_task(task_id: str, db: Session = Depends(get_db)):
    source = get_task_or_404(db, task_id)
    clone = TrainingTask(
        id=uuid.uuid4().hex[:16], name=f"{source.name} - 副本", task_type=source.task_type,
        model_path=source.model_path, data_path=source.data_path, epochs=source.epochs, batch=source.batch,
        imgsz=source.imgsz, device=source.device, workers=source.workers, optimizer=source.optimizer,
        config=source.config.copy(), canvas_layout=source.canvas_layout.copy(),
    )
    db.add(clone)
    db.flush()
    add_event(db, clone.id, "cloned", f"从任务 {source.name} 克隆", payload={"source_id": source.id})
    db.commit()
    db.refresh(clone)
    return clone


@app.delete("/api/tasks/{task_id}", status_code=204)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "运行中或排队中的任务不能删除")
    db.delete(task)
    db.commit()


@app.patch("/api/tasks/{task_id}/layout", response_model=TaskRead)
def save_layout(task_id: str, payload: LayoutUpdate, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    task.canvas_layout = payload.layout
    db.commit()
    db.refresh(task)
    return task


@app.patch("/api/tasks/{task_id}/config", response_model=TaskRead)
def update_task_config(task_id: str, payload: TaskConfigUpdate, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "任务正在运行或排队，不能修改训练参数")
    if payload.epochs < task.completed_epochs:
        raise HTTPException(422, f"训练轮数不能小于已有进度 {task.completed_epochs}")
    unknown = set(payload.config) - EDITABLE_TRAINING_KEYS
    if unknown:
        raise HTTPException(422, f"包含不支持的参数：{', '.join(sorted(unknown))}")
    if len(payload.custom_config) > 30:
        raise HTTPException(422, "自定义参数最多添加 30 项")
    for key, value in payload.custom_config.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise HTTPException(422, f"自定义参数名不合法：{key}")
        if key in RESERVED_CUSTOM_KEYS or key in EDITABLE_TRAINING_KEYS:
            raise HTTPException(422, f"参数 {key} 由平台或预设参数管理，不能自定义覆盖")
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, str) and len(value) > 500:
            raise HTTPException(422, f"自定义参数 {key} 的值必须是布尔、数字或短文本")
    task.epochs = payload.epochs
    task.batch = payload.batch
    task.imgsz = payload.imgsz
    task.device = payload.device
    task.workers = payload.workers
    task.optimizer = payload.optimizer
    preserved = {key: value for key, value in (task.config or {}).items() if key not in EDITABLE_TRAINING_KEYS and key != "custom_args"}
    task.config = {**preserved, **payload.config, "custom_args": payload.custom_config}
    task.progress = min(100.0, round(task.completed_epochs / max(task.epochs, 1) * 100, 2))
    add_event(db, task.id, "config_updated", "训练参数已更新", payload=payload.model_dump())
    db.commit()
    db.refresh(task)
    return task


@app.patch("/api/tasks/{task_id}/resources", response_model=TaskRead)
def update_task_resources(task_id: str, payload: TaskResourceUpdate, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "任务正在运行或排队，不能更换模型或数据集")
    if payload.model_path is None and payload.data_path is None:
        raise HTTPException(422, "至少需要提供一个模型或数据集路径")
    try:
        if payload.model_path is not None:
            task.model_path = str(validate_input_path(payload.model_path, MODEL_ROOTS, "模型文件"))
        if payload.data_path is not None:
            task.data_path = str(validate_input_path(payload.data_path, DATASET_ROOTS, "数据集配置"))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    current_config = dict(task.config or {})
    issues = dict(current_config.get("path_issues") or {})
    if payload.model_path is not None:
        issues["model_missing"] = False
        current_config["model"] = task.model_path
    if payload.data_path is not None:
        issues["data_missing"] = False
        current_config["data"] = task.data_path
    current_config["path_issues"] = issues
    task.config = current_config
    changed = "模型" if payload.model_path is not None else "数据集"
    add_event(db, task.id, "resource_updated", f"{changed}路径已更新", payload=payload.model_dump(exclude_none=True))
    db.commit()
    db.refresh(task)
    return task


@app.get("/api/tasks/{task_id}/metrics", response_model=list[MetricRead])
def task_metrics(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    sync_metrics(db, task)
    db.commit()
    return db.scalars(select(MetricPoint).where(MetricPoint.task_id == task_id).order_by(MetricPoint.epoch)).all()


@app.get("/api/tasks/{task_id}/artifacts", response_model=list[ArtifactRead])
def task_artifacts(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    sync_artifacts(db, task)
    db.commit()
    return db.scalars(select(Artifact).where(Artifact.task_id == task_id).order_by(Artifact.kind, Artifact.name)).all()


@app.get("/api/tasks/{task_id}/events", response_model=list[EventRead])
def task_events(task_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    return db.scalars(select(TaskEvent).where(TaskEvent.task_id == task_id).order_by(TaskEvent.id.desc()).limit(100)).all()


@app.get("/api/tasks/{task_id}/logs")
def task_logs(task_id: str, tail: int = Query(default=400, ge=20, le=5000), db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if not task.log_path or not Path(task.log_path).exists():
        return {"lines": [], "path": task.log_path}
    lines = Path(task.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-tail:], "path": task.log_path}


@app.get("/api/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: int, db: Session = Depends(get_db)):
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "产物不存在")
    path = Path(artifact.path).resolve()
    task = get_task_or_404(db, artifact.task_id)
    if not task.run_dir or not path_is_within(path, [Path(task.run_dir)]):
        raise HTTPException(403, "产物路径不受信任")
    if not path.exists():
        raise HTTPException(404, "产物文件已经不存在")
    return FileResponse(path, filename=path.name, media_type=mimetypes.guess_type(path.name)[0])


@app.post("/api/import", status_code=202)
def import_runs():
    return {"imported": import_existing_runs()}


@app.get("/api/stream")
async def stream():
    async def events():
        while True:
            with SessionLocal() as db:
                tasks = db.scalars(select(TrainingTask).order_by(TrainingTask.updated_at.desc())).all()
                payload = [TaskRead.model_validate(task).model_dump(mode="json") for task in tasks]
            yield f"event: tasks\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            await asyncio.sleep(2)

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


frontend_dist = WEB_ROOT / "frontend" / "dist"
assets_dir = frontend_dist / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_app(full_path: str):
    """Serve built frontend files and fall back to index.html for client-side routes."""
    if not frontend_dist.exists():
        raise HTTPException(404, "前端尚未构建")
    candidate = (frontend_dist / full_path).resolve()
    if path_is_within(candidate, [frontend_dist]) and candidate.is_file():
        return FileResponse(candidate)
    return FileResponse(frontend_dist / "index.html")
