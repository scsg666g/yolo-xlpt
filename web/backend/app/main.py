from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .config import DATASET_ROOTS, MODEL_ROOTS, PLATFORM_INSTANCE_NAME, RUNS_ROOT, WEB_ROOT
from .compute_nodes import ensure_local_node, local_snapshot, router as compute_nodes_router
from .database import Base, SessionLocal, engine, get_db, migrate_database
from .dataset_uploads import router as dataset_upload_router
from .inference import router as inference_router, worker as inference_worker
from .models import Artifact, ComputeNode, MetricPoint, TaskEvent, TrainingRun, TrainingTask, utcnow
from .local_deletions import local_deletion_worker, queue_local_delete
from .parameter_catalog import load_parameter_catalog
from .remote_operations import matching_fetch, operation_file, queue_delete, queue_scan, router as remote_operations_router
from .schemas import ArtifactRead, EventRead, LayoutUpdate, MetricRead, RunRead, TaskConfigUpdate, TaskCreate, TaskOrderUpdate, TaskRead, TaskRename, TaskResourceUpdate
from .services import (
    add_event,
    backfill_training_runs,
    create_training_run,
    current_run,
    discover_files,
    gpu_inventory,
    import_existing_runs,
    migrate_training_logs,
    mirror_run_to_task,
    path_is_within,
    resume_checkpoint,
    scheduler,
    sync_metrics,
    validate_dataset_config,
    validate_input_path,
)

logger = logging.getLogger("yolo-platform.requests")
_task_snapshot = ""
_task_stream_subscribers: set[asyncio.Queue[str]] = set()

EDITABLE_TRAINING_KEYS = {
    "patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay",
    "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction",
    "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl",
    "save", "cls_remap", "verbose", "single_cls", "classes", "profile", "compile", "overlap_mask",
    "mask_ratio", "dropout", "val", "split", "save_json", "conf", "iou", "max_det", "plots", "lrf",
    "warmup_momentum", "warmup_bias_lr", "cls_pw", "pose", "kobj", "rle", "angle", "dlog", "dgrad",
    "dlam", "nbs", "bgr", "cutmix", "copy_paste_mode", "auto_augment", "erasing",
}


def editable_training_keys() -> set[str]:
    return set(load_parameter_catalog()) or EDITABLE_TRAINING_KEYS


RESERVED_CUSTOM_KEYS = {
    "model", "data", "project", "name", "device", "epochs", "batch", "imgsz", "workers", "optimizer",
    "task", "mode", "exist_ok", "save_dir", "resume", "source",
}


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _task_snapshot
    migrate_database()
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        ensure_local_node(db)
    import_existing_runs()
    backfill_training_runs()
    migrate_training_logs()
    scheduler.start()
    local_deletion_worker.start()
    inference_worker.start()
    _task_snapshot = await asyncio.to_thread(load_task_snapshot)
    snapshot_task = asyncio.create_task(task_snapshot_pump(), name="task-snapshot-pump")
    try:
        yield
    finally:
        snapshot_task.cancel()
        try:
            await snapshot_task
        except asyncio.CancelledError:
            pass
        local_deletion_worker.stop()
        inference_worker.stop()
        scheduler.stop()


app = FastAPI(title="YOLO 模型训练平台", version="0.1.0", lifespan=lifespan)
app.include_router(dataset_upload_router)
app.include_router(inference_router)
app.include_router(compute_nodes_router)
app.include_router(remote_operations_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_slow_requests(request: Request, call_next):
    started = time.perf_counter()
    try:
        return await call_next(request)
    finally:
        elapsed = time.perf_counter() - started
        if elapsed >= 1.0 and request.url.path.startswith("/api/"):
            logger.warning("slow_request method=%s path=%s duration_ms=%.1f", request.method,
                           request.url.path, elapsed * 1000)


def get_task_or_404(db: Session, task_id: str, *, include_deleted: bool = False) -> TrainingTask:
    task = db.get(TrainingTask, task_id)
    if not task or (task.deleted_at and not include_deleted):
        raise HTTPException(404, "训练任务不存在")
    return task


def get_run_or_404(db: Session, task_id: str, run_id: str) -> TrainingRun:
    run = db.scalar(select(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.task_id == task_id))
    if not run:
        raise HTTPException(404, "训练运行不存在")
    return run


@app.get("/api/health")
def health():
    started = time.perf_counter()
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1").scalar_one()
    except Exception as exc:
        logger.warning("health_database_error error=%s", type(exc).__name__)
        raise HTTPException(503, "数据库暂不可用") from exc
    return {"status": "ok", "instance": PLATFORM_INSTANCE_NAME,
            "scheduler": "running" if scheduler._thread and scheduler._thread.is_alive() else "stopped",
            "database": "ok", "database_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "runs_root": str(RUNS_ROOT)}


@app.get("/api/resources")
def resources():
    return {
        "gpus": gpu_inventory(),
        "models": discover_files(MODEL_ROOTS, {".pt", ".pth", ".onnx", ".yaml", ".yml"}),
        "datasets": discover_files(DATASET_ROOTS, {".yaml", ".yml"}),
    }


@app.get("/api/compute")
def compute_resources():
    return local_snapshot()


@app.get("/api/training-parameters")
def training_parameters():
    catalog = load_parameter_catalog()
    return {
        "source": "default.yaml" if catalog else "fallback",
        "parameters": list(catalog.values()),
    }


@app.get("/api/tasks", response_model=list[TaskRead])
def list_tasks(status: str | None = Query(default=None), include_deleted: bool = Query(default=False),
               deleted_only: bool = Query(default=False), db: Session = Depends(get_db)):
    query = select(TrainingTask).order_by(TrainingTask.display_order, TrainingTask.created_at.desc(), TrainingTask.id)
    if deleted_only:
        query = query.where(TrainingTask.deleted_at.is_not(None))
    elif not include_deleted:
        query = query.where(TrainingTask.deleted_at.is_(None))
    if status:
        query = query.where(TrainingTask.status == status)
    # Listing tasks must stay a pure read: disk scans and metric writes belong in
    # the scheduler/sync worker, never in the request serving the dashboard.
    return db.scalars(query).all()


@app.patch("/api/tasks/order", status_code=204)
def update_task_order(payload: TaskOrderUpdate, db: Session = Depends(get_db)):
    requested = list(dict.fromkeys(payload.task_ids))
    if len(requested) != len(payload.task_ids):
        raise HTTPException(422, "任务排序列表不能包含重复任务")
    tasks = db.scalars(select(TrainingTask).where(TrainingTask.deleted_at.is_(None))).all()
    by_id = {task.id: task for task in tasks}
    if set(requested) != set(by_id):
        raise HTTPException(409, "任务列表已更新，请刷新页面后重新排序")
    for index, task_id in enumerate(requested, start=1):
        by_id[task_id].display_order = index * 1000
    db.commit()


@app.post("/api/tasks", response_model=TaskRead, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    node = db.get(ComputeNode, payload.execution_node_id) if payload.execution_node_id else None
    if payload.execution_node_id:
        if not node or node.kind != "agent" or not node.enabled:
            raise HTTPException(422, "请选择已启用的远程 Agent 服务器")
        model_path, data_path = payload.model_path.strip(), payload.data_path.strip()
        output_root = (payload.remote_output_root or "").strip()
        if not Path(model_path).is_absolute() or not Path(data_path).is_absolute() or not Path(output_root).is_absolute():
            raise HTTPException(422, "远程模型、数据 YAML 和输出目录都必须是远程服务器上的绝对路径")
        if any("\x00" in value for value in (model_path, data_path, output_root)):
            raise HTTPException(422, "路径不能包含空字符")
    else:
        try:
            model_path = str(validate_input_path(payload.model_path, MODEL_ROOTS, "模型文件"))
            data_path = str(validate_dataset_config(payload.data_path))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    task_id = uuid.uuid4().hex[:16]
    config = payload.model_dump(exclude={"name", "task_type", "model_path", "data_path", "execution_node_id", "remote_output_root", "epochs", "batch", "imgsz", "device", "workers", "optimizer", "advanced"})
    if node:
        config["remote_output_root"] = payload.remote_output_root.strip()
    supported_keys = editable_training_keys()
    config.update({key: value for key, value in payload.advanced.items() if key in supported_keys})
    task = TrainingTask(
        id=task_id,
        name=payload.name.strip(),
        task_type=payload.task_type,
        execution_node_id=node.id if node else None,
        model_path=model_path,
        data_path=data_path,
        epochs=payload.epochs,
        batch=payload.batch,
        imgsz=payload.imgsz,
        device=payload.device,
        workers=payload.workers,
        optimizer=payload.optimizer,
        config=config,
        display_order=(db.scalar(select(func.coalesce(func.max(TrainingTask.display_order), 0))) or 0) + 1000,
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
    db.commit()
    return task


@app.post("/api/tasks/{task_id}/start", response_model=TaskRead)
def start_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status != "draft":
        raise HTTPException(409, f"当前状态 {task.status} 不能启动")
    if not task.execution_node_id:
        if not Path(task.model_path).exists() or not Path(task.data_path).exists():
            raise HTTPException(409, "模型或数据集路径已经失效，请克隆任务并选择有效路径")
        try:
            validate_dataset_config(task.data_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    run = create_training_run(db, task, "fresh")
    add_event(db, task.id, "queued", f"Attempt {run.attempt_number:03d} 已加入训练队列", run_id=run.id)
    db.commit()
    db.refresh(task)
    return task


@app.post("/api/tasks/{task_id}/resume", response_model=TaskRead)
def resume_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status not in {"stopped", "interrupted"}:
        raise HTTPException(409, f"当前状态 {task.status} 不能继续训练")
    if task.execution_node_id:
        previous = current_run(db, task)
        if previous and (task.epochs != previous.parameter_snapshot.get("epochs") or task.data_path != previous.data_path):
            raise HTTPException(409, "断点续训沿用检查点的总轮数和数据集，请恢复原配置或选择重新训练")
        checkpoint_artifact = db.scalar(select(Artifact).where(
            Artifact.run_id == previous.id, Artifact.name == "weights/last.pt", Artifact.size > 0,
        )) if previous else None
        checkpoint = Path(checkpoint_artifact.path) if checkpoint_artifact else None
        if previous:
            queue_scan(db, task, previous)
            db.commit()
    else:
        checkpoint = resume_checkpoint(task)
    if not checkpoint:
        raise HTTPException(409, "尚未发现可用的 last.pt 检查点；远程任务请等待 Agent 同步产物后重试")
    if not task.execution_node_id and not Path(task.data_path).exists():
        raise HTTPException(409, "数据集路径已经失效，无法继续训练")
    if not task.execution_node_id:
        try:
            validate_dataset_config(task.data_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    if task.epochs <= task.completed_epochs:
        if task.execution_node_id:
            raise HTTPException(409, f"断点已经完成全部 {task.completed_epochs} 个 Epoch，请选择重新训练")
        raise HTTPException(409, f"断点已经完成 {task.completed_epochs} 个 Epoch，请增加训练轮数或选择重新训练")
    run = create_training_run(db, task, "resume", checkpoint)
    add_event(
        db,
        task.id,
        "resume_queued",
        f"Attempt {run.attempt_number:03d} 已加入断点续训队列",
        payload={"checkpoint": str(checkpoint), "completed_epochs": task.completed_epochs},
        run_id=run.id,
    )
    db.commit()
    db.refresh(task)
    return task


@app.post("/api/tasks/{task_id}/retry", response_model=TaskRead)
def retry_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status not in {"failed", "stopped", "interrupted", "completed"}:
        raise HTTPException(409, f"当前状态 {task.status} 不能重新训练")
    if not task.execution_node_id:
        try:
            validate_input_path(task.model_path, MODEL_ROOTS, "模型文件")
            validate_dataset_config(task.data_path)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    run = create_training_run(db, task, "retry")
    add_event(db, task.id, "retry_queued", f"Attempt {run.attempt_number:03d} 已加入重新训练队列", run_id=run.id)
    db.commit()
    db.refresh(task)
    return task


@app.post("/api/tasks/{task_id}/stop", response_model=TaskRead)
def stop_task(task_id: str, request: Request, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.execution_node_id:
        run = current_run(db, task)
        if not run or run.status not in {"queued", "starting", "running", "stopping"}:
            raise HTTPException(409, "该远程任务当前无法停止")
        if run.status == "queued":
            run.status = "stopped"
            run.stop_requested = True
            run.error_message = "平台取消排队，训练尚未启动"
            run.finished_at = utcnow()
        else:
            run.stop_requested = True
            run.status = "stopping"
        mirror_run_to_task(task, run)
        source_ip = request.client.host if request.client else "unknown"
        message = "已取消远程训练排队" if run.status == "stopped" else "已向远程 Agent 请求停止"
        add_event(db, task.id, "stop_requested", f"{message}；请求来源：{source_ip}",
                  payload={"source_ip": source_ip, "method": request.method, "path": request.url.path,
                           "requested_at": utcnow().isoformat(), "run_id": run.id}, run_id=run.id)
        db.commit()
        db.refresh(task)
        return task
    if not scheduler.stop_task(task_id):
        raise HTTPException(409, "该任务当前无法停止")
    with SessionLocal() as refreshed_db:
        return get_task_or_404(refreshed_db, task_id)


def clone_name_parts(name: str) -> tuple[str, int]:
    """Strip generated clone suffixes, including names from older versions."""
    number = 0
    while match := re.search(r" - 副本(?: ([1-9][0-9]*))?$", name):
        number = max(number, int(match.group(1) or 1))
        name = name[:match.start()]
    return name or "未命名任务", number


def next_clone_name(db: Session, source_name: str) -> str:
    base, number = clone_name_parts(source_name)
    existing_names = set(db.scalars(select(TrainingTask.name)))
    for name in existing_names:
        existing_base, existing_number = clone_name_parts(name)
        suffix = re.search(r" - 副本(?: ([1-9][0-9]*))?$", name)
        if existing_base == base or (
            suffix and existing_base == base[:160 - len(suffix.group())]
        ):
            number = max(number, existing_number)

    while True:
        number += 1
        suffix = f" - 副本 {number}"
        candidate = f"{base[:160 - len(suffix)]}{suffix}"
        if candidate not in existing_names:
            return candidate


@app.post("/api/tasks/{task_id}/clone", response_model=TaskRead, status_code=201)
def clone_task(task_id: str, db: Session = Depends(get_db)):
    source = get_task_or_404(db, task_id)
    clone = TrainingTask(
        id=uuid.uuid4().hex[:16], name=next_clone_name(db, source.name), task_type=source.task_type,
        execution_node_id=source.execution_node_id,
        model_path=source.model_path, data_path=source.data_path, epochs=source.epochs, batch=source.batch,
        imgsz=source.imgsz, device=source.device, workers=source.workers, optimizer=source.optimizer,
        config={key: value for key, value in source.config.items() if key != "_resume_checkpoint"}, canvas_layout=source.canvas_layout.copy(),
        display_order=(db.scalar(select(func.coalesce(func.max(TrainingTask.display_order), 0))) or 0) + 1000,
    )
    db.add(clone)
    db.flush()
    add_event(db, clone.id, "cloned", f"从任务 {source.name} 克隆", payload={"source_id": source.id})
    db.commit()
    db.refresh(clone)
    return clone


@app.patch("/api/tasks/{task_id}/name", response_model=TaskRead)
def rename_task(task_id: str, payload: TaskRename, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.name != payload.name:
        old_name = task.name
        task.name = payload.name
        add_event(
            db, task.id, "renamed", f"任务已重命名：{old_name} → {task.name}",
            payload={"old_name": old_name, "new_name": task.name},
        )
        db.commit()
        db.refresh(task)
    return task


@app.post("/api/tasks/{task_id}/trash", status_code=204)
@app.delete("/api/tasks/{task_id}", status_code=204, deprecated=True)
def delete_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "运行中或排队中的任务不能删除")
    task.deleted_at = utcnow()
    task.deleted_reason = "soft_delete"
    add_event(db, task.id, "soft_deleted", "任务已移入回收站")
    db.commit()


@app.post("/api/tasks/{task_id}/restore", response_model=TaskRead)
def restore_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id, include_deleted=True)
    if not task.deleted_at:
        raise HTTPException(409, "任务当前不在回收站")
    if task.deleted_reason == "remote_delete_pending":
        raise HTTPException(409, "远程文件清理已提交，不能恢复；请等待 Agent 返回结果")
    if task.deleted_reason == "remote_delete_failed":
        raise HTTPException(409, "远程清理可能已删除部分文件，请重试清理；该任务不能恢复")
    if task.deleted_reason == "local_delete_pending":
        raise HTTPException(409, "本机文件清理已提交，不能恢复；请等待清理结果")
    if task.deleted_reason == "local_delete_failed":
        raise HTTPException(409, "本机文件可能已部分清理，请重试清理；该任务不能恢复")
    task.deleted_at = None
    task.deleted_reason = None
    add_event(db, task.id, "restored", "任务已从回收站恢复")
    db.commit()
    db.refresh(task)
    return task


@app.delete("/api/tasks/{task_id}/permanent", status_code=202)
def permanently_delete_task(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id, include_deleted=True)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "运行中或排队中的任务不能永久删除")
    if task.execution_node_id:
        queue_delete(db, task)
        db.commit()
        return JSONResponse(status_code=202, content={"status": "pending", "message": "等待远程 Agent 清理文件；确认后删除任务记录"})
    queue_local_delete(db, task)
    db.commit()
    return JSONResponse(status_code=202, content={"status": "pending", "message": "已提交本机文件清理；可在回收站查看进度"})


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
    supported_keys = editable_training_keys()
    unknown = set(payload.config) - supported_keys
    if unknown:
        raise HTTPException(422, f"包含不支持的参数：{', '.join(sorted(unknown))}")
    if len(payload.custom_config) > 30:
        raise HTTPException(422, "自定义参数最多添加 30 项")
    for key, value in payload.custom_config.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise HTTPException(422, f"自定义参数名不合法：{key}")
        if key in RESERVED_CUSTOM_KEYS or key in supported_keys:
            raise HTTPException(422, f"参数 {key} 由平台或预设参数管理，不能自定义覆盖")
        if not isinstance(value, (str, int, float, bool)) or isinstance(value, str) and len(value) > 500:
            raise HTTPException(422, f"自定义参数 {key} 的值必须是布尔、数字或短文本")
    task.epochs = payload.epochs
    task.batch = payload.batch
    task.imgsz = payload.imgsz
    task.device = payload.device
    task.workers = payload.workers
    task.optimizer = payload.optimizer
    preserved = {key: value for key, value in (task.config or {}).items() if key not in supported_keys and key != "custom_args"}
    task.config = {**preserved, **payload.config, "custom_args": payload.custom_config}
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
            if task.execution_node_id:
                if not Path(payload.model_path).is_absolute() or "\x00" in payload.model_path:
                    raise ValueError("远程模型必须是绝对路径")
                task.model_path = payload.model_path.strip()
            else:
                task.model_path = str(validate_input_path(payload.model_path, MODEL_ROOTS, "模型文件"))
        if payload.data_path is not None:
            if task.execution_node_id:
                if not Path(payload.data_path).is_absolute() or "\x00" in payload.data_path:
                    raise ValueError("远程数据集 YAML 必须是绝对路径")
                task.data_path = payload.data_path.strip()
            else:
                task.data_path = str(validate_dataset_config(payload.data_path))
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
    run = current_run(db, task)
    if not run:
        return []
    return db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id).order_by(MetricPoint.epoch)).all()


@app.get("/api/tasks/{task_id}/artifacts", response_model=list[ArtifactRead])
def task_artifacts(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    run = current_run(db, task)
    if not run:
        return []
    if run.execution_node_id:
        queue_scan(db, task, run)
        db.commit()
    return db.scalars(select(Artifact).where(Artifact.run_id == run.id).order_by(Artifact.kind, Artifact.name)).all()


@app.get("/api/tasks/{task_id}/events", response_model=list[EventRead])
def task_events(task_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    run = current_run(db, task)
    if not run:
        return []
    return db.scalars(select(TaskEvent).where(TaskEvent.run_id == run.id).order_by(TaskEvent.id.desc()).limit(100)).all()


@app.get("/api/tasks/{task_id}/logs")
def task_logs(task_id: str, tail: int = Query(default=400, ge=20, le=5000), db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    run = current_run(db, task)
    if run and run.execution_node_id:
        return {"lines": (run.remote_log_tail or [])[-tail:], "path": run.log_path}
    if not run or not run.log_path or not Path(run.log_path).exists():
        return {"lines": [], "path": run.log_path if run else None}
    lines = Path(run.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-tail:], "path": run.log_path}


@app.get("/api/tasks/{task_id}/runs", response_model=list[RunRead])
def task_runs(task_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    return db.scalars(select(TrainingRun).where(TrainingRun.task_id == task_id).order_by(TrainingRun.attempt_number.desc())).all()


@app.get("/api/tasks/{task_id}/runs/{run_id}", response_model=RunRead)
def task_run(task_id: str, run_id: str, db: Session = Depends(get_db)):
    return get_run_or_404(db, task_id, run_id)


@app.get("/api/tasks/{task_id}/runs/{run_id}/metrics", response_model=list[MetricRead])
def run_metrics(task_id: str, run_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    return db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id).order_by(MetricPoint.epoch)).all()


@app.get("/api/tasks/{task_id}/runs/{run_id}/artifacts", response_model=list[ArtifactRead])
def run_artifacts(task_id: str, run_id: str, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    if run.execution_node_id:
        queue_scan(db, task, run)
        db.commit()
    return db.scalars(select(Artifact).where(Artifact.run_id == run.id).order_by(Artifact.kind, Artifact.name)).all()


@app.get("/api/tasks/{task_id}/runs/{run_id}/events", response_model=list[EventRead])
def run_events(task_id: str, run_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    return db.scalars(select(TaskEvent).where(TaskEvent.run_id == run.id).order_by(TaskEvent.id.desc()).limit(100)).all()


@app.get("/api/tasks/{task_id}/runs/{run_id}/logs")
def run_logs(task_id: str, run_id: str, tail: int = Query(default=400, ge=20, le=5000), db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    if run.execution_node_id:
        return {"lines": (run.remote_log_tail or [])[-tail:], "path": run.log_path}
    if not run.log_path or not Path(run.log_path).exists():
        return {"lines": [], "path": run.log_path}
    lines = Path(run.log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lines": lines[-tail:], "path": run.log_path}


@app.get("/api/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: int, db: Session = Depends(get_db)):
    artifact = db.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "产物不存在")
    task = get_task_or_404(db, artifact.task_id)
    run = db.get(TrainingRun, artifact.run_id) if artifact.run_id else current_run(db, task)
    if run and run.execution_node_id:
        operation = matching_fetch(db, artifact)
        if not operation or operation.status != "succeeded" or not operation_file(operation).is_file():
            raise HTTPException(409, "远程文件尚未传输完成，请先准备下载")
        return FileResponse(operation_file(operation), filename=Path(artifact.name).name,
                            media_type=mimetypes.guess_type(artifact.name)[0])
    path = Path(artifact.path).resolve()
    if not run or not run.run_dir or not path_is_within(path, [Path(run.run_dir)]):
        raise HTTPException(403, "产物路径不受信任")
    if not path.exists():
        raise HTTPException(404, "产物文件已经不存在")
    return FileResponse(path, filename=path.name, media_type=mimetypes.guess_type(path.name)[0])


@app.post("/api/import", status_code=202)
def import_runs():
    return {"imported": import_existing_runs()}


def load_task_snapshot() -> str:
    with SessionLocal() as db:
        tasks = db.scalars(select(TrainingTask).where(TrainingTask.deleted_at.is_(None)).order_by(
            TrainingTask.display_order, TrainingTask.created_at.desc(), TrainingTask.id
        )).all()
        payload = [TaskRead.model_validate(task).model_dump(mode="json") for task in tasks]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


async def task_snapshot_pump() -> None:
    """Read task state once per interval and broadcast only changed snapshots."""
    global _task_snapshot
    while True:
        try:
            # There is no reason to poll SQLite while nobody has an open stream.
            # A new stream performs one fresh read below before subscribing.
            if not _task_stream_subscribers:
                await asyncio.sleep(5)
                continue
            snapshot = await asyncio.to_thread(load_task_snapshot)
            if snapshot != _task_snapshot:
                _task_snapshot = snapshot
                for subscriber in tuple(_task_stream_subscribers):
                    if subscriber.full():
                        try:
                            subscriber.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        subscriber.put_nowait(snapshot)
                    except asyncio.QueueFull:
                        pass
        except Exception:
            logger.exception("task_snapshot_refresh_failed")
        await asyncio.sleep(2)


@app.get("/api/stream")
async def stream():
    async def events():
        subscriber: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
        _task_stream_subscribers.add(subscriber)
        last_sent = ""
        try:
            # Always load once on connect so a stream opened after an idle period
            # cannot receive a stale in-memory snapshot.
            snapshot = await asyncio.to_thread(load_task_snapshot)
            global _task_snapshot
            _task_snapshot = snapshot
            last_sent = snapshot
            yield f"event: tasks\ndata: {snapshot}\n\n"
            while True:
                try:
                    snapshot = await asyncio.wait_for(subscriber.get(), timeout=15)
                except TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                if snapshot != last_sent:
                    last_sent = snapshot
                    yield f"event: tasks\ndata: {snapshot}\n\n"
        finally:
            _task_stream_subscribers.discard(subscriber)

    return StreamingResponse(events(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


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
