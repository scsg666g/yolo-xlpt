from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from .config import DATASET_ROOTS, MODEL_ROOTS, PLATFORM_INSTANCE_NAME, PLATFORM_VERSION, RUNS_ROOT, WEB_ROOT
from .compute_nodes import (
    as_utc,
    ensure_local_node,
    local_snapshot,
    remote_output_root_unavailable_reason,
    remote_trace_root,
    remote_training_unavailable_reason,
    router as compute_nodes_router,
)
from .database import Base, SessionLocal, engine, get_db, migrate_database
from .gpu_reservations import acquire_node_claim_lock, release_gpu_reservations
from .dataset_uploads import (
    cleanup_managed_dataset_workdirs,
    managed_dataset_mutation,
    management_router as dataset_management_router,
    router as dataset_upload_router,
)
from .inference import (
    cancel_task_tests,
    model_test_lifecycle,
    router as inference_router,
    stage_task_test_cancellation,
    worker as inference_worker,
)
from .models import Artifact, ComputeNode, MetricPoint, RemoteOperation, TaskEvent, TaskGroup, TrainingRun, TrainingTask, utcnow
from .local_deletions import local_deletion_worker, queue_local_delete, queue_local_run_delete
from .model_assets import router as model_assets_router
from .parameter_catalog import is_finite_number, load_parameter_catalog, validate_parameter_values
from .remote_operations import (
    active_remote_delete,
    matching_fetch,
    operation_file,
    queue_attempt_delete,
    queue_delete,
    queue_scan,
    router as remote_operations_router,
)
from .schemas import ArtifactRead, EffectiveBestWeightRead, EventRead, LayoutUpdate, MetricRead, RemoteInterruptConfirmation, RunRead, TaskConfigUpdate, TaskCreate, TaskGroupAssignment, TaskGroupCreate, TaskGroupCreateFromTasks, TaskGroupRead, TaskGroupRename, TaskOrderUpdate, TaskRead, TaskRename, TaskResourceUpdate
from .services import (
    add_event,
    backfill_training_runs,
    create_training_run,
    current_run,
    discover_files,
    effective_best_weight,
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
from .run_deletions import active_attempt_delete_operation, attempt_deletion_state
from .deletion_jobs import deletion_preview, list_deletion_jobs, reject_local_task_cleanup

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
    try:
        cleanup_managed_dataset_workdirs()
    except OSError as exc:
        logger.warning("managed_dataset_workdir_cleanup_failed error=%s", type(exc).__name__)
    with managed_dataset_mutation():
        import_existing_runs()
    backfill_training_runs()
    migrate_training_logs()
    scheduler.start()
    inference_worker.start()
    local_deletion_worker.start()
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


app = FastAPI(title="YOLO 模型训练平台", version=PLATFORM_VERSION, lifespan=lifespan)
app.include_router(dataset_upload_router)
app.include_router(dataset_management_router)
app.include_router(inference_router)
app.include_router(model_assets_router)
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


def serialize_managed_dataset_references():
    with managed_dataset_mutation():
        yield


def get_task_or_404(db: Session, task_id: str, *, include_deleted: bool = False) -> TrainingTask:
    task = db.get(TrainingTask, task_id)
    if not task or (task.deleted_at and not include_deleted):
        raise HTTPException(404, "训练任务不存在")
    return task


def get_task_group_or_404(db: Session, group_id: str) -> TaskGroup:
    group = db.get(TaskGroup, group_id)
    if not group:
        raise HTTPException(404, "训练分组不存在")
    return group


def reject_duplicate_group_name(db: Session, name: str, *, exclude_id: str | None = None) -> None:
    query = select(TaskGroup).where(func.lower(TaskGroup.name) == name.casefold())
    if exclude_id:
        query = query.where(TaskGroup.id != exclude_id)
    if db.scalar(query):
        raise HTTPException(409, "训练分组名称已存在")


def next_task_group_display_order(db: Session) -> int:
    first_order = db.scalar(select(func.min(TaskGroup.display_order)))
    return (first_order if first_order is not None else 0) - 1000


def get_run_or_404(db: Session, task_id: str, run_id: str) -> TrainingRun:
    run = db.scalar(select(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.task_id == task_id))
    if not run:
        raise HTTPException(404, "训练运行不存在")
    return run


def serialize_run(db: Session, run: TrainingRun) -> RunRead:
    """Add durable deletion operation state without changing the run table."""
    status, error, operation = attempt_deletion_state(db, run)
    return RunRead.model_validate(run).model_copy(update={
        "deletion_status": status,
        "deletion_error": error,
        "deletion_requested_at": operation.created_at if operation else None,
    })


def reject_active_remote_delete(db: Session, node_id: str) -> None:
    if active_remote_delete(db, node_id):
        raise HTTPException(
            409,
            "该远程服务器正在清理任务文件，暂不能新建、克隆或更换模型和数据集；请等待清理完成后重试",
        )


def reject_unsupported_remote_parameters(node: ComputeNode | None, config: dict, status_code: int = 422) -> None:
    capability = (node.snapshot or {}).get("training") if node else None
    if not isinstance(capability, dict) or "supported_parameters" not in capability:
        return
    reported = capability.get("supported_parameters")
    if not isinstance(reported, list):
        return
    requested = {
        key for key, value in config.items()
        if key in editable_training_keys() and value is not None
    }
    unsupported = requested - {key for key in reported if isinstance(key, str)}
    if unsupported:
        raise HTTPException(
            status_code,
            f"远程 Agent 不支持参数 {', '.join(sorted(unsupported))}，请升级 Agent 或移除该参数",
        )


def reject_unavailable_remote_training(node: ComputeNode | None, status_code: int) -> None:
    reason = remote_training_unavailable_reason(node)
    if reason:
        raise HTTPException(status_code, f"远程 Agent 当前不可训练：{reason}")


def reject_invalid_remote_output_root(node: ComputeNode | None, config: dict, status_code: int = 422) -> None:
    reason = remote_output_root_unavailable_reason(node, (config or {}).get("remote_output_root"))
    if reason:
        raise HTTPException(status_code, reason)


@app.get("/api/health")
def health():
    started = time.perf_counter()
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("SELECT 1").scalar_one()
    except Exception as exc:
        logger.warning("health_database_error error=%s", type(exc).__name__)
        raise HTTPException(503, "数据库暂不可用") from exc
    scheduler_running = bool(
        getattr(scheduler, "is_alive", False)
        or (getattr(scheduler, "_thread", None) and scheduler._thread.is_alive())
    )
    scheduler_error = getattr(scheduler, "active_error", None)
    scheduler_status = "error" if scheduler_running and scheduler_error else "running" if scheduler_running else "stopped"
    scheduler_last_error = getattr(scheduler, "last_error", None)
    scheduler_last_error_at = getattr(scheduler, "last_error_at", None)
    if hasattr(scheduler_last_error_at, "isoformat"):
        scheduler_last_error_at = scheduler_last_error_at.isoformat()
    deletion_status = "running" if local_deletion_worker.is_alive else "stopped"
    inference_status = "running" if inference_worker.is_alive else "stopped"
    return {"status": "ok" if all(status == "running" for status in (
                scheduler_status, deletion_status, inference_status,
            )) else "degraded",
            "instance": PLATFORM_INSTANCE_NAME,
            "version": PLATFORM_VERSION,
            "scheduler": scheduler_status,
            "scheduler_error": scheduler_error,
            "scheduler_last_error": scheduler_last_error,
            "scheduler_last_error_at": scheduler_last_error_at,
            "local_deletion_worker": deletion_status,
            "inference_worker": inference_status,
            "database": "ok", "database_latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "runs_root": str(RUNS_ROOT)}


@app.get("/api/health/storage")
def storage_health():
    """Inspect the platform output directory without creating test files."""
    try:
        if not RUNS_ROOT.exists():
            return {"path": str(RUNS_ROOT), "status": "missing", "free_bytes": None, "total_bytes": None}
        if not RUNS_ROOT.is_dir():
            return {"path": str(RUNS_ROOT), "status": "not_directory", "free_bytes": None, "total_bytes": None}
        usage = shutil.disk_usage(RUNS_ROOT)
        status = "writable" if os.access(RUNS_ROOT, os.W_OK) else "read_only"
        return {"path": str(RUNS_ROOT), "status": status, "free_bytes": usage.free, "total_bytes": usage.total}
    except OSError:
        return {"path": str(RUNS_ROOT), "status": "unavailable", "free_bytes": None, "total_bytes": None}


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


@app.get("/api/task-groups", response_model=list[TaskGroupRead])
def list_task_groups(db: Session = Depends(get_db)):
    return db.scalars(select(TaskGroup).order_by(TaskGroup.display_order, TaskGroup.created_at, TaskGroup.id)).all()


@app.post("/api/task-groups", response_model=TaskGroupRead, status_code=201)
def create_task_group(payload: TaskGroupCreate, db: Session = Depends(get_db)):
    reject_duplicate_group_name(db, payload.name)
    group = TaskGroup(
        id=uuid.uuid4().hex[:16],
        name=payload.name,
        display_order=next_task_group_display_order(db),
    )
    db.add(group)
    db.commit()
    db.refresh(group)
    return group


@app.post("/api/task-groups/from-tasks", response_model=TaskGroupRead, status_code=201)
def create_task_group_from_tasks(payload: TaskGroupCreateFromTasks, db: Session = Depends(get_db)):
    reject_duplicate_group_name(db, payload.name)
    tasks = list(db.scalars(
        select(TrainingTask).where(TrainingTask.id.in_(payload.task_ids))
    ).all())
    tasks_by_id = {task.id: task for task in tasks}
    if len(tasks_by_id) != len(payload.task_ids):
        raise HTTPException(404, "部分训练任务已不存在，请刷新后重试")
    if any(task.deleted_at for task in tasks):
        raise HTTPException(409, "已删除的训练任务不能加入分组")
    if any(task.group_id for task in tasks):
        raise HTTPException(409, "部分训练任务已加入其他分组，请刷新后重试")

    group = TaskGroup(
        id=uuid.uuid4().hex[:16],
        name=payload.name,
        display_order=next_task_group_display_order(db),
    )
    db.add(group)
    db.flush()
    for task_id in payload.task_ids:
        task = tasks_by_id[task_id]
        task.group_id = group.id
        add_event(
            db,
            task.id,
            "group_changed",
            f"训练分组已调整为 {group.name}",
            payload={
                "old_group_id": None,
                "old_group_name": None,
                "group_id": group.id,
                "group_name": group.name,
            },
        )
    db.commit()
    db.refresh(group)
    return group


@app.patch("/api/task-groups/{group_id}", response_model=TaskGroupRead)
def rename_task_group(group_id: str, payload: TaskGroupRename, db: Session = Depends(get_db)):
    group = get_task_group_or_404(db, group_id)
    if group.name != payload.name:
        reject_duplicate_group_name(db, payload.name, exclude_id=group.id)
        group.name = payload.name
        db.commit()
        db.refresh(group)
    return group


@app.delete("/api/task-groups/{group_id}", status_code=204)
def delete_task_group(group_id: str, db: Session = Depends(get_db)):
    group = get_task_group_or_404(db, group_id)
    db.execute(update(TrainingTask).where(TrainingTask.group_id == group.id).values(group_id=None))
    db.delete(group)
    db.commit()


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


@app.patch("/api/tasks/{task_id}/group", response_model=TaskRead)
def assign_task_group(task_id: str, payload: TaskGroupAssignment, db: Session = Depends(get_db)):
    task = get_task_or_404(db, task_id)
    group = get_task_group_or_404(db, payload.group_id) if payload.group_id else None
    if task.group_id != payload.group_id:
        old_group = db.get(TaskGroup, task.group_id) if task.group_id else None
        task.group_id = payload.group_id
        add_event(
            db,
            task.id,
            "group_changed",
            f"训练分组已调整为 {group.name if group else '未分组'}",
            payload={
                "old_group_id": old_group.id if old_group else None,
                "old_group_name": old_group.name if old_group else None,
                "group_id": group.id if group else None,
                "group_name": group.name if group else None,
            },
        )
        db.commit()
        db.refresh(task)
    return task


@app.post("/api/tasks", response_model=TaskRead, status_code=201)
def create_task(payload: TaskCreate, db: Session = Depends(get_db)):
    group = get_task_group_or_404(db, payload.group_id) if payload.group_id else None
    node = db.get(ComputeNode, payload.execution_node_id) if payload.execution_node_id else None
    if payload.execution_node_id:
        reject_unavailable_remote_training(node, 422)
    guard = nullcontext() if node else managed_dataset_mutation()
    with guard:
        if node:
            acquire_node_claim_lock(db, node.id)
            db.refresh(node)
            reject_unavailable_remote_training(node, 422)
            model_path, data_path = payload.model_path.strip(), payload.data_path.strip()
            output_root = (payload.remote_output_root or "").strip()
            if not Path(model_path).is_absolute() or not Path(data_path).is_absolute() or not Path(output_root).is_absolute():
                raise HTTPException(422, "远程模型、数据 YAML 和输出目录都必须是远程服务器上的绝对路径")
            if any("\x00" in value for value in (model_path, data_path, output_root)):
                raise HTTPException(422, "路径不能包含空字符")
            reject_active_remote_delete(db, node.id)
        else:
            reject_local_task_cleanup(db)
            try:
                model_path = str(validate_input_path(payload.model_path, MODEL_ROOTS, "模型文件"))
                data_path = str(validate_dataset_config(payload.data_path))
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        task_id = uuid.uuid4().hex[:16]
        config = payload.model_dump(exclude={"name", "task_type", "group_id", "model_path", "data_path", "execution_node_id", "remote_output_root", "epochs", "batch", "imgsz", "device", "workers", "optimizer", "advanced"})
        if node:
            config["remote_output_root"] = payload.remote_output_root.strip()
        supported_keys = editable_training_keys()
        unknown = set(payload.advanced) - supported_keys
        if unknown:
            raise HTTPException(422, f"包含不支持的参数：{', '.join(sorted(unknown))}")
        try:
            validate_parameter_values(payload.advanced)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        config.update(payload.advanced)
        if node:
            reject_invalid_remote_output_root(node, config)
            reject_unsupported_remote_parameters(node, config)
        task = TrainingTask(
            id=task_id,
            name=payload.name.strip(),
            task_type=payload.task_type,
            group_id=group.id if group else None,
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
    if task.execution_node_id:
        node = db.get(ComputeNode, task.execution_node_id)
        reject_unavailable_remote_training(node, 409)
        reject_invalid_remote_output_root(node, task.config, 409)
        reject_unsupported_remote_parameters(node, task.config, 409)
    else:
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
    previous = current_run(db, task)
    previous_epochs = (previous.parameter_snapshot or {}).get("epochs") if previous else None
    if previous and (
        (previous_epochs is not None and task.epochs != previous_epochs)
        or task.model_path != previous.model_path
        or task.data_path != previous.data_path
    ):
        raise HTTPException(409, "断点续训沿用上一 Attempt 的总轮数、基础模型和数据集，请恢复原配置或选择重新训练")
    if task.execution_node_id:
        node = db.get(ComputeNode, task.execution_node_id)
        reject_unavailable_remote_training(node, 409)
        reject_invalid_remote_output_root(node, task.config, 409)
        reject_unsupported_remote_parameters(node, task.config, 409)
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
        raise HTTPException(409, f"断点已经完成全部 {task.completed_epochs} 个 Epoch，请选择重新训练")
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
    if task.execution_node_id:
        node = db.get(ComputeNode, task.execution_node_id)
        reject_unavailable_remote_training(node, 409)
        reject_invalid_remote_output_root(node, task.config, 409)
        reject_unsupported_remote_parameters(node, task.config, 409)
    else:
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
        if not run or (run.status not in {"queued", "starting", "running", "stopping"}
                       and not (run.status == "stopped" and run.stop_requested)):
            raise HTTPException(409, "该远程任务当前无法停止")
        requested_at = utcnow()
        transition = None
        cancelled = db.execute(update(TrainingRun).where(
            TrainingRun.id == run.id,
            TrainingRun.status == "queued",
        ).values(
            status="stopped", stop_requested=True, pid=None,
            error_message="平台取消排队，训练尚未启动", finished_at=requested_at,
        ))
        if cancelled.rowcount == 1:
            transition = "stopped"
            release_gpu_reservations(db, run.id)
        else:
            stopping = db.execute(update(TrainingRun).where(
                TrainingRun.id == run.id,
                TrainingRun.status.in_(["starting", "running", "stopping"]),
                TrainingRun.stop_requested.is_(False) | (TrainingRun.status != "stopping"),
            ).values(status="stopping", stop_requested=True))
            if stopping.rowcount == 1:
                transition = "stopping"

        # A repeated request is successful but must not create duplicate events.
        # Conditional updates also prevent a concurrent poll from overwriting a
        # queued cancellation, or a stale queued view from cancelling a claim.
        db.expire_all()
        run = db.get(TrainingRun, run.id)
        if not run or (transition is None and not (run.stop_requested and run.status in {"stopping", "stopped"})):
            db.rollback()
            raise HTTPException(409, "该远程任务当前无法停止")
        mirror_run_to_task(task, run)
        if transition is not None:
            source_ip = request.client.host if request.client else "unknown"
            message = "已取消远程训练排队" if transition == "stopped" else "已向远程 Agent 请求停止"
            add_event(db, task.id, "stop_requested", f"{message}；请求来源：{source_ip}",
                      payload={"source_ip": source_ip, "method": request.method, "path": request.url.path,
                               "requested_at": requested_at.isoformat(), "run_id": run.id}, run_id=run.id)
        db.commit()
        db.refresh(task)
        return task
    if not scheduler.stop_task(task_id):
        raise HTTPException(409, "该任务当前无法停止")
    with SessionLocal() as refreshed_db:
        return get_task_or_404(refreshed_db, task_id)


@app.post("/api/tasks/{task_id}/force-interrupt", response_model=TaskRead)
def force_interrupt_remote_task(task_id: str, payload: RemoteInterruptConfirmation,
                                request: Request, db: Session = Depends(get_db)):
    """Release a remote node only after an operator verifies its process is gone."""
    task = get_task_or_404(db, task_id)
    if not task.execution_node_id:
        raise HTTPException(409, "该操作只用于无法完成停止的远程训练")
    run = current_run(db, task)
    if not run or run.id != payload.run_id:
        raise HTTPException(409, "当前 Attempt 已变化，请刷新页面后重试")
    if run.status != "stopping" or not run.stop_requested:
        raise HTTPException(409, "请先停止远程训练，再确认远程进程已经结束")
    requested_at = utcnow()
    stop_requested_at = db.scalar(select(TaskEvent.created_at).where(
        TaskEvent.run_id == run.id,
        TaskEvent.event_type == "stop_requested",
    ).order_by(TaskEvent.id.desc()).limit(1))
    # Agent heartbeats update run.updated_at while a stop is pending. The
    # recovery delay starts from the operator's stop request, not the latest
    # heartbeat; old imported rows without an event retain a safe fallback.
    recovery_started_at = as_utc(stop_requested_at or run.updated_at)
    if requested_at - recovery_started_at < timedelta(seconds=30):
        raise HTTPException(409, "停止请求仍在处理，请等待 30 秒；若仍未结束，再到远程服务器确认进程状态")
    interrupted = db.execute(update(TrainingRun).where(
        TrainingRun.id == run.id,
        TrainingRun.status == "stopping",
        TrainingRun.stop_requested.is_(True),
        TrainingRun.updated_at == run.updated_at,
    ).values(
        status="interrupted", pid=None, finished_at=requested_at,
        error_message="人工确认远程训练进程已结束，节点占用已解除",
    ))
    if interrupted.rowcount != 1:
        db.rollback()
        raise HTTPException(409, "远程训练状态刚刚发生变化，请刷新页面后确认")
    db.expire_all()
    run = db.get(TrainingRun, payload.run_id)
    task = db.get(TrainingTask, task_id)
    if not run or not task:
        db.rollback()
        raise HTTPException(409, "训练记录已变化，请刷新页面")
    mirror_run_to_task(task, run)
    source_ip = request.client.host if request.client else "unknown"
    add_event(
        db, task.id, "force_interrupted", run.error_message, "warning",
        payload={"source_ip": source_ip, "confirmed_at": requested_at.isoformat(), "run_id": run.id},
        run_id=run.id,
    )
    # The operator explicitly confirmed the remote process has exited, so its
    # durable GPU reservation can now be released with the terminal transition.
    release_gpu_reservations(db, run.id)
    db.commit()
    db.refresh(task)
    return task


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
    guard = nullcontext() if source.execution_node_id else managed_dataset_mutation()
    with guard:
        if not source.execution_node_id:
            db.refresh(source)
            reject_local_task_cleanup(db)
        else:
            acquire_node_claim_lock(db, source.execution_node_id)
            db.refresh(source)
            reject_active_remote_delete(db, source.execution_node_id)
        # Reserve the source position and shift every visible task after it.
        # This keeps the clone immediately before its source for all clients,
        # including after a refresh or when another browser opens the dashboard.
        source_order = source.display_order
        db.execute(update(TrainingTask).where(
            TrainingTask.deleted_at.is_(None),
            TrainingTask.display_order >= source_order,
        ).values(display_order=TrainingTask.display_order + 1000))
        clone = TrainingTask(
            id=uuid.uuid4().hex[:16], name=next_clone_name(db, source.name), task_type=source.task_type,
            group_id=source.group_id,
            execution_node_id=source.execution_node_id,
            model_path=source.model_path, data_path=source.data_path, epochs=source.epochs, batch=source.batch,
            imgsz=source.imgsz, device=source.device, workers=source.workers, optimizer=source.optimizer,
            config={key: value for key, value in source.config.items() if key != "_resume_checkpoint"}, canvas_layout=source.canvas_layout.copy(),
            display_order=source_order,
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
    with model_test_lifecycle():
        task = get_task_or_404(db, task_id)
        if task.status in {"queued", "starting", "running", "stopping"}:
            raise HTTPException(409, "运行中或排队中的任务不能删除")
        if active_attempt_delete_operation(db, task.id):
            raise HTTPException(409, "该任务已有 Attempt 文件清理正在进行，请等待完成后再移入回收站")
        task.deleted_at = utcnow()
        task.deleted_reason = "soft_delete"
        stage_task_test_cancellation(task.id, db)
        add_event(db, task.id, "soft_deleted", "任务已移入回收站")
        db.commit()
        cancel_task_tests(task.id, bind=db.get_bind())


@app.post("/api/tasks/{task_id}/restore", response_model=TaskRead)
def restore_task(task_id: str, db: Session = Depends(get_db)):
    with model_test_lifecycle():
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
    with model_test_lifecycle():
        task = get_task_or_404(db, task_id, include_deleted=True)
        if task.status in {"queued", "starting", "running", "stopping"}:
            raise HTTPException(409, "运行中或排队中的任务不能永久删除")
        if active_attempt_delete_operation(db, task.id):
            raise HTTPException(409, "该任务已有 Attempt 文件清理正在进行，请等待完成后再删除任务")
        guard = nullcontext() if task.execution_node_id else managed_dataset_mutation()
        with guard:
            try:
                # Queue first: remote serialization may restart a read-only
                # transaction. Intent and cancellation markers commit together.
                operation = queue_delete(db, task) if task.execution_node_id else queue_local_delete(db, task)
                stage_task_test_cancellation(task.id, db, purge=True)
                db.commit()
            except ValueError as exc:
                db.rollback()
                raise HTTPException(409, str(exc)) from exc
        return JSONResponse(status_code=202, content={
            "status": operation.status,
            "operation_id": operation.id,
            "message": "已提交文件清理；可在回收站查看进度和清理结果",
        })


@app.get("/api/task-deletions")
def task_deletion_jobs(limit: int = Query(default=50, ge=1, le=200), db: Session = Depends(get_db)):
    return list_deletion_jobs(db, limit=limit)


@app.get("/api/tasks/{task_id}/deletion-preview")
def task_deletion_preview(task_id: str, db: Session = Depends(get_db)):
    return deletion_preview(db, get_task_or_404(db, task_id, include_deleted=True))


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
    if task.execution_node_id:
        node = db.get(ComputeNode, task.execution_node_id)
        reject_unsupported_remote_parameters(node, payload.config)
    if task.status in {"queued", "starting", "running", "stopping"}:
        raise HTTPException(409, "任务正在运行或排队，不能修改训练参数")
    supported_keys = editable_training_keys()
    unknown = set(payload.config) - supported_keys
    if unknown:
        raise HTTPException(422, f"包含不支持的参数：{', '.join(sorted(unknown))}")
    try:
        validate_parameter_values(payload.config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if len(payload.custom_config) > 30:
        raise HTTPException(422, "自定义参数最多添加 30 项")
    for key, value in payload.custom_config.items():
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
            raise HTTPException(422, f"自定义参数名不合法：{key}")
        if key in RESERVED_CUSTOM_KEYS or key in supported_keys:
            raise HTTPException(422, f"参数 {key} 由平台或预设参数管理，不能自定义覆盖")
        if (
            not isinstance(value, (str, int, float, bool))
            or isinstance(value, str) and len(value) > 500
            or isinstance(value, (int, float)) and not isinstance(value, bool) and not is_finite_number(value)
        ):
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
    guard = nullcontext() if task.execution_node_id else managed_dataset_mutation()
    with guard:
        if not task.execution_node_id:
            db.refresh(task)
            reject_local_task_cleanup(db)
        else:
            acquire_node_claim_lock(db, task.execution_node_id)
            db.refresh(task)
        if task.status in {"queued", "starting", "running", "stopping"}:
            raise HTTPException(409, "任务正在运行或排队，不能更换模型或数据集")
        if payload.model_path is None and payload.data_path is None:
            raise HTTPException(422, "至少需要提供一个模型或数据集路径")
        if task.execution_node_id:
            reject_active_remote_delete(db, task.execution_node_id)
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
    runs = db.scalars(select(TrainingRun).where(TrainingRun.task_id == task_id).order_by(TrainingRun.attempt_number.desc())).all()
    return [serialize_run(db, run) for run in runs]


@app.get("/api/tasks/{task_id}/runs/{run_id}", response_model=RunRead)
def task_run(task_id: str, run_id: str, db: Session = Depends(get_db)):
    return serialize_run(db, get_run_or_404(db, task_id, run_id))


@app.delete("/api/tasks/{task_id}/runs/{run_id}", status_code=202)
def delete_training_run(task_id: str, run_id: str, db: Session = Depends(get_db)):
    """Queue deletion of one terminal Attempt and its owned files."""
    with model_test_lifecycle():
        task = get_task_or_404(db, task_id)
        run = get_run_or_404(db, task_id, run_id)
        try:
            if task.execution_node_id or run.execution_node_id:
                node_id = task.execution_node_id or run.execution_node_id
                reject_active_remote_delete(db, node_id)
                operation = queue_attempt_delete(db, task, run)
                message = "已提交远程 Attempt 文件清理，完成后删除记录"
            else:
                operation = queue_local_run_delete(db, task, run)
                message = "已提交本机 Attempt 文件清理，完成后删除记录"
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        db.commit()
        return JSONResponse(status_code=202, content={
            "status": operation.status,
            "operation_id": operation.id,
            "message": message,
        })


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


@app.get("/api/tasks/{task_id}/runs/{run_id}/effective-best-weight", response_model=EffectiveBestWeightRead)
def run_effective_best_weight(task_id: str, run_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    artifact, source_run = effective_best_weight(db, run)
    return {
        "artifact": artifact,
        "source_run_id": source_run.id if source_run else None,
        "source_attempt_number": source_run.attempt_number if source_run else None,
        "inherited": bool(source_run and source_run.id != run.id),
    }


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


@app.get("/api/tasks/{task_id}/runs/{run_id}/trace")
def run_training_trace(task_id: str, run_id: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    if run.task_id != task_id:
        return {"available": False, "reason": "训练运行与任务不匹配", "samples": []}
    trace_path = remote_trace_root(run.id) / "trace.json" if run.execution_node_id else Path(run.run_dir or "") / "training-trace" / "trace.json"
    if not trace_path.is_file():
        source = "远程 Agent" if run.execution_node_id else "本机训练"
        return {"available": False, "reason": f"该运行尚未产生过程采集数据，请确认{source}已升级并重新打开", "samples": []}
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HTTPException(503, f"训练过程数据暂不可读：{exc}") from exc
    for sample in payload.get("samples", []):
        if sample.get("image"):
            sample["image_url"] = f"/api/tasks/{task_id}/runs/{run_id}/trace/images/{sample['image']}"
    return payload


@app.get("/api/tasks/{task_id}/runs/{run_id}/trace/images/{name}")
def run_training_trace_image(task_id: str, run_id: str, name: str, db: Session = Depends(get_db)):
    get_task_or_404(db, task_id)
    run = get_run_or_404(db, task_id, run_id)
    if Path(name).name != name:
        raise HTTPException(404, "训练过程图片不存在")
    trace_root = (remote_trace_root(run.id) if run.execution_node_id else Path(run.run_dir or "") / "training-trace").resolve()
    image = (trace_root / name).resolve()
    if trace_root not in image.parents or not image.is_file():
        raise HTTPException(404, "训练过程图片不存在")
    return FileResponse(image)


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


@app.post("/api/import", status_code=202, dependencies=[Depends(serialize_managed_dataset_references)])
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
