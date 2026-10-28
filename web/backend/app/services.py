from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError

from .config import (
    DATASET_ROOTS,
    LOG_ROOT,
    MODEL_ROOTS,
    NVIDIA_SMI_BIN,
    platform_instance_id,
    RUNTIME_ROOT,
    RUNS_ROOT,
    TRAINING_MKL_THREADING_LAYER,
    TRAINING_STOP_TIMEOUT_SECONDS,
    ULTRALYTICS_ROOT,
    YOLO_BIN,
)
from .database import SessionLocal
from .gpu_reservations import (
    acquire_node_claim_lock,
    node_serial_assignment,
    non_gpu_assignment,
    reserve_recovery_conflict,
    release_gpu_reservations,
    release_terminal_gpu_reservations,
    reserve_gpu_resources,
    resolve_gpu_assignments,
)
from .models import Artifact, MetricPoint, RemoteOperation, TaskEvent, TrainingRun, TrainingTask, utcnow
from .log_files import tail_log
from .metric_files import MetricCsvReader
from .parameter_catalog import training_parameter_keys
from .training_outcome import classify_training_exit
from .batch_history import BatchLogReader, append_reductions, scan_batch_log


logger = logging.getLogger("yolo-platform.training-scheduler")
_METRIC_CSV_READER = MetricCsvReader(maximum_files=16)
_BATCH_LOG_READER = BatchLogReader()


TRAINING_KEYS = {
    "patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay",
    "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction",
    "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl",
    "save", "cls_remap", "verbose", "single_cls", "classes", "profile", "compile", "overlap_mask",
    "mask_ratio", "dropout", "val", "split", "save_json", "conf", "iou", "max_det", "plots", "lrf",
    "warmup_momentum", "warmup_bias_lr", "cls_pw", "pose", "kobj", "rle", "angle", "dlog", "dgrad",
    "dlam", "nbs", "bgr", "cutmix", "copy_paste_mode", "auto_augment", "erasing",
}


def supported_training_keys() -> set[str]:
    """Use the installed Ultralytics config, retaining a safe fallback."""
    return training_parameter_keys() or TRAINING_KEYS


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def training_environment(*, task_id: str | None = None, run_id: str | None = None) -> dict[str, str]:
    """Build a stable environment for PyTorch and its DDP worker processes."""
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = TRAINING_MKL_THREADING_LAYER
    if task_id and run_id:
        environment.update({
            "YOLO_FORGE_INSTANCE_ID": platform_instance_id(),
            "YOLO_FORGE_NODE_ID": "local",
            "YOLO_FORGE_RUN_ID": run_id,
            "YOLO_FORGE_TASK_ID": task_id,
        })
    return environment


def _process_identity(pid: int) -> tuple[list[str], datetime] | None:
    """Read a Linux process command and start time without adding a runtime dependency."""
    if pid <= 0:
        return None
    proc = Path("/proc") / str(pid)
    try:
        stat_fields = (proc / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        if not stat_fields or stat_fields[0] == "Z":
            return None
        command = [part.decode("utf-8", errors="replace") for part in (proc / "cmdline").read_bytes().split(b"\0") if part]
        boot_line = next(line for line in Path("/proc/stat").read_text(encoding="utf-8").splitlines() if line.startswith("btime "))
        boot_time = int(boot_line.split()[1])
        clock_ticks = os.sysconf("SC_CLK_TCK")
        # stat field 22 is index 19 after removing pid and the parenthesized command.
        started_at = datetime.fromtimestamp(boot_time + int(stat_fields[19]) / clock_ticks, timezone.utc)
    except (FileNotFoundError, PermissionError, OSError, StopIteration, ValueError, IndexError):
        return None
    return command, started_at


def process_matches_training(pid: int, command: list[str], started_at: datetime | None) -> bool:
    """Reject stale/reused PIDs by matching both argv and process creation time."""
    identity = _process_identity(pid)
    if not identity or not command or not started_at:
        return False
    actual_command, actual_started_at = identity
    expected_started_at = started_at if started_at.tzinfo else started_at.replace(tzinfo=timezone.utc)
    return actual_command == [str(part) for part in command] and abs(
        (actual_started_at - expected_started_at.astimezone(timezone.utc)).total_seconds()
    ) <= 10


@dataclass
class ManagedTrainingProcess:
    run_id: str
    pid: int
    command: list[str]
    started_at: datetime
    popen: subprocess.Popen | None = None
    adopted: bool = False
    stop_deadline: float | None = None
    stop_escalated: bool = False

    def poll(self) -> int | None:
        if self.popen is not None:
            return self.popen.poll()
        return None if process_matches_training(self.pid, self.command, self.started_at) else 0


def failure_detail(log_path: str | None) -> str:
    if not log_path:
        return ""
    lines = [
        ANSI_ESCAPE.sub("", line).strip()
        for line in tail_log(log_path, 2000)
        if line.strip()
    ]
    root_markers = ("Error:", "RuntimeError:", "ValueError:", "FileNotFoundError:", "OutOfMemoryError:")
    normalized = [re.sub(r"^\[rank\d+\]:\s*", "", line) for line in lines]
    detail = next((line for line in reversed(normalized) if line.startswith(root_markers)), "")
    return detail or (lines[-1] if lines else "")


def add_event(
    db, task_id: str, event_type: str, message: str, level: str = "info",
    payload: dict | None = None, run_id: str | None = None,
):
    db.add(TaskEvent(task_id=task_id, run_id=run_id, event_type=event_type, message=message, level=level, payload=payload or {}))


def parameter_snapshot(task: TrainingTask) -> dict[str, Any]:
    return {
        "epochs": task.epochs,
        "batch": task.batch,
        "imgsz": task.imgsz,
        "device": task.device,
        "workers": task.workers,
        "optimizer": task.optimizer,
        "config": dict(task.config or {}),
        "remote_output_root": (task.config or {}).get("remote_output_root"),
    }


def set_queue_reason(db, run: TrainingRun, reason: str) -> None:
    # Conditional writes cannot put a stale queued explanation onto a Run
    # concurrently claimed or cancelled by another request.
    if run.status != "queued":
        return
    db.execute(update(TrainingRun).where(
        TrainingRun.id == run.id, TrainingRun.status == "queued",
        (TrainingRun.queue_reason.is_(None)) | (TrainingRun.queue_reason != reason),
    ).values(queue_reason=reason))
    db.execute(update(TrainingTask).where(
        TrainingTask.id == run.task_id, TrainingTask.current_run_id == run.id,
        TrainingTask.status == "queued",
        (TrainingTask.queue_reason.is_(None)) | (TrainingTask.queue_reason != reason),
    ).values(queue_reason=reason))


def mirror_run_to_task(task: TrainingTask, run: TrainingRun) -> None:
    """Keep the old task response useful while run-aware clients migrate."""
    task.current_run_id = run.id
    task.status = run.status
    task.run_dir = run.run_dir
    task.log_path = run.log_path
    task.completed_epochs = run.completed_epochs
    task.progress = run.progress
    task.pid = run.pid
    task.exit_code = run.exit_code
    if run.status != "queued":
        run.queue_reason = None
    task.queue_reason = run.queue_reason
    task.error_message = run.error_message
    task.completion_info = dict(run.completion_info or {})
    task.latest_metrics = dict(run.latest_metrics or {})
    task.batch_history = list(run.batch_history or [])
    task.queued_at = run.queued_at
    task.started_at = run.started_at
    task.finished_at = run.finished_at


def sync_local_batch_history(task: TrainingTask, run: TrainingRun, *, final: bool = False) -> None:
    """Publish each confirmed reduction without rescanning the whole active log."""
    if run.execution_node_id:
        return
    try:
        snapshot = run.parameter_snapshot if isinstance(run.parameter_snapshot, dict) else {}
        configured = int(snapshot.get("batch") or task.batch)
    except (TypeError, ValueError, OverflowError):
        return
    if not 1 <= configured <= 1024:
        return
    history, found = (scan_batch_log(run.log_path, configured) if final else
                      (_BATCH_LOG_READER.read(run.log_path, configured), False))
    known = list(run.batch_history or [])
    if history and len(history) > len(known) and history[:len(known)] == known:
        run.batch_history = history
        if task.current_run_id == run.id:
            task.batch_history = list(history)
    if final and found:
        run.batch_history_scanned = True


def backfill_batch_histories() -> int:
    """Scan legacy local logs and any cached remote log snapshot once at startup."""
    recovered = 0
    with SessionLocal() as db:
        snapshots_by_run: dict[str, list[RemoteOperation]] = {}
        for operation in db.scalars(select(RemoteOperation).where(
            RemoteOperation.kind == "fetch_log", RemoteOperation.status == "succeeded",
        )):
            run_id = (operation.payload or {}).get("run_id")
            if isinstance(run_id, str):
                snapshots_by_run.setdefault(run_id, []).append(operation)
        run_ids = db.scalars(select(TrainingRun.id).order_by(TrainingRun.task_id, TrainingRun.attempt_number)).all()
        for index, run_id in enumerate(run_ids, 1):
            run = db.get(TrainingRun, run_id)
            if not run:
                continue
            task = db.get(TrainingTask, run.task_id)
            if not task:
                continue
            try:
                snapshot = run.parameter_snapshot if isinstance(run.parameter_snapshot, dict) else {}
                configured = int(snapshot.get("batch") or task.batch)
            except (TypeError, ValueError, OverflowError):
                continue
            if not 1 <= configured <= 1024:
                continue
            found_complete = False
            if run.execution_node_id:
                history = append_reductions(list(run.batch_history or []), run.remote_log_tail or [], configured)
                if not run.batch_history_scanned and run.status not in {"queued", "starting", "running", "stopping"}:
                    from .remote_operations import operation_file
                    for operation in snapshots_by_run.get(run.id, []):
                        cached, found = scan_batch_log(operation_file(operation), configured)
                        if found:
                            history = cached
                            found_complete = run.status not in {"queued", "starting", "running", "stopping"}
                            break
            elif not run.batch_history_scanned and run.status not in {"queued", "starting", "running", "stopping"}:
                history, found = scan_batch_log(run.log_path, configured)
                found_complete = found
            else:
                history = list(run.batch_history or [])
            # The scheduler or a remote result may have advanced while a large
            # historical file was read. Never overwrite a newer live chain.
            db.refresh(run)
            db.refresh(task)
            if found_complete:
                run.batch_history_scanned = True
            known = list(run.batch_history or [])
            if history and len(history) > len(known) and history[:len(known)] == known:
                run.batch_history = history
                recovered += 1
            if task.current_run_id == run.id and task.batch_history != run.batch_history:
                task.batch_history = list(run.batch_history or [])
            if index % 32 == 0:
                db.commit()
                db.expunge_all()
        db.commit()
    return recovered


def log_path_for_run(task_id: str, attempt_number: int, when: datetime | None = None) -> Path:
    """Return a collision-resistant log name with a YYYYMMDD date prefix."""
    date_stamp = (when or datetime.now()).strftime("%Y%m%d")
    return LOG_ROOT / f"{date_stamp}_{task_id}_attempt-{attempt_number:03d}.log"


def current_run(db, task: TrainingTask) -> TrainingRun | None:
    if task.current_run_id:
        run = db.get(TrainingRun, task.current_run_id)
        if run:
            return run
    return db.scalar(
        select(TrainingRun).where(TrainingRun.task_id == task.id).order_by(TrainingRun.attempt_number.desc()).limit(1)
    )


def remote_log_path(output_root: str, task_id: str, attempt_number: int, when: datetime | None = None) -> Path:
    date_stamp = (when or datetime.now()).strftime("%Y%m%d")
    return Path(output_root) / ".platform-logs" / date_stamp / task_id / f"attempt-{attempt_number:03d}.log"


def create_training_run(
    db, task: TrainingTask, launch_mode: str, checkpoint_path: Path | None = None,
) -> TrainingRun:
    attempt = db.scalar(select(func.max(TrainingRun.attempt_number)).where(TrainingRun.task_id == task.id)) or 0
    # Keep Attempt numbers monotonic after a deleted run.  Completed deletion
    # operations are durable tombstones, so a later run cannot reuse an old
    # directory/log identity and be confused with historical records.
    for operation in db.scalars(select(RemoteOperation).where(RemoteOperation.task_id == task.id)):
        payload = operation.payload or {}
        if payload.get("scope") != "attempt":
            continue
        try:
            attempt = max(attempt, int(payload.get("attempt_number", 0)))
        except (TypeError, ValueError):
            continue
    attempt += 1
    attempt_name = f"attempt-{attempt:03d}"
    run = TrainingRun(
        id=uuid.uuid4().hex,
        task_id=task.id,
        attempt_number=attempt,
        launch_mode=launch_mode,
        status="queued",
        queue_reason="等待远程 Agent 领取任务" if task.execution_node_id else "等待本地调度器分配资源",
        execution_node_id=task.execution_node_id,
        parameter_snapshot=parameter_snapshot(task),
        model_path=task.model_path,
        data_path=task.data_path,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        run_dir=(str(Path((task.config or {}).get("remote_output_root")) / task.task_type / task.id / attempt_name)
                 if task.execution_node_id and (task.config or {}).get("remote_output_root") else str(RUNS_ROOT / task.task_type / task.id / attempt_name)),
        log_path=(str(remote_log_path((task.config or {})["remote_output_root"], task.id, attempt))
                  if task.execution_node_id and (task.config or {}).get("remote_output_root") else str(log_path_for_run(task.id, attempt))),
        completed_epochs=task.completed_epochs if launch_mode == "resume" else 0,
        progress=task.progress if launch_mode == "resume" else 0.0,
        latest_metrics=dict(task.latest_metrics or {}) if launch_mode == "resume" else {},
        queued_at=utcnow(),
    )
    db.add(run)
    db.flush()
    mirror_run_to_task(task, run)
    return run


def backfill_training_runs() -> int:
    """Register pre-migration task outputs as Attempt 001 in place."""
    created = 0
    with SessionLocal() as db:
        tasks = db.scalars(select(TrainingTask).order_by(TrainingTask.created_at)).all()
        for task in tasks:
            run = current_run(db, task)
            if not run and task.status == "draft" and not task.run_dir and not task.log_path:
                continue
            if not run:
                run = TrainingRun(
                    id=uuid.uuid4().hex,
                    task_id=task.id,
                    attempt_number=1,
                    launch_mode="fresh",
                    status=task.status,
                    execution_node_id=task.execution_node_id,
                    pid=task.pid,
                    process_started_at=task.started_at,
                    parameter_snapshot=parameter_snapshot(task),
                    model_path=task.model_path,
                    data_path=task.data_path,
                    run_dir=task.run_dir,
                    log_path=task.log_path,
                    completed_epochs=task.completed_epochs,
                    progress=task.progress,
                    latest_metrics=dict(task.latest_metrics or {}),
                    exit_code=task.exit_code,
                    error_message=task.error_message,
                    completion_info=dict(task.completion_info or {}),
                    queued_at=task.queued_at,
                    started_at=task.started_at,
                    finished_at=task.finished_at,
                    created_at=task.created_at,
                    updated_at=task.updated_at,
                )
                db.add(run)
                db.flush()
                created += 1
            task.current_run_id = run.id
            db.execute(update(MetricPoint).where(MetricPoint.task_id == task.id, MetricPoint.run_id.is_(None)).values(run_id=run.id))
            db.execute(update(Artifact).where(Artifact.task_id == task.id, Artifact.run_id.is_(None)).values(run_id=run.id))
            db.execute(update(TaskEvent).where(TaskEvent.task_id == task.id, TaskEvent.run_id.is_(None)).values(run_id=run.id))
            sync_metrics(db, task, run)
            sync_artifacts(db, task, run)
            if task.source == "imported" and run.completed_epochs >= task.epochs:
                run.status = "completed"
                run.progress = 100.0
                mirror_run_to_task(task, run)
        db.commit()
    return created


def migrate_training_logs() -> int:
    """Move inactive legacy run logs into LOG_ROOT and update stored paths."""
    migrated = 0
    active_statuses = {"queued", "starting", "running", "stopping"}
    with SessionLocal() as db:
        runs = db.scalars(select(TrainingRun).order_by(TrainingRun.task_id, TrainingRun.attempt_number)).all()
        for run in runs:
            if not run.log_path or run.execution_node_id or run.status in active_statuses:
                continue
            source = Path(run.log_path).expanduser().resolve()
            if path_is_within(source, [LOG_ROOT]) or not source.is_file():
                continue
            destination = log_path_for_run(run.task_id, run.attempt_number, run.started_at or run.created_at).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                source.replace(destination)
            run.log_path = str(destination)
            task = db.get(TrainingTask, run.task_id)
            if task and task.current_run_id == run.id:
                task.log_path = run.log_path
            migrated += 1
        db.commit()
    return migrated


def path_is_within(path: Path, roots: list[Path]) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def _deletion_path(path: Path) -> Path:
    """Resolve parent components while preserving the final symlink itself."""
    expanded = path.expanduser()
    return expanded.parent.resolve() / expanded.name


def _validate_deletion_target(path: Path, roots: list[Path]) -> Path:
    candidate = _deletion_path(path)
    allowed = [root.expanduser().resolve() for root in roots]
    if not any(candidate != root and root in candidate.parents for root in allowed):
        raise ValueError(f"训练目录不在允许的删除范围内：{candidate}")
    return candidate


def _safe_delete_directory(path: Path, roots: list[Path]) -> Path | None:
    """Delete a run directory only when it is below an explicitly allowed root."""
    candidate = _validate_deletion_target(path, roots)
    if candidate.is_symlink():
        candidate.unlink()
    elif not candidate.exists():
        return None
    elif not candidate.is_dir():
        candidate.unlink()
    else:
        shutil.rmtree(candidate)
    return candidate


def permanently_delete_task_files(task: TrainingTask) -> list[str]:
    """Remove only task-owned run/runtime directories; model and dataset paths are never targets."""
    if task.execution_node_id:
        raise ValueError("远程训练目录不在本机，当前不支持永久删除远程任务的文件")
    run_dirs = {Path(run.run_dir) for run in task.runs if run.run_dir}
    log_files = {Path(run.log_path) for run in task.runs if run.log_path}
    if task.run_dir:
        run_dirs.add(Path(task.run_dir))
    if task.log_path:
        log_files.add(Path(task.log_path))
    runtime_dir = RUNTIME_ROOT / task.id
    deletion_roots = [RUNS_ROOT, ULTRALYTICS_ROOT / "runs", RUNTIME_ROOT, LOG_ROOT]
    targets = sorted(run_dirs | log_files | {runtime_dir}, key=lambda path: len(str(path)), reverse=True)
    for target in targets:
        _validate_deletion_target(target, deletion_roots)
    deleted: list[str] = []
    for target in targets:
        removed = _safe_delete_directory(target, deletion_roots)
        if removed is not None:
            deleted.append(str(removed))
    return deleted


def validate_input_path(raw: str, roots: list[Path], label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label}不存在：{path}")
    if not path_is_within(path, roots):
        raise ValueError(f"{label}必须位于允许的目录中")
    for root in roots:
        try:
            relative = path.relative_to(root.expanduser().resolve())
        except ValueError:
            continue
        if any(part.startswith(".") for part in relative.parts):
            raise ValueError(f"{label}不能位于平台隐藏目录中")
        break
    return path


def training_dataset_roots(raw: str) -> list[Path]:
    """Keep legacy roots, adding only this registered, ready local dataset."""
    from .models import DatasetEntry
    from .config import DATASET_STORAGE_ALLOWED_ROOTS
    from sqlalchemy import inspect
    candidate = Path(raw).expanduser().resolve()
    roots = list(DATASET_ROOTS)
    if path_is_within(candidate, roots):
        return roots
    with SessionLocal() as db:
        if inspect(db.get_bind()).has_table('dataset_entries'):
            for entry in db.scalars(select(DatasetEntry).where(DatasetEntry.node_id == 'local', DatasetEntry.status == 'ready')):
                directory = Path(entry.path)
                if path_is_within(directory, DATASET_STORAGE_ALLOWED_ROOTS) and path_is_within(candidate, [directory]):
                    roots.append(directory)
                    break
    return roots


def validate_dataset_config(raw: str) -> Path:
    """Validate the dataset YAML and resolve train/val entries before queuing a run."""
    path = validate_input_path(raw, training_dataset_roots(raw), "数据集配置")
    if not path.is_file() or path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(f"非分类任务的数据集必须是 YAML 文件：{path}")
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"数据集配置无法读取：{path}（{exc}）") from exc
    if not isinstance(config, dict):
        raise ValueError(f"数据集配置格式错误：{path}")
    root_value = config.get("path")
    root = Path(str(root_value)).expanduser() if root_value else path.parent
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    entries: list[tuple[str, Any]] = [("train", config.get("train")), ("val", config.get("val"))]
    missing: list[str] = []
    for name, value in entries:
        if value is None:
            if name == "train": missing.append("缺少 train 配置")
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            candidate = Path(str(item)).expanduser()
            if not candidate.is_absolute():
                candidate = root / candidate
            if not candidate.exists():
                missing.append(f"{name}: {candidate}")
    if missing:
        raise ValueError(f"数据集图片路径不存在（{path}）：" + "; ".join(missing))
    return path


def validate_classification_dataset(raw: str) -> Path:
    """Validate an Ultralytics classification dataset directory without mutating it."""
    path = validate_input_path(raw, DATASET_ROOTS, "分类数据集目录")
    if not path.is_dir():
        raise ValueError(f"分类数据集必须是目录，不能是 YAML 文件：{path}")
    train = path / "train"
    if not train.is_dir():
        raise ValueError(f"分类数据集缺少 train 目录：{train}")
    try:
        has_class = any(child.is_dir() and not child.name.startswith(".") for child in train.iterdir())
    except OSError as exc:
        raise ValueError(f"分类数据集无法读取：{path}（{exc}）") from exc
    if not has_class:
        raise ValueError(f"分类数据集 train 目录中没有类别子目录：{train}")
    return path


def validate_training_data_path(task_type: str, raw: str) -> Path:
    """Validate the task-specific Ultralytics data input."""
    if task_type == "classify":
        return validate_classification_dataset(raw)
    return validate_dataset_config(raw)


def validate_remote_training_data_path(task_type: str, raw: str) -> str:
    """Validate task-specific remote path syntax before the Agent checks the filesystem."""
    value = raw.strip()
    path = Path(value)
    if not value or not path.is_absolute() or any(ord(character) < 32 for character in value):
        kind = "分类数据集目录" if task_type == "classify" else "数据集 YAML"
        raise ValueError(f"远程{kind}必须是不含控制字符的绝对路径")
    suffix = path.suffix.lower()
    if task_type == "classify":
        if suffix in {".yaml", ".yml"}:
            raise ValueError("图像分类任务需要远程数据集目录，不能使用 YAML 文件")
    elif suffix not in {".yaml", ".yml"}:
        raise ValueError("非分类任务需要远程 .yaml 或 .yml 数据集配置")
    return value


def prepare_dataset_config(raw: str, destination: Path) -> Path:
    """Write a run-local dataset YAML with an absolute root for Ultralytics."""
    source = validate_dataset_config(raw)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    root_value = config.get("path")
    root = Path(str(root_value)).expanduser() if root_value else source.parent
    if not root.is_absolute():
        root = (source.parent / root).resolve()
    config["path"] = str(root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return destination


def prepare_training_data_path(task_type: str, raw: str, destination: Path) -> Path:
    """Return a validated directory for classification or a run-local YAML otherwise."""
    if task_type == "classify":
        return validate_classification_dataset(raw)
    return prepare_dataset_config(raw, destination)


def discover_files(roots: list[Path], suffixes: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for directory, child_directories, filenames in os.walk(root, topdown=True, followlinks=False):
            child_directories[:] = [name for name in child_directories if not name.startswith(".")]
            parent = Path(directory)
            for filename in filenames:
                if filename.startswith("."):
                    continue
                path = parent / filename
                try:
                    if not path.is_file() or path.suffix.lower() not in suffixes:
                        continue
                    stat = path.stat()
                except OSError:
                    # A managed dataset can be atomically detached while this
                    # directory snapshot is being traversed.
                    continue
                name = path.parent.name if path.name in {"data.yaml", "data.yml"} and root in DATASET_ROOTS else path.name
                files.append({"name": name, "path": str(path), "size": stat.st_size, "root": str(root)})
    return sorted(files, key=lambda item: item["name"].lower())


def discover_classification_datasets(roots: list[Path]) -> list[dict[str, Any]]:
    """List dataset roots that contain a train split with at least one class directory."""
    datasets: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for directory, child_directories, _filenames in os.walk(root, topdown=True, followlinks=False):
            child_directories[:] = [name for name in child_directories if not name.startswith(".")]
            parent = Path(directory)
            if "train" not in child_directories:
                continue
            train = parent / "train"
            try:
                if parent.is_symlink() or not any(
                    child.is_dir() and not child.name.startswith(".") for child in train.iterdir()
                ):
                    continue
                resolved = parent.resolve()
            except OSError:
                continue
            if resolved in seen or not path_is_within(resolved, roots):
                continue
            seen.add(resolved)
            datasets.append({"name": parent.name, "path": str(resolved), "size": 0, "root": str(root)})
            child_directories.clear()
    return sorted(datasets, key=lambda item: item["name"].lower())


def _nullable_number(value: str, integer: bool = False) -> int | float | None:
    value = value.strip()
    if not value or value.lower() in {"n/a", "[n/a]", "not supported"}:
        return None
    try:
        number = float(value)
        return int(number) if integer else number
    except ValueError:
        return None


def _run_nvidia_smi(arguments: list[str]) -> str:
    return subprocess.check_output(
        [NVIDIA_SMI_BIN, *arguments], text=True, timeout=5, stderr=subprocess.STDOUT,
    )


def _gpu_processes() -> dict[str, list[dict[str, Any]]]:
    command = [
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = _run_nvidia_smi(command)
    except (OSError, subprocess.SubprocessError):
        return {}
    result: dict[str, list[dict[str, Any]]] = {}
    for row in csv.reader(output.splitlines()):
        if len(row) != 4:
            continue
        uuid, pid, name, memory = (value.strip() for value in row)
        parsed_pid = _nullable_number(pid, integer=True)
        if parsed_pid is None:
            continue
        result.setdefault(uuid, []).append({
            "pid": parsed_pid,
            "name": Path(name).name or name,
            "memory_used": _nullable_number(memory, integer=True),
        })
    return result


def gpu_inventory() -> list[dict[str, Any]]:
    fields = [
        "index", "uuid", "name", "driver_version", "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw", "power.limit",
        "clocks.current.graphics", "clocks.current.memory",
    ]
    command = [f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"]
    try:
        output = _run_nvidia_smi(command)
    except (OSError, subprocess.SubprocessError):
        return []
    processes = _gpu_processes()
    result: list[dict[str, Any]] = []
    for row in csv.reader(output.splitlines()):
        if len(row) != len(fields):
            continue
        values = [value.strip() for value in row]
        index = _nullable_number(values[0], integer=True)
        if index is None:
            continue
        uuid = values[1]
        result.append({
            "index": index,
            "uuid": uuid,
            "name": values[2],
            "driver_version": values[3],
            "memory_total": _nullable_number(values[4], integer=True),
            "memory_used": _nullable_number(values[5], integer=True),
            "memory_free": _nullable_number(values[6], integer=True),
            "utilization": _nullable_number(values[7], integer=True),
            "memory_utilization": _nullable_number(values[8], integer=True),
            "temperature": _nullable_number(values[9], integer=True),
            "power_draw": _nullable_number(values[10]),
            "power_limit": _nullable_number(values[11]),
            "graphics_clock": _nullable_number(values[12], integer=True),
            "memory_clock": _nullable_number(values[13], integer=True),
            "processes": processes.get(uuid, []),
        })
    return result


def _host_memory() -> tuple[int | None, int | None]:
    try:
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        total = values.get("MemTotal")
        available = values.get("MemAvailable")
        return total, total - available if total is not None and available is not None else None
    except (OSError, ValueError, IndexError):
        return None, None


def compute_inventory() -> dict[str, Any]:
    """Return live host/GPU data without assuming a specific server or GPU model."""
    total_memory, used_memory = _host_memory()
    try:
        load_average = [round(value, 2) for value in os.getloadavg()]
    except OSError:
        load_average = []
    gpus = gpu_inventory()
    return {
        "available": bool(gpus),
        "provider": "NVIDIA" if gpus else None,
        "message": None if gpus else "未检测到 NVIDIA GPU，或 nvidia-smi 当前不可用",
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "load_average": load_average,
            "memory_total": total_memory,
            "memory_used": used_memory,
        },
        "gpus": gpus,
    }


def parse_number(value: Any) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return str(value)


def sync_metrics(db, task: TrainingTask, run: TrainingRun | None = None) -> None:
    run = run or current_run(db, task)
    if not run or run.execution_node_id or not run.run_dir:
        return
    if any(
        operation.kind in {"local_delete", "delete"}
        and operation.status in {"pending", "running"}
        and (operation.payload or {}).get("scope") == "attempt"
        and (operation.payload or {}).get("run_id") == run.id
        for operation in db.scalars(select(RemoteOperation).where(RemoteOperation.task_id == task.id))
    ):
        return
    csv_path = Path(run.run_dir) / "results.csv"
    if not csv_path.exists():
        return
    snapshot = _METRIC_CSV_READER.read(
        csv_path, final=run.status in {"completed", "failed", "stopped", "interrupted"},
    )
    existing = {
        point.epoch: point
        for point in db.scalars(select(MetricPoint).where(MetricPoint.run_id == run.id)).all()
    }
    for point in existing.values():
        normalized_values = {
            key: None if isinstance(value, (int, float)) and not isinstance(value, bool)
            and not math.isfinite(value) else value
            for key, value in (point.values or {}).items()
        }
        if point.values != normalized_values:
            point.values = normalized_values
        if not isinstance(point.elapsed, (int, float)) or not math.isfinite(point.elapsed):
            # elapsed is a required numeric API/database field; match the
            # fallback used below for newly parsed invalid CSV durations.
            point.elapsed = 0.0
    latest: dict[str, Any] = {}
    latest_epoch = run.completed_epochs
    has_valid_epoch = False
    for metric in snapshot["metrics"]:
        epoch, values, elapsed = metric["epoch"], metric["values"], metric["elapsed"]
        has_valid_epoch = True
        if epoch not in existing:
            try:
                with db.begin_nested():
                    point = MetricPoint(
                        task_id=task.id, run_id=run.id, epoch=epoch, elapsed=elapsed, values=values,
                    )
                    db.add(point)
                    db.flush()
                    existing[epoch] = point
            except IntegrityError:
                pass
        else:
            point = existing[epoch]
            if point.values != values:
                point.values = values
            if point.elapsed != elapsed:
                point.elapsed = elapsed
        latest, latest_epoch = values, max(latest_epoch, epoch)
    if has_valid_epoch:
        epochs = int(
            (run.completion_info or {}).get("expected_epochs")
            or (run.parameter_snapshot or {}).get("epochs")
            or task.epochs
        )
        progress = 100.0 if run.status == "completed" else min(100.0, round(latest_epoch / max(epochs, 1) * 100, 2))
        changed = (run.completed_epochs != latest_epoch or run.progress != progress or run.latest_metrics != latest)
        if run.completed_epochs != latest_epoch:
            run.completed_epochs = latest_epoch
        if run.progress != progress:
            run.progress = progress
        if run.latest_metrics != latest:
            run.latest_metrics = latest
        if changed and task.current_run_id == run.id:
            mirror_run_to_task(task, run)


def artifact_kind(path: Path) -> str:
    if path.suffix.lower() in {".pt", ".onnx", ".engine", ".torchscript"}:
        return "weight"
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        return "image"
    if path.suffix.lower() in {".yaml", ".yml"}:
        return "config"
    if path.suffix.lower() in {".csv", ".json"}:
        return "metrics"
    if path.suffix.lower() in {".log", ".txt"}:
        return "log"
    return "file"


def _artifact_timestamp(value: datetime | None) -> datetime | None:
    """Normalize SQLite's naive DateTime values before comparing file mtimes."""
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def resume_checkpoint(task: TrainingTask) -> Path | None:
    """Return the checkpoint written by the latest incomplete run, if available."""
    if not task.run_dir:
        return None
    checkpoint = Path(task.run_dir) / "weights" / "last.pt"
    return checkpoint if checkpoint.is_file() else None


def effective_best_weight(db, run: TrainingRun) -> tuple[Artifact | None, TrainingRun | None]:
    """Find best.pt on this run or on the exact checkpoint ancestry of a resumed run."""
    visited: set[str] = set()
    candidate: TrainingRun | None = run
    while candidate and candidate.id not in visited:
        visited.add(candidate.id)
        weights = db.scalars(select(Artifact).where(
            Artifact.run_id == candidate.id,
            Artifact.kind == "weight",
            Artifact.size > 0,
        )).all()
        best = next((artifact for artifact in weights if Path(artifact.name).name == "best.pt"), None)
        if best:
            return best, candidate
        if candidate.launch_mode != "resume" or not candidate.checkpoint_path:
            break
        checkpoint = db.scalar(select(Artifact).where(
            Artifact.task_id == candidate.task_id,
            Artifact.path == candidate.checkpoint_path,
            Artifact.run_id.is_not(None),
        ))
        source = db.get(TrainingRun, checkpoint.run_id) if checkpoint and checkpoint.run_id else None
        if not source or source.task_id != candidate.task_id or source.attempt_number >= candidate.attempt_number:
            break
        candidate = source
    return None, None


def sync_artifacts(db, task: TrainingTask, run: TrainingRun | None = None) -> None:
    run = run or current_run(db, task)
    if not run or run.execution_node_id or not run.run_dir:
        return
    if any(
        operation.kind in {"local_delete", "delete"}
        and operation.status in {"pending", "running"}
        and (operation.payload or {}).get("scope") == "attempt"
        and (operation.payload or {}).get("run_id") == run.id
        for operation in db.scalars(select(RemoteOperation).where(RemoteOperation.task_id == task.id))
    ):
        return
    root = Path(run.run_dir)
    known = {
        artifact.path: artifact
        for artifact in db.scalars(select(Artifact).where(Artifact.run_id == run.id)).all()
    }
    if not root.is_dir():
        for artifact in known.values():
            db.delete(artifact)
        return
    discovered = []
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        try:
            raw = str(path.resolve())
            stat = path.stat()
        except OSError:
            continue
        discovered.append((path, raw, stat.st_size, datetime.fromtimestamp(stat.st_mtime, timezone.utc)))
    discovered_paths: set[str] = set()
    for path, raw, size, modified_at in discovered:
        discovered_paths.add(raw)
        name = str(path.relative_to(root))
        existing = known.get(raw)
        if existing:
            values = (artifact_kind(path), name, size, modified_at)
            existing_values = (existing.kind, existing.name, existing.size, _artifact_timestamp(existing.modified_at))
            if existing_values != values:
                existing.kind, existing.name, existing.size, existing.modified_at = values
            continue
        values = {
            "task_id": task.id,
            "run_id": run.id,
            "kind": artifact_kind(path),
            "name": name,
            "path": raw,
            "size": size,
            "modified_at": modified_at,
        }
        # The scheduler and API can discover the same file concurrently. A savepoint
        # makes a uniqueness race harmless on SQLite and other supported databases.
        artifact = Artifact(**values)
        try:
            with db.begin_nested():
                db.add(artifact)
                db.flush()
        except IntegrityError:
            existing = db.scalar(
                select(Artifact).where(Artifact.task_id == task.id, Artifact.path == raw)
            )
            if existing:
                existing.kind = values["kind"]
                existing.name = values["name"]
                existing.size = values["size"]
                existing.modified_at = values["modified_at"]
            artifact = existing
        if artifact:
            known[raw] = artifact
    for raw, artifact in known.items():
        if raw not in discovered_paths:
            db.delete(artifact)


def import_existing_runs() -> int:
    runs_root = ULTRALYTICS_ROOT / "runs"
    if not runs_root.exists():
        return 0
    imported = 0
    with SessionLocal() as db:
        for args_path in runs_root.rglob("args.yaml"):
            imported_this_run = False
            try:
                with db.begin_nested():
                    run_dir = args_path.parent.resolve()
                    task_id = "legacy-" + hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]
                    task = db.get(TrainingTask, task_id)
                    if task:
                        if task.deleted_at:
                            continue
                        sync_metrics(db, task)
                        sync_artifacts(db, task)
                        continue
                    args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
                    if not isinstance(args, dict):
                        raise ValueError("args.yaml 顶层必须是映射")
                    if args.get("mode") != "train":
                        continue
                    # Imported configuration is persisted as JSON by SQLAlchemy.
                    # Reject YAML-only objects in this directory without blocking
                    # discovery of independent runs.
                    json.dumps(args, allow_nan=False)
                    epochs = int(args.get("epochs") or 100)
                    task = TrainingTask(
                        id=task_id,
                        name=str(args.get("name") or run_dir.name),
                        task_type=str(args.get("task") or "detect"),
                        status="interrupted",
                        source="imported",
                        model_path=str(args.get("model") or ""),
                        data_path=str(args.get("data") or ""),
                        run_dir=str(run_dir),
                        log_path=str(run_dir / "platform.log"),
                        epochs=epochs,
                        batch=int(args.get("batch") or 16),
                        imgsz=int(args.get("imgsz") or 640),
                        device=str(args.get("device") or "0"),
                        workers=int(args.get("workers") or 8),
                        optimizer=str(args.get("optimizer") or "auto"),
                        config={**args, "path_issues": {
                            "model_missing": bool(args.get("model")) and not Path(str(args["model"])).exists(),
                            "data_missing": bool(args.get("data")) and not Path(str(args["data"])).exists(),
                        }},
                        display_order=(db.scalar(select(func.coalesce(func.max(TrainingTask.display_order), 0))) or 0) + 1000,
                    )
                    db.add(task)
                    db.flush()
                    sync_metrics(db, task)
                    if task.completed_epochs >= task.epochs:
                        task.status = "completed"
                        task.progress = 100.0
                    sync_artifacts(db, task)
                    add_event(db, task.id, "imported", "从现有 Ultralytics 训练目录导入")
                    imported_this_run = True
            except Exception as exc:
                logger.warning(
                    "existing_run_import_skipped args_path=%s error=%s: %s",
                    args_path, type(exc).__name__, exc,
                )
                continue
            if imported_this_run:
                imported += 1
        db.commit()
    return imported


class TrainingScheduler:
    def __init__(self, session_factory=SessionLocal) -> None:
        self.session_factory = session_factory
        self.processes: dict[str, ManagedTrainingProcess] = {}
        self._process_lock = threading.Lock()
        self._last_artifact_scan: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._health_lock = threading.Lock()
        self._active_error: str | None = None
        self._last_error: str | None = None
        self._last_error_at: datetime | None = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def active_error(self) -> str | None:
        with self._health_lock:
            return self._active_error

    @property
    def last_error(self) -> str | None:
        with self._health_lock:
            return self._last_error

    @property
    def last_error_at(self) -> datetime | None:
        with self._health_lock:
            return self._last_error_at

    def _record_error(self, exc: Exception) -> None:
        detail = f"{type(exc).__name__}: {exc}"[:1000]
        with self._health_lock:
            self._active_error = detail
            self._last_error = detail
            self._last_error_at = utcnow()

    def _record_success(self) -> None:
        with self._health_lock:
            self._active_error = None

    @staticmethod
    def _find_run_process(run: TrainingRun) -> int | None:
        candidates = [run.pid] if run.pid else []
        try:
            candidates.extend(int(path.name) for path in Path("/proc").iterdir() if path.name.isdigit())
        except OSError:
            pass
        for pid in dict.fromkeys(candidates):
            if pid and process_matches_training(pid, list(run.command or []), run.process_started_at):
                return pid
        return None

    @staticmethod
    def _finalize_run(
        task: TrainingTask, run: TrainingRun, *, actual_exit_code: int | None = None,
        interrupted_message: str = "平台接管的训练进程已退出，无法取得原进程退出码",
        only_verified: bool = False,
    ) -> tuple[str, str] | None:
        outcome = classify_training_exit(
            run.run_dir, run.log_path, run.id, task.id,
            int((run.parameter_snapshot or {}).get("epochs") or task.epochs),
            run.completed_epochs, run.process_started_at or run.started_at,
            actual_exit_code=actual_exit_code,
        )
        if only_verified and outcome["status"] == "interrupted":
            return None
        run.status = outcome["status"]
        run.exit_code = outcome["exit_code"]
        run.pid = None
        run.completion_info = outcome["completion_info"] or {}
        run.finished_at = utcnow()
        if run.completion_info.get("finished_at"):
            run.finished_at = datetime.fromisoformat(run.completion_info["finished_at"].replace("Z", "+00:00"))
        run.completed_epochs = max(run.completed_epochs, run.completion_info.get("completed_epochs", 0))
        run.error_message = outcome["error_message"]
        if run.status == "completed":
            run.progress = 100.0
            message = "训练已完成 · 提前停止" if run.completion_info.get("reason") == "early_stopping" else "训练已完成"
            return message, "info"
        if run.status == "interrupted":
            run.error_message = interrupted_message
            return run.error_message, "warning"
        if actual_exit_code is not None and actual_exit_code != 0:
            detail = failure_detail(run.log_path)
            run.error_message = f"训练进程异常退出，退出码 {actual_exit_code}"
            if detail or outcome["error_message"]:
                run.error_message += f"：{(detail or outcome['error_message'])[-500:]}"
        return run.error_message or "训练异常结束", "error"

    def _recover_processes(self) -> None:
        with self.session_factory() as db:
            release_terminal_gpu_reservations(db)
            local_gpus = gpu_inventory()
            candidates = db.scalars(select(TrainingRun).where(
                TrainingRun.execution_node_id.is_(None),
                (
                    TrainingRun.status.in_(["starting", "running", "stopping"])
                    | ((TrainingRun.status == "interrupted") & TrainingRun.error_message.in_([
                        "平台服务重启后未发现受管训练进程",
                        "平台接管的训练进程已退出，无法取得原进程退出码",
                    ]))
                ),
            )).all()
            for run in candidates:
                task = db.get(TrainingTask, run.task_id)
                previous_status = run.status
                historical = previous_status == "interrupted"
                if not task or (historical and (task.deleted_at or run.stop_requested)):
                    continue
                # A training process can finish writing checkpoints while the platform is
                # restarting. Reconcile the run directory before deciding whether the
                # process can be reattached so completed checkpoints do not remain
                # invisible after the run is marked interrupted.
                sync_artifacts(db, task, run)
                pid = self._find_run_process(run)
                if pid and task.current_run_id != run.id:
                    # Reconciliation of an old Attempt must not replace the current one.
                    continue
                if pid and run.process_started_at:
                    run.status = "stopping" if previous_status == "stopping" else "running"
                    run.pid = pid
                    run.error_message = None
                    run.finished_at = None
                    mirror_run_to_task(task, run)
                    device = str((run.parameter_snapshot or {}).get("device", task.device) or "").strip().lower()
                    if device in {"cpu", "mps"}:
                        assignments = [non_gpu_assignment("local", run.id, device)]
                    else:
                        assignments = resolve_gpu_assignments(
                            device, local_gpus, node_id="local",
                        ) or [node_serial_assignment("local")]
                    if not reserve_gpu_resources(
                        db, run_id=run.id, node_id="local", assignments=assignments,
                    ):
                        # The process is already alive and cannot safely be
                        # interrupted during reconciliation. Keep a durable
                        # node-wide guard until this process reaches terminal
                        # state, so another active run's eventual cleanup
                        # cannot expose the disputed GPU to a new launch.
                        reserve_recovery_conflict(db, run_id=run.id, node_id="local")
                        logger.warning("local_gpu_reservation_recovery_conflict run_id=%s", run.id)
                    with self._process_lock:
                        self.processes[task.id] = ManagedTrainingProcess(
                            run_id=run.id, pid=pid, command=list(run.command or []),
                            started_at=run.process_started_at, adopted=True,
                        )
                    add_event(
                        db, task.id, "reattached", f"平台重启后已重新接管 Attempt {run.attempt_number:03d}，PID {pid}",
                        payload={"pid": pid, "previous_status": previous_status}, run_id=run.id,
                    )
                    if previous_status == "stopping":
                        try:
                            os.killpg(pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    continue
                sync_metrics(db, task, run)
                sync_local_batch_history(task, run, final=True)
                stopped_during_start = run.status == "stopping" and run.stop_requested
                if stopped_during_start:
                    run.status = "stopped"
                    run.pid = None
                    run.finished_at = utcnow()
                    run.completion_info = {}
                    run.error_message = "平台停止请求已确认，训练进程已退出或尚未启动"
                    message, level = run.error_message, "info"
                else:
                    result = self._finalize_run(
                        task, run, interrupted_message="平台服务重启后未发现受管训练进程",
                        only_verified=historical,
                    )
                    if result is None:
                        # No new evidence: leave the historical terminal state untouched.
                        continue
                    message, level = result
                release_gpu_reservations(db, run.id)
                if task.current_run_id == run.id:
                    mirror_run_to_task(task, run)
                add_event(
                    db, task.id, "reconciled" if historical else run.status, message, level,
                    payload={"previous_status": previous_status, "status": run.status,
                             "exit_code": run.exit_code, "completion_info": run.completion_info}, run_id=run.id,
                )
            db.commit()

    def start(self) -> None:
        self._recover_processes()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="training-scheduler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    @staticmethod
    def requested_gpus(device: str) -> set[str]:
        return set() if device in {"cpu", "mps"} else {part.strip() for part in device.split(",")}

    def _available(self, db, task: TrainingTask) -> bool:
        wanted = self.requested_gpus(task.device)
        active = db.scalars(select(TrainingTask).where(
            TrainingTask.execution_node_id.is_(None),
            TrainingTask.status.in_(["starting", "running", "stopping"]),
        )).all()
        used: set[str] = set()
        for other in active:
            if other.id != task.id:
                used |= self.requested_gpus(other.device)
        return not wanted.intersection(used)

    def _command(
        self,
        task: TrainingTask,
        run: TrainingRun,
        data_path: Path | None = None,
    ) -> list[str]:
        snapshot = run.parameter_snapshot or {}
        config = snapshot.get("config") or {}
        resolved_data_path = data_path or Path(run.data_path)
        attempt_name = f"attempt-{run.attempt_number:03d}"
        command = [
            str(YOLO_BIN), task.task_type, "train",
            f"model={run.model_path}", f"data={resolved_data_path}", f"epochs={snapshot.get('epochs', task.epochs)}",
            f"batch={snapshot.get('batch', task.batch)}", f"imgsz={snapshot.get('imgsz', task.imgsz)}",
            f"device={snapshot.get('device', task.device)}", f"workers={snapshot.get('workers', task.workers)}",
            f"optimizer={snapshot.get('optimizer', task.optimizer)}",
            f"project={RUNS_ROOT / task.task_type / task.id}", f"name={attempt_name}", "exist_ok=True",
        ]
        if run.checkpoint_path:
            command.extend([f"resume={run.checkpoint_path}", f"save_dir={run.run_dir}"])
        supported_keys = supported_training_keys()
        for key, value in config.items():
            if key in supported_keys and value is not None:
                command.append(f"{key}={value}")
        for key, value in (config.get("custom_args") or {}).items():
            command.append(f"{key}={value}")
        if not run.execution_node_id:
            trace_dir = Path(run.run_dir or RUNS_ROOT / task.task_type / task.id / f"attempt-{run.attempt_number:03d}") / "training-trace"
            trace_dir.mkdir(parents=True, exist_ok=True)
            python_bin = YOLO_BIN.parent / "python"
            if not python_bin.is_file():
                python_bin = Path(sys.executable)
            return [
                str(python_bin), str(Path(__file__).with_name("training_trace.py")),
                "--trace-dir", str(trace_dir), "--maximum", "24", "--interval", "10",
                *([] if task.task_type == "detect" else ["--disable-trace"]),
                "--command-json", json.dumps(command, ensure_ascii=False),
                *command,
            ]
        return command

    def _claim_queued_run(
        self,
        db,
        run_id: str,
        assignments: list[dict[str, Any]] | None = None,
    ) -> bool:
        """Durably claim a queued local run before any process can be created."""
        acquire_node_claim_lock(db, "local")
        from .dataset_store import mutation_pending
        if mutation_pending(db, 'local'):
            db.rollback()
            return False
        candidate = db.get(TrainingRun, run_id)
        candidate_task = db.get(TrainingTask, candidate.task_id) if candidate else None
        if not candidate or not candidate_task:
            db.rollback()
            return False
        if assignments is None:
            device = str((candidate.parameter_snapshot or {}).get("device", candidate_task.device) or "").strip().lower()
            if device in {"cpu", "mps"}:
                assignments = [non_gpu_assignment("local", candidate.id, device)]
            elif device and all(part.strip().isdigit() for part in device.split(",")):
                assignments = resolve_gpu_assignments(device, gpu_inventory(), node_id="local")
                if assignments is None:
                    db.rollback()
                    return False
            else:
                assignments = [node_serial_assignment("local")]
        claimed = db.execute(
            update(TrainingRun).where(
                TrainingRun.id == run_id,
                TrainingRun.execution_node_id.is_(None),
                TrainingRun.status == "queued",
                TrainingRun.stop_requested.is_(False),
            ).values(status="starting", queue_reason=None, error_message=None, finished_at=None)
        )
        if claimed.rowcount != 1:
            db.rollback()
            return False
        if not reserve_gpu_resources(
            db, run_id=run_id, node_id="local", assignments=assignments,
        ):
            db.rollback()
            return False
        db.expire_all()
        run = db.get(TrainingRun, run_id)
        task = db.get(TrainingTask, run.task_id) if run else None
        if not run or not task or task.deleted_at or task.current_run_id != run.id:
            if run:
                run.status = "interrupted"
                run.error_message = "训练队列记录已失效，未启动进程"
                run.finished_at = utcnow()
                release_gpu_reservations(db, run.id)
            db.commit()
            return False
        mirror_run_to_task(task, run)
        db.commit()
        return True

    def _finish_unlaunched_stop(self, db, run_id: str) -> bool:
        """Complete a stop that won before Popen created a process."""
        stopped = db.execute(
            update(TrainingRun).where(
                TrainingRun.id == run_id,
                TrainingRun.status == "stopping",
                TrainingRun.pid.is_(None),
            ).values(
                status="stopped", finished_at=utcnow(),
                error_message="平台已取消本机训练启动，训练进程尚未启动",
            )
        )
        if stopped.rowcount != 1:
            db.rollback()
            return False
        db.expire_all()
        run = db.get(TrainingRun, run_id)
        task = db.get(TrainingTask, run.task_id) if run else None
        if task and run:
            mirror_run_to_task(task, run)
            add_event(db, task.id, "stopped", run.error_message, run_id=run.id)
            release_gpu_reservations(db, run.id)
        db.commit()
        return True

    def _fail_before_launch(self, db, run_id: str, message: str, event_message: str) -> None:
        failed = db.execute(
            update(TrainingRun).where(
                TrainingRun.id == run_id,
                TrainingRun.status == "starting",
                TrainingRun.stop_requested.is_(False),
            ).values(status="failed", error_message=message, finished_at=utcnow(), pid=None)
        )
        if failed.rowcount != 1:
            db.rollback()
            self._finish_unlaunched_stop(db, run_id)
            return
        db.expire_all()
        run = db.get(TrainingRun, run_id)
        task = db.get(TrainingTask, run.task_id) if run else None
        if task and run:
            mirror_run_to_task(task, run)
            add_event(db, task.id, "failed", event_message, "error", run_id=run.id)
            release_gpu_reservations(db, run.id)
        db.commit()

    def _launch(self, db, task: TrainingTask, run: TrainingRun) -> None:
        run_dir = Path(run.run_dir or RUNS_ROOT / task.task_type / task.id / f"attempt-{run.attempt_number:03d}")
        runtime_dir = RUNTIME_ROOT / task.id / f"attempt-{run.attempt_number:03d}"
        log_path = Path(run.log_path) if run.log_path else log_path_for_run(task.id, run.attempt_number)
        try:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path = prepare_training_data_path(
                task.task_type, run.data_path, runtime_dir / "dataset.yaml",
            )
        except (OSError, ValueError, yaml.YAMLError) as exc:
            self._fail_before_launch(db, run.id, str(exc), f"数据集配置校验失败：{exc}")
            return
        resume_path = Path(run.checkpoint_path).resolve() if run.checkpoint_path else None
        if resume_path and not resume_path.is_file():
            message = f"断点检查点不存在：{resume_path}"
            self._fail_before_launch(db, run.id, message, message)
            return
        command = self._command(task, run, dataset_path)
        from .dataset_provenance import capture as capture_dataset
        from .run_dataset_snapshot import enrich as enrich_dataset
        snapshot = dict(run.parameter_snapshot or {})
        if 'dataset_snapshot' not in snapshot:
            snapshot['dataset_snapshot'] = enrich_dataset(db, capture_dataset(
                run.data_path, task_type=task.task_type, node_id='local',
                runtime_path=dataset_path,
            ), run.data_path, launch_mode=run.launch_mode)
        prepared_at = utcnow()
        prepared = db.execute(
            update(TrainingRun).where(
                TrainingRun.id == run.id,
                TrainingRun.status == "starting",
                TrainingRun.stop_requested.is_(False),
            ).values(
                command=command, run_dir=str(run_dir), log_path=str(log_path),
                started_at=prepared_at, process_started_at=prepared_at,
                parameter_snapshot=snapshot,
            )
        )
        if prepared.rowcount != 1:
            db.rollback()
            self._finish_unlaunched_stop(db, run.id)
            return
        db.expire_all()
        prepared_run = db.get(TrainingRun, run.id)
        prepared_task = db.get(TrainingTask, task.id)
        if prepared_run and prepared_task:
            mirror_run_to_task(prepared_task, prepared_run)
        db.commit()

        log_handle = None
        try:
            log_handle = log_path.open("a", encoding="utf-8", buffering=1)
            log_handle.write(f"\n[{utcnow().isoformat()}] COMMAND {json.dumps(command, ensure_ascii=False)}\n")
            log_handle.write(f"[{utcnow().isoformat()}] ENV MKL_THREADING_LAYER={TRAINING_MKL_THREADING_LAYER}\n")
            process = subprocess.Popen(
                command,
                cwd=ULTRALYTICS_ROOT,
                env=training_environment(task_id=task.id, run_id=run.id),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            self._fail_before_launch(db, run.id, str(exc), f"训练进程启动失败：{exc}")
            return
        finally:
            if log_handle is not None:
                log_handle.close()

        managed = ManagedTrainingProcess(
            run_id=run.id, pid=process.pid, command=command,
            started_at=prepared_at, popen=process,
        )
        with self._process_lock:
            self.processes[task.id] = managed
        running = db.execute(
            update(TrainingRun).where(
                TrainingRun.id == run.id,
                TrainingRun.status == "starting",
                TrainingRun.stop_requested.is_(False),
            ).values(status="running", pid=process.pid, error_message=None)
        )
        if running.rowcount == 1:
            db.expire_all()
            launched_run = db.get(TrainingRun, run.id)
            launched_task = db.get(TrainingTask, task.id)
            if launched_run and launched_task:
                mirror_run_to_task(launched_task, launched_run)
                add_event(
                    db, launched_task.id, "started",
                    f"Attempt {launched_run.attempt_number:03d} 已启动，PID {process.pid}",
                    payload={"command": command}, run_id=launched_run.id,
                )
            db.commit()
            return

        # A stop request can win after the pre-launch barrier and before Popen
        # returns. Persist the PID for recovery, then terminate the new process.
        db.rollback()
        db.expire_all()
        cancelled_run = db.get(TrainingRun, run.id)
        cancelled_task = db.get(TrainingTask, task.id)
        if cancelled_run and cancelled_run.status in {"stopping", "stopped"} and cancelled_run.stop_requested:
            cancelled_run.pid = process.pid
            cancelled_run.command = command
            cancelled_run.process_started_at = prepared_at
            if cancelled_task:
                mirror_run_to_task(cancelled_task, cancelled_run)
            db.commit()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        raise RuntimeError(f"Attempt {run.id} 启动后的数据库状态无效，已终止训练进程")

    def stop_task(self, task_id: str) -> bool:
        with self.session_factory() as db:
            task = db.get(TrainingTask, task_id)
            if not task or task.execution_node_id:
                return False
            run = current_run(db, task)
            if not run:
                return False
            run_id = run.id
            now = utcnow()
            transition = None
            cancelled = db.execute(
                update(TrainingRun).where(
                    TrainingRun.id == run_id,
                    TrainingRun.status == "queued",
                    TrainingRun.stop_requested.is_(False),
                ).values(
                    status="stopped", stop_requested=True, pid=None, finished_at=now,
                    error_message="平台取消排队，训练尚未启动",
                )
            )
            if cancelled.rowcount == 1:
                transition = "stopped"
                release_gpu_reservations(db, run_id)
            else:
                stopping = db.execute(
                    update(TrainingRun).where(
                        TrainingRun.id == run_id,
                        TrainingRun.status.in_(["starting", "running"]),
                        TrainingRun.stop_requested.is_(False),
                    ).values(status="stopping", stop_requested=True)
                )
                if stopping.rowcount == 1:
                    transition = "stopping"
            db.expire_all()
            run = db.get(TrainingRun, run_id)
            task = db.get(TrainingTask, task_id)
            if not run or not task:
                db.rollback()
                return False
            if transition is None:
                if not (run.stop_requested and run.status in {"stopping", "stopped"}):
                    db.rollback()
                    return False
                transition = run.status
                db.rollback()
            else:
                mirror_run_to_task(task, run)
                message = "已从等待队列移除" if transition == "stopped" else "正在停止训练进程"
                add_event(db, task.id, transition, message, run_id=run.id)
                db.commit()
        with self._process_lock:
            managed = self.processes.get(task_id)
        if transition == "stopping" and managed and managed.run_id == run_id:
            try:
                os.killpg(managed.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return True

    def _enforce_stop(self, db, task, run, managed) -> None:
        # poll() verifies adopted identities; check again before signalling.
        if managed.adopted and not process_matches_training(managed.pid, managed.command, managed.started_at):
            return
        now = time.monotonic()
        if managed.stop_deadline is None:
            managed.stop_deadline = now + TRAINING_STOP_TIMEOUT_SECONDS
            signum = signal.SIGTERM
        elif now >= managed.stop_deadline and not managed.stop_escalated:
            managed.stop_escalated = True
            signum = signal.SIGKILL
            add_event(db, task.id, "stop_escalated", "训练未在停止宽限期内退出，已强制终止进程组", "warning", run_id=run.id)
        else:
            return
        try:
            os.killpg(managed.pid, signum)
        except ProcessLookupError:
            pass

    def _poll_processes(self, db) -> None:
        with self._process_lock:
            processes = list(self.processes.items())
        for task_id, managed in processes:
            run_id = managed.run_id
            task = db.get(TrainingTask, task_id)
            run = db.get(TrainingRun, run_id)
            if not task or not run:
                continue
            if run.status == "starting":
                run.status = "running"
                run.pid = managed.pid
                run.command = managed.command
                run.process_started_at = managed.started_at
                run.started_at = run.started_at or managed.started_at
                mirror_run_to_task(task, run)
                add_event(
                    db, task.id, "started", f"Attempt {run.attempt_number:03d} 启动状态已恢复，PID {managed.pid}",
                    payload={"command": managed.command}, run_id=run.id,
                )
            elif run.status == "stopping" and run.pid is None:
                run.pid = managed.pid
                run.command = managed.command
                run.process_started_at = managed.started_at
                mirror_run_to_task(task, run)
            sync_metrics(db, task, run)
            sync_local_batch_history(task, run)
            code = managed.poll()
            if code is None:
                if run.status == "stopping" or run.stop_requested:
                    self._enforce_stop(db, task, run, managed)
                now = time.monotonic()
                if now - self._last_artifact_scan.get(run_id, 0.0) >= 10:
                    sync_artifacts(db, task, run)
                    self._last_artifact_scan[run_id] = now
                continue
            run.pid = None
            run.finished_at = utcnow()
            if run.status == "stopping" or (run.status == "stopped" and run.stop_requested):
                run.status = "stopped"
                run.exit_code = None if managed.adopted else code
                run.completion_info = {}
                message, level = "训练已停止", "info"
            else:
                message, level = self._finalize_run(
                    task, run, actual_exit_code=None if managed.adopted else code,
                )
            if task.current_run_id == run.id:
                mirror_run_to_task(task, run)
            sync_local_batch_history(task, run, final=True)
            sync_artifacts(db, task, run)
            self._last_artifact_scan.pop(run_id, None)
            add_event(db, task.id, run.status, message, level,
                      payload={"exit_code": run.exit_code, "completion_info": run.completion_info}, run_id=run.id)
            release_gpu_reservations(db, run.id)
            with self._process_lock:
                if self.processes.get(task_id) is managed:
                    self.processes.pop(task_id, None)

    def _run_once(self) -> None:
        with self.session_factory() as db:
            self._poll_processes(db)
            from .compute_nodes import remote_training_unavailable_reason
            from .models import ComputeNode
            for waiting in db.scalars(select(TrainingRun).where(
                TrainingRun.status == "queued", TrainingRun.execution_node_id.is_not(None),
            )):
                reason = remote_training_unavailable_reason(db.get(ComputeNode, waiting.execution_node_id))
                # Prefix distinguishes node availability from a GPU reason
                # last reported by the Agent; clear it when the node recovers.
                if reason:
                    set_queue_reason(db, waiting, "节点不可用：" + reason)
                elif not waiting.queue_reason or waiting.queue_reason.startswith("节点不可用："):
                    set_queue_reason(db, waiting, "等待远程 Agent 领取任务")
            db.commit()
            queued_ids = db.scalars(select(TrainingRun.id).where(
                TrainingRun.status == "queued", TrainingRun.execution_node_id.is_(None),
            ).order_by(TrainingRun.queued_at)).all()
            local_gpus = gpu_inventory()
            for run_id in queued_ids:
                db.expire_all()
                run = db.get(TrainingRun, run_id)
                task = db.get(TrainingTask, run.task_id) if run else None
                if not run or not task or task.current_run_id != run.id:
                    continue
                if not self._available(db, task):
                    set_queue_reason(db, run, "所选 GPU 正被其他本地任务占用")
                    db.commit()
                    continue
                device = str((run.parameter_snapshot or {}).get("device", task.device) or "").strip().lower()
                if device in {"cpu", "mps"}:
                    assignments = [non_gpu_assignment("local", run.id, device)]
                elif device and all(part.strip().isdigit() for part in device.split(",")):
                    assignments = resolve_gpu_assignments(device, local_gpus, node_id="local")
                    if assignments is None:
                        set_queue_reason(db, run, "所选 GPU 不在本地设备清单中，请检查设备编号及驱动")
                        db.commit()
                        continue
                else:
                    assignments = [node_serial_assignment("local")]
                if not self._claim_queued_run(db, run.id, assignments):
                    waiting = db.get(TrainingRun, run_id)
                    if waiting and waiting.status == "queued":
                        set_queue_reason(db, waiting, "GPU 或节点资源已被预约，等待释放")
                        db.commit()
                    continue
                db.expire_all()
                claimed_run = db.get(TrainingRun, run.id)
                claimed_task = db.get(TrainingTask, task.id)
                if claimed_run and claimed_task and claimed_run.status == "starting":
                    self._launch(db, claimed_task, claimed_run)

    def _loop(self) -> None:
        while not self._stop.wait(2):
            try:
                self._run_once()
                self._record_success()
            except Exception as exc:
                self._record_error(exc)
                logger.exception("training_scheduler_iteration_failed")
                time.sleep(1)


scheduler = TrainingScheduler()
