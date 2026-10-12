from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import threading
import time
import uuid
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
    RUNTIME_ROOT,
    RUNS_ROOT,
    TRAINING_MKL_THREADING_LAYER,
    ULTRALYTICS_ROOT,
    YOLO_BIN,
)
from .database import SessionLocal
from .models import Artifact, MetricPoint, TaskEvent, TrainingRun, TrainingTask, utcnow
from .parameter_catalog import training_parameter_keys


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


def training_environment() -> dict[str, str]:
    """Build a stable environment for PyTorch and its DDP worker processes."""
    environment = os.environ.copy()
    environment["MKL_THREADING_LAYER"] = TRAINING_MKL_THREADING_LAYER
    return environment


def failure_detail(log_path: str | None) -> str:
    if not log_path or not Path(log_path).exists():
        return ""
    lines = [
        ANSI_ESCAPE.sub("", line).strip()
        for line in Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    root_markers = ("Error:", "RuntimeError:", "ValueError:", "FileNotFoundError:", "OutOfMemoryError:")
    detail = next((line for line in reversed(lines) if line.startswith(root_markers)), "")
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
    task.error_message = run.error_message
    task.latest_metrics = dict(run.latest_metrics or {})
    task.queued_at = run.queued_at
    task.started_at = run.started_at
    task.finished_at = run.finished_at


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


def create_training_run(
    db, task: TrainingTask, launch_mode: str, checkpoint_path: Path | None = None,
) -> TrainingRun:
    attempt = (db.scalar(select(func.max(TrainingRun.attempt_number)).where(TrainingRun.task_id == task.id)) or 0) + 1
    attempt_name = f"attempt-{attempt:03d}"
    run = TrainingRun(
        id=uuid.uuid4().hex,
        task_id=task.id,
        attempt_number=attempt,
        launch_mode=launch_mode,
        status="queued",
        execution_node_id=task.execution_node_id,
        parameter_snapshot=parameter_snapshot(task),
        model_path=task.model_path,
        data_path=task.data_path,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        run_dir=(str(Path((task.config or {}).get("remote_output_root")) / task.task_type / task.id / attempt_name)
                 if task.execution_node_id and (task.config or {}).get("remote_output_root") else str(RUNS_ROOT / task.task_type / task.id / attempt_name)),
        log_path=(str(Path((task.config or {}).get("remote_output_root")) / task.task_type / task.id / attempt_name / "platform.log")
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


def _safe_delete_directory(path: Path, roots: list[Path]) -> None:
    """Delete a run directory only when it is below an explicitly allowed root."""
    candidate = path.expanduser().resolve()
    allowed = [root.expanduser().resolve() for root in roots]
    if not path_is_within(candidate, allowed) or candidate in allowed:
        raise ValueError(f"训练目录不在允许的删除范围内：{candidate}")
    if not candidate.exists():
        return
    if candidate.is_symlink() or not candidate.is_dir():
        candidate.unlink()
    else:
        shutil.rmtree(candidate)


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
        candidate = target.expanduser().resolve()
        allowed = [root.expanduser().resolve() for root in deletion_roots]
        if not path_is_within(candidate, allowed) or candidate in allowed:
            raise ValueError(f"训练目录不在允许的删除范围内：{candidate}")
    deleted: list[str] = []
    for target in targets:
        if target.exists():
            _safe_delete_directory(target, deletion_roots)
            deleted.append(str(target.resolve()))
    return deleted


def validate_input_path(raw: str, roots: list[Path], label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label}不存在：{path}")
    if not path_is_within(path, roots):
        raise ValueError(f"{label}必须位于允许的目录中")
    return path


def validate_dataset_config(raw: str) -> Path:
    """Validate the dataset YAML and resolve train/val entries before queuing a run."""
    path = validate_input_path(raw, DATASET_ROOTS, "数据集配置")
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


def discover_files(roots: list[Path], suffixes: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                stat = path.stat()
                name = path.parent.name if path.name in {"data.yaml", "data.yml"} and root in DATASET_ROOTS else path.name
                files.append({"name": name, "path": str(path), "size": stat.st_size, "root": str(root)})
    return sorted(files, key=lambda item: item["name"].lower())


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
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def sync_metrics(db, task: TrainingTask, run: TrainingRun | None = None) -> None:
    run = run or current_run(db, task)
    if not run or run.execution_node_id or not run.run_dir:
        return
    csv_path = Path(run.run_dir) / "results.csv"
    if not csv_path.exists():
        return
    existing = set(db.scalars(select(MetricPoint.epoch).where(MetricPoint.run_id == run.id)).all())
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return
    latest: dict[str, Any] = {}
    latest_epoch = run.completed_epochs
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "0")))
        except ValueError:
            continue
        values = {key.strip(): parse_number(value) for key, value in row.items() if key and key.strip() not in {"epoch", "time"}}
        elapsed = float(row.get("time") or 0)
        if epoch not in existing:
            try:
                with db.begin_nested():
                    db.add(MetricPoint(task_id=task.id, run_id=run.id, epoch=epoch, elapsed=elapsed, values=values))
                    db.flush()
            except IntegrityError:
                pass
        latest, latest_epoch = values, max(latest_epoch, epoch)
    if rows:
        epochs = int((run.parameter_snapshot or {}).get("epochs") or task.epochs)
        run.completed_epochs = latest_epoch
        run.progress = min(100.0, round(latest_epoch / max(epochs, 1) * 100, 2))
        run.latest_metrics = latest
        if task.current_run_id == run.id:
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


def resume_checkpoint(task: TrainingTask) -> Path | None:
    """Return the checkpoint written by the latest incomplete run, if available."""
    if not task.run_dir:
        return None
    checkpoint = Path(task.run_dir) / "weights" / "last.pt"
    return checkpoint if checkpoint.is_file() else None


def sync_artifacts(db, task: TrainingTask, run: TrainingRun | None = None) -> None:
    run = run or current_run(db, task)
    if not run or run.execution_node_id or not run.run_dir or not Path(run.run_dir).exists():
        return
    known = {
        artifact.path: artifact
        for artifact in db.scalars(select(Artifact).where(Artifact.run_id == run.id)).all()
    }
    for path in Path(run.run_dir).rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        raw = str(path.resolve())
        stat = path.stat()
        modified_at = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        existing = known.get(raw)
        if existing:
            existing.kind = artifact_kind(path)
            existing.name = str(path.relative_to(run.run_dir))
            existing.size = stat.st_size
            existing.modified_at = modified_at
            continue
        values = {
            "task_id": task.id,
            "run_id": run.id,
            "kind": artifact_kind(path),
            "name": str(path.relative_to(run.run_dir)),
            "path": raw,
            "size": stat.st_size,
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


def import_existing_runs() -> int:
    runs_root = ULTRALYTICS_ROOT / "runs"
    if not runs_root.exists():
        return 0
    imported = 0
    with SessionLocal() as db:
        for args_path in runs_root.rglob("args.yaml"):
            run_dir = args_path.parent.resolve()
            task_id = "legacy-" + hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]
            if db.get(TrainingTask, task_id):
                task = db.get(TrainingTask, task_id)
                if task.deleted_at:
                    continue
                sync_metrics(db, task)
                sync_artifacts(db, task)
                continue
            try:
                args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if args.get("mode") != "train":
                continue
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
            imported += 1
        db.commit()
    return imported


class TrainingScheduler:
    def __init__(self) -> None:
        self.processes: dict[str, tuple[str, subprocess.Popen]] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with SessionLocal() as db:
            running = db.scalars(select(TrainingRun).where(
                TrainingRun.status.in_(["starting", "running", "stopping"]),
                TrainingRun.execution_node_id.is_(None),
            )).all()
            for run in running:
                task = db.get(TrainingTask, run.task_id)
                run.status = "interrupted"
                run.pid = None
                run.error_message = "平台服务重启后未发现受管训练进程"
                if task:
                    mirror_run_to_task(task, run)
                    add_event(db, task.id, "interrupted", run.error_message, "warning", run_id=run.id)
            db.commit()
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
        active = db.scalars(select(TrainingTask).where(TrainingTask.status.in_(["starting", "running", "stopping"]))).all()
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
        return command

    def _launch(self, db, task: TrainingTask, run: TrainingRun) -> None:
        run_dir = Path(run.run_dir or RUNS_ROOT / task.task_type / task.id / f"attempt-{run.attempt_number:03d}")
        runtime_dir = RUNTIME_ROOT / task.id / f"attempt-{run.attempt_number:03d}"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(run.log_path) if run.log_path else log_path_for_run(task.id, run.attempt_number)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            dataset_path = prepare_dataset_config(run.data_path, runtime_dir / "dataset.yaml")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            run.status = "failed"
            run.error_message = str(exc)
            run.finished_at = utcnow()
            mirror_run_to_task(task, run)
            add_event(db, task.id, "failed", f"数据集配置校验失败：{exc}", "error", run_id=run.id)
            return
        resume_path = Path(run.checkpoint_path).resolve() if run.checkpoint_path else None
        if resume_path and not resume_path.is_file():
            run.status = "failed"
            run.error_message = f"断点检查点不存在：{resume_path}"
            run.finished_at = utcnow()
            mirror_run_to_task(task, run)
            add_event(db, task.id, "failed", run.error_message, "error", run_id=run.id)
            return
        command = self._command(task, run, dataset_path)
        run.command = command
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(f"\n[{utcnow().isoformat()}] COMMAND {json.dumps(command, ensure_ascii=False)}\n")
        log_handle.write(f"[{utcnow().isoformat()}] ENV MKL_THREADING_LAYER={TRAINING_MKL_THREADING_LAYER}\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=ULTRALYTICS_ROOT,
                env=training_environment(),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            run.status = "failed"
            run.error_message = str(exc)
            run.finished_at = utcnow()
            mirror_run_to_task(task, run)
            add_event(db, task.id, "failed", f"训练进程启动失败：{exc}", "error", run_id=run.id)
            return
        log_handle.close()
        run.status = "running"
        run.pid = process.pid
        run.started_at = utcnow()
        run.process_started_at = run.started_at
        run.run_dir = str(run_dir)
        run.log_path = str(log_path)
        run.error_message = None
        mirror_run_to_task(task, run)
        self.processes[task.id] = (run.id, process)
        add_event(db, task.id, "started", f"Attempt {run.attempt_number:03d} 已启动，PID {process.pid}", payload={"command": command}, run_id=run.id)

    def stop_task(self, task_id: str) -> bool:
        with SessionLocal() as db:
            task = db.get(TrainingTask, task_id)
            if not task:
                return False
            if task.status == "queued":
                run = current_run(db, task)
                if not run:
                    return False
                run.status = "stopped"
                run.finished_at = utcnow()
                mirror_run_to_task(task, run)
                add_event(db, task.id, "stopped", "已从等待队列移除", run_id=run.id)
                db.commit()
                return True
            managed = self.processes.get(task_id)
            if task.status not in {"starting", "running"} or not managed:
                return False
            run_id, process = managed
            run = db.get(TrainingRun, run_id)
            if not run:
                return False
            run.status = "stopping"
            mirror_run_to_task(task, run)
            add_event(db, task.id, "stopping", "正在停止训练进程", run_id=run.id)
            db.commit()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return True

    def _poll_processes(self, db) -> None:
        for task_id, (run_id, process) in list(self.processes.items()):
            task = db.get(TrainingTask, task_id)
            run = db.get(TrainingRun, run_id)
            if not task or not run:
                continue
            sync_metrics(db, task, run)
            code = process.poll()
            if code is None:
                continue
            run.exit_code = code
            run.pid = None
            run.finished_at = utcnow()
            if run.status == "stopping":
                run.status = "stopped"
                message, level = "训练已停止", "info"
            elif code == 0:
                run.status = "completed"
                run.progress = 100.0
                message, level = "训练已完成", "info"
            else:
                run.status = "failed"
                detail = failure_detail(run.log_path)
                run.error_message = f"训练进程异常退出，退出码 {code}"
                if detail:
                    run.error_message += f"：{detail[-500:]}"
                message, level = run.error_message, "error"
            mirror_run_to_task(task, run)
            sync_artifacts(db, task, run)
            add_event(db, task.id, run.status, message, level, run_id=run.id)
            self.processes.pop(task_id, None)

    def _loop(self) -> None:
        while not self._stop.wait(2):
            try:
                with SessionLocal() as db:
                    self._poll_processes(db)
                    queued = db.scalars(select(TrainingRun).where(TrainingRun.status == "queued", TrainingRun.execution_node_id.is_(None)).order_by(TrainingRun.queued_at)).all()
                    for run in queued:
                        task = db.get(TrainingTask, run.task_id)
                        if task and task.current_run_id == run.id and self._available(db, task):
                            run.status = "starting"
                            mirror_run_to_task(task, run)
                            self._launch(db, task, run)
                    db.commit()
            except Exception:
                # Keep the scheduler alive; errors are surfaced by the next API health check/log inspection.
                time.sleep(1)


scheduler = TrainingScheduler()
