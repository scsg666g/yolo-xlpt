from __future__ import annotations

import csv
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select

from .config import DATASET_ROOTS, MODEL_ROOTS, RUNS_ROOT, ULTRALYTICS_ROOT, YOLO_BIN
from .database import SessionLocal
from .models import Artifact, MetricPoint, TaskEvent, TrainingTask, utcnow


TRAINING_KEYS = {
    "patience", "lr0", "pretrained", "cache", "amp", "close_mosaic", "cos_lr", "weight_decay",
    "momentum", "warmup_epochs", "freeze", "multi_scale", "rect", "seed", "deterministic", "fraction",
    "save_period", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste", "box", "cls", "dfl",
}


def add_event(db, task_id: str, event_type: str, message: str, level: str = "info", payload: dict | None = None):
    db.add(TaskEvent(task_id=task_id, event_type=event_type, message=message, level=level, payload=payload or {}))


def path_is_within(path: Path, roots: list[Path]) -> bool:
    resolved = path.expanduser().resolve()
    return any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots)


def validate_input_path(raw: str, roots: list[Path], label: str) -> Path:
    path = Path(raw).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{label}不存在：{path}")
    if not path_is_within(path, roots):
        raise ValueError(f"{label}必须位于允许的目录中")
    return path


def discover_files(roots: list[Path], suffixes: set[str]) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in suffixes:
                stat = path.stat()
                files.append({"name": path.name, "path": str(path), "size": stat.st_size, "root": str(root)})
    return sorted(files, key=lambda item: item["name"].lower())


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    result = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 6:
            result.append({
                "index": int(parts[0]), "name": parts[1], "memory_total": int(parts[2]),
                "memory_used": int(parts[3]), "utilization": int(parts[4]), "temperature": int(parts[5]),
            })
    return result


def parse_number(value: Any) -> float | str | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def sync_metrics(db, task: TrainingTask) -> None:
    if not task.run_dir:
        return
    csv_path = Path(task.run_dir) / "results.csv"
    if not csv_path.exists():
        return
    existing = set(db.scalars(select(MetricPoint.epoch).where(MetricPoint.task_id == task.id)).all())
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return
    latest: dict[str, Any] = {}
    latest_epoch = task.completed_epochs
    for row in rows:
        try:
            epoch = int(float(row.get("epoch", "0")))
        except ValueError:
            continue
        values = {key.strip(): parse_number(value) for key, value in row.items() if key and key.strip() not in {"epoch", "time"}}
        elapsed = float(row.get("time") or 0)
        if epoch not in existing:
            db.add(MetricPoint(task_id=task.id, epoch=epoch, elapsed=elapsed, values=values))
        latest, latest_epoch = values, max(latest_epoch, epoch)
    if rows:
        task.completed_epochs = latest_epoch
        task.progress = min(100.0, round(latest_epoch / max(task.epochs, 1) * 100, 2))
        task.latest_metrics = latest


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


def sync_artifacts(db, task: TrainingTask) -> None:
    if not task.run_dir or not Path(task.run_dir).exists():
        return
    known = set(db.scalars(select(Artifact.path).where(Artifact.task_id == task.id)).all())
    for path in Path(task.run_dir).rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        raw = str(path.resolve())
        if raw in known:
            continue
        stat = path.stat()
        db.add(Artifact(
            task_id=task.id,
            kind=artifact_kind(path),
            name=str(path.relative_to(task.run_dir)),
            path=raw,
            size=stat.st_size,
            modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc),
        ))


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
        self.processes: dict[str, subprocess.Popen] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with SessionLocal() as db:
            running = db.scalars(select(TrainingTask).where(TrainingTask.status.in_(["starting", "running", "stopping"]))).all()
            for task in running:
                task.status = "interrupted"
                task.pid = None
                task.error_message = "平台服务重启后未发现受管训练进程"
                add_event(db, task.id, "interrupted", task.error_message, "warning")
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

    def _command(self, task: TrainingTask) -> list[str]:
        config = task.config or {}
        command = [
            str(YOLO_BIN), task.task_type, "train",
            f"model={task.model_path}", f"data={task.data_path}", f"epochs={task.epochs}",
            f"batch={task.batch}", f"imgsz={task.imgsz}", f"device={task.device}",
            f"workers={task.workers}", f"optimizer={task.optimizer}",
            f"project={RUNS_ROOT / task.task_type}", f"name={task.id}", "exist_ok=True",
        ]
        for key, value in config.items():
            if key in TRAINING_KEYS and value is not None:
                command.append(f"{key}={value}")
        for key, value in (config.get("custom_args") or {}).items():
            command.append(f"{key}={value}")
        return command

    def _launch(self, db, task: TrainingTask) -> None:
        run_dir = RUNS_ROOT / task.task_type / task.id
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "platform.log"
        command = self._command(task)
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        log_handle.write(f"\n[{utcnow().isoformat()}] COMMAND {json.dumps(command, ensure_ascii=False)}\n")
        try:
            process = subprocess.Popen(
                command,
                cwd=ULTRALYTICS_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            log_handle.close()
            task.status = "failed"
            task.error_message = str(exc)
            task.finished_at = utcnow()
            add_event(db, task.id, "failed", f"训练进程启动失败：{exc}", "error")
            return
        task.status = "running"
        task.pid = process.pid
        task.started_at = utcnow()
        task.run_dir = str(run_dir)
        task.log_path = str(log_path)
        task.error_message = None
        self.processes[task.id] = process
        add_event(db, task.id, "started", f"训练已启动，PID {process.pid}", payload={"command": command})

    def stop_task(self, task_id: str) -> bool:
        with SessionLocal() as db:
            task = db.get(TrainingTask, task_id)
            if not task:
                return False
            if task.status == "queued":
                task.status = "stopped"
                task.finished_at = utcnow()
                add_event(db, task.id, "stopped", "已从等待队列移除")
                db.commit()
                return True
            process = self.processes.get(task_id)
            if task.status not in {"starting", "running"} or not process:
                return False
            task.status = "stopping"
            add_event(db, task.id, "stopping", "正在停止训练进程")
            db.commit()
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            return True

    def _poll_processes(self, db) -> None:
        for task_id, process in list(self.processes.items()):
            task = db.get(TrainingTask, task_id)
            if not task:
                continue
            sync_metrics(db, task)
            code = process.poll()
            if code is None:
                continue
            task.exit_code = code
            task.pid = None
            task.finished_at = utcnow()
            if task.status == "stopping":
                task.status = "stopped"
                message, level = "训练已停止", "info"
            elif code == 0:
                task.status = "completed"
                task.progress = 100.0
                message, level = "训练已完成", "info"
            else:
                task.status = "failed"
                task.error_message = f"训练进程异常退出，退出码 {code}"
                message, level = task.error_message, "error"
            sync_artifacts(db, task)
            add_event(db, task.id, task.status, message, level)
            self.processes.pop(task_id, None)

    def _loop(self) -> None:
        while not self._stop.wait(2):
            try:
                with SessionLocal() as db:
                    self._poll_processes(db)
                    queued = db.scalars(
                        select(TrainingTask).where(TrainingTask.status == "queued").order_by(TrainingTask.queued_at)
                    ).all()
                    for task in queued:
                        if self._available(db, task):
                            task.status = "starting"
                            self._launch(db, task)
                    db.commit()
            except Exception:
                # Keep the scheduler alive; errors are surfaced by the next API health check/log inspection.
                time.sleep(1)


scheduler = TrainingScheduler()
