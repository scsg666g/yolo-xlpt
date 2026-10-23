"""Resolve a finished training attempt without inventing a recovered exit code."""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


RESULT_FILENAME = "training-result.json"
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EPOCHS_COMPLETED = re.compile(r"\b(\d+)\s+epochs? completed in\b", re.IGNORECASE)
FATAL_ERROR = re.compile(
    r"^(?:Traceback \(most recent call last\):|(?:[\w.]*Error|Exception):|"
    r"(?:torch\.)?distributed\.elastic\.multiprocessing\.errors\.ChildFailedError|"
    r"Killed(?:\s|$)|Segmentation fault(?:\s|$))"
)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _integer(value: Any, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _read_record(run_dir: Path | None, run_id: str, task_id: str, started_at: datetime | None) -> dict | None:
    if run_dir is None or not run_id or not task_id:
        return None
    path = run_dir / RESULT_FILENAME
    try:
        if path.stat().st_size > 128 * 1024:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or type(record.get("schema_version")) is not int or record["schema_version"] != 1:
            return None
        if record.get("run_id") != run_id or record.get("task_id") != task_id:
            return None
        status, code = record.get("status"), record.get("exit_code")
        if not isinstance(code, int) or isinstance(code, bool) or status not in {"completed", "failed"}:
            return None
        if (status == "completed" and code != 0) or (status == "failed" and code == 0):
            return None
        if not _integer(record.get("completed_epochs")) or not _integer(record.get("expected_epochs"), 1):
            return None
        if record.get("reason") not in {"early_stopping", "epochs_completed", "training_completed", "exception"}:
            return None
        if (status == "failed") != (record["reason"] == "exception"):
            return None
        if record.get("best_epoch") is not None and not _integer(record["best_epoch"], 1):
            return None
        if record.get("best_epoch", 0) > record["completed_epochs"]:
            return None
        if record.get("patience") is not None and not _integer(record["patience"]):
            return None
        if record.get("error_message") is not None and not isinstance(record["error_message"], str):
            return None
        finished_at = datetime.fromisoformat(record["finished_at"].replace("Z", "+00:00"))
        if finished_at.tzinfo is None or _utc(finished_at) > datetime.now(timezone.utc) + timedelta(seconds=10):
            return None
        if started_at is not None and (
            _utc(finished_at) < _utc(started_at)
            or path.stat().st_mtime < _utc(started_at).timestamp() - 1
        ):
            return None
        metrics = record.get("final_metrics")
        if metrics is not None and (
            not isinstance(metrics, dict)
            or any(not isinstance(key, str) or isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for key, value in metrics.items())
        ):
            return None
        return record
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


def _log_lines(log_path: Path | None) -> list[str]:
    if log_path is None:
        return []
    try:
        # Completion and final validation are at the end, even for very long runs.
        with log_path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024))
            content = handle.read().decode("utf-8", errors="replace")
        return [re.sub(r"^\[rank\d+\]:\s*", "", ANSI_ESCAPE.sub("", line)).strip()
                for line in content.splitlines() if line.strip()]
    except OSError:
        return []


def _legacy_completion(run_dir: Path | None, lines: list[str], started_at: datetime | None,
                       expected_epochs: int, completed_epochs: int) -> dict | None:
    if run_dir is None or started_at is None:
        return None
    completion = [(index, match) for index, line in enumerate(lines) if (match := EPOCHS_COMPLETED.search(line))]
    if not completion:
        return None
    index, match = completion[-1]
    tail = lines[index + 1:]
    def matches_directory(line: str) -> bool:
        saved = re.search(r"results saved to\s+(.+)$", line, re.IGNORECASE)
        if not saved:
            return False
        path = Path(saved[1].strip().strip("\"'"))
        if path.is_absolute():
            return path.resolve() == run_dir.resolve()
        # Ultralytics also logs paths relative to its training working directory.
        return len(path.parts) >= 2 and run_dir.resolve().parts[-len(path.parts):] == path.parts

    if not any(matches_directory(line) for line in tail) or any(FATAL_ERROR.match(line) for line in tail):
        return None
    try:
        started = _utc(started_at).timestamp() - 1
        csv_path = run_dir / "results.csv"
        if csv_path.stat().st_mtime < started:
            return None
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            epochs = [int(float(value)) for row in reader for key, value in row.items()
                      if key and key.strip() == "epoch" and value is not None]
        actual_epochs = max(epochs, default=0)
        if actual_epochs < max(int(match[1]), completed_epochs, 1):
            return None
        weights = [run_dir / "weights" / name for name in ("best.pt", "last.pt")]
        if not any(path.is_file() and path.stat().st_size > 0 and path.stat().st_mtime >= started for path in weights):
            return None
    except (OSError, ValueError, TypeError):
        return None
    info: dict[str, Any] = {
        "source": "legacy_log", "reason": "epochs_completed" if actual_epochs >= expected_epochs else "training_completed",
        "completed_epochs": actual_epochs, "expected_epochs": expected_epochs,
    }
    early_text = "\n".join(lines[max(0, index - 12):index + 1])
    best = re.search(r"Best results observed at epoch\s+(\d+)", early_text, re.IGNORECASE)
    patience = re.search(r"no improvement observed in last\s+(\d+)\s+epochs", early_text, re.IGNORECASE)
    if actual_epochs < expected_epochs and "earlystopping:" in early_text.lower() and best and patience:
        info.update(reason="early_stopping", best_epoch=int(best[1]), patience=int(patience[1]))
    return info


def classify_training_exit(
    run_dir: Path | str | None, log_path: Path | str | None, run_id: str, task_id: str,
    expected_epochs: int, completed_epochs: int, started_at: datetime | None,
    actual_exit_code: int | None = None,
) -> dict[str, Any]:
    """Prefer a bound runner receipt; use strict legacy evidence for adopted runs."""
    directory, log = Path(run_dir) if run_dir else None, Path(log_path) if log_path else None
    record = _read_record(directory, run_id, task_id, started_at)
    lines = _log_lines(log)
    error = next((line for line in reversed(lines) if FATAL_ERROR.match(line)), None)
    result: dict[str, Any] = {"status": "interrupted", "exit_code": actual_exit_code,
                              "error_message": "平台接管的训练进程已退出，缺少可靠的训练结束记录", "completion_info": None}
    info = None
    if record is not None:
        info = {key: record[key] for key in ("reason", "completed_epochs", "expected_epochs", "best_epoch", "patience",
                                            "finished_at", "final_metrics") if key in record}
        info["source"] = "training_result"
    if actual_exit_code is not None and actual_exit_code != 0:
        result.update(status="failed", error_message=(record or {}).get("error_message") or error
                      or f"训练进程异常退出，退出码 {actual_exit_code}")
        if record is not None and record["status"] == "failed":
            result["completion_info"] = info
        return result
    if record is not None:
        result.update(status=record["status"], exit_code=record["exit_code"],
                      error_message=(record.get("error_message") or "训练执行程序记录到异常退出") if record["status"] == "failed" else None,
                      completion_info=info)
        return result
    info = _legacy_completion(directory, lines, started_at, expected_epochs, completed_epochs)
    if actual_exit_code == 0:
        result.update(status="completed", error_message=None, completion_info=info or {
            "source": "process_exit", "reason": "epochs_completed" if completed_epochs >= expected_epochs else "training_completed",
            "completed_epochs": completed_epochs, "expected_epochs": expected_epochs,
        })
    elif info is not None:
        result.update(status="completed", error_message=None, completion_info=info)
    elif error is not None:
        result.update(status="failed", error_message=error)
    return result
