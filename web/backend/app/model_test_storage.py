"""Filesystem ownership, capacity accounting, and cleanup for model-test history."""
from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DATA_ROOT,
    MODEL_TEST_HISTORY_QUOTA_BYTES,
    MODEL_TEST_IMAGE_RESERVE_BYTES,
    MODEL_TEST_MIN_FREE_BYTES,
    MODEL_TEST_TASK_QUOTA_BYTES,
)


HISTORY_ROOT = DATA_ROOT / "model-test-history"
WORK_ROOT = DATA_ROOT / "model-test-work"
_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TEST_ID = re.compile(r"^[0-9a-f]{32}$")
_RESERVATION_FILE = ".storage-reservation.json"
_RESERVATION_VERSION = 1
_FIXED_TEST_RESERVE_BYTES = 64 * 1024**2
_capacity_lock = threading.RLock()
STANDALONE_HISTORY_SCOPE = "_standalone"


class ModelTestCapacityError(RuntimeError):
    """A deterministic capacity refusal suitable for an HTTP 507 response."""

    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(detail)


@dataclass(frozen=True)
class ModelTestStorageEstimate:
    """The conservative space reservation for one pending model test."""

    history_bytes: int
    peak_bytes: int
    model_bytes: int
    declared_upload_bytes: int
    image_bytes: int
    fixed_bytes: int


@dataclass(frozen=True)
class ModelTestStorageUsage:
    """Accounted storage before a new test is admitted."""

    history_bytes: int
    task_bytes: int
    free_bytes: int
    pending_peak_bytes: int


def history_scope(task_id: str | None) -> str:
    """Map taskless tests to a reserved owner that cannot collide with UUID task IDs."""
    return task_id or STANDALONE_HISTORY_SCOPE


def _safe_child(root: Path, *parts: str) -> Path:
    resolved_root = root.resolve()
    candidate = resolved_root
    for part in parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError("拒绝访问符号链接模型检测目录")
    resolved_candidate = candidate.resolve()
    if resolved_candidate == resolved_root or resolved_root not in resolved_candidate.parents:
        raise ValueError("模型检测目录超出允许范围")
    return candidate


def task_history_root(task_id: str) -> Path:
    if not _TASK_ID.fullmatch(task_id):
        raise ValueError("训练任务 ID 不合法")
    return _safe_child(HISTORY_ROOT, task_id)


def test_history_root(task_id: str, test_id: str) -> Path:
    if not _TEST_ID.fullmatch(test_id):
        raise ValueError("模型检测 ID 不合法")
    return _safe_child(HISTORY_ROOT, task_id, test_id)


def test_work_root(test_id: str) -> Path:
    if not _TEST_ID.fullmatch(test_id):
        raise ValueError("模型检测 ID 不合法")
    return _safe_child(WORK_ROOT, test_id)


def reservation_path(task_id: str, test_id: str) -> Path:
    return test_history_root(task_id, test_id) / _RESERVATION_FILE


def _delete_directory(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"拒绝清理非受管目录：{path}")
    shutil.rmtree(path)


def delete_test_history(task_id: str, test_id: str) -> None:
    with _capacity_lock:
        _delete_directory(test_history_root(task_id, test_id))
        parent = task_history_root(task_id)
        try:
            parent.rmdir()
        except OSError:
            pass


def delete_task_history(task_id: str) -> None:
    """Idempotently remove every persistent model-test file owned by a task."""
    with _capacity_lock:
        _delete_directory(task_history_root(task_id))


def delete_test_work(test_id: str) -> None:
    _delete_directory(test_work_root(test_id))


def directory_size(path: Path) -> int:
    total = 0
    if not path.is_dir() or path.is_symlink():
        return total
    for child in path.rglob("*"):
        if child.is_file() and not child.is_symlink():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def estimate_test_storage(*, model_bytes: int, declared_upload_bytes: int, image_count: int) -> ModelTestStorageEstimate:
    """Estimate peak disk use until one test becomes a terminal history record.

    Each image can exist as a normalized original, an annotated result, and a
    ZIP member. The source upload and weight copy are also present while the
    worker runs. The image reserve is intentionally independent of the browser
    supplied byte count because a small compressed image may expand when it is
    normalized to JPEG.
    """
    if model_bytes < 0 or declared_upload_bytes < 0 or image_count < 0:
        raise ValueError("模型测试存储预留不能为负数")
    image_bytes = max(declared_upload_bytes, image_count * MODEL_TEST_IMAGE_RESERVE_BYTES)
    history_bytes = image_bytes * 3 + _FIXED_TEST_RESERVE_BYTES
    return ModelTestStorageEstimate(
        history_bytes=history_bytes,
        peak_bytes=history_bytes + model_bytes + declared_upload_bytes,
        model_bytes=model_bytes,
        declared_upload_bytes=declared_upload_bytes,
        image_bytes=image_bytes,
        fixed_bytes=_FIXED_TEST_RESERVE_BYTES,
    )


def _reserved_bytes(path: Path) -> int:
    candidate = path / _RESERVATION_FILE
    if candidate.is_symlink() or not candidate.is_file():
        return 0
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    value = payload.get("history_bytes") if isinstance(payload, dict) and payload.get("version") == _RESERVATION_VERSION else None
    return value if isinstance(value, int) and value >= 0 else 0


def _reserved_peak_bytes(path: Path) -> int:
    candidate = path / _RESERVATION_FILE
    if candidate.is_symlink() or not candidate.is_file():
        return 0
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return 0
    value = payload.get("peak_bytes") if isinstance(payload, dict) and payload.get("version") == _RESERVATION_VERSION else None
    return value if isinstance(value, int) and value >= 0 else 0


def _task_reservation_adjustment(task: Path) -> int:
    """Add only reservation capacity that is not already occupied on disk."""
    if not task.is_dir() or task.is_symlink():
        return 0
    adjustment = 0
    for test in task.iterdir():
        if test.is_symlink() or not test.is_dir() or not _TEST_ID.fullmatch(test.name):
            continue
        actual = directory_size(test)
        adjustment += max(0, _reserved_bytes(test) - actual)
    return adjustment


def accounted_history_bytes(task_id: str | None = None) -> int:
    """Physical bytes plus reservations for active tests, including orphan files."""
    if task_id is not None:
        root = task_history_root(task_id)
        return directory_size(root) + _task_reservation_adjustment(root)
    root = HISTORY_ROOT.resolve()
    if not root.is_dir():
        return 0
    adjustment = 0
    for task in root.iterdir():
        if task.is_symlink() or not task.is_dir() or not _TASK_ID.fullmatch(task.name):
            continue
        adjustment += _task_reservation_adjustment(task)
    return directory_size(root) + adjustment


def pending_peak_bytes() -> int:
    """Return reserved peak bytes that have not materialized on disk yet."""
    root = HISTORY_ROOT.resolve()
    if not root.is_dir():
        return 0
    pending = 0
    for task in root.iterdir():
        if task.is_symlink() or not task.is_dir() or not _TASK_ID.fullmatch(task.name):
            continue
        for test in task.iterdir():
            if test.is_symlink() or not test.is_dir() or not _TEST_ID.fullmatch(test.name):
                continue
            peak = _reserved_peak_bytes(test)
            if not peak:
                continue
            materialized = directory_size(test) + directory_size(test_work_root(test.name))
            pending += max(0, peak - materialized)
    return pending


def storage_usage(task_id: str) -> ModelTestStorageUsage:
    try:
        disk_path = HISTORY_ROOT.resolve() if HISTORY_ROOT.exists() else HISTORY_ROOT.parent.resolve()
        free_bytes = shutil.disk_usage(disk_path).free
        history_bytes = accounted_history_bytes()
        task_bytes = accounted_history_bytes(task_id)
        pending_bytes = pending_peak_bytes()
    except OSError as exc:
        raise ModelTestCapacityError("disk_unavailable", "无法读取模型测试存储所在磁盘的可用空间") from exc
    return ModelTestStorageUsage(
        history_bytes=history_bytes,
        task_bytes=task_bytes,
        free_bytes=free_bytes,
        pending_peak_bytes=pending_bytes,
    )


def _format_gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def ensure_test_capacity(task_id: str, estimate: ModelTestStorageEstimate) -> ModelTestStorageUsage:
    """Refuse a new test before any files are created.

    The caller must hold `_capacity_lock` through `prepare_test_history()` so a
    concurrent request observes the first request's durable reservation.
    """
    usage = storage_usage(task_id)
    if MODEL_TEST_HISTORY_QUOTA_BYTES and usage.history_bytes + estimate.history_bytes > MODEL_TEST_HISTORY_QUOTA_BYTES:
        raise ModelTestCapacityError(
            "history_quota_exceeded",
            "模型测试历史总配额不足：已占用 "
            f"{_format_gib(usage.history_bytes)}，本次需预留 {_format_gib(estimate.history_bytes)}，"
            f"总配额为 {_format_gib(MODEL_TEST_HISTORY_QUOTA_BYTES)}",
        )
    if MODEL_TEST_TASK_QUOTA_BYTES and usage.task_bytes + estimate.history_bytes > MODEL_TEST_TASK_QUOTA_BYTES:
        raise ModelTestCapacityError(
            "task_quota_exceeded",
            "当前训练任务的模型测试历史配额不足：已占用 "
            f"{_format_gib(usage.task_bytes)}，本次需预留 {_format_gib(estimate.history_bytes)}，"
            f"任务配额为 {_format_gib(MODEL_TEST_TASK_QUOTA_BYTES)}",
        )
    required_free = usage.pending_peak_bytes + estimate.peak_bytes + MODEL_TEST_MIN_FREE_BYTES
    if usage.free_bytes < required_free:
        raise ModelTestCapacityError(
            "low_disk_space",
            "服务器模型测试存储空间不足：可用 "
            f"{_format_gib(usage.free_bytes)}，现有测试尚需 {_format_gib(usage.pending_peak_bytes)}，"
            f"本次峰值需预留 {_format_gib(estimate.peak_bytes)}，"
            f"并保留最低 {_format_gib(MODEL_TEST_MIN_FREE_BYTES)}",
        )
    return usage


def _write_reservation(path: Path, estimate: ModelTestStorageEstimate) -> None:
    temporary = path / f".{_RESERVATION_FILE}.part"
    temporary.write_text(
        json.dumps({
            "version": _RESERVATION_VERSION,
            "history_bytes": estimate.history_bytes,
            "peak_bytes": estimate.peak_bytes,
        }, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path / _RESERVATION_FILE)


def prepare_test_history(task_id: str, test_id: str, estimate: ModelTestStorageEstimate) -> Path:
    """Reserve capacity and create a private history directory atomically enough for API use."""
    directory = test_history_root(task_id, test_id)
    with _capacity_lock:
        ensure_test_capacity(task_id, estimate)
        created = False
        try:
            directory.mkdir(parents=True, exist_ok=False)
            created = True
            _write_reservation(directory, estimate)
            (directory / "images").mkdir()
            return directory
        except Exception as exc:
            if created:
                try:
                    _delete_directory(directory)
                except (OSError, ValueError):
                    pass
            if isinstance(exc, OSError):
                raise ModelTestCapacityError(
                    "reservation_write_failed", "无法在服务器上预留模型测试存储空间",
                ) from exc
            raise


def release_test_reservation(task_id: str, test_id: str) -> None:
    """Replace an active reservation with the actual terminal record size."""
    with _capacity_lock:
        path = reservation_path(task_id, test_id)
        if path.is_symlink():
            raise ValueError("拒绝清理符号链接模型测试预留文件")
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def reconcile_test_reservations(active_tests: set[tuple[str, str]]) -> None:
    """Release stale reservations left after a terminal commit or interrupted create."""
    root = HISTORY_ROOT.resolve()
    if not root.is_dir():
        return
    with _capacity_lock:
        for task in root.iterdir():
            if task.is_symlink() or not task.is_dir() or not _TASK_ID.fullmatch(task.name):
                continue
            for test in task.iterdir():
                if test.is_symlink() or not test.is_dir() or not _TEST_ID.fullmatch(test.name):
                    continue
                if (task.name, test.name) not in active_tests:
                    release_test_reservation(task.name, test.name)
