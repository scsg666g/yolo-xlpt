"""Shared safety and cleanup helpers for deleting one training Attempt.

Task deletion has a separate lifecycle.  This module deliberately keeps an
Attempt deletion scoped to the selected run so model/data files and other
Attempts remain untouched.
"""
from __future__ import annotations

import hashlib
import posixpath
import re
from pathlib import Path, PurePosixPath

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from .config import LOG_ROOT, RUNTIME_ROOT, RUNS_ROOT, ULTRALYTICS_ROOT
from .models import Artifact, GpuReservation, MetricPoint, ModelTest, RemoteOperation, TaskEvent, TrainingRun, TrainingTask
from .services import (
    _safe_delete_directory,
    _validate_deletion_target,
    current_run,
    mirror_run_to_task,
)

ACTIVE_RUN_STATUSES = {"queued", "starting", "running", "stopping"}
ACTIVE_TEST_STATUSES = {"uploading", "preparing", "queued", "running", "cancel_requested"}
RUN_DELETE_KINDS = {"local_delete", "delete"}
ACTIVE_DELETE_STATUSES = {"pending", "running", "finalizing", "center_cleaning"}
CANONICAL_LAYOUT = "canonical"
IMPORTED_LAYOUT = "imported"
LEGACY_FLAT_LAYOUT = "legacy_flat"
ATTEMPT_DIRECTORY_PATTERN = re.compile(r"attempt-\d{3,}")
TASK_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TASK_TYPE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
ATTEMPT_PLAN_VERSION = 1


def is_attempt_payload(payload: dict | None, run_id: str) -> bool:
    payload = payload or {}
    return payload.get("scope") == "attempt" and payload.get("run_id") == run_id


def attempt_delete_operation(db: Session, run: TrainingRun, *, active_only: bool = False):
    """Return the newest center operation for this Attempt, if one exists."""
    query = select(RemoteOperation).where(
        RemoteOperation.task_id == run.task_id,
        RemoteOperation.kind.in_(RUN_DELETE_KINDS),
    ).order_by(RemoteOperation.created_at.desc())
    if active_only:
        query = query.where(RemoteOperation.status.in_(ACTIVE_DELETE_STATUSES))
    for operation in db.scalars(query):
        if is_attempt_payload(operation.payload, run.id):
            return operation
    return None


def active_attempt_delete_operation(db: Session, task_id: str):
    """Return active Attempt cleanup for a task, if any."""
    return db.scalar(select(RemoteOperation).where(
        RemoteOperation.task_id == task_id,
        RemoteOperation.kind.in_(RUN_DELETE_KINDS),
        RemoteOperation.status.in_(ACTIVE_DELETE_STATUSES),
        RemoteOperation.payload["scope"].as_string() == "attempt",
    ).order_by(RemoteOperation.created_at.desc()))


def attempt_deletion_state(db: Session, run: TrainingRun) -> tuple[str | None, str | None, object | None]:
    operation = attempt_delete_operation(db, run)
    if not operation:
        return None, None, None
    status = operation.status
    if status == "succeeded":
        return "succeeded", None, operation
    if status == "failed":
        return "failed", operation.error, operation
    if status in ACTIVE_DELETE_STATUSES:
        return ("pending" if status == "pending" else "running"), operation.error, operation
    return None, operation.error, operation


def _stored_run_dir(task: TrainingTask, run: TrainingRun) -> str | None:
    return run.run_dir or (task.run_dir if task.current_run_id == run.id else None)


def local_attempt_layout(task: TrainingTask, run: TrainingRun) -> tuple[str, Path]:
    """Classify a local Attempt without treating a legacy task root as disposable."""
    expected_run_dir = RUNS_ROOT / task.task_type / task.id / f"attempt-{run.attempt_number:03d}"
    task_root = _deletion_path(RUNS_ROOT / task.task_type / task.id)
    stored_run_dir = _stored_run_dir(task, run)
    run_dir = Path(stored_run_dir) if stored_run_dir else expected_run_dir
    run_dir = _deletion_path(run_dir)
    expected_run_dir = _deletion_path(expected_run_dir)
    if run_dir == expected_run_dir:
        return CANONICAL_LAYOUT, run_dir
    if _is_imported_legacy_run_dir(task, run, run_dir):
        return IMPORTED_LAYOUT, run_dir
    # Before TrainingRun used one directory per Attempt, the first platform
    # run wrote directly below <runs>/<type>/<task>.  That directory is now a
    # container for newer attempt-NNN children and must never be removed.
    if task.source == "platform" and run.attempt_number == 1 and run_dir == task_root:
        return LEGACY_FLAT_LAYOUT, run_dir
    raise ValueError("Attempt 训练目录与该任务/轮次不匹配，已拒绝清理")


def attempt_path_candidates(task: TrainingTask, run: TrainingRun) -> list[Path]:
    """Return only paths owned by this run on the local platform."""
    _, _, output_targets, log_target, runtime_target = _local_attempt_components(task, run)
    paths = list(output_targets)
    if log_target is not None:
        paths.append(log_target)
    paths.append(runtime_target)
    return list(dict.fromkeys(paths))


def _local_attempt_components(
    task: TrainingTask,
    run: TrainingRun,
) -> tuple[str, Path, list[Path], Path | None, Path]:
    layout, run_dir = local_attempt_layout(task, run)
    if layout == LEGACY_FLAT_LAYOUT:
        # Delete only root-level legacy output. Preserve every Attempt-shaped
        # child even when an old database lacks the corresponding run row.
        output_targets = [
            _deletion_path(child)
            for child in run_dir.iterdir()
            if not ATTEMPT_DIRECTORY_PATTERN.fullmatch(child.name)
        ] if run_dir.is_dir() else []
    else:
        output_targets = [run_dir]
    runtime_target = _deletion_path(RUNTIME_ROOT / task.id / f"attempt-{run.attempt_number:03d}")
    log_target: Path | None = None
    if run.log_path:
        log_path = _deletion_path(Path(run.log_path))
        allowed_logs = {
            _deletion_path(run_dir / "platform.log"),
            _deletion_path(runtime_target / "platform.log"),
        }
        log_root = _deletion_path(LOG_ROOT)
        is_platform_log = (
            log_path.parent == log_root
            and re.fullmatch(
                rf"\d{{8}}_{re.escape(task.id)}_attempt-{run.attempt_number:03d}\.log",
                log_path.name,
            ) is not None
        )
        if log_path not in allowed_logs and not is_platform_log:
            raise ValueError("Attempt 日志路径与该任务/轮次不匹配，已拒绝清理")
        log_target = log_path
    return layout, run_dir, output_targets, log_target, runtime_target


def remote_output_root_for_run(task: TrainingTask, run: TrainingRun) -> str | None:
    """Resolve the configured or canonical output root for a remote Attempt."""
    snapshot = run.parameter_snapshot or {}
    output_root = snapshot.get("remote_output_root") or (task.config or {}).get("remote_output_root")
    if output_root:
        return str(output_root)
    stored_run_dir = _stored_run_dir(task, run)
    if not stored_run_dir:
        return None
    candidate = PurePosixPath(stored_run_dir)
    expected_attempt = f"attempt-{run.attempt_number:03d}"
    if (
        candidate.is_absolute()
        and candidate.name == expected_attempt
        and candidate.parent.name == task.id
        and candidate.parent.parent.name == task.task_type
        and ".." not in candidate.parts
    ):
        return str(candidate.parents[2])
    if (
        task.source == "platform"
        and run.attempt_number == 1
        and candidate.is_absolute()
        and candidate.name == task.id
        and candidate.parent.name == task.task_type
        and ".." not in candidate.parts
    ):
        return str(candidate.parents[1])
    return None


def remote_attempt_layout(task: TrainingTask, run: TrainingRun) -> tuple[str, PurePosixPath, PurePosixPath]:
    """Classify a remote Attempt using lexical POSIX paths only."""
    output_root = remote_output_root_for_run(task, run)
    if not output_root or "\x00" in output_root:
        raise ValueError("远程 Attempt 缺少可验证的输出根目录，请检查远程训练路径配置")
    root = PurePosixPath(output_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("远程 Attempt 输出根目录无效，已拒绝清理")
    attempt_name = f"attempt-{run.attempt_number:03d}"
    expected_run_dir = root / task.task_type / task.id / attempt_name
    task_root = root / task.task_type / task.id
    stored_run_dir = _stored_run_dir(task, run)
    run_dir = PurePosixPath(stored_run_dir) if stored_run_dir else expected_run_dir
    if run_dir == expected_run_dir:
        return CANONICAL_LAYOUT, root, run_dir
    if task.source == "platform" and run.attempt_number == 1 and run_dir == task_root:
        return LEGACY_FLAT_LAYOUT, root, run_dir
    raise ValueError("远程 Attempt 目录与该任务/轮次不匹配，已拒绝清理")


def remote_attempt_path_candidates(
    task: TrainingTask,
    run: TrainingRun,
    artifact_paths: list[str] | tuple[str, ...] = (),
) -> list[PurePosixPath]:
    """Validate stored remote paths lexically without resolving them on this host."""
    layout, root, run_dir = remote_attempt_layout(task, run)
    attempt_name = f"attempt-{run.attempt_number:03d}"
    if layout == LEGACY_FLAT_LAYOUT:
        paths = []
        for value in artifact_paths:
            artifact = PurePosixPath(value)
            try:
                relative = artifact.relative_to(run_dir)
            except ValueError as exc:
                raise ValueError("旧版远程 Attempt 产物路径超出任务目录，已拒绝清理") from exc
            if not relative.parts or ATTEMPT_DIRECTORY_PATTERN.fullmatch(relative.parts[0]):
                raise ValueError("旧版远程 Attempt 产物与其他 Attempt 目录重叠，已拒绝清理")
            paths.append(run_dir / relative.parts[0])
    else:
        paths = [run_dir]
    if run.log_path:
        log_path = PurePosixPath(run.log_path)
        legacy_log = run_dir / "platform.log"
        date = log_path.parent.parent.name
        expected_log = root / ".platform-logs" / date / task.id / f"{attempt_name}.log"
        is_platform_log = (
            log_path == expected_log
            and re.fullmatch(r"\d{8}", date) is not None
        )
        if log_path != legacy_log and not is_platform_log:
            raise ValueError("远程 Attempt 日志路径与该任务/轮次不匹配，已拒绝清理")
        paths.append(log_path)
    return list(dict.fromkeys(paths))


def _deletion_path(path: Path) -> Path:
    """Resolve parent components while preserving the final symlink itself."""
    expanded = path.expanduser()
    return expanded.parent.resolve(strict=False) / expanded.name


def _is_imported_legacy_run_dir(task: TrainingTask, run: TrainingRun, run_dir: Path) -> bool:
    """Recognize only the exact Ultralytics run directory imported for this task."""
    legacy_root = _deletion_path(ULTRALYTICS_ROOT / "runs")
    try:
        relative = run_dir.relative_to(legacy_root)
    except ValueError:
        return False
    stored_run_dir = run.run_dir or (task.run_dir if task.current_run_id == run.id else None)
    if task.source != "imported" or not stored_run_dir or len(relative.parts) < 2 or relative.parts[0] != task.task_type:
        return False
    canonical = Path(stored_run_dir).expanduser().resolve(strict=False)
    expected_id = "legacy-" + hashlib.sha1(str(canonical).encode()).hexdigest()[:12]
    return task.id == expected_id


def _overlaps(first: Path, second: Path) -> bool:
    try:
        left, right = first.expanduser().resolve(strict=False), second.expanduser().resolve(strict=False)
    except OSError:
        left, right = Path(first), Path(second)
    return left == right or left in right.parents or right in left.parents


def _remote_overlaps(first: PurePosixPath, second: PurePosixPath) -> bool:
    left = PurePosixPath(posixpath.normpath(str(first)))
    right = PurePosixPath(posixpath.normpath(str(second)))
    return left == right or left in right.parents or right in left.parents


def validate_attempt_deletion(
    db: Session,
    task: TrainingTask,
    run: TrainingRun,
    *,
    ignore_operation_id: str | None = None,
) -> None:
    """Reject unsafe or concurrently referenced Attempts before queuing work."""
    if task.deleted_at:
        raise ValueError("任务已进入删除流程，不能单独删除 Attempt")
    if task.execution_node_id and run.execution_node_id not in {None, task.execution_node_id}:
        raise ValueError("Attempt 所属远程节点与任务不一致，已拒绝清理")
    if not task.execution_node_id and run.execution_node_id:
        raise ValueError("Attempt 标记为远程运行但任务未关联远程节点，已拒绝清理")
    if run.status in ACTIVE_RUN_STATUSES:
        raise ValueError("运行中或排队中的 Attempt 不能删除")
    existing_operation = attempt_delete_operation(db, run, active_only=True)
    if existing_operation and existing_operation.id != ignore_operation_id:
        raise ValueError("该 Attempt 已提交删除，请等待清理完成")
    task_operations = db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id,
        RemoteOperation.kind.in_(RUN_DELETE_KINDS),
        RemoteOperation.status.in_(ACTIVE_DELETE_STATUSES),
    ))
    if any(not is_attempt_payload(operation.payload, run.id) for operation in task_operations):
        raise ValueError("任务整体文件正在清理，不能单独删除 Attempt")

    active_tests = db.scalars(select(ModelTest).where(ModelTest.task_id == task.id, ModelTest.status.in_(ACTIVE_TEST_STATUSES))).all()
    target_artifacts = list(db.scalars(select(Artifact).where(Artifact.run_id == run.id)))
    target_artifact_ids = {artifact.id for artifact in target_artifacts}
    if any(
        test.run_id == run.id or test.source_run_id == run.id or test.artifact_id in target_artifact_ids
        for test in active_tests
    ):
        raise ValueError("该 Attempt 正被模型检测使用，请等待检测结束或取消后重试")

    remote = bool(task.execution_node_id)
    target_paths = (
        remote_attempt_path_candidates(task, run, [artifact.path for artifact in target_artifacts])
        if remote else attempt_path_candidates(task, run)
    )
    convert_path = PurePosixPath if remote else Path
    overlaps = _remote_overlaps if remote else _overlaps
    protected_paths = [convert_path(path) for path in (task.model_path, task.data_path) if path]
    if not remote:
        # Import lazily because whole-task deletion shares the constants above.
        from .task_deletions import validate_local_model_references

        validate_local_model_references(
            db,
            task.id,
            [str(path) for path in target_paths],
            include_task_active_tests=True,
        )
    related_runs = db.scalars(select(TrainingRun).where(
        TrainingRun.task_id == task.id, TrainingRun.id != run.id,
    )).all()
    for other in related_runs:
        other_artifacts = list(db.scalars(select(Artifact).where(Artifact.run_id == other.id)))
        try:
            other_layout = (
                remote_attempt_layout(task, other)[0]
                if remote else local_attempt_layout(task, other)[0]
            )
        except ValueError:
            other_layout = None
        # A verified legacy-flat run owns root-level entries, not the task
        # container itself. Treating that container as exclusive would make
        # every newer canonical Attempt look overlapped and undeletable.
        other_paths = [convert_path(other.log_path)] if other.log_path else []
        if other.run_dir and other_layout != LEGACY_FLAT_LAYOUT:
            other_paths.append(convert_path(other.run_dir))
        protected_paths.extend(other_paths)
        if any(overlaps(target, candidate) for target in target_paths for candidate in other_paths):
            raise ValueError("该 Attempt 的训练目录或日志与其他 Attempt 重叠，不能安全删除")
        if other.checkpoint_path and any(
            overlaps(convert_path(other.checkpoint_path), target) for target in target_paths
        ):
            raise ValueError("该 Attempt 的检查点仍被其他续训 Attempt 引用")
        for artifact in other_artifacts:
            artifact_path = convert_path(artifact.path)
            protected_paths.append(artifact_path)
            if any(overlaps(artifact_path, target) for target in target_paths):
                raise ValueError("该 Attempt 的产物仍被其他 Attempt 引用")
    other_tasks = db.scalars(select(TrainingTask).where(
        TrainingTask.id != task.id,
        TrainingTask.execution_node_id == task.execution_node_id if remote else TrainingTask.execution_node_id.is_(None),
    ))
    for other_task in other_tasks:
        protected_paths.extend(convert_path(path) for path in (other_task.model_path, other_task.data_path) if path)
        protected_paths.extend(
            convert_path(artifact.path)
            for artifact in db.scalars(select(Artifact).where(Artifact.task_id == other_task.id))
        )
        other_runs = db.scalars(select(TrainingRun).where(TrainingRun.task_id == other_task.id)).all()
        for other in other_runs:
            protected_paths.extend(convert_path(path) for path in (other.run_dir, other.log_path, other.checkpoint_path) if path)
            protected_paths.extend(
                convert_path(artifact.path)
                for artifact in db.scalars(select(Artifact).where(Artifact.run_id == other.id))
            )
    protected_paths.extend(
        convert_path(artifact.path)
        for artifact in db.scalars(select(Artifact).where(
            Artifact.task_id == task.id, Artifact.run_id != run.id,
        ))
    )
    if any(overlaps(target, protected) for target in target_paths for protected in protected_paths):
        raise ValueError("该 Attempt 路径仍被其他任务或资源引用，不能安全删除")


def build_local_attempt_delete_plan(
    db: Session,
    task: TrainingTask,
    run: TrainingRun,
    *,
    ignore_operation_id: str | None = None,
) -> dict:
    """Validate ownership/references and freeze one local Attempt's targets."""
    if task.execution_node_id or run.execution_node_id:
        raise ValueError("远程 Attempt 不能进入本机删除队列")
    validate_attempt_deletion(
        db,
        task,
        run,
        ignore_operation_id=ignore_operation_id,
    )
    layout, run_dir, output_targets, log_target, runtime_target = _local_attempt_components(task, run)
    targets = list(dict.fromkeys([
        *output_targets,
        *([log_target] if log_target is not None else []),
        runtime_target,
    ]))
    roots = [RUNS_ROOT, ULTRALYTICS_ROOT / "runs", RUNTIME_ROOT, LOG_ROOT]
    for target in targets:
        _validate_deletion_target(target, roots)
    return {
        "version": ATTEMPT_PLAN_VERSION,
        "task_id": task.id,
        "task_type": task.task_type,
        "source": task.source,
        "run_id": run.id,
        "attempt_number": run.attempt_number,
        "layout": layout,
        "run_dir": str(run_dir),
        "output_targets": sorted(str(path) for path in output_targets),
        "log_target": str(log_target) if log_target is not None else None,
        "runtime_target": str(runtime_target),
        "targets": sorted(str(path) for path in targets),
    }


def _validate_frozen_attempt_plan(
    plan: dict,
    task: TrainingTask | None = None,
    run: TrainingRun | None = None,
) -> list[Path]:
    """Validate an immutable plan while allowing already removed targets."""
    if plan.get("version") != ATTEMPT_PLAN_VERSION:
        raise ValueError("本机 Attempt 删除计划版本无效")
    task_id, task_type = plan.get("task_id"), plan.get("task_type")
    run_id, attempt_number = plan.get("run_id"), plan.get("attempt_number")
    if not isinstance(task_id, str) or not TASK_IDENTIFIER_PATTERN.fullmatch(task_id):
        raise ValueError("本机 Attempt 删除计划的任务标识无效")
    if not isinstance(task_type, str) or not TASK_TYPE_PATTERN.fullmatch(task_type):
        raise ValueError("本机 Attempt 删除计划的任务类型无效")
    if not isinstance(run_id, str) or not run_id or type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("本机 Attempt 删除计划的轮次标识无效")
    if task is not None and (
        task.id != task_id or task.task_type != task_type or task.source != plan.get("source")
    ):
        raise ValueError("本机 Attempt 删除计划与任务记录不匹配")
    if run is not None and (
        run.id != run_id or run.task_id != task_id or run.attempt_number != attempt_number
    ):
        raise ValueError("本机 Attempt 删除计划与轮次记录不匹配")

    run_dir_value = plan.get("run_dir")
    if not isinstance(run_dir_value, str) or not run_dir_value:
        raise ValueError("本机 Attempt 删除计划缺少训练目录")
    run_dir = _deletion_path(Path(run_dir_value))
    canonical_task_root = _deletion_path(RUNS_ROOT / task_type / task_id)
    canonical_run_dir = canonical_task_root / f"attempt-{attempt_number:03d}"
    layout = plan.get("layout")
    if layout == CANONICAL_LAYOUT:
        if run_dir != canonical_run_dir:
            raise ValueError("本机 Attempt 删除计划的标准目录已变更")
    elif layout == LEGACY_FLAT_LAYOUT:
        if plan.get("source") != "platform" or attempt_number != 1 or run_dir != canonical_task_root:
            raise ValueError("本机 Attempt 删除计划的旧版目录无法验证")
    elif layout == IMPORTED_LAYOUT:
        legacy_root = _deletion_path(ULTRALYTICS_ROOT / "runs")
        try:
            relative = run_dir.relative_to(legacy_root)
        except ValueError as exc:
            raise ValueError("本机 Attempt 删除计划的导入目录超出允许范围") from exc
        expected_id = "legacy-" + hashlib.sha1(str(run_dir.resolve(strict=False)).encode()).hexdigest()[:12]
        if plan.get("source") != "imported" or len(relative.parts) < 2 or relative.parts[0] != task_type or expected_id != task_id:
            raise ValueError("本机 Attempt 删除计划的导入目录归属无法验证")
    else:
        raise ValueError("本机 Attempt 删除计划包含未知目录布局")
    if task is not None and run is not None:
        current_layout, current_run_dir = local_attempt_layout(task, run)
        if current_layout != layout or current_run_dir != run_dir:
            raise ValueError("Attempt 训练目录在排队后发生变化，请重新提交删除")

    output_values = plan.get("output_targets")
    if not isinstance(output_values, list) or any(not isinstance(value, str) for value in output_values):
        raise ValueError("本机 Attempt 删除计划缺少产物清单")
    output_targets = [_deletion_path(Path(value)) for value in output_values]
    if len(set(output_targets)) != len(output_targets):
        raise ValueError("本机 Attempt 删除计划包含重复产物")
    if layout in {CANONICAL_LAYOUT, IMPORTED_LAYOUT}:
        if output_targets != [run_dir]:
            raise ValueError("本机 Attempt 删除计划的训练目录不完整")
    else:
        if any(
            target.parent != run_dir or ATTEMPT_DIRECTORY_PATTERN.fullmatch(target.name)
            for target in output_targets
        ):
            raise ValueError("本机 Attempt 删除计划的旧版产物超出允许范围")
        current_targets = {
            _deletion_path(child)
            for child in run_dir.iterdir()
            if not ATTEMPT_DIRECTORY_PATTERN.fullmatch(child.name)
        } if run_dir.is_dir() else set()
        unexpected = current_targets - set(output_targets)
        if unexpected:
            raise ValueError("Attempt 排队后出现新的旧版产物，已拒绝扩大删除范围")

    runtime_value = plan.get("runtime_target")
    if not isinstance(runtime_value, str):
        raise ValueError("本机 Attempt 删除计划缺少运行时目录")
    runtime_target = _deletion_path(Path(runtime_value))
    expected_runtime = _deletion_path(RUNTIME_ROOT / task_id / f"attempt-{attempt_number:03d}")
    if runtime_target != expected_runtime:
        raise ValueError("本机 Attempt 删除计划的运行时目录已变更")

    log_value = plan.get("log_target")
    log_target = _deletion_path(Path(log_value)) if isinstance(log_value, str) and log_value else None
    if log_value is not None and log_target is None:
        raise ValueError("本机 Attempt 删除计划包含无效日志路径")
    if log_target is not None:
        platform_match = re.fullmatch(
            rf"\d{{8}}_{re.escape(task_id)}_attempt-{attempt_number:03d}\.log",
            log_target.name,
        )
        allowed_logs = {
            _deletion_path(run_dir / "platform.log"),
            _deletion_path(runtime_target / "platform.log"),
        }
        if log_target not in allowed_logs and not (
            log_target.parent == _deletion_path(LOG_ROOT) and platform_match is not None
        ):
            raise ValueError("本机 Attempt 删除计划的日志路径超出允许范围")

    expected_targets = sorted(str(path) for path in dict.fromkeys([
        *output_targets,
        *([log_target] if log_target is not None else []),
        runtime_target,
    ]))
    if plan.get("targets") != expected_targets:
        raise ValueError("本机 Attempt 删除计划的目标清单不完整")
    roots = [RUNS_ROOT, ULTRALYTICS_ROOT / "runs", RUNTIME_ROOT, LOG_ROOT]
    targets = [_deletion_path(Path(value)) for value in expected_targets]
    for target in targets:
        _validate_deletion_target(target, roots)
    return targets


def verify_local_attempt_plan_absent(plan: dict) -> int:
    """Verify a prior cleanup receipt against the frozen target list."""
    targets = _validate_frozen_attempt_plan(plan)
    remaining = [str(target) for target in targets if target.exists() or target.is_symlink()]
    if remaining:
        raise RuntimeError("已记录的 Attempt 删除回执与文件系统不一致：" + "、".join(remaining))
    return len(targets)


def execute_local_attempt_plan(plan: dict) -> dict:
    """Apply a frozen Attempt plan outside a database write transaction."""
    targets = _validate_frozen_attempt_plan(plan)
    roots = [RUNS_ROOT, ULTRALYTICS_ROOT / "runs", RUNTIME_ROOT, LOG_ROOT]
    deleted: list[str] = []
    already_missing: list[str] = []
    for target in sorted(targets, key=lambda path: len(str(path)), reverse=True):
        if not target.exists() and not target.is_symlink():
            already_missing.append(str(target))
            continue
        removed = _safe_delete_directory(target, roots)
        if removed is not None:
            deleted.append(str(removed))
    remaining = [str(target) for target in targets if target.exists() or target.is_symlink()]
    if remaining:
        raise RuntimeError("Attempt 文件清理后仍有目标存在：" + "、".join(remaining))
    if plan.get("layout") == LEGACY_FLAT_LAYOUT:
        # Re-run validation so a file created during cleanup is reported and
        # left untouched instead of silently widening the frozen plan.
        _validate_frozen_attempt_plan(plan)
    return {
        "deleted_targets": sorted(deleted),
        "already_missing_targets": sorted(already_missing),
        "verified_target_count": len(targets),
    }


def delete_attempt_files(task: TrainingTask, run: TrainingRun) -> list[str]:
    """Delete local files for one run after validating every target first."""
    if task.execution_node_id:
        raise ValueError("远程 Attempt 文件必须由远程 Agent 清理")
    targets = attempt_path_candidates(task, run)
    roots = [RUNS_ROOT, ULTRALYTICS_ROOT / "runs", RUNTIME_ROOT, LOG_ROOT]
    for target in targets:
        _validate_deletion_target(target, roots)
    deleted: list[str] = []
    for target in sorted(targets, key=lambda path: len(str(path)), reverse=True):
        removed = _safe_delete_directory(target, roots)
        if removed is not None:
            deleted.append(str(removed))
    return deleted


def delete_attempt_records(db: Session, task_id: str, run_id: str) -> TrainingTask | None:
    """Drop run-owned rows while preserving ModelTest history and the task."""
    run = db.get(TrainingRun, run_id)
    task = db.get(TrainingTask, task_id)
    if not run or not task or run.task_id != task.id:
        return task
    artifacts = list(db.scalars(select(Artifact).where(Artifact.run_id == run.id)))
    artifact_ids = [artifact.id for artifact in artifacts]
    artifact_paths = {artifact.path for artifact in artifacts}
    # Detection history is independent and must survive; only its live FK links
    # are cleared when their source Attempt disappears.
    db.execute(update(ModelTest).where(ModelTest.task_id == task.id, ModelTest.run_id == run.id).values(run_id=None))
    db.execute(update(ModelTest).where(ModelTest.task_id == task.id, ModelTest.source_run_id == run.id).values(source_run_id=None))
    if artifact_ids:
        db.execute(update(ModelTest).where(
            ModelTest.task_id == task.id,
            ModelTest.artifact_id.in_(artifact_ids),
        ).values(artifact_id=None))
    db.execute(delete(MetricPoint).where(MetricPoint.run_id == run.id))
    db.execute(delete(TaskEvent).where(TaskEvent.run_id == run.id))
    db.execute(delete(GpuReservation).where(GpuReservation.run_id == run.id))
    if artifact_ids:
        db.execute(delete(Artifact).where(Artifact.id.in_(artifact_ids)))
    config = dict(task.config or {})
    if config.get("_resume_checkpoint") in artifact_paths:
        config.pop("_resume_checkpoint", None)
        task.config = config
    db.delete(run)
    db.flush()
    db.expire(task, ["runs"])
    remaining = current_run(db, task)
    if remaining:
        mirror_run_to_task(task, remaining)
    else:
        task.current_run_id = None
        task.status = "draft"
        task.run_dir = task.log_path = None
        task.completed_epochs = 0
        task.progress = 0.0
        task.pid = task.exit_code = None
        task.error_message = None
        task.latest_metrics = {}
        task.completion_info = {}
        task.queued_at = task.started_at = task.finished_at = None
    return task
