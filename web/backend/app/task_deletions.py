"""Safety checks and idempotent file cleanup for whole local tasks."""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from . import model_assets as model_asset_service
from . import model_test_storage, services
from .models import Artifact, ModelAsset, ModelTest, RemoteOperation, TrainingRun, TrainingTask
from .run_deletions import ACTIVE_RUN_STATUSES, ACTIVE_TEST_STATUSES


PLAN_VERSION = 1
ACTIVE_DELETE_STATUSES = {"pending", "running", "finalizing", "center_cleaning"}
TASK_IDENTIFIER = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TASK_TYPE = re.compile(r"^[A-Za-z0-9_-]{1,24}$")
PLATFORM_LOG = re.compile(r"^\d{8}_(?P<task>[A-Za-z0-9_-]{1,64})_attempt-(?P<attempt>\d{3,})\.log$")


def _deletion_path(path: Path | str) -> Path:
    return services._deletion_path(Path(path))


def _absolute_reference_path(path: Path | str, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (base or model_asset_service.PROJECT_ROOT) / candidate
    return Path(os.path.abspath(candidate))


def _target_overlaps_reference(
    target: Path | str,
    reference: Path | str,
    *,
    reference_base: Path | None = None,
) -> bool:
    """Compare both stored paths and resolved aliases without following a deletion symlink."""
    target_lexical = _absolute_reference_path(target)
    reference_lexical = _absolute_reference_path(reference, base=reference_base)
    if (
        target_lexical == reference_lexical
        or target_lexical in reference_lexical.parents
        or reference_lexical in target_lexical.parents
    ):
        return True
    # A normal target directory owns its resolved contents. Resolve the
    # reference as well so an alias outside the run tree cannot hide a live
    # dependency on a weight inside it. A final target symlink is only unlinked,
    # so its external destination is deliberately not treated as owned.
    if target_lexical.is_symlink():
        return False
    target_resolved = target_lexical.resolve(strict=False)
    reference_resolved = reference_lexical.resolve(strict=False)
    return (
        target_resolved == reference_resolved
        or target_resolved in reference_resolved.parents
        or reference_resolved in target_resolved.parents
    )


def _stored_run_dir(task: TrainingTask, run: TrainingRun) -> str | None:
    return run.run_dir or (task.run_dir if task.current_run_id == run.id else None)


def _stored_log_path(task: TrainingTask, run: TrainingRun) -> str | None:
    return run.log_path or (task.log_path if task.current_run_id == run.id else None)


def _imported_run_dir(task: TrainingTask, run_dir: Path) -> bool:
    legacy_root = _deletion_path(services.ULTRALYTICS_ROOT / "runs")
    try:
        relative = run_dir.relative_to(legacy_root)
    except ValueError:
        return False
    if task.source != "imported" or len(relative.parts) < 2 or relative.parts[0] != task.task_type:
        return False
    canonical = Path(run_dir).expanduser().resolve(strict=False)
    expected_id = "legacy-" + hashlib.sha1(str(canonical).encode()).hexdigest()[:12]
    return task.id == expected_id


def _classify_run_dir(task: TrainingTask, run: TrainingRun) -> tuple[str, Path]:
    canonical_task_root = _deletion_path(services.RUNS_ROOT / task.task_type / task.id)
    expected = canonical_task_root / f"attempt-{run.attempt_number:03d}"
    stored = _stored_run_dir(task, run)
    candidate = _deletion_path(stored) if stored else expected
    if candidate == expected:
        return "canonical", candidate
    if task.source == "platform" and run.attempt_number == 1 and candidate == canonical_task_root:
        return "legacy_flat", candidate
    if _imported_run_dir(task, candidate):
        return "imported", candidate
    raise ValueError(
        f"Attempt {run.attempt_number:03d} 训练目录与任务记录不匹配，已拒绝永久删除"
    )


def _validate_log_path(task: TrainingTask, run: TrainingRun, run_dir: Path) -> Path | None:
    stored = _stored_log_path(task, run)
    if not stored:
        return None
    candidate = _deletion_path(stored)
    runtime_log = _deletion_path(
        services.RUNTIME_ROOT / task.id / f"attempt-{run.attempt_number:03d}" / "platform.log"
    )
    run_log = _deletion_path(run_dir / "platform.log")
    match = PLATFORM_LOG.fullmatch(candidate.name)
    platform_log = (
        candidate.parent == _deletion_path(services.LOG_ROOT)
        and match is not None
        and match.group("task") == task.id
        and int(match.group("attempt")) == run.attempt_number
    )
    if candidate not in {run_log, runtime_log} and not platform_log:
        raise ValueError(
            f"Attempt {run.attempt_number:03d} 日志路径与任务记录不匹配，已拒绝永久删除"
        )
    return candidate


def _target_contains(path: str, run_roots: list[dict], log_targets: list[str], runtime_root: str) -> bool:
    candidate = _deletion_path(path)
    return (
        any(
            candidate == _deletion_path(item["path"])
            or _deletion_path(item["path"]) in candidate.parents
            for item in run_roots
        )
        or candidate in {_deletion_path(value) for value in log_targets}
        or candidate == _deletion_path(runtime_root)
        or _deletion_path(runtime_root) in candidate.parents
    )


def _protected_model_assets(
    db: Session,
    deleting_task_id: str,
) -> list[tuple[ModelAsset, bool]]:
    """Return active registrations and removed registrations retained by external history."""
    protected: list[tuple[ModelAsset, bool]] = []
    assets = db.scalars(select(ModelAsset).where(ModelAsset.execution_node_id == "local")).all()
    for asset in assets:
        retained_by_history = asset.deleted_at is not None and db.scalar(select(ModelTest.id).where(
            ModelTest.model_asset_id == asset.id,
            (ModelTest.task_id.is_(None)) | (ModelTest.task_id != deleting_task_id),
        ).limit(1)) is not None
        if asset.deleted_at is None or retained_by_history:
            protected.append((asset, retained_by_history))
    return protected


def _validate_references(
    db: Session,
    task: TrainingTask,
    targets: list[str],
) -> None:
    protected: list[str] = [path for path in (task.model_path, task.data_path) if path]
    for run in task.runs:
        protected.extend(path for path in (run.model_path, run.data_path) if path)

    other_tasks = db.scalars(select(TrainingTask).where(
        TrainingTask.id != task.id,
        TrainingTask.execution_node_id.is_(None),
    )).all()
    for other in other_tasks:
        protected.extend(path for path in (other.model_path, other.data_path, other.run_dir, other.log_path) if path)
        for run in other.runs:
            protected.extend(path for path in (
                run.model_path, run.data_path, run.checkpoint_path, run.run_dir, run.log_path,
            ) if path)
        protected.extend(
            artifact.path for artifact in db.scalars(select(Artifact).where(Artifact.task_id == other.id))
            if artifact.path
        )

    external_active_tests = db.scalars(select(ModelTest).where(
        ModelTest.status.in_(ACTIVE_TEST_STATUSES),
        or_(ModelTest.execution_node_id.is_(None), ModelTest.execution_node_id == "local"),
        or_(ModelTest.task_id.is_(None), ModelTest.task_id != task.id),
    )).all()
    protected.extend(test.model_source_path for test in external_active_tests if test.model_source_path)

    for target in targets:
        for asset, retained_by_history in _protected_model_assets(db, task.id):
            if asset.path and _target_overlaps_reference(
                target,
                asset.path,
                reference_base=model_asset_service.PROJECT_ROOT,
            ):
                if retained_by_history:
                    raise ValueError(
                        f"已移除的模型登记「{asset.name}」仍有模型测试历史引用；"
                        "请先删除相关历史记录后再永久删除任务"
                    )
                raise ValueError(f"训练产物仍被模型来源「{asset.name}」登记，请先移除该模型来源")
        if any(_target_overlaps_reference(target, candidate) for candidate in protected):
            raise ValueError("待清理本机路径仍被基础资源、模型测试或其他任务引用，已拒绝永久删除")


def validate_task_deletion(
    db: Session,
    task: TrainingTask,
    *,
    ignore_operation_id: str | None = None,
) -> None:
    """Reject a whole-task deletion whose ownership or lifecycle is ambiguous."""
    if task.execution_node_id:
        raise ValueError("远程任务不能进入本机删除队列")
    if not TASK_IDENTIFIER.fullmatch(task.id) or not TASK_TYPE.fullmatch(task.task_type):
        raise ValueError("任务标识或任务类型不合法，已拒绝永久删除")
    if task.status in ACTIVE_RUN_STATUSES:
        raise ValueError("运行中或排队中的任务不能永久删除")
    if task.current_run_id and all(run.id != task.current_run_id for run in task.runs):
        raise ValueError("当前 Attempt 记录不存在，已拒绝永久删除")
    for run in task.runs:
        if run.status in ACTIVE_RUN_STATUSES:
            raise ValueError(f"Attempt {run.attempt_number:03d} 仍在运行或排队，不能永久删除")
        if run.execution_node_id:
            raise ValueError(
                f"Attempt {run.attempt_number:03d} 属于远程服务器，不能由本机清理"
            )
        if run.attempt_number < 1:
            raise ValueError("Attempt 编号无效，已拒绝永久删除")

    operations = db.scalars(select(RemoteOperation).where(
        RemoteOperation.task_id == task.id,
        RemoteOperation.kind.in_({"local_delete", "delete"}),
        RemoteOperation.status.in_(ACTIVE_DELETE_STATUSES),
    )).all()
    if any(operation.id != ignore_operation_id for operation in operations):
        raise ValueError("该任务已有文件清理正在进行，请等待完成后重试")


def build_local_task_plan(
    db: Session,
    task: TrainingTask,
    *,
    ignore_operation_id: str | None = None,
) -> dict:
    """Validate current ownership and freeze every local filesystem target."""
    validate_task_deletion(db, task, ignore_operation_id=ignore_operation_id)
    canonical_root = _deletion_path(services.RUNS_ROOT / task.task_type / task.id)
    run_root_map: dict[str, dict] = {
        str(canonical_root): {"path": str(canonical_root), "layout": "canonical_task"},
    }
    log_targets: set[str] = set()
    runs = sorted(task.runs, key=lambda item: (item.attempt_number, item.id))
    for run in runs:
        layout, run_dir = _classify_run_dir(task, run)
        if layout == "imported":
            run_root_map[str(run_dir)] = {"path": str(run_dir), "layout": "imported"}
        log_path = _validate_log_path(task, run, run_dir)
        if log_path:
            log_targets.add(str(log_path))

    run_roots = [run_root_map[key] for key in sorted(run_root_map)]
    runtime_root = str(_deletion_path(services.RUNTIME_ROOT / task.id))
    logs = sorted(log_targets)

    if task.run_dir and not _target_contains(task.run_dir, run_roots, logs, runtime_root):
        raise ValueError("任务训练目录不属于任何可验证的 Attempt，已拒绝永久删除")
    if task.log_path and not _target_contains(task.log_path, run_roots, logs, runtime_root):
        raise ValueError("任务日志路径不属于任何可验证的 Attempt，已拒绝永久删除")

    artifacts = list(db.scalars(select(Artifact).where(Artifact.task_id == task.id)))
    run_ids = {run.id for run in runs}
    for artifact in artifacts:
        if artifact.run_id and artifact.run_id not in run_ids:
            raise ValueError("任务产物关联了不存在的 Attempt，已拒绝永久删除")
        if artifact.path and not _target_contains(artifact.path, run_roots, logs, runtime_root):
            raise ValueError(f"任务产物「{artifact.name}」位于不可验证的目录，已拒绝永久删除")

    targets = sorted({
        *(item["path"] for item in run_roots),
        runtime_root,
        *logs,
    })
    _validate_references(db, task, targets)
    test_ids = sorted(db.scalars(select(ModelTest.id).where(ModelTest.task_id == task.id)).all())
    return {
        "version": PLAN_VERSION,
        "task_id": task.id,
        "task_type": task.task_type,
        "source": task.source,
        "run_ids": [run.id for run in runs],
        "attempt_count": len(runs),
        "run_roots": run_roots,
        "runtime_root": runtime_root,
        "log_targets": logs,
        "targets": targets,
        "test_ids": test_ids,
    }


def _validate_frozen_plan(plan: dict) -> list[Path]:
    if plan.get("version") != PLAN_VERSION:
        raise ValueError("本机删除计划版本无效")
    task_id, task_type = plan.get("task_id"), plan.get("task_type")
    if not isinstance(task_id, str) or not TASK_IDENTIFIER.fullmatch(task_id):
        raise ValueError("本机删除计划的任务标识无效")
    if not isinstance(task_type, str) or not TASK_TYPE.fullmatch(task_type):
        raise ValueError("本机删除计划的任务类型无效")
    canonical_root = _deletion_path(services.RUNS_ROOT / task_type / task_id)
    legacy_root = _deletion_path(services.ULTRALYTICS_ROOT / "runs")
    run_roots = plan.get("run_roots")
    if not isinstance(run_roots, list) or not run_roots:
        raise ValueError("本机删除计划缺少训练目录")
    run_paths: list[Path] = []
    for item in run_roots:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("本机删除计划包含无效训练目录")
        candidate = _deletion_path(item["path"])
        if item.get("layout") == "canonical_task":
            if candidate != canonical_root:
                raise ValueError("本机删除计划的标准任务目录已变更")
        elif item.get("layout") == "imported":
            try:
                relative = candidate.relative_to(legacy_root)
            except ValueError as exc:
                raise ValueError("本机删除计划的导入目录超出允许范围") from exc
            expected = "legacy-" + hashlib.sha1(str(candidate.resolve(strict=False)).encode()).hexdigest()[:12]
            if plan.get("source") != "imported" or len(relative.parts) < 2 or relative.parts[0] != task_type or expected != task_id:
                raise ValueError("本机删除计划的导入目录归属无法验证")
        else:
            raise ValueError("本机删除计划包含未知目录布局")
        run_paths.append(candidate)

    runtime_root = _deletion_path(plan.get("runtime_root", ""))
    if runtime_root != _deletion_path(services.RUNTIME_ROOT / task_id):
        raise ValueError("本机删除计划的运行时目录已变更")
    log_values = plan.get("log_targets")
    if not isinstance(log_values, list) or any(not isinstance(value, str) for value in log_values):
        raise ValueError("本机删除计划包含无效日志路径")
    log_paths = [_deletion_path(value) for value in log_values]
    for log in log_paths:
        within_owned_root = any(root == log or root in log.parents for root in [*run_paths, runtime_root])
        if not within_owned_root and log.parent != _deletion_path(services.LOG_ROOT):
            raise ValueError("本机删除计划的日志路径超出允许范围")

    expected_targets = sorted({str(path) for path in [*run_paths, runtime_root, *log_paths]})
    if plan.get("targets") != expected_targets:
        raise ValueError("本机删除计划的目标清单不完整")
    roots = [services.RUNS_ROOT, services.ULTRALYTICS_ROOT / "runs", services.RUNTIME_ROOT, services.LOG_ROOT]
    targets = [_deletion_path(value) for value in expected_targets]
    for target in targets:
        services._validate_deletion_target(target, roots)
    return targets


def execute_local_task_plan(plan: dict) -> dict:
    """Apply a frozen plan and prove that each target is absent afterward."""
    targets = _validate_frozen_plan(plan)
    roots = [services.RUNS_ROOT, services.ULTRALYTICS_ROOT / "runs", services.RUNTIME_ROOT, services.LOG_ROOT]
    deleted: list[str] = []
    already_missing: list[str] = []
    for target in sorted(targets, key=lambda path: len(str(path)), reverse=True):
        if not target.exists() and not target.is_symlink():
            already_missing.append(str(target))
            continue
        removed = services._safe_delete_directory(target, roots)
        if removed is not None:
            deleted.append(str(removed))
    remaining = [str(target) for target in targets if target.exists() or target.is_symlink()]
    if remaining:
        raise RuntimeError("文件清理后仍有目标存在：" + "、".join(remaining))

    canonical_parent = _deletion_path(services.RUNS_ROOT / plan["task_type"])
    try:
        canonical_parent.rmdir()
    except OSError:
        pass
    return {
        "deleted_targets": sorted(deleted),
        "already_missing_targets": sorted(already_missing),
        "verified_target_count": len(targets),
    }


def cleanup_task_test_files(
    task_id: str,
    test_ids: list[str],
    *,
    delete_history: Callable[[str], None] = model_test_storage.delete_task_history,
) -> list[str]:
    """Remove task-owned test work/history and verify the cleanup."""
    if not TASK_IDENTIFIER.fullmatch(task_id):
        raise ValueError("训练任务 ID 不合法")
    cleaned: list[str] = []
    for test_id in sorted(set(test_ids)):
        work = model_test_storage.test_work_root(test_id)
        model_test_storage.delete_test_work(test_id)
        if work.exists() or work.is_symlink():
            raise RuntimeError(f"模型测试临时目录清理失败：{test_id}")
        cleaned.append(test_id)
    delete_history(task_id)
    history = model_test_storage.task_history_root(task_id)
    if history.exists() or history.is_symlink():
        raise RuntimeError("模型测试历史目录清理后仍然存在")
    return cleaned
