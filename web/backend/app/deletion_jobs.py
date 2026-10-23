"""Read-only task cleanup previews and durable operation presentation."""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .compute_nodes import agent_version_at_least, as_utc
from .models import ComputeNode, ModelTest, RemoteOperation, TrainingTask, utcnow


TASK_DELETE_AGENT_VERSION = "2.0.4"
ACTIVE = {"pending", "running", "finalizing", "center_cleaning"}
PHASES = {
    "remote_cleanup": "waiting_agent",
    "remote_pending": "waiting_agent",
    "remote_verified": "verifying",
    "recording": "center_cleanup",
    "finalizing": "center_cleanup",
    "center_cleanup_failed": "center_cleanup",
    "complete": "completed",
}
MESSAGES = {
    "queued": "清理任务已排队",
    "stopping_tests": "正在停止该任务的模型测试与图片上传",
    "validating": "正在核对文件归属和模型引用",
    "waiting_agent": "等待远程 Agent 清理文件",
    "deleting_files": "正在清理训练产物和日志",
    "verifying": "正在核验文件清理结果",
    "center_cleanup": "正在清理中心缓存、测试文件及任务记录",
    "completed": "文件清理完成，任务记录已删除",
    "failed": "清理未完成，请查看原因后重试",
    "agent_upgrade_required": "请升级远程 Agent 后重试",
}


def reject_local_task_cleanup(db: Session) -> None:
    """Call under the shared local resource mutation guard."""
    operation = db.scalar(select(RemoteOperation.id).where(
        RemoteOperation.kind == "local_delete",
        RemoteOperation.status.in_(ACTIVE),
    ).limit(1))
    if operation:
        raise HTTPException(409, "本机正在清理训练文件，暂不能新增文件引用；请等待清理完成后重试")


def _whole_task_query():
    scope = RemoteOperation.payload["scope"].as_string()
    return select(RemoteOperation).where(
        RemoteOperation.kind.in_({"local_delete", "delete"}),
        scope.is_(None) | (scope != "attempt"),
    )


def _node_status(node: ComputeNode | None) -> str:
    if node is None:
        return "offline"
    if not node.enabled:
        return "disabled"
    if node.last_seen_at is None:
        return "pending"
    timeout = max(30, node.report_interval_seconds * 3)
    return "online" if (utcnow() - as_utc(node.last_seen_at)).total_seconds() <= timeout else "offline"


def serialize_deletion_job(operation: RemoteOperation, task: TrainingTask | None, node: ComputeNode | None) -> dict:
    payload = operation.payload or {}
    remote = operation.kind == "delete"
    node_status = _node_status(node) if remote else "online"
    raw_phase = payload.get("phase") or ("remote_cleanup" if remote else "queued")
    phase = PHASES.get(raw_phase, raw_phase)
    status = operation.status
    if remote and status == "pending" and phase == "stopping_tests":
        # Remote cancellation is settled when the Agent claims the operation.
        # Before that, accurately describe an offline job as waiting for Agent.
        phase = "waiting_agent"
    if status == "succeeded":
        phase = "completed"
    elif status in {"failed", "cancelled"}:
        phase = "failed"
    elif phase == "waiting_agent" and status == "running":
        phase = "deleting_files"
    if status == "center_cleaning" or status in ACTIVE and phase == "center_cleanup":
        status = "finalizing"
    message = MESSAGES.get(phase, "正在处理文件清理")
    if status in {"failed", "cancelled"}:
        message = operation.error or message
    elif remote and phase in {"waiting_agent", "deleting_files"}:
        if not agent_version_at_least(node.agent_version if node else None, (2, 0, 4)):
            phase = "agent_upgrade_required"
            message = f"完整任务清理需要 Agent {TASK_DELETE_AGENT_VERSION} 或以上版本"
        elif node_status != "online":
            message = "远程 Agent 已禁用，请启用后继续" if node_status == "disabled" else "等待远程 Agent 重新连接；文件清理尚未确认完成"
    return {
        "id": operation.id, "task_id": operation.task_id,
        "task_name": payload.get("task_name") or (task.name if task else operation.task_id),
        "executor": "remote" if remote else "local",
        "node_id": operation.node_id if remote else None,
        "node_name": payload.get("node_name") or (node.name if node else "远程服务器" if remote else "本机服务器"),
        "node_status": node_status,
        "last_seen_at": as_utc(node.last_seen_at) if node and remote and node.last_seen_at else None,
        "agent_version": node.agent_version if node and remote else None,
        "required_agent_version": TASK_DELETE_AGENT_VERSION if remote else None,
        "status": status, "phase": phase, "message": message,
        "error": operation.error,
        "retryable": task is not None and operation.status in {"failed", "cancelled"},
        "created_at": as_utc(operation.created_at), "updated_at": as_utc(operation.updated_at),
        "receipt": ({**payload["receipt"], "center_cleanup": payload["center_cleanup"]}
                    if isinstance(payload.get("receipt"), dict) and payload.get("center_cleanup")
                    else payload.get("receipt")),
    }


def list_deletion_jobs(db: Session, *, limit: int = 50) -> list[dict]:
    # The history limit must never hide progress/errors for a task still in
    # trash. Keep its latest operation, then append bounded historical receipts.
    current: dict[str, RemoteOperation] = {}
    for operation in db.scalars(_whole_task_query().join(
        TrainingTask, TrainingTask.id == RemoteOperation.task_id,
    ).where(TrainingTask.deleted_at.is_not(None)).order_by(
        RemoteOperation.created_at.desc(), RemoteOperation.id.desc(),
    )):
        current.setdefault(operation.task_id, operation)
    history = db.scalars(_whole_task_query().where(
        RemoteOperation.task_id.not_in(current),
    ).order_by(
        RemoteOperation.status.in_(ACTIVE).desc(),
        RemoteOperation.updated_at.desc(), RemoteOperation.id.desc(),
    ).limit(limit)).all()
    operations = sorted([*current.values(), *history], key=lambda operation: (
        operation.status in ACTIVE, as_utc(operation.updated_at), operation.id,
    ), reverse=True)
    task_ids = {operation.task_id for operation in operations}
    node_ids = {operation.node_id for operation in operations if operation.kind == "delete"}
    tasks = {task.id: task for task in db.scalars(select(TrainingTask).where(TrainingTask.id.in_(task_ids)))}
    nodes = {node.id: node for node in db.scalars(select(ComputeNode).where(ComputeNode.id.in_(node_ids)))}
    return [serialize_deletion_job(operation, tasks.get(operation.task_id), nodes.get(operation.node_id)) for operation in operations]


def deletion_preview(db: Session, task: TrainingTask) -> dict:
    """Validate without queueing work, cancelling tests or touching files."""
    from .remote_operations import build_remote_task_delete_plan
    from .task_deletions import build_local_task_plan

    operation = db.scalar(_whole_task_query().where(
        RemoteOperation.task_id == task.id,
    ).order_by(RemoteOperation.created_at.desc()).limit(1))
    remote = bool(task.execution_node_id)
    node = db.get(ComputeNode, task.execution_node_id) if remote else None
    reason = None
    try:
        if operation and operation.status in ACTIVE:
            raise ValueError("该任务已提交文件清理，请在回收站查看进度")
        if not (operation and (operation.payload or {}).get("remote_verified")):
            if remote:
                build_remote_task_delete_plan(db, task, acquire_lock=False)
            else:
                build_local_task_plan(db, task)
    except (ValueError, HTTPException) as exc:
        reason = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
    test_ids = db.scalars(select(ModelTest.id).where(ModelTest.task_id == task.id)).all()
    return {
        "task_id": task.id, "task_name": task.name,
        "executor": "remote" if remote else "local",
        "node_name": node.name if node else "远程服务器" if remote else "本机服务器",
        "attempt_count": len(task.runs), "test_count": len(test_ids),
        "can_delete": reason is None, "blocking_reason": reason,
        "delete_items": [
            f"{'远程服务器' if remote else '本机'}上属于该任务的全部 {len(task.runs)} 次训练产物及日志",
            "中心保存的任务记录、指标和事件",
            f"该任务的 {len(test_ids)} 条模型测试历史、原图、结果及临时权重副本",
            "该任务的远程产物缓存" if remote else "该任务的本机运行时配置",
        ],
        "keep_items": [
            "基础模型、数据集及独立登记的模型来源（存在引用冲突时阻止删除）",
            "其他任务及独立模型测试的文件和历史",
            "清理进度、失败原因及完成回执",
        ],
    }
