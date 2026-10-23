"""Bounded, batch reads for the experiment picker and active comparisons."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .database import get_db
from .models import RemoteOperation, TrainingRun, TrainingTask
from .run_deletions import ACTIVE_DELETE_STATUSES
from .schemas import RunRead, TaskRead

router = APIRouter()


@router.get("/api/experiments")
def list_experiments(
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=200, ge=1, le=200),
    run_ids: str | None = Query(default=None, max_length=14000),
    db: Session = Depends(get_db),
):
    filters = [TrainingTask.deleted_at.is_(None)]
    if run_ids is not None:
        ids = list(dict.fromkeys(value for value in run_ids.split(",") if value))
        if not ids or len(ids) > 200 or any(len(value) > 64 for value in ids):
            raise HTTPException(422, "一次最多查询 200 个训练运行")
        filters.append(TrainingRun.id.in_(ids))
    query = select(TrainingRun, TrainingTask).join(TrainingTask, TrainingTask.id == TrainingRun.task_id).where(*filters)
    total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
    rows = db.execute(query.order_by(
        TrainingTask.display_order, TrainingTask.created_at.desc(), TrainingTask.id,
        TrainingRun.attempt_number.desc(), TrainingRun.id,
    ).offset(offset).limit(limit)).all()
    operations = {}
    if rows:
        ids = [run.id for run, _ in rows]
        for operation in db.scalars(select(RemoteOperation).where(
            RemoteOperation.kind.in_({"local_delete", "delete"}),
            RemoteOperation.payload["scope"].as_string() == "attempt",
            RemoteOperation.payload["run_id"].as_string().in_(ids),
        ).order_by(RemoteOperation.created_at.desc(), RemoteOperation.id.desc())):
            operations.setdefault(operation.payload["run_id"], operation)
    items = []
    for run, task in rows:
        operation = operations.get(run.id)
        status = None
        if operation:
            if operation.status in ACTIVE_DELETE_STATUSES:
                status = "pending" if operation.status == "pending" else "running"
            elif operation.status in {"failed", "cancelled"}:
                status = "failed"
            elif operation.status == "succeeded":
                status = "succeeded"
        view = RunRead.model_validate(run).model_copy(update={
            "deletion_status": status,
            "deletion_error": operation.error if operation else None,
            "deletion_requested_at": operation.created_at if operation else None,
        })
        items.append({"task": TaskRead.model_validate(task), "run": view})
    return {"items": items, "total": total, "offset": offset, "limit": limit}
