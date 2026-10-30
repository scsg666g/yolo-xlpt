from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import get_db
from .models import TaskWorkspace, TrainingTask, utcnow


router = APIRouter(prefix="/api/task-workspace", tags=["task workspace"])

WORKSPACE_ID = "default"
WORKSPACE_VERSION = 1
MAX_SAFE_REVISION = 2**53 - 1
MAX_COORDINATE = 10_000_000
MAX_NODES = 2_000
MAX_EDGES = 4_000
MAX_MODES = 12

# TrainingTask historically only constrained identifier length at the database layer.
# Keep that compatibility here instead of imposing a new character whitelist.
TaskId = Annotated[str, StringConstraints(min_length=1, max_length=64)]
EdgeId = Annotated[str, StringConstraints(min_length=1, max_length=160)]
ModeId = Annotated[str, StringConstraints(min_length=1, max_length=40, pattern=r"^[A-Za-z0-9_-]+$")]
FiniteCoordinate = Annotated[float, Field(ge=-MAX_COORDINATE, le=MAX_COORDINATE, allow_inf_nan=False)]


class WorkspacePosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: FiniteCoordinate
    y: FiniteCoordinate


class WorkspaceNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: TaskId
    position: WorkspacePosition


class WorkspaceEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: EdgeId
    source: TaskId
    target: TaskId
    label: str | None = Field(default=None, max_length=80)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip() or None

    @model_validator(mode="after")
    def reject_self_link(self) -> "WorkspaceEdge":
        if self.source == self.target:
            raise ValueError("关联线不能连接同一个任务")
        return self


class WorkspaceViewport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: FiniteCoordinate = 0
    y: FiniteCoordinate = 0
    zoom: float = Field(default=1, ge=0.1, le=4, allow_inf_nan=False)


class WorkspaceMode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[WorkspaceNode] = Field(default_factory=list, max_length=MAX_NODES)
    edges: list[WorkspaceEdge] = Field(default_factory=list, max_length=MAX_EDGES)
    viewport: WorkspaceViewport = Field(default_factory=WorkspaceViewport)

    @model_validator(mode="after")
    def validate_unique_ids_and_endpoints(self) -> "WorkspaceMode":
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("同一视图不能重复保存任务节点")
        edge_ids = [edge.id for edge in self.edges]
        if len(edge_ids) != len(set(edge_ids)):
            raise ValueError("同一视图不能重复保存关联线")
        known_nodes = set(node_ids)
        if any(edge.source not in known_nodes or edge.target not in known_nodes for edge in self.edges):
            raise ValueError("关联线端点必须属于当前视图中的任务节点")
        return self


class WorkspaceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=0, lt=MAX_SAFE_REVISION, strict=True)
    modes: dict[ModeId, WorkspaceMode] = Field(default_factory=dict, max_length=MAX_MODES)


class WorkspaceRead(BaseModel):
    version: int
    revision: int
    modes: dict[str, WorkspaceMode]
    updated_at: datetime | None


def _active_task_ids(db: Session) -> set[str]:
    return set(db.scalars(select(TrainingTask.id).where(TrainingTask.deleted_at.is_(None))).all())


def _filter_modes(modes: dict[str, WorkspaceMode], task_ids: set[str]) -> dict[str, dict[str, Any]]:
    filtered: dict[str, dict[str, Any]] = {}
    for mode_id, mode in modes.items():
        nodes = [node for node in mode.nodes if node.id in task_ids]
        node_ids = {node.id for node in nodes}
        edges = [
            edge for edge in mode.edges
            if edge.source in node_ids and edge.target in node_ids
        ]
        filtered[mode_id] = {
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "edges": [edge.model_dump(mode="json") for edge in edges],
            "viewport": mode.viewport.model_dump(mode="json"),
        }
    return filtered


def _read_record(record: TaskWorkspace | None, db: Session) -> WorkspaceRead:
    if record is None:
        return WorkspaceRead(version=WORKSPACE_VERSION, revision=0, modes={}, updated_at=None)

    try:
        modes = WorkspaceUpdate(revision=record.revision, modes=record.modes or {}).modes
    except (TypeError, ValueError):
        # A malformed legacy/manual row must not prevent the task centre from opening.
        modes = {}
    filtered = _filter_modes(modes, _active_task_ids(db))
    return WorkspaceRead(
        version=WORKSPACE_VERSION,
        revision=record.revision,
        modes=filtered,
        updated_at=record.updated_at,
    )


@router.get("", response_model=WorkspaceRead)
def get_task_workspace(db: Session = Depends(get_db)) -> WorkspaceRead:
    return _read_record(db.get(TaskWorkspace, WORKSPACE_ID), db)


@router.put("", response_model=WorkspaceRead)
def update_task_workspace(body: WorkspaceUpdate, db: Session = Depends(get_db)) -> WorkspaceRead:
    modes = _filter_modes(body.modes, _active_task_ids(db))
    now = utcnow()
    current = db.get(TaskWorkspace, WORKSPACE_ID)

    if current is None:
        if body.revision != 0:
            raise HTTPException(409, "任务总览布局已被其他窗口更新，请刷新后重试")
        record = TaskWorkspace(
            id=WORKSPACE_ID,
            version=WORKSPACE_VERSION,
            revision=1,
            modes=modes,
            updated_at=now,
        )
        db.add(record)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(409, "任务总览布局已被其他窗口更新，请刷新后重试") from exc
        db.refresh(record)
        return _read_record(record, db)

    result = db.execute(
        update(TaskWorkspace)
        .where(TaskWorkspace.id == WORKSPACE_ID, TaskWorkspace.revision == body.revision)
        .values(
            version=WORKSPACE_VERSION,
            revision=body.revision + 1,
            modes=modes,
            updated_at=now,
        )
    )
    if result.rowcount != 1:
        db.rollback()
        raise HTTPException(409, "任务总览布局已被其他窗口更新，请刷新后重试")
    db.commit()
    db.expire_all()
    return _read_record(db.get(TaskWorkspace, WORKSPACE_ID), db)
