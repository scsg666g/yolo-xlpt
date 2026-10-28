from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    task_type: Literal["detect", "segment", "classify", "pose", "obb"] = "detect"
    group_id: str | None = Field(default=None, min_length=1, max_length=64)
    model_path: str
    data_path: str
    execution_node_id: str | None = None
    remote_output_root: str | None = None
    epochs: int = Field(default=100, ge=1, le=10000)
    batch: int = Field(default=16, ge=1, le=1024)
    imgsz: int = Field(default=640, ge=32, le=4096)
    device: str = "0"
    workers: int = Field(default=8, ge=0, le=64)
    optimizer: str = Field(default="auto", min_length=1, max_length=40)
    patience: int = Field(default=50, ge=0, le=10000)
    lr0: float | None = Field(default=None, gt=0, le=1, allow_inf_nan=False)
    pretrained: bool = True
    cache: bool | str = False
    amp: bool = True
    close_mosaic: int = Field(default=10, ge=0, le=1000)
    advanced: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "optimizer", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value in {"cpu", "mps"}:
            return value
        parts = value.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise ValueError("device 必须是 cpu、mps 或逗号分隔的 GPU 编号")
        normalized = [str(int(part.strip())) for part in parts]
        if len(normalized) != len(set(normalized)):
            raise ValueError("device 不能包含重复的 GPU 编号")
        return ",".join(normalized)


class LayoutUpdate(BaseModel):
    layout: dict[str, Any]


class TaskOrderUpdate(BaseModel):
    task_ids: list[str] = Field(min_length=1)


class TaskGroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=160, strict=True)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class TaskGroupRename(TaskGroupCreate):
    pass


class TaskGroupCreateFromTasks(TaskGroupCreate):
    model_config = ConfigDict(extra="forbid")

    task_ids: list[str] = Field(min_length=1, max_length=500)

    @field_validator("task_ids")
    @classmethod
    def reject_duplicate_task_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("任务列表不能包含重复项")
        return value


class TaskGroupAssignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: str | None = Field(default=None, min_length=1, max_length=64)


class TaskGroupRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    display_order: int
    created_at: datetime
    updated_at: datetime


class TaskRename(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=160, strict=True)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class TaskConfigUpdate(BaseModel):
    epochs: int = Field(ge=1, le=10000)
    batch: int = Field(ge=1, le=1024)
    imgsz: int = Field(ge=32, le=4096)
    device: str
    workers: int = Field(ge=0, le=64)
    optimizer: str = Field(min_length=1, max_length=40)
    config: dict[str, Any] = Field(default_factory=dict)
    custom_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("optimizer", mode="before")
    @classmethod
    def strip_optimizer(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        value = value.strip().lower()
        if value in {"cpu", "mps"}:
            return value
        parts = value.split(",")
        if not parts or any(not part.strip().isdigit() for part in parts):
            raise ValueError("device 必须是 cpu、mps 或逗号分隔的 GPU 编号")
        normalized = [str(int(part.strip())) for part in parts]
        if len(normalized) != len(set(normalized)):
            raise ValueError("device 不能包含重复的 GPU 编号")
        return ",".join(normalized)


class TaskResourceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str | None = Field(default=None, min_length=1)
    data_path: str | None = Field(default=None, min_length=1)

    @field_validator("model_path", "data_path", mode="before")
    @classmethod
    def strip_resource_path(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value


class RemoteInterruptConfirmation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    confirm_process_stopped: Literal[True]


class TaskRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    task_type: str
    status: str
    source: str
    group_id: str | None = None
    execution_node_id: str | None = None
    model_path: str
    data_path: str
    run_dir: str | None
    log_path: str | None
    epochs: int
    completed_epochs: int
    batch: int
    batch_history: list[int] = Field(default_factory=list)
    imgsz: int
    device: str
    workers: int
    optimizer: str
    progress: float
    pid: int | None
    exit_code: int | None
    queue_reason: str | None = None
    error_message: str | None
    completion_info: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any]
    latest_metrics: dict[str, Any]
    canvas_layout: dict[str, Any]
    created_at: datetime
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    updated_at: datetime
    display_order: int
    current_run_id: str | None
    deleted_at: datetime | None
    deleted_reason: str | None


class RunRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    task_id: str
    attempt_number: int
    launch_mode: str
    status: str
    execution_node_id: str | None = None
    monitoring_status: Literal["local", "online", "reconnecting", "unavailable"] | None = None
    monitoring_last_seen_at: datetime | None = None
    pid: int | None
    process_started_at: datetime | None
    command: list[Any]
    parameter_snapshot: dict[str, Any]
    batch_history: list[int] = Field(default_factory=list)
    model_path: str
    data_path: str
    checkpoint_source: dict[str, Any] = Field(default_factory=dict)
    checkpoint_path: str | None
    run_dir: str | None
    log_path: str | None
    completed_epochs: int
    progress: float
    latest_metrics: dict[str, Any]
    exit_code: int | None
    queue_reason: str | None = None
    error_message: str | None
    completion_info: dict[str, Any] = Field(default_factory=dict)
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime
    deletion_status: Literal["pending", "running", "failed", "succeeded"] | None = None
    deletion_error: str | None = None
    deletion_requested_at: datetime | None = None


class MetricRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    epoch: int
    elapsed: float
    values: dict[str, Any]
    created_at: datetime


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    kind: str
    name: str
    path: str
    size: int
    modified_at: datetime | None


class EffectiveBestWeightRead(BaseModel):
    artifact: ArtifactRead | None = None
    source_run_id: str | None = None
    source_attempt_number: int | None = None
    inherited: bool = False


class EventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    level: str
    event_type: str
    message: str
    payload: dict[str, Any]
    created_at: datetime
