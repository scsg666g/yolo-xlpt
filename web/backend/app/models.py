from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ComputeNode(Base):
    __tablename__ = "compute_nodes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160))
    address: Mapped[str] = mapped_column(String(255), default="")
    notes: Mapped[str] = mapped_column(String(1000), default="")
    kind: Mapped[str] = mapped_column(String(16), default="agent")
    enabled: Mapped[bool] = mapped_column(default=True)
    token_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    report_interval_seconds: Mapped[int] = mapped_column(Integer, default=5)
    snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    agent_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TaskGroup(Base):
    __tablename__ = "task_groups"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    display_order: Mapped[int] = mapped_column(Integer, default=0, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    tasks: Mapped[list["TrainingTask"]] = relationship(back_populates="group")


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    task_type: Mapped[str] = mapped_column(String(24), default="detect")
    status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    source: Mapped[str] = mapped_column(String(24), default="platform")
    group_id: Mapped[str | None] = mapped_column(
        ForeignKey("task_groups.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    execution_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_path: Mapped[str] = mapped_column(Text)
    data_path: Mapped[str] = mapped_column(Text)
    run_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    epochs: Mapped[int] = mapped_column(Integer, default=100)
    completed_epochs: Mapped[int] = mapped_column(Integer, default=0)
    batch: Mapped[int] = mapped_column(Integer, default=16)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    device: Mapped[str] = mapped_column(String(32), default="0")
    workers: Mapped[int] = mapped_column(Integer, default=8)
    optimizer: Mapped[str] = mapped_column(String(40), default="auto")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    latest_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    canvas_layout: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    # Stable presentation order shared by every browser and client.
    display_order: Mapped[int] = mapped_column(Integer, default=0, index=True)
    current_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    deleted_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)

    runs: Mapped[list["TrainingRun"]] = relationship(back_populates="task", cascade="all, delete-orphan", foreign_keys="TrainingRun.task_id")
    metrics: Mapped[list["MetricPoint"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    events: Mapped[list["TaskEvent"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    model_tests: Mapped[list["ModelTest"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    group: Mapped[TaskGroup | None] = relationship(back_populates="tasks")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (UniqueConstraint("task_id", "attempt_number", name="uq_run_task_attempt"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    launch_mode: Mapped[str] = mapped_column(String(24), default="fresh")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
    execution_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    stop_requested: Mapped[bool] = mapped_column(default=False)
    claim_lease: Mapped[str | None] = mapped_column(String(32), nullable=True)
    claim_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    claim_confirmed: Mapped[bool] = mapped_column(default=False)
    remote_log_tail: Mapped[list] = mapped_column(JSON, default=list)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    process_started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    command: Mapped[list] = mapped_column(JSON, default=list)
    parameter_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    model_path: Mapped[str] = mapped_column(Text)
    data_path: Mapped[str] = mapped_column(Text)
    checkpoint_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_dir: Mapped[str | None] = mapped_column(Text, nullable=True)
    log_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    completed_epochs: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    latest_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    exit_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    task: Mapped[TrainingTask] = relationship(back_populates="runs", foreign_keys=[task_id])
    metrics: Mapped[list["MetricPoint"]] = relationship(back_populates="run")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")
    events: Mapped[list["TaskEvent"]] = relationship(back_populates="run")
    gpu_reservations: Mapped[list["GpuReservation"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
    )


class GpuReservation(Base):
    """Durable ownership of a physical GPU or a node-wide serial slot."""
    __tablename__ = "gpu_reservations"

    resource_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("training_runs.id", ondelete="CASCADE"), index=True,
    )
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    gpu_uuid: Mapped[str | None] = mapped_column(String(160), nullable=True)
    gpu_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    run: Mapped[TrainingRun] = relationship(back_populates="gpu_reservations")


class MetricPoint(Base):
    __tablename__ = "metric_points"
    __table_args__ = (UniqueConstraint("run_id", "epoch", name="uq_metric_run_epoch"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True, index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    elapsed: Mapped[float] = mapped_column(Float, default=0.0)
    values: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    task: Mapped[TrainingTask] = relationship(back_populates="metrics")
    run: Mapped[TrainingRun | None] = relationship(back_populates="metrics")


class Artifact(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("task_id", "path", name="uq_artifact_task_path"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text)
    size: Mapped[int] = mapped_column(Integer, default=0)
    modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    task: Mapped[TrainingTask] = relationship(back_populates="artifacts")
    run: Mapped[TrainingRun | None] = relationship(back_populates="artifacts")


class TaskEvent(Base):
    __tablename__ = "task_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    run_id: Mapped[str | None] = mapped_column(ForeignKey("training_runs.id"), nullable=True, index=True)
    level: Mapped[str] = mapped_column(String(16), default="info")
    event_type: Mapped[str] = mapped_column(String(40))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    task: Mapped[TrainingTask] = relationship(back_populates="events")
    run: Mapped[TrainingRun | None] = relationship(back_populates="events")


class ModelAsset(Base):
    """A reusable model source registered for the standalone testing centre."""
    __tablename__ = "model_assets"
    __table_args__ = (
        UniqueConstraint(
            "source_type", "execution_node_id", "path", name="uq_model_asset_source_path",
        ),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    execution_node_id: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(Text)
    artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="ready", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    artifact: Mapped[Artifact | None] = relationship(foreign_keys=[artifact_id])
    model_tests: Mapped[list["ModelTest"]] = relationship(back_populates="model_asset")


class ModelTest(Base):
    __tablename__ = "model_tests"
    __table_args__ = (
        Index("ix_model_tests_task_created_id", "task_id", "created_at", "id"),
        Index("ix_model_tests_created_id", "created_at", "id"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    task_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_tasks.id", ondelete="CASCADE"), nullable=True, index=True,
    )
    # run_id is the Attempt open in the canvas. source_run_id produced the tested weight.
    run_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    source_run_id: Mapped[str | None] = mapped_column(
        ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    artifact_id: Mapped[int | None] = mapped_column(
        ForeignKey("artifacts.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    model_asset_id: Mapped[str | None] = mapped_column(
        ForeignKey("model_assets.id", ondelete="SET NULL"), nullable=True, index=True,
    )
    model_source_type: Mapped[str] = mapped_column(
        String(32), default="training_artifact", index=True,
    )
    execution_node_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_name: Mapped[str] = mapped_column(String(255))
    model_name: Mapped[str] = mapped_column(String(255))
    # Content identity is immutable even when the linked Artifact row is
    # refreshed in place by a later local scan or remote manifest.
    model_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rerun_of_test_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    attempt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_attempt_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="uploading", index=True)
    confidence: Mapped[float] = mapped_column(Float)
    iou: Mapped[float] = mapped_column(Float)
    imgsz: Mapped[int] = mapped_column(Integer, default=640)
    device: Mapped[str] = mapped_column(String(32), default="CPU")
    model_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    storage_bytes: Mapped[int] = mapped_column(Integer, default=0)
    delete_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    task: Mapped[TrainingTask | None] = relationship(back_populates="model_tests")
    run: Mapped[TrainingRun | None] = relationship(foreign_keys=[run_id])
    source_run: Mapped[TrainingRun | None] = relationship(foreign_keys=[source_run_id])
    artifact: Mapped[Artifact | None] = relationship(foreign_keys=[artifact_id])
    model_asset: Mapped[ModelAsset | None] = relationship(back_populates="model_tests")
    images: Mapped[list["ModelTestImage"]] = relationship(
        back_populates="test", cascade="all, delete-orphan", order_by="ModelTestImage.ordinal",
    )


class ModelTestImage(Base):
    __tablename__ = "model_test_images"
    __table_args__ = (UniqueConstraint("test_id", "ordinal", name="uq_model_test_image_ordinal"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    test_id: Mapped[str] = mapped_column(ForeignKey("model_tests.id", ondelete="CASCADE"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    original_name: Mapped[str] = mapped_column(String(255))
    declared_size: Mapped[int] = mapped_column(Integer)
    stored_size: Mapped[int] = mapped_column(Integer, default=0)
    uploaded: Mapped[bool] = mapped_column(Boolean, default=False)
    detections: Mapped[int | None] = mapped_column(Integer, nullable=True)
    objects: Mapped[list] = mapped_column(JSON, default=list)
    timings: Mapped[dict] = mapped_column(JSON, default=dict)

    test: Mapped[ModelTest] = relationship(back_populates="images")


class RemoteOperation(Base):
    """Durable node-scoped work queue; completed deletion rows are ACK tombstones."""
    __tablename__ = "remote_operations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    node_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    lease: Mapped[str | None] = mapped_column(String(32), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
