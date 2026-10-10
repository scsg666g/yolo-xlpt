from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TrainingTask(Base):
    __tablename__ = "training_tasks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    task_type: Mapped[str] = mapped_column(String(24), default="detect")
    status: Mapped[str] = mapped_column(String(24), default="draft", index=True)
    source: Mapped[str] = mapped_column(String(24), default="platform")
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
    current_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    deleted_reason: Mapped[str | None] = mapped_column(String(40), nullable=True)

    runs: Mapped[list["TrainingRun"]] = relationship(back_populates="task", cascade="all, delete-orphan", foreign_keys="TrainingRun.task_id")
    metrics: Mapped[list["MetricPoint"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="task", cascade="all, delete-orphan")
    events: Mapped[list["TaskEvent"]] = relationship(back_populates="task", cascade="all, delete-orphan")


class TrainingRun(Base):
    __tablename__ = "training_runs"
    __table_args__ = (UniqueConstraint("task_id", "attempt_number", name="uq_run_task_attempt"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("training_tasks.id"), index=True)
    attempt_number: Mapped[int] = mapped_column(Integer)
    launch_mode: Mapped[str] = mapped_column(String(24), default="fresh")
    status: Mapped[str] = mapped_column(String(24), default="queued", index=True)
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
