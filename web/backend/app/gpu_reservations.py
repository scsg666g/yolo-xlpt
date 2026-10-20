"""Persistent GPU ownership shared by local and remote training claims."""
from __future__ import annotations

from typing import Any

from sqlalchemy import delete, select, update
from sqlalchemy.exc import IntegrityError

from .models import ComputeNode, GpuReservation, TrainingRun


TERMINAL_RUN_STATUSES = {"completed", "failed", "stopped", "interrupted"}


def requested_gpu_indices(device: Any) -> list[int]:
    value = str(device or "").strip().lower()
    if value in {"", "cpu", "mps"}:
        return []
    indices: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part.isdigit():
            return []
        index = int(part)
        if index not in indices:
            indices.append(index)
    return indices


def resolve_gpu_assignments(
    device: Any,
    gpus: list[dict[str, Any]] | None,
    *,
    node_id: str,
    require_uuid: bool = False,
) -> list[dict[str, Any]] | None:
    """Resolve configured device indices to durable identities.

    Old/local environments may fall back to a node-scoped index. Parallel
    remote claims require UUIDs because indices can change across runtimes.
    """
    indices = requested_gpu_indices(device)
    if not indices:
        return []
    inventory = {
        int(gpu["index"]): gpu
        for gpu in (gpus or [])
        if isinstance(gpu, dict) and type(gpu.get("index")) is int
    }
    assignments: list[dict[str, Any]] = []
    for index in indices:
        gpu = inventory.get(index)
        # An unknown index must not be treated as a reservable GPU. Callers
        # keep the task queued until inventory confirms the requested device.
        if gpu is None:
            return None
        raw_uuid = gpu.get("uuid")
        gpu_uuid = raw_uuid.strip() if isinstance(raw_uuid, str) else ""
        if require_uuid and not gpu_uuid:
            return None
        # UUIDs identify physical GPUs within one host; scope them to the
        # execution node so matching UUID values on different hosts don't clash.
        resource_key = (
            f"node:{node_id}:gpu:{gpu_uuid}"
            if gpu_uuid else f"node:{node_id}:gpu-index:{index}"
        )
        assignments.append({
            "resource_key": resource_key,
            "uuid": gpu_uuid or None,
            "index": index,
        })
    return assignments


def node_serial_assignment(node_id: str) -> dict[str, Any]:
    return {"resource_key": f"node:{node_id}:serial", "uuid": None, "index": None}


def acquire_node_claim_lock(db, node_id: str) -> None:
    """Serialize a new claim before it reads competing active runs.

    PostgreSQL takes a row lock. SQLite has no row-level FOR UPDATE, so begin
    the claim transaction with a no-op UPDATE; in WAL mode this obtains the
    single-writer reservation before the caller reads queue/resource state.
    Callers must invoke this before making changes in the transaction.
    """
    if db.get_bind().dialect.name == "sqlite":
        db.rollback()
        db.execute(update(ComputeNode).where(ComputeNode.id == node_id).values(id=ComputeNode.id))
        return
    db.scalar(select(ComputeNode.id).where(ComputeNode.id == node_id).with_for_update())


def non_gpu_assignment(node_id: str, run_id: str, device: str) -> dict[str, Any]:
    """Give non-GPU work an explicit reservation without blocking GPU slots."""
    normalized = str(device or "").strip().lower()
    if normalized == "cpu":
        # CPU tasks follow the local scheduler's existing behavior: they can
        # coexist with GPU work and with other CPU runs.
        key = f"node:{node_id}:cpu-run:{run_id}"
    elif normalized == "mps":
        # MPS is a single device per host, but independent of CUDA GPUs.
        key = f"node:{node_id}:mps"
    else:
        # Automatic/unspecified selection may use any accelerator, so preserve
        # safe node-wide exclusivity until the Agent can report an assignment.
        return node_serial_assignment(node_id)
    return {"resource_key": key, "uuid": None, "index": None}


def reserve_gpu_resources(
    db,
    *,
    run_id: str,
    node_id: str,
    assignments: list[dict[str, Any]],
    exclusive_node: bool = False,
    allow_recovery_conflict: bool = False,
) -> bool:
    """Acquire all requested resources in the caller's open transaction.

    Every claimant for a node locks the same ComputeNode row before examining
    active runs or reservations. This arbitrates unlike resource keys too
    (for example, a node-serial CPU claim against a UUID-backed GPU claim).
    """
    # PostgreSQL uses this row lock to serialize all claims on one node while
    # still allowing claims for different nodes to proceed independently.
    # SQLite ignores FOR UPDATE; its single-writer transaction and unique
    # resource_key constraint provide the corresponding arbitration.
    db.scalar(select(ComputeNode.id).where(ComputeNode.id == node_id).with_for_update())

    # Old processes/runs may predate reservations. They remain exclusive until
    # they finish, otherwise their unknown device usage could overlap a new run.
    active_without_reservation = select(TrainingRun.id).where(
        TrainingRun.execution_node_id == (None if node_id == "local" else node_id),
        TrainingRun.status.in_(["starting", "running", "stopping"]),
        TrainingRun.id != run_id,
        ~select(GpuReservation.run_id).where(
            GpuReservation.run_id == TrainingRun.id
        ).exists(),
    ).limit(1)
    if db.scalar(active_without_reservation):
        return False

    existing_for_node = db.scalars(
        select(GpuReservation).where(GpuReservation.node_id == node_id)
    ).all()
    conflict_prefix = f"node:{node_id}:recovery-conflict:"
    if not allow_recovery_conflict and any(
        item.resource_key.startswith(conflict_prefix) and item.run_id != run_id
        for item in existing_for_node
    ):
        return False
    if exclusive_node and any(item.run_id != run_id for item in existing_for_node):
        return False
    serial_key = node_serial_assignment(node_id)["resource_key"]
    if not exclusive_node and any(
        item.resource_key == serial_key and item.run_id != run_id for item in existing_for_node
    ):
        return False

    keys = [str(item["resource_key"]) for item in assignments]
    existing = {
        item.resource_key: item
        for item in db.scalars(
            select(GpuReservation).where(GpuReservation.resource_key.in_(keys))
        ).all()
    } if keys else {}
    if any(existing[key].run_id != run_id for key in keys if key in existing):
        return False
    try:
        # Isolate a unique-key race so callers can safely try another queued
        # candidate without discarding unrelated work in their transaction.
        # Add rows only after entering the savepoint: Session.begin_nested()
        # flushes pending state on entry, so adding first would let a unique
        # key violation escape before the savepoint exists.
        with db.begin_nested():
            for assignment in assignments:
                key = str(assignment["resource_key"])
                if key in existing:
                    continue
                db.add(GpuReservation(
                    resource_key=key,
                    run_id=run_id,
                    node_id=node_id,
                    gpu_uuid=assignment.get("uuid"),
                    gpu_index=assignment.get("index"),
                ))
            db.flush()
    except IntegrityError:
        return False
    return True


def reserve_recovery_conflict(db, *, run_id: str, node_id: str) -> None:
    """Persist a node-wide guard while a recovered process has unknown ownership.

    Each affected run gets its own key, so the guard remains until every
    conflicting live process has independently reached a terminal state.
    """
    key = f"node:{node_id}:recovery-conflict:{run_id}"
    db.scalar(select(ComputeNode.id).where(ComputeNode.id == node_id).with_for_update())
    if db.get(GpuReservation, key):
        return
    try:
        with db.begin_nested():
            db.add(GpuReservation(
                resource_key=key, run_id=run_id, node_id=node_id,
                gpu_uuid=None, gpu_index=None,
            ))
            db.flush()
    except IntegrityError:
        # Another recovery pass may have installed the same guard already.
        return


def release_gpu_reservations(db, run_id: str) -> None:
    db.execute(
        delete(GpuReservation).where(GpuReservation.run_id == run_id)
        .execution_options(synchronize_session=False)
    )


def release_terminal_gpu_reservations(db) -> int:
    result = db.execute(
        delete(GpuReservation).where(
            GpuReservation.run_id.in_(
                select(TrainingRun.id).where(TrainingRun.status.in_(TERMINAL_RUN_STATUSES))
            )
        ).execution_options(synchronize_session=False)
    )
    return result.rowcount or 0
