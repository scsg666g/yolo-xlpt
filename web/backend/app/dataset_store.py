"""Durable dataset catalogue and verified, node-neutral file operations.

Only a verified snapshot is published. Transfers pass through the centre; agents
need no inbound port. A phase token is stable across retries for crash receipts.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import logging
import os
from pathlib import Path, PurePosixPath
import shutil
import threading
import time
import uuid
from typing import Literal
from datetime import datetime, timedelta, timezone

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from . import dataset_files as files
from .dataset_progress import record_progress as record_work_progress, progress_reporter
from .config import (DATA_ROOT, DATASET_ROOTS, DATASET_STORAGE_ALLOWED_ROOTS, DATASET_STORAGE_QUOTA_BYTES,
                     DATASET_STORAGE_RETENTION_SECONDS, DATASET_FAILED_RETENTION_SECONDS, PROJECT_ROOT,
                     DATASET_REFRESH_READ_BYTES_PER_SECOND, DATASET_REFRESH_IO_PRESSURE_PCT,
                     DATASET_REFRESH_MIN_FREE_BYTES)
from .database import SessionLocal, get_db
from .models import ComputeNode, DatasetEntry, DatasetLocation, DatasetOperation, DatasetOperationHistoryRemoval, TrainingRun, TrainingTask, utcnow
from .gpu_reservations import acquire_node_claim_lock
from .dataset_uploads import managed_dataset_mutation, dataset_references, managed_marker, validate_name

router = APIRouter(prefix='/api/dataset-storage', tags=['dataset storage'])
STORE_ROOT = DATA_ROOT / 'dataset_storage'
ACTIVE = ('pending', 'running', 'committing')
LIGHTWEIGHT_PHASES = ('check', 'discover')
TERMINAL = ('completed', 'cancelled')
logger = logging.getLogger(__name__)


def uid():
    return uuid.uuid4().hex


def job_dir(identity):
    if len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('操作编号无效')
    return STORE_ROOT / identity


@contextmanager
def snapshot_access(identity, *, blocking=True):
    """Keep a snapshot alive across local work and centre-side Agent I/O.

    The lock lives outside the directory being renamed for reclamation. A
    bounded set of stripes avoids leaving one lock file per historical job.
    """
    job_dir(identity)
    locks = STORE_ROOT / '.snapshot-locks'
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / (identity[:3] + '.lock')).open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def failed_snapshot_expired(op, now=None):
    """Only disposable pre-commit snapshots may expire; receipts stay durable."""
    now = now or utcnow()
    return bool(DATASET_FAILED_RETENTION_SECONDS and op.status == 'failed'
                and op.kind in ('upload', 'register', 'refresh', 'migrate')
                and not op.payload.get('published') and not op.payload.get('commit_started')
                and (not op.lease_until or operation_time(op.lease_until) <= now)
                and (op.payload.get('snapshot_reclaimed') or
                     (now - operation_time(op.updated_at)).total_seconds() >= DATASET_FAILED_RETENTION_SECONDS))


def restart_snapshot(op):
    """Rotate the token before rebuilding a discarded/changed source snapshot."""
    for name in ('bundle.part', 'bundle.tar', 'bundle.tmp', 'manifest.json'):
        (job_dir(op.id) / name).unlink(missing_ok=True)
    payload = dict(op.payload)
    for key in ('archive_size', 'archive_sha256', 'fingerprint', 'snapshot_reclaimed', 'verified_archive',
                'snapshot_lock_v1', 'cache_cleaned_at', 'progress', 'progress_steps', 'started_at', 'heartbeat_at'):
        payload.pop(key, None)
    if op.kind != 'upload':
        for key in ('names', 'stats', 'warnings'):
            payload.pop(key, None)
    op.payload = payload
    op.result = {}
    set_phase(op, 'prepare_upload' if op.kind == 'upload' else 'export',
              'local' if op.kind == 'upload' else payload['source_node'])


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    temp.replace(path)


def node_or_error(db, node_id):
    if node_id == 'local':
        return None
    node = db.get(ComputeNode, node_id)
    if not node or not node.enabled:
        raise HTTPException(409, '远程节点不存在或已禁用')
    if tuple(int(p) for p in (node.agent_version or '0').split('.') if p.isdigit()) < (2, 1, 0):
        raise HTTPException(409, '远程数据集管理需要 Agent 2.1.0 或以上版本')
    capability = (node.snapshot or {}).get('training') or {}
    if not capability.get('dataset_roots'):
        raise HTTPException(409, '请在远程 Agent 配置 dataset_roots 并重新启动 Agent')
    return node


def node_has_dataset_work(db, node_id):
    return any(node_id in (op.node_id, op.payload.get('source_node'), op.payload.get('target_node'))
               for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_(ACTIVE))))


def operation_time(value):
    """SQLite returns naive UTC timestamps; APIs always include the timezone."""
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def training_active(db, node_id):
    node_filter = TrainingRun.execution_node_id == node_id
    if node_id == 'local':
        node_filter = or_(TrainingRun.execution_node_id.is_(None), node_filter)
    return bool(db.scalar(select(TrainingRun.id).where(
        node_filter,
        TrainingRun.status.in_(('starting', 'running', 'stopping'))).limit(1)))


def concurrent_dataset_reads(db, node_id):
    if node_id == 'local':
        return True
    node = db.get(ComputeNode, node_id)
    return bool(node and ((node.snapshot or {}).get('training') or {}).get('dataset_concurrent_reads') is True)


def concurrent_dataset_refresh(db, node_id):
    """Only a reporting Agent can opt into training-safe, read-only exports."""
    if node_id == 'local':
        return True
    node = db.get(ComputeNode, node_id)
    capability = ((node.snapshot or {}).get('training') or {}) if node else {}
    return capability.get('dataset_concurrent_reads') is True and capability.get('dataset_concurrent_refresh') is True


def read_only_refresh(op):
    return op.kind == 'refresh' and op.payload.get('phase') == 'export'


def training_safe_operation(op, *, concurrent_reads, concurrent_refresh):
    phase = op.payload.get('phase')
    return ((phase in LIGHTWEIGHT_PHASES and concurrent_reads)
            or (read_only_refresh(op) and concurrent_refresh))


def lightweight_operation(op):
    return op.payload.get('phase') in LIGHTWEIGHT_PHASES


def local_refresh(op):
    return op.kind == 'refresh' and op.payload.get('source_node') == 'local'


def refresh_lane_operation(op):
    """Centre-side read work, whether its immutable source is local or remote."""
    return (op.kind == 'refresh' and op.node_id == 'local'
            and (local_refresh(op) or op.payload.get('phase') == 'validate'))


def local_mutation_paths(op):
    """Paths the current local execution phase may write."""
    paths = []
    p = op.payload
    phase = p.get('phase')
    if phase == 'import' and op.kind in ('upload', 'migrate') and p.get('target_node') == 'local' and p.get('destination'):
        paths.append(p['destination'])
    if (phase == 'remove' and (op.kind == 'delete' or (op.kind == 'migrate' and p.get('remove_source')))
            and p.get('source_node') == 'local' and p.get('source_path')):
        paths.append(p['source_path'])
    if phase == 'check' and op.kind == 'check' and op.node_id == 'local' and p.get('path'):
        paths.append(p['path'])
    return paths


def local_paths_overlap(first, second):
    a, b = Path(first).resolve(), Path(second).resolve()
    return a == b or a in b.parents or b in a.parents


def local_mutation_conflicts(op, source):
    if op.kind == 'check':
        # Checking a parent location creates its tempfile in that parent, not
        # inside the dataset being reloaded.
        root = Path(source).resolve()
        return any((path := Path(raw).resolve()) == root or root in path.parents
                   for raw in local_mutation_paths(op))
    return any(local_paths_overlap(path, source) for path in local_mutation_paths(op))


def local_mutation_blocked_by_refresh(db, op):
    if not local_mutation_paths(op):
        return False
    for refresh in db.scalars(select(DatasetOperation).where(
            DatasetOperation.kind == 'refresh', DatasetOperation.status.in_(ACTIVE))):
        source = refresh.payload.get('source_path')
        # An older mutation has priority over a queued reload; otherwise both
        # lanes could wait on each other forever.
        if refresh.status == 'pending' and (operation_time(op.created_at), op.id) < (operation_time(refresh.created_at), refresh.id):
            continue
        if refresh.id != op.id and source and local_refresh(refresh) and local_mutation_conflicts(op, source):
            return True
    return False


def local_refresh_blocked_by_older_mutation(db, op):
    source = op.payload.get('source_path')
    if not source:
        return False
    for other in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_(ACTIVE))):
        if (other.id != op.id and (operation_time(other.created_at), other.id) < (operation_time(op.created_at), op.id)
                and local_mutation_conflicts(other, source)):
            return True
    return False


def operation_read(op, db=None):
    value = {'id': op.id, 'kind': op.kind, 'status': op.status, 'phase': op.payload.get('phase'),
             'node_id': op.node_id, 'name': op.payload.get('name', ''), 'error': op.error,
             'transferred': op.transferred, 'total_bytes': op.payload.get('archive_size', 0),
             'created_at': operation_time(op.created_at), 'updated_at': operation_time(op.updated_at), 'result': op.result,
             'source_path': op.payload.get('source_path') or op.payload.get('path'), 'destination': op.payload.get('destination'),
             'can_delete_history': op.status in TERMINAL,
             'can_cancel': op.status in ('pending', 'running', 'failed') and not op.payload.get('published') and not op.payload.get('commit_started')}
    if db:
        def node_name(identity):
            node = db.get(ComputeNode, identity) if identity else None
            return node.name if node else ('本机' if identity == 'local' else identity)
        value['node_name'] = node_name(op.node_id)
        for key in ('source_node', 'target_node'):
            value[key] = op.payload.get(key)
            value[key + '_name'] = node_name(op.payload.get(key))
        if not value['name'] and op.payload.get('location_id'):
            location = db.get(DatasetLocation, op.payload['location_id'])
            value['name'] = location.name if location else op.payload.get('location_name', '')
    progress = (op.payload.get('progress') or {}).copy()
    if progress:
        for private in ('run_id', 'baseline'):
            progress.pop(private, None)
    value.update(progress=progress or None, progress_steps=op.payload.get('progress_steps', []),
                 heartbeat_at=op.payload.get('heartbeat_at'),
                 started_at=op.payload.get('started_at'), queued_at=op.payload.get('queued_at'), queue_ahead=0,
                 waiting_lock=bool(progress and progress.get('stage') == 'waiting_lock' and op.status in ACTIVE))
    if db and op.status == 'pending':
        value['queue_ahead'] = db.scalar(select(func.count()).select_from(DatasetOperation).where(
            DatasetOperation.node_id == op.node_id, DatasetOperation.status.in_(ACTIVE),
            or_(DatasetOperation.created_at < op.created_at,
                (DatasetOperation.created_at == op.created_at) & (DatasetOperation.id < op.id)))) or 0
    if db and op.node_id != 'local' and op.status in ACTIVE:
        node = db.get(ComputeNode, op.node_id)
        seen = node.last_seen_at if node else None
        value['waiting_offline'] = not seen or (utcnow() - seen.replace(tzinfo=timezone.utc)).total_seconds() > 90
        capability = ((node.snapshot or {}).get('training') or {}) if node else {}
        concurrent = concurrent_dataset_reads(db, op.node_id)
        busy = training_active(db, op.node_id)
        refresh = concurrent_dataset_refresh(db, op.node_id)
        safe = training_safe_operation(op, concurrent_reads=concurrent, concurrent_refresh=refresh)
        value['waiting_training'] = op.status == 'pending' and busy and not safe
        value['waiting_lock'] = value['waiting_lock'] or (op.status == 'pending' and not safe and not value['waiting_training'] and bool(capability.get('lock_occupied')))
        if op.status == 'pending' and safe:
            value['waiting_lock'] = False  # Discard a pre-upgrade lock-wait sample.
        if op.status == 'pending' and safe and (busy or capability.get('lock_occupied')):
            # Blocked mutations do not precede runnable read-only work.
            earlier = db.scalars(select(DatasetOperation).where(
                DatasetOperation.node_id == op.node_id, DatasetOperation.status.in_(ACTIVE),
                or_(DatasetOperation.created_at < op.created_at,
                    (DatasetOperation.created_at == op.created_at) & (DatasetOperation.id < op.id))))
            value['queue_ahead'] = sum(training_safe_operation(
                item, concurrent_reads=concurrent, concurrent_refresh=refresh) or bool(
                item.lease_until and operation_time(item.lease_until) > utcnow()) for item in earlier)
    if db and op.node_id == 'local':
        value['waiting_path_lock'] = op.status == 'pending' and local_mutation_blocked_by_refresh(db, op)
        earlier_lane = [item for item in db.scalars(select(DatasetOperation).where(
            DatasetOperation.kind == 'refresh', DatasetOperation.node_id == 'local',
            DatasetOperation.status.in_(ACTIVE))) if refresh_lane_operation(item)
            and (operation_time(item.created_at), item.id) < (operation_time(op.created_at), op.id)]
        if refresh_lane_operation(op):
            value['waiting_training'] = False
            value['waiting_path_lock'] = (op.status == 'pending' and local_refresh(op)
                                          and local_refresh_blocked_by_older_mutation(db, op))
            runnable_refreshes = sum(1 for item in earlier_lane if item.status != 'pending'
                                     or not local_refresh(item)
                                     or not local_refresh_blocked_by_older_mutation(db, item))
            value['queue_ahead'] = runnable_refreshes
            value['waiting_resource'] = (op.status == 'pending' and runnable_refreshes > 0) or (
                op.status in ACTIVE and bool(progress and progress.get('stage') == 'paused_resource'))
        elif op.status == 'pending':
            value['queue_ahead'] -= len(earlier_lane)
    return value


def reference_rows(db, entry, context=None):
    if entry.node_id == 'local':
        tasks = dataset_references(db, str(Path(entry.path) / 'data.yaml'),
            local_tasks=context['tasks_by_node'].get('local', []) if context else None,
            historical=context['historical_by_node'].get('local', {}) if context else None)
    else:
        # We cannot open an unregistered remote YAML at the centre. Treat those
        # paths conservatively as potential aliases instead of guessing no refs.
        tasks = []
        candidates = (context['tasks_by_node'].get(entry.node_id, []) if context else
            list(db.scalars(select(TrainingTask).where(or_(TrainingTask.execution_node_id == entry.node_id,
                TrainingTask.id.in_(select(TrainingRun.task_id).where(TrainingRun.execution_node_id == entry.node_id)))))))
        for task in candidates:
            historical = (context['historical_by_node'].get(entry.node_id, {}).get(task.id, []) if context else
                db.scalars(select(TrainingRun.data_path).where(TrainingRun.task_id == task.id,
                    or_(TrainingRun.execution_node_id == entry.node_id,
                        TrainingRun.execution_node_id.is_(None) & (task.execution_node_id == entry.node_id)))))
            paths = [task.data_path if task.execution_node_id == entry.node_id else None, *historical]
            for raw in paths:
                if not raw:
                    continue
                p, root = PurePosixPath(raw), PurePosixPath(entry.path)
                known = (context['entries_by_path'].get((entry.node_id, str(p.parent))) if context else
                    db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == entry.node_id,
                                                        DatasetEntry.path == str(p.parent), DatasetEntry.status != 'deleted')))
                if p == root or root in p.parents or ((not known or p.name != 'data.yaml' or not known.details.get('config_sha256')) and p.suffix in {'.yaml', '.yml'}):
                    tasks.append(task)
                    break
    return [{'id': t.id, 'name': t.name, 'status': t.status, 'deleted': bool(t.deleted_at)} for t in tasks]


def remote_config_guards(db, node_id, source_path):
    if node_id == 'local':
        return []
    guards = []
    referenced = set(db.scalars(select(TrainingTask.data_path).where(TrainingTask.execution_node_id == node_id)))
    referenced.update(db.scalars(select(TrainingRun.data_path).where(TrainingRun.execution_node_id == node_id)))
    for entry in db.scalars(select(DatasetEntry).where(DatasetEntry.node_id == node_id, DatasetEntry.status == 'ready')):
        checksum = entry.details.get('config_sha256')
        path = str(PurePosixPath(entry.path) / 'data.yaml')
        if checksum and path in referenced and entry.path != source_path:
            guards.append({'path': path, 'sha256': checksum})
    return guards


def managed_path_problem(db, node_id, path, *, exclude_id=None, context=None):
    """Use the same ownership boundary before adoption, preview and removal."""
    candidate = PurePosixPath(path)
    if not candidate.is_absolute() or '..' in candidate.parts:
        return '目录必须为可解析的绝对路径，不能包含上级跳转'
    node = (context['nodes'].get(node_id) if context else db.get(ComputeNode, node_id)) if node_id != 'local' else None
    allowed = DATASET_STORAGE_ALLOWED_ROOTS if node_id == 'local' else (
        ((node.snapshot or {}).get('training') or {}).get('dataset_roots', []) if node else [])
    roots = [PurePosixPath(str(root)) for root in allowed]
    if roots and not any(root == candidate or root in candidate.parents for root in roots):
        return '目录不在数据集允许范围内，请核对服务器配置'
    locations = (context['locations_by_node'].get(node_id, []) if context else db.scalars(
        select(DatasetLocation).where(DatasetLocation.node_id == node_id)))
    location_roots = [PurePosixPath(loc.resolved_path or loc.path) for loc in locations]
    if any(candidate == root or candidate in root.parents for root in (*roots, *location_roots)):
        return '保存位置或允许路径的根目录及其上级不能接管或删除，请使用独立子目录'
    entries = (context['entries_by_node'].get(node_id, []) if context else db.scalars(select(DatasetEntry).where(
        DatasetEntry.node_id == node_id, DatasetEntry.status != 'deleted')))
    for entry in entries:
        if entry.id == exclude_id:
            continue
        existing = PurePosixPath(entry.path)
        if candidate == existing or candidate in existing.parents or existing in candidate.parents:
            return '目录与已登记的数据集相同或互相包含，请使用独立目录'
    return None


def active_training_on_node(db, node_id):
    node_filter = TrainingTask.execution_node_id == node_id
    if node_id == 'local':
        node_filter = or_(TrainingTask.execution_node_id.is_(None), node_filter)
    return bool(db.scalar(select(TrainingTask.id).where(
        node_filter, TrainingTask.status.in_(('running', 'starting', 'stopping', 'queued'))).limit(1)))


def removal_problem(db, entry, references=None, blocking=None, context=None):
    if entry.ownership != 'managed':
        return '数据集未接管，请先确认文件管理权'
    if entry.status != 'ready':
        return '数据集当前状态不可删除'
    if problem := managed_path_problem(db, entry.node_id, entry.path, exclude_id=entry.id, context=context):
        return problem
    if (references if references is not None else reference_rows(db, entry)):
        return '数据集仍被训练任务或历史实验引用'
    if blocking:
        return '数据集已有未完成操作，请先处理该操作'
    node = (context['nodes'].get(entry.node_id) if context else db.get(ComputeNode, entry.node_id)) if entry.node_id != 'local' else None
    if entry.node_id != 'local' and (not node or not node.enabled):
        return '远程节点不存在或已禁用'
    if entry.node_id != 'local' and not (entry.details or {}).get('fingerprint'):
        return '历史远程数据缺少校验指纹，请先重新加载校验'
    if (entry.node_id in context['active_nodes'] if context else active_training_on_node(db, entry.node_id)):
        return '该节点仍有活动训练任务，请停止后重试清理'
    return None


def catalogue_context(db, entries):
    """A response-local snapshot avoids one history/reference query per row."""
    context = {'nodes': {node.id: node for node in db.scalars(select(ComputeNode))},
               'entries_by_path': {(entry.node_id, entry.path): entry for entry in entries},
               'locations_by_node': {}, 'entries_by_node': {}, 'tasks_by_node': {},
               'historical_by_node': {}, 'busy': {}, 'refresh': {}, 'active_nodes': set()}
    for loc in db.scalars(select(DatasetLocation)):
        context['locations_by_node'].setdefault(loc.node_id, []).append(loc)
    for entry in entries:
        context['entries_by_node'].setdefault(entry.node_id, []).append(entry)
    tasks = list(db.scalars(select(TrainingTask).order_by(
        TrainingTask.deleted_at.is_not(None), TrainingTask.created_at.desc(), TrainingTask.id)))
    tasks_by_id = {task.id: task for task in tasks}
    candidates = {}
    for task in tasks:
        node_id = task.execution_node_id or 'local'
        candidates.setdefault(node_id, set()).add(task.id)
        if task.status in ('running', 'starting', 'stopping', 'queued'):
            context['active_nodes'].add(node_id)
    for task_id, path, run_node in db.execute(select(TrainingRun.task_id, TrainingRun.data_path, TrainingRun.execution_node_id)):
        if path:
            node_ids = {run_node or 'local'}
            # Old remote attempts may predate the node snapshot column. Keep
            # their current remote node protected as well as the local default.
            if run_node is None and task_id in tasks_by_id:
                node_ids.add(tasks_by_id[task_id].execution_node_id or 'local')
            for node_id in node_ids:
                context['historical_by_node'].setdefault(node_id, {}).setdefault(task_id, []).append(path)
                candidates.setdefault(node_id, set()).add(task_id)
    for node_id, identities in candidates.items():
        context['tasks_by_node'][node_id] = [task for task in tasks if task.id in identities]
    for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed')))):
        if identity := op.payload.get('entry_id'):
            context['busy'].setdefault(identity, op)
    latest = select(DatasetOperation.id, func.row_number().over(
        partition_by=DatasetOperation.payload['entry_id'].as_string(),
        order_by=(DatasetOperation.created_at.desc(), DatasetOperation.id.desc())).label('position')).where(
            DatasetOperation.kind == 'refresh',
            DatasetOperation.payload['entry_id'].as_string().in_([entry.id for entry in entries])).subquery()
    for op in db.scalars(select(DatasetOperation).where(DatasetOperation.id.in_(
            select(latest.c.id).where(latest.c.position == 1)))):
        context['refresh'][op.payload.get('entry_id')] = op
    return context


def entry_read(entry, db, context=None):
    d = entry.details or {}
    references = reference_rows(db, entry, context)
    blocking = context['busy'].get(entry.id) if context else busy_entry(db, entry.id)
    refresh = context['refresh'].get(entry.id) if context else db.scalar(select(DatasetOperation).where(
        DatasetOperation.kind == 'refresh', DatasetOperation.payload['entry_id'].as_string() == entry.id)
        .order_by(DatasetOperation.created_at.desc(), DatasetOperation.id.desc()).limit(1))
    delete_problem = removal_problem(db, entry, references, blocking, context)
    node = context['nodes'].get(entry.node_id) if context else db.get(ComputeNode, entry.node_id)
    return {'id': entry.id, 'location_id': entry.location_id, 'node_id': entry.node_id,
            'name': entry.name, 'node_name': (node.name if node else ('本机' if entry.node_id == 'local' else entry.node_id)), 'directory': entry.path, 'config_path': str(PurePosixPath(entry.path) / 'data.yaml'),
            'ownership': entry.ownership, 'status': entry.status, 'created_at': entry.created_at,
            'total_bytes': d.get('total_bytes', 0), 'stats': d.get('stats', {}), 'classes': d.get('names', []),
            'warnings': d.get('warnings', []), 'fingerprint': d.get('fingerprint'),
            'history': d.get('history', []), 'references': references,
            'checked_at': d.get('checked_at'),
            'refresh_status': refresh.status if refresh else None,
            'refresh_error': refresh.error if refresh else None,
            'blocking_operation_kind': blocking.kind if blocking else None,
            'blocking_operation_id': blocking.id if blocking else None,
            'can_delete': delete_problem is None, 'delete_block_reason': delete_problem}


def bootstrap(db):
    """Register old platform-owned folders without moving or rehashing them."""
    from .dataset_uploads import DATASET_ROOTS as legacy_roots
    for root in legacy_roots:
        identity = hashlib.sha256(('local:' + str(root.resolve())).encode()).hexdigest()[:32]
        location = db.get(DatasetLocation, identity)
        if not location:
            location = DatasetLocation(id=identity, node_id='local', name='原有数据集目录' if root == legacy_roots[0] else '兼容数据集目录',
                                       path=str(root), resolved_path=str(root.resolve()), enabled=True,
                                       is_default=root == legacy_roots[0] and not db.scalar(select(DatasetLocation.id).where(
                                           DatasetLocation.node_id == 'local', DatasetLocation.is_default.is_(True))),
                                       status='ready' if root.is_dir() else 'pending')
            db.add(location)
            db.flush()
        if not root.is_dir() or root.is_symlink():
            continue
        for child in root.iterdir():
            if child.name.startswith('.') or child.is_symlink() or not child.is_dir():
                continue
            metadata = managed_marker(child)
            if not metadata or not metadata.get('upload_id') or db.scalar(select(DatasetEntry.id).where(DatasetEntry.node_id == 'local', DatasetEntry.path == str(child.resolve()), DatasetEntry.status != 'deleted')):
                continue
            db.add(DatasetEntry(id=uid(), node_id='local', location_id=identity, name=child.name,
                                path=str(child.resolve()), ownership='managed', status='ready',
                                details={'names': metadata.get('classes', []), 'stats': metadata.get('stats', {}),
                                         'total_bytes': metadata.get('total_bytes', 0), 'warnings': metadata.get('warnings', [])}))
    db.commit()


def busy_entry(db, identity):
    return db.scalar(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed')),
                     DatasetOperation.payload['entry_id'].as_string() == identity).limit(1))


def mutation_pending(db, node_id):
    """Training claims and reference changes share the node DB claim lock."""
    for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed')))):
        if op.status == 'failed' and not op.payload.get('commit_started'):
            continue
        if op.kind == 'delete' or (op.kind == 'migrate' and op.payload.get('remove_source')):
            if op.payload.get('source_node') == node_id:
                return True
    return False


def node_dataset_file_work(db, node_id):
    concurrent = concurrent_dataset_reads(db, node_id)
    refresh = concurrent_dataset_refresh(db, node_id)
    return any(op.payload.get('phase') in ('export', 'import', 'remove', 'discover')
               and not training_safe_operation(
                   op, concurrent_reads=concurrent, concurrent_refresh=refresh) for op in db.scalars(
        select(DatasetOperation).where(DatasetOperation.node_id == node_id, DatasetOperation.status.in_(ACTIVE))))


def reject_training_mutation(db, node_id):
    if mutation_pending(db, node_id or 'local'):
        raise HTTPException(409, '该服务器正在迁移或删除数据集，请等待操作结束后再调整或启动训练')


def new_operation(db, kind, node_id, payload):
    payload = {**payload, 'queued_at': utcnow().isoformat()}
    op = DatasetOperation(id=uid(), kind=kind, node_id=node_id, status='pending',
                          payload=payload, result={}, lease=uid())
    db.add(op)
    db.flush()
    return op


def set_phase(op, phase, node_id):
    op.payload = {**op.payload, 'phase': phase, 'commit_started': False, 'queued_at': utcnow().isoformat()}
    op.node_id, op.status, op.lease, op.lease_until = node_id, 'pending', uid(), None
    op.transferred = 0
    payload = dict(op.payload)
    previous = payload.pop('progress', None)
    if previous:
        payload['progress_steps'] = [*payload.get('progress_steps', []),
            {k: previous.get(k) for k in ('stage', 'phase', 'executor')}][-16:]
    payload['progress_run_id'] = uid()
    op.payload = payload


def location_get(db, identity, *, ready=False):
    loc = db.get(DatasetLocation, identity)
    if not loc:
        raise HTTPException(404, '数据集保存位置不存在')
    node_or_error(db, loc.node_id)
    if ready and (not loc.enabled or loc.status != 'ready' or not loc.resolved_path):
        raise HTTPException(409, '保存位置尚未校验通过或已禁用')
    return loc


class LocationCreate(BaseModel):
    node_id: str = Field(default='local', max_length=64)
    name: str = Field(min_length=1, max_length=160)
    path: str = Field(min_length=1, max_length=4096)
    create: bool = False
    is_default: bool = False


class LocationUpdate(BaseModel):
    enabled: bool = True
    is_default: bool = False


class Register(BaseModel):
    location_id: str
    name: str = ''
    relative_path: str | None = Field(default=None, max_length=4096)
    ownership: str = 'external'
    acknowledge: bool = False


class Move(BaseModel):
    location_id: str
    name: str = ''
    remove_source: bool = False


@router.get('/locations')
def list_locations(db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        bootstrap(db)
    return [dict(id=l.id, node_id=l.node_id, node_name=(db.get(ComputeNode, l.node_id).name if db.get(ComputeNode, l.node_id) else ('本机' if l.node_id == 'local' else l.node_id)), name=l.name, path=l.path, resolved_path=l.resolved_path,
                 enabled=l.enabled, is_default=l.is_default, status=l.status, details=l.details, updated_at=l.updated_at)
            for l in db.scalars(select(DatasetLocation).order_by(DatasetLocation.updated_at))]


@router.post('/locations', status_code=201)
def create_location(body: LocationCreate, db: Session = Depends(get_db)):
    if not body.name.strip():
        raise HTTPException(422, '位置名称不能为空')
    with managed_dataset_mutation():
        acquire_node_claim_lock(db, body.node_id)
        node_or_error(db, body.node_id)
        if any(ord(c) < 32 for c in body.path) or '..' in PurePosixPath(body.path).parts:
            raise HTTPException(422, '目录不能包含控制字符或上级跳转')
        path = str((PROJECT_ROOT / body.path).absolute()) if body.node_id == 'local' else body.path
        if db.scalar(select(DatasetLocation).where(DatasetLocation.node_id == body.node_id, DatasetLocation.path == path)):
            raise HTTPException(409, '该保存位置已登记，请使用已有目录')
        loc = DatasetLocation(id=uid(), node_id=body.node_id, name=body.name.strip(), path=path,
                              enabled=True, is_default=False, status='pending', details={})
        db.add(loc)
        op = new_operation(db, 'check', body.node_id, {'phase': 'check', 'location_id': loc.id,
                           'path': path, 'create': body.create, 'make_default': body.is_default})
        db.commit()
        return {'id': loc.id, 'operation_id': op.id}


@router.patch('/locations/{identity}')
def update_location(identity: str, body: LocationUpdate, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        loc = location_get(db, identity)
        acquire_node_claim_lock(db, loc.node_id)
        if body.is_default and (not body.enabled or loc.status != 'ready'):
            raise HTTPException(409, '默认保存位置必须已启用且校验通过')
        if not body.enabled and any(op.payload.get('location_id') == identity for op in db.scalars(
                select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed'))))):
            raise HTTPException(409, '该位置仍有未完成的传输或校验')
        if body.is_default:
            for other in db.scalars(select(DatasetLocation).where(DatasetLocation.node_id == loc.node_id)):
                other.is_default = other.id == identity
        loc.enabled, loc.is_default = body.enabled, body.is_default
        db.commit()
    return {'ok': True}


@router.post('/locations/{identity}/check', status_code=202)
def recheck_location(identity: str, body: dict = Body(default={}), db: Session = Depends(get_db)):
    loc = location_get(db, identity)
    op = new_operation(db, 'check', loc.node_id, {'phase': 'check', 'location_id': identity, 'path': loc.path, 'create': body.get('create') is True})
    db.commit()
    return operation_read(op)


@router.delete('/locations/{identity}', status_code=204)
def forget_location(identity: str, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        loc = location_get(db, identity)
        if loc.is_default or db.scalar(select(DatasetEntry.id).where(
                DatasetEntry.location_id == identity, DatasetEntry.status != 'deleted')) or any(
                op.payload.get('location_id') == identity for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed'))))):
            raise HTTPException(409, '请先取消默认设置，并处理该位置的数据集和操作记录')
        snapshot = {'id': loc.id, 'name': loc.name, 'node_id': loc.node_id,
                    'path': loc.resolved_path or loc.path}
        # Keep audit records even after forgetting a now-empty location. There
        # is intentionally no cascading deletion of entries or receipts.
        for entry in db.scalars(select(DatasetEntry).where(DatasetEntry.location_id == identity)):
            entry.details = {**(entry.details or {}), 'location_snapshot': snapshot}
        for op in db.scalars(select(DatasetOperation).where(
                DatasetOperation.payload['location_id'].as_string() == identity)):
            op.payload = {**op.payload, 'location_name': loc.name, 'location_snapshot': snapshot}
        db.delete(loc)
        db.commit()


@router.get('/datasets')
def list_entries(db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        bootstrap(db)
    entries = list(db.scalars(select(DatasetEntry).where(DatasetEntry.status != 'deleted').order_by(DatasetEntry.created_at.desc())))
    context = catalogue_context(db, entries)
    return [entry_read(e, db, context) for e in entries]


@router.post('/datasets/register', status_code=202)
def register(body: Register, db: Session = Depends(get_db)):
    if body.ownership not in ('external', 'managed') or (body.ownership == 'managed' and not body.acknowledge):
        raise HTTPException(422, '接管文件管理需要明确确认；普通登记不会取得文件删除权')
    with managed_dataset_mutation():
        loc = location_get(db, body.location_id, ready=True)
        relative = body.relative_path if body.relative_path is not None else body.name
        op = queue_registration(db, loc, relative, body.name or PurePosixPath(relative).name or PurePosixPath(loc.resolved_path).name, body.ownership)
        db.commit()
        return operation_read(op)


def queue_registration(db, loc, relative, name, ownership, config_sha256=None, batch_id=None, *, claim_locked=False):
    from .dataset_discovery import relative_path
    relative = relative_path(relative)
    validate_name(name)
    if not claim_locked:
        acquire_node_claim_lock(db, loc.node_id)
    path = str(PurePosixPath(loc.resolved_path) / relative)
    if ownership == 'managed' and (problem := managed_path_problem(db, loc.node_id, path)):
        raise HTTPException(409, problem)
    for entry in db.scalars(select(DatasetEntry).where(DatasetEntry.node_id == loc.node_id, DatasetEntry.status != 'deleted')):
        if entry.path == path:
            raise HTTPException(409, '该目录已登记')
        if (ownership == 'managed' or entry.ownership == 'managed') and (
                PurePosixPath(entry.path) in PurePosixPath(path).parents or PurePosixPath(path) in PurePosixPath(entry.path).parents):
            raise HTTPException(409, '目录与已登记的平台管理数据集互相包含，请使用独立目录')
    if db.scalar(select(DatasetOperation.id).where(DatasetOperation.kind == 'register',
            DatasetOperation.payload['source_node'].as_string() == loc.node_id,
            DatasetOperation.payload['source_path'].as_string() == path,
            DatasetOperation.status.in_((*ACTIVE, 'failed')))):
        raise HTTPException(409, '该目录已有登记操作，请等待完成或重试／取消原操作')
    return new_operation(db, 'register', loc.node_id, {'phase': 'export', 'source_node': loc.node_id,
        'source_path': path, 'relative_path': relative, 'location_id': loc.id, 'name': name,
        'ownership': ownership, 'scan_config_sha256': config_sha256, 'batch_id': batch_id})


class RefreshRequest(BaseModel):
    request_id: str | None = Field(default=None, min_length=1, max_length=64, pattern=r'^[A-Za-z0-9_-]+$')


@router.post('/datasets/{identity}/refresh', status_code=202)
def refresh_entry(identity: str, body: RefreshRequest | None = Body(default=None), db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        request_id = body.request_id if body else None
        if request_id:
            previous = db.scalar(select(DatasetOperation).where(
                DatasetOperation.kind == 'refresh', DatasetOperation.payload['request_id'].as_string() == request_id))
            if previous:
                if previous.payload.get('entry_id') != identity:
                    raise HTTPException(409, '重新加载请求编号已用于另一个数据集')
                return operation_read(previous, db)
        entry = db.get(DatasetEntry, identity)
        if not entry or entry.status == 'deleted':
            raise HTTPException(404, '数据集不存在')
        if entry.status != 'ready' or busy_entry(db, identity):
            raise HTTPException(409, '数据集已有未完成操作，请等待完成或在操作记录中重试／取消')
        node_or_error(db, entry.node_id)
        acquire_node_claim_lock(db, entry.node_id)
        # Reuse the existing export protocol, including old Agents. Never import,
        # replace source files, change ownership, or create another catalogue row.
        op = new_operation(db, 'refresh', entry.node_id, {
            'phase': 'export', 'entry_id': entry.id, 'source_node': entry.node_id,
            'source_path': entry.path, 'name': entry.name, 'request_id': request_id})
        db.commit()
        return operation_read(op, db)


@router.get('/datasets/{identity}/delete-preview')
def delete_preview(identity: str, db: Session = Depends(get_db)):
    e = db.get(DatasetEntry, identity)
    if not e or e.status == 'deleted':
        raise HTTPException(404, '数据集不存在')
    value = entry_read(e, db)
    return value


@router.post('/datasets/{identity}/adopt')
def adopt(identity: str, body: dict, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        e = db.get(DatasetEntry, identity)
        if not e or e.status != 'ready' or busy_entry(db, identity):
            raise HTTPException(409, '数据集不可接管')
        if body.get('confirm_path') != e.path:
            raise HTTPException(422, '请填写完整目录以确认平台将获得文件删除权')
        acquire_node_claim_lock(db, e.node_id)
        node_or_error(db, e.node_id)
        if problem := managed_path_problem(db, e.node_id, e.path, exclude_id=e.id):
            raise HTTPException(409, problem)
        e.ownership = 'managed'
        db.commit()
    return {'ok': True}


@router.post('/datasets/{identity}/migrate', status_code=202)
def migrate(identity: str, body: Move, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        e = db.get(DatasetEntry, identity)
        if not e or e.status != 'ready' or busy_entry(db, identity):
            raise HTTPException(409, '数据集不存在或已有未完成操作')
        loc = location_get(db, body.location_id, ready=True)
        acquire_node_claim_lock(db, e.node_id)
        name = body.name or e.name
        validate_name(name)
        destination = str(PurePosixPath(loc.resolved_path) / name)
        if loc.node_id == e.node_id and (destination == e.path or PurePosixPath(e.path) in PurePosixPath(destination).parents or PurePosixPath(destination) in PurePosixPath(e.path).parents):
            raise HTTPException(409, '源与目标目录不能相同、互相包含')
        if body.remove_source and (problem := removal_problem(db, e)):
            raise HTTPException(409, problem + '；可选择仅复制并保留源数据')
        if loc.node_id != 'local':
            node_or_error(db, loc.node_id)
        op = new_operation(db, 'migrate', e.node_id, {'phase': 'export', 'entry_id': identity,
                           'source_node': e.node_id, 'source_path': e.path, 'location_id': loc.id,
                           'name': name, 'destination': destination, 'target_node': loc.node_id,
                           'remove_source': body.remove_source, 'config_guards': remote_config_guards(db, e.node_id, e.path) if body.remove_source else []})
        db.commit()
        return operation_read(op)


@router.delete('/datasets/{identity}', status_code=202)
def delete_entry(identity: str, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        e = db.get(DatasetEntry, identity)
        if not e or e.status == 'deleted':
            raise HTTPException(404, '数据集不存在')
        acquire_node_claim_lock(db, e.node_id)
        node_or_error(db, e.node_id)
        if problem := removal_problem(db, e, blocking=busy_entry(db, identity)):
            raise HTTPException(409, problem)
        op = new_operation(db, 'delete', e.node_id, {'phase': 'remove', 'entry_id': identity,
                           'source_node': e.node_id, 'source_path': e.path, 'name': e.name,
                           'fingerprint': (e.details or {}).get('fingerprint'), 'config_guards': remote_config_guards(db, e.node_id, e.path)})
        # Old managed entries get a first snapshot at deletion under the same lock.
        if e.node_id == 'local' and not op.payload['fingerprint']:
            op.payload = {**op.payload, 'initial_snapshot': True}
        db.commit()
        return operation_read(op)


def visible_operation():
    return ~select(DatasetOperationHistoryRemoval.operation_id).where(
        DatasetOperationHistoryRemoval.operation_id == DatasetOperation.id).exists()


def history_query(status='all', kind='all', node_id='all', q='', before=None, after=None):
    query = select(DatasetOperation).where(visible_operation(), DatasetOperation.created_at <= operation_time(before or utcnow()))
    if after:
        query = query.where(DatasetOperation.created_at >= operation_time(after))
    if status != 'all':
        statuses = (*ACTIVE, 'failed') if status == 'attention' else ACTIVE if status == 'active' else (status,)
        query = query.where(DatasetOperation.status.in_(statuses))
    if kind != 'all':
        query = query.where(DatasetOperation.kind == kind)
    if node_id != 'all':
        query = query.where(or_(DatasetOperation.node_id == node_id,
                               DatasetOperation.payload['source_node'].as_string() == node_id,
                               DatasetOperation.payload['target_node'].as_string() == node_id))
    if q.strip():
        location_ids = select(DatasetLocation.id).where(DatasetLocation.name.contains(q.strip(), autoescape=True))
        query = query.where(or_(DatasetOperation.payload['name'].as_string().contains(q.strip(), autoescape=True),
                               DatasetOperation.payload['location_name'].as_string().contains(q.strip(), autoescape=True),
                               (DatasetOperation.kind == 'check') & DatasetOperation.payload['location_id'].as_string().in_(location_ids)))
    return query


@router.get('/operations')
def list_operations(db: Session = Depends(get_db)):
    return [operation_read(op, db) for op in db.scalars(history_query().order_by(DatasetOperation.created_at.desc(), DatasetOperation.id.desc()).limit(200))]


@router.get('/operations/summary')
def operation_summary(db: Session = Depends(get_db)):
    counts = dict(db.execute(select(DatasetOperation.status, func.count()).where(visible_operation()).group_by(DatasetOperation.status)).all())
    node_ids = set(db.scalars(select(DatasetOperation.node_id).where(visible_operation()).distinct()))
    for key in ('source_node', 'target_node'):
        node_ids.update(db.scalars(select(DatasetOperation.payload[key].as_string()).where(visible_operation()).distinct()))
    nodes = {n.id: n.name for n in db.scalars(select(ComputeNode))}
    catalog_changed = db.scalar(select(func.max(DatasetEntry.updated_at)))
    finished_changed = db.scalar(select(func.max(DatasetOperation.updated_at)).where(
        DatasetOperation.status.in_((*TERMINAL, 'failed'))))
    # Digest reference-visible fields, not progress heartbeat/metric timestamps.
    reference_digest = hashlib.sha256()
    for row in db.execute(select(TrainingTask.id, TrainingTask.name, TrainingTask.data_path,
                                  TrainingTask.execution_node_id, TrainingTask.status, TrainingTask.deleted_at)
                          .order_by(TrainingTask.id)):
        reference_digest.update(json.dumps(tuple(row), default=str, ensure_ascii=False).encode())
    run_reference_state = db.execute(select(func.count(TrainingRun.id), func.max(TrainingRun.created_at))).one()
    reference_digest.update(json.dumps(tuple(run_reference_state), default=str).encode())
    revision = json.dumps([counts, str(catalog_changed), str(finished_changed), reference_digest.hexdigest()], sort_keys=True)
    return {'total': sum(counts.values()), 'active': sum(counts.get(s, 0) for s in ACTIVE),
            'failed': counts.get('failed', 0), 'counts': counts, 'revision': revision,
            'reference_revision': reference_digest.hexdigest(), 'managed_task_types': ['detect'],
            'latest_created_at': operation_time(db.scalar(select(func.max(DatasetOperation.created_at)).where(visible_operation()))),
            'snapshot': utcnow(),
            'nodes': [{'id': identity, 'name': nodes.get(identity, '本机' if identity == 'local' else identity)}
                      for identity in sorted(n for n in node_ids if n)]}


@router.get('/operations/page')
def operation_page(
    page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
    status: Literal['all', 'attention', 'active', 'pending', 'running', 'committing', 'completed', 'failed', 'cancelled'] = 'all',
    kind: Literal['all', 'check', 'upload', 'register', 'refresh', 'migrate', 'delete', 'discover'] = 'all',
    node_id: str = Query('all', max_length=64), q: str = Query('', max_length=160),
    before: datetime | None = None, after: datetime | None = None, through: datetime | None = None, db: Session = Depends(get_db),
):
    cutoff = operation_time(before or utcnow())
    upper = min(cutoff, operation_time(through)) if through else cutoff
    if after and operation_time(after) > upper:
        raise HTTPException(422, '开始时间不能晚于结束时间')
    query = history_query(status, kind, node_id, q, upper, after)
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    # Clamp after cancellations/filter changes without changing the creation snapshot.
    page = min(page, max(1, (total + page_size - 1) // page_size))
    rows = db.scalars(query.order_by(DatasetOperation.created_at.desc(), DatasetOperation.id.desc())
                      .offset((page - 1) * page_size).limit(page_size))
    return {'items': [operation_read(op, db) for op in rows], 'total': total,
            'page': page, 'page_size': page_size, 'snapshot': cutoff}


class HistoryRemovalRequest(BaseModel):
    ids: list[str] | None = Field(default=None, min_length=1, max_length=500)
    status: Literal['all', 'attention', 'active', 'pending', 'running', 'committing', 'completed', 'failed', 'cancelled'] = 'all'
    kind: Literal['all', 'check', 'upload', 'register', 'refresh', 'migrate', 'delete', 'discover'] = 'all'
    node_id: str = Field(default='all', max_length=64)
    q: str = Field(default='', max_length=160)
    before: datetime
    after: datetime | None = None
    fingerprint: str | None = None


def removal_candidates(db, body):
    if body.after and operation_time(body.after) > operation_time(body.before):
        raise HTTPException(422, '开始时间不能晚于结束时间')
    if body.ids is not None:
        rows = db.execute(select(DatasetOperation.id, DatasetOperation.status).where(DatasetOperation.id.in_(body.ids))).all()
        if {row.id for row in rows} != set(body.ids):
            raise HTTPException(404, '部分操作记录不存在，请刷新后重试')
        if any(row.status not in TERMINAL for row in rows):
            raise HTTPException(409, '只能删除已完成或已取消的操作记录；失败操作请先重试或取消')
        removed = set(db.scalars(select(DatasetOperationHistoryRemoval.operation_id).where(DatasetOperationHistoryRemoval.operation_id.in_(body.ids))))
        identities = [row.id for row in rows if row.id not in removed]
    else:
        identities = list(db.scalars(history_query(body.status, body.kind, body.node_id, body.q, body.before, body.after)
                               .where(DatasetOperation.status.in_(TERMINAL)).with_only_columns(DatasetOperation.id)))
    fingerprint = hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest()
    return identities, fingerprint


@router.post('/operations/history-removal-preview')
def history_removal_preview(body: HistoryRemovalRequest, db: Session = Depends(get_db)):
    rows, fingerprint = removal_candidates(db, body)
    return {'count': len(rows), 'fingerprint': fingerprint}


@router.post('/operations/history-removal')
def remove_history(body: HistoryRemovalRequest, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        rows, fingerprint = removal_candidates(db, body)
        if not body.fingerprint or fingerprint != body.fingerprint:
            raise HTTPException(409, '可清理记录已变化，请重新预览并确认数量')
        # Journal rows, leases and timestamps remain intact for replay and cleanup.
        db.add_all([DatasetOperationHistoryRemoval(operation_id=identity) for identity in rows])
        db.commit()
    return {'removed': len(rows)}


@router.delete('/operations/{identity}/history')
def remove_history_entry(identity: str, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        op = db.get(DatasetOperation, identity)
        if not op:
            raise HTTPException(404, '操作记录不存在')
        if op.status not in TERMINAL:
            raise HTTPException(409, '只能删除已完成或已取消的操作记录')
        if not db.get(DatasetOperationHistoryRemoval, identity):
            db.add(DatasetOperationHistoryRemoval(operation_id=identity))
            db.commit()
    return {'removed': 1}


@router.get('/operations/{identity}')
def get_operation(identity: str, db: Session = Depends(get_db)):
    op = db.get(DatasetOperation, identity)
    if not op:
        raise HTTPException(404, '操作记录不存在')
    return operation_read(op, db)


@router.post('/operations/{identity}/{action}')
def control_operation(identity: str, action: str, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        op = db.get(DatasetOperation, identity)
        if not op:
            raise HTTPException(404, '操作记录不存在')
        if action == 'retry' and op.status == 'failed':
            if op.kind == 'refresh' or op.payload.get('snapshot_reclaimed') or failed_snapshot_expired(op):
                # A corrected source must be exported again, with a new receipt
                # token. Revalidating the cached failed snapshot cannot see fixes.
                restart_snapshot(op)
            op.status, op.error, op.lease_until = 'pending', None, None
            op.payload = {**op.payload, 'progress_run_id': uid(), 'progress': None, 'queued_at': utcnow().isoformat()}
        elif action == 'cancel' and op.status in ('pending', 'running', 'failed') and not op.payload.get('published') and not op.payload.get('commit_started'):
            op.status = 'cancelled'
        else:
            raise HTTPException(409, '当前阶段不能执行该操作；提交文件后只能完成或重试')
        db.commit()
    return operation_read(op)


def queue_upload(db, upload_id, directory, manifest, names, stats, warnings):
    loc = location_get(db, manifest['location_id'], ready=True)
    # Frozen destination belongs to this session, never silently follow a new default.
    if loc.node_id != manifest['node_id'] or loc.resolved_path != manifest['root']:
        raise HTTPException(409, '上传目标位置已改变，请新建上传')
    existing = db.get(DatasetOperation, upload_id)
    if existing:
        return operation_read(existing)
    op = DatasetOperation(id=upload_id, kind='upload', node_id='local', status='pending', lease=uid(),
                          payload={'phase': 'prepare_upload', 'location_id': loc.id, 'target_node': loc.node_id,
                                   'destination': str(PurePosixPath(manifest['root']) / manifest['name']),
                                   'name': manifest['name'], 'staging': str(directory / 'files'),
                                   'names': names, 'stats': stats, 'warnings': warnings}, result={})
    db.add(op)
    db.commit()
    return operation_read(op)


def verify_snapshot(op, check=None):
    from .dataset_uploads import class_names, validate_contents, validate_file_path, MAX_FILE_BYTES, MAX_TEXT_BYTES
    directory = job_dir(op.id)
    manifest = json.loads((directory / 'manifest.json').read_text())
    archive = directory / 'bundle.tar'
    receipt = op.payload.get('verified_archive') or {}
    # Remote receipt acceptance already hashes the complete archive. Reuse only
    # this process-generated receipt while the exact file identity is unchanged;
    # historical operations and modified files still get a complete hash read.
    if (receipt.get('sha256') != manifest['archive_sha256']
            or receipt.get('file') != archive_identity(archive)):
        if files.digest(archive, check, stage='archive_hash') != manifest['archive_sha256']:
            raise ValueError('归档 SHA-256 校验失败')
    target = directory / 'validate'
    if target.exists():
        shutil.rmtree(target)
    try:
        files.unpack(directory / 'bundle.tar', target, manifest, check)
        for item in manifest['files']:
            relative = validate_file_path(item['path'])
            if item['size'] > (MAX_FILE_BYTES if relative.parts[0] == 'images' else MAX_TEXT_BYTES):
                raise ValueError('已有数据集单文件超过上传大小限制')
        names = class_names(target, op.payload.get('names', []))
        stats, warnings = validate_contents(target, manifest, names, check)
        if op.payload.get('scan_config_sha256') and files.digest(target / 'data.yaml') != op.payload['scan_config_sha256']:
            raise ValueError('data.yaml 在扫描后发生变化，请取消本次登记并重新扫描确认')
        if op.kind in ('register', 'refresh', 'migrate'):
            cfg = yaml.safe_load((target / 'data.yaml').read_text(encoding='utf-8-sig'))
            root = PurePosixPath(op.payload['source_path'])
            configured_root = PurePosixPath(str(cfg.get('path', root)))
            if not configured_root.is_absolute():
                configured_root = root / configured_root
            if str(configured_root).rstrip('/') != str(root) or cfg.get('train') != 'images/train' or cfg.get('val') != 'images/val':
                raise ValueError('登记要求自包含目录：data.yaml 的 path 指向本目录，train/val 为 images/train、images/val；请先整理或通过上传导入')
        return manifest, names, stats, warnings
    finally:
        shutil.rmtree(target, ignore_errors=True)


def archive_identity(path):
    value = Path(path).stat()
    return [value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def action_payload(op, *, concurrent_refresh=False):
    p, phase = op.payload, op.payload['phase']
    common = {'id': op.id, 'token': op.lease, 'phase': phase, 'progress_run_id': p.get('progress_run_id')}
    if phase == 'discover':
        return {**common, 'path': p['source_path'], 'limits': p['limits']}
    if phase == 'check':
        return {**common, 'path': p['path'], 'create': p.get('create', False)}
    if phase == 'export':
        return {**common, 'path': p['source_path'],
                **({'kind': 'refresh', 'read_only_during_training': True}
                   if read_only_refresh(op) and concurrent_refresh else {})}
    if phase == 'import':
        return {**common, 'root': str(PurePosixPath(p['destination']).parent), 'name': p['name'],
                'manifest': json.loads((job_dir(op.id) / 'manifest.json').read_text()),
                'config_text': yaml.safe_dump({'path': p['destination'], 'train': 'images/train', 'val': 'images/val',
                                              'names': dict(enumerate(p['names']))}, allow_unicode=True, sort_keys=False)}
    if phase == 'remove':
        return {**common, 'path': p['source_path'], 'fingerprint': p['fingerprint'], 'config_guards': p.get('config_guards', [])}
    return common


def phase_success(db, op, result):
    p, phase = op.payload, op.payload['phase']
    if phase == 'discover':
        from .dataset_discovery import record_progress
        record_progress(db, op, result)
        op.status = 'completed'
    elif phase == 'check':
        loc = db.get(DatasetLocation, p['location_id'])
        loc.resolved_path, loc.status, loc.details = result['resolved_path'], 'ready', {**result, 'checked_at': utcnow().isoformat()}
        if p.get('make_default'):
            for other in db.scalars(select(DatasetLocation).where(DatasetLocation.node_id == loc.node_id)):
                other.is_default = other.id == loc.id
        op.result = result
        op.status = 'completed'
    elif phase in ('export', 'prepare_upload'):
        op.payload = {**p, **{k: result[k] for k in ('archive_size', 'archive_sha256', 'fingerprint')}}
        set_phase(op, 'validate', 'local')
    elif phase == 'validate':
        manifest, names, stats, warnings = result
        op.payload = {**p, 'names': names, 'stats': stats, 'warnings': warnings}
        if op.kind == 'refresh':
            entry = db.get(DatasetEntry, p['entry_id'])
            if not entry or entry.status != 'ready' or entry.node_id != p['source_node'] or entry.path != p['source_path']:
                raise ValueError('数据集位置或状态已改变，请取消并重新加载')
            entry.details = {**(entry.details or {}), 'fingerprint': manifest['fingerprint'],
                             'names': names, 'stats': stats, 'warnings': warnings,
                             'total_bytes': manifest['total_bytes'], 'checked_at': utcnow().isoformat(),
                             'config_sha256': next(f['sha256'] for f in manifest['files'] if f['path'] == 'data.yaml')}
            op.result = {'dataset_id': entry.id}
            op.status = 'completed'
        elif op.kind == 'register':
            existing = db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == p['source_node'], DatasetEntry.path == p['source_path'], DatasetEntry.status != 'deleted'))
            if existing:
                raise ValueError('目录已经登记，拒绝重复登记')
            if p['ownership'] == 'managed' and (problem := managed_path_problem(db, p['source_node'], p['source_path'])):
                raise ValueError(problem)
            e = DatasetEntry(id=uid(), node_id=p['source_node'], location_id=p['location_id'], name=p['name'],
                             path=p['source_path'], ownership=p['ownership'], status='ready',
                             details={'fingerprint': manifest['fingerprint'], 'names': names, 'stats': stats,
                                      'warnings': warnings, 'total_bytes': manifest['total_bytes'], 'checked_at': utcnow().isoformat(), 'config_sha256': next(f['sha256'] for f in manifest['files'] if f['path'] == 'data.yaml')})
            db.add(e)
            db.flush()
            op.result = {'dataset_id': e.id}
            op.status = 'completed'
        else:
            set_phase(op, 'import', p['target_node'])
    elif phase == 'import':
        dest = db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == p['target_node'], DatasetEntry.path == p['destination'], DatasetEntry.status != 'deleted'))
        if dest and dest.status != 'deleted' and dest.details.get('operation_id') != op.id:
            raise ValueError('目标目录已有其他数据集登记')
        history = []
        if p.get('entry_id'):
            source = db.get(DatasetEntry, p['entry_id'])
            history = [*(source.details or {}).get('history', []), {'node_id': source.node_id, 'path': source.path, 'operation_id': op.id}]
        details = {'operation_id': op.id, 'fingerprint': result['fingerprint'], 'names': p['names'],
                   'stats': p['stats'], 'warnings': p['warnings'], 'total_bytes': result.get('total_bytes', 0), 'config_sha256': result.get('config_sha256'), 'history': history, 'checked_at': utcnow().isoformat()}
        if not dest:
            dest = DatasetEntry(id=uid(), node_id=p['target_node'], location_id=p['location_id'], name=p['name'], path=p['destination'])
            db.add(dest)
        dest.status, dest.ownership, dest.details = 'ready', 'managed', details
        db.flush()
        op.result = {'dataset_id': dest.id, 'name': dest.name, 'directory': dest.path, 'node_id': dest.node_id,
                     'names': p['names'], 'stats': p['stats'], 'warnings': p['warnings'],
                     'resource': {'name': dest.name, 'path': str(PurePosixPath(dest.path) / 'data.yaml'),
                                  'root': str(PurePosixPath(dest.path).parent), 'size': 0}}
        op.payload = {**p, 'published': True}
        if p.get('remove_source'):
            set_phase(op, 'remove', p['source_node'])
        else:
            op.status = 'completed'
    elif phase == 'remove':
        source = db.get(DatasetEntry, p['entry_id'])
        if op.kind == 'migrate':
            # Move the stable identity only after source cleanup is confirmed.

            destination = db.get(DatasetEntry, op.result.get('dataset_id'))
            destination_id = destination.id
            new_details = {**destination.details, 'copied_entry_id': destination_id}
            new_path, new_node, new_location, new_name = destination.path, destination.node_id, destination.location_id, destination.name
            db.delete(destination)
            db.flush()
            source.path, source.node_id, source.location_id, source.name = new_path, new_node, new_location, new_name
            source.details = new_details
            op.result = {**op.result, 'dataset_id': source.id}
        else:
            source.status = 'deleted'
        op.status = 'completed'
        op.result = {**op.result, 'source_removed': True}
    op.updated_at = utcnow()
    db.commit()


def guard_remove(db, op):
    entry = db.get(DatasetEntry, op.payload['entry_id'])
    acquire_node_claim_lock(db, op.payload['source_node'])
    if not entry:
        raise ValueError('源数据集不存在，拒绝删除')
    if problem := removal_problem(db, entry):
        raise ValueError(problem)


def mark_error(db, op, exc):
    db.rollback()
    op = db.get(DatasetOperation, op.id)
    if op.status == 'cancelled':
        return
    op.status, op.error = 'failed', str(getattr(exc, 'detail', None) or str(exc) or type(exc).__name__)[:1200]
    if op.kind == 'check':
        loc = db.get(DatasetLocation, op.payload['location_id'])
        if loc:
            loc.status, loc.details = 'error', {'message': op.error}
    db.commit()


def lease_check(db, identity, token, phase=None, committing=False):
    db.expire_all()
    op = db.get(DatasetOperation, identity)
    if not op or op.lease != token or op.status not in ACTIVE or (phase and op.payload['phase'] != phase):
        raise HTTPException(409, '操作已取消或执行阶段已变化')
    if committing:
        if op.payload['phase'] == 'remove':
            guard_remove(db, op)
        op.status = 'committing'
        op.payload = {**op.payload, 'commit_started': True}
    op.lease_until = utcnow() + timedelta(seconds=120)
    op.payload = {**op.payload, 'heartbeat_at': utcnow().isoformat()}
    db.commit()
    return op


def local_io_pressure_pct():
    """Linux PSI is optional; missing support leaves the byte-rate budget active."""
    path = os.getenv('PLATFORM_DATASET_REFRESH_IO_PRESSURE_FILE', '/proc/pressure/io')
    if not path:
        return 0.0
    try:
        pressure_path = Path(path).expanduser()
        if not pressure_path.is_absolute():
            pressure_path = PROJECT_ROOT / pressure_path
        line = next((line for line in pressure_path.read_text().splitlines() if line.startswith('some ')), '')
        return float(next(part[6:] for part in line.split() if part.startswith('avg10=')))
    except (OSError, StopIteration, ValueError):
        return 0.0


def local_refresh_free_bytes():
    return shutil.disk_usage(STORE_ROOT).free


def refresh_snapshot_disk_budget(manifest):
    """Conservative tar headers and unpacked filesystem metadata allowance."""
    paths = [len(item['path'].encode('utf-8')) for item in manifest['files']]
    # GNU/PAX headers and block rounding can exceed a simple two-header
    # allowance for long or non-ASCII paths. The unpacked tree also consumes
    # directory entries and filesystem metadata.
    metadata = sum(4096 + 2 * size for size in paths)
    archive = manifest['total_bytes'] + metadata
    unpacked = manifest['total_bytes'] + metadata
    return archive, unpacked


class LocalRefreshBudget:
    """Throttle only read-heavy reload work, including scans and archive copies."""
    def __init__(self, check):
        self.check = check
        self.last_training_probe = 0.0
        self.training = False
        self.next_read_at = 0.0
        self.last_resource_probe = 0.0
        self.pressure = 0.0
        self.free_bytes = None
        self.paused = False
        self.resume_progress = None

    def remember_progress(self, value):
        if value.get('stage') != 'paused_resource':
            self.resume_progress = dict(value)

    def training_now(self):
        now = time.monotonic()
        if now - self.last_training_probe >= 1:
            with SessionLocal() as db:
                self.training = training_active(db, 'local')
            self.last_training_probe = now
        return self.training

    def guard(self):
        while True:
            training = self.training_now()
            now = time.monotonic()
            if now - self.last_resource_probe >= 2:
                if DATASET_REFRESH_MIN_FREE_BYTES:
                    self.free_bytes = local_refresh_free_bytes()
                if training and DATASET_REFRESH_IO_PRESSURE_PCT:
                    self.pressure = local_io_pressure_pct()
                self.last_resource_probe = now
            reason = None
            if (DATASET_REFRESH_MIN_FREE_BYTES and self.free_bytes is not None
                    and self.free_bytes < DATASET_REFRESH_MIN_FREE_BYTES):
                reason = 'low_disk_space'
            elif training and DATASET_REFRESH_IO_PRESSURE_PCT and self.pressure >= DATASET_REFRESH_IO_PRESSURE_PCT:
                reason = 'disk_pressure'
            if not reason:
                if self.paused and self.resume_progress:
                    self.check.progress(self.resume_progress)
                self.paused = False
                if not training:
                    self.next_read_at = 0.0
                return training
            self.paused = True
            self.check.progress({'stage': 'paused_resource', 'resource_reason': reason})
            self.check()
            time.sleep(0.25)

    def throttle(self, byte_count):
        if not byte_count or not self.guard():
            return
        rate = DATASET_REFRESH_READ_BYTES_PER_SECOND
        if not rate:
            return
        self.next_read_at = max(self.next_read_at, time.monotonic()) + byte_count / rate
        while self.training_now():
            remaining = self.next_read_at - time.monotonic()
            if remaining <= 0:
                return
            self.check()
            time.sleep(min(0.25, remaining))
        self.next_read_at = 0.0


def execute_local(db, op):
    identity, token, phase = op.id, op.lease, op.payload['phase']
    run_id = uid()
    with managed_dataset_mutation():
        lease_check(db, identity, token, phase)
        op.status = 'running' if op.status != 'committing' else 'committing'
        op.payload = {**op.payload, 'progress_run_id': run_id,
                      'snapshot_lock_v1': True,
                      'started_at': op.payload.get('started_at') or utcnow().isoformat()}
        db.commit()
    last = [0.0]
    def check(stage=None):
        if stage or time.monotonic() - last[0] > 4:
            with managed_dataset_mutation():
                lease_check(db, identity, token, phase, committing=stage == 'committing')
            last[0] = time.monotonic()
    def persist(value):
        with managed_dataset_mutation():
            current = lease_check(db, identity, token, phase)
            if record_work_progress(current, value, run_id, 'local'):
                db.commit()
    check.progress = progress_reporter(persist)
    if refresh_lane_operation(op):
        budget = LocalRefreshBudget(check)
        report = check.progress
        def budgeted_progress(value):
            budget.remember_progress(value)
            report(value)
        check.progress = budgeted_progress
        check.throttle_bytes = budget.throttle
    check.progress({'stage': phase})
    check()
    p = op.payload
    directory = job_dir(op.id)
    directory.mkdir(parents=True, exist_ok=True)
    if phase == 'discover':
        from .dataset_discovery import record_progress
        from .dataset_discovery_files import discover
        def report(progress):
            with managed_dataset_mutation():
                current = lease_check(db, identity, token, phase)
                record_progress(db, current, progress)
                db.commit()
            check.progress({'stage': 'discover', 'processed': progress['visited'],
                            'unit': 'directories', 'current_file': progress.get('current_path', '')})
        result = discover(p['source_path'], DATASET_STORAGE_ALLOWED_ROOTS, p['limits'], report, check)
    elif phase == 'check':
        result = files.check_root(p['path'], DATASET_STORAGE_ALLOWED_ROOTS, p.get('create', False))
    elif phase in ('export', 'prepare_upload'):
        source = p['staging'] if phase == 'prepare_upload' else str(files.inside(p['source_path'], DATASET_STORAGE_ALLOWED_ROOTS))
        manifest = files.scan(source, check)
        archive_budget, unpack_budget = refresh_snapshot_disk_budget(manifest) if local_refresh(op) else (
            manifest['total_bytes'], manifest['total_bytes'])
        if DATASET_STORAGE_QUOTA_BYTES and sum(f.stat().st_size for f in STORE_ROOT.glob('*/bundle*')) + archive_budget + unpack_budget > DATASET_STORAGE_QUOTA_BYTES:
            raise ValueError('平台数据集中转配额不足，请先处理失败操作或增加 PLATFORM_DATASET_STORAGE_QUOTA_BYTES')
        reserve = max(100 * 1024**2, DATASET_REFRESH_MIN_FREE_BYTES) if local_refresh(op) else 100 * 1024**2
        required = archive_budget + unpack_budget + reserve
        if shutil.disk_usage(directory).free < required:
            raise ValueError('平台中转磁盘空间不足')
        result = files.pack(source, directory / 'bundle.tar', check, manifest=manifest)
        save_json(directory / 'manifest.json', result)
    elif phase == 'validate':
        if refresh_lane_operation(op):
            manifest = json.loads((directory / 'manifest.json').read_text())
            _, unpack_budget = refresh_snapshot_disk_budget(manifest)
            if shutil.disk_usage(directory).free < unpack_budget + DATASET_REFRESH_MIN_FREE_BYTES:
                raise ValueError('平台中转磁盘空间不足，无法安全展开重新加载快照')
        result = verify_snapshot(op, check)
        if local_refresh(op):
            source = files.inside(p['source_path'], DATASET_STORAGE_ALLOWED_ROOTS)
            if files.scan(source, check, stage='source_verify')['fingerprint'] != result[0]['fingerprint']:
                raise ValueError('源数据在重新加载期间发生修改，保留原有统计，请重试')
    elif phase == 'import':
        action = action_payload(op)
        root = files.inside(action['root'], DATASET_STORAGE_ALLOWED_ROOTS)
        result = files.publish(directory / 'bundle.tar', root, p['name'], action['manifest'], token, action['config_text'], check)
    elif phase == 'remove':
        guard_remove(db, op)
        if not p.get('fingerprint') and p.get('initial_snapshot'):
            fingerprint = files.scan(p['source_path'], check)['fingerprint']
            with managed_dataset_mutation():
                lease_check(db, identity, token, phase)
                op.payload = {**op.payload, 'fingerprint': fingerprint}
                db.commit()
            p = op.payload
        result = files.remove_dataset(p['source_path'], DATASET_STORAGE_ALLOWED_ROOTS, p['fingerprint'], check)
    else:
        raise ValueError('未知的数据集操作阶段')
    with managed_dataset_mutation():
        lease_check(db, identity, token, phase)
        phase_success(db, op, result)


def reclaim_snapshot(db, identity, *, now=None):
    """Detach a disposable centre snapshot only after its last reader exits."""
    directory = job_dir(identity)
    if not directory.is_dir() or directory.is_symlink():
        return False
    now = now or utcnow()
    # Always take the snapshot lock before the mutation lock. Executors use
    # the same order, so cancellation cannot deadlock with a running export.
    with snapshot_access(identity, blocking=False) as available:
        if not available:
            return False
        with managed_dataset_mutation():
            if not directory.is_dir() or directory.is_symlink():
                return False
            db.expire_all()
            op = db.get(DatasetOperation, identity)
            if op:
                terminal = op.status in TERMINAL
                failed = failed_snapshot_expired(op, now)
                if not terminal and not failed:
                    return False
                # A terminal operation from a rolling upgrade may still have
                # an old executor without the snapshot lock.
                if (terminal and not op.payload.get('snapshot_lock_v1')
                        and op.lease_until and operation_time(op.lease_until) > now):
                    return False
                if failed:
                    # Keep the journal retryable after its cached bytes expire.
                    op.payload = {**op.payload, 'snapshot_reclaimed': True}
                    db.commit()
            else:
                try:
                    orphan_age = time.time() - directory.stat().st_mtime
                except OSError:
                    return False
                if not DATASET_FAILED_RETENTION_SECONDS or orphan_age < DATASET_FAILED_RETENTION_SECONDS:
                    return False
            quarantine = STORE_ROOT / ('.reclaimed-' + uid())
            directory.rename(quarantine)
    try:
        shutil.rmtree(quarantine)
    except FileNotFoundError:
        pass  # Another cleanup pass already finished this detached directory.
    except OSError:
        logger.warning('dataset_snapshot_cleanup_deferred id=%s', identity)
    return True


def reclaim_terminal_now(db, identity):
    """Best-effort fast path; the periodic scan retries interrupted cleanup."""
    try:
        reclaim_snapshot(db, identity)
    except Exception:
        logger.exception('dataset_terminal_snapshot_cleanup_deferred id=%s', identity)


class DatasetWorker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None
        self.refresh_thread = None
        self.last_tick_at = None
        self.refresh_last_tick_at = None
        self.last_error = None
        self.last_error_at = None
        self.active_error = None
        self.refresh_error = None
        self.refresh_error_at = None
        self.refresh_active_error = None
        self._last_cleanup = 0.0
        self._last_terminal_cleanup = 0.0

    def health_state(self):
        alive = bool(self.thread and self.thread.is_alive())
        refresh_alive = bool(self.refresh_thread and self.refresh_thread.is_alive())
        return {'status': 'error' if alive and (self.active_error or self.refresh_active_error or not refresh_alive)
                else 'running' if alive else 'stopped',
                'last_tick_at': self.last_tick_at.isoformat() if self.last_tick_at else None,
                'last_error': self.last_error,
                'last_error_at': self.last_error_at.isoformat() if self.last_error_at else None,
                'active_error': self.active_error,
                'refresh_last_tick_at': self.refresh_last_tick_at.isoformat() if self.refresh_last_tick_at else None,
                'refresh_error': self.refresh_error,
                'refresh_error_at': self.refresh_error_at.isoformat() if self.refresh_error_at else None,
                'refresh_active_error': self.refresh_active_error}

    def start(self):
        self.stop_event.clear()
        if not self.thread or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True, name='dataset-storage')
            self.thread.start()
        if not self.refresh_thread or not self.refresh_thread.is_alive():
            self.refresh_thread = threading.Thread(target=self.run_refresh, daemon=True, name='dataset-refresh')
            self.refresh_thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.refresh_thread:
            self.refresh_thread.join(timeout=5)

    def tick(self):
        STORE_ROOT.mkdir(parents=True, exist_ok=True)
        with (STORE_ROOT / '.worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            with SessionLocal() as db:
                candidates = db.scalars(select(DatasetOperation).where(
                    DatasetOperation.node_id == 'local', DatasetOperation.status.in_(ACTIVE))
                    .order_by(DatasetOperation.created_at, DatasetOperation.id))
                op = next((item for item in candidates if not refresh_lane_operation(item)
                           and not local_mutation_blocked_by_refresh(db, item)), None)
                if op:
                    try:
                        with snapshot_access(op.id):
                            execute_local(db, op)
                    except Exception as exc:
                        mark_error(db, op, exc)
                        logger.warning('dataset_operation_failed id=%s error=%s', op.id, type(exc).__name__)
                    reclaim_terminal_now(db, op.id)
                self.cleanup(db)
                self.last_tick_at = utcnow()

    def tick_refresh(self):
        STORE_ROOT.mkdir(parents=True, exist_ok=True)
        with (STORE_ROOT / '.refresh.worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            with SessionLocal() as db:
                candidates = db.scalars(select(DatasetOperation).where(
                    DatasetOperation.kind == 'refresh', DatasetOperation.node_id == 'local',
                    DatasetOperation.status.in_(ACTIVE))
                    .order_by(DatasetOperation.created_at, DatasetOperation.id))
                op = next((item for item in candidates if refresh_lane_operation(item)
                           and (not local_refresh(item) or not local_refresh_blocked_by_older_mutation(db, item))), None)
                if op:
                    try:
                        with snapshot_access(op.id):
                            execute_local(db, op)
                    except Exception as exc:
                        mark_error(db, op, exc)
                        logger.warning('dataset_refresh_failed id=%s error=%s', op.id, type(exc).__name__)
                    reclaim_terminal_now(db, op.id)
                self.cleanup(db)
                self.refresh_last_tick_at = utcnow()

    def cleanup(self, db):
        # Filesystem directories bound the scan, not the ever-growing journal.
        clock = time.monotonic()
        full_cleanup = clock - self._last_cleanup >= 60
        if (self.thread and self.thread.is_alive() and not full_cleanup
                and clock - self._last_terminal_cleanup < 5):
            return
        self._last_terminal_cleanup = clock
        if full_cleanup:
            self._last_cleanup = clock
        now = utcnow()
        terminal_cutoff = now - timedelta(seconds=DATASET_STORAGE_RETENTION_SECONDS)
        if not STORE_ROOT.is_dir():
            return
        for directory in STORE_ROOT.iterdir():
            if directory.name.startswith('.reclaimed-') and directory.is_dir() and not directory.is_symlink():
                suffix = directory.name.removeprefix('.reclaimed-')
                if len(suffix) == 32 and all(c in '0123456789abcdef' for c in suffix):
                    if full_cleanup:
                        try:
                            shutil.rmtree(directory)
                        except FileNotFoundError:
                            pass
                        except OSError:
                            logger.warning('dataset_snapshot_quarantine_cleanup_deferred path=%s', directory.name)
                continue
            if directory.is_symlink() or not directory.is_dir() or len(directory.name) != 32:
                continue
            try:
                identity = directory.name
                if job_dir(identity) != directory:
                    continue
            except ValueError:
                continue
            if full_cleanup:
                reclaim_snapshot(db, identity, now=now)
            else:
                # The frequent pass only targets terminal operations. Keep
                # failed/orphan retention scans at the original minute cadence.
                db.expire_all()
                op = db.get(DatasetOperation, identity)
                terminal = bool(op and op.status in TERMINAL)
                if terminal:
                    reclaim_terminal_now(db, identity)
        if full_cleanup or not self.thread or not self.thread.is_alive():
            self.cleanup_upload_staging(db, terminal_cutoff)

    def cleanup_upload_staging(self, db, terminal_cutoff):
        """Retry owned terminal uploads independently of removed bundle caches."""
        from .dataset_uploads import UPLOAD_ROOT, upload_session
        if not DATASET_STORAGE_RETENTION_SECONDS or not UPLOAD_ROOT.is_dir() or UPLOAD_ROOT.is_symlink():
            return
        for directory in UPLOAD_ROOT.iterdir():
            reclaimed = directory.name.startswith('.reclaimed-upload-')
            identity = directory.name.removeprefix('.reclaimed-upload-') if reclaimed else directory.name
            if directory.is_symlink() or not directory.is_dir() or len(identity) != 32 or any(
                    char not in '0123456789abcdef' for char in identity):
                continue
            try:
                with managed_dataset_mutation():
                    db.expire_all()
                    op = db.get(DatasetOperation, identity)
                    if (not op or op.kind != 'upload' or op.status not in TERMINAL
                            or operation_time(op.updated_at) > terminal_cutoff
                            or Path(op.payload.get('staging', '')) != UPLOAD_ROOT / identity / 'files'):
                        continue
                if reclaimed:
                    shutil.rmtree(directory)
                    continue
                # Completion acquires these locks in the same order. Unknown
                # or pre-journal sessions remain under their existing TTL rules.
                with upload_session(identity, touch=False) as (_, manifest):
                    if manifest.get('operation_id') != identity:
                        continue
                    with managed_dataset_mutation():
                        db.expire_all()
                        op = db.get(DatasetOperation, identity)
                        if not op or op.status not in TERMINAL or operation_time(op.updated_at) > terminal_cutoff:
                            continue
                        quarantine = UPLOAD_ROOT / ('.reclaimed-upload-' + identity)
                        if quarantine.exists() or quarantine.is_symlink():
                            continue
                        directory.rename(quarantine)
                shutil.rmtree(quarantine)
            except (OSError, HTTPException, ValueError, TypeError):
                logger.warning('dataset_upload_staging_cleanup_deferred id=%s', identity)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
                self.active_error = None
            except Exception as exc:
                self.last_error = self.active_error = str(exc)[:600] or type(exc).__name__
                self.last_error_at = utcnow()
                logger.exception('dataset_worker_failed')
            self.stop_event.wait(1)

    def run_refresh(self):
        while not self.stop_event.is_set():
            try:
                self.tick_refresh()
                self.refresh_active_error = None
            except Exception as exc:
                self.refresh_error = self.refresh_active_error = str(exc)[:600] or type(exc).__name__
                self.refresh_error_at = utcnow()
                logger.exception('dataset_refresh_worker_failed')
            self.stop_event.wait(1)


worker = DatasetWorker()


def auth(node_id, request, db):
    from .compute_nodes import authenticated_agent
    return authenticated_agent(node_id, request, db)


class DatasetReceipt(BaseModel):
    id: str = Field(pattern=r'^[0-9a-f]{32}$')
    token: str = Field(pattern=r'^[0-9a-f]{32}$')


class ReconcileRequest(BaseModel):
    receipts: list[DatasetReceipt] = Field(default_factory=list, max_length=128)


@router.post('/agent/{node_id}/reconcile')
def reconcile_agent_cache(node_id: str, body: ReconcileRequest, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    reclaimable = []
    with managed_dataset_mutation():
        for receipt in body.receipts:
            op = db.get(DatasetOperation, receipt.id)
            if op and node_id not in (op.node_id, op.payload.get('source_node'), op.payload.get('target_node')):
                continue
            # Every phase owns a new token. Old phase receipts are no longer
            # needed by centre replay, including source export after migration.
            if (not op or op.lease != receipt.token or op.status in TERMINAL
                    or failed_snapshot_expired(op)):
                reclaimable.append({'id': receipt.id, 'token': receipt.token})
    return {'reclaimable': reclaimable}


@router.post('/agent/{node_id}/poll')
def agent_poll(node_id: str, request: Request, lightweight_only: bool = False, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    with managed_dataset_mutation():
        acquire_node_claim_lock(db, node_id)
        operations = list(db.scalars(select(DatasetOperation).where(
            DatasetOperation.node_id == node_id, DatasetOperation.status.in_(ACTIVE))
            .order_by(DatasetOperation.created_at, DatasetOperation.id)))
        # Never bypass a live executor, including one running a heavier phase.
        if any(item.lease_until and operation_time(item.lease_until) > utcnow() for item in operations):
            return {'operation': None}
        node = db.get(ComputeNode, node_id)
        capability = ((node.snapshot or {}).get('training') or {}) if node else {}
        busy = training_active(db, node_id)
        concurrent = concurrent_dataset_reads(db, node_id)
        refresh = concurrent_dataset_refresh(db, node_id)
        reads_only = lightweight_only or ((concurrent or refresh) and (busy or capability.get('lock_occupied')))
        op = next((item for item in operations if not reads_only or item.payload.get('phase') == 'check'
                   or training_safe_operation(item, concurrent_reads=concurrent, concurrent_refresh=refresh)), None)
        if not op:
            return {'operation': None}
        if op.payload['phase'] == 'discover':
            from .dataset_discovery import supports_discovery
            if not supports_discovery(db, node_id):
                return {'operation': None, 'message': '目录扫描需要 Agent 2.2.0 及 dataset_discovery 能力'}
        # An expired lease reuses its token, so a saved commit receipt is replayed.
        if busy and not (op.payload.get('phase') == 'check' or training_safe_operation(
                op, concurrent_reads=concurrent, concurrent_refresh=refresh)):
            return {'operation': None}
        if op.payload['phase'] == 'remove':
            try:
                guard_remove(db, op)
            except ValueError as exc:
                mark_error(db, op, exc)
                return {'operation': None}
        op.payload = {**op.payload, 'progress_run_id': uid(),
                      'snapshot_lock_v1': True,
                      'started_at': op.payload.get('started_at') or utcnow().isoformat(),
                      'heartbeat_at': utcnow().isoformat()}
        op.lease_until, op.status = utcnow() + timedelta(seconds=120), 'running' if op.status != 'committing' else 'committing'
        db.commit()
        return {'operation': action_payload(op, concurrent_refresh=refresh)}


class AgentMessage(BaseModel):
    token: str
    phase: str
    committing: bool = False
    error: str | None = Field(default=None, max_length=1200)
    result: dict = Field(default_factory=dict)
    progress: dict | None = None
    progress_run_id: str | None = Field(default=None, max_length=32)


@router.post('/agent/{node_id}/{identity}/heartbeat')
def agent_heartbeat(node_id: str, identity: str, body: AgentMessage, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    with managed_dataset_mutation():
        op = db.get(DatasetOperation, identity)
        if not op or op.node_id != node_id:
            raise HTTPException(404, '操作不存在')
        lease_check(db, identity, body.token, body.phase, body.committing)
        if body.progress is not None:
            try:
                if record_work_progress(op, body.progress, body.progress_run_id, node_id):
                    db.commit()
            except (ValueError, TypeError) as exc:
                raise HTTPException(422, str(exc)) from exc
        if body.phase == 'discover' and body.result:
            from .dataset_discovery import record_progress
            try:
                record_progress(db, op, body.result)
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(422, str(exc)) from exc
            db.commit()
    return {'ok': True}


@router.post('/agent/{node_id}/{identity}/finish')
def agent_finish(node_id: str, identity: str, body: AgentMessage, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    try:
        with snapshot_access(identity):
            return finish_agent_operation(node_id, identity, body, request, db)
    finally:
        reclaim_terminal_now(db, identity)


def finish_agent_operation(node_id, identity, body, request, db):
    # Slow archive hashing stays outside the global mutation lock. The Agent
    # has stopped writing before sending this receipt; recheck its phase before
    # the short durable transition to reject concurrent cancellation.
    with managed_dataset_mutation():
        op = db.get(DatasetOperation, identity)
        if not op:
            raise HTTPException(404, '操作不存在')
        if op.status == 'completed' or op.payload.get('phase') != body.phase:
            if node_id not in (op.node_id, op.payload.get('source_node'), op.payload.get('target_node')):
                raise HTTPException(404, '操作不存在')
            return {'ok': True}
        if op.node_id != node_id:
            raise HTTPException(404, '操作不存在')
        op = lease_check(db, identity, body.token, body.phase)
        if body.error:
            mark_error(db, op, ValueError(body.error))
            return {'ok': True}
    try:
        if body.phase == 'export':
            manifest = body.result
            directory = job_dir(identity)
            archive = directory / 'bundle.part'
            if not archive.exists():
                archive = directory / 'bundle.tar'
            def check():
                nonlocal op
                if not getattr(check, 'last', 0) or time.monotonic() - check.last > 4:
                    with managed_dataset_mutation():
                        op = lease_check(db, identity, body.token, body.phase)
                    check.last = time.monotonic()
            def persist(value):
                with managed_dataset_mutation():
                    current = lease_check(db, identity, body.token, body.phase)
                    if record_work_progress(current, value, current.payload.get('progress_run_id'), 'local'):
                        db.commit()
            check.progress = progress_reporter(persist, start_sequence=(op.payload.get('progress') or {}).get('sequence') or 0)
            before = archive_identity(archive) if archive.is_file() else None
            if (not before or archive.stat().st_size != manifest['archive_size']
                    or files.digest(archive, check, stage='archive_hash') != manifest['archive_sha256']
                    or before != archive_identity(archive)):
                raise ValueError('远程数据包大小或 SHA-256 不匹配')
            with managed_dataset_mutation():
                op = lease_check(db, identity, body.token, body.phase)
                if before != archive_identity(archive):
                    raise ValueError('远程数据包在校验期间发生修改')
                save_json(directory / 'manifest.json', manifest)
                if archive.name != 'bundle.tar':
                    archive.replace(directory / 'bundle.tar')
                op.payload = {**op.payload, 'verified_archive': {
                    'sha256': manifest['archive_sha256'], 'file': archive_identity(directory / 'bundle.tar')}}
                phase_success(db, op, body.result)
        else:
            with managed_dataset_mutation():
                op = lease_check(db, identity, body.token, body.phase)
                phase_success(db, op, body.result)
    except Exception as exc:
        with managed_dataset_mutation():
            mark_error(db, op, exc)
        raise HTTPException(409, str(getattr(exc, 'detail', None) or str(exc))) from exc
    return {'ok': True}


def authorized_transfer(db, node_id, identity, token, request):
    auth(node_id, request, db)
    op = db.get(DatasetOperation, identity)
    if not op or op.node_id != node_id:
        raise HTTPException(404, '操作不存在')
    return lease_check(db, identity, token)


@router.get('/agent/{node_id}/{identity}/chunk')
def download_chunk(node_id: str, identity: str, token: str, offset: int, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    with snapshot_access(identity):
        op = authorized_transfer(db, node_id, identity, token, request)
        if op.payload['phase'] != 'import' or offset < 0:
            raise HTTPException(409, '传输阶段不匹配')
        with (job_dir(identity) / 'bundle.tar').open('rb') as handle:
            handle.seek(offset)
            data = handle.read(512 * 1024)
        op.transferred = max(op.transferred, offset + len(data))
        db.commit()
        return {'data': base64.b64encode(data).decode(), 'offset': offset}


class Chunk(BaseModel):
    token: str
    offset: int = Field(ge=0)
    data: str = Field(max_length=710000)


@router.post('/agent/{node_id}/{identity}/chunk')
def upload_chunk(node_id: str, identity: str, body: Chunk, request: Request, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        op = authorized_transfer(db, node_id, identity, body.token, request)
        if op.payload['phase'] != 'export':
            raise HTTPException(409, '传输阶段不匹配')
        directory = job_dir(identity)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / 'bundle.part'
        current = target.stat().st_size if target.exists() else 0
        try:
            data = base64.b64decode(body.data, validate=True)
        except ValueError as exc:
            raise HTTPException(422, '分块编码无效') from exc
        if body.offset != current:
            return {'offset': current}
        if len(data) > 512 * 1024 or current + len(data) > files.MAX_BYTES + files.MAX_FILES * 2048:
            raise HTTPException(413, '数据包超出限制')
        if data:
            if shutil.disk_usage(directory).free < len(data) + 100 * 1024**2 or (DATASET_STORAGE_QUOTA_BYTES and sum(p.stat().st_size for p in STORE_ROOT.glob('*/bundle*')) + len(data) > DATASET_STORAGE_QUOTA_BYTES):
                raise HTTPException(507, '平台中转磁盘或配额不足')
            with target.open('ab') as handle:
                handle.write(data)
            current += len(data)
        op.transferred = current
        db.commit()
        return {'offset': current}
