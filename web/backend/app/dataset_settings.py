"""Dataset settings and bounded, cached accounting of disposable centre files.

An HTTP read never walks dataset roots. Only the disposable transfer/upload
trees are sampled, on one background thread, with a shared time/entry budget.
"""
from __future__ import annotations

import copy
import os
import shutil
import stat
import threading
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from . import dataset_policy
from .database import get_db
from .models import ComputeNode, DatasetEntry, DatasetOperation, utcnow

router = APIRouter()
SAMPLE_SECONDS = 0.4
SAMPLE_ENTRIES = 20000
SAMPLE_OPERATIONS = 50
SAMPLE_INTERVAL = 30


def store_module():
    from . import dataset_store
    return dataset_store


def valid_id(value):
    return len(value) == 32 and all(char in '0123456789abcdef' for char in value)


class ScanBudget:
    def __init__(self):
        self.deadline = time.monotonic() + SAMPLE_SECONDS
        self.entries = 0
        self.limited = False
        self.errors = []

    def step(self):
        if self.entries >= SAMPLE_ENTRIES or time.monotonic() >= self.deadline:
            self.limited = True
            return False
        self.entries += 1
        return True


def tree_bytes(path, budget, depth=0, *, parent_fd=None):
    """Count file metadata without following links or opening image contents."""
    total = bundle = 0
    complete = True
    if depth > 32:
        budget.limited = True
        return total, bundle, False
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
                             dir_fd=parent_fd)
    except FileNotFoundError:
        return 0, 0, True
    except OSError as exc:
        budget.errors.append(type(exc).__name__)
        return 0, 0, False
    try:
        with os.scandir(descriptor) as iterator:
            for item in iterator:
                if not budget.step():
                    complete = False
                    break
                try:
                    metadata = item.stat(follow_symlinks=False)
                    if stat.S_ISREG(metadata.st_mode):
                        total += metadata.st_size
                        if depth == 0 and item.name.startswith('bundle'):
                            bundle += metadata.st_size
                    elif stat.S_ISDIR(metadata.st_mode):
                        size, _, child_complete = tree_bytes(item.name, budget, depth + 1, parent_fd=descriptor)
                        total += size
                        complete = complete and child_complete
                    elif stat.S_ISLNK(metadata.st_mode):
                        complete = False
                except FileNotFoundError:
                    complete = False  # A live upload/cleanup changed during this sample.
                except OSError as exc:
                    budget.errors.append(type(exc).__name__)
                    complete = False
                if budget.limited:
                    break
    finally:
        os.close(descriptor)
    return total, bundle, complete and not budget.limited


def sample_roots(transfer, uploads):
    budget = ScanBudget()
    records = {}
    totals = {'transfer_bytes': 0, 'upload_staging_bytes': 0, 'quota_used_bytes': 0}
    complete = True
    for root, field in ((transfer, 'transfer_bytes'), (uploads, 'upload_staging_bytes')):
        try:
            descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            continue
        except OSError as exc:
            budget.errors.append(type(exc).__name__)
            complete = False
            continue
        try:
            with os.scandir(descriptor) as iterator:
                for item in iterator:
                    if not budget.step():
                        complete = False
                        break
                    if item.is_symlink():
                        complete = False
                        continue
                    if not item.is_dir(follow_symlinks=False):
                        continue
                    identity = item.name
                    identity = identity.removeprefix('.reclaimed-upload-').removeprefix('.reclaimed-')
                    known_identity = valid_id(identity)
                    if not known_identity and field != 'transfer_bytes':
                        continue
                    size, bundle, found_complete = tree_bytes(item.name, budget, parent_fd=descriptor)
                    totals[field] += size
                    # Path.glob('*/bundle*') also includes reclamation quarantine.
                    if field == 'transfer_bytes':
                        totals['quota_used_bytes'] += bundle
                    if not known_identity:
                        # Count unexpected transfer subdirectories in the same
                        # quota as the executor without inventing an operation.
                        complete = complete and found_complete
                        if budget.limited:
                            break
                        continue
                    row = records.setdefault(identity, {'id': identity, 'bundle_bytes': 0, 'transfer_bytes': 0,
                        'upload_staging_bytes': 0, 'total_bytes': 0, 'sampled_complete': True})
                    row['total_bytes'] += size
                    row['bundle_bytes'] += bundle if field == 'transfer_bytes' else 0
                    row['transfer_bytes'] += size if field == 'transfer_bytes' else 0
                    row['upload_staging_bytes'] += size if field == 'upload_staging_bytes' else 0
                    row['sampled_complete'] = row['sampled_complete'] and found_complete
                    complete = complete and found_complete
                    if budget.limited:
                        break
        finally:
            os.close(descriptor)
        if budget.limited:
            break
    rows = sorted(records.values(), key=lambda row: (-row['total_bytes'], row['id']))
    return {**totals, 'total_bytes': totals['transfer_bytes'] + totals['upload_staging_bytes'],
            'status': 'complete' if complete and not budget.errors else 'partial',
            'sampled_at': utcnow().isoformat(), 'entries_scanned': budget.entries,
            'limit_reached': budget.limited, 'error': ', '.join(sorted(set(budget.errors))) or None,
            'operations': rows[:SAMPLE_OPERATIONS], 'displayed_operation_count': min(len(rows), SAMPLE_OPERATIONS),
            'truncated_operation_count': max(0, len(rows) - SAMPLE_OPERATIONS)}


class StorageSampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.cache = {}

    def get(self, transfer, uploads, *, refresh=False):
        key = (str(transfer), str(uploads))
        with self.lock:
            cached = self.cache.get(key)
            running = bool(self.thread and self.thread.is_alive())
            if not running and (refresh or not cached or time.monotonic() - cached[0] >= SAMPLE_INTERVAL):
                self.thread = threading.Thread(target=self.run, args=(key, transfer, uploads),
                                               name='dataset-space-sample', daemon=True)
                self.thread.start()
                running = True
            value = copy.deepcopy(cached[1]) if cached else {
                'status': 'pending', 'sampled_at': None, 'transfer_bytes': None,
                'upload_staging_bytes': None, 'total_bytes': None, 'quota_used_bytes': None,
                'entries_scanned': 0, 'limit_reached': False, 'error': None, 'operations': [],
                'displayed_operation_count': 0, 'truncated_operation_count': 0}
            return {**value, 'sampling': running}

    def run(self, key, transfer, uploads):
        try:
            result = sample_roots(transfer, uploads)
        except Exception as exc:
            result = {'status': 'error', 'sampled_at': utcnow().isoformat(), 'transfer_bytes': None,
                      'upload_staging_bytes': None, 'total_bytes': None, 'quota_used_bytes': None,
                      'entries_scanned': 0, 'limit_reached': False, 'error': type(exc).__name__,
                      'operations': [], 'displayed_operation_count': 0, 'truncated_operation_count': 0}
        with self.lock:
            self.cache[key] = (time.monotonic(), result)
            # Bound retained state even when tests or instances change roots.
            for old in list(self.cache)[:-8]:
                del self.cache[old]


sampler = StorageSampler()


def cleanup_state(op, row, store):
    if op is None:
        return 'orphan', '未关联到操作记录；由现有上传会话或孤立快照保留规则处理'
    if op.status in store.ACTIVE:
        return 'active', '操作正在执行或排队，临时文件仍被使用'
    if op.status == 'failed':
        if row['upload_staging_bytes']:
            return 'retained', '失败上传暂存供重试保留；取消上传后释放，快照另按失败保留期回收'
        if store.failed_snapshot_expired(op):
            return 'pending', '失败快照已到保留期，等待清理器取得快照锁后回收'
        return 'retained', ('发布回执或提交状态需保留，不能自动回收' if op.payload.get('published') or op.payload.get('commit_started')
                            else '失败快照供重试保留；保留期为 0 时不自动回收')
    if op.status in store.TERMINAL:
        if (not op.payload.get('snapshot_lock_v1') and op.lease_until
                and store.operation_time(op.lease_until) > utcnow()):
            return 'reserved', '旧执行器租约仍有效，等待执行器释放后清理'
        if row['upload_staging_bytes']:
            if not store.DATASET_STORAGE_RETENTION_SECONDS:
                return 'retained', '上传暂存自动清理已由环境配置关闭；完成回执仍保留'
            age = (utcnow() - store.operation_time(op.updated_at)).total_seconds()
            if age < store.DATASET_STORAGE_RETENTION_SECONDS:
                return 'reserved', '上传暂存和完成回执处于兼容续传保留期；快照在读者释放后清理'
        return 'pending', '检测到终态临时文件，等待清理器重试；快照锁或文件权限可能使回收延后'
    return 'retained', '当前状态不符合自动回收条件'


def catalogue_sizes(db):
    byte_value = DatasetEntry.details['total_bytes'].as_integer()
    rows = db.execute(select(DatasetEntry.node_id, func.count(), func.sum(byte_value),
                             func.count(byte_value)).where(DatasetEntry.status != 'deleted')
                      .group_by(DatasetEntry.node_id)).all()
    result = {'local': {'bytes': 0, 'count': 0, 'unknown_count': 0},
              'remote': {'bytes': 0, 'count': 0, 'unknown_count': 0},
              'measured_at': utcnow().isoformat(), 'source': 'catalogue', 'nodes': []}
    for node, count, size, known in rows:
        bucket = result['local' if node == 'local' else 'remote']
        bucket['bytes'] += int(size or 0)
        bucket['count'] += count
        bucket['unknown_count'] += count - known
        result['nodes'].append({'node_id': node, 'bytes': int(size or 0), 'count': count, 'unknown_count': count - known})
    return result


def operation_metadata(op, db, store):
    """Use only journal/capability information; avoid source-path resolution.

    The full management-page serializer resolves paths to explain conflicting
    filesystem mutations. Settings only report persisted progress, active counts
    and node state so opening this page cannot probe a training dataset mount.
    """
    value = store.operation_read(op)
    node = db.get(ComputeNode, op.node_id) if op.node_id != 'local' else None
    value['node_name'] = node.name if node else ('本机' if op.node_id == 'local' else op.node_id)
    # The two local execution lanes and path conflicts determine exact queue
    # position. Do not invent a serial count here or inspect original paths.
    value['queue_ahead'] = 0
    if node:
        capability = (node.snapshot or {}).get('training') or {}
        seen = store.operation_time(node.last_seen_at)
        value['waiting_offline'] = not seen or (utcnow() - seen).total_seconds() > max(30, node.report_interval_seconds * 3)
        safe = store.training_safe_operation(op,
            concurrent_reads=capability.get('dataset_concurrent_reads') is True,
            concurrent_refresh=capability.get('dataset_concurrent_reads') is True and capability.get('dataset_concurrent_refresh') is True,
            concurrent_delete=capability.get('dataset_concurrent_delete') is True)
        value['waiting_training'] = op.status == 'pending' and not safe and store.training_active(db, op.node_id)
        value['waiting_lock'] = (value['waiting_lock'] or
            (op.status == 'pending' and not safe and not value['waiting_training'] and bool(capability.get('lock_occupied'))))
        if op.status == 'pending' and safe:
            value['waiting_lock'] = False
    elif op.node_id != 'local':
        value['waiting_offline'] = True
    else:
        value['waiting_training'] = False
        progress = op.payload.get('progress') or {}
        value['waiting_resource'] = progress.get('stage') == 'paused_resource'
    return value


def settings_response(db, *, refresh=False):
    store = store_module()
    from .dataset_uploads import UPLOAD_ROOT
    state = dataset_policy.read(store.STORE_ROOT, store.dataset_policy_defaults())
    storage = sampler.get(store.STORE_ROOT, UPLOAD_ROOT, refresh=refresh)
    for row in storage['operations']:
        op = db.get(DatasetOperation, row['id'])
        row.update(name=op.payload.get('name', '') if op else '', kind=op.kind if op else 'unknown',
                   status=op.status if op else 'unknown')
        row['cleanup_state'], row['cleanup_reason'] = cleanup_state(op, row, store)
    used = storage['quota_used_bytes']
    quota = state['policy']['storage_quota_bytes']
    storage['quota_remaining_bytes'] = max(0, quota - used) if quota and used is not None and storage['status'] == 'complete' else None
    storage['quota_basis'] = 'transfer_bundle_files'
    try:
        disk = shutil.disk_usage(store.STORE_ROOT if store.STORE_ROOT.exists() else store.STORE_ROOT.parent)
        storage['disk'] = {'free_bytes': disk.free, 'total_bytes': disk.total}
    except OSError:
        storage['disk'] = {'free_bytes': None, 'total_bytes': None}
    storage['registered_bytes'] = catalogue_sizes(db)
    counts = dict(db.execute(select(DatasetOperation.status, func.count()).group_by(DatasetOperation.status)).all())
    operations = list(db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_(store.ACTIVE))
                                .order_by(DatasetOperation.created_at, DatasetOperation.id).limit(30)))
    remotes = []
    for node in db.scalars(select(ComputeNode).where(ComputeNode.enabled.is_(True), ComputeNode.kind == 'agent').order_by(ComputeNode.name)):
        capability = (node.snapshot or {}).get('training') or {}
        last_seen = store.operation_time(node.last_seen_at)
        timeout = max(30, node.report_interval_seconds * 3)
        remotes.append({'id': node.id, 'name': node.name, 'online': bool(last_seen and (utcnow() - last_seen).total_seconds() <= timeout),
                        'last_seen_at': last_seen.isoformat() if last_seen else None, 'agent_version': node.agent_version,
                        'dataset_roots': capability.get('dataset_roots') or [],
                        'concurrent_reads': capability.get('dataset_concurrent_reads'),
                        'concurrent_refresh': capability.get('dataset_concurrent_refresh')})
    return {**state, 'defaults': store.dataset_policy_defaults(), 'scope': 'platform_local',
            'paths': {'transfer': str(store.STORE_ROOT), 'upload_staging': str(UPLOAD_ROOT)},
            'cleanup': {'terminal_snapshots': 'after_release',
                        'upload_staging_retention_seconds': store.DATASET_STORAGE_RETENTION_SECONDS,
                        'failed_retention_seconds': state['policy']['failed_retention_seconds']},
            'storage': storage, 'worker': store.worker.health_state(),
            'operations': {'active': [operation_metadata(op, db, store) for op in operations],
                           'active_count': sum(counts.get(status, 0) for status in store.ACTIVE),
                           'failed_count': counts.get('failed', 0)}, 'remote_nodes': remotes}


class PolicyPatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: StrictInt = Field(ge=0, le=dataset_policy.MAX_SAFE_INTEGER)
    policy: dict[str, StrictInt]


@router.get('/settings')
def get_settings(refresh: bool = Query(False), db: Session = Depends(get_db)):
    return settings_response(db, refresh=refresh)


@router.patch('/settings')
def patch_settings(body: PolicyPatch, db: Session = Depends(get_db)):
    store = store_module()
    try:
        dataset_policy.update(store.STORE_ROOT, store.dataset_policy_defaults(), body.revision, body.policy)
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(507, '数据集设置保存失败，请检查平台数据目录权限和磁盘空间') from exc
    return settings_response(db)
