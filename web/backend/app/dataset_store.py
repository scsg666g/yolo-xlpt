"""Durable dataset catalogue and verified, node-neutral file operations.

Only a verified snapshot is published. Transfers pass through the centre; agents
need no inbound port. A phase token is stable across retries for crash receipts.
"""
from __future__ import annotations

import base64
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
from datetime import timedelta, timezone

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from . import dataset_files as files
from .config import DATA_ROOT, DATASET_ROOTS, DATASET_STORAGE_ALLOWED_ROOTS, DATASET_STORAGE_QUOTA_BYTES, PROJECT_ROOT
from .database import SessionLocal, get_db
from .models import ComputeNode, DatasetEntry, DatasetLocation, DatasetOperation, TrainingRun, TrainingTask, utcnow
from .gpu_reservations import acquire_node_claim_lock
from .dataset_uploads import managed_dataset_mutation, dataset_references, managed_marker, validate_name

router = APIRouter(prefix='/api/dataset-storage', tags=['dataset storage'])
STORE_ROOT = DATA_ROOT / 'dataset_storage'
ACTIVE = ('pending', 'running', 'committing')
TERMINAL = ('completed', 'cancelled')
logger = logging.getLogger(__name__)


def uid():
    return uuid.uuid4().hex


def job_dir(identity):
    if len(identity) != 32 or any(c not in '0123456789abcdef' for c in identity):
        raise ValueError('操作编号无效')
    return STORE_ROOT / identity


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


def operation_read(op, db=None):
    value = {'id': op.id, 'kind': op.kind, 'status': op.status, 'phase': op.payload.get('phase'),
             'node_id': op.node_id, 'name': op.payload.get('name', ''), 'error': op.error,
             'transferred': op.transferred, 'total_bytes': op.payload.get('archive_size', 0),
             'created_at': op.created_at, 'updated_at': op.updated_at, 'result': op.result,
             'source_path': op.payload.get('source_path'), 'destination': op.payload.get('destination'),
             'can_cancel': op.status in ('pending', 'running', 'failed') and not op.payload.get('published') and not op.payload.get('commit_started')}
    if db and op.node_id != 'local' and op.status in ACTIVE:
        node = db.get(ComputeNode, op.node_id)
        seen = node.last_seen_at if node else None
        value['waiting_offline'] = not seen or (utcnow() - seen.replace(tzinfo=timezone.utc)).total_seconds() > 90
        value['waiting_training'] = bool(db.scalar(select(TrainingRun.id).where(TrainingRun.execution_node_id == op.node_id, TrainingRun.status.in_(('starting', 'running', 'stopping')))))
    return value


def reference_rows(db, entry):
    if entry.node_id == 'local':
        tasks = dataset_references(db, str(Path(entry.path) / 'data.yaml'))
    else:
        # We cannot open an unregistered remote YAML at the centre. Treat those
        # paths conservatively as potential aliases instead of guessing no refs.
        tasks = []
        candidates = list(db.scalars(select(TrainingTask).where(TrainingTask.execution_node_id == entry.node_id)))
        for task in candidates:
            paths = [task.data_path, *db.scalars(select(TrainingRun.data_path).where(TrainingRun.task_id == task.id))]
            for raw in paths:
                if not raw:
                    continue
                p, root = PurePosixPath(raw), PurePosixPath(entry.path)
                known = db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == entry.node_id,
                                                            DatasetEntry.path == str(p.parent)))
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


def entry_read(entry, db):
    d = entry.details or {}
    return {'id': entry.id, 'location_id': entry.location_id, 'node_id': entry.node_id,
            'name': entry.name, 'node_name': (db.get(ComputeNode, entry.node_id).name if db.get(ComputeNode, entry.node_id) else ('本机' if entry.node_id == 'local' else entry.node_id)), 'directory': entry.path, 'config_path': str(PurePosixPath(entry.path) / 'data.yaml'),
            'ownership': entry.ownership, 'status': entry.status, 'created_at': entry.created_at,
            'total_bytes': d.get('total_bytes', 0), 'stats': d.get('stats', {}), 'classes': d.get('names', []),
            'warnings': d.get('warnings', []), 'fingerprint': d.get('fingerprint'),
            'history': d.get('history', []), 'references': reference_rows(db, entry)}


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
    return next((op for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed'))))
                 if op.payload.get('entry_id') == identity), None)


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
    return any(op.payload.get('phase') in ('export', 'import', 'remove') for op in db.scalars(
        select(DatasetOperation).where(DatasetOperation.node_id == node_id, DatasetOperation.status.in_(ACTIVE))))


def reject_training_mutation(db, node_id):
    if mutation_pending(db, node_id or 'local'):
        raise HTTPException(409, '该服务器正在迁移或删除数据集，请等待操作结束后再调整或启动训练')


def new_operation(db, kind, node_id, payload):
    op = DatasetOperation(id=uid(), kind=kind, node_id=node_id, status='pending',
                          payload=payload, result={}, lease=uid())
    db.add(op)
    db.flush()
    return op


def set_phase(op, phase, node_id):
    op.payload = {**op.payload, 'phase': phase, 'commit_started': False}
    op.node_id, op.status, op.lease, op.lease_until = node_id, 'pending', uid(), None
    op.transferred = 0


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
    name: str
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
        if loc.is_default or db.scalar(select(DatasetEntry.id).where(DatasetEntry.location_id == identity)) or any(
                op.payload.get('location_id') == identity for op in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_((*ACTIVE, 'failed'))))):
            raise HTTPException(409, '请先取消默认设置，并处理该位置的数据集和操作记录')
        db.delete(loc)
        db.commit()


@router.get('/datasets')
def list_entries(db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        bootstrap(db)
    return [entry_read(e, db) for e in db.scalars(select(DatasetEntry).where(DatasetEntry.status != 'deleted').order_by(DatasetEntry.created_at.desc()))]


@router.post('/datasets/register', status_code=202)
def register(body: Register, db: Session = Depends(get_db)):
    validate_name(body.name)
    if body.ownership not in ('external', 'managed') or (body.ownership == 'managed' and not body.acknowledge):
        raise HTTPException(422, '接管文件管理需要明确确认；普通登记不会取得文件删除权')
    with managed_dataset_mutation():
        loc = location_get(db, body.location_id, ready=True)
        acquire_node_claim_lock(db, loc.node_id)
        path = str(PurePosixPath(loc.resolved_path) / body.name)
        if db.scalar(select(DatasetEntry.id).where(DatasetEntry.node_id == loc.node_id, DatasetEntry.path == path, DatasetEntry.status != 'deleted')):
            raise HTTPException(409, '该目录已登记')
        op = new_operation(db, 'register', loc.node_id, {'phase': 'export', 'source_node': loc.node_id,
                           'source_path': path, 'location_id': loc.id, 'name': body.name, 'ownership': body.ownership})
        db.commit()
        return operation_read(op)


@router.get('/datasets/{identity}/delete-preview')
def delete_preview(identity: str, db: Session = Depends(get_db)):
    e = db.get(DatasetEntry, identity)
    if not e or e.status == 'deleted':
        raise HTTPException(404, '数据集不存在')
    value = entry_read(e, db)
    value['can_delete'] = not value['references'] and e.ownership == 'managed' and not busy_entry(db, identity)
    return value


@router.post('/datasets/{identity}/adopt')
def adopt(identity: str, body: dict, db: Session = Depends(get_db)):
    with managed_dataset_mutation():
        e = db.get(DatasetEntry, identity)
        if not e or e.status != 'ready' or busy_entry(db, identity):
            raise HTTPException(409, '数据集不可接管')
        if body.get('confirm_path') != e.path:
            raise HTTPException(422, '请填写完整目录以确认平台将获得文件删除权')
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
        if body.remove_source and (e.ownership != 'managed' or reference_rows(db, e)):
            raise HTTPException(409, '源目录仍被任务引用或未接管，请选择仅复制并保留源数据')
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
        if e.ownership != 'managed' or reference_rows(db, e) or busy_entry(db, identity):
            raise HTTPException(409, '数据集仍被引用、未接管或已有未完成操作，拒绝删除')
        op = new_operation(db, 'delete', e.node_id, {'phase': 'remove', 'entry_id': identity,
                           'source_node': e.node_id, 'source_path': e.path, 'name': e.name,
                           'fingerprint': (e.details or {}).get('fingerprint'), 'config_guards': remote_config_guards(db, e.node_id, e.path)})
        # Old managed entries get a first snapshot at deletion under the same lock.
        if e.node_id == 'local' and not op.payload['fingerprint']:
            op.payload = {**op.payload, 'initial_snapshot': True}
        if e.node_id != 'local' and not op.payload['fingerprint']:
            raise HTTPException(409, '历史远程数据缺少校验指纹，请先重新登记校验')
        db.commit()
        return operation_read(op)


@router.get('/operations')
def list_operations(db: Session = Depends(get_db)):
    return [operation_read(op, db) for op in db.scalars(select(DatasetOperation).order_by(DatasetOperation.created_at.desc()).limit(200))]


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
            op.status, op.error, op.lease_until = 'pending', None, None
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


def verify_snapshot(op):
    from .dataset_uploads import class_names, validate_contents, validate_file_path, MAX_FILE_BYTES, MAX_TEXT_BYTES
    directory = job_dir(op.id)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if files.digest(directory / 'bundle.tar') != manifest['archive_sha256']:
        raise ValueError('归档 SHA-256 校验失败')
    target = directory / 'validate'
    if target.exists():
        shutil.rmtree(target)
    try:
        files.unpack(directory / 'bundle.tar', target, manifest)
        for item in manifest['files']:
            relative = validate_file_path(item['path'])
            if item['size'] > (MAX_FILE_BYTES if relative.parts[0] == 'images' else MAX_TEXT_BYTES):
                raise ValueError('已有数据集单文件超过上传大小限制')
        names = class_names(target, op.payload.get('names', []))
        stats, warnings = validate_contents(target, manifest, names)
        if op.kind in ('register', 'migrate'):
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


def action_payload(op):
    p, phase = op.payload, op.payload['phase']
    common = {'id': op.id, 'token': op.lease, 'phase': phase}
    if phase == 'check':
        return {**common, 'path': p['path'], 'create': p.get('create', False)}
    if phase == 'export':
        return {**common, 'path': p['source_path']}
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
    if phase == 'check':
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
        if op.kind == 'register':
            existing = db.scalar(select(DatasetEntry).where(DatasetEntry.node_id == p['source_node'], DatasetEntry.path == p['source_path'], DatasetEntry.status != 'deleted'))
            if existing:
                raise ValueError('目录已经登记，拒绝重复登记')
            e = DatasetEntry(id=uid(), node_id=p['source_node'], location_id=p['location_id'], name=p['name'],
                             path=p['source_path'], ownership=p['ownership'], status='ready',
                             details={'fingerprint': manifest['fingerprint'], 'names': names, 'stats': stats,
                                      'warnings': warnings, 'total_bytes': manifest['total_bytes'], 'config_sha256': next(f['sha256'] for f in manifest['files'] if f['path'] == 'data.yaml')})
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
                   'stats': p['stats'], 'warnings': p['warnings'], 'total_bytes': result.get('total_bytes', 0), 'config_sha256': result.get('config_sha256'), 'history': history}
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
    if not entry or entry.ownership != 'managed' or reference_rows(db, entry):
        raise ValueError('源目录仍被任务引用或未接管，拒绝删除')
    if any(t.status in ('running', 'starting', 'stopping', 'queued') for t in db.scalars(select(TrainingTask).where(
            TrainingTask.execution_node_id == (None if entry.node_id == 'local' else entry.node_id)))):
        raise ValueError('该节点仍有活动训练任务，请停止后重试清理')


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
    db.commit()
    return op


def execute_local(db, op):
    identity, token, phase = op.id, op.lease, op.payload['phase']
    last = [0.0]
    def check(stage=None):
        if stage or time.monotonic() - last[0] > 4:
            with managed_dataset_mutation():
                lease_check(db, identity, token, phase, committing=stage == 'committing')
            last[0] = time.monotonic()
    check()
    p = op.payload
    directory = job_dir(op.id)
    directory.mkdir(parents=True, exist_ok=True)
    if phase == 'check':
        result = files.check_root(p['path'], DATASET_STORAGE_ALLOWED_ROOTS, p.get('create', False))
    elif phase in ('export', 'prepare_upload'):
        source = p['staging'] if phase == 'prepare_upload' else str(files.inside(p['source_path'], DATASET_STORAGE_ALLOWED_ROOTS))
        if DATASET_STORAGE_QUOTA_BYTES and sum(f.stat().st_size for f in STORE_ROOT.glob('*/bundle*')) + files.scan(source, check)['total_bytes'] * 2 > DATASET_STORAGE_QUOTA_BYTES:
            raise ValueError('平台数据集中转配额不足，请先处理失败操作或增加 PLATFORM_DATASET_STORAGE_QUOTA_BYTES')
        required = files.scan(source, check)['total_bytes'] * 2 + 100 * 1024**2
        if shutil.disk_usage(directory).free < required:
            raise ValueError('平台中转磁盘空间不足')
        result = files.pack(source, directory / 'bundle.tar', check)
        save_json(directory / 'manifest.json', result)
    elif phase == 'validate':
        result = verify_snapshot(op)
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


class DatasetWorker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True, name='dataset-storage')
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)

    def tick(self):
        STORE_ROOT.mkdir(parents=True, exist_ok=True)
        with (STORE_ROOT / '.worker.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
            with SessionLocal() as db:
                op = db.scalar(select(DatasetOperation).where(DatasetOperation.node_id == 'local', DatasetOperation.status.in_(ACTIVE)).order_by(DatasetOperation.created_at))
                if op:
                    try:
                        execute_local(db, op)
                    except Exception as exc:
                        mark_error(db, op, exc)
                        logger.warning('dataset_operation_failed id=%s error=%s', op.id, type(exc).__name__)
                # Terminal bundle retention is bounded; journal rows stay durable.
                for old in db.scalars(select(DatasetOperation).where(DatasetOperation.status.in_(TERMINAL))):
                    age = (utcnow() - old.updated_at.replace(tzinfo=timezone.utc)).total_seconds()
                    if age > 24 * 3600:
                        shutil.rmtree(job_dir(old.id), ignore_errors=True)
                        if old.kind == 'upload':
                            from .dataset_uploads import UPLOAD_ROOT
                            shutil.rmtree(UPLOAD_ROOT / old.id, ignore_errors=True)

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.tick()
            except Exception:
                logger.exception('dataset_worker_failed')
            self.stop_event.wait(1)


worker = DatasetWorker()


def auth(node_id, request, db):
    from .compute_nodes import authenticated_agent
    return authenticated_agent(node_id, request, db)


@router.post('/agent/{node_id}/poll')
def agent_poll(node_id: str, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    with managed_dataset_mutation():
        acquire_node_claim_lock(db, node_id)
        op = db.scalar(select(DatasetOperation).where(DatasetOperation.node_id == node_id, DatasetOperation.status.in_(ACTIVE)).order_by(DatasetOperation.created_at))
        if not op:
            return {'operation': None}
        # One Agent process executes a token at a time using its private flock.
        # An expired lease reuses that token, so a saved commit receipt is replayed.
        if op.lease_until and op.lease_until.replace(tzinfo=timezone.utc) > utcnow():
            return {'operation': None}
        if op.payload['phase'] in ('export', 'import') and db.scalar(select(TrainingRun.id).where(
                TrainingRun.execution_node_id == node_id, TrainingRun.status.in_(('starting', 'running', 'stopping')))):
            return {'operation': None}
        if op.payload['phase'] == 'remove':
            try:
                guard_remove(db, op)
            except ValueError as exc:
                mark_error(db, op, exc)
                return {'operation': None}
        op.lease_until, op.status = utcnow() + timedelta(seconds=120), 'running' if op.status != 'committing' else 'committing'
        db.commit()
        return {'operation': action_payload(op)}


class AgentMessage(BaseModel):
    token: str
    phase: str
    committing: bool = False
    error: str | None = Field(default=None, max_length=1200)
    result: dict = Field(default_factory=dict)


@router.post('/agent/{node_id}/{identity}/heartbeat')
def agent_heartbeat(node_id: str, identity: str, body: AgentMessage, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
    with managed_dataset_mutation():
        op = db.get(DatasetOperation, identity)
        if not op or op.node_id != node_id:
            raise HTTPException(404, '操作不存在')
        lease_check(db, identity, body.token, body.phase, body.committing)
    return {'ok': True}


@router.post('/agent/{node_id}/{identity}/finish')
def agent_finish(node_id: str, identity: str, body: AgentMessage, request: Request, db: Session = Depends(get_db)):
    auth(node_id, request, db)
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
            if not archive.is_file() or archive.stat().st_size != manifest['archive_size'] or files.digest(archive, check) != manifest['archive_sha256']:
                raise ValueError('远程数据包大小或 SHA-256 不匹配')
            with managed_dataset_mutation():
                op = lease_check(db, identity, body.token, body.phase)
                save_json(directory / 'manifest.json', manifest)
                if archive.name != 'bundle.tar':
                    archive.replace(directory / 'bundle.tar')
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
