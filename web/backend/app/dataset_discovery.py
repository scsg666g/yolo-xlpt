"""Persistent, node-neutral dataset discovery and idempotent batch registration."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
import re
from typing import Literal

import yaml
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import case, delete, exists, func, select
from sqlalchemy.orm import Session

from .database import get_db
from .models import (ComputeNode, DatasetDiscoverySession, DatasetDiscoveryCandidate,
                     DatasetRegistrationBatch, DatasetOperation, DatasetEntry, utcnow)
from . import dataset_store as store

router = APIRouter(prefix='/api/dataset-storage/discovery', tags=['dataset discovery'])
ID_PATTERN = r'^[0-9a-f]{32}$'


def relative_path(value):
    if not isinstance(value, str) or not value or len(value) > 4096:
        raise HTTPException(422, '数据集相对目录无效')
    if value != '.' and (PurePosixPath(value).is_absolute() or any(
            part in ('', '.', '..') or part.startswith('.') for part in value.split('/'))):
        raise HTTPException(422, '数据集相对目录不能包含上级跳转、隐藏目录或空路径')
    if '\\' in value or any(ord(c) < 32 for c in value):
        raise HTTPException(422, '数据集相对目录不能包含控制字符或反斜杠')
    return value


def supports_discovery(db, node_id):
    if node_id == 'local':
        return True
    node = db.get(ComputeNode, node_id)
    from .compute_nodes import agent_version_at_least
    return bool(node and node.enabled and agent_version_at_least(node.agent_version, (2, 2, 0)) and
                ((node.snapshot or {}).get('training') or {}).get('dataset_discovery'))


class ScanRequest(BaseModel):
    request_id: str = Field(pattern=ID_PATTERN)
    location_ids: list[str] = Field(min_length=1, max_length=16)
    depth: int = Field(default=3, ge=0, le=12)
    max_candidates: int = Field(default=500, ge=1, le=2000)
    max_directories: int = Field(default=10000, ge=1, le=50000)
    max_seconds: int = Field(default=120, ge=5, le=600)


def session_or_error(db, identity):
    session = db.get(DatasetDiscoverySession, identity)
    if not session:
        raise HTTPException(404, '扫描记录不存在')
    return session


def scan_read(db, session):
    ops = list(db.scalars(select(DatasetOperation).where(
        DatasetOperation.kind == 'discover', DatasetOperation.payload['session_id'].as_string() == session.id)
        .order_by(DatasetOperation.created_at)))
    counts = {status: sum(op.status == status for op in ops) for status in ('pending', 'running', 'completed', 'failed', 'cancelled')}
    active = counts['pending'] + counts['running']
    return {'id': session.id, 'specification': session.specification, 'created_at': session.created_at,
            'status': 'running' if active else 'failed' if counts['failed'] else 'cancelled' if counts['cancelled'] else 'completed',
            'operations': [{**store.operation_read(op, db), 'progress': op.result} for op in ops],
            'counts': counts}


@router.post('/scans', status_code=202)
def create_scan(body: ScanRequest, db: Session = Depends(get_db)):
    spec = body.model_dump(exclude={'request_id'})
    spec['location_ids'] = sorted(set(spec['location_ids']))
    with store.managed_dataset_mutation():
        prior = db.get(DatasetDiscoverySession, body.request_id)
        if prior:
            if prior.specification != spec:
                raise HTTPException(409, '请求编号已用于其他扫描，请重新提交')
            return scan_read(db, prior)
        locations = [store.location_get(db, identity, ready=True) for identity in spec['location_ids']]
        for loc in locations:
            if not supports_discovery(db, loc.node_id):
                raise HTTPException(409, '远程目录扫描需要 Agent 2.2.0 及 dataset_discovery 能力，请升级并等待上报')
        session = DatasetDiscoverySession(id=body.request_id, specification=spec)
        db.add(session); db.flush()
        for loc in locations:
            store.new_operation(db, 'discover', loc.node_id, {
                'session_id': session.id, 'phase': 'discover', 'location_id': loc.id,
                'name': loc.name, 'source_node': loc.node_id, 'source_path': loc.resolved_path,
                'limits': {key: value for key, value in spec.items() if key != 'location_ids'}})
        db.commit()
    return scan_read(db, session)


@router.get('/scans')
def list_scans(db: Session = Depends(get_db)):
    return [scan_read(db, s) for s in db.scalars(select(DatasetDiscoverySession)
            .order_by(DatasetDiscoverySession.created_at.desc()).limit(50))]


@router.get('/scans/{identity}')
def read_scan(identity: str, db: Session = Depends(get_db)):
    return scan_read(db, session_or_error(db, identity))


@router.post('/scans/{identity}/cancel')
def cancel_scan(identity: str, db: Session = Depends(get_db)):
    with store.managed_dataset_mutation():
        session = session_or_error(db, identity)
        for op in db.scalars(select(DatasetOperation).where(DatasetOperation.kind == 'discover',
                DatasetOperation.payload['session_id'].as_string() == identity, DatasetOperation.status.in_((*store.ACTIVE, 'failed')))):
            op.status = 'cancelled'; op.updated_at = utcnow()
        db.commit()
    return scan_read(db, session)


def classify(item, path):
    """Cheap preview only. Full validation is still mandatory before publication."""
    state, reason = item['state'], item.get('reason', '')
    details = {'reason': reason[:1200], 'config_sha256': item.get('config_sha256', ''), 'task': 'unknown', 'classes': []}
    if state != 'candidate':
        return state, details
    try:
        cfg = yaml.safe_load(item['config_text'])
        if not isinstance(cfg, dict):
            raise ValueError('data.yaml 必须为配置映射')
        task = cfg.get('task')
        if task in ('classify', 'segment', 'pose', 'obb') or cfg.get('kpt_shape'):
            details.update(task=str(task or 'pose'), reason='该类型暂不支持当前目标检测登记校验器')
            return 'unsupported', details
        configured = PurePosixPath(str(cfg.get('path', path)))
        if not configured.is_absolute():
            configured = PurePosixPath(path) / configured
        if str(configured).rstrip('/') != path.rstrip('/') or cfg.get('train') != 'images/train' or cfg.get('val') != 'images/val':
            raise ValueError('需自包含检测目录：path 指向本目录，train/val 为 images/train、images/val')
        if not all(item.get('structure', {}).get(n) is True for n in ('images/train', 'images/val', 'labels')):
            raise ValueError('缺少 images/train、images/val 或 labels 目录，或目录包含符号链接')
        names = cfg.get('names')
        if not isinstance(names, (list, dict)) or not names or len(names) > 2000:
            raise ValueError('缺少有效的 names 类别配置，或类别超过扫描上限')
        values = list(names.values()) if isinstance(names, dict) else names
        if any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in values):
            raise ValueError('names 中包含无效类别名称')
        details.update(classes=values[:30], task='detect', reason='结构初检通过；登记时仍需完整校验图片、标签及 SHA-256')
        return 'ready', details
    except (ValueError, TypeError, yaml.YAMLError, RecursionError) as exc:
        details['reason'] = str(exc)[:1200]
        return 'needs_attention', details


def record_progress(db, op, result):
    """Handle both local callbacks and bounded, lease-checked Agent chunks."""
    generation = result.get('generation', '')
    items = result.get('candidates', [])
    if not isinstance(generation, str) or not re.fullmatch(ID_PATTERN, generation) or not isinstance(items, list) or len(items) > 20:
        raise ValueError('扫描分块格式无效')
    progress = {'generation': generation}
    for key in ('visited', 'found', 'skipped_links', 'skipped_depth'):
        value = result.get(key, 0)
        if not isinstance(value, int) or not 0 <= value <= 10_000_000:
            raise ValueError('扫描进度无效')
        progress[key] = value
    if progress['found'] > op.payload['limits']['max_candidates'] or progress['visited'] > op.payload['limits']['max_directories']:
        raise ValueError('扫描结果超出约定限制')
    progress.update(limited=bool(result.get('limited')), limit_reason=str(result.get('limit_reason', ''))[:600],
                    current_path=str(result.get('current_path', ''))[:4096])
    root = PurePosixPath(op.payload['source_path'])
    if (op.result or {}).get('generation') != generation:
        db.execute(delete(DatasetDiscoveryCandidate).where(DatasetDiscoveryCandidate.operation_id == op.id))
    for item in items:
        if not isinstance(item, dict) or item.get('state') not in ('candidate', 'needs_attention', 'unsupported', 'inaccessible'):
            raise ValueError('候选数据格式无效')
        relative = relative_path(item.get('relative_path'))
        path = str(root / relative)
        if len(path) > 4096 or len(str(item.get('config_text', ''))) > 65536 or not isinstance(item.get('config_text', ''), str):
            raise ValueError('候选配置或路径超出限制')
        if item['state'] == 'candidate' and not re.fullmatch(r'[0-9a-f]{64}', item.get('config_sha256', '')):
            raise ValueError('候选缺少配置指纹')
        state, details = classify(item, path)
        identity = hashlib.sha256((op.id + ':' + path).encode()).hexdigest()[:32]
        candidate = db.get(DatasetDiscoveryCandidate, identity)
        if not candidate:
            candidate = DatasetDiscoveryCandidate(id=identity, session_id=op.payload['session_id'], operation_id=op.id,
                location_id=op.payload['location_id'], node_id=op.payload['source_node'], path=path, relative_path=relative)
            db.add(candidate)
        candidate.generation, candidate.state, candidate.details = generation, state, details
    db.flush()
    actual = db.scalar(select(func.count()).select_from(DatasetDiscoveryCandidate).where(
        DatasetDiscoveryCandidate.operation_id == op.id, DatasetDiscoveryCandidate.generation == generation))
    if actual > op.payload['limits']['max_candidates']:
        raise ValueError('扫描累计候选超出约定限制')
    op.result = progress
    op.updated_at = utcnow()


def candidate_query(identity):
    c = DatasetDiscoveryCandidate
    registered = exists(select(DatasetEntry.id).where(DatasetEntry.node_id == c.node_id,
            DatasetEntry.path == c.path, DatasetEntry.status != 'deleted').correlate(c))
    registering = exists(select(DatasetOperation.id).where(DatasetOperation.kind == 'register',
            DatasetOperation.payload['source_node'].as_string() == c.node_id,
            DatasetOperation.payload['source_path'].as_string() == c.path,
            DatasetOperation.status.in_((*store.ACTIVE, 'failed'))).correlate(c))
    state = case((registered, 'registered'), (registering, 'registering'),
                 (DatasetOperation.status != 'completed', 'scan_incomplete'), else_=c.state).label('current_state')
    return select(c, state).join(DatasetOperation, c.operation_id == DatasetOperation.id).where(
        c.session_id == identity, c.generation == DatasetOperation.result['generation'].as_string())


def candidate_read(db, candidate, state):
    node = db.get(ComputeNode, candidate.node_id)
    return {'id': candidate.id, 'location_id': candidate.location_id, 'node_id': candidate.node_id,
            'node_name': node.name if node else ('本机服务器' if candidate.node_id == 'local' else candidate.node_id),
            'path': candidate.path, 'relative_path': candidate.relative_path, 'name': PurePosixPath(candidate.path).name,
            'state': state, 'selection_token': candidate_token(candidate), **candidate.details}


def candidate_token(candidate):
    return hashlib.sha256(json.dumps([candidate.id, candidate.generation, candidate.details.get('config_sha256')]).encode()).hexdigest()


@router.get('/scans/{identity}/candidates')
def list_candidates(identity: str, page: int = Query(1, ge=1), page_size: int = Query(20, ge=1, le=100),
                    state: str = Query('all', max_length=32), q: str = Query('', max_length=160),
                    node_id: str = Query('all', max_length=64), db: Session = Depends(get_db)):
    session_or_error(db, identity)
    query = candidate_query(identity)
    if node_id != 'all':
        query = query.where(DatasetDiscoveryCandidate.node_id == node_id)
    if q.strip():
        query = query.where(DatasetDiscoveryCandidate.path.contains(q.strip(), autoescape=True))
    # Overlapping roots are a single visible candidate per node + canonical path.
    sub = query.subquery()
    ranked = select(sub, func.row_number().over(partition_by=(sub.c.node_id, sub.c.path),
        order_by=(case((sub.c.current_state == 'ready', 0), else_=1), sub.c.id)).label('rank')).subquery()
    base = select(ranked).where(ranked.c.rank == 1)
    counts = dict(db.execute(select(ranked.c.current_state, func.count()).where(ranked.c.rank == 1)
                            .group_by(ranked.c.current_state)).all())
    if state != 'all':
        base = base.where(ranked.c.current_state == state)
    total = db.scalar(select(func.count()).select_from(base.subquery()))
    page = min(page, max(1, (total + page_size - 1) // page_size))
    rows = db.execute(base.order_by(ranked.c.node_id, ranked.c.path).offset((page - 1) * page_size).limit(page_size)).mappings()
    items = [candidate_read(db, db.get(DatasetDiscoveryCandidate, row['id']), row['current_state']) for row in rows]
    return {'items': items, 'total': total, 'counts': counts, 'page': page, 'page_size': page_size}


class BatchRequest(BaseModel):
    request_id: str = Field(pattern=ID_PATTERN)
    candidate_ids: list[str] = Field(min_length=1, max_length=500)
    candidate_tokens: dict[str, str] = Field(max_length=500)
    ownership: Literal['external', 'managed'] = 'external'
    acknowledge: bool = False


def batch_read(db, batch):
    items = []
    operation_ids = [i['operation_id'] for i in batch.items if i.get('operation_id')]
    operations = {op.id: op for op in db.scalars(select(DatasetOperation).where(DatasetOperation.id.in_(operation_ids)))} if operation_ids else {}
    for original in batch.items:
        item = dict(original)
        if item.get('operation_id'):
            op = operations.get(item['operation_id'])
            if op:
                item.update(status=op.status, error=op.error, phase=op.payload['phase'], result=op.result,
                            can_cancel=store.operation_read(op)['can_cancel'])
        items.append(item)
    counts = {s: sum(i['status'] == s for i in items) for s in ('pending', 'running', 'committing', 'completed', 'failed', 'cancelled', 'skipped')}
    active = any(counts[s] for s in store.ACTIVE)
    return {'id': batch.id, 'session_id': batch.session_id, 'created_at': batch.created_at, 'items': items, 'counts': counts,
            'status': 'running' if active else 'completed_with_errors' if counts['failed'] or counts['skipped'] else 'completed'}


@router.get('/scans/{identity}/batches')
def list_batches(identity: str, db: Session = Depends(get_db)):
    session_or_error(db, identity)
    return [batch_read(db, b) for b in db.scalars(select(DatasetRegistrationBatch)
            .where(DatasetRegistrationBatch.session_id == identity).order_by(DatasetRegistrationBatch.created_at.desc()).limit(50))]


@router.post('/scans/{identity}/batches', status_code=202)
def register_batch(identity: str, body: BatchRequest, db: Session = Depends(get_db)):
    if body.ownership == 'managed' and not body.acknowledge:
        raise HTTPException(422, '批量接管需明确确认文件删除权限')
    selection = sorted(set(body.candidate_ids))
    if set(body.candidate_tokens) != set(selection):
        raise HTTPException(422, '请提供所有选中候选的扫描快照')
    fingerprint = hashlib.sha256(json.dumps([identity, selection, body.candidate_tokens, body.ownership, body.acknowledge], sort_keys=True).encode()).hexdigest()
    with store.managed_dataset_mutation():
        # SQLite's node lock starts by rolling back the read transaction. Take
        # it once before creating any child operations, never once per item.
        if db.get_bind().dialect.name == 'sqlite':
            store.acquire_node_claim_lock(db, 'local')
        session_or_error(db, identity)
        batch = db.get(DatasetRegistrationBatch, body.request_id)
        if batch:
            if batch.fingerprint != fingerprint:
                raise HTTPException(409, '请求编号已用于其他批次，请重新提交')
            return batch_read(db, batch)
        candidates = {c.id: (c, state) for c, state in db.execute(candidate_query(identity).where(DatasetDiscoveryCandidate.id.in_(selection)))}
        if set(candidates) != set(selection):
            raise HTTPException(409, '部分候选已失效或不属于本次扫描，请刷新结果')
        if db.get_bind().dialect.name != 'sqlite':
            for node_id in sorted({c.node_id for c, _ in candidates.values()}):
                store.acquire_node_claim_lock(db, node_id)
        items = []
        for candidate_id in selection:
            c, state = candidates[candidate_id]
            item = {'candidate_id': c.id, 'name': PurePosixPath(c.path).name, 'path': c.path, 'node_id': c.node_id,
                    'status': 'skipped', 'error': None}
            try:
                if body.candidate_tokens[c.id] != candidate_token(c):
                    raise HTTPException(409, '候选在预览后发生变化，请重新扫描并选择')
                if state != 'ready':
                    raise HTTPException(409, '候选状态已变化或尚未通过结构初检，请刷新后查看')
                loc = store.location_get(db, c.location_id, ready=True)
                if str(PurePosixPath(loc.resolved_path) / c.relative_path) != c.path:
                    raise HTTPException(409, '保存位置发生变化，请重新扫描')
                op = store.queue_registration(db, loc, c.relative_path, item['name'], body.ownership,
                                              c.details['config_sha256'], body.request_id, claim_locked=True)
                item.update(operation_id=op.id, status=op.status)
            except HTTPException as exc:
                item['error'] = str(exc.detail)
            items.append(item)
        batch = DatasetRegistrationBatch(id=body.request_id, session_id=identity, fingerprint=fingerprint, items=items)
        db.add(batch); db.commit()
    return batch_read(db, batch)


@router.post('/batches/{identity}/cancel')
def cancel_batch(identity: str, db: Session = Depends(get_db)):
    with store.managed_dataset_mutation():
        batch = db.get(DatasetRegistrationBatch, identity)
        if not batch:
            raise HTTPException(404, '登记批次不存在')
        for item in batch.items:
            op = db.get(DatasetOperation, item.get('operation_id')) if item.get('operation_id') else None
            if op and store.operation_read(op)['can_cancel']:
                op.status = 'cancelled'; op.updated_at = utcnow()
        db.commit()
    return batch_read(db, batch)
