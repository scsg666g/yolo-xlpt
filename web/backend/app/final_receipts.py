"""Verify lost final ACKs against durable, completed remote deletion records."""
from __future__ import annotations

import hashlib
import hmac
import posixpath

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String, cast, select

from .models import RemoteOperation, TrainingRun


class ReceiptReconciliation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    version: int = Field(ge=1, le=1)
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    node_id: str = Field(min_length=1, max_length=64)
    claim_lease: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    run_dir: str | None = Field(default=None, max_length=4096)
    nonce: str = Field(pattern=r'^[0-9a-f]{32}$')


def receipt_proofs(runs) -> dict:
    return {run.id: {
        'lease_sha256': hashlib.sha256(run.claim_lease.encode()).hexdigest() if run.claim_lease else None,
        'run_dir': run.run_dir,
    } for run in runs}


def reconcile_deleted_run(db, node_id: str, run_id: str, body: ReceiptReconciliation) -> dict:
    if body.node_id != node_id or body.run_id != run_id:
        raise HTTPException(409, '训练回执身份不匹配')
    # A live run must use the normal update protocol, even if an old failed
    # cleanup exists. Unknown 404s must never be treated as deletion ACKs.
    if db.get(TrainingRun, run_id) is not None:
        raise HTTPException(409, '训练运行仍存在，请通过正常上报接口确认')
    query = select(RemoteOperation).where(
        RemoteOperation.node_id == node_id,
        RemoteOperation.kind == 'delete', RemoteOperation.status == 'succeeded',
        cast(RemoteOperation.payload, String).contains(run_id, autoescape=True),
    ).order_by(RemoteOperation.created_at.desc())
    for operation in db.scalars(query):
        payload = operation.payload or {}
        plans = [plan for plan in payload.get('runs', [])
                 if isinstance(plan, dict) and plan.get('run_id') == run_id]
        if payload.get('scope') not in {'task', 'attempt'} or len(plans) != 1:
            continue
        if payload.get('scope') == 'attempt' and payload.get('run_id') != run_id:
            continue
        proof = (payload.get('final_receipt_proofs') or {}).get(run_id)
        expected_path = (proof or {}).get('run_dir') or plans[0].get('run_dir')
        verified_lease = False
        if proof is not None:
            expected_lease = proof.get('lease_sha256')
            supplied_lease = hashlib.sha256(body.claim_lease.encode()).hexdigest() if body.claim_lease else None
            if bool(expected_lease) != bool(supplied_lease):
                continue
            if expected_lease and not hmac.compare_digest(expected_lease, supplied_lease):
                continue
            verified_lease = bool(expected_lease)
        if body.run_dir is not None:
            if not expected_path or posixpath.normpath(body.run_dir) != posixpath.normpath(expected_path):
                continue
        elif not verified_lease:
            # Historical early-failure receipts omitted the path. A matching
            # durable lease digest proves their identity; path-only legacy
            # tombstones cannot safely acknowledge such incomplete receipts.
            continue
        # Legacy successful operations have a node-bound frozen run ID/path
        # but no lease digest. Preserve that historical evidence, never infer
        # deletion just because a database record is missing.
        return {'version': 1, 'disposition': 'deleted', 'run_id': run_id,
                'node_id': node_id, 'nonce': body.nonce, 'operation_id': operation.id}
    raise HTTPException(404, '没有可核验的已完成训练删除记录，保留本地回执')
