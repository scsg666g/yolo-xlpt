import base64
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from sqlalchemy import select

import app.remote_operations as operations
from app.models import RemoteOperation, TrainingRun, utcnow
from test_compute_nodes import headers, monitoring_client
from test_remote_operations import remote_run


def test_simultaneous_log_snapshots_reserve_capacity_before_upload(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential, _, source = remote_run(client, factory, tmp_path)
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    monkeypatch.setattr(operations, 'REMOTE_LOG_CACHE_QUOTA_BYTES', 5)
    monkeypatch.setattr(operations, 'REMOTE_LOG_MIN_FREE_BYTES', 0)
    node_id = credential['node']['id']
    with factory() as db:
        second = TrainingRun(
            id='second-log-run', task_id=source['task_id'], attempt_number=2,
            execution_node_id=node_id, model_path='base.pt', data_path='data.yaml',
            log_path=str(tmp_path / 'second.log'), status='completed',
        )
        db.add(second)
        for index, run in enumerate([db.get(TrainingRun, source['run_id']), second]):
            db.add(RemoteOperation(
                id=str(index + 1) * 32, node_id=node_id, task_id=source['task_id'],
                kind='fetch_log', status='running', lease=str(index + 3) * 32,
                payload={'run_id': run.id, 'log_path': run.log_path, 'max_bytes': 100},
            ))
        db.commit()

    original_reserve = operations.reserve_log_cache

    def delayed_reserve(db):
        original_reserve(db)
        # Keep the first reservation uncommitted while the other request arrives.
        time.sleep(0.05)

    monkeypatch.setattr(operations, 'reserve_log_cache', delayed_reserve)
    ready = Barrier(2)
    modified_at = utcnow().isoformat()

    def upload(index):
        ready.wait(timeout=5)
        return client.post(
            f'/api/compute/nodes/{node_id}/agent/operations/{str(index + 1) * 32}/chunk',
            headers=headers(credential),
            json={
                'lease': str(index + 3) * 32, 'offset': 0,
                'data': base64.b64encode(b'ab').decode(), 'final': False,
                'size': 4, 'modified_at': modified_at, 'mtime_ns': 1,
            },
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(upload, range(2)))
    assert sorted(response.status_code for response in responses) == [200, 507]
    with factory() as db:
        records = list(db.scalars(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log')))
        assert sorted(record.status for record in records) == ['failed', 'running']
        assert sum(operations.operation_file(record, False).exists() for record in records) == 1
