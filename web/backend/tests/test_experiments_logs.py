import hashlib
from datetime import timedelta

import pytest
from sqlalchemy import select

import app.remote_operations as operations
import app.training_logs as training_logs
from app.log_files import tail_log
from app.models import ComputeNode, RemoteOperation, TrainingRun, TrainingTask, utcnow
from test_compute_nodes import monitoring_client
from test_remote_agent import agent
from test_remote_operations import agent_api, execute_through, remote_run


def store_log_snapshot(factory, *, operation_id, node_id, payload, content):
    digest = hashlib.sha256(content).hexdigest()
    with factory() as db:
        operation = RemoteOperation(
            id=operation_id,
            node_id=node_id,
            task_id=payload['task_id'],
            kind='fetch_log',
            status='succeeded',
            payload={
                **payload,
                'size': len(content),
                'sha256': digest,
            },
        )
        db.add(operation)
        db.commit()
        path = operations.operation_file(operation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_experiment_batch_paginates_excludes_trash_and_reports_cleanup(monitoring_client):
    client, factory = monitoring_client
    with factory() as db:
        for index in range(3):
            task = TrainingTask(id=f"task-{index}", name=f"Task {index}", task_type="detect",
                                model_path="base.pt", data_path="data.yaml", display_order=index,
                                deleted_at=utcnow() if index == 2 else None)
            run = TrainingRun(id=f"run-{index}", task=task, attempt_number=1,
                              model_path=task.model_path, data_path=task.data_path, status="completed")
            db.add_all([task, run])
        db.add(RemoteOperation(id="a" * 32, node_id="__local__", task_id="task-1", kind="local_delete",
                               status="failed", error="disk error", payload={"scope": "attempt", "run_id": "run-1"}))
        db.commit()
    first = client.get('/api/experiments?limit=1').json()
    assert first['total'] == 2
    assert first['items'][0]['run']['id'] == 'run-0'
    second = client.get('/api/experiments?offset=1&limit=1').json()
    assert second['items'][0]['run']['deletion_error'] == 'disk error'
    selected = client.get('/api/experiments?run_ids=run-1,run-2').json()
    assert [item['run']['id'] for item in selected['items']] == ['run-1']
    assert client.get('/api/experiments?limit=201').status_code == 422


def test_tail_log_reads_unicode_tail_and_bounds_single_long_line(tmp_path):
    path = tmp_path / 'train.log'
    path.write_text(''.join(f'第 {i} 轮训练\n' for i in range(100000)), encoding='utf-8')
    assert tail_log(path, 2) == ['第 99998 轮训练', '第 99999 轮训练']
    path.write_bytes(b'x' * 100000)
    assert len(tail_log(path, 2, max_bytes=1024)[0]) == 1024
    link = tmp_path / 'link.log'
    link.symlink_to(path)
    assert tail_log(link, 2) == []


@pytest.mark.parametrize('status, expected', [
    ('finalizing', 'running'), ('center_cleaning', 'running'), ('cancelled', 'failed'),
])
def test_experiment_batch_preserves_cleanup_blocking_states(monitoring_client, status, expected):
    client, factory = monitoring_client
    with factory() as db:
        task = TrainingTask(id='cleanup-state', name='Cleanup', task_type='detect',
                            model_path='base.pt', data_path='data.yaml')
        db.add(TrainingRun(id='cleanup-run', task=task, attempt_number=1,
                           model_path='base.pt', data_path='data.yaml', status='completed'))
        db.add(RemoteOperation(id='phase-operation', task_id=task.id, node_id='__local__',
                               kind='local_delete', status=status,
                               payload={'scope': 'attempt', 'run_id': 'cleanup-run'}))
        db.commit()
    assert client.get('/api/experiments').json()['items'][0]['run']['deletion_status'] == expected


def test_attempt_history_filters_task_before_applying_history_limit(monitoring_client):
    client, factory = monitoring_client
    with factory() as db:
        for index, task_id in enumerate(['older-task', 'newer-task']):
            timestamp = utcnow() - timedelta(days=1 - index)
            db.add(RemoteOperation(
                id=f'history-{index}', task_id=task_id, node_id='__local__',
                kind='local_delete', status='succeeded', created_at=timestamp, updated_at=timestamp,
                payload={'scope': 'attempt', 'run_id': f'historical-run-{index}', 'attempt_number': 1},
            ))
        db.commit()
    assert [job['task_id'] for job in client.get('/api/attempt-deletions?limit=1').json()] == ['newer-task']
    response = client.get('/api/attempt-deletions?limit=1&task_id=older-task')
    assert [job['task_id'] for job in response.json()] == ['older-task']


def test_local_full_log_download_preserves_early_lines(monitoring_client, tmp_path, monkeypatch):
    client, factory = monitoring_client
    monkeypatch.setattr(training_logs, 'LOG_ROOT', tmp_path)
    path = tmp_path / 'complete.log'
    content = ''.join(f'epoch {i}\n' for i in range(1000)).encode()
    path.write_bytes(content)
    with factory() as db:
        task = TrainingTask(id='local-logs', name='Logs', task_type='detect', model_path='base.pt', data_path='data.yaml')
        db.add(TrainingRun(id='local-run', task=task, attempt_number=1, model_path='base.pt', data_path='data.yaml', log_path=str(path)))
        db.commit()
    endpoint = '/api/tasks/local-logs/runs/local-run/logs'
    assert client.post(endpoint + '/prepare').json()['status'] == 'ready'
    response = client.get(endpoint + '/download')
    assert response.status_code == 200 and response.content == content
    monkeypatch.setattr(training_logs, 'MAX_LOG_BYTES', 10)
    assert client.get(endpoint + '/download').status_code == 413
    path.unlink(); path.symlink_to(tmp_path / 'other')
    assert client.get(endpoint + '/download').status_code == 403


@pytest.mark.parametrize('content', [b'', b'first epoch\n' + b'complete log line\n' * 40000], ids=['empty', 'multiple-chunks'])
def test_remote_full_log_transfer_and_download(monitoring_client, tmp_path, monkeypatch, agent, content):
    client, factory = monitoring_client
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    credential, config, payload = remote_run(client, factory, tmp_path)
    from pathlib import Path
    Path(payload['log_path']).write_bytes(content)
    with factory() as db:
        node = db.get(ComputeNode, credential['node']['id'])
        node.agent_version = '2.0.6'
        db.commit()
    monkeypatch.setattr(agent, 'api_json', agent_api(client, credential))
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs"
    assert client.post(endpoint + '/prepare?refresh=true').json()['status'] == 'pending'
    with factory() as db:
        operation = db.scalar(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log'))
        operation_id = operation.id
    execute_through(client, credential, config, agent, operation_id)
    ready = client.post(endpoint + '/prepare').json()
    assert ready['status'] == 'ready', ready
    response = client.get(endpoint + '/download')
    assert response.status_code == 200 and response.content == content


def test_remote_log_requires_updated_agent(monitoring_client, tmp_path):
    client, factory = monitoring_client
    _, _, payload = remote_run(client, factory, tmp_path)
    response = client.post(f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs/prepare")
    assert response.status_code == 409 and '2.0.6' in response.json()['detail']


def test_remote_log_snapshot_quota_fails_without_publishing(monitoring_client, tmp_path, monkeypatch, agent):
    client, factory = monitoring_client
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    monkeypatch.setattr(operations, 'REMOTE_LOG_CACHE_QUOTA_BYTES', 1)
    credential, config, payload = remote_run(client, factory, tmp_path)
    with factory() as db:
        db.get(ComputeNode, credential['node']['id']).agent_version = '2.0.6'
        db.commit()
    monkeypatch.setattr(agent, 'api_json', agent_api(client, credential))
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs"
    assert client.post(endpoint + '/prepare').json()['status'] == 'pending'
    with factory() as db:
        operation = db.scalar(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log'))
        operation_id = operation.id
    execute_through(client, credential, config, agent, operation_id)
    result = client.post(endpoint + '/prepare').json()
    assert result['status'] == 'failed' and '容量' in result['message']
    assert client.get(endpoint + '/download').status_code == 409
    with factory() as db:
        operation = db.get(RemoteOperation, operation_id)
        assert not operations.operation_file(operation).exists()


def test_remote_log_snapshot_requires_current_source_and_supports_legacy_run_node(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    credential, _, payload = remote_run(client, factory, tmp_path)
    node_id = credential['node']['id']
    with factory() as db:
        db.get(ComputeNode, node_id).agent_version = '2.0.6'
        run = db.get(TrainingRun, payload['run_id'])
        run.execution_node_id = None
        db.commit()
    content = b'legacy remote run log\n'
    old_path = store_log_snapshot(
        factory,
        operation_id='b' * 32,
        node_id=node_id,
        payload=payload,
        content=content,
    )
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs"
    ready = client.post(endpoint + '/prepare').json()
    assert ready['status'] == 'ready' and ready['snapshot_at']
    assert client.get(endpoint + '/download').content == content

    changed_path = str(tmp_path / 'output' / 'changed.log')
    with factory() as db:
        db.get(TrainingRun, payload['run_id']).log_path = changed_path
        db.commit()
    assert client.get(endpoint + '/download').status_code == 409
    response = client.post(endpoint + '/prepare')
    assert response.status_code == 200 and response.json()['status'] == 'pending'
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log')))
        assert len(fetches) == 1
        assert fetches[0].status == 'pending'
        assert fetches[0].node_id == node_id
        assert fetches[0].payload['log_path'] == changed_path
    assert not old_path.exists()


def test_remote_log_refresh_falls_back_then_replaces_previous_snapshot(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    credential, _, payload = remote_run(client, factory, tmp_path)
    node_id = credential['node']['id']
    with factory() as db:
        db.get(ComputeNode, node_id).agent_version = '2.0.6'
        db.commit()
    old_content = b'usable snapshot before refresh\n'
    old_path = store_log_snapshot(
        factory,
        operation_id='c' * 32,
        node_id=node_id,
        payload=payload,
        content=old_content,
    )
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs"

    assert client.post(endpoint + '/prepare?refresh=true').json()['status'] == 'pending'
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log')))
        assert {operation.status for operation in fetches} == {'succeeded', 'pending'}
        assert len(fetches) == 2
        refresh = next(operation for operation in fetches if operation.status == 'pending')
        refresh_id = refresh.id
        refresh.status = 'failed'
        refresh.error = 'temporary transfer failure'
        db.commit()
    fallback = client.post(endpoint + '/prepare').json()
    assert fallback['status'] == 'ready' and fallback['snapshot_at']
    assert client.get(endpoint + '/download').content == old_content
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log')))
        assert [operation.id for operation in fetches] == ['c' * 32]
        assert db.get(RemoteOperation, refresh_id) is None
    assert old_path.exists()

    assert client.post(endpoint + '/prepare?refresh=true').json()['status'] == 'pending'
    new_content = b'new successful snapshot\n'
    with factory() as db:
        refresh = db.scalar(select(RemoteOperation).where(RemoteOperation.status == 'pending'))
        refresh_id = refresh.id
        refresh.status = 'succeeded'
        refresh.payload = {
            **refresh.payload,
            'size': len(new_content),
            'sha256': hashlib.sha256(new_content).hexdigest(),
        }
        db.commit()
        new_path = operations.operation_file(refresh)
    new_path.write_bytes(new_content)
    current = client.post(endpoint + '/prepare').json()
    assert current['status'] == 'ready' and current['snapshot_at']
    assert client.get(endpoint + '/download').content == new_content
    with factory() as db:
        fetches = list(db.scalars(select(RemoteOperation).where(RemoteOperation.kind == 'fetch_log')))
        assert [operation.id for operation in fetches] == [refresh_id]
    assert not old_path.exists()


def test_remote_log_download_streams_verified_open_inode(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    monkeypatch.setattr(operations, 'CACHE_ROOT', tmp_path / 'cache')
    credential, _, payload = remote_run(client, factory, tmp_path)
    node_id = credential['node']['id']
    with factory() as db:
        db.get(ComputeNode, node_id).agent_version = '2.0.6'
        db.commit()
    content = b'verified snapshot bytes\n'
    path = store_log_snapshot(
        factory,
        operation_id='d' * 32,
        node_id=node_id,
        payload=payload,
        content=content,
    )
    original_open = training_logs.open_remote_snapshot

    def replace_path_after_open(operation):
        handle, size = original_open(operation)
        path.unlink()
        path.write_bytes(b'replacement must not be streamed')
        return handle, size

    monkeypatch.setattr(training_logs, 'open_remote_snapshot', replace_path_after_open)
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/logs/download"
    response = client.get(endpoint)
    assert response.status_code == 200
    assert response.content == content
    assert response.headers['content-length'] == str(len(content))
