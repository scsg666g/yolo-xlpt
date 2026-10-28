import importlib.util
from pathlib import Path
from urllib.error import HTTPError

import pytest

from app.final_receipts import receipt_proofs
from app.models import RemoteOperation, TrainingRun, TrainingTask
from test_compute_nodes import create, headers, monitoring_client


RUN_ID = 'a' * 32
LEASE = 'b' * 32


def deleted_run(factory, node_id, *, legacy=False, status='succeeded', scope='attempt', lease=LEASE):
    task = TrainingTask(id='deleted-task', name='Deleted', model_path='model.pt', data_path='data.yaml',
                        execution_node_id=node_id)
    run = TrainingRun(id=RUN_ID, task=task, attempt_number=1, status='completed',
                      model_path='model.pt', data_path='data.yaml', execution_node_id=node_id,
                      claim_lease=lease, run_dir='/portable/output/attempt-001')
    payload = {'scope': scope, 'runs': [{'run_id': RUN_ID, 'run_dir': run.run_dir}]}
    if scope == 'attempt':
        payload['run_id'] = RUN_ID
    if not legacy:
        payload['final_receipt_proofs'] = receipt_proofs([run])
    with factory() as db:
        db.add(RemoteOperation(id='c' * 32, node_id=node_id, task_id=task.id,
                               kind='delete', status=status, payload=payload))
        db.commit()
    return payload


def request_body(identity, **values):
    return {'version': 1, 'run_id': RUN_ID, 'node_id': identity, 'claim_lease': LEASE,
            'run_dir': '/portable/output/attempt-001', 'nonce': 'd' * 32, **values}


@pytest.mark.parametrize('scope', ['attempt', 'task'])
@pytest.mark.parametrize('legacy', [False, True])
def test_known_completed_delete_reconciles_old_and_new_receipts(monitoring_client, scope, legacy):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    payload = deleted_run(factory, node_id, legacy=legacy, scope=scope)
    assert LEASE not in str(payload)
    body = request_body(node_id)
    url = f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile'
    for _ in range(2):
        response = client.post(url, headers=headers(credential), json=body)
        assert response.status_code == 200, response.text
        assert response.json() == {'version': 1, 'disposition': 'deleted', 'run_id': RUN_ID,
            'node_id': node_id, 'nonce': body['nonce'], 'operation_id': 'c' * 32}


@pytest.mark.parametrize('change', [{'claim_lease': 'e' * 32}, {'claim_lease': None},
    {'run_dir': '/other/run'}, {'run_dir': ''}, {'node_id': 'other-node'}, {'run_id': 'e' * 32}])
def test_identity_or_path_mismatch_is_never_acknowledged(monitoring_client, change):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id)
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=request_body(node_id, **change))
    assert response.status_code in {404, 409}


@pytest.mark.parametrize('scope', ['attempt', 'task'])
@pytest.mark.parametrize('include_null_path', [False, True])
def test_historical_early_failure_receipt_uses_verified_lease_without_run_dir(
    monitoring_client, scope, include_null_path,
):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id, scope=scope)
    body = request_body(node_id, run_dir=None)
    if not include_null_path:
        body.pop('run_dir')
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=body)
    assert response.status_code == 200, response.text
    assert response.json()['disposition'] == 'deleted'


@pytest.mark.parametrize('legacy,lease,supplied_lease', [
    (True, LEASE, LEASE), (True, None, None), (False, None, None),
    (False, LEASE, None), (False, LEASE, 'e' * 32),
])
def test_missing_run_dir_cannot_reconcile_without_matching_durable_lease_proof(
    monitoring_client, legacy, lease, supplied_lease,
):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id, legacy=legacy, lease=lease)
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
        headers=headers(credential), json=request_body(node_id, claim_lease=supplied_lease, run_dir=None))
    assert response.status_code == 404


@pytest.mark.parametrize('legacy', [True, False])
def test_historical_unleased_run_reconciles_with_exact_path(monitoring_client, legacy):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id, legacy=legacy, lease=None)
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=request_body(node_id, claim_lease=None))
    assert response.status_code == 200, response.text


def test_agent_replays_historical_pathless_failure_against_real_reconciliation_endpoint(
    monitoring_client, tmp_path, monkeypatch,
):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id)
    script = Path(__file__).resolve().parents[2] / 'agent/monitor_agent.py'
    spec = importlib.util.spec_from_file_location('historical_receipt_agent', script)
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    config = {'node_id': node_id, 'training_state_dir': str(tmp_path / 'agent-state')}
    original = {'status': 'failed', 'exit_code': 1, 'error_message': 'Historical launch failure',
                'metrics': [], '_claim_lease': LEASE}
    pending = agent.save_pending_final(config, RUN_ID, original)
    original_bytes = pending.read_bytes()
    endpoints = []

    def api(_config, endpoint, method='GET', payload=None, extra_headers=None):
        endpoints.append(endpoint)
        response = client.request(method, endpoint, json=payload,
                                  headers={**headers(credential), **(extra_headers or {})})
        if response.status_code >= 400:
            raise HTTPError(endpoint, response.status_code, response.text, None, None)
        return response.json()

    monkeypatch.setattr(agent, 'api_json', api)
    assert agent.retry_pending_finals(config) is True
    assert [endpoint.rsplit('/', 1)[1] for endpoint in endpoints] == ['update', 'reconcile']
    assert not pending.exists()
    assert (pending.parent / 'archived' / f'{RUN_ID}.deleted.json').read_bytes() == original_bytes


@pytest.mark.parametrize('status', ['pending', 'running', 'failed', 'cancelled'])
def test_unfinished_delete_never_reconciles_receipt(monitoring_client, status):
    client, factory = monitoring_client
    credential = create(client)
    node_id = credential['node']['id']
    deleted_run(factory, node_id, status=status)
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=request_body(node_id))
    assert response.status_code == 404


def test_unknown_and_other_node_receipts_are_not_acknowledged(monitoring_client):
    client, factory = monitoring_client
    credential, other = create(client), create(client)
    node_id = credential['node']['id']
    deleted_run(factory, other['node']['id'])
    response = client.post(f'/api/compute/nodes/{node_id}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=request_body(node_id))
    assert response.status_code == 404
    response = client.post(f'/api/compute/nodes/{other["node"]["id"]}/agent/runs/{RUN_ID}/reconcile',
                           headers=headers(credential), json=request_body(other['node']['id']))
    assert response.status_code in {401, 403}
