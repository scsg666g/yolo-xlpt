"""A capable Agent can verify a read-only reload while training continues."""
import threading

from app.models import ComputeNode, TrainingRun, TrainingTask, utcnow
from test_dataset_concurrency import config
from test_dataset_refresh import setup_entry
from test_dataset_storage import env, entries, operation, pump, store


def test_remote_refresh_runs_during_training_without_opening_mutation_gate(env):
    client, factory, root, agent = env
    entry, folder, node_id = setup_entry(env, True)
    with factory() as db:
        node = db.get(ComputeNode, node_id)
        node.agent_version = '2.5.1'
        node.last_seen_at = utcnow()
        node.snapshot = {'training': {'dataset_roots': [str(folder.parent)],
                                      'dataset_concurrent_reads': True,
                                      'dataset_concurrent_refresh': True,
                                      'lock_occupied': True}}
        task = TrainingTask(id='active-refresh-task', name='训练中', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='running', execution_node_id=node_id)
        db.add(task)
        db.flush()
        db.add(TrainingRun(id='active-refresh-run', task_id=task.id, attempt_number=1,
                           status='running', execution_node_id=node_id,
                           model_path=task.model_path, data_path=task.data_path))
        db.commit()

    response = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh")
    assert response.status_code == 202, response.text
    op_id = response.json()['id']
    view = operation(client, op_id)
    assert not view['waiting_training'] and not view['waiting_lock']
    with factory() as db:
        assert not store.node_dataset_file_work(db, node_id)

    cfg = config(folder.parent, node_id)
    work = agent.poll_dataset_operation(cfg, busy=True)
    assert work['id'] == op_id
    assert work['phase'] == 'export' and work['kind'] == 'refresh'
    assert work['read_only_during_training'] is True
    agent.run_dataset_operation(cfg, work, threading.Event())
    pump(client, factory, folder.parent, agent, node_id)
    assert operation(client, op_id)['status'] == 'completed'
    assert entries(client)[0]['id'] == entry['id']
    with factory() as db:
        assert db.get(TrainingRun, 'active-refresh-run').status == 'running'


def test_remote_refresh_old_agent_still_waits_for_training(env):
    client, factory, root, agent = env
    entry, folder, node_id = setup_entry(env, True)
    with factory() as db:
        node = db.get(ComputeNode, node_id)
        node.last_seen_at = utcnow()
        node.snapshot = {'training': {'dataset_roots': [str(folder.parent)],
                                      'dataset_concurrent_reads': True, 'lock_occupied': True}}
        task = TrainingTask(id='legacy-refresh-task', name='旧 Agent', model_path='model.pt',
                            data_path=str(folder / 'data.yaml'), status='running', execution_node_id=node_id)
        db.add(task)
        db.flush()
        db.add(TrainingRun(id='legacy-refresh-run', task_id=task.id, attempt_number=1,
                           status='running', execution_node_id=node_id,
                           model_path=task.model_path, data_path=task.data_path))
        db.commit()
    op_id = client.post(f"/api/dataset-storage/datasets/{entry['id']}/refresh").json()['id']
    assert operation(client, op_id)['waiting_training']
    assert agent.poll_dataset_operation(config(folder.parent, node_id), busy=True) is None
    with factory() as db:
        assert store.node_dataset_file_work(db, node_id)
