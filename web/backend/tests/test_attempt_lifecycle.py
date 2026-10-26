"""Isolated local/Agent lifecycle regressions; never touch deployed task files."""
from pathlib import Path
import signal
import sys
import threading
import time

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Artifact, ComputeNode, TaskEvent, TrainingRun, TrainingTask, utcnow
import app.database as database_module
import app.run_deletions as deletions
import app.services as services
from app.local_deletions import LocalDeletionWorker, queue_local_run_delete
from test_compute_nodes import monitoring_client
from test_remote_agent import agent, training
from test_remote_operations import remote_run, agent_api, claim


def local_chain(tmp_path, monkeypatch, status="completed", legacy=False):
    root = tmp_path / "runs"
    monkeypatch.setattr(deletions, "RUNS_ROOT", root)
    monkeypatch.setattr(deletions, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(deletions, "LOG_ROOT", tmp_path / "logs")
    engine = create_engine(f"sqlite:///{tmp_path / 'db.sqlite'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as db:
        task = TrainingTask(id="chain", name="Chain", task_type="detect", model_path="base.pt", data_path="data.yaml")
        first_dir = root / "detect/chain" / ("" if legacy else "attempt-001")
        next_dir = root / "detect/chain/attempt-002"
        source = TrainingRun(id="source", task=task, attempt_number=1, status="stopped", model_path="base.pt", data_path="data.yaml", run_dir=str(first_dir))
        child = TrainingRun(id="child", task=task, attempt_number=2, status=status, completed_epochs=10, launch_mode="resume",
                            model_path="base.pt", data_path="data.yaml", run_dir=str(next_dir), checkpoint_path=str(first_dir / "weights/last.pt"))
        task.current_run_id = child.id
        db.add_all([task, source, child])
        db.flush()
        for run, directory in ((source, first_dir), (child, next_dir)):
            (directory / "weights").mkdir(parents=True)
            for name in ("best.pt", "last.pt"):
                path = directory / "weights" / name
                path.write_bytes(b"weight")
                db.add(Artifact(task_id=task.id, run_id=run.id, kind="weight", name="weights/" + name, path=str(path), size=6))
        db.commit()
    return factory, first_dir, next_dir


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("status", ["completed", "stopped"])
def test_delete_old_checkpoint_retains_independent_results_and_provenance(tmp_path, monkeypatch, legacy, status):
    factory, first_dir, next_dir = local_chain(tmp_path, monkeypatch, status, legacy)
    with factory() as db:
        queue_local_run_delete(db, db.get(TrainingTask, "chain"), db.get(TrainingRun, "source"))
        db.commit()
    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        assert db.get(TrainingRun, "source") is None
        child = db.get(TrainingRun, "child")
        assert child.checkpoint_path is None
        assert child.checkpoint_source["source_deleted"] is True
        assert child.checkpoint_source["attempt_number"] == 1
        assert child.checkpoint_source["checkpoint_path"] == str(first_dir / "weights/last.pt")
    assert not (first_dir / "weights").exists()
    assert (next_dir / "weights/best.pt").read_bytes() == b"weight"


@pytest.mark.parametrize("problem", ["running", "queued", "failed", "interrupted", "missing_best", "missing_last", "symlink"])
def test_checkpoint_dependency_not_released_without_safe_descendant(tmp_path, monkeypatch, problem):
    factory, first_dir, next_dir = local_chain(tmp_path, monkeypatch)
    with factory() as db:
        child = db.get(TrainingRun, "child")
        if problem in {"running", "queued", "failed", "interrupted"}:
            child.status = problem
        else:
            path = next_dir / "weights" / ("last.pt" if problem == "missing_last" else "best.pt")
            path.unlink()
            if problem == "symlink":
                path.symlink_to(first_dir / "weights/best.pt")
        db.commit()
        with pytest.raises(ValueError, match="检查点"):
            queue_local_run_delete(db, db.get(TrainingTask, "chain"), db.get(TrainingRun, "source"))
    assert (first_dir / "weights/last.pt").exists()


def remote_chain(client, factory, tmp_path):
    credential, config, payload = remote_run(client, factory, tmp_path)
    with factory() as db:
        task = db.get(TrainingTask, payload["task_id"])
        db.get(ComputeNode, task.execution_node_id).agent_version = "2.0.8"
        source = db.get(TrainingRun, payload["run_id"])
        child = services.create_training_run(db, task, "resume", Path(source.run_dir) / "weights/last.pt")
        child.status, child.completed_epochs = "completed", 10
        services.mirror_run_to_task(task, child)
        (Path(child.run_dir) / "weights").mkdir(parents=True)
        for name in ("best.pt", "last.pt"):
            path = Path(child.run_dir) / "weights" / name
            path.write_bytes(b"independent")
            db.add(Artifact(task_id=task.id, run_id=child.id, kind="weight", name="weights/" + name, path=str(path), size=11))
        db.commit()
        child_id = child.id
    return credential, config, payload, child_id


@pytest.mark.parametrize("missing", [False, True])
def test_remote_old_checkpoint_cleanup_checks_real_independent_files(monitoring_client, agent, tmp_path, monkeypatch, missing):
    client, factory = monitoring_client
    credential, config, payload, child_id = remote_chain(client, factory, tmp_path)
    monkeypatch.setattr(agent, "api_json", agent_api(client, credential))
    endpoint = f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}"
    preview = client.get(endpoint + "/deletion-preview")
    assert preview.status_code == 200, preview.text
    assert preview.json()["allowed"]
    assert preview.json()["checkpoint_releases"][0]["run_id"] == child_id
    assert client.delete(endpoint).status_code == 202
    work = claim(client, credential)
    assert len(work["payload"]["required_files"]) == 2
    if missing:
        Path(work["payload"]["required_files"][0]["path"]).unlink()
    agent.execute_operation(config, work, threading.Event())
    with factory() as db:
        source = db.get(TrainingRun, payload["run_id"])
        child = db.get(TrainingRun, child_id)
        if missing:
            assert source and child.checkpoint_path
            assert Path(payload["run_dir"]).is_dir()
        else:
            assert source is None
            assert child.checkpoint_path is None and child.checkpoint_source["source_deleted"]
            assert not Path(payload["run_dir"]).exists()


def test_preview_blocks_old_agent_without_mutation(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, config, payload, child_id = remote_chain(client, factory, tmp_path)
    with factory() as db:
        db.get(ComputeNode, credential["node"]["id"]).agent_version = "2.0.7"
        db.commit()
    response = client.get(f"/api/tasks/{payload['task_id']}/runs/{payload['run_id']}/deletion-preview")
    assert response.status_code == 200
    assert not response.json()["allowed"]
    assert "2.0.8" in response.json()["blockers"][0]
    with factory() as db:
        assert db.get(TrainingRun, child_id).checkpoint_path


def test_local_stop_escalates_once_and_checks_adopted_identity(tmp_path, monkeypatch):
    factory, _, _ = local_chain(tmp_path, monkeypatch)
    scheduler = services.TrainingScheduler(session_factory=factory)
    signals = []
    now = [100.0]
    monkeypatch.setattr(services.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(services, "TRAINING_STOP_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(services.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(services, "process_matches_training", lambda *args: True)
    managed = services.ManagedTrainingProcess("child", 987654, ["test"], utcnow(), adopted=True)
    with factory() as db:
        run, task = db.get(TrainingRun, "child"), db.get(TrainingTask, "chain")
        scheduler._enforce_stop(db, task, run, managed)
        now[0] = 104
        scheduler._enforce_stop(db, task, run, managed)
        scheduler._enforce_stop(db, task, run, managed)
        assert signals == [signal.SIGTERM, signal.SIGKILL]
        assert db.scalar(select(TaskEvent).where(TaskEvent.event_type == "stop_escalated"))
        managed.stop_deadline, managed.stop_escalated = None, False
        monkeypatch.setattr(services, "process_matches_training", lambda *args: False)
        scheduler._enforce_stop(db, task, run, managed)
        assert len(signals) == 2


def test_agent_kills_process_ignoring_term_after_platform_stop(agent, training, monkeypatch):
    config, task = training
    config["training_stop_timeout_seconds"] = 1
    original = agent.build_training_command
    def command(config, task):
        original(config, task)
        return [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(60)"]
    monkeypatch.setattr(agent, "build_training_command", command)
    updates = []
    def api(config, path, method="GET", payload=None):
        if path.endswith("/update"):
            updates.append(payload)
            if payload.get("status") == "running":
                deadline = time.monotonic() + 3
                while 'ready' not in Path(task["log_path"]).read_text().splitlines() and time.monotonic() < deadline:
                    time.sleep(0.01)
                return {"stop_requested": True}
        return {}
    monkeypatch.setattr(agent, "api_json", api)
    started = time.monotonic()
    agent.run_remote_task(config, task, threading.Event())
    assert time.monotonic() - started < 8
    assert updates[-1]["status"] == "stopped"
    assert updates[-1]["exit_code"] == -signal.SIGKILL


def test_lifecycle_migration_preserves_old_runs_and_is_idempotent(tmp_path, monkeypatch):
    factory, _, _ = local_chain(tmp_path, monkeypatch)
    engine = factory.kw["bind"]
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE training_tasks DROP COLUMN queue_reason"))
        conn.execute(text("ALTER TABLE training_runs DROP COLUMN queue_reason"))
        conn.execute(text("ALTER TABLE training_runs DROP COLUMN checkpoint_source"))
    monkeypatch.setattr(database_module, "engine", engine)
    assert database_module.migrate_database().is_file()
    with factory() as db:
        child = db.get(TrainingRun, "child")
        assert child.status == "completed" and child.checkpoint_path
        assert child.checkpoint_source == {} and child.queue_reason is None
    assert database_module.migrate_database() is None


def test_local_queue_reason_persists_and_clears_on_claim(tmp_path, monkeypatch):
    factory, _, _ = local_chain(tmp_path, monkeypatch)
    scheduler = services.TrainingScheduler(session_factory=factory)
    monkeypatch.setattr(services, "gpu_inventory", lambda: [])
    with factory() as db:
        task, run = db.get(TrainingTask, "chain"), db.get(TrainingRun, "child")
        run.status = "queued"
        run.parameter_snapshot = {"device": "99"}
        task.device = "99"
        services.mirror_run_to_task(task, run)
        db.commit()
    scheduler._run_once()
    with factory() as db:
        run, task = db.get(TrainingRun, "child"), db.get(TrainingTask, "chain")
        assert "设备清单" in run.queue_reason
        assert task.queue_reason == run.queue_reason
        run.parameter_snapshot = {"device": "cpu"}
        db.commit()
        assert scheduler._claim_queued_run(db, run.id)
        db.expire_all()
        assert db.get(TrainingRun, "child").queue_reason is None
        assert db.get(TrainingTask, "chain").queue_reason is None


def test_offline_remote_queue_reason_clears_after_reconnect(monitoring_client, tmp_path):
    client, factory = monitoring_client
    credential, config, payload, child_id = remote_chain(client, factory, tmp_path)
    with factory() as db:
        run = db.get(TrainingRun, child_id)
        run.status = "queued"
        services.mirror_run_to_task(db.get(TrainingTask, run.task_id), run)
        node = db.get(ComputeNode, credential["node"]["id"])
        node.last_seen_at = None
        db.commit()
    scheduler = services.TrainingScheduler(session_factory=factory)
    scheduler._run_once()
    with factory() as db:
        assert "节点不可用" in db.get(TrainingRun, child_id).queue_reason
        db.get(ComputeNode, credential["node"]["id"]).last_seen_at = utcnow()
        db.commit()
    scheduler._run_once()
    with factory() as db:
        assert db.get(TrainingRun, child_id).queue_reason == "等待远程 Agent 领取任务"


def test_legacy_unindexed_source_is_archived_after_cleanup(tmp_path, monkeypatch):
    factory, first_dir, next_dir = local_chain(tmp_path, monkeypatch, legacy=True)
    with factory() as db:
        for artifact in db.scalars(select(Artifact).where(Artifact.run_id == "source")):
            db.delete(artifact)
        db.commit()
        queue_local_run_delete(db, db.get(TrainingTask, "chain"), db.get(TrainingRun, "source"))
        db.commit()
    assert LocalDeletionWorker(session_factory=factory).process_one()
    with factory() as db:
        child = db.get(TrainingRun, "child")
        assert child.checkpoint_path is None and child.checkpoint_source["source_deleted"]
        assert db.get(TrainingRun, "source") is None
    assert (next_dir / "weights/last.pt").is_file()


def test_recovered_agent_process_ignoring_term_is_stopped(agent, training, monkeypatch, tmp_path):
    import json
    import subprocess
    config, task = training
    config["training_stop_timeout_seconds"] = 1
    ready = tmp_path / "child-ready"
    command = [sys.executable, "-c", "import pathlib,signal,sys,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); pathlib.Path(sys.argv[1]).touch(); time.sleep(60)", str(ready)]
    lock = agent.acquire_training_lock(config)
    process = subprocess.Popen(command, start_new_session=True, pass_fds=(lock.fileno(),))
    lock.close()
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists()
        state_path = agent.save_active_run(config, task, "running", command=command, identity=agent.process_identity(process.pid))
        updates = []
        def api(config, path, method="GET", payload=None, extra_headers=None):
            if path.endswith("/control"):
                return {"status": "stopping", "stop": True}
            updates.append(payload)
            return {}
        monkeypatch.setattr(agent, "api_json", api)
        started = time.monotonic()
        agent.recover_remote_task(config, json.loads(state_path.read_text()), threading.Event(), process_holds_lock=True)
        assert time.monotonic() - started < 8
        assert process.wait(timeout=2) == -signal.SIGKILL
        assert updates[-1]["status"] == "stopped"
        assert agent.load_active_run(config) is None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
