import asyncio
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import app.main as main_module


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_lifespan_cleans_managed_dataset_workdirs_before_run_import_and_workers(
    monkeypatch, cleanup_fails,
):
    calls: list[str] = []

    class SessionContext:
        def __enter__(self):
            return object()

        def __exit__(self, *_):
            return None

    class Worker:
        def __init__(self, name: str):
            self.name = name

        def start(self):
            calls.append(f"{self.name}.start")

        def stop(self):
            calls.append(f"{self.name}.stop")

    @contextmanager
    def dataset_mutation():
        calls.append("dataset_lock.enter")
        try:
            yield
        finally:
            calls.append("dataset_lock.exit")

    async def snapshot_pump():
        await asyncio.Event().wait()

    def cleanup_workdirs():
        calls.append("cleanup_managed_dataset_workdirs")
        if cleanup_fails:
            raise OSError("temporary filesystem error")

    monkeypatch.setattr(main_module, "migrate_database", lambda: calls.append("migrate_database"))
    monkeypatch.setattr(
        main_module,
        "Base",
        SimpleNamespace(metadata=SimpleNamespace(create_all=lambda _: calls.append("create_all"))),
    )
    monkeypatch.setattr(main_module, "SessionLocal", SessionContext)
    monkeypatch.setattr(main_module, "ensure_local_node", lambda _: calls.append("ensure_local_node"))
    monkeypatch.setattr(
        main_module,
        "cleanup_managed_dataset_workdirs",
        cleanup_workdirs,
    )
    monkeypatch.setattr(main_module, "managed_dataset_mutation", dataset_mutation)
    monkeypatch.setattr(main_module, "import_existing_runs", lambda: calls.append("import_existing_runs"))
    monkeypatch.setattr(main_module, "backfill_training_runs", lambda: calls.append("backfill_training_runs"))
    monkeypatch.setattr(main_module, "migrate_training_logs", lambda: calls.append("migrate_training_logs"))
    # Schema creation is mocked here, so the real background backfill must not
    # query the still-unmigrated test database after this lifespan exits.
    monkeypatch.setattr(main_module, "backfill_batch_histories", lambda: None)
    monkeypatch.setattr(main_module, "scheduler", Worker("scheduler"))
    monkeypatch.setattr(main_module, "local_deletion_worker", Worker("local_deletion_worker"))
    monkeypatch.setattr(main_module, "inference_worker", Worker("inference_worker"))
    monkeypatch.setattr(main_module, "load_task_snapshot", lambda: "snapshot")
    monkeypatch.setattr(main_module, "task_snapshot_pump", snapshot_pump)

    async def exercise_lifespan():
        async with main_module.lifespan(main_module.app):
            calls.append("serving")

    asyncio.run(exercise_lifespan())

    assert calls.index("cleanup_managed_dataset_workdirs") < calls.index("dataset_lock.enter")
    assert calls.index("cleanup_managed_dataset_workdirs") < calls.index("scheduler.start")
    assert calls[calls.index("dataset_lock.enter") + 1:calls.index("dataset_lock.exit")] == [
        "import_existing_runs",
    ]
    assert calls[-3:] == ["local_deletion_worker.stop", "inference_worker.stop", "scheduler.stop"]
