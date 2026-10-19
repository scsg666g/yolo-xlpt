import pytest
import app.dataset_uploads as dataset_uploads
import app.main as main_module
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import TaskEvent, TaskGroup, TrainingTask, utcnow


@pytest.fixture
def clone_client(tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_uploads, "UPLOAD_ROOT", tmp_path / "staging")
    engine = create_engine(f"sqlite:///{tmp_path / 'clone.db'}")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, expire_on_commit=False)() as db:
        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        try:
            # Do not enter the lifespan, which starts the real training scheduler.
            client = TestClient(app)
            try:
                yield client, db
            finally:
                client.close()
        finally:
            app.dependency_overrides.pop(get_db, None)
    engine.dispose()


def add_task(db, task_id, name, **values):
    task = TrainingTask(
        id=task_id, name=name, model_path="model.pt", data_path="data.yaml",
        config=values.pop("config", {}), **values,
    )
    db.add(task)
    db.commit()
    return task


def clone(client, task_id):
    response = client.post(f"/api/tasks/{task_id}/clone")
    assert response.status_code == 201, response.text
    return response.json()


def test_cloning_original_and_copy_share_numbering(clone_client):
    client, db = clone_client
    db.add(TaskGroup(id="fire-group", name="火灾检测", display_order=1000))
    db.commit()
    source = add_task(
        db, "source", "火", status="completed", epochs=150, batch=4, imgsz=960,
        group_id="fire-group",
        device="0,1", optimizer="MuSGD", completed_epochs=150,
        config={"lr0": 0.001, "_resume_checkpoint": "last.pt"},
        canvas_layout={"model": {"x": 10, "y": 20}},
    )
    first = clone(client, source.id)
    second = clone(client, source.id)
    third = clone(client, first["id"])
    assert [item["name"] for item in (first, second, third)] == [
        "火 - 副本 1", "火 - 副本 2", "火 - 副本 3",
    ]
    assert first["status"] == "draft"
    assert first["group_id"] == "fire-group"
    assert first["completed_epochs"] == 0
    assert first["config"] == {"lr0": 0.001}
    for key in ("model_path", "data_path", "epochs", "batch", "imgsz", "device", "optimizer", "canvas_layout"):
        assert first[key] == getattr(source, key)
    copied = db.get(TrainingTask, first["id"])
    assert copied.runs == []
    event = db.scalar(select(TaskEvent).where(TaskEvent.task_id == copied.id))
    assert event.payload["source_id"] == source.id
    assert source.name == "火"
    assert source.config["_resume_checkpoint"] == "last.pt"


def test_remote_clone_does_not_take_local_dataset_lock(clone_client, monkeypatch):
    client, db = clone_client
    source = add_task(db, "remote-source", "远程任务", execution_node_id="remote-node")

    def forbidden_lock():
        raise AssertionError("远程任务不应获取本机数据集锁")

    monkeypatch.setattr(main_module, "managed_dataset_mutation", forbidden_lock)
    result = clone(client, source.id)
    assert result["execution_node_id"] == "remote-node"


@pytest.mark.parametrize("source_name", ["火", "火 - 副本", "火 - 副本 - 副本", "火 - 副本 2"])
def test_legacy_names_and_trashed_copies_reserve_numbers(clone_client, source_name):
    client, db = clone_client
    source = add_task(db, "source", source_name)
    add_task(db, "legacy", "火 - 副本")
    add_task(db, "trashed", "火 - 副本 7", deleted_at=utcnow())
    add_task(db, "other", "烟 - 副本 20")
    assert clone(client, source.id)["name"] == "火 - 副本 8"


@pytest.mark.parametrize("legacy_name, expected", [
    ("火 - 副本", "火 - 副本 2"),
    ("火 - 副本 - 副本", "火 - 副本 2"),
    ("火 - 副本 5 - 副本", "火 - 副本 6"),
])
def test_legacy_suffixes_determine_next_number(clone_client, legacy_name, expected):
    client, db = clone_client
    source = add_task(db, "source", "火")
    add_task(db, "legacy", legacy_name)
    assert clone(client, source.id)["name"] == expected


def test_long_names_fit_limit_and_remain_distinct(clone_client):
    client, db = clone_client
    source = add_task(db, "source", "火" * 160)
    first = clone(client, source.id)
    second = clone(client, source.id)
    third = clone(client, first["id"])
    assert len(first["name"]) == len(second["name"]) == len(third["name"]) == 160
    assert first["name"].endswith(" - 副本 1")
    assert second["name"].endswith(" - 副本 2")
    assert third["name"].endswith(" - 副本 3")
