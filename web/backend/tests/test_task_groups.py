import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import TaskEvent, TaskGroup, TrainingTask, utcnow


@pytest.fixture
def group_client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'groups.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)
    with session_factory() as db:
        def override_db():
            yield db

        app.dependency_overrides[get_db] = override_db
        client = TestClient(app)
        try:
            yield client, db
        finally:
            client.close()
            app.dependency_overrides.pop(get_db, None)
    engine.dispose()


def add_task(db, task_id: str, name: str, group_id: str | None = None, **values) -> TrainingTask:
    task = TrainingTask(
        id=task_id,
        name=name,
        group_id=group_id,
        model_path="model.pt",
        data_path="data.yaml",
        config={},
        **values,
    )
    db.add(task)
    db.commit()
    return task


def create_group(client: TestClient, name: str = "火灾检测") -> dict:
    response = client.post("/api/task-groups", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


def test_group_crud_rejects_duplicate_names(group_client):
    client, _ = group_client
    first = create_group(client, "  火灾检测  ")
    assert first["name"] == "火灾检测"
    assert client.post("/api/task-groups", json={"name": "火灾检测"}).status_code == 409

    renamed = client.patch(f"/api/task-groups/{first['id']}", json={"name": "烟雾检测"})
    assert renamed.status_code == 200
    assert renamed.json()["name"] == "烟雾检测"
    assert [group["id"] for group in client.get("/api/task-groups").json()] == [first["id"]]


def test_new_groups_are_listed_before_existing_groups(group_client):
    client, _ = group_client
    existing = create_group(client, "已有分组")
    created = create_group(client, "新分组")

    assert [group["id"] for group in client.get("/api/task-groups").json()] == [created["id"], existing["id"]]


def test_create_group_from_ungrouped_tasks_is_atomic(group_client):
    client, db = group_client
    existing = create_group(client, "已有分组")
    first = add_task(db, "first", "火")
    second = add_task(db, "second", "烟", status="completed")

    response = client.post("/api/task-groups/from-tasks", json={
        "name": "火灾检测",
        "task_ids": [first.id, second.id],
    })

    assert response.status_code == 201
    group = response.json()
    assert [item["id"] for item in client.get("/api/task-groups").json()] == [group["id"], existing["id"]]
    db.expire_all()
    assert db.get(TrainingTask, first.id).group_id == group["id"]
    assert db.get(TrainingTask, second.id).group_id == group["id"]
    events = list(db.scalars(select(TaskEvent).where(
        TaskEvent.task_id.in_([first.id, second.id]),
        TaskEvent.event_type == "group_changed",
    )).all())
    assert len(events) == 2

    rejected = client.post("/api/task-groups/from-tasks", json={
        "name": "重复整理",
        "task_ids": [first.id],
    })
    assert rejected.status_code == 409
    assert db.scalar(select(TaskGroup).where(TaskGroup.name == "重复整理")) is None

    missing = client.post("/api/task-groups/from-tasks", json={
        "name": "不完整分组",
        "task_ids": ["missing"],
    })
    assert missing.status_code == 404
    assert db.scalar(select(TaskGroup).where(TaskGroup.name == "不完整分组")) is None


def test_assignment_clone_and_delete_group_preserve_tasks(group_client):
    client, db = group_client
    group = create_group(client)
    source = add_task(db, "source", "火", status="completed")
    deleted = add_task(db, "deleted", "历史火", group_id=group["id"], deleted_at=utcnow())

    moved = client.patch(f"/api/tasks/{source.id}/group", json={"group_id": group["id"]})
    assert moved.status_code == 200
    assert moved.json()["group_id"] == group["id"]
    event = db.scalar(select(TaskEvent).where(TaskEvent.task_id == source.id, TaskEvent.event_type == "group_changed"))
    assert event and event.payload["group_name"] == "火灾检测"

    clone = client.post(f"/api/tasks/{source.id}/clone")
    assert clone.status_code == 201
    assert clone.json()["group_id"] == group["id"]

    response = client.delete(f"/api/task-groups/{group['id']}")
    assert response.status_code == 204
    db.expire_all()
    assert db.get(TaskGroup, group["id"]) is None
    assert db.get(TrainingTask, source.id).group_id is None
    assert db.get(TrainingTask, clone.json()["id"]).group_id is None
    assert db.get(TrainingTask, deleted.id).group_id is None


def test_assignment_validates_group_and_allows_ungrouping(group_client):
    client, db = group_client
    group = create_group(client)
    task = add_task(db, "task", "任务", group_id=group["id"])

    missing = client.patch(f"/api/tasks/{task.id}/group", json={"group_id": "missing"})
    assert missing.status_code == 404
    unchanged = client.get(f"/api/tasks/{task.id}").json()
    assert unchanged["group_id"] == group["id"]

    ungrouped = client.patch(f"/api/tasks/{task.id}/group", json={"group_id": None})
    assert ungrouped.status_code == 200
    assert ungrouped.json()["group_id"] is None
