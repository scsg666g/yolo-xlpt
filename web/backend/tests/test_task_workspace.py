from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import TaskWorkspace, TrainingTask, utcnow


@pytest.fixture
def workspace_client(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'workspace.db'}")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, expire_on_commit=False)

    def override_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    client = TestClient(app)
    try:
        yield client, session_factory
    finally:
        client.close()
        app.dependency_overrides.pop(get_db, None)
        engine.dispose()


def add_task(session_factory, task_id: str, *, deleted: bool = False) -> None:
    with session_factory() as db:
        db.add(TrainingTask(
            id=task_id,
            name=task_id,
            model_path="model.pt",
            data_path="data.yaml",
            config={},
            deleted_at=utcnow() if deleted else None,
        ))
        db.commit()


def workspace_payload(revision: int = 0) -> dict:
    return {
        "revision": revision,
        "modes": {
            "all": {
                "nodes": [
                    {"id": "first", "position": {"x": 10.5, "y": -20}},
                    {"id": "second", "position": {"x": 450, "y": 30}},
                ],
                "edges": [{
                    "id": "first:second", "source": "first",
                    "target": "second", "label": "  关联  ",
                }],
                "viewport": {"x": 24, "y": 12, "zoom": 0.8},
            }
        },
    }


def test_workspace_defaults_and_persists_with_revision(workspace_client):
    client, session_factory = workspace_client
    add_task(session_factory, "first")
    add_task(session_factory, "second")

    empty = client.get("/api/task-workspace")
    assert empty.status_code == 200
    assert empty.json() == {"version": 1, "revision": 0, "modes": {}, "updated_at": None}

    saved = client.put("/api/task-workspace", json=workspace_payload())
    assert saved.status_code == 200, saved.text
    result = saved.json()
    assert result["revision"] == 1
    assert result["modes"]["all"]["edges"][0]["label"] == "关联"
    assert client.get("/api/task-workspace").json() == result

    replacement = workspace_payload(revision=1)
    replacement["modes"]["all"]["viewport"]["zoom"] = 1.25
    updated = client.put("/api/task-workspace", json=replacement)
    assert updated.status_code == 200
    assert updated.json()["revision"] == 2
    assert updated.json()["modes"]["all"]["viewport"]["zoom"] == 1.25

    reset = client.put("/api/task-workspace", json={"revision": 2, "modes": {}})
    assert reset.status_code == 200
    assert reset.json()["revision"] == 3
    assert reset.json()["modes"] == {}


def test_workspace_rejects_stale_revision(workspace_client):
    client, session_factory = workspace_client
    add_task(session_factory, "first")
    add_task(session_factory, "second")
    assert client.put("/api/task-workspace", json=workspace_payload()).status_code == 200

    stale = client.put("/api/task-workspace", json=workspace_payload())
    assert stale.status_code == 409
    assert client.get("/api/task-workspace").json()["revision"] == 1


def test_workspace_filters_missing_and_deleted_tasks(workspace_client):
    client, session_factory = workspace_client
    add_task(session_factory, "first")
    add_task(session_factory, "second", deleted=True)
    payload = workspace_payload()
    payload["modes"]["all"]["nodes"].append(
        {"id": "missing", "position": {"x": 0, "y": 0}}
    )

    saved = client.put("/api/task-workspace", json=payload)
    assert saved.status_code == 200, saved.text
    mode = saved.json()["modes"]["all"]
    assert [node["id"] for node in mode["nodes"]] == ["first"]
    assert mode["edges"] == []


def test_workspace_keeps_historical_task_ids_without_new_character_rules(workspace_client):
    client, session_factory = workspace_client
    add_task(session_factory, "legacy.task:001")
    payload = {
        "revision": 0,
        "modes": {
            "all": {
                "nodes": [{"id": "legacy.task:001", "position": {"x": 3, "y": 4}}],
                "edges": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1},
            }
        },
    }
    response = client.put("/api/task-workspace", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["modes"]["all"]["nodes"][0]["id"] == "legacy.task:001"


def test_workspace_preserves_historical_task_identifier_characters(workspace_client):
    client, session_factory = workspace_client
    historical_id = "legacy/imported run:1"
    add_task(session_factory, historical_id)
    payload = {
        "revision": 0,
        "modes": {
            "all": {
                "nodes": [{"id": historical_id, "position": {"x": 0, "y": 0}}],
                "edges": [],
                "viewport": {"x": 0, "y": 0, "zoom": 1},
            }
        },
    }

    saved = client.put("/api/task-workspace", json=payload)
    assert saved.status_code == 200, saved.text
    assert saved.json()["modes"]["all"]["nodes"][0]["id"] == historical_id


@pytest.mark.parametrize("mutate", [
    lambda payload: payload.update({"unknown": True}),
    lambda payload: payload["modes"]["all"]["nodes"].append(
        {"id": "first", "position": {"x": 1, "y": 2}}),
    lambda payload: payload["modes"]["all"]["edges"].append(
        {"id": "self", "source": "first", "target": "first"}),
    lambda payload: payload["modes"]["all"]["viewport"].update({"zoom": 0}),
    lambda payload: payload["modes"]["all"]["nodes"][0]["position"].update({"x": 10_000_001}),
])
def test_workspace_rejects_invalid_layout(workspace_client, mutate):
    client, session_factory = workspace_client
    add_task(session_factory, "first")
    add_task(session_factory, "second")
    payload = workspace_payload()
    mutate(payload)
    response = client.put("/api/task-workspace", json=payload)
    assert response.status_code == 422
    assert client.get("/api/task-workspace").json()["revision"] == 0


def test_malformed_stored_workspace_degrades_to_empty_layout(workspace_client):
    client, session_factory = workspace_client
    with session_factory() as db:
        db.add(TaskWorkspace(id="default", version=1, revision=7, modes={"all": {"broken": True}}))
        db.commit()

    response = client.get("/api/task-workspace")
    assert response.status_code == 200
    assert response.json()["revision"] == 7
    assert response.json()["modes"] == {}


def test_create_all_adds_workspace_table_to_legacy_database(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    legacy_tables = [table for table in Base.metadata.sorted_tables if table.name != "task_workspaces"]
    Base.metadata.create_all(engine, tables=legacy_tables)
    assert "task_workspaces" not in inspect(engine).get_table_names()

    Base.metadata.create_all(engine)
    assert "task_workspaces" in inspect(engine).get_table_names()
    engine.dispose()
