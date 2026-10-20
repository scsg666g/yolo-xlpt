"""GPU reservation semantics shared by local scheduling and remote claims."""
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.gpu_reservations import reserve_gpu_resources, resolve_gpu_assignments
from app.models import GpuReservation, TrainingRun, TrainingTask


def reservation_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'gpu-reservations.db'}")
    Base.metadata.create_all(engine)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def add_run(db, run_id):
    task = TrainingTask(
        id=f"task-{run_id}", name=run_id, task_type="detect", status="queued",
        model_path="model.pt", data_path="data.yaml", config={},
    )
    run = TrainingRun(
        id=run_id, task=task, attempt_number=1, launch_mode="fresh", status="queued",
        model_path=task.model_path, data_path=task.data_path, parameter_snapshot={},
    )
    db.add_all([task, run])
    db.flush()


def gpu(key, index, uuid):
    return {"resource_key": key, "index": index, "uuid": uuid}


def test_same_gpu_is_exclusive_but_different_uuids_can_reserve_in_parallel(tmp_path):
    engine, sessions = reservation_session(tmp_path)
    try:
        with sessions() as db:
            add_run(db, "run-a")
            add_run(db, "run-b")
            add_run(db, "run-c")
            db.commit()

        with sessions.begin() as db:
            assert reserve_gpu_resources(
                db, run_id="run-a", node_id="local", assignments=[gpu("gpu:GPU-A", 0, "GPU-A")],
            )
        with sessions.begin() as db:
            assert not reserve_gpu_resources(
                db, run_id="run-b", node_id="local", assignments=[gpu("gpu:GPU-A", 0, "GPU-A")],
            )
        with sessions.begin() as db:
            assert reserve_gpu_resources(
                db, run_id="run-c", node_id="local", assignments=[gpu("gpu:GPU-B", 1, "GPU-B")],
            )

        with sessions() as db:
            reservations = db.scalars(select(GpuReservation).order_by(GpuReservation.resource_key)).all()
            assert [(item.run_id, item.resource_key) for item in reservations] == [
                ("run-a", "gpu:GPU-A"), ("run-c", "gpu:GPU-B"),
            ]
    finally:
        engine.dispose()


def test_uuid_reservations_are_scoped_to_execution_node(tmp_path):
    engine, sessions = reservation_session(tmp_path)
    try:
        with sessions() as db:
            add_run(db, "node-a-run")
            add_run(db, "node-b-run")
            db.commit()

        assignment_a = resolve_gpu_assignments(
            "0", [{"index": 0, "uuid": "GPU-SAME"}], node_id="node-a",
        )
        assignment_b = resolve_gpu_assignments(
            "0", [{"index": 0, "uuid": "GPU-SAME"}], node_id="node-b",
        )
        assert assignment_a and assignment_b
        assert assignment_a[0]["resource_key"] != assignment_b[0]["resource_key"]
        with sessions.begin() as db:
            assert reserve_gpu_resources(db, run_id="node-a-run", node_id="node-a", assignments=assignment_a)
        with sessions.begin() as db:
            assert reserve_gpu_resources(db, run_id="node-b-run", node_id="node-b", assignments=assignment_b)
    finally:
        engine.dispose()


def test_gpu_assignment_requires_inventory_match_but_uuid_is_optional_for_local_nodes():
    assert resolve_gpu_assignments("3", [], node_id="local") is None
    assert resolve_gpu_assignments("3", [{"index": 3}], node_id="local") == [{
        "resource_key": "node:local:gpu-index:3", "uuid": None, "index": 3,
    }]


def test_multi_gpu_reservation_is_all_or_nothing_on_partial_conflict(tmp_path):
    engine, sessions = reservation_session(tmp_path)
    try:
        with sessions() as db:
            add_run(db, "owner")
            add_run(db, "multi")
            db.commit()
        with sessions.begin() as db:
            assert reserve_gpu_resources(
                db, run_id="owner", node_id="remote", assignments=[gpu("gpu:GPU-1", 1, "GPU-1")],
            )
        with sessions.begin() as db:
            assert not reserve_gpu_resources(
                db, run_id="multi", node_id="remote",
                assignments=[gpu("gpu:GPU-0", 0, "GPU-0"), gpu("gpu:GPU-1", 1, "GPU-1")],
            )
        with sessions() as db:
            assert [(item.run_id, item.resource_key) for item in
                    db.scalars(select(GpuReservation).order_by(GpuReservation.resource_key))] == [
                        ("owner", "gpu:GPU-1"),
                    ]
    finally:
        engine.dispose()


def test_node_serial_reservation_conflicts_with_any_gpu_claim(tmp_path):
    engine, sessions = reservation_session(tmp_path)
    try:
        with sessions() as db:
            add_run(db, "serial")
            add_run(db, "gpu")
            db.commit()
        with sessions.begin() as db:
            assert reserve_gpu_resources(
                db, run_id="serial", node_id="remote",
                assignments=[gpu("node:remote:serial", None, None)], exclusive_node=True,
            )
        with sessions.begin() as db:
            assert not reserve_gpu_resources(
                db, run_id="gpu", node_id="remote", assignments=[gpu("gpu:GPU-0", 0, "GPU-0")],
            )
    finally:
        engine.dispose()
