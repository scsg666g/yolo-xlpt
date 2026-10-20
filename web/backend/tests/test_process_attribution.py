from __future__ import annotations

from pathlib import Path

from app.process_attribution import inspect_process_owner


INSTANCE_ID = "a" * 32
NODE_ID = "local"
RUN_ID = "b" * 32
TASK_ID = "c" * 32


def write_process(proc_root: Path, pid: int, parent: int, environment: dict[str, str] | None = None) -> None:
    directory = proc_root / str(pid)
    directory.mkdir(parents=True)
    fields = ["S", str(parent), *("0" for _ in range(18))]
    fields[19] = "100"
    (directory / "stat").write_text(f"{pid} (python worker) {' '.join(fields)}\n", encoding="ascii")
    if environment is not None:
        value = "\0".join(f"{key}={item}" for key, item in environment.items()).encode()
        (directory / "environ").write_bytes(value + b"\0")


def process_tree(proc_root: Path, leaf_environment: dict[str, str] | None = None) -> None:
    write_process(proc_root, 1, 0, {})
    write_process(proc_root, 400, 1, {})
    write_process(proc_root, 500, 400, {})
    write_process(proc_root, 501, 500, leaf_environment)


def marker(instance_id: str = INSTANCE_ID, run_id: str = RUN_ID) -> dict[str, str]:
    return {
        "YOLO_FORGE_INSTANCE_ID": instance_id,
        "YOLO_FORGE_NODE_ID": NODE_ID,
        "YOLO_FORGE_RUN_ID": run_id,
        "YOLO_FORGE_TASK_ID": TASK_ID,
    }


def test_process_owner_uses_inherited_runtime_marker(tmp_path: Path):
    process_tree(tmp_path, marker())

    hint = inspect_process_owner(501, validated_roots={}, proc_root=tmp_path)

    assert hint == {
        "instance_id": INSTANCE_ID,
        "node_id": NODE_ID,
        "run_id": RUN_ID,
        "task_id": TASK_ID,
        "match_method": "runtime_marker",
        "scan_complete": True,
    }


def test_process_owner_falls_back_to_verified_launcher_ancestor(tmp_path: Path):
    process_tree(tmp_path)
    root = {\"instance_id\": INSTANCE_ID, \"node_id\": NODE_ID, \"run_id\": RUN_ID,
            \"task_id\": TASK_ID, \"root_pid\": 400}

    hint = inspect_process_owner(501, validated_roots={400: [root]}, proc_root=tmp_path)

    assert hint[\"match_method\"] == \"ancestor\"
    assert hint[\"run_id\"] == RUN_ID
    assert hint[\"root_pid\"] == 400


def test_process_owner_reports_conflicting_evidence_as_ambiguous(tmp_path: Path):
    process_tree(marker(run_id="d" * 32))
    root = {\"instance_id\": INSTANCE_ID, \"node_id\": NODE_ID, \"run_id\": RUN_ID,
            \"task_id\": TASK_ID, \"root_pid\": 400}

    hint = inspect_process_owner(501, validated_roots={400: [root]}, proc_root=tmp_path)

    assert hint == {\"match_method\": \"ambiguous\", \"scan_complete\": True}


def test_process_owner_does_not_call_incomplete_scan_unmanaged(tmp_path: Path):
    process_tree()
    (tmp_path / \"501\" / \"environ\").unlink()

    hint = inspect_process_owner(501, validated_roots={}, proc_root=tmp_path)

    assert hint == {\"match_method\": \"none\", \"scan_complete\": False}


def test_process_owner_marks_complete_unmarked_process_as_unmanaged(tmp_path: Path):
    process_tree({})

    hint = inspect_process_owner(501, validated_roots={}, proc_root=tmp_path)

    assert hint == {\"match_method\": \"none\", \"scan_complete\": True}
