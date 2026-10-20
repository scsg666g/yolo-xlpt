"""Best-effort Linux process ownership evidence for GPU process snapshots."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_ID = re.compile(r"^[0-9a-f]{32}$")
_MARKERS = {
    "instance_id": "YOLO_FORGE_INSTANCE_ID",
    "node_id": "YOLO_FORGE_NODE_ID",
    "run_id": "YOLO_FORGE_RUN_ID",
    "task_id": "YOLO_FORGE_TASK_ID",
}
_TASK_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")


def _proc_stat(pid: int, proc_root: Path) -> tuple[int, int] | None:
    try:
        raw = (proc_root / str(pid) / "stat").read_text(encoding="ascii")
        closing = raw.rfind(")")
        fields = raw[closing + 2:].split()
        if closing < 0 or len(fields) <= 19 or fields[0] == "Z":
            return None
        return int(fields[1]), int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _proc_environment(pid: int, proc_root: Path) -> dict[str, str] | None:
    try:
        values = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    except OSError:
        return None
    environment: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition(b"=")
        if separator:
            environment[os.fsdecode(key)] = os.fsdecode(value)
    return environment


def _marker(environment: dict[str, str] | None) -> dict[str, str] | None:
    if environment is None:
        return None
    values = {key: environment.get(variable, "") for key, variable in _MARKERS.items()}
    if not (_ID.fullmatch(values["instance_id"]) and _ID.fullmatch(values["run_id"])
            and _TASK_ID.fullmatch(values["task_id"])
            and (values["node_id"] == "local" or _ID.fullmatch(values["node_id"]))):
        return None
    return values


def inspect_process_owner(
    pid: int,
    *,
    validated_roots: dict[int, list[dict[str, Any]]],
    proc_root: Path = Path("/proc"),
    max_depth: int = 32,
) -> dict[str, Any]:
    """Read marker and parent-chain evidence without treating missing evidence as external."""
    current = pid
    chain: list[int] = []
    markers: list[dict[str, str]] = []
    leaf_environment_readable = False
    complete = False
    for _ in range(max_depth):
        if current <= 0 or current in chain:
            break
        stat = _proc_stat(current, proc_root)
        if stat is None:
            break
        chain.append(current)
        parent, _start_ticks = stat
        environment = _proc_environment(current, proc_root)
        if len(chain) == 1:
            leaf_environment_readable = environment is not None
        candidate = _marker(environment)
        if candidate and candidate not in markers:
            markers.append(candidate)
        if current == 1 or parent == 0:
            complete = True
            break
        current = parent

    root_candidates = [
        root
        for ancestor in chain
        for root in validated_roots.get(ancestor, [])
    ]
    root_keys = {(item.get("instance_id"), item.get("node_id"), item.get("run_id"), item.get("task_id"))
                 for item in root_candidates}
    marker_keys = {(item["instance_id"], item["node_id"], item["run_id"], item["task_id"])
                   for item in markers}
    if len(root_keys | marker_keys) > 1:
        return {"match_method": "ambiguous", "scan_complete": complete}

    if markers:
        owner = markers[0]
        return {**owner, "match_method": "runtime_marker", "scan_complete": complete}
    if root_candidates:
        return {**root_candidates[0], "match_method": "ancestor", "scan_complete": complete}
    return {"match_method": "none", "scan_complete": complete and leaf_environment_readable}
