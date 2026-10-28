"""Recover confirmed automatic batch reductions from training logs."""
from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_REDUCTION = re.compile(r"\bwith batch=(\d+)\.\s+Reducing to batch=(\d+) and retrying\b", re.I)
MAX_STEPS = 8


def append_reductions(history: list[int], lines, configured: int) -> list[int]:
    """Accept only contiguous, decreasing transitions explicitly logged by the trainer."""
    result = list(history or [])
    for raw in lines:
        match = _REDUCTION.search(_ANSI.sub("", raw))
        if not match:
            continue
        old, new = map(int, match.groups())
        if not 1 <= new < old <= 1024:
            continue
        if not result:
            if configured > 0 and old != configured:
                # A tail that starts at the second reduction is incomplete.
                continue
            result = [old]
        if result[-1] == new:
            continue
        if result[-1] == old and len(result) < MAX_STEPS:
            result.append(new)
    return result if len(result) > 1 else []


def scan_batch_log(path: str | Path | None, configured: int) -> tuple[list[int], bool]:
    """Stream an existing log once; a missing log remains eligible for later recovery."""
    if not path:
        return [], False
    try:
        with Path(path).open("r", encoding="utf-8", errors="replace") as source:
            return append_reductions([], source, configured), True
    except (OSError, UnicodeError):
        return [], False


class BatchLogReader:
    """Read only appended complete lines while an active training log grows."""

    def __init__(self):
        self._files: OrderedDict[str, tuple[int, int, int, list[int]]] = OrderedDict()

    def read(self, path: str | Path | None, configured: int) -> list[int]:
        if not path:
            return []
        key = str(path)
        try:
            with Path(path).open("rb") as source:
                stat = os.fstat(source.fileno())
                state = self._files.get(key)
                if not state or state[0] != stat.st_dev or state[1] != stat.st_ino or stat.st_size < state[2]:
                    state = (stat.st_dev, stat.st_ino, 0, [])
                source.seek(state[2])
                history = list(state[3])
                while True:
                    start = source.tell()
                    line = source.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        source.seek(start)
                        break
                    history = append_reductions(history, (line.decode("utf-8", errors="replace"),), configured)
                self._files.pop(key, None)
                self._files[key] = (stat.st_dev, stat.st_ino, source.tell(), history)
                if len(self._files) > 256:
                    self._files.popitem(last=False)
                return history
        except OSError:
            return list(self._files.get(key, (None, None, None, []))[3])
