"""Incremental training CSV reads; standard library only for standalone Agents."""
from __future__ import annotations

import csv
import math
import threading
from collections import OrderedDict
from pathlib import Path


class MetricCsvReader:
    """Keep bounded per-file caches; incomplete trailing records wait for a newline."""

    def __init__(self, maximum_files=128):
        self.maximum_files = max(1, maximum_files)
        self._files = OrderedDict()
        self._lock = threading.RLock()

    def read(self, path, *, final=False, incremental=False):
        path = Path(path)
        key = str(path.absolute())
        with self._lock:
            try:
                info = path.stat()
                signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
                state = self._files.pop(key, None)
                with path.open('rb') as stream:
                    reset = state is None or state['signature'][:2] != signature[:2]
                    reset = reset or info.st_size < state['offset']
                    if state is not None and not reset:
                        # An append must retain the already consumed suffix. Same-size
                        # rewrites and resumed/truncated files start a fresh generation.
                        reset = info.st_size <= state['signature'][2] and signature != state['signature']
                        if not reset and state['anchor']:
                            stream.seek(state['offset'] - len(state['anchor']))
                            reset = stream.read(len(state['anchor'])) != state['anchor']
                        if not reset and state['signature'] == signature and (not final or not state['partial']):
                            state['delta'] = []
                            self._files[key] = state
                            return self._snapshot(state, incremental=incremental)
                    if reset:
                        state = {'offset': 0, 'header': None, 'metrics': [], 'completed': 0,
                                 'latest': {}, 'anchor': b'', 'partial': False, 'delta': [],
                                 'generation': (state['generation'] + 1) if state else 1}
                    state['delta'] = []
                    stream.seek(state['offset'])
                    state['partial'] = False
                    while True:
                        start = stream.tell()
                        line = stream.readline()
                        if not line:
                            break
                        if not line.endswith(b'\n') and not final:
                            stream.seek(start)
                            state['partial'] = True
                            break
                        try:
                            values = next(csv.reader([line.decode('utf-8-sig', errors='replace')]))
                        except (csv.Error, StopIteration):
                            continue
                        if state['header'] is None:
                            state['header'] = [value.strip() for value in values]
                            continue
                        # CSV output contains one numeric record per physical line.
                        # Reject truncated/extra fields rather than publishing half an epoch.
                        if len(values) != len(state['header']):
                            continue
                        metric = self._metric(dict(zip(state['header'], values)))
                        if metric is not None:
                            state['metrics'].append(metric)
                            state['delta'].append(metric)
                            state['completed'] = max(state['completed'], metric['epoch'])
                            state['latest'] = metric['values']
                    state['offset'] = stream.tell()
                    stream.seek(max(0, state['offset'] - 256))
                    state['anchor'] = stream.read(min(256, state['offset']))
                state['signature'] = signature
                self._files[key] = state
                while len(self._files) > self.maximum_files:
                    self._files.popitem(last=False)
                return self._snapshot(state, incremental=incremental)
            except (OSError, csv.Error):
                # Missing files can be normal during trainer startup/rotation.
                return {'completed': 0, 'latest': {}, 'metrics': [], 'generation': 0}

    @staticmethod
    def _snapshot(state, *, incremental=False):
        return {'completed': state['completed'], 'latest': dict(state['latest']),
                'metrics': list(state['delta'] if incremental else state['metrics']),
                'generation': state['generation']}

    @staticmethod
    def _metric(row):
        try:
            epoch_number = float(row.get('epoch', ''))
            if not math.isfinite(epoch_number):
                return None
            epoch = int(epoch_number)
        except (ValueError, TypeError, OverflowError):
            return None
        values = {}
        for key, raw in row.items():
            if not key or key in {'epoch', 'time'}:
                continue
            if raw is None or raw == '':
                values[key] = None
                continue
            try:
                number = float(raw)
                values[key] = number if math.isfinite(number) else None
            except (ValueError, TypeError, OverflowError):
                values[key] = str(raw)[:200]
        try:
            elapsed = float(row.get('time') or 0)
            if not math.isfinite(elapsed) or elapsed < 0:
                elapsed = 0.0
        except (ValueError, TypeError, OverflowError):
            elapsed = 0.0
        return {'epoch': epoch, 'elapsed': elapsed, 'values': values}
