"""Instance-local dataset limits, atomically persisted without a DB migration."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import stat as stat_module
import threading
import uuid

MAX_SAFE_INTEGER = 2**53 - 1
LIMITS = {
    'storage_quota_bytes': MAX_SAFE_INTEGER,
    'failed_retention_seconds': 3650 * 24 * 3600,
    'refresh_read_bytes_per_second': 1024**4,
    'refresh_io_pressure_pct': 100,
    'refresh_min_free_bytes': MAX_SAFE_INTEGER,
}
_cache_lock = threading.Lock()
_cache = {}


def validate(values):
    if not isinstance(values, dict) or set(values) - LIMITS.keys():
        raise ValueError('数据集设置包含未知字段')
    for key, value in values.items():
        if type(value) is not int or not 0 <= value <= LIMITS[key]:
            raise ValueError(f'{key} 必须是 0 至 {LIMITS[key]} 之间的整数')
    return values


def read(root: Path, defaults: dict):
    """Stat-only cache checks keep executing workers cheap and cross-process fresh.

    The defaults are supplied at call time, preserving environment settings and
    existing integrations that replace dataset_store's constants in tests.
    """
    path = root / '.settings.json'
    try:
        if root.is_symlink():
            raise ValueError('中转目录本身不能是符号链接')
        stat = path.lstat()
        if not stat_module.S_ISREG(stat.st_mode):
            raise ValueError('设置文件必须是普通文件')
        key = (str(path), stat.st_ino, stat.st_mtime_ns, stat.st_size)
        with _cache_lock:
            cached = _cache.get(key)
        if cached is None:
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(descriptor, 'r', encoding='utf-8') as handle:
                if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError('设置文件必须是普通文件')
                raw = handle.read(32769)
            if len(raw) > 32768:
                raise ValueError('设置文件超出大小限制')
            document = json.loads(raw)
            if (not isinstance(document, dict) or document.get('version') != 1
                    or type(document.get('revision')) is not int
                    or not 1 <= document['revision'] <= MAX_SAFE_INTEGER):
                raise ValueError('设置文件版本或修订号无效')
            cached = (document['revision'], dict(validate(document['policy'])))
            with _cache_lock:
                _cache.clear()
                _cache[key] = cached
        revision, overrides = cached
        return {'revision': revision, 'policy': {**defaults, **overrides},
                'policy_source': 'instance', 'policy_error': None}
    except FileNotFoundError:
        return {'revision': 0, 'policy': dict(defaults), 'policy_source': 'environment', 'policy_error': None}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return {'revision': 0, 'policy': dict(defaults), 'policy_source': 'environment',
                'policy_error': f'实例设置无法读取，当前使用环境默认值：{exc}'}


@contextmanager
def write_lock(root):
    if root.is_symlink():
        raise ValueError('中转目录本身不能是符号链接')
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / '.settings.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a+b') as handle:
        if not stat_module.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError('设置锁文件必须是普通文件')
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def update(root, defaults, revision, values):
    validate(values)
    with write_lock(root):
        current = read(root, defaults)
        if current['policy_error']:
            raise ValueError(current['policy_error'])
        if revision != current['revision']:
            raise FileExistsError('数据集设置已被其他页面更新，请刷新后重试')
        if revision >= MAX_SAFE_INTEGER:
            raise ValueError('设置修订号已达到安全整数上限')
        document = {'version': 1, 'revision': revision + 1,
                    'policy': {**current['policy'], **values}}
        temporary = root / ('.settings-' + uuid.uuid4().hex + '.tmp')
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
                json.dump(document, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, root / '.settings.json')
            directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        return read(root, defaults)
