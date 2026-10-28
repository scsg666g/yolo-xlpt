"""Small, portable evidence of the dataset configuration used by a run.

A catalogue fingerprint describes the last completed validation, not an
immutable copy of the training images. Never scan a large dataset at launch.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path


MAX_CONFIG_BYTES = 1024 * 1024


def capture(raw, *, task_type='detect', node_id='local', runtime_path=None):
    snapshot = {
        'version': 1, 'node_id': str(node_id), 'task_type': str(task_type),
        'source': 'runtime_config', 'verification': 'path_only' if task_type == 'classify' else 'config_only',
        'captured_at': datetime.now(timezone.utc).isoformat(),
    }
    if task_type == 'classify':
        return snapshot
    try:
        with Path(raw).open('rb') as handle:
            content = handle.read(MAX_CONFIG_BYTES + 1)
        if len(content) <= MAX_CONFIG_BYTES:
            snapshot['config_sha256'] = hashlib.sha256(content).hexdigest()
        else:
            snapshot['verification'] = 'unavailable'
    except OSError:
        snapshot['verification'] = 'unavailable'
    if runtime_path is not None:
        try:
            with Path(runtime_path).open('rb') as handle:
                content = handle.read(MAX_CONFIG_BYTES + 1)
            if len(content) <= MAX_CONFIG_BYTES:
                snapshot['runtime_config_sha256'] = hashlib.sha256(content).hexdigest()
        except OSError:
            pass
    return snapshot


def clean(snapshot):
    """Accept only bounded configuration evidence from an authenticated Agent."""
    if not isinstance(snapshot, dict) or snapshot.get('version') != 1:
        raise ValueError('数据集快照版本无效')
    result = {'version': 1, 'source': 'runtime_config'}
    for key in ('node_id', 'task_type', 'captured_at'):
        value = snapshot.get(key)
        if not isinstance(value, str) or not value or len(value) > 100:
            raise ValueError('数据集快照字段无效')
        result[key] = value
    try:
        stamp = datetime.fromisoformat(result['captured_at'].replace('Z', '+00:00'))
        if stamp.tzinfo is None:
            raise ValueError()
    except ValueError:
        raise ValueError('数据集快照时间无效') from None
    verification = snapshot.get('verification')
    if verification not in ('config_only', 'path_only', 'unavailable'):
        raise ValueError('数据集快照验证类型无效')
    result['verification'] = verification
    for key in ('config_sha256', 'runtime_config_sha256'):
        digest = snapshot.get(key)
        if digest is not None:
            if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('数据集配置摘要无效')
            result[key] = digest
    return result
