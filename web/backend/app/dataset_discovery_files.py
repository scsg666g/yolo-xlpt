"""Read-only, bounded discovery. Standard library only; also embedded in the Agent."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import time
import uuid

CONFIG_LIMIT = 64 * 1024
SKIP = {'images', 'labels', 'train', 'val', 'test', 'runs', 'weights', 'node_modules', '__pycache__'}


def discover(path, roots, limits, report, check):
    """Publish small idempotent chunks; never read image/label contents or write sources.

    A restart begins a new generation, so old partial candidates cannot be mistaken
    for the completed result. The centre retains the last progress across restarts.
    """
    root = Path(path)
    if not root.is_absolute() or root != root.resolve() or not root.is_dir():
        raise ValueError('扫描目录不存在或包含符号链接')
    if not any(root == Path(r).resolve() or Path(r).resolve() in root.parents for r in roots):
        raise ValueError('扫描目录不在允许范围内')
    depth = int(limits.get('depth', 3))
    maximum = int(limits.get('max_candidates', 500))
    directories = int(limits.get('max_directories', 10000))
    seconds = int(limits.get('max_seconds', 120))
    if not (0 <= depth <= 12 and 1 <= maximum <= 2000 and 1 <= directories <= 50000 and 5 <= seconds <= 600):
        raise ValueError('扫描限制无效')
    generation = uuid.uuid4().hex
    progress = {'generation': generation, 'visited': 0, 'found': 0, 'skipped_links': 0,
                'skipped_depth': 0, 'limited': False, 'limit_reason': '', 'current_path': str(root)}
    pending, stack = [], [(root, 0)]
    started = last_report = time.monotonic()
    entries_seen = 0

    def flush(force=False):
        nonlocal pending, last_report
        if force or len(pending) >= 20 or time.monotonic() - last_report >= 2:
            report({**progress, 'candidates': pending})
            pending = []
            last_report = time.monotonic()

    def limit(reason):
        progress.update(limited=True, limit_reason=reason)

    flush(True)
    while stack:
        check()
        if progress['visited'] >= directories:
            limit('达到目录数量上限，请缩小扫描位置或调高限制'); break
        if time.monotonic() - started >= seconds:
            limit('达到扫描时间上限，请缩小扫描位置或调高限制'); break
        directory, level = stack.pop()
        progress.update(visited=progress['visited'] + 1, current_path=str(directory))
        item = None
        try:
            # Recheck every parent, rather than following a link swapped in mid-scan.
            if directory != directory.resolve() or not directory.is_dir():
                progress['skipped_links'] += 1
                continue
            children, names = [], set()
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    entries_seen += 1
                    if entries_seen % 128 == 0:
                        check(); flush()
                    if time.monotonic() - started >= seconds or entries_seen > directories * 100:
                        limit('达到扫描时间或目录项数量上限，请缩小扫描范围'); break
                    if entry.name.startswith('.'):
                        continue
                    if entry.name in {'data.yaml', 'data.yml', 'images', 'train', 'val'}:
                        names.add(entry.name)
                    if entry.is_symlink():
                        progress['skipped_links'] += 1
                    elif entry.name not in SKIP and entry.is_dir(follow_symlinks=False):
                        children.append(Path(entry.path))
                        if len(children) + len(stack) > directories:
                            limit('达到待检查目录数量上限，请缩小扫描范围'); break
            likely = 'data.yaml' in names or 'data.yml' in names or 'images' in names or ('train' in names and 'val' in names)
            if likely:
                item = {'relative_path': directory.relative_to(root).as_posix(), 'state': 'needs_attention',
                        'reason': '', 'config_text': '', 'config_sha256': '',
                        'structure': {n: (directory / n).is_dir() and not (directory / n).is_symlink()
                                      for n in ('images/train', 'images/val', 'labels', 'train', 'val')}}
                config = directory / 'data.yaml'
                if 'data.yaml' not in names:
                    item['reason'] = '缺少 data.yaml；请整理目录或使用上传导入'
                    if 'train' in names and 'val' in names and 'images' not in names:
                        item.update(state='unsupported', reason='疑似分类数据集，当前登记校验器仅支持目标检测')
                else:
                    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
                    with os.fdopen(os.open(config, flags), 'rb') as handle:
                        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                            raise ValueError('data.yaml 必须是普通文件')
                        raw = handle.read(CONFIG_LIMIT + 1)
                    if len(raw) > CONFIG_LIMIT:
                        item['reason'] = 'data.yaml 超过扫描配置大小上限（64 KiB）'
                    else:
                        item.update(state='candidate', config_text=raw.decode('utf-8-sig'),
                                    config_sha256=hashlib.sha256(raw).hexdigest())
            # Dataset internals are excluded, but nested sibling versions remain discoverable.
            if level < depth:
                stack.extend((child, level + 1) for child in sorted(children, reverse=True))
            else:
                progress['skipped_depth'] += len(children)
        except (OSError, UnicodeError, ValueError) as exc:
            item = {'relative_path': directory.relative_to(root).as_posix(), 'state': 'inaccessible',
                    'reason': ('无法检查目录或配置：' + str(exc))[:600], 'config_text': '', 'config_sha256': '', 'structure': {}}
        if item:
            pending.append(item); progress['found'] += 1
        flush()
        if progress['limited']:
            break
        if progress['found'] >= maximum:
            if stack:
                limit('达到候选数量上限，请缩小扫描位置或调高限制')
            break
    flush(True)
    return progress
