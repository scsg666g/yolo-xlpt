"""Read-only evidence for dataset path conflict decisions.

A false result proves separation. Missing or ambiguous evidence returns None.
"""
from __future__ import annotations

from pathlib import Path
import os
import time
import yaml

MAX_CONFIG_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_LINES = 200_000
MAX_TREE_ENTRIES = 250_000
MAX_TREE_SECONDS = 2.0


def canonical(raw, base=None):
    if not isinstance(raw, (str, Path)) or not str(raw).strip() or chr(0) in str(raw) or '://' in str(raw):
        return None
    candidate = Path(str(raw).strip()).expanduser()
    if not candidate.is_absolute():
        if base is None:
            return None
        candidate = Path(base) / candidate
    try:
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return None


def overlaps(first, second):
    a, b = canonical(first), canonical(second)
    return None if a is None or b is None else a == b or a in b.parents or b in a.parents


def read_yaml(path):
    try:
        with Path(path).open('rb') as stream:
            content = stream.read(MAX_CONFIG_BYTES + 1)
        if len(content) > MAX_CONFIG_BYTES:
            return None
        value = yaml.safe_load(content.decode('utf-8-sig'))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeError, ValueError, yaml.YAMLError):
        return None


def manifest_conflict(path, target):
    try:
        with path.open('rb') as stream:
            content = stream.read(MAX_MANIFEST_BYTES + 1)
        if len(content) > MAX_MANIFEST_BYTES:
            return None
        lines = content.decode('utf-8-sig').splitlines()
        if len(lines) > MAX_MANIFEST_LINES:
            return None
        unknown = False
        for line in lines:
            raw = line.strip()
            if not raw or raw.startswith('#'):
                continue
            # Ultralytics only anchors ./ entries to the manifest. Other
            # relative entries use trainer CWD, unavailable from this helper.
            if not Path(raw).is_absolute() and not raw.startswith('./'):
                unknown = True
                continue
            resolved = canonical(raw, path.parent)
            if resolved is None:
                unknown = True
            elif overlaps(resolved, target):
                return True
            if resolved is not None:
                original = Path(raw) if Path(raw).is_absolute() else path.parent / raw
                labels = image_labels_path(original)
                if labels is not None and overlaps(labels, target):
                    return True
                if labels is None:
                    unknown = True
        return None if unknown else False
    except (OSError, UnicodeError, ValueError):
        return None


def image_labels_path(path):
    parts = list(Path(path).parts)
    if 'images' not in parts:
        return None
    parts[len(parts) - 1 - parts[::-1].index('images')] = 'labels'
    return Path(*parts)


def directory_link_conflict(path, target, evidence_cache=None):
    """Inspect names only, never image bytes, and bound expensive/unknown trees."""
    pending = [Path(path)]
    key = ('directory_links', str(path))
    if evidence_cache is not None and key in evidence_cache:
        return evidence_cache[key]
    visited = 0
    deadline = time.monotonic() + MAX_TREE_SECONDS
    try:
        while pending:
            with os.scandir(pending.pop()) as items:
                for item in items:
                    visited += 1
                    if visited > MAX_TREE_ENTRIES or time.monotonic() > deadline:
                        if evidence_cache is not None:
                            evidence_cache[key] = None
                        return None
                    if item.is_symlink():
                        # Even a link to an unrelated directory can contain a
                        # further link. Unknown links retain the deletion guard.
                        return True if overlaps(item.path, target) else None
                    if item.is_dir(follow_symlinks=False):
                        pending.append(Path(item.path))
        if evidence_cache is not None:
            evidence_cache[key] = False
        return False
    except (OSError, ValueError):
        return None


def data_path_conflict(raw, target, *, task_type='detect', evidence_cache=None):
    path, target = canonical(raw), canonical(target)
    if path is None or target is None:
        return None
    if overlaps(path, target):
        return True
    if task_type == 'classify':
        return directory_link_conflict(path, target, evidence_cache) if path.is_dir() else None
    if path.suffix.lower() not in ('.yaml', '.yml'):
        return None
    config = read_yaml(path)
    if config is None:
        return None
    root_raw = config.get('path')
    # An explicit relative root uses trainer CWD (or DATASETS_DIR), neither
    # established by the YAML location. Only an omitted root uses its parent.
    root = (canonical(root_raw) if isinstance(root_raw, (str, Path)) and Path(root_raw).is_absolute()
            else None) if root_raw else path.parent
    if root is not None and (root == target or target in root.parents):
        return True
    unknown = root is None or not config.get('train') or not config.get('val')
    for key in ('train', 'val', 'test', 'minival'):
        values = config.get(key)
        if values is None:
            continue
        values = values if isinstance(values, list) else [values]
        for value in values:
            resolved = canonical(value, root)
            if resolved is None:
                unknown = True
                continue
            if overlaps(resolved, target):
                return True
            if resolved.suffix.lower() == '.txt':
                result = manifest_conflict(resolved, target)
                if result is True:
                    return True
                unknown = unknown or result is None
            elif resolved.is_dir():
                result = directory_link_conflict(resolved, target, evidence_cache)
                if result is True:
                    return True
                unknown = unknown or result is None
                # Ultralytics derives labels by replacing the images path
                # component, so protect links in that sibling tree as well.
                original = Path(value) if Path(value).is_absolute() else root / value
                labels = image_labels_path(original)
                if labels is not None:
                    if overlaps(labels, target):
                        return True
                    if labels.is_dir():
                        result = directory_link_conflict(labels, target, evidence_cache)
                        if result is True:
                            return True
                        unknown = unknown or result is None
                else:
                    unknown = True
            elif not resolved.exists():
                unknown = True
    return None if unknown else False


def run_data_paths(run):
    paths = [run.data_path]
    command = run.command or []
    if isinstance(command, list):
        paths.extend(value[5:] for value in command if isinstance(value, str) and value.startswith('data='))
    snapshot = run.parameter_snapshot or {}
    if snapshot.get('data'):
        paths.append(snapshot['data'])
    return list(dict.fromkeys(str(path) for path in paths if path))


def local_run_conflict(run, target, *, task_type='detect', active=False, evidence_cache=None):
    paths = run_data_paths(run)
    unknown = not paths
    for raw in paths:
        result = data_path_conflict(raw, target, task_type=task_type, evidence_cache=evidence_cache)
        if result is True:
            return True
        unknown = unknown or result is None
    if active:
        # Resume may restore args.data from the checkpoint. Only actual trainer
        # args establish this value; never unpickle a checkpoint in the API.
        args_path = Path(run.run_dir) / 'args.yaml' if run.run_dir else None
        if args_path and args_path.is_file():
            args = read_yaml(args_path)
            result = data_path_conflict(args.get('data') if args else None, target, task_type=task_type, evidence_cache=evidence_cache)
            if result is True:
                return True
            unknown = unknown or result is None
        elif run.launch_mode == 'resume' or (run.parameter_snapshot or {}).get('resume'):
            unknown = True
    return None if unknown else False
