"""Portable dataset file operations; embedded unchanged into the standalone Agent."""
from __future__ import annotations
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import tarfile

MAX_BYTES = 50 * 1024**3
MAX_FILES = 200_010
MARKER = ".yolo-platform-dataset.json"


def report_progress(check, stage, processed=0, total=None, unit='files', current_file=''):
    """Optional progress hook; legacy callers keep the same cancellation callback."""
    report = getattr(check, 'progress', None)
    if report:
        report({'stage': stage, 'processed': processed, 'total': total, 'unit': unit,
                'current_file': str(current_file)[-400:]})


def digest(path, check=None, stage=None):
    value = hashlib.sha256()
    processed = 0
    total = Path(path).stat().st_size if stage else None
    if stage:
        report_progress(check, stage, 0, total, 'bytes')
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            if check:
                check()
            value.update(chunk)
            processed += len(chunk)
            if stage:
                report_progress(check, stage, processed, total, 'bytes')
    return value.hexdigest()


def inside(path, roots, create=False):
    raw = Path(path).expanduser()
    resolved = raw.resolve()
    if not raw.is_absolute() or raw != resolved:
        raise ValueError("目录必须为可解析的绝对路径，不能包含符号链接")
    allowed = [Path(root).resolve() for root in roots]
    if not any(root == resolved or root in resolved.parents for root in allowed):
        raise ValueError("目录不在数据集允许范围内，请核对服务器配置")
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def check_root(path, roots, create=False):
    root = inside(path, roots, create)
    if not root.is_dir():
        raise ValueError("目录不存在或不是文件夹")
    # Exercise actual filesystem permissions, rather than only reporting os.access.
    import tempfile
    with tempfile.TemporaryFile(dir=root):
        pass
    usage = shutil.disk_usage(root)
    return {"resolved_path": str(root), "free_bytes": usage.free, "total_bytes": usage.total,
            "status": "ready"}


def scan(root, check=None, stage="scan"):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("数据集目录不存在或为符号链接")
    files, total = [], 0
    report_progress(check, stage)
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError("数据集不能包含符号链接目录")
        for name in sorted(names):
            if name.endswith('.cache') or name in {MARKER, '.dataset-content.json'}:
                continue
            path = Path(directory) / name
            if path.is_symlink() or not path.is_file():
                raise ValueError("数据集不能包含符号链接或特殊文件")
            relative = path.relative_to(root).as_posix()
            safe_member(relative)
            size = path.stat().st_size
            total += size
            if total > MAX_BYTES or len(files) >= MAX_FILES:
                raise ValueError("数据集超过 50 GB 或 20 万文件限制")
            if check:
                check()
            report_progress(check, stage, len(files), current_file=relative)
            before = path.stat()
            sha = digest(path, check)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError("数据集在校验中发生修改，请停止修改后重试")
            files.append({"path": relative, "size": size, "sha256": sha})
    report_progress(check, stage, len(files), len(files))
    files.sort(key=lambda x: x['path'])
    signature = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"files": files, "total_bytes": total, "fingerprint": signature}


def safe_member(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or '\\' in name or any(part in {'', '.', '..'} or part.startswith('.') for part in name.split('/')) or any(ord(c) < 32 for c in name):
        raise ValueError("数据文件相对路径无效")
    return p


def pack(root, archive, check=None):
    manifest = scan(root, check)
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_suffix('.tmp')
    processed = 0
    report_progress(check, 'pack', 0, manifest['total_bytes'], 'bytes')
    class Reader:
        def __init__(self, stream, name):
            self.stream, self.name = stream, name
        def read(self, size=-1):
            nonlocal processed
            data = self.stream.read(size)
            processed += len(data)
            if check:
                check()
            report_progress(check, 'pack', processed, manifest['total_bytes'], 'bytes', self.name)
            return data
    with tarfile.open(temporary, 'w') as tar:
        for file in manifest['files']:
            if check:
                check()
            path = Path(root) / file['path']
            info = tar.gettarinfo(str(path), arcname=file['path'])
            if not info.isfile():
                raise ValueError('数据集不能包含符号链接或特殊文件')
            with path.open('rb') as stream:
                tar.addfile(info, Reader(stream, file['path']))
    # Reject concurrent source changes before handing a snapshot to another node.
    if scan(root, check, stage='source_verify')['fingerprint'] != manifest['fingerprint']:
        temporary.unlink(missing_ok=True)
        raise ValueError("源数据在导出期间发生修改")
    temporary.replace(archive)
    return {**manifest, 'archive_size': archive.stat().st_size, 'archive_sha256': digest(archive, check, stage='archive_hash')}


def unpack(archive, target, manifest, check=None):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    entries = {item['path']: item for item in manifest['files']}
    if len(entries) != len(manifest['files']) or len(entries) > MAX_FILES or sum(i['size'] for i in entries.values()) > MAX_BYTES:
        raise ValueError("数据清单大小或路径无效")
    seen = set()
    processed = 0
    total = sum(i['size'] for i in entries.values())
    report_progress(check, 'unpack', 0, total, 'bytes')
    with tarfile.open(archive, 'r:') as tar:
        for item in tar:
            if check:
                check()
            safe_member(item.name)
            spec = entries.get(item.name)
            if not item.isfile() or item.name in seen or not spec or item.size != spec['size']:
                raise ValueError("归档文件类型、大小或清单不匹配")
            seen.add(item.name)
            path = target / item.name
            path.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(item)
            report_progress(check, 'unpack', processed, total, 'bytes', item.name)
            with path.open('xb') as output:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(chunk)
                    if check:
                        check()
            if digest(path, check) != spec['sha256']:
                raise ValueError("数据文件 SHA-256 校验失败")
            processed += item.size
            report_progress(check, 'unpack', processed, total, 'bytes', item.name)
    if seen != set(entries):
        raise ValueError("归档缺少数据文件")


def publish(archive, root, name, manifest, identity, config_text, check=None):
    if not name or name.startswith('.') or '/' in name or '\\' in name:
        raise ValueError("数据集名称无效")
    if not re.fullmatch(r'[0-9a-f]{32}', identity):
        raise ValueError('操作编号无效')
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('目标根目录不可用')
    destination = root / name
    if destination.exists() or destination.is_symlink():
        try:
            metadata = json.loads((destination / MARKER).read_text())
        except (OSError, ValueError):
            metadata = {}
        if metadata.get('operation_id') == identity:
            actual = scan(destination, check)
            if actual['fingerprint'] != metadata.get('fingerprint'):
                raise ValueError('已发布的目录内容发生变化，请人工核对')
            return {'path': str(destination), 'fingerprint': actual['fingerprint'], 'total_bytes': actual['total_bytes'], 'config_sha256': digest(destination / 'data.yaml'), 'already_published': True}
        raise ValueError("目标存在同名目录，不允许覆盖")
    work = root / ('.dataset-publish-' + identity)
    if work.is_symlink():
        raise ValueError('临时目录不能是符号链接')
    if work.exists():
        shutil.rmtree(work)
    required = manifest['total_bytes'] + 100 * 1024**2
    if shutil.disk_usage(root).free < required:
        raise ValueError("目标磁盘空间不足")
    try:
        unpack(archive, work, manifest, check)
        if config_text is not None:
            (work / 'data.yaml').write_text(config_text, encoding='utf-8')
        actual = scan(work, check, stage='target_verify')
        (work / MARKER).write_text(json.dumps({'version': 1, 'name': name, 'operation_id': identity,
                                               'fingerprint': actual['fingerprint']}, ensure_ascii=False), encoding='utf-8')
        if check:
            check('committing')
        if destination.exists() or destination.is_symlink():
            raise ValueError("目标目录已出现，拒绝覆盖")
        work.rename(destination)
        return {'path': str(destination), 'fingerprint': actual['fingerprint'], 'total_bytes': actual['total_bytes'], 'config_sha256': digest(destination / 'data.yaml')}
    finally:
        if work.exists():
            shutil.rmtree(work)


def remove_dataset(path, roots, expected_fingerprint, check=None):
    path = inside(path, roots)
    allowed = [Path(root).resolve() for root in roots]
    if any(path == root or path in root.parents for root in allowed) or not path.parent.is_dir():
        raise ValueError("不能删除根目录或离线磁盘上的数据")
    if not expected_fingerprint:
        raise ValueError("缺少内容指纹，拒绝删除")
    quarantine = path.parent / ('.dataset-delete-' + hashlib.sha256(
        (str(path) + expected_fingerprint).encode()).hexdigest()[:32])
    if quarantine.is_symlink():
        raise ValueError("删除暂存目录不能为符号链接")
    if quarantine.exists():
        # A crash may have interrupted rmtree. Only this path+snapshot's private
        # quarantine may be resumed; a newly created source is never removed.
        if check:
            check('committing')
        shutil.rmtree(quarantine)
        return {'deleted': True, 'recovered': True}
    if not path.exists():
        return {'already_missing': True}
    if scan(path, check, stage='delete_verify')['fingerprint'] != expected_fingerprint:
        raise ValueError("数据集内容已变化，拒绝删除，请重新校验")
    if check:
        check('committing')
    report_progress(check, 'remove')
    path.rename(quarantine)
    shutil.rmtree(quarantine)
    return {'deleted': True}
