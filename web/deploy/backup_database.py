#!/usr/bin/env python3
"""Create a consistent, private SQLite backup without stopping the platform."""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[1]


def resolve(value: Path) -> Path:
    return (value if value.is_absolute() else WEB_ROOT / value).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="安全备份当前项目自己的 SQLite 数据库")
    parser.add_argument("--source", type=Path, default=Path("data/platform.db"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/backups"))
    args = parser.parse_args()
    source, output_dir = resolve(args.source), resolve(args.output_dir)
    if not source.is_file():
        parser.error(f"数据库不存在：{source}")
    output_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination = output_dir / f"{source.name}.backup-{stamp}"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with sqlite3.connect(source.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(temporary) as dst:
            src.backup(dst)
            result = dst.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                raise RuntimeError("备份完整性检查失败")
        os.chmod(temporary, 0o600)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
