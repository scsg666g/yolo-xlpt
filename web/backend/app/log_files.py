"""Read recent output without loading a complete, potentially large log."""
import os
import stat
from pathlib import Path


def tail_log(path: str | Path, count: int, *, max_bytes: int = 2 * 1024 * 1024) -> list[str]:
    descriptor = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            metadata = os.fstat(handle.fileno())
            if not stat.S_ISREG(metadata.st_mode):
                return []
            position = metadata.st_size
            chunks = []
            read_bytes = breaks = 0
            while position > 0 and read_bytes < max_bytes and breaks <= count:
                size = min(position, 65536, max_bytes - read_bytes)
                position -= size
                handle.seek(position)
                block = handle.read(size)
                chunks.append(block)
                read_bytes += len(block)
                breaks += block.count(b"\n")
            content = b"".join(reversed(chunks))
            if position > 0:
                # The leading fragment is not a whole line (or UTF-8 character).
                boundary = content.find(b"\n")
                content = content[boundary + 1:] if boundary >= 0 else content
            return content.decode("utf-8", errors="replace").splitlines()[-count:]
    except OSError:
        return []
    finally:
        if descriptor is not None:
            os.close(descriptor)
