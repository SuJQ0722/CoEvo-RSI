"""Durable replacement and incremental JSONL writes, without owning caller locks.

Atomic replacement publishes a complete file or leaves the old one intact when
serialization fails. Append preserves completed records across interruptions;
the caller owns the open stream and any thread/process synchronization. Neither
operation provides a transaction across multiple files or validates task data.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4


@contextmanager
def atomic_text(path: Path) -> Iterator[TextIO]:
    """Replace UTF-8 text using a unique sibling file, fsync and rename.

    Preserve existing permission bits. Writers still need an external lock for
    read/modify/write operations; concurrent complete replacements are last-wins.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            yield stream
            if path.exists():
                temporary.chmod(path.stat().st_mode & 0o777)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any, *, sort_keys: bool = False) -> None:
    with atomic_text(path) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=sort_keys)
        stream.write("\n")


def append_jsonl(
    stream: TextIO,
    value: Any,
    *,
    flush: bool = True,
    durable: bool = False,
    sort_keys: bool = False,
) -> None:
    """Serialize before writing so a bad record cannot leave a partial JSON line.

    Flush by default for resumable progress logs; durable additionally fsyncs.
    A process/disk failure during the write may still leave a torn last line;
    domain readers retain their explicit malformed-record/recovery policies.
    """
    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=sort_keys) + "\n")
    if flush or durable:
        stream.flush()
    if durable:
        os.fsync(stream.fileno())


def atomic_jsonl(path: Path, records: Iterable[Any], *, sort_keys: bool = False) -> int:
    count = 0
    with atomic_text(path) as stream:
        for record in records:
            append_jsonl(stream, record, flush=False, sort_keys=sort_keys)
            count += 1
    return count
