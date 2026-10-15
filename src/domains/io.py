"""Small domain-neutral record I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    """Read object records from a JSON array or JSONL file."""
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".jsonl":
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError(f"JSONL record must be an object: {path}")
                    yield record
            return
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f"JSON must contain a list of objects: {path}")
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"JSON record must be an object: {path}")
        yield record
