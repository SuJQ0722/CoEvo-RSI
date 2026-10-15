"""Validation and idempotent copying for the shared evidence envelope."""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Iterable

from src.utils.persistence import atomic_jsonl

from .io import read_records

SCHEMA_VERSION = "coevox.evidence.v1"


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result |= _keys(child)
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for child in value:
            result |= _keys(child)
        return result
    return set()


def write_records(records: Iterable[dict[str, Any]], output: Path) -> Path:
    records = iter(records)
    try:
        first = next(records)
    except StopIteration as exc:
        raise ValueError("normalized evidence is empty") from exc
    atomic_jsonl(output, chain([first], records))
    return output


def copy_if_normalized(
    source: Path,
    output: Path,
    *,
    domain: str,
    forbidden_keys: set[str] | None = None,
) -> bool:
    """Copy a normalized JSON/JSONL source after validating its boundary."""
    if not source.is_file():
        return False
    records = list(read_records(source))
    if not records or records[0].get("schema_version") != SCHEMA_VERSION:
        return False
    forbidden_keys = forbidden_keys or set()
    for index, record in enumerate(records, start=1):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"mixed evidence schema at record {index}: {source}")
        if record.get("domain") != domain:
            raise ValueError(
                f"evidence domain mismatch at record {index}: "
                f"expected {domain!r}, got {record.get('domain')!r}"
            )
        leaked = sorted(_keys(record) & forbidden_keys)
        if leaked:
            raise ValueError(f"evidence contains forbidden keys: {leaked}")
    write_records(records, output)
    return True
