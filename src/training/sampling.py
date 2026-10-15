"""Shared rejection-sampling records, persistence, and counters.

Domains own rollout execution and their correctness score.  They do not own
the serialized training schema or JSONL persistence protocol.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, TextIO

from src.training.message_schema import (
    canonicalize_chat_messages,
    validate_training_record,
)
from src.utils.persistence import append_jsonl, atomic_jsonl


@dataclass
class SamplingStats:
    items: int = 0
    attempts: int = 0
    accepted: int = 0
    strict_accepted: int = 0
    repaired_accepted: int = 0
    skipped_existing: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)


def accepted_record(
    *,
    sample_id: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    sampling: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate the canonical accepted-rollout representation."""

    record: dict[str, Any] = {
        "sample_id": str(sample_id),
        **dict(metadata or {}),
        "messages": canonicalize_chat_messages(messages),
        "tools": list(tools),
        "sampling": dict(sampling),
    }
    validate_training_record(record, require_tool_calls=True)
    return record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL artifact with actionable line-number errors."""

    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"record at {path}:{line_number} must be an object")
            records.append(value)
    return records


def read_accepted_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and contract-check records before a sampling run resumes them."""

    records = read_jsonl(path)
    for line_number, record in enumerate(records, start=1):
        try:
            validate_training_record(record, require_tool_calls=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid accepted rollout at {path}:{line_number}: {exc}") from exc
    return records


class SamplingRecordWriter:
    """Thread-safe writer shared by all domain sampling loops."""

    def __init__(self, accepted_path: Path, attempts_path: Path, *, append: bool):
        self.accepted_path = accepted_path
        self.attempts_path = attempts_path
        self.append = append
        self._accepted: TextIO | None = None
        self._attempts: TextIO | None = None
        self._lock = Lock()

    def __enter__(self) -> "SamplingRecordWriter":
        self.accepted_path.parent.mkdir(parents=True, exist_ok=True)
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if self.append else "w"
        self._accepted = self.accepted_path.open(mode, encoding="utf-8")
        self._attempts = self.attempts_path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, *_: Any) -> None:
        if self._accepted is not None:
            self._accepted.close()
        if self._attempts is not None:
            self._attempts.close()

    def write_accepted(self, record: dict[str, Any]) -> None:
        validate_training_record(record, require_tool_calls=True)
        if self._accepted is None:
            raise RuntimeError("sampling writer is not open")
        with self._lock:
            append_jsonl(self._accepted, record)

    def write_attempt(self, record: dict[str, Any]) -> None:
        if self._attempts is None:
            raise RuntimeError("sampling writer is not open")
        with self._lock:
            append_jsonl(self._attempts, record)


def write_accepted_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    """Write validated accepted records when no diagnostics stream is needed."""

    def validated():
        for record in records:
            validate_training_record(record, require_tool_calls=True)
            yield record

    atomic_jsonl(path, validated())
