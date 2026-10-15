"""Load normalized binary RBAC records from CSV or JSON/JSONL."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class RBACSample:
    sample_id: str
    instruction: str
    test_role: str
    valid: int
    output: str = ""
    test_type: str = "train"
    required_role: str = ""
    db_id: str = ""
    difficulty: str = "unknown"
    question_group_id: str = ""
    source_split: str = ""
    gold_sql: str | None = None


def _bit(value: object, *, path: Path, row_number: int) -> int:
    try:
        bit = int(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid access label at {path}:{row_number}: {value!r}") from exc
    if bit not in {0, 1}:
        raise ValueError(f"access label must be 0/1 at {path}:{row_number}")
    return bit


def _sample(row: dict[str, Any], *, path: Path, index: int) -> RBACSample:
    instruction = str(row.get("instruction") or "").strip()
    test_role = str(row.get("test_role") or "").strip()
    if not instruction or not test_role:
        raise ValueError(f"empty instruction/test_role at {path}:{index + 1}")
    target = row.get("target") or {}
    target_output = target.get("normalized", "") if isinstance(target, dict) else ""
    return RBACSample(
        sample_id=str(row.get("sample_id") or f"{path.stem}:{index:06d}"),
        instruction=instruction,
        test_role=test_role,
        valid=_bit(row.get("valid"), path=path, row_number=index + 1),
        output=str(row.get("output") or target_output),
        test_type=str(row.get("test_type") or "train"),
        required_role=str(row.get("role") or ""),
        db_id=str(row.get("db_id") or ""),
        difficulty=str(row.get("difficulty") or "unknown"),
        question_group_id=str(row.get("question_group_id") or ""),
        source_split=str(row.get("source_split") or ""),
        gold_sql=row.get("gold_sql") or None,
    )


def _json_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        if path.suffix.lower() == ".jsonl":
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid RBAC JSONL at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"RBAC JSONL record must be an object: {path}:{line_number}")
                yield row
            return
        rows = json.load(stream)
    if not isinstance(rows, list):
        raise ValueError(f"RBAC JSON must contain a list of objects: {path}")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"RBAC JSON record must be an object: {path}:{index}")
        yield row


def load(path: Path, *, limit: int | None = None, **_: Any) -> Iterator[RBACSample]:
    path = path.resolve()
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {"instruction", "test_role", "valid"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"RBAC CSV is missing columns {sorted(missing)}: {path}")
            for index, row in enumerate(reader, start=1):
                if limit is not None and index > limit:
                    return
                yield _sample(dict(row), path=path, index=index)
        return
    if suffix not in {".json", ".jsonl"}:
        raise ValueError(f"unsupported RBAC data format {suffix!r}: {path}")
    for index, row in enumerate(_json_rows(path), start=1):
        if limit is not None and index > limit:
            return
        yield _sample(row, path=path, index=index)


def load_policy_rows(path: Path) -> list[dict[str, Any]]:
    """Load only the fields a harness may compile from the training split."""
    return [
        {
            "instruction": sample.instruction,
            "role": sample.required_role,
            "test_role": sample.test_role,
            "valid": sample.valid,
        }
        for sample in load(path)
    ]
