"""HomeBench manifests. Answers are never part of the public environment."""

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from src.domains.io import read_records

PROTOCOL = "homebench_counter_v1"
EXPORT_FIELDS = {
    "id",
    "input",
    "output",
    "home_id",
    "type",
    "category",
    "home_status_json",
    "methods_json",
    "source_split",
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Sample:
    record: dict
    home: dict

    @property
    def sample_id(self):
        return self.record["sample_id"]

    @property
    def instruction(self):
        return self.record["instruction"]

    @cached_property
    def digest(self):
        return digest({"record": self.record, "home": self.home})


def _decode_embedded(value, *, field, sample_id):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {field} JSON: {sample_id}") from exc
    return value


def _export_sample(row, home_cache):
    missing = EXPORT_FIELDS - row.keys()
    if missing:
        raise ValueError(f"missing HomeBench export fields: {sorted(missing)}")
    sample_id = row.get("id")
    instruction = row.get("input")
    commands = row.get("output")
    home_id = row.get("home_id")
    task_type = row.get("type")
    category = row.get("category")
    source_split = row.get("source_split")
    if not sample_id or not instruction or home_id is None or not task_type:
        raise ValueError(f"invalid HomeBench export sample: {sample_id!r}")
    if not isinstance(commands, str):
        raise TypeError(f"missing reference commands: {sample_id}")
    if category not in {"VM", "IM", "MM"}:
        raise ValueError(f"invalid HomeBench category: {sample_id}")
    if source_split not in {"train", "test", "validation", "val"}:
        raise ValueError(f"invalid HomeBench source split: {sample_id}")

    raw_context = (row.get("home_status_json"), row.get("methods_json"))
    cached = home_cache.get(home_id)
    if cached is not None:
        if raw_context != cached[0]:
            raise ValueError(f"inconsistent embedded home context: {home_id}")
        home = cached[1]
    else:
        home_status = _decode_embedded(
            raw_context[0], field="home_status_json", sample_id=sample_id
        )
        methods = _decode_embedded(raw_context[1], field="methods_json", sample_id=sample_id)
        if not isinstance(home_status, dict):
            raise TypeError(f"home_status_json must decode to an object: {sample_id}")
        if not isinstance(methods, list):
            raise TypeError(f"methods_json must decode to a list: {sample_id}")
        home = {"home_id": home_id, "home_status": home_status, "method": methods}
        home_cache[home_id] = (raw_context, home)

    # Normalize the portable/Hugging Face export to the internal task schema.
    # system_prompt is deliberately not exposed: the active harness owns its
    # prompt, while only public household state/methods become tool results.
    record = {
        "sample_id": str(sample_id),
        "domain": "homebench",
        "instruction": instruction,
        "task_type": task_type,
        "environment": {"protocol": PROTOCOL, "home_id": home_id},
        "target": {"commands": commands},
        "metadata": {
            "source_split": source_split,
            "category": category,
            "label_conflict": False,
        },
    }
    return record, home


def load(path, *, limit=None, **kwargs):
    path = Path(path)
    homes = None
    embedded_homes = {}
    seen = set()
    for index, row in enumerate(read_records(path)):
        if limit is not None and index >= limit:
            break
        if "sample_id" in row:
            if homes is None:
                homes_path = path.parent / "homes.jsonl"
                if not homes_path.is_file():
                    raise FileNotFoundError(
                        f"canonical HomeBench data requires sidecar: {homes_path}"
                    )
                homes = {r["home_id"]: r for r in read_records(homes_path)}
            record = row
            sid = record.get("sample_id")
            env = record.get("environment", {})
            if env.get("protocol") != PROTOCOL or env.get("home_id") not in homes:
                raise ValueError(f"unknown environment: {sid}")
            home = homes[env["home_id"]]
        else:
            record, home = _export_sample(row, embedded_homes)
            sid = record["sample_id"]
            env = record["environment"]
        if not sid or sid in seen or not record.get("instruction"):
            raise ValueError(f"invalid or duplicate sample at {path}:{index + 1}")
        if not isinstance(record.get("target", {}).get("commands"), str):
            raise TypeError(f"missing reference commands: {sid}")
        seen.add(sid)
        yield Sample(record, home)
