"""Who&When data model and loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from src.traces import AgentTrace

from src.domains.io import read_records


class WhoWhenDataLoader:
    """Load complete Who&When trajectories."""

    def __init__(self, path: Path, limit: int | None = None):
        self.path = path
        self.limit = limit

    def __iter__(self) -> Iterator[AgentTrace]:
        emitted = 0
        for line_number, record in enumerate(read_records(self.path), start=1):
            trace = AgentTrace.from_record(record)
            if not trace.sample_id or not trace.task or not trace.steps:
                raise ValueError(
                    f"{self.path}:{line_number} is not a complete Who&When trace"
                )
            yield trace
            emitted += 1
            if self.limit is not None and emitted >= self.limit:
                return


def load(path: Path, *, limit: int | None = None, **_: object) -> WhoWhenDataLoader:
    return WhoWhenDataLoader(path, limit)
