"""Validated settings for the seed HomeBench harness."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HarnessSettings:
    config: dict[str, Any]

    @property
    def prompt_mode(self) -> str:
        mode = str(self.config.get("harness", {}).get("prompt_mode", "icl"))
        if mode not in {"icl", "zero_shot"}:
            raise ValueError("harness.prompt_mode must be icl or zero_shot")
        return mode
