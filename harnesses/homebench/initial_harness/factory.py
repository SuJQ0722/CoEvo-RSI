"""Stable factory for the HomeBench runtime contract."""

from __future__ import annotations

from typing import Any

from .harness import HomeBenchHarness
from .settings import HarnessSettings


def build_agent(settings: HarnessSettings, *, client: Any) -> HomeBenchHarness:
    return HomeBenchHarness(settings, client)
