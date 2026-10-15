"""Stable construction adapter for the RBAC harness contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .harness import RBACHarness


def build_harness(policy_rows: Sequence[Mapping[str, Any]]) -> RBACHarness:
    return RBACHarness(policy_rows)

