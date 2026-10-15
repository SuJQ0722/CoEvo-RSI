"""Load the seed, the final evolved (``generated``) or a candidate RBAC harness."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping, Sequence
from typing import Any

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def build_harness(name: str, policy_rows: Sequence[Mapping[str, Any]]) -> Any:
    if not _NAME.fullmatch(name):
        raise ValueError(f"invalid harness name: {name!r}")
    module_name = (
        f"harnesses.rbac.{name}"
        if name in {"initial_harness", "system_zero", "generated"}
        else f"harnesses.rbac.candidates.{name}"
    )
    if name == "system_zero":
        module_name = "harnesses.rbac.initial_harness"
    module = importlib.import_module(module_name)
    builder = getattr(module, "build_harness", None)
    if not callable(builder):
        raise TypeError(f"{module_name} must expose build_harness(policy_rows)")
    harness = builder(policy_rows)
    required = ("route", "messages", "call_llm", "parse_access")
    missing = [member for member in required if not callable(getattr(harness, member, None))]
    if missing:
        raise TypeError(f"{module_name} harness is missing: {', '.join(missing)}")
    if name not in {"initial_harness", "system_zero"}:
        setattr(harness, "name", name)
    return harness
