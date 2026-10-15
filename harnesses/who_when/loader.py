"""Load initial_harness, the final evolved harness, or a candidate by name."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path
from typing import Any

from src.traces import AgentTrace

_HARNESS_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Ensure the harnesses root is importable
_HARNESSES_ROOT = Path(__file__).resolve().parent
if str(_HARNESSES_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_HARNESSES_ROOT.parent))


def build_harness(name: str, trace: AgentTrace) -> Any:
    """Build a harness instance by name.

    ``initial_harness`` is the seed, ``generated`` is the final evolved harness,
    and any other name refers to a candidate under ``candidates/``.
    """
    if not _HARNESS_NAME.fullmatch(name):
        raise ValueError(f"invalid harness name: {name!r}")

    if name in ("initial_harness", "system_zero"):
        module_name = "harnesses.who_when.initial_harness"
    elif name == "generated":
        module_name = "harnesses.who_when.generated"
    else:
        module_name = f"harnesses.who_when.candidates.{name}"

    module = importlib.import_module(module_name)
    builder = getattr(module, "build_harness", None)
    if not callable(builder):
        raise TypeError(f"{module_name} must expose build_harness(trace)")

    harness = builder(trace)
    required = (
        "tools",
        "initial_messages",
        "handle_tool_call",
        "is_done",
        "result",
        "inspected_step_ids",
    )
    missing = [attr for attr in required if not hasattr(harness, attr)]
    if missing:
        raise TypeError(f"{module_name} harness is missing: {', '.join(missing)}")

    if name not in ("initial_harness", "system_zero"):
        try:
            setattr(harness, "name", name)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                f"{module_name} harness identity cannot be bound to {name!r}"
            ) from exc

    return harness
