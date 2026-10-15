"""Prompt resources and deterministic initial-message construction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .context import build_household_index
from .settings import HarnessSettings

RESOURCE_DIR = Path(__file__).resolve().parent


def _read_resource(name: str) -> str:
    return (RESOURCE_DIR / name).read_text(encoding="utf-8").strip()


def build_user_prompt(
    instruction: str,
    home: Mapping[str, Any],
    *,
    prompt_mode: str,
) -> str:
    sections = [
        "<household_index>\n"
        "Available room and device names (details must be retrieved with tools):\n"
        + json.dumps(
            build_household_index(home),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n</household_index>"
    ]
    if prompt_mode == "icl":
        sections.append(
            "<icl_examples>\n"
            "These examples show output syntax for other household states; never copy "
            "their device availability or values into the current task.\n"
            + _read_resource("icl_examples.txt")
            + "\n</icl_examples>"
        )
    sections.append("<User instructions>\n" + str(instruction))
    return "\n\n".join(sections)


def initial_messages(
    settings: HarnessSettings,
    instruction: str,
    home: Mapping[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _read_resource("prompt.txt")},
        {
            "role": "user",
            "content": build_user_prompt(
                instruction,
                home,
                prompt_mode=settings.prompt_mode,
            ),
        },
    ]
