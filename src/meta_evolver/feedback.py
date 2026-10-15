"""Shared machine-readable feedback for every rejected harness attempt."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.persistence import atomic_json


def build_failure_feedback(
    *,
    status: str,
    reason: str,
    causes: list[str],
    evidence: dict[str, Any] | None = None,
    retryable: bool = True,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "selection_reason": reason,
        "failure_causes": list(dict.fromkeys(causes)),
        "evidence": evidence or {},
        "retryable": retryable,
        "next_round_guidance": [
            "Keep the current parent as the mutation base.",
            "Use this rejection evidence together with the prior analysis.",
            "Do not repeat the rejected change without new rollout evidence.",
        ],
        **(details or {}),
    }


def write_failure_feedback(path: Path, payload: dict[str, Any]) -> Path:
    atomic_json(path, payload)
    return path
