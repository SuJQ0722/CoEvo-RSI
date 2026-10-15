"""Shared policy rejection-sampling orchestration.

Domains own rollout execution, correctness checks, and training-record creation.
This module owns the common attempt schedule: run bounded policy attempts until
enough accepted records are collected.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.training.sampling import SamplingStats

AttemptResult = tuple[dict[str, Any] | None, dict[str, Any]]
AttemptRunner = Callable[[int], AttemptResult]


@dataclass
class SamplingProgress:
    """Apply one shared accounting/logging schema to a completed sample."""

    stats: SamplingStats
    existing_records: int = 0
    _outcomes: Counter[str] = field(default_factory=Counter)

    def complete(
        self,
        records: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        self.stats.items += 1
        self.stats.attempts += len(diagnostics)
        self.stats.accepted += len(records)
        for record in records:
            tier = str((record.get("sampling") or {}).get("quality_tier") or "")
            if tier == "strict":
                self.stats.strict_accepted += 1
            else:
                self.stats.repaired_accepted += 1
        for diagnostic in diagnostics:
            self._outcomes[str(diagnostic.get("reason") or "unknown")] += 1
        self.stats.outcome_counts = dict(self._outcomes)
        return [
            {
                **diagnostic,
                "items": self.stats.items,
                "attempts_total": self.stats.attempts,
                "accepted_new": self.stats.accepted,
                "accepted_total": self.existing_records + self.stats.accepted,
            }
            for diagnostic in diagnostics
        ]


def policy_attempts(
    *,
    attempts_per_item: int,
    accepted_per_item: int,
    policy_attempt: AttemptRunner,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run bounded policy attempts until enough records are accepted.

    The shared phase label keeps diagnostics and accounting consistent while
    callbacks retain all domain-specific rollout and acceptance semantics.
    """

    if attempts_per_item < 1 or accepted_per_item < 1:
        raise ValueError("attempts and accepted count must be positive")

    accepted: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for attempt in range(1, attempts_per_item + 1):
        record, diagnostic = policy_attempt(attempt)
        diagnostics.append(
            {
                **diagnostic,
                "attempt": attempt,
                "phase": "policy",
                "accepted": record is not None,
            }
        )
        if record is not None:
            accepted.append(record)
            if len(accepted) >= accepted_per_item:
                break
    return accepted, diagnostics
