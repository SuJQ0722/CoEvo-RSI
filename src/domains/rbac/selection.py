"""Six-category gated selection for RBAC harnesses."""

from __future__ import annotations

from collections.abc import Mapping

from src.domains.base import EvaluationResult, SelectionDecision

QUALITY_METRICS = ("safe_ex", "ac_f1")
COST_METRICS = ("llm_rate", "rule_rate", "rule_accuracy")


def select(
    parent: EvaluationResult,
    child: EvaluationResult,
    config: Mapping[str, object],
) -> SelectionDecision:
    all_names = QUALITY_METRICS + COST_METRICS
    missing = [n for n in all_names if n not in parent.metrics or n not in child.metrics]
    if missing:
        raise ValueError(f"RBAC selection metrics are missing: {missing}")

    deltas = {n: float(child.metrics[n]) - float(parent.metrics[n]) for n in all_names}
    epsilon = float(config.get("epsilon", 1e-12))
    safe_ex_tolerance = float(config.get("safe_ex_tolerance", 0.0))
    ac_f1_tolerance = float(config.get("ac_f1_tolerance", 0.0))

    # Primary gate: safe_ex must not regress
    if deltas["safe_ex"] < -safe_ex_tolerance - epsilon:
        return SelectionDecision(False, "SafeEX regression exceeds tolerance", deltas)

    # Primary gate: ac_f1 must not regress
    if deltas["ac_f1"] < -ac_f1_tolerance - epsilon:
        return SelectionDecision(False, "AC-F1 regression exceeds tolerance", deltas)

    # Secondary: cost metrics should not regress (soft)
    # rule_accuracy: if child uses rules, accuracy shouldn't decrease
    if (
        child.metrics["rule_rate"] > epsilon
        and deltas["rule_accuracy"] < -epsilon
    ):
        return SelectionDecision(False, "rule accuracy regressed", deltas)

    # Acceptance: at least one improvement
    safe_ex_improves = deltas["safe_ex"] > epsilon
    ac_f1_improves = deltas["ac_f1"] > epsilon
    cost_improves = deltas["llm_rate"] < -epsilon  # lower llm_rate is better

    if safe_ex_improves:
        return SelectionDecision(True, "SafeEX improved", deltas)
    if ac_f1_improves:
        return SelectionDecision(True, "AC-F1 improved without SafeEX regression", deltas)
    if cost_improves:
        return SelectionDecision(True, "LLM call rate decreased without quality regression", deltas)

    return SelectionDecision(False, "no improvement in SafeEX, AC-F1, or LLM call rate", deltas)
