"""Declarative multi-metric selection shared across domains."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from .base import EvaluationResult, SelectionDecision


def quality_names(selection):
    names = tuple(selection.get("quality_metrics") or ())
    if not names:
        raise ValueError("selection.quality_metrics must declare the quality ordering")
    primary = selection.get("primary_quality_metric")
    if primary:
        if primary not in names:
            raise ValueError("selection.primary_quality_metric must be a quality metric")
        names = (primary, *(name for name in names if name != primary))
    return names


def ranking_groups(selection):
    """Ordered objectives; each group contributes a possibly weighted mean."""
    groups = selection.get("ranking_metric_groups")
    if groups is None:
        return tuple((name,) for name in quality_names(selection))
    if not isinstance(groups, (list, tuple)) or not groups or any(
        not isinstance(group, (list, tuple)) or not group
        or any(not isinstance(name, str) or not name for name in group)
        for group in groups
    ):
        raise ValueError("selection.ranking_metric_groups must contain nonempty metric lists")
    return groups


def quality_score(scores, selection):
    values = []
    weights = selection.get("ranking_metric_weights", {})
    if not isinstance(weights, Mapping):
        raise TypeError("selection.ranking_metric_weights must be a metric-to-weight mapping")
    for group in ranking_groups(selection):
        missing = [name for name in group if name not in scores]
        if missing:
            raise ValueError(f"ranking metrics are missing: {missing}")
        raw = [float(scores[name]) for name in group]
        if not all(math.isfinite(value) for value in raw):
            raise ValueError("non-finite quality score")
        group_weights = [float(weights.get(name, 1.0)) for name in group]
        if any(not math.isfinite(weight) or weight <= 0 for weight in group_weights):
            raise ValueError("ranking metric weights must be finite and positive")
        values.append(
            sum(value * weight for value, weight in zip(raw, group_weights))
            / sum(group_weights)
        )
    return tuple(values)


def split_quality_scores(payload, names):
    """Expose official split metrics without treating an absent split as zero."""
    return {
        f"{split}.{name}": float(metrics[name])
        for split, metrics in (payload.get("by_split") or {}).items()
        if metrics.get("total", 0) > 0
        for name in names
        if metrics.get(name) is not None
    }


def select_any_improves_no_regression(
    parent: EvaluationResult,
    child: EvaluationResult,
    config: Mapping[str, Any],
) -> SelectionDecision:
    metrics = list(config.get("quality_metrics") or parent.metrics)
    tolerances = dict(config.get("tolerances") or {})
    epsilon = float(config.get("epsilon", 1e-12))
    missing = [name for name in metrics if name not in parent.metrics or name not in child.metrics]
    if missing:
        raise ValueError(f"selection metrics are missing: {missing}")
    deltas = {
        name: float(child.metrics[name]) - float(parent.metrics[name])
        for name in metrics
    }
    improved = any(delta > epsilon for delta in deltas.values())
    regressions = {
        name: delta
        for name, delta in deltas.items()
        if delta < -float(tolerances.get(name, 0.0)) - epsilon
    }
    accepted = improved and not regressions
    reason = (
        "at least one quality metric improved and none regressed beyond tolerance"
        if accepted
        else "no quality metric improved"
        if not improved
        else f"quality regressions exceed tolerance: {regressions}"
    )
    return SelectionDecision(accepted, reason, deltas)
