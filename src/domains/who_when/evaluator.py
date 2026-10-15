"""Who&When official metric adapter."""

from dataclasses import asdict
from typing import Any

from src.domains.base import EvaluationResult
from src.domains.selection import split_quality_scores
from src.domains.who_when.runtime.evaluation import evaluate as evaluate_official


def evaluate(records: list[dict[str, Any]], **_: Any) -> EvaluationResult:
    result = evaluate_official(records)
    payload = asdict(result)
    metrics = {
        key: float(payload[key])
        for key in (
            "agent_accuracy",
            "step_accuracy",
            "both_accuracy",
            "agent_accuracy_completed",
            "step_accuracy_completed",
            "both_accuracy_completed",
            "coverage",
            "error_rate",
        )
    }
    metrics.update(
        {
            "agent_accuracy_all": metrics["agent_accuracy"],
            "step_accuracy_all": metrics["step_accuracy"],
            "both_accuracy_all": metrics["both_accuracy"],
        }
    )
    metrics.update(split_quality_scores(payload, ("agent_accuracy", "step_accuracy")))
    return EvaluationResult(metrics=metrics, diagnostics=payload)
