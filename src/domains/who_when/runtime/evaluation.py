"""Evaluate failure attribution using deterministic final-answer matching."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class AttributionDecision:
    agent_correct: bool
    step_correct: bool


class AttributionJudge(Protocol):
    def judge(self, record: dict[str, Any]) -> AttributionDecision: ...


@dataclass(frozen=True)
class AttributionSlice:
    total: int
    completed_total: int
    predictions_parsed: int
    agent_labeled_total: int
    completed_agent_labeled_total: int
    correct_agent: int
    correct_step: int
    correct_both: int
    agent_accuracy: float
    step_accuracy: float
    both_accuracy: float
    agent_accuracy_completed: float
    step_accuracy_completed: float
    both_accuracy_completed: float
    coverage: float
    error_count: int
    error_rate: float
    max_turn_failures: int
    context_overflow_failures: int
    no_tool_call_failures: int
    avg_turns: float
    avg_inspected_steps: float
    token_usage_records: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    avg_max_context_chars: float
    avg_cumulative_context_chars: float
    avg_repeated_inspections: float
    error_counts: dict[str, int]


@dataclass(frozen=True)
class WhoAndWhenMetrics:
    protocol: str
    total: int
    completed_total: int
    predictions_parsed: int
    agent_labeled_total: int
    completed_agent_labeled_total: int
    correct_agent: int
    correct_step: int
    correct_both: int
    agent_accuracy: float
    step_accuracy: float
    both_accuracy: float
    agent_accuracy_completed: float
    step_accuracy_completed: float
    both_accuracy_completed: float
    coverage: float
    error_count: int
    error_rate: float
    max_turn_failures: int
    context_overflow_failures: int
    no_tool_call_failures: int
    avg_turns: float
    avg_inspected_steps: float
    token_usage_records: int
    avg_prompt_tokens: float
    avg_completion_tokens: float
    avg_total_tokens: float
    avg_max_context_chars: float
    avg_cumulative_context_chars: float
    avg_repeated_inspections: float
    error_counts: dict[str, int]
    by_split: dict[str, dict[str, Any]]


class StrictAttributionJudge:
    """Deterministic scorer for parsed final-answer predictions."""

    def judge(self, record: dict[str, Any]) -> AttributionDecision:
        prediction = _official_prediction(record)
        if prediction is None:
            return AttributionDecision(False, False)
        gold = record.get("gold") or {}
        predicted_agent, predicted_step = prediction
        return AttributionDecision(
            _normalize_agent(gold.get("responsible_agent")) == _normalize_agent(predicted_agent),
            _normalize_step(gold.get("decisive_step")) == _normalize_step(predicted_step),
        )


def _normalize_agent(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def _normalize_step(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _benchmark_split(record: dict[str, Any]) -> str:
    value = record.get("benchmark_split")
    if value:
        return str(value)
    sample_id = str(record.get("sample_id") or "")
    parts = sample_id.split(":", 2)
    if len(parts) == 3 and parts[0] == "who_and_when":
        return parts[1]
    return str(record.get("source_dataset") or "validation")


def _official_prediction(record: dict[str, Any]) -> tuple[str, str] | None:
    prediction = record.get("prediction")
    if not isinstance(prediction, dict):
        return None
    agent = str(prediction.get("responsible_agent") or "").strip()
    evidence = prediction.get("evidence_step_ids")
    if not isinstance(evidence, list) or not evidence:
        return None
    gold_agent = str((record.get("gold") or {}).get("responsible_agent") or "").strip()
    if gold_agent and not agent:
        return None
    step = str(evidence[0]).strip()
    return (agent, step) if step else None


def _evaluate_attribution_slice(
    records: list[dict[str, Any]],
    decisions: dict[int, AttributionDecision],
) -> AttributionSlice:
    correct_agent = correct_step = correct_both = parsed = 0
    completed_correct_agent = completed_correct_step = completed_correct_both = 0
    completed_records = [record for record in records if not record.get("error")]
    agent_labeled_total = sum(
        bool(str((record.get("gold") or {}).get("responsible_agent") or "").strip())
        for record in records
    )
    completed_agent_labeled_total = sum(
        bool(str((record.get("gold") or {}).get("responsible_agent") or "").strip())
        for record in completed_records
    )
    for record in records:
        if _official_prediction(record) is None:
            continue
        parsed += 1
        decision = decisions[id(record)]
        usable = not record.get("error")
        correct_step += int(usable and decision.step_correct)
        completed_correct_step += int(usable and decision.step_correct)
        if str((record.get("gold") or {}).get("responsible_agent") or "").strip():
            correct_agent += int(usable and decision.agent_correct)
            correct_both += int(usable and decision.agent_correct and decision.step_correct)
            completed_correct_agent += int(usable and decision.agent_correct)
            completed_correct_both += int(
                usable and decision.agent_correct and decision.step_correct
            )
    total = len(records)
    completed_total = len(completed_records)
    errors = Counter(
        str(record.get("error") or "").strip()
        for record in records
        if str(record.get("error") or "").strip()
    )
    error_count = sum(errors.values())
    context_markers = (
        "context length",
        "context window",
        "maximum context",
        "max_model_len",
        "too many tokens",
    )
    usage_records = [record for record in records if record.get("token_usage_available") is True]

    def average(key: str, source: list[dict[str, Any]] = records) -> float:
        return (
            sum(float(record.get(key) or 0) for record in source) / len(source) if source else 0.0
        )

    return AttributionSlice(
        total=total,
        completed_total=completed_total,
        predictions_parsed=parsed,
        agent_labeled_total=agent_labeled_total,
        completed_agent_labeled_total=completed_agent_labeled_total,
        correct_agent=correct_agent,
        correct_step=correct_step,
        correct_both=correct_both,
        agent_accuracy=(
            correct_agent / agent_labeled_total * 100.0 if agent_labeled_total else 0.0
        ),
        step_accuracy=correct_step / total * 100.0 if total else 0.0,
        both_accuracy=(correct_both / agent_labeled_total * 100.0 if agent_labeled_total else 0.0),
        agent_accuracy_completed=(
            completed_correct_agent / completed_agent_labeled_total * 100.0
            if completed_agent_labeled_total
            else 0.0
        ),
        step_accuracy_completed=(
            completed_correct_step / completed_total * 100.0 if completed_total else 0.0
        ),
        both_accuracy_completed=(
            completed_correct_both / completed_agent_labeled_total * 100.0
            if completed_agent_labeled_total
            else 0.0
        ),
        coverage=parsed / total if total else 0.0,
        error_count=error_count,
        error_rate=error_count / total if total else 0.0,
        max_turn_failures=sum(
            count for message, count in errors.items() if "max turns" in message.lower()
        ),
        context_overflow_failures=sum(
            count
            for message, count in errors.items()
            if any(marker in message.lower() for marker in context_markers)
        ),
        no_tool_call_failures=sum(
            count for message, count in errors.items() if "no tool call" in message.lower()
        ),
        avg_turns=average("turns"),
        avg_inspected_steps=(
            sum(len(record.get("inspected_step_ids") or []) for record in records) / total
            if total
            else 0.0
        ),
        token_usage_records=len(usage_records),
        avg_prompt_tokens=average("prompt_tokens", usage_records),
        avg_completion_tokens=average("completion_tokens", usage_records),
        avg_total_tokens=average("total_tokens", usage_records),
        avg_max_context_chars=average("max_context_chars"),
        avg_cumulative_context_chars=average("cumulative_context_chars"),
        avg_repeated_inspections=average("repeated_inspections"),
        error_counts=dict(sorted(errors.items())),
    )


def evaluate(
    records: list[dict[str, Any]], judge: AttributionJudge | None = None
) -> WhoAndWhenMetrics:
    selected_judge = judge or StrictAttributionJudge()
    decisions = {
        id(record): selected_judge.judge(record)
        for record in records
        if _official_prediction(record) is not None
    }
    overall = _evaluate_attribution_slice(records, decisions)
    split_names = sorted({_benchmark_split(record) for record in records})
    by_split = {
        split: asdict(
            _evaluate_attribution_slice(
                [record for record in records if _benchmark_split(record) == split],
                decisions,
            )
        )
        for split in split_names
    }
    return WhoAndWhenMetrics(
        protocol="who_and_when_official",
        total=overall.total,
        completed_total=overall.completed_total,
        predictions_parsed=overall.predictions_parsed,
        agent_labeled_total=overall.agent_labeled_total,
        completed_agent_labeled_total=overall.completed_agent_labeled_total,
        correct_agent=overall.correct_agent,
        correct_step=overall.correct_step,
        correct_both=overall.correct_both,
        agent_accuracy=overall.agent_accuracy,
        step_accuracy=overall.step_accuracy,
        both_accuracy=overall.both_accuracy,
        agent_accuracy_completed=overall.agent_accuracy_completed,
        step_accuracy_completed=overall.step_accuracy_completed,
        both_accuracy_completed=overall.both_accuracy_completed,
        coverage=overall.coverage,
        error_count=overall.error_count,
        error_rate=overall.error_rate,
        max_turn_failures=overall.max_turn_failures,
        context_overflow_failures=overall.context_overflow_failures,
        no_tool_call_failures=overall.no_tool_call_failures,
        avg_turns=overall.avg_turns,
        avg_inspected_steps=overall.avg_inspected_steps,
        token_usage_records=overall.token_usage_records,
        avg_prompt_tokens=overall.avg_prompt_tokens,
        avg_completion_tokens=overall.avg_completion_tokens,
        avg_total_tokens=overall.avg_total_tokens,
        avg_max_context_chars=overall.avg_max_context_chars,
        avg_cumulative_context_chars=overall.avg_cumulative_context_chars,
        avg_repeated_inspections=overall.avg_repeated_inspections,
        error_counts=overall.error_counts,
        by_split=by_split,
    )


def build_judge(_config: dict[str, Any]) -> AttributionJudge:
    """Return the offline scorer used for final-answer evaluation."""
    return StrictAttributionJudge()


def format_score_summary(metrics: WhoAndWhenMetrics) -> str:
    """Format the two official scores for compact terminal output."""
    return (
        f"Agent Accuracy: {metrics.agent_accuracy:.2f}%\n"
        f"Step Accuracy:  {metrics.step_accuracy:.2f}%\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/who_when.yaml"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    metrics = evaluate(records, build_judge({}))
    payload = json.dumps(asdict(metrics), indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(format_score_summary(metrics), end="")


if __name__ == "__main__":
    main()
