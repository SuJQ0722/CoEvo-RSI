"""Conservatively select a child harness from staged parent/child rollouts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.config import active_harness, load_config
from src.domains.base import SmokeCheck, SmokeReport
from src.domains.who_when.runtime.evaluation import AttributionJudge, build_judge, evaluate
from src.meta_evolver.evolution import (
    load_harness_tree,
    record_selection,
)
from src.meta_evolver.feedback import (
    build_failure_feedback,
    write_failure_feedback,
)
from src.utils.persistence import atomic_json

QUALITY_METRICS = ("agent_accuracy", "step_accuracy", "coverage")
COST_METRICS = ("error_rate", "avg_turns", "avg_inspected_steps")


@dataclass(frozen=True)
class HarnessMetrics:
    agent_accuracy: float
    step_accuracy: float
    coverage: float
    error_rate: float
    avg_turns: float
    avg_inspected_steps: float
    samples: int
    context_overflow_failures: int = 0
    max_turn_failures: int = 0
    no_tool_call_failures: int = 0
    token_usage_records: int = 0
    avg_prompt_tokens: float = 0.0
    avg_completion_tokens: float = 0.0
    avg_total_tokens: float = 0.0
    avg_max_context_chars: float = 0.0
    avg_cumulative_context_chars: float = 0.0
    avg_repeated_inspections: float = 0.0
    error_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_test_harness(harness_name: str) -> SmokeReport:
    """Who&When smoke helper for staged selection and standalone domain callers."""
    from src.domains.who_when.smoke import test

    report = test(
        harness_name,
        package="harnesses.who_when",
        root=Path(__file__).resolve().parents[3] / "harnesses/who_when",
    )
    return SmokeReport(
        report.harness,
        report.passed,
        tuple(SmokeCheck(item.name, item.passed, item.detail) for item in report.checks),
    )


def metrics_from_records(
    records: list[dict[str, Any]], judge: AttributionJudge | None = None
) -> HarnessMetrics:
    detection = evaluate(records, judge)
    total = len(records)
    turns = sum(float(record.get("turns") or 0) for record in records)
    inspections = sum(len(record.get("inspected_step_ids") or []) for record in records)
    return HarnessMetrics(
        agent_accuracy=detection.agent_accuracy,
        step_accuracy=detection.step_accuracy,
        coverage=detection.coverage,
        error_rate=detection.error_rate,
        avg_turns=turns / total if total else 0.0,
        avg_inspected_steps=inspections / total if total else 0.0,
        samples=total,
        context_overflow_failures=detection.context_overflow_failures,
        max_turn_failures=detection.max_turn_failures,
        no_tool_call_failures=detection.no_tool_call_failures,
        token_usage_records=detection.token_usage_records,
        avg_prompt_tokens=detection.avg_prompt_tokens,
        avg_completion_tokens=detection.avg_completion_tokens,
        avg_total_tokens=detection.avg_total_tokens,
        avg_max_context_chars=detection.avg_max_context_chars,
        avg_cumulative_context_chars=detection.avg_cumulative_context_chars,
        avg_repeated_inspections=detection.avg_repeated_inspections,
        error_counts=detection.error_counts,
    )


def validate_comparable_records(
    parent_records: list[dict[str, Any]], child_records: list[dict[str, Any]]
) -> None:
    def index(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for record in records:
            sample_id = record.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{label} rollout is missing sample_id")
            gold_record = record.get("gold", {})
            gold = (
                gold_record.get("responsible_agent"),
                gold_record.get("decisive_step"),
            )
            if not str(gold[0] or "").strip() or gold[1] is None:
                raise ValueError(
                    f"{label} rollout has invalid attribution labels for {sample_id!r}"
                )
            if sample_id in indexed:
                raise ValueError(f"{label} rollout repeats sample_id={sample_id!r}")
            indexed[sample_id] = gold
        return indexed

    parent_index = index(parent_records, "parent")
    child_index = index(child_records, "child")
    if parent_index != child_index:
        raise ValueError(
            "parent and child rollouts must contain identical sample IDs and attribution labels"
        )
    if not parent_index:
        raise ValueError("selection stage cannot be empty")


def _dominates(left: HarnessMetrics, right: HarnessMetrics) -> bool:
    no_worse = all(getattr(left, name) >= getattr(right, name) for name in QUALITY_METRICS)
    no_worse = no_worse and all(
        getattr(left, name) <= getattr(right, name) for name in COST_METRICS
    )
    strictly_better = any(
        getattr(left, name) > getattr(right, name) for name in QUALITY_METRICS
    ) or any(getattr(left, name) < getattr(right, name) for name in COST_METRICS)
    return no_worse and strictly_better


def pareto_relation(parent: HarnessMetrics, child: HarnessMetrics) -> str:
    if _dominates(child, parent):
        return "child_dominates"
    if _dominates(parent, child):
        return "parent_dominates"
    return "non_dominated"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def decide_selection(
    child_smoke_passed: bool,
    stage_reports: list[dict[str, Any]],
    *,
    quality_tolerance: float = 100.0,
) -> tuple[str, str]:
    """Pre-filter a harness for full-policy SFT using official attribution metrics.

    Coverage is reported separately as a target-health signal.  It is not a
    hard comparison metric because the fair promotion decision happens only
    after the child harness has trained its own policy.
    """
    if not child_smoke_passed:
        return "retain_parent", "child failed smoke tests"
    excessive: list[str] = []
    for stage in stage_reports:
        parent = stage["parent_metrics"]
        child = stage["child_metrics"]
        for metric in ("agent_accuracy", "step_accuracy"):
            delta = float(child[metric]) - float(parent[metric])
            if delta < -quality_tolerance - 1e-12:
                excessive.append(f"stage {stage['stage']} {metric} delta={delta:+.4f}")
    if excessive:
        return (
            "retain_parent",
            f"child exceeds {quality_tolerance:.2f} prefilter tolerance: " + "; ".join(excessive),
        )
    return (
        "shortlist_child",
        (
            "child Agent Accuracy and Step Accuracy are within the prefilter tolerance; "
            "defer promotion until full-policy validation"
        ),
    )


def summarize_failed_child(
    reason: str, stage_reports: list[dict[str, Any]], child_smoke: dict[str, Any]
) -> dict[str, Any]:
    causes: list[str] = []
    if not child_smoke.get("passed"):
        causes.append("child_failed_smoke_contract")
    for stage in stage_reports:
        parent = stage["parent_metrics"]
        child = stage["child_metrics"]
        for metric in ("agent_accuracy", "step_accuracy"):
            if child[metric] < parent[metric]:
                causes.append(f"lower_{metric}")
        if stage.get("regression_cases"):
            causes.append("lost_parent_correct_cases")
    causes = list(dict.fromkeys(causes)) or ["no_strict_pareto_improvement"]
    stage_comparisons = [
        {
            "stage": stage["stage"],
            "relation": stage["relation"],
            "metric_delta_child_minus_parent": {
                key: stage["child_metrics"][key] - stage["parent_metrics"][key]
                for key in (
                    "agent_accuracy",
                    "step_accuracy",
                    "coverage",
                    "error_rate",
                    "avg_turns",
                    "avg_inspected_steps",
                )
            },
            "regression_cases": stage.get("regression_cases", []),
        }
        for stage in stage_reports
    ]
    return build_failure_feedback(
        status="child_not_promoted",
        reason=reason,
        causes=causes,
        evidence={"child_smoke": child_smoke},
        details={"stage_comparisons": stage_comparisons},
    )


def regression_cases(
    parent_records: list[dict[str, Any]],
    child_records: list[dict[str, Any]],
    limit: int = 20,
) -> list[dict[str, Any]]:
    parent_by_id = {record["sample_id"]: record for record in parent_records}
    cases: list[dict[str, Any]] = []
    for child in child_records:
        parent = parent_by_id[child["sample_id"]]
        gold = child.get("gold", {})

        def attribution_correct(record: dict[str, Any], gold=gold) -> bool:
            prediction = record.get("prediction") or {}
            evidence = prediction.get("evidence_step_ids") or []
            return bool(
                str(gold.get("responsible_agent")) in str(prediction.get("responsible_agent") or "")
                and evidence
                and str(gold.get("decisive_step")) in str(evidence[0])
            )

        parent_correct = attribution_correct(parent)
        child_correct = attribution_correct(child)
        if parent_correct and not child_correct:
            cases.append(
                {
                    "sample_id": child["sample_id"],
                    "gold": {
                        "responsible_agent": gold.get("responsible_agent"),
                        "decisive_step": gold.get("decisive_step"),
                    },
                    "parent_prediction": parent.get("prediction"),
                    "child_prediction": child.get("prediction"),
                    "child_error": str(child.get("error") or ""),
                    "turn_delta": float(child.get("turns") or 0) - float(parent.get("turns") or 0),
                    "inspected_steps_delta": len(child.get("inspected_step_ids") or [])
                    - len(parent.get("inspected_step_ids") or []),
                }
            )
            if len(cases) >= limit:
                break
    return cases


def select_candidate(
    *,
    config_path: Path,
    registry_path: Path,
    parent_name: str,
    child_name: str,
    stages: list[tuple[Path, Path]],
    report_path: Path,
    judge: AttributionJudge | None = None,
) -> dict[str, Any]:
    """Shortlist a child using official Who&When attribution metrics."""
    if not stages:
        raise ValueError("at least one staged parent/child rollout pair is required")
    config = load_config(config_path)
    selected_judge = judge or build_judge(config)
    if active_harness(config) != parent_name:
        raise ValueError(
            f"stale selection: active harness is {active_harness(config)!r}, "
            f"not parent {parent_name!r}"
        )
    tree = load_harness_tree(registry_path)
    child_node = tree["nodes"].get(child_name)
    if not isinstance(child_node, dict) or child_node.get("parent") != parent_name:
        raise ValueError(f"{child_name!r} is not a registered child of {parent_name!r}")

    parent_smoke = smoke_test_harness(parent_name)
    child_smoke = smoke_test_harness(child_name)
    if not parent_smoke.passed:
        raise RuntimeError("active parent failed smoke tests; selection cannot proceed safely")

    stage_reports: list[dict[str, Any]] = []
    if child_smoke.passed:
        for index, (parent_path, child_path) in enumerate(stages, start=1):
            parent_records = _read_jsonl(parent_path)
            child_records = _read_jsonl(child_path)
            validate_comparable_records(parent_records, child_records)
            parent_metrics = metrics_from_records(parent_records, selected_judge)
            child_metrics = metrics_from_records(child_records, selected_judge)
            relation = pareto_relation(parent_metrics, child_metrics)
            frontier = (
                [child_name]
                if relation == "child_dominates"
                else [parent_name]
                if relation == "parent_dominates"
                else [parent_name, child_name]
            )
            stage_reports.append(
                {
                    "stage": index,
                    "parent_rollouts": str(parent_path),
                    "child_rollouts": str(child_path),
                    "parent_metrics": parent_metrics.to_dict(),
                    "child_metrics": child_metrics.to_dict(),
                    "relation": relation,
                    "pareto_frontier": frontier,
                    "regression_cases": regression_cases(parent_records, child_records),
                }
            )

    selection_config = dict(config.get("selection") or {})
    quality_tolerance = float(selection_config.get("quality_tolerance", 100.0))
    coverage_target = float(selection_config.get("coverage_target", 0.80))
    decision, reason = decide_selection(
        child_smoke.passed,
        stage_reports,
        quality_tolerance=quality_tolerance,
    )
    coverage_status = [
        {
            "stage": stage["stage"],
            "target": coverage_target,
            "parent": stage["parent_metrics"]["coverage"],
            "child": stage["child_metrics"]["coverage"],
            "parent_met": stage["parent_metrics"]["coverage"] >= coverage_target,
            "child_met": stage["child_metrics"]["coverage"] >= coverage_target,
            "both_met": (
                stage["parent_metrics"]["coverage"] >= coverage_target
                and stage["child_metrics"]["coverage"] >= coverage_target
            ),
        }
        for stage in stage_reports
    ]

    payload = {
        "selection_rule": "shortlist_if_agent_and_step_accuracy_regress_by_at_most_tolerance",
        "quality_metrics": ["agent_accuracy", "step_accuracy"],
        "quality_tolerance": quality_tolerance,
        "coverage_is_hard_gate": False,
        "coverage_status": coverage_status,
        "parent": parent_name,
        "child": child_name,
        "decision": decision,
        "reason": reason,
        "active_after_selection": parent_name,
        "smoke": {
            "parent": parent_smoke.to_dict(),
            "child": child_smoke.to_dict(),
        },
        "stages": stage_reports,
    }
    if decision == "retain_parent":
        feedback = summarize_failed_child(reason, stage_reports, child_smoke.to_dict())
        feedback_path = report_path.with_name("failure_feedback.json")
        write_failure_feedback(feedback_path, feedback)
        payload["failure_feedback"] = feedback
        payload["failure_feedback_path"] = str(feedback_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report_path, payload)
    record_selection(registry_path, parent_name, child_name, decision, report_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/who_when.yaml"))
    parser.add_argument("--registry", type=Path, default=Path("registries/harness_tree.json"))
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        nargs=2,
        metavar=("PARENT_JSONL", "CHILD_JSONL"),
        required=True,
        help="Repeat for progressively larger filtering stages.",
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    payload = select_candidate(
        config_path=config_path,
        registry_path=args.registry.resolve(),
        parent_name=args.parent,
        child_name=args.child,
        stages=[(Path(parent).resolve(), Path(child).resolve()) for parent, child in args.stage],
        report_path=args.report.resolve(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


def select(parent, child, config):
    from src.domains.selection import select_any_improves_no_regression

    return select_any_improves_no_regression(parent, child, config)
