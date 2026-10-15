"""final_answer tool: submit the failure-detection decision."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from src.traces import AgentTrace, DetectionResult

_PARENTHETICAL_ROLE_SUFFIX = re.compile(r"\s+\((?:thought|analysis|termination condition|->[^)]*)\)\s*$", re.IGNORECASE)
_OBSERVER_AGENT_ALIASES = {"computerterminal", "tool"}
_BOOLEAN_STRING_VALUES = {"true": True, "false": False}
COVERAGE_TRACE_MIN_STEPS = 20
COVERAGE_GAP_MIN_STEPS = 8
COVERAGE_TAIL_GAP_MIN_STEPS = 3
COVERAGE_MAX_NUDGES = 2
COVERAGE_SUGGESTION_LIMIT = 5
COVERAGE_WINDOW_RADIUS = 2

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the failure-detection decision and terminate.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_failure": {
                    "type": "boolean",
                    "description": "Whether the trajectory contains a failure.",
                },
                "decisive_step": {
                    "type": "integer",
                    "description": (
                        "Exact inspected step_id where the earliest decisive "
                        "failure was introduced. This is the scored attribution "
                        "step; supporting evidence_step_ids cannot substitute "
                        "for choosing this singleton."
                    ),
                },
                "evidence_step_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "Inspected steps supporting the decision. Include any "
                        "symptom or confirmation steps after decisive_step; the "
                        "tool will put decisive_step first for evaluator "
                        "compatibility."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": "Short explanation based on inspected evidence.",
                },
                "responsible_agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Exact agent or component responsible for the earliest decisive failure.",
                },
            },
            "required": ["is_failure", "decisive_step", "responsible_agent"],
            "additionalProperties": False,
        },
    },
}


def schema_for_task(task_mode: str) -> dict[str, Any]:
    """Return strict attribution constraints only where labels support them."""
    schema = deepcopy(SCHEMA)
    properties = schema["function"]["parameters"]["properties"]
    if task_mode != "failure_attribution":
        properties["responsible_agent"].pop("minLength", None)
    if task_mode == "failure_detection":
        required = schema["function"]["parameters"]["required"]
        schema["function"]["parameters"]["required"] = [
            field for field in required if field != "decisive_step"
        ]
    return schema


def _tool_error(*missing: str) -> str:
    return json.dumps(
        {
            "error": "final_answer is missing required attribution fields",
            "missing": list(missing),
            "retryable": True,
            "instruction": "Inspect relevant steps and call final_answer again.",
        }
    )


def _uninspected_evidence_error(step_id: int) -> str:
    return json.dumps(
        {
            "error": "final_answer decisive step has not been inspected",
            "uninspected_step_id": step_id,
            "retryable": True,
            "instruction": (
                "Call inspect with a step_id_list containing this step, or submit "
                "an inspected step as decisive_step."
            ),
        }
    )


def _coverage_gap_error(
    suggestions: list[int],
    largest_gap: tuple[int, int, int],
    *,
    nudge_count: int,
) -> str:
    return json.dumps(
        {
            "error": "inspection coverage gap before final_answer",
            "retryable": True,
            "suggested_step_id_list": suggestions,
            "largest_uninspected_gap": {
                "start_step_id": largest_gap[0],
                "end_step_id": largest_gap[1],
                "step_count": largest_gap[2],
            },
            "remaining_nudges": max(0, COVERAGE_MAX_NUDGES - nudge_count - 1),
            "instruction": (
                "Call inspect once with suggested_step_id_list, compare the newly "
                "returned windows against the current decisive_step, then call "
                "final_answer again."
            ),
        }
    )


def _normalize_responsible_agent(value: Any) -> str:
    agent = str(value or "").strip()
    return _PARENTHETICAL_ROLE_SUFFIX.sub("", agent).strip()


def _agent_key(value: Any) -> str:
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def _coerce_boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _BOOLEAN_STRING_VALUES:
            return _BOOLEAN_STRING_VALUES[normalized]
    return None


def _coerce_step_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _contiguous_uninspected_position_gaps(
    trace: AgentTrace,
    inspected_step_ids: set[int],
) -> list[list[int]]:
    gaps: list[list[int]] = []
    current: list[int] = []
    for position, step in enumerate(trace.steps):
        if step.step_id in inspected_step_ids:
            if current:
                gaps.append(current)
                current = []
            continue
        current.append(position)
    if current:
        gaps.append(current)
    return gaps


def _anchor_id_for_position(trace: AgentTrace, position: int) -> int:
    bounded = min(max(position, 0), len(trace.steps) - 1)
    return trace.steps[bounded].step_id


def _coverage_gap_suggestions(
    trace: AgentTrace | None,
    inspected_step_ids: list[int] | None,
    *,
    nudge_count: int,
) -> tuple[list[int], tuple[int, int, int]] | None:
    if (
        trace is None
        or inspected_step_ids is None
        or nudge_count >= COVERAGE_MAX_NUDGES
        or len(trace.steps) < COVERAGE_TRACE_MIN_STEPS
    ):
        return None

    inspected: set[int] = set()
    for value in inspected_step_ids:
        try:
            inspected.add(int(value))
        except (TypeError, ValueError):
            continue
    if not inspected:
        return None

    gaps = _contiguous_uninspected_position_gaps(trace, inspected)
    candidate_gaps = [gap for gap in gaps if len(gap) >= COVERAGE_GAP_MIN_STEPS]
    candidate_gaps.extend(
        gap
        for gap in gaps
        if (
            gap
            and gap[-1] == len(trace.steps) - 1
            and len(gap) >= COVERAGE_TAIL_GAP_MIN_STEPS
            and gap not in candidate_gaps
        )
    )
    if not candidate_gaps:
        return None

    largest_gap_positions = max(candidate_gaps, key=len)
    largest_gap = (
        _anchor_id_for_position(trace, largest_gap_positions[0]),
        _anchor_id_for_position(trace, largest_gap_positions[-1]),
        len(largest_gap_positions),
    )

    suggestions: list[int] = []

    def add_anchor(position: int) -> None:
        step_id = _anchor_id_for_position(trace, position)
        if step_id not in inspected and step_id not in suggestions:
            suggestions.append(step_id)

    # First cover the start of chronological gaps, where earliest-failure labels
    # often sit just after an observed window. Then fill with midpoints.
    for gap in sorted(candidate_gaps, key=lambda item: item[0]):
        add_anchor(gap[min(COVERAGE_WINDOW_RADIUS, len(gap) - 1)])
        if len(suggestions) >= COVERAGE_SUGGESTION_LIMIT:
            return suggestions, largest_gap

    for gap in sorted(candidate_gaps, key=len, reverse=True):
        add_anchor(gap[len(gap) // 2])
        if len(suggestions) >= COVERAGE_SUGGESTION_LIMIT:
            return suggestions, largest_gap

    for gap in sorted(candidate_gaps, key=len, reverse=True):
        add_anchor(gap[max(0, len(gap) - COVERAGE_WINDOW_RADIUS - 1)])
        if len(suggestions) >= COVERAGE_SUGGESTION_LIMIT:
            return suggestions, largest_gap

    return (suggestions, largest_gap) if suggestions else None


def _canonicalize_observer_attribution(
    evidence: list[int],
    responsible_agent: str,
    trace: AgentTrace | None,
) -> tuple[list[int], str]:
    """Map terminal/tool observation blame to the preceding causal actor.

    The tool/result row is still useful evidence, but in the benchmark labels
    available to this harness it is not the responsible actor. Keep this narrow
    so genuine executor labels such as bash remain selectable.
    """
    if not evidence or trace is None:
        return evidence, responsible_agent
    if _agent_key(responsible_agent) not in _OBSERVER_AGENT_ALIASES:
        return evidence, responsible_agent

    first_step_id = evidence[0]
    try:
        first_step_id = int(first_step_id)
    except (TypeError, ValueError):
        return evidence, responsible_agent

    observer_index: int | None = None
    for index, step in enumerate(trace.steps):
        if step.step_id == first_step_id:
            observer_index = index
            break
    if observer_index is None:
        return evidence, responsible_agent

    observer_step = trace.steps[observer_index]
    if _agent_key(observer_step.agent_id) not in _OBSERVER_AGENT_ALIASES:
        return evidence, responsible_agent

    for previous_step in reversed(trace.steps[:observer_index]):
        if _agent_key(previous_step.agent_id) in _OBSERVER_AGENT_ALIASES:
            continue
        canonical_agent = _normalize_responsible_agent(previous_step.agent_id)
        if not canonical_agent:
            continue
        canonical_evidence = [previous_step.step_id, *evidence]
        return list(dict.fromkeys(canonical_evidence)), canonical_agent
    return evidence, responsible_agent


def execute(
    arguments: dict[str, Any],
    *,
    trace: AgentTrace | None = None,
    task_mode: str = "failure_attribution",
    inspected_step_ids: list[int] | None = None,
    coverage_nudge_count: int = 0,
) -> tuple[str, DetectionResult | None]:
    is_failure = _coerce_boolean(arguments.get("is_failure"))
    if is_failure is None:
        return json.dumps(
            {
                "error": "is_failure must be a JSON boolean",
                "retryable": True,
                "instruction": (
                    "Call final_answer again with is_failure set to true or "
                    "false. Only exact boolean-like strings are normalized; "
                    "other strings are rejected."
                ),
            }
        ), None

    evidence: list[int] = []
    raw_evidence = arguments.get("evidence_step_ids", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    for value in raw_evidence:
        step_id = _coerce_step_id(value)
        if step_id is not None:
            evidence.append(step_id)

    decisive_step = _coerce_step_id(
        arguments.get("decisive_step", arguments.get("decisive_step_id"))
    )
    if decisive_step is None and evidence:
        # Backward compatibility for models that still follow the parent
        # evidence-first protocol despite the stricter schema.
        decisive_step = evidence[0]

    responsible_agent = _normalize_responsible_agent(arguments.get("responsible_agent", ""))
    missing: list[str] = []
    if task_mode in {"step_attribution", "failure_attribution"} and decisive_step is None:
        missing.append("decisive_step")
    if task_mode == "failure_attribution" and not responsible_agent:
        missing.append("responsible_agent")
    if missing:
        return _tool_error(*missing), None

    if decisive_step is not None:
        evidence = [decisive_step, *evidence]
    evidence = list(dict.fromkeys(evidence))

    if task_mode == "failure_attribution":
        evidence, responsible_agent = _canonicalize_observer_attribution(
            evidence,
            responsible_agent,
            trace,
        )

    if (
        task_mode in {"step_attribution", "failure_attribution"}
        and evidence
        and inspected_step_ids is not None
    ):
        inspected: set[int] = set()
        for value in inspected_step_ids:
            try:
                inspected.add(int(value))
            except (TypeError, ValueError):
                continue
        if evidence[0] not in inspected:
            return _uninspected_evidence_error(evidence[0]), None

        coverage_gap = _coverage_gap_suggestions(
            trace,
            inspected_step_ids,
            nudge_count=coverage_nudge_count,
        )
        if coverage_gap is not None:
            suggestions, largest_gap = coverage_gap
            return (
                _coverage_gap_error(
                    suggestions,
                    largest_gap,
                    nudge_count=coverage_nudge_count,
                ),
                None,
            )

    result = DetectionResult(
        is_failure=is_failure,
        evidence_step_ids=tuple(evidence),
        rationale=str(arguments.get("rationale", "")),
        responsible_agent=responsible_agent,
    )
    return json.dumps({"status": "submitted", "is_failure": result.is_failure}), result
