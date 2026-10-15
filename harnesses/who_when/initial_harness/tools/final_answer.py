"""final_answer tool: submit the failure-detection decision."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from src.traces import DetectionResult

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
                "evidence_step_ids": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "description": "Inspected steps supporting the decision.",
                },
                "rationale": {
                    "type": "string",
                    "description": "Short explanation based on inspected evidence.",
                },
                "responsible_agent": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Agent responsible for the decisive failure.",
                },
            },
            "required": ["is_failure", "evidence_step_ids", "responsible_agent"],
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
        properties["evidence_step_ids"].pop("minItems", None)
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


def execute(
    arguments: dict[str, Any], *, task_mode: str = "failure_attribution"
) -> tuple[str, DetectionResult | None]:
    if not isinstance(arguments.get("is_failure"), bool):
        return json.dumps({"error": "is_failure must be a boolean"}), None

    evidence: list[int] = []
    raw_evidence = arguments.get("evidence_step_ids", [])
    if not isinstance(raw_evidence, list):
        raw_evidence = []
    for value in raw_evidence:
        try:
            evidence.append(int(value))
        except (TypeError, ValueError):
            continue

    responsible_agent = str(arguments.get("responsible_agent", "")).strip()
    missing: list[str] = []
    if task_mode in {"step_attribution", "failure_attribution"} and not evidence:
        missing.append("evidence_step_ids")
    if task_mode == "failure_attribution" and not responsible_agent:
        missing.append("responsible_agent")
    if missing:
        return _tool_error(*missing), None

    result = DetectionResult(
        is_failure=arguments["is_failure"],
        evidence_step_ids=tuple(dict.fromkeys(evidence)),
        rationale=str(arguments.get("rationale", "")),
        responsible_agent=responsible_agent,
    )
    return json.dumps({"status": "submitted", "is_failure": result.is_failure}), result
