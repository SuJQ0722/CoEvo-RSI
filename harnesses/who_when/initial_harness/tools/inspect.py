"""inspect tool: retrieve full records for selected trajectory steps."""

from __future__ import annotations

import json
from typing import Any

from src.traces import AgentTrace

MAX_INSPECT_ITEMS = 3
MAX_TOOL_RESPONSE_CHARS = 16_000
TRUNCATION_NOTICE = (
    "[The tool response exceeded the 16,000-character limit and was truncated.]"
)

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Return the full records for selected trajectory steps. "
            f"Maximum {MAX_INSPECT_ITEMS} steps per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step_id_list": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": MAX_INSPECT_ITEMS,
                    "description": f"Step IDs to inspect (max {MAX_INSPECT_ITEMS} per call).",
                }
            },
            "required": ["step_id_list"],
            "additionalProperties": False,
        },
    },
}


def execute(
    trace: AgentTrace,
    step_id_list: list[int],
    inspected_step_ids: list[int],
) -> str:
    if not isinstance(step_id_list, list) or not step_id_list:
        return json.dumps({"error": "step_id_list must be a non-empty list"})

    if len(step_id_list) > MAX_INSPECT_ITEMS:
        return json.dumps({
            "error": f"Too many steps ({len(step_id_list)}), max {MAX_INSPECT_ITEMS}."
        })

    records: list[dict[str, Any]] = []
    for value in step_id_list:
        try:
            step_id = int(value)
        except (TypeError, ValueError):
            records.append({"step_id": value, "error": "invalid step ID"})
            continue
        step = trace.find_step(step_id)
        if step is None:
            records.append({"step_id": step_id, "error": "step not found"})
            continue
        if step_id not in inspected_step_ids:
            inspected_step_ids.append(step_id)
        records.append({"step_id": step_id, "record": step.raw})
    response = json.dumps({"steps": records}, ensure_ascii=False)
    if len(response) <= MAX_TOOL_RESPONSE_CHARS:
        return response

    # Keep the tool message valid JSON after truncation. Encoding the preview as
    # a JSON string can expand quotes and backslashes, so use a binary search to
    # guarantee the serialized response itself never exceeds the hard limit.
    low = 0
    high = len(response)
    bounded = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = json.dumps(
            {
                "truncated": True,
                "content": response[:midpoint] + "\n" + TRUNCATION_NOTICE,
            },
            ensure_ascii=False,
        )
        if len(candidate) <= MAX_TOOL_RESPONSE_CHARS:
            bounded = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    return bounded
