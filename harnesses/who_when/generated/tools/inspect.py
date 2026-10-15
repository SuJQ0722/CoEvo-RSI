"""inspect tool: retrieve compact records for selected trajectory steps."""

from __future__ import annotations

import json
import re
from typing import Any

from src.traces import AgentTrace, TraceStep

MAX_INSPECT_ITEMS = 5
MAX_TOOL_RESPONSE_CHARS = 10_000
INSPECTION_CONTEXT_RADIUS = 2
SINGLE_ANCHOR_CONTEXT_RADIUS = 2
AUTO_SCOUT_MIN_STEPS = 6
AUTO_SCOUT_PREFIX_FRACTION = 0.25
AUTO_SCOUT_MAX_ANCHORS = 4
AUTO_SCOUT_FRACTIONS = (0.2, 0.45, 0.75, 1.0)
DEFAULT_TEXT_FIELD_CHARS = 900
MIN_TEXT_FIELD_CHARS = 96
MAX_DICT_KEYS = 24
MAX_LIST_ITEMS = 6
TRUNCATION_NOTICE = (
    "[The compact inspect response exceeded its budget; details were reduced.]"
)
IDENTITY_KEYS = ("step_id", "step", "agent_name", "name", "agent_id", "role", "phase")
TEXT_KEYS = ("content", "thought", "action", "observation", "input", "output", "error")
PRIORITY_KEYS = IDENTITY_KEYS + TEXT_KEYS + ("substeps",)
COMPONENT_MARKER_PATTERN = re.compile(
    r"</?\s*(memory|reflection|plan|planning|action)\b", re.IGNORECASE
)
COMPONENT_BLOCK_PATTERN = re.compile(
    r"<\s*(memory|reflection|plan|planning|action)\b[^>]*>(.*?)</\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "inspect",
        "description": (
            "Return compact head/tail records for selected trajectory step "
            "anchors and their local neighboring context, including step "
            "identity, available fields, and text shapes. On the first narrow "
            "early inspection of a long trace, the response may also include "
            "automatic scout windows from later parts of the trace. "
            f"Maximum {MAX_INSPECT_ITEMS} requested anchors per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "step_id_list": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 1,
                    "maxItems": MAX_INSPECT_ITEMS,
                    "description": (
                        f"Step IDs to inspect as anchors (max {MAX_INSPECT_ITEMS} "
                        "per call). Neighboring steps may be returned too."
                    ),
                }
            },
            "required": ["step_id_list"],
            "additionalProperties": False,
        },
    },
}


def _shape(value: str) -> dict[str, int]:
    return {"characters": len(value), "lines": value.count("\n") + 1}


def _clip_text(value: str, limit: int) -> str | dict[str, Any]:
    if len(value) <= limit:
        return value
    head_chars = max(40, limit // 2)
    tail_chars = max(40, limit - head_chars)
    return {
        "truncated": True,
        **_shape(value),
        "head": value[:head_chars],
        "tail": value[-tail_chars:],
        "omitted_characters": max(0, len(value) - head_chars - tail_chars),
    }


def _ordered_keys(record: dict[str, Any]) -> list[str]:
    keys = [key for key in PRIORITY_KEYS if key in record and record[key] not in (None, "", [], {})]
    keys.extend(
        key
        for key in sorted(record)
        if key not in keys and record[key] not in (None, "", [], {})
    )
    return keys[:MAX_DICT_KEYS]


def _compact_value(value: Any, text_limit: int, depth: int = 0) -> Any:
    if isinstance(value, str):
        return _clip_text(value, text_limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        if depth >= 3:
            return {
                "type": "object",
                "keys": sorted(str(key) for key in value.keys())[:MAX_DICT_KEYS],
                "truncated": True,
            }
        keys = _ordered_keys(value)
        compact = {
            key: _compact_value(value[key], max(MIN_TEXT_FIELD_CHARS, text_limit // 2), depth + 1)
            for key in keys
        }
        if len(keys) < len([key for key, item in value.items() if item not in (None, "", [], {})]):
            compact["_omitted_keys"] = [
                key
                for key in sorted(value)
                if key not in keys and value[key] not in (None, "", [], {})
            ]
        return compact
    if isinstance(value, list):
        if not value:
            return []
        omitted = max(0, len(value) - MAX_LIST_ITEMS)
        if len(value) <= MAX_LIST_ITEMS:
            selected = value
        else:
            head_count = max(1, MAX_LIST_ITEMS // 2)
            tail_count = MAX_LIST_ITEMS - head_count
            selected = value[:head_count] + value[-tail_count:]
        compact_items = [
            _compact_value(item, max(MIN_TEXT_FIELD_CHARS, text_limit // 2), depth + 1)
            for item in selected
        ]
        if omitted:
            return {
                "type": "list",
                "items": compact_items,
                "original_length": len(value),
                "omitted_middle_items": omitted,
            }
        return compact_items
    return _clip_text(str(value), text_limit)


def _text_shapes(record: dict[str, Any]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value:
            result[key] = _shape(value)
    return result


def _iter_text_values(value: Any, depth: int = 0) -> list[str]:
    if depth > 4:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        text_values: list[str] = []
        for key, item in value.items():
            if key in TEXT_KEYS or key == "substeps" or isinstance(item, (dict, list)):
                text_values.extend(_iter_text_values(item, depth + 1))
        return text_values
    if isinstance(value, list):
        text_values = []
        for item in value[:MAX_LIST_ITEMS]:
            text_values.extend(_iter_text_values(item, depth + 1))
        return text_values
    return []


def _component_markers(record: dict[str, Any]) -> list[str]:
    markers: list[str] = []
    for text in _iter_text_values(record):
        for match in COMPONENT_MARKER_PATTERN.finditer(text):
            marker = match.group(1).lower()
            if marker not in markers:
                markers.append(marker)
    return markers


def _clean_preview_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _component_snippets(
    record: dict[str, Any],
    *,
    limit: int = 220,
    max_items: int = 5,
) -> list[dict[str, Any]]:
    snippets: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for text in _iter_text_values(record):
        for match in COMPONENT_BLOCK_PATTERN.finditer(text):
            component = match.group(1).lower()
            preview = _clean_preview_text(match.group(2))
            if not preview:
                continue
            key = (component, preview[:80])
            if key in seen:
                continue
            seen.add(key)
            snippets.append(
                {
                    "component": component,
                    "preview": _clip_text(preview, limit),
                }
            )
            if len(snippets) >= max_items:
                return snippets
    return snippets


def _substep_outline(
    record: dict[str, Any],
    *,
    limit: int = 180,
    max_items: int = 4,
) -> list[dict[str, Any]]:
    raw_substeps = record.get("substeps")
    if not isinstance(raw_substeps, list):
        return []

    outline: list[dict[str, Any]] = []
    for index, substep in enumerate(raw_substeps):
        if not isinstance(substep, dict):
            continue
        item: dict[str, Any] = {"index": index}
        for key in IDENTITY_KEYS:
            value = substep.get(key)
            if value not in (None, "", [], {}):
                item[key] = _compact_value(value, limit)
        for key in TEXT_KEYS:
            value = substep.get(key)
            if isinstance(value, str) and value:
                item[key] = _clip_text(_clean_preview_text(value), limit)
                break
            if isinstance(value, dict) and value:
                item[f"{key}_fields"] = sorted(str(field) for field in value)[:MAX_DICT_KEYS]
                break
        nested_components = _component_snippets(substep, limit=max(96, limit // 2), max_items=2)
        if nested_components:
            item["component_snippets"] = nested_components
        if len(item) > 1:
            outline.append(item)
        if len(outline) >= max_items:
            break
    return outline


def _short_text_previews(record: dict[str, Any], limit: int = 96) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []

    def visit(value: Any, path: str, depth: int = 0) -> None:
        if len(previews) >= 3 or depth > 4:
            return
        if isinstance(value, str):
            if value:
                previews.append({"path": path, "preview": _clip_text(value, limit)})
            return
        if isinstance(value, dict):
            ordered = [key for key in TEXT_KEYS if key in value]
            ordered.extend(
                key
                for key in ("substeps",)
                if key in value and key not in ordered
            )
            ordered.extend(
                key
                for key in sorted(value)
                if key not in ordered and isinstance(value[key], (dict, list))
            )
            for key in ordered:
                visit(value[key], f"{path}.{key}" if path else str(key), depth + 1)
                if len(previews) >= 3:
                    return
            return
        if isinstance(value, list):
            for index, item in enumerate(value[:MAX_LIST_ITEMS]):
                visit(item, f"{path}[{index}]", depth + 1)
                if len(previews) >= 3:
                    return

    visit(record, "")
    return previews


def _lean_step(step: TraceStep, text_limit: int) -> dict[str, Any]:
    raw = step.raw
    lean: dict[str, Any] = {
        "step_id": step.step_id,
        "agent": step.agent_id,
    }
    for key in ("role", "phase"):
        if raw.get(key) not in (None, "", [], {}):
            lean[key] = raw[key]
    shapes = _text_shapes(raw)
    if shapes:
        lean["text_shapes"] = shapes
    markers = _component_markers(raw)
    if markers:
        lean["component_markers"] = markers
    previews = _short_text_previews(raw, limit=text_limit)
    if previews:
        lean["text_previews"] = previews
    snippets = _component_snippets(raw, limit=max(120, text_limit), max_items=3)
    if snippets:
        lean["component_snippets"] = snippets
    outline = _substep_outline(raw, limit=max(120, text_limit), max_items=3)
    if outline:
        lean["substep_outline"] = outline
    return lean


def _compact_step(step: TraceStep, text_limit: int) -> dict[str, Any]:
    raw = step.raw
    identity = {
        key: raw[key]
        for key in IDENTITY_KEYS
        if key in raw and raw[key] not in (None, "")
    }
    identity.setdefault("agent", step.agent_id)
    compact_record = {
        key: _compact_value(raw[key], text_limit)
        for key in _ordered_keys(raw)
    }
    summary = step.summary()
    markers = _component_markers(raw)
    if markers:
        summary["component_markers"] = markers
    result = {
        "step_id": step.step_id,
        "agent": step.agent_id,
        "identity": identity,
        "summary": summary,
        "record_keys": sorted(str(key) for key in raw.keys()),
        "text_shapes": _text_shapes(raw),
        "record": compact_record,
    }
    snippets = _component_snippets(raw, limit=max(160, text_limit // 2), max_items=4)
    if snippets:
        result["component_snippets"] = snippets
    outline = _substep_outline(raw, limit=max(140, text_limit // 3), max_items=4)
    if outline:
        result["substep_outline"] = outline
    return result


def _build_response(
    step_records: list[TraceStep | dict[str, Any]],
    coverage_guidance: dict[str, Any] | None = None,
) -> str:
    for text_limit in (DEFAULT_TEXT_FIELD_CHARS, 700, 520, 360, 240, 160, MIN_TEXT_FIELD_CHARS):
        steps: list[dict[str, Any]] = []
        for item in step_records:
            if isinstance(item, TraceStep):
                steps.append(_compact_step(item, text_limit))
            else:
                steps.append(item)
        payload: dict[str, Any] = {
            "steps": steps,
            "budget": {
                "max_anchors_per_call": MAX_INSPECT_ITEMS,
                "max_response_chars": MAX_TOOL_RESPONSE_CHARS,
                "inspection_window": (
                    "requested steps are anchors; compact neighboring steps "
                    "may be included and count as inspected"
                ),
                "text_truncation": "long strings preserve head and tail; middle text may be omitted",
                "text_field_chars": text_limit,
            },
            "attribution_hint": (
                "Use the earliest inspected step that introduced the decisive failure. "
                "Terminal/output steps often report symptoms from an earlier agent action."
            ),
        }
        if coverage_guidance:
            payload["coverage_guidance"] = coverage_guidance
        response = json.dumps(
            payload,
            ensure_ascii=False,
        )
        if len(response) <= MAX_TOOL_RESPONSE_CHARS:
            return response

    for text_limit in (320, 260, 200, 150, 110, MIN_TEXT_FIELD_CHARS):
        lean_steps: list[dict[str, Any]] = []
        for item in step_records:
            if isinstance(item, TraceStep):
                lean_steps.append(_lean_step(item, text_limit))
            else:
                lean_steps.append(item)
        payload = {
            "steps": lean_steps,
            "truncated": True,
            "notice": TRUNCATION_NOTICE,
            "budget": {
                "max_anchors_per_call": MAX_INSPECT_ITEMS,
                "max_response_chars": MAX_TOOL_RESPONSE_CHARS,
                "lean_text_field_chars": text_limit,
                "inspection_window": (
                    "requested steps are anchors; compact neighboring steps "
                    "may be included and count as inspected"
                ),
            },
            "attribution_hint": (
                "This lean response keeps text previews for every returned step. "
                "If the decisive boundary is unclear, inspect a narrower local "
                "window before final_answer."
            ),
        }
        if coverage_guidance:
            payload["coverage_guidance"] = coverage_guidance
        response = json.dumps(payload, ensure_ascii=False)
        if len(response) <= MAX_TOOL_RESPONSE_CHARS:
            return response

    minimal_steps: list[dict[str, Any]] = []
    for item in step_records:
        if isinstance(item, TraceStep):
            summary = item.summary()
            markers = _component_markers(item.raw)
            if markers:
                summary["component_markers"] = markers
            minimal_step = {
                "step_id": item.step_id,
                "agent": item.agent_id,
                "summary": summary,
                "record_keys": sorted(str(key) for key in item.raw.keys()),
                "text_shapes": _text_shapes(item.raw),
                "truncated": True,
            }
            previews = _short_text_previews(item.raw)
            if previews:
                minimal_step["text_previews"] = previews
            snippets = _component_snippets(item.raw, limit=180, max_items=4)
            if snippets:
                minimal_step["component_snippets"] = snippets
            outline = _substep_outline(item.raw, limit=160, max_items=4)
            if outline:
                minimal_step["substep_outline"] = outline
            minimal_steps.append(minimal_step)
        else:
            minimal_steps.append(item)
    payload = {
        "steps": minimal_steps,
        "truncated": True,
        "notice": TRUNCATION_NOTICE,
    }
    if coverage_guidance:
        payload["coverage_guidance"] = coverage_guidance
    return json.dumps(payload, ensure_ascii=False)


def _auto_scout_anchor_ids(
    trace: AgentTrace,
    anchor_ids: list[int],
    step_positions: dict[int, int],
    inspected_step_ids: list[int],
) -> list[int]:
    if inspected_step_ids or len(trace.steps) < AUTO_SCOUT_MIN_STEPS or not anchor_ids:
        return []

    anchor_positions = [
        step_positions[step_id]
        for step_id in anchor_ids
        if step_id in step_positions
    ]
    if not anchor_positions:
        return []

    prefix_limit = max(2, int((len(trace.steps) - 1) * AUTO_SCOUT_PREFIX_FRACTION))
    if max(anchor_positions) > prefix_limit:
        return []

    used = set(anchor_ids)
    scout_ids: list[int] = []
    continuation_position = min(
        len(trace.steps) - 1,
        max(anchor_positions) + (2 * INSPECTION_CONTEXT_RADIUS) + 1,
    )
    if continuation_position > max(anchor_positions):
        step_id = trace.steps[continuation_position].step_id
        if step_id not in used:
            used.add(step_id)
            scout_ids.append(step_id)

    for fraction in AUTO_SCOUT_FRACTIONS:
        position = round((len(trace.steps) - 1) * fraction)
        step_id = trace.steps[position].step_id
        if step_id in used:
            continue
        used.add(step_id)
        scout_ids.append(step_id)
        if len(scout_ids) >= AUTO_SCOUT_MAX_ANCHORS:
            break
    return scout_ids


def _coverage_gap_anchor_ids(
    trace: AgentTrace,
    inspected_step_ids: list[int],
    *,
    limit: int = MAX_INSPECT_ITEMS,
    min_gap: int = 8,
) -> list[int]:
    inspected = set(inspected_step_ids)
    suggestions: list[int] = []
    gaps: list[tuple[int, int, int]] = []
    start: int | None = None
    for position, step in enumerate(trace.steps):
        if step.step_id in inspected:
            if start is not None:
                gaps.append((start, position - 1, position - start))
                start = None
            continue
        if start is None:
            start = position
    if start is not None:
        gaps.append((start, len(trace.steps) - 1, len(trace.steps) - start))

    candidate_gaps = [gap for gap in gaps if gap[2] >= min_gap]
    for first, _last, length in sorted(candidate_gaps, key=lambda item: item[0]):
        position = first + min(INSPECTION_CONTEXT_RADIUS, length - 1)
        step_id = trace.steps[position].step_id
        if step_id not in inspected and step_id not in suggestions:
            suggestions.append(step_id)
        if len(suggestions) >= limit:
            return suggestions
    for first, _last, length in sorted(candidate_gaps, key=lambda item: item[2], reverse=True):
        position = first + (length // 2)
        step_id = trace.steps[position].step_id
        if step_id not in inspected and step_id not in suggestions:
            suggestions.append(step_id)
        if len(suggestions) >= limit:
            return suggestions
    return suggestions


def execute(
    trace: AgentTrace,
    step_id_list: list[int],
    inspected_step_ids: list[int],
) -> str:
    if not isinstance(step_id_list, list) or not step_id_list:
        return json.dumps({"error": "step_id_list must be a non-empty list"})

    if len(step_id_list) > MAX_INSPECT_ITEMS:
        return json.dumps({
            "error": f"Too many step anchors ({len(step_id_list)}), max {MAX_INSPECT_ITEMS}."
        })

    step_positions = {step.step_id: position for position, step in enumerate(trace.steps)}
    anchor_ids: list[int] = []
    records_by_id: dict[int, TraceStep] = {}
    errors: list[dict[str, Any]] = []
    for value in step_id_list:
        try:
            step_id = int(value)
        except (TypeError, ValueError):
            errors.append({"step_id": value, "error": "invalid step ID"})
            continue
        position = step_positions.get(step_id)
        if position is None:
            errors.append({"step_id": step_id, "error": "step not found"})
            continue
        if step_id not in anchor_ids:
            anchor_ids.append(step_id)

    scout_anchor_ids = _auto_scout_anchor_ids(
        trace,
        anchor_ids,
        step_positions,
        inspected_step_ids,
    )
    all_anchor_ids = [*anchor_ids, *scout_anchor_ids]
    scout_anchor_set = set(scout_anchor_ids)

    for step_id in all_anchor_ids:
        position = step_positions.get(step_id)
        if position is None:
            continue
        radius = (
            SINGLE_ANCHOR_CONTEXT_RADIUS
            if len(step_id_list) == 1 and step_id not in scout_anchor_set
            else INSPECTION_CONTEXT_RADIUS
        )
        start = max(0, position - radius)
        stop = min(len(trace.steps), position + radius + 1)
        for neighboring_step in trace.steps[start:stop]:
            records_by_id.setdefault(neighboring_step.step_id, neighboring_step)

    records: list[TraceStep | dict[str, Any]] = [
        step for step in trace.steps if step.step_id in records_by_id
    ]
    for step in records:
        if isinstance(step, TraceStep) and step.step_id not in inspected_step_ids:
            inspected_step_ids.append(step.step_id)
    if errors:
        records.extend(errors)
    coverage_guidance = None
    if scout_anchor_ids:
        returned_ids = [
            step.step_id for step in records if isinstance(step, TraceStep)
        ]
        coverage_guidance = {
            "auto_scout": True,
            "reason": (
                "The first inspection was clustered near the beginning of a long "
                "trace, so later scout windows were added for coverage."
            ),
            "requested_anchor_step_ids": anchor_ids,
            "scout_anchor_step_ids": scout_anchor_ids,
            "returned_step_ids": returned_ids,
            "instruction": (
                "Compare early evidence with all returned scout windows before "
                "final_answer. If a scout step looks more causal, inspect a local "
                "window around it before submitting."
            ),
        }
        suggested_anchors = _coverage_gap_anchor_ids(trace, inspected_step_ids)
        if suggested_anchors:
            coverage_guidance["suggested_next_anchor_step_ids"] = suggested_anchors
            coverage_guidance["suggestion_reason"] = (
                "These anchors cover the starts or midpoints of the largest "
                "uninspected chronological gaps."
            )
    return _build_response(records, coverage_guidance)
