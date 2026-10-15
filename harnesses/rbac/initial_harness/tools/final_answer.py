"""final_answer tool: submit one formal RBAC authorization decision."""

from __future__ import annotations

import json
from typing import Any

ACCESS_GRANTED = "ACCESS_GRANTED"
ACCESS_DENIED = "ACCESS_DENIED"

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the authorization decision. If access is granted, include the SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "access_granted": {
                    "type": "boolean",
                    "description": "True only when the supplied role may access the request.",
                },
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute. Required when access_granted is true.",
                },
            },
            "required": ["access_granted"],
            "additionalProperties": False,
        },
    },
}


def marker(access_granted: bool) -> str:
    return ACCESS_GRANTED if access_granted else ACCESS_DENIED


def execute(arguments: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Returns (tool_response_json, access_marker, sql_or_none)."""
    value = arguments.get("access_granted")
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        value = value.strip().lower() == "true"
    if not isinstance(value, bool):
        return (
            json.dumps(
                {
                    "error": "access_granted must be a boolean",
                    "retryable": False,
                }
            ),
            None,
            None,
        )
    sql = arguments.get("sql")
    if isinstance(sql, str):
        sql = sql.strip() or None
    result = marker(value)
    return json.dumps({"status": "submitted", "result": result, "sql": sql}), result, sql
