"""test_sql tool: execute a SQL query against the sample's database for verification."""
from __future__ import annotations
import json
import sqlite3
from typing import Any

SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "test_sql",
        "description": (
            "Execute a SQL query against the database and return the result or error. "
            "Use this to verify your SQL before submitting a final answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "The SQL query to execute.",
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
    },
}

def execute(arguments: dict[str, Any], db_path: str | None = None) -> str:
    """Execute SQL and return JSON result. Safe: catches all exceptions."""
    sql = arguments.get("sql", "")
    if not sql or not isinstance(sql, str):
        return json.dumps({"error": "sql must be a non-empty string"})
    if not db_path:
        return json.dumps({"error": "no database available for this sample"})
    try:
        conn = sqlite3.connect(db_path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(50)  # limit to 50 rows for context efficiency
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        total_rows = len(rows)
        # Check if there are more rows
        has_more = bool(cursor.fetchone())
        cursor.close()
        conn.close()
        result = {
            "columns": columns,
            "rows": [list(row) for row in rows],
            "row_count": total_rows,
            "truncated": has_more,
        }
        return json.dumps(result, default=str)
    except Exception as exc:
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"})
