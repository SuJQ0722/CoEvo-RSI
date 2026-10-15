"""Conservative post-generation checks for column-policy violations.

The guard intentionally recognizes only one unambiguous condition: generated SQL
reads a table declared in the supplied schema, while that table is wholly absent
from the role's accessible-column list.  Anything it cannot parse is left to the
LLM rather than guessed from request words or role names.
"""

from __future__ import annotations

import re


_ACCESS_LINE = re.compile(r"(?mi)^Accessible Columns:\s*(.+?)\s*$")
_SCHEMA_TABLE = re.compile(
    r"(?mi)^Table\s+([^\n]+?)\s+has columns such as\s+.+?\.\s*$"
)
_SQL_IDENTIFIER = r'(?:"[^"]+"|`[^`]+`|\[[^\]]+\]|[A-Za-z_]\w*)'
_FROM_OR_JOIN = re.compile(
    rf"(?i)\b(?:from|join)\s+({_SQL_IDENTIFIER})"
    rf"(?:\s*\.\s*({_SQL_IDENTIFIER}))?"
)
_CTE_NAME = re.compile(rf"(?i)(?:\bwith\b|,)\s*({_SQL_IDENTIFIER})\s+as\s*\(")
_SQL_COMMENTS_OR_STRINGS = re.compile(
    r"--[^\n]*|/\*.*?\*/|'(?:''|[^'])*'", re.DOTALL
)


def _normalize_identifier(value: object) -> str:
    """Normalize quoted SQL/schema identifiers without fuzzy matching."""
    text = str(value or "").strip().strip('`"[]').casefold()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def inaccessible_tables(instruction: str, sql: str) -> tuple[str, ...]:
    """Return referenced schema tables having no listed accessible columns.

    An empty tuple also represents an unsupported or incomplete input format.  The
    request body is excluded before parsing so request text cannot manufacture
    schema or policy declarations.
    """
    if not isinstance(instruction, str) or not isinstance(sql, str) or not sql.strip():
        return ()

    header, separator, _request = instruction.partition("##Access Request:")
    if not separator or "##Role Access Policy" not in header:
        return ()

    access_match = _ACCESS_LINE.search(header)
    if not access_match:
        return ()

    schema_tables = {
        _normalize_identifier(match.group(1))
        for match in _SCHEMA_TABLE.finditer(header)
    }
    accessible_tables = {
        _normalize_identifier(entry.split(":", 1)[0])
        for entry in access_match.group(1).split(";")
        if ":" in entry
    }
    if not schema_tables or not accessible_tables:
        return ()

    scrubbed_sql = _SQL_COMMENTS_OR_STRINGS.sub(" ", sql)
    cte_names = {
        _normalize_identifier(match.group(1)) for match in _CTE_NAME.finditer(scrubbed_sql)
    }
    referenced_tables: set[str] = set()
    for match in _FROM_OR_JOIN.finditer(scrubbed_sql):
        # For database-qualified names, the second identifier is the table.
        table = _normalize_identifier(match.group(2) or match.group(1))
        if table in schema_tables and table not in cte_names:
            referenced_tables.add(table)

    return tuple(sorted(referenced_tables - accessible_tables))
