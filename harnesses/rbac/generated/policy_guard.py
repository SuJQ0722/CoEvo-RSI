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
_SCHEMA_TABLE_COLUMNS = re.compile(
    r"(?mi)^Table\s+([^\n]+?)\s+has columns such as\s+(.+?)\.\s*$"
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

# Grant-side citation parsing.  ``_FROM_OR_JOIN_ALIAS`` captures the optional
# table qualifier and the optional alias of every FROM/JOIN source; the alias is
# accepted only when it is not a SQL keyword, so grammar words such as ``WHERE``
# or ``JOIN`` are never mistaken for a table name.
_FROM_OR_JOIN_ALIAS = re.compile(
    rf"(?i)\b(?:from|join)\s+({_SQL_IDENTIFIER})"
    rf"(?:\s*\.\s*({_SQL_IDENTIFIER}))?"
    rf"(?:\s+(?:as\s+)?({_SQL_IDENTIFIER}))?"
)
_QUALIFIED_REFERENCE = re.compile(
    rf"(?i)({_SQL_IDENTIFIER})\s*\.\s*({_SQL_IDENTIFIER})"
)
_SQL_KEYWORDS = frozenset(
    {
        "all", "and", "as", "between", "case", "cross", "distinct", "else",
        "end", "except", "exists", "from", "full", "group", "having", "in",
        "inner", "intersect", "is", "join", "left", "like", "limit", "natural",
        "not", "null", "offset", "on", "or", "order", "outer", "right",
        "select", "set", "then", "union", "using", "values", "when", "where",
    }
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


def named_unavailable_column(instruction: str, claimed: object) -> str:
    """Return the normalized name of a claimed column the role cannot access.

    The claim is accepted only when it resolves to a column declared in the
    supplied schema and every schema table that declares that column blocks it.
    The decision is derived from the structured schema and policy header alone:
    the request body is excluded, unsupported formats return an empty string,
    and a claim the role can reach anywhere is rejected so the calling agent
    cannot manufacture denial evidence from request text or a role name.
    """
    if not isinstance(instruction, str) or not isinstance(claimed, str):
        return ""
    if not claimed.strip():
        return ""

    header, separator, _request = instruction.partition("##Access Request:")
    if not separator or "##Role Access Policy" not in header:
        return ""

    access_match = _ACCESS_LINE.search(header)
    if not access_match:
        return ""

    policy: dict[str, set[str] | None] = {}
    for entry in access_match.group(1).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            return ""
        table, columns = entry.split(":", 1)
        table_key = _normalize_identifier(table)
        if not table_key or table_key in policy:
            return ""
        if _normalize_identifier(columns) == "all columns":
            policy[table_key] = None
            continue
        column_keys = {
            _normalize_identifier(column)
            for column in columns.split(",")
            if _normalize_identifier(column)
        }
        if not column_keys:
            return ""
        policy[table_key] = column_keys

    schema: dict[str, set[str]] = {}
    for match in _SCHEMA_TABLE_COLUMNS.finditer(header):
        table_key = _normalize_identifier(match.group(1))
        if not table_key:
            continue
        column_keys = {
            _normalize_identifier(column)
            for column in match.group(2).split(",")
            if _normalize_identifier(column)
        }
        if column_keys:
            schema[table_key] = column_keys
    if not schema:
        return ""

    claim_key = _normalize_identifier(claimed)
    if not claim_key:
        return ""

    declaring_tables = [
        table for table, columns in schema.items() if claim_key in columns
    ]
    if not declaring_tables:
        return ""

    for table in declaring_tables:
        if table not in policy:
            # A table absent from the policy grants none of its columns, so it
            # blocks the claim rather than making it reachable.
            continue
        allowed = policy[table]
        if allowed is None or claim_key in allowed:
            # The role can reach the claimed column somewhere, so it is not a
            # valid blocking claim.
            return ""
    return claim_key


def _parsed_header(
    header: str,
) -> tuple[dict[str, set[str] | None], dict[str, set[str]]] | None:
    """Parse a policy header into (table -> allowed columns, table -> schema columns).

    ``None`` is returned whenever the header is not exact enough to judge a grant,
    so an unparseable header abstains instead of guessing.  Only the text before
    ``##Access Request:`` is ever passed here.
    """
    access_match = _ACCESS_LINE.search(header)
    if not access_match:
        return None

    policy: dict[str, set[str] | None] = {}
    for entry in access_match.group(1).split(";"):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            return None
        table, columns = entry.split(":", 1)
        table_key = _normalize_identifier(table)
        if not table_key or table_key in policy:
            return None
        if _normalize_identifier(columns) == "all columns":
            policy[table_key] = None
            continue
        column_keys = {
            _normalize_identifier(column)
            for column in columns.split(",")
            if _normalize_identifier(column)
        }
        if not column_keys:
            return None
        policy[table_key] = column_keys

    schema: dict[str, set[str]] = {}
    for match in _SCHEMA_TABLE_COLUMNS.finditer(header):
        table_key = _normalize_identifier(match.group(1))
        if not table_key:
            continue
        column_keys = {
            _normalize_identifier(column)
            for column in match.group(2).split(",")
            if _normalize_identifier(column)
        }
        if column_keys:
            schema[table_key] = column_keys

    if not policy or not schema:
        return None
    return policy, schema


def _grant_grounding(instruction: str, sql: str) -> tuple[bool | None, dict[str, str]]:
    """Shared tri-state grounding verdict plus the first ungrounded citation.

    Returns ``(verdict, failure)``.  ``failure`` names the offending citation
    (``{"table": ..., "column": ...}``) whenever ``verdict`` is ``False`` and is an
    empty dict otherwise.  Only the structured policy header before
    ``##Access Request:`` and the tables/qualified columns the SQL itself cites are
    read; request text, role names and unqualified SQL identifiers are never used.
    """
    if not isinstance(instruction, str) or not isinstance(sql, str) or not sql.strip():
        return None, {}

    header, separator, _request = instruction.partition("##Access Request:")
    if not separator or "##Role Access Policy" not in header:
        return None, {}

    parsed = _parsed_header(header)
    if parsed is None:
        return None, {}
    policy, schema = parsed

    scrubbed_sql = _SQL_COMMENTS_OR_STRINGS.sub(" ", sql)
    cte_names = {
        _normalize_identifier(match.group(1)) for match in _CTE_NAME.finditer(scrubbed_sql)
    }

    aliases: dict[str, str] = {}
    referenced_tables: set[str] = set()
    for match in _FROM_OR_JOIN_ALIAS.finditer(scrubbed_sql):
        table = _normalize_identifier(match.group(2) or match.group(1))
        if table not in schema or table in cte_names:
            continue
        referenced_tables.add(table)
        alias = _normalize_identifier(match.group(3) or "")
        if alias and alias not in _SQL_KEYWORDS and alias not in schema:
            aliases.setdefault(alias, table)

    if not referenced_tables:
        # Nothing in the SQL resolves to a declared schema table, so no citation
        # can be checked; abstain rather than guess from unqualified identifiers.
        return None, {}

    for table in sorted(referenced_tables):
        if table not in policy:
            return False, {"table": table, "column": ""}

    for match in _QUALIFIED_REFERENCE.finditer(scrubbed_sql):
        qualifier = _normalize_identifier(match.group(1))
        column = _normalize_identifier(match.group(2))
        if not qualifier or not column:
            continue
        table = qualifier if qualifier in referenced_tables else aliases.get(qualifier)
        if table is None:
            # A derived-table alias or a database qualifier; no column evidence.
            continue
        allowed = policy[table]
        if column in schema.get(table, set()) and allowed is not None and column not in allowed:
            return False, {"table": table, "column": column}

    return True, {}


def grant_is_grounded(instruction: str, sql: str) -> bool | None:
    """Tri-state grant-side citation verdict for a submitted authorization grant.

    The check reads only the structured Role Access Policy header before
    ``##Access Request:`` and only references the model itself cited: the tables
    named by the SQL's FROM/JOIN sources (including aliases) and the columns the
    SQL qualifies with one of those tables.  It returns:

    * ``False`` when a cited table is absent from the policy, or a cited column is
      declared by a referenced table but withheld there -- the grant is not
      grounded;
    * ``None`` (abstain) when the header or SQL cannot be parsed, or when nothing in
      the SQL resolves to a declared schema table;
    * ``True`` when every resolvable citation is authorized.

    Nothing here is inferred from request text, role names, or unqualified SQL
    identifiers, and the verdict can only restrict grants; it never manufactures a
    denial.
    """
    return _grant_grounding(instruction, sql)[0]


def grant_grounding_context(instruction: str, sql: str) -> dict[str, object]:
    """Header-grounded repair context for a definite ``False`` grant verdict.

    Returns ``{}`` for a ``True`` or ``None`` verdict so a caller can never turn an
    abstention or an authorized grant into denial evidence.  For a definite
    ``False`` verdict it returns:

    * ``evidence`` -- the normalized offending citation (``table.column`` for a
      column the referenced table withholds, ``table`` for a table with no
      Accessible Columns entry);
    * ``declared_tables`` -- the header's declared schema tables, in header order;
    * ``alternative_tables`` -- declared tables other than the offending one where
      the withheld column is granted (empty when the citation names no column);
    * ``column`` -- the display name of the withheld column, or ``""``.

    Every name is resolved from the structured policy header before
    ``##Access Request:`` and from the SQL's own resolvable citation.  Request
    text, role names and unqualified identifiers are never used.
    """
    verdict, failure = _grant_grounding(instruction, sql)
    if verdict is not False:
        return {}

    table_key = failure.get("table", "")
    column_key = failure.get("column", "")
    evidence = f"{table_key}.{column_key}" if column_key else table_key

    header, separator, _request = instruction.partition("##Access Request:")
    if not separator:
        return {
            "evidence": evidence,
            "declared_tables": [],
            "alternative_tables": [],
            "column": "",
        }

    declared: list[tuple[str, str, set[str]]] = []
    offending_columns: dict[str, str] = {}
    for match in _SCHEMA_TABLE_COLUMNS.finditer(header):
        table_name = _normalize_identifier(match.group(1))
        if not table_name:
            continue
        raw_columns: dict[str, str] = {}
        for raw in match.group(2).split(","):
            normalized = _normalize_identifier(raw)
            if normalized and normalized not in raw_columns:
                raw_columns[normalized] = raw.strip()
        declared.append((table_name, match.group(1).strip(), set(raw_columns)))
        if table_name == table_key:
            offending_columns = raw_columns

    parsed = _parsed_header(header)
    policy = parsed[0] if parsed is not None else {}

    alternatives: list[str] = []
    if column_key:
        for table_name, display, _columns in declared:
            if table_name == table_key:
                continue
            if table_name not in policy:
                continue
            allowed = policy[table_name]
            if allowed is None or column_key in allowed:
                alternatives.append(display)

    return {
        "evidence": evidence,
        "declared_tables": [display for _name, display, _columns in declared],
        "alternative_tables": alternatives,
        "column": offending_columns.get(column_key, column_key),
    }


def grant_grounding_evidence(instruction: str, sql: str) -> str:
    """Name the first ungrounded citation for a definite ``False`` verdict.

    Returns ``""`` for ``None`` (abstain) and ``True`` so a caller can never turn an
    abstention or an authorized grant into denial evidence.  A withheld column is
    reported as ``table.column``; an absent table is reported as ``table``.  The
    single-citation accessor delegates to ``grant_grounding_context`` so the
    evidence string and the repair guidance can never disagree.
    """
    return str(grant_grounding_context(instruction, sql).get("evidence", ""))
