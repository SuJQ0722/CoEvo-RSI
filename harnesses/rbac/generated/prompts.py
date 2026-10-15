"""Prompts used by the seed RBAC cascade."""
from __future__ import annotations

import re

# Prompt-size budget.  Observed aborts ("exhausted 10 turns without calling
# final_answer", completion_tokens saturated at 40960) happen above roughly 12k
# prompt tokens; 9k estimated tokens leaves headroom for the token limit and for
# multi-turn transcript growth.  ``MODEL_PROMPT_CHAR_BUDGET`` is consumed by
# ``llm.LLMCaller`` and is exported here so the prompt and loop layers share one
# documented ceiling without an import cycle.
_MODEL_PROMPT_TOKEN_BUDGET = 9000
_CHARS_PER_TOKEN = 3.5
MODEL_PROMPT_CHAR_BUDGET = int(_MODEL_PROMPT_TOKEN_BUDGET * _CHARS_PER_TOKEN)
MAX_TOOL_RESULT_CHARS = 6000

# A schema section is compacted only once the raw instruction alone threatens the
# assembled-prompt budget; ordinary instructions are embedded verbatim.  The first
# pass caps very wide column lists, and the halving / trailing-drop passes only run
# for pathological large-schema inputs.
_SCHEMA_COMPACTION_TRIGGER = MODEL_PROMPT_CHAR_BUDGET
MAX_SCHEMA_COLUMNS_PER_TABLE = 64
_MIN_SCHEMA_COLUMNS_PER_TABLE = 16

_REQUEST_MARKER = "##Access Request:"
_SCHEMA_TABLE_LINE = re.compile(
    r"(?mi)^Table\s+([^\n]+?)\s+has columns such as\s+(.+?)\.\s*$"
)


def _compact_schema_line(match: re.Match[str], max_columns: int) -> str:
    """Shorten one schema table line to at most ``max_columns`` column names."""
    prefix = match.group(0)[: match.start(2) - match.start(0)]
    columns = [part.strip() for part in match.group(2).split(",") if part.strip()]
    if len(columns) > max_columns:
        columns = columns[:max_columns]
    return f"{prefix}{', '.join(columns)}."


def compact_instruction(instruction: str) -> str:
    """Bound the schema section of an oversized instruction.

    The Role Access Policy block and the Access Request body are never edited and
    are reattached verbatim; only ``Table ... has columns such as ...`` lines that
    sit in the policy header are shortened.  Instructions whose header already fits
    the prompt budget are returned unchanged, so ordinary samples are untouched.
    """
    if not isinstance(instruction, str) or len(instruction) <= _SCHEMA_COMPACTION_TRIGGER:
        return instruction
    header, separator, request = instruction.partition(_REQUEST_MARKER)
    if not separator or "##Role Access Policy" not in header:
        return instruction

    budget = MODEL_PROMPT_CHAR_BUDGET - len(separator) - len(request)
    if len(header) <= budget:
        return instruction

    def compact(header_text: str, max_columns: int) -> str:
        return _SCHEMA_TABLE_LINE.sub(
            lambda match: _compact_schema_line(match, max_columns), header_text
        )

    compacted = compact(header, MAX_SCHEMA_COLUMNS_PER_TABLE)
    max_columns = MAX_SCHEMA_COLUMNS_PER_TABLE
    while len(compacted) > budget and max_columns > _MIN_SCHEMA_COLUMNS_PER_TABLE:
        max_columns = max(_MIN_SCHEMA_COLUMNS_PER_TABLE, max_columns // 2)
        compacted = compact(header, max_columns)
    # Last resort for a header that is still over budget after column capping:
    # drop trailing schema table lines only; the policy block and the request
    # body are never touched and stay in place.
    while len(compacted) > budget:
        matches = list(_SCHEMA_TABLE_LINE.finditer(compacted))
        if not matches:
            break
        last = matches[-1]
        compacted = compacted[: last.start()] + compacted[last.end() :]
    return compacted + separator + request


def access_messages(instruction: str, test_role: str) -> list[dict[str, str]]:
    instruction = compact_instruction(instruction)
    return [
        {
            "role": "system",
            "content": (
                "You are an RBAC-aware SQL assistant. You receive a database schema, "
                "a role's column-level access policy, and a natural-language query.\n\n"
                "Your task:\n"
                "1. First plan the semantically correct SQL from the supplied schema. "
                "Follow stated foreign-key relationships and use the table that really "
                "stores each requested value; do not invent a shortcut query merely "
                "because its columns are accessible.\n"
                "2. Decide access from the tables and columns that SQL reads, judged "
                "against the role's Accessible Columns list only:\n"
                "   - If every table and column that SQL reads is granted by the "
                "policy, the request is authorized and you MUST call "
                "final_answer(access_granted=true, sql=<your SQL>).\n"
                "   - If the SQL must read a table with no Accessible Columns entry at "
                "all, the request is not authorized: deny it.\n"
                "   - If the SQL must read a column the policy does not grant in any "
                "table that declares it—including columns used in filters, joins, "
                "grouping, ordering, and calculations—the request is not authorized: "
                "deny it.\n"
                "   Declared tables and columns are the only evidence; request words "
                "and role names never decide access.\n"
                "3. When access is authorized, generate the SQL query. You may call "
                "test_sql when execution feedback is useful. Successful execution—and "
                "especially an empty or zero result—does not prove that a query has the "
                "right joins, identifiers, or meaning. Check those against the schema.\n"
                "4. Stop testing once satisfied and terminate with exactly one "
                "final_answer call: final_answer(access_granted=true, sql=<your SQL>) "
                "for an authorized request, otherwise final_answer(access_granted="
                "false, unauthorized_column=<the blocked column>).\n\n"
                "An authorized request must be granted: deny only when the SQL "
                "genuinely needs a table or column the Accessible Columns policy "
                "withholds."
            ),
        },
        {"role": "user", "content": f"Active role: {test_role}\n\n{instruction}"},
    ]


def preauthorized_messages(instruction: str, test_role: str) -> list[dict[str, str]]:
    instruction = compact_instruction(instruction)
    return [
        {
            "role": "system",
            "content": (
                "The external RBAC policy engine has authorized this request. "
                "Generate semantically correct SQL from the supplied schema, following "
                "stated foreign-key relationships and using the table that actually "
                "stores each requested value. You may call test_sql for execution "
                "feedback, but an empty or zero result is not proof of correct joins or "
                "identifiers. Then call final_answer(access_granted=true, sql=<your SQL>) "
                "exactly once to submit."
            ),
        },
        {"role": "user", "content": f"Active role: {test_role}\n\n{instruction}"},
    ]
