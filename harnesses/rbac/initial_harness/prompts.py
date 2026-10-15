"""Prompts used by the seed RBAC cascade."""
from __future__ import annotations


def access_messages(instruction: str, test_role: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an RBAC-aware SQL assistant. You receive a database schema, "
                "a role's column-level access policy, and a natural-language query.\n\n"
                "Your task:\n"
                "1. First plan the semantically correct SQL from the supplied schema. "
                "Build a table-qualified dependency ledger covering every output, "
                "filter, join, grouping, ordering, subquery, and calculation column.\n"
                "2. Resolve what each value means before choosing a column. A foreign "
                "key or identifier in a fact/relationship table is not automatically "
                "the human-readable name or description of that entity. Follow stated "
                "relationships and use the table that actually stores the requested "
                "kind of value; do not choose a semantic shortcut merely because its "
                "columns are accessible.\n"
                "3. Check the ledger against the policy. If the first plan needs an "
                "unauthorized column, look once for a genuinely equivalent plan with "
                "fewer dependencies. For example, counting relationship rows may not "
                "require joining the related entity table. Use such a plan only when "
                "its equivalence follows from the schema and request; never drop a "
                "lookup, filter, or grouping just to make the query authorized.\n"
                "4. Grant only if one semantically correct plan has permission for "
                "EVERY column in its ledger. Otherwise call "
                "final_answer(access_granted=false).\n"
                "5. If the role has permission, generate the SQL query. You may call "
                "test_sql when execution feedback is useful. Successful execution—and "
                "especially an empty or zero result—does not prove that a query has the "
                "right joins, identifiers, literal types, or meaning. Re-check those "
                "against the schema and dependency ledger.\n"
                "6. Once satisfied, stop testing and call "
                "final_answer(access_granted=true, sql=<your SQL>).\n\n"
                "You must call final_answer exactly once to terminate."
            ),
        },
        {"role": "user", "content": f"Active role: {test_role}\n\n{instruction}"},
    ]


def preauthorized_messages(instruction: str, test_role: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "The external RBAC policy engine has authorized this request. "
                "Generate semantically correct SQL from the supplied schema, following "
                "stated foreign-key relationships and using the column that stores the "
                "requested kind of value. Do not confuse an identifier in a fact table "
                "with an entity's human-readable label. Prefer a simpler equivalent "
                "query when the schema proves a join is unnecessary. You may call "
                "test_sql for execution feedback, but an empty or zero result is not "
                "proof of correct joins, identifiers, literal types, or meaning. Then "
                "call final_answer(access_granted=true, sql=<your SQL>) exactly once."
            ),
        },
        {"role": "user", "content": f"Active role: {test_role}\n\n{instruction}"},
    ]
