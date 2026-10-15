# Evolved RBAC harness

This harness is deliberately conservative for Spider v3 column-level RBAC. It skips
the LLM for malformed inputs, a role/policy mismatch, a policy that explicitly
grants every column in every table, or a literal unavailable multi-token schema
column with complete per-table policy evidence. Unknown formats and ambiguous
requests defer to the LLM and force exactly one
`final_answer(access_granted: bool)` tool call. It does not contain fuzzy/keyword
rules.

The all-columns preauthorization is now materialized with an SQL body. The frozen
runtime only ever records SQL from the LLM path (`route.decision is None`); a
terminal rule decision is stored as `sql=None`, which is why the incumbent's 13
all-columns grants scored zero on SafeEX despite being access-correct. So the rule
still decides the policy by header only, and a policy declaration with no request
body remains the terminal `decision=1` grant the contract smoke probe checks. An
instruction that carries a request body after the structured policy header is
instead routed through the preauthorized LLM call purely to obtain that SQL, and
`RBACHarness.call_llm` forces the response back to `ACCESS_GRANTED` regardless of
the model's marker or error. The access decision (a grant), the 0/1 output and the
`error=""` terminal are therefore unchanged; only the SQL body is borrowed, and
only when there is a request to answer. No new tool, argument, message or budget is
introduced.

Request presence is detected structurally rather than by matching one literal
marker string. The structured policy header is the span covered by the recognized
header grammars (the `Role:` / `Total Tables:` / `Accessible Columns:` policy block
and the declared-schema `Table ... has columns such as ...` lines), and the router
looks for content beyond that span that survives removal of blank lines and generic
section/key heading lines. A differently punctuated heading, a differently named
heading, or no heading at all are therefore treated identically; the incumbent's
literal `##Access Request:` test never fired on the observed score-0 subset. Only
the presence of a request body is read, never its wording, and a trailing block
made up solely of blank lines and heading lines stays a terminal rule grant.

The all-columns rule normalizes policy text only: casing, spacing, and separators
(semicolons or line breaks) across multi-line `Accessible Columns` lists, with the
entry count required to match `Total Tables` exactly and every entry required to
resolve to `ALL COLUMNS`. Requests are never used as evidence, unknown or
restricted policies defer to the LLM, and the grant route still generates SQL
through the schema/foreign-key-aware prompt rather than short-circuiting it.

The prompt layer also enforces one documented size ceiling. `prompts.py` exports
`MODEL_PROMPT_CHAR_BUDGET` and `MAX_TOOL_RESULT_CHARS`; `llm.py` imports both,
caps each tool observation, and shrinks an oversized live transcript (tool-result
text first, then assistant prose, oldest-first) before every model call. The Role
Access Policy block and the Access Request body are never rewritten: schema
`Table ... has columns such as ...` lines alone may be compacted, and only when the
header already exceeds the budget. Rule routing and `policy_guard` still read the
original instruction.

The LLM-fallback denial path is evidence-checked. `final_answer` accepts an
optional `unauthorized_column`, and `policy_guard.named_unavailable_column`
validates it against the structured header only: the claim must name a column the
schema declares and that every schema table declaring it blocks. A denial with no
such evidence is not accepted immediately; the loop returns one bounded
observation asking for a header-grounded column or a grant, then honours whatever
the model decides next. The all-columns preauthorized message path and the
whole-table `policy_guard.inaccessible_tables` override are untouched, and a whole
table omitted from the policy still blocks a grant before it is returned.

The model-path access prompt is grant-forward rather than caution-forward. It tells
the assistant to decide access from the tables and columns the SQL it will submit
actually reads: a request whose SQL stays inside the role's declared Accessible
Columns policy **must** be granted, and denial is reserved for a table with no
Accessible Columns entry or a column the policy withholds in every table that
declares it. Declared tables and columns remain the only evidence; request words
and role names never decide access. The prompt edit changes system-message wording
only, so the Role Access Policy block and the Access Request body still reach the
model byte-identical to the source instruction.

Turn exhaustion on the LLM-fallback route is now a formal decision rather than an
abort. `policy_guard.grant_is_grounded(instruction, sql)` is a tri-state grant-side
citation check that reads only the policy header and only the tables and qualified
columns the model's own SQL cites: it returns `False` when a cited table is absent
from the policy or a cited column is declared by a referenced table but withheld
there, `None` when nothing in the SQL resolves to a declared schema table, and
`True` otherwise. `llm.py` remembers the most recent `test_sql` submission that
clears both this check and the whole-table `inaccessible_tables` veto. If the loop
exhausts its ten turns without a `final_answer`, that guard-cleared submission is
returned as a formal `ACCESS_GRANTED` with its SQL; when no guard-cleared
submission exists the decision fails closed to a formal `ACCESS_DENIED`. Both
terminal branches report `error=""`, so no exhausted sample returns
`predicted_access=None`. The branch is gated to `route.source == "llm_fallback"`,
the turn and `test_sql` budgets are unchanged, and the normal `final_answer` grant,
whole-table veto, denial-evidence checkpoint, and preauthorized all-columns route
are untouched.

The same tri-state citation verdict now also guards ordinary `final_answer` grants on
the LLM-fallback route. After the unpreempted whole-table veto, a grant whose SQL the
check reports as a definite `False` (a cited table absent from the policy, or a
column the SQL qualifies with a referenced table but the policy withholds there) is
not accepted immediately: the loop returns one bounded `not_accepted` observation
carrying the offending citation and grant-preserving guidance, so the model can
either re-grant with SQL rewritten to authorized columns or deny with a
header-grounded `unauthorized_column`. That guidance is header-grounded rather than
generic: it names the schema's declared tables, the exact offending `table.column`
(or the policy-absent table), and, when the same withheld column is granted by
another declared table, that alternative table so a shortcut-table plan can be
rewritten instead of resubmitted verbatim. Every name is resolved from the policy
header before `##Access Request:`; request words, role names and unqualified
identifiers are never quoted. A second ungrounded grant fails closed to a formal
`ACCESS_DENIED` that carries the offending citation. A `None` verdict (unparseable
header or SQL, or nothing resolvable to a declared schema table) abstains: the
model's own grant is returned untouched, so ambiguity can never manufacture a
denial. The checkpoint is gated to `route.source == "llm_fallback"`, adds no model
call and no new tool, and leaves the `all_columns_policy` decision, the
denial-evidence checkpoint and both terminal markers unchanged.

Candidates may evolve `__init__.py` and any other local module while preserving the
`build_harness(policy_rows)` public contract and required runtime behavior. The seed
separates evolvable behavior into `routing.py`, `harness.py`, `prompts.py`, and
`response.py`; `llm.py` owns model calls plus usage, latency, and error normalization.
